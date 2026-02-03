# SPDX-License-Identifier: MIT
# Copyright (c) 2025 WuResearchGroup

"""LAMMPS-based MD sampler that runs via subprocess.

Uses LAMMPS ``real`` units (Angstroms, kcal/mol, fs) and provides the same
interface as :class:`MolecularDynamics` so it can be used interchangeably
in all three learning workflows (DiffSim, IBI, Relative Entropy).
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from diffcg._core.logger import get_logger
from diffcg._core.units import (
    NM_TO_ANGSTROM,
    ANGSTROM_TO_NM,
    KJMOL_TO_KCALMOL,
    KCALMOL_TO_KJMOL,
    positions_from_lammps,
    cell_from_lammps,
    energy_from_lammps,
)
from diffcg.io.lammps import LAMMPSDumpReader, LAMMPSDataReader
from diffcg.io.lammps_writer import write_lammps_data, write_lammps_table
from diffcg.system import AtomicSystem, Trajectory

logger = get_logger(__name__)

# Default LAMMPS input template
_DEFAULT_TEMPLATE = """\
units real
atom_style full
boundary p p p
read_data {data_file}

{pair_style_block}
{bond_style_block}
{angle_style_block}
{dihedral_style_block}

{special_bonds_block}

velocity all create {temperature} {seed} dist gaussian

{thermostat_block}

{extra_commands}

timestep {timestep}
thermo_style custom step temp press pe ke etotal vol
thermo {loginterval}
dump 1 all custom {loginterval} {dump_file} id type x y z
run {steps}
write_data final.data nocoeff
write_restart restart.lmp
"""

_RESTART_TEMPLATE = """\
units real
atom_style full
boundary p p p
read_restart {restart_file}

{pair_style_block}
{bond_style_block}
{angle_style_block}
{dihedral_style_block}

{special_bonds_block}

{thermostat_block}

{extra_commands}

timestep {timestep}
thermo_style custom step temp press pe ke etotal vol
thermo {loginterval}
dump 1 all custom {loginterval} {dump_file} id type x y z
run {steps}
write_data final.data nocoeff
write_restart restart.lmp
"""


class LAMMPSSampler:
    """Molecular dynamics sampler that runs LAMMPS via subprocess.

    Provides the same public interface as :class:`MolecularDynamics`:
    ``run(steps)``, ``get_trajectory()``, ``get_final_system()``,
    ``set_system(system)``, plus ``update_potentials(energy_params)``
    for iterative learning workflows.
    """

    def __init__(
        self,
        system: AtomicSystem,
        *,
        energy_params: Optional[Dict] = None,
        energy_objects: Optional[List] = None,
        topology: Dict,
        ensemble: str = "nvt",
        thermostat: str = "langevin",
        temperature: float = 300.0,
        timestep: float = 2.0,
        friction: float = 1.0,
        cutoff: float = 1.0,
        r_onset: float = 0.8,
        mol_ids: Optional[np.ndarray] = None,
        trajectory: Optional[str] = None,
        logfile: Optional[str] = None,
        loginterval: int = 100,
        lammps_exe: str = "lmp",
        work_dir: Optional[str] = None,
        random_seed: int = 0,
        extra_lammps_commands: Optional[List[str]] = None,
        input_template: Optional[str] = None,
        restart_file: Optional[str] = None,
        special_bonds: str = "lj 0.0 0.0 0.0",
    ) -> None:
        self._system = system
        self.energy_params = energy_params
        self.energy_objects = energy_objects
        self.topology = topology
        self.ensemble = ensemble.lower()
        self.thermostat = thermostat.lower()
        self.temperature = temperature
        self.timestep = timestep
        self.friction = friction
        self.cutoff = cutoff
        self.r_onset = r_onset
        self.mol_ids = mol_ids
        self.trajectory_path = trajectory
        self.logfile = logfile
        self.loginterval = loginterval
        self.lammps_exe = lammps_exe
        self.random_seed = random_seed
        self.extra_lammps_commands = extra_lammps_commands or []
        self._input_template = input_template
        self._restart_file = restart_file
        self.special_bonds = special_bonds

        # Build LAMMPS specs from energy objects if provided
        self._lammps_specs = None
        if energy_objects is not None:
            from diffcg.energy import build_lammps_specs
            self._lammps_specs = build_lammps_specs(*energy_objects)

        if energy_params is None and energy_objects is None:
            raise ValueError("Either energy_params or energy_objects must be provided")

        if work_dir is None:
            self._work_dir_obj = tempfile.TemporaryDirectory(prefix="diffcg_lammps_")
            self._work_dir = self._work_dir_obj.name
        else:
            self._work_dir = work_dir
            os.makedirs(work_dir, exist_ok=True)
            self._work_dir_obj = None

        self._iteration = 0
        self._last_trajectory: Optional[Trajectory] = None
        self._last_thermo: Optional[Dict] = None

        logger.debug(
            "Created LAMMPSSampler: ensemble=%s, thermostat=%s, T=%s K, dt=%s fs, exe=%s",
            self.ensemble, self.thermostat, temperature, timestep, lammps_exe,
        )

    # --- Public interface ---

    def run(self, steps: int) -> Trajectory:
        """Run LAMMPS for the given number of steps.

        Each call creates a new iteration subdirectory preserving all files.
        """
        iter_dir = self._work_dir

        self._write_input_files(iter_dir, steps)

        log_path = os.path.join(iter_dir, "lammps.log")

        result = subprocess.run(
            [self.lammps_exe, "-in", "input.lmp"],
            cwd=iter_dir,
            capture_output=True,
            text=True,
        )

        # Save stdout/stderr
        with open(log_path, "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n--- STDERR ---\n")
                f.write(result.stderr)

        if result.returncode != 0:
            raise RuntimeError(
                f"LAMMPS failed (return code {result.returncode}).\n"
                f"Working directory: {iter_dir}\n"
                f"STDERR: {result.stderr[-2000:] if result.stderr else '(empty)'}"
            )

        # Parse thermo from stdout
        self._last_thermo = self._parse_thermo_log(result.stdout)
        thermo_csv = os.path.join(iter_dir, "thermo.csv")
        self._save_thermo_csv(self._last_thermo, thermo_csv)

        # Read trajectory
        dump_file = os.path.join(iter_dir, "trajectory.dump")
        self._last_trajectory = self._read_trajectory(dump_file)

        # Copy trajectory to user-requested path
        if self.trajectory_path is not None:
            self._last_trajectory.save(self.trajectory_path)

        self._iteration += 1

        logger.debug("LAMMPS completed, %d frames read", len(self._last_trajectory))
        return self._last_trajectory

    def get_trajectory(self) -> Trajectory:
        if self._last_trajectory is None:
            import jax.numpy as jnp
            return Trajectory(
                positions=jnp.zeros((0, self._system.n_atoms, 3)),
                Z=self._system.Z,
                cell=self._system.cell,
                masses=self._system.masses,
                pbc=self._system.pbc,
            )
        return self._last_trajectory

    def get_final_system(self) -> AtomicSystem:
        """Read the final state from the last LAMMPS run."""
        if self._iteration == 0:
            return self._system

        iter_dir = self._work_dir
        final_data = os.path.join(iter_dir, "final.data")

        if os.path.exists(final_data):
            reader = LAMMPSDataReader(final_data)
            data = reader.read()
            import jax.numpy as jnp

            coords_nm = positions_from_lammps(data["coords"][0])
            cell_nm = cell_from_lammps(data["cells"][0])

            return AtomicSystem(
                R=jnp.array(coords_nm, dtype=jnp.float32),
                Z=self._system.Z,
                cell=jnp.array(cell_nm, dtype=jnp.float32),
                masses=self._system.masses,
                pbc=self._system.pbc,
            )

        # Fallback: use last frame of trajectory
        if self._last_trajectory is not None and len(self._last_trajectory) > 0:
            return self._last_trajectory[-1]
        return self._system

    def set_system(self, system: AtomicSystem) -> None:
        self._system = system

    def update_potentials(self, energy_params: Optional[Dict] = None,
                          energy_objects: Optional[List] = None) -> None:
        """Update potentials for next run (used in iterative learning).

        Args:
            energy_params: Legacy dict-based energy params.
            energy_objects: List of energy class instances with ``to_lammps()`` methods.
        """
        if energy_objects is not None:
            self.energy_objects = energy_objects
            from diffcg.energy import build_lammps_specs
            self._lammps_specs = build_lammps_specs(*energy_objects)
        if energy_params is not None:
            self.energy_params = energy_params

    def get_restart_file(self) -> Optional[str]:
        """Return path to the restart file from the last run, or None."""
        restart_path = os.path.join(self._work_dir, "restart.lmp")
        if os.path.exists(restart_path):
            return restart_path
        return None

    def get_thermo_log(self) -> Optional[Dict]:
        """Return parsed thermodynamic log from last run.

        PE values are converted to kJ/mol. Other values are in LAMMPS real units.
        """
        return self._last_thermo

    def get_input_script(self, steps: int, production: bool = True) -> str:
        """Return the LAMMPS input script that would be generated for inspection."""
        return self._generate_input_script(steps)

    def set_input_template(self, template: str) -> None:
        """Set a custom LAMMPS input template (overrides auto-generation)."""
        self._input_template = template

    # --- Internal methods ---

    def _generate_input_script(self, steps: int) -> str:
        """Build the LAMMPS input script string."""
        if self._input_template:
            template = self._input_template
        elif self._restart_file is not None:
            template = _RESTART_TEMPLATE
        else:
            template = _DEFAULT_TEMPLATE

        pair_block = self._make_pair_style_block()
        bond_block = self._make_bond_style_block()
        angle_block = self._make_angle_style_block()
        dihedral_block = self._make_dihedral_style_block()
        thermostat_block = self._make_thermostat_block()
        extra = "\n".join(self.extra_lammps_commands)

        seed = self.random_seed if self.random_seed != 0 else 12345

        fmt_kwargs = dict(
            pair_style_block=pair_block,
            bond_style_block=bond_block,
            angle_style_block=angle_block,
            dihedral_style_block=dihedral_block,
            special_bonds_block=f"special_bonds {self.special_bonds}",
            thermostat_block=thermostat_block,
            timestep=self.timestep,
            temperature=self.temperature,
            steps=steps,
            loginterval=self.loginterval,
            dump_file="trajectory.dump",
            extra_commands=extra,
            seed=seed,
        )

        if self._restart_file is not None:
            fmt_kwargs["restart_file"] = os.path.basename(self._restart_file)
        else:
            fmt_kwargs["data_file"] = "system.data"

        return template.format(**fmt_kwargs)

    def _make_pair_style_block(self) -> str:
        # New path: use _lammps_specs if available
        if self._lammps_specs is not None:
            return self._make_pair_style_block_from_specs()

        # Legacy path: use energy_params dict
        if self.energy_params is None or "pair" not in self.energy_params:
            return "# No pair interactions"

        cutoff_ang = self.cutoff * NM_TO_ANGSTROM
        pair_data = self.energy_params["pair"]

        if "types" in pair_data:
            type_list = pair_data["types"]
        else:
            type_list = [None]

        lines = [f"pair_style table linear {len(pair_data['x'])}"]
        lines.append(f"pair_modify shift yes")

        if isinstance(type_list[0], (list, tuple)):
            for idx, (ti, tj) in enumerate(type_list):
                keyword = f"PAIR_{idx}"
                lines.append(
                    f"pair_coeff {ti + 1} {tj + 1} pair_table.txt {keyword} {cutoff_ang:.4f}"
                )
        else:
            atom_types = np.asarray(self._system.Z)
            n_types = int(atom_types.max()) + 1
            for ti in range(n_types):
                for tj in range(ti, n_types):
                    lines.append(
                        f"pair_coeff {ti + 1} {tj + 1} pair_table.txt PAIR_0 {cutoff_ang:.4f}"
                    )

        return "\n".join(lines)

    def _make_pair_style_block_from_specs(self) -> str:
        if "pair" not in self._lammps_specs:
            return "# No pair interactions"

        spec = self._lammps_specs["pair"]
        style = spec["lammps_style"]
        lines = []

        if style == "hybrid/overlay":
            # hybrid/overlay style1 cutoff1 style2 ...
            parts = " ".join(
                f"{s} {c:.4f}" if c is not None else s for s, c in spec["style_parts"]
            )
            lines.append(f"pair_style hybrid/overlay {parts}")
            lines.append("pair_modify shift yes")

            for item_style, item in spec["items"]:
                if item_style == "table":
                    types = item.get("types")
                    kw = item["keyword"]
                    if types is not None:
                        ti, tj = types
                        lines.append(
                            f"pair_coeff {ti+1} {tj+1} table pair_table.txt {kw}"
                        )
                    else:
                        atom_types = np.asarray(self._system.Z)
                        n_types = int(atom_types.max()) + 1
                        for ti in range(n_types):
                            for tj in range(ti, n_types):
                                lines.append(
                                    f"pair_coeff {ti+1} {tj+1} table pair_table.txt {kw}"
                                )
                else:
                    types = item.get("types")
                    params_str = " ".join(f"{p:.6f}" for p in item["params"])
                    if types is not None:
                        ti, tj = types
                        lines.append(
                            f"pair_coeff {ti+1} {tj+1} {item_style} {params_str}"
                        )
                    else:
                        lines.append(f"pair_coeff * * {item_style} {params_str}")

        elif style == "table":
            n_points = spec.get("n_points", 500)
            cutoff = spec.get("cutoff", self.cutoff * NM_TO_ANGSTROM)
            lines.append(f"pair_style table linear {n_points}")
            lines.append("pair_modify shift yes")

            for table in spec["tables"]:
                types = table.get("types")
                kw = table["keyword"]
                if types is not None:
                    ti, tj = types
                    lines.append(
                        f"pair_coeff {ti+1} {tj+1} pair_table.txt {kw} {cutoff:.4f}"
                    )
                else:
                    atom_types = np.asarray(self._system.Z)
                    n_types = int(atom_types.max()) + 1
                    for ti in range(n_types):
                        for tj in range(ti, n_types):
                            lines.append(
                                f"pair_coeff {ti+1} {tj+1} pair_table.txt {kw} {cutoff:.4f}"
                            )
        else:
            # Single native style (lj/cut, morse, soft, mie/cut)
            cutoff = spec.get("cutoff", self.cutoff * NM_TO_ANGSTROM)
            lines.append(f"pair_style {style} {cutoff:.4f}")
            lines.append("pair_modify shift yes")

            for coeff in spec["coeffs"]:
                types = coeff.get("types")
                params_str = " ".join(f"{p:.6f}" for p in coeff["params"])
                if types is not None:
                    ti, tj = types
                    lines.append(f"pair_coeff {ti+1} {tj+1} {params_str}")
                else:
                    lines.append(f"pair_coeff * * {params_str}")

        return "\n".join(lines)

    def _make_bond_style_block(self) -> str:
        if self._lammps_specs is not None:
            return self._make_bonded_style_block_from_specs("bond")

        if self.energy_params is None or "bond" not in self.energy_params:
            return "# No bond interactions"

        bond_data = self.energy_params["bond"]
        n_points = len(bond_data["x"])

        if "types" in bond_data:
            n_bond_types = len(bond_data["types"])
        else:
            bond_types_arr = self.topology.get("bond_types", np.empty(0, dtype=int))
            n_bond_types = int(bond_types_arr.max()) + 1 if len(bond_types_arr) > 0 else 1

        lines = [f"bond_style table linear {n_points}"]
        for bt in range(n_bond_types):
            keyword = f"BOND_{bt}"
            lines.append(f"bond_coeff {bt + 1} bond_table.txt {keyword}")

        return "\n".join(lines)

    def _make_angle_style_block(self) -> str:
        if self._lammps_specs is not None:
            return self._make_bonded_style_block_from_specs("angle")

        if self.energy_params is None or "angle" not in self.energy_params:
            return "# No angle interactions"

        angle_data = self.energy_params["angle"]
        n_points = len(angle_data["x"])

        if "types" in angle_data:
            n_angle_types = len(angle_data["types"])
        else:
            angle_types_arr = self.topology.get("angle_types", np.empty(0, dtype=int))
            n_angle_types = int(angle_types_arr.max()) + 1 if len(angle_types_arr) > 0 else 1

        lines = [f"angle_style table linear {n_points}"]
        for at in range(n_angle_types):
            keyword = f"ANGLE_{at}"
            lines.append(f"angle_coeff {at + 1} angle_table.txt {keyword}")

        return "\n".join(lines)

    def _make_dihedral_style_block(self) -> str:
        if self._lammps_specs is not None:
            return self._make_bonded_style_block_from_specs("dihedral")

        if self.energy_params is None or "dihedral" not in self.energy_params:
            return "# No dihedral interactions"

        dihedral_data = self.energy_params["dihedral"]
        n_points = len(dihedral_data["x"])

        if "types" in dihedral_data:
            n_dihedral_types = len(dihedral_data["types"])
        else:
            dihedral_types_arr = self.topology.get("dihedral_types", np.empty(0, dtype=int))
            n_dihedral_types = int(dihedral_types_arr.max()) + 1 if len(dihedral_types_arr) > 0 else 1

        lines = [f"dihedral_style table linear {n_points}"]
        for dt in range(n_dihedral_types):
            keyword = f"DIHEDRAL_{dt}"
            lines.append(f"dihedral_coeff {dt + 1} dihedral_table.txt {keyword}")

        return "\n".join(lines)

    def _make_bonded_style_block_from_specs(self, itype: str) -> str:
        """Generate bond/angle/dihedral style block from _lammps_specs."""
        if itype not in self._lammps_specs:
            return f"# No {itype} interactions"

        spec = self._lammps_specs[itype]
        style = spec["lammps_style"]
        table_file = f"{itype}_table.txt"
        lines = []

        if style == "hybrid":
            parts = " ".join(spec["style_parts"])
            lines.append(f"{itype}_style hybrid {parts}")

            for item_style, item in spec["items"]:
                if item_style == "table":
                    idx = item.get("type_idx", 0)
                    kw = item["keyword"]
                    lines.append(f"{itype}_coeff {idx+1} table {table_file} {kw}")
                else:
                    idx = item.get("type_idx", 0)
                    params_str = " ".join(f"{p:.6f}" for p in item["params"])
                    lines.append(f"{itype}_coeff {idx+1} {item_style} {params_str}")

        elif style == "table":
            n_points = spec.get("n_points", 500)
            lines.append(f"{itype}_style table linear {n_points}")
            for table in spec["tables"]:
                idx = table.get("type_idx", 0)
                kw = table["keyword"]
                lines.append(f"{itype}_coeff {idx+1} {table_file} {kw}")

        else:
            # Single native style (harmonic)
            lines.append(f"{itype}_style {style}")
            for coeff in spec["coeffs"]:
                idx = coeff.get("type_idx", 0)
                params_str = " ".join(f"{p:.6f}" for p in coeff["params"])
                lines.append(f"{itype}_coeff {idx+1} {params_str}")

        return "\n".join(lines)

    def _make_thermostat_block(self) -> str:
        seed = self.random_seed if self.random_seed != 0 else 12345
        T = self.temperature

        if self.ensemble == "nve":
            return "fix 1 all nve"

        if self.thermostat == "langevin":
            damp = self.friction  # fs
            lines = [
                "fix 1 all nve",
                f"fix 2 all langevin {T} {T} {damp} {seed}",
            ]
            return "\n".join(lines)
        elif self.thermostat in ("nose-hoover", "nosehoover", "nh"):
            taut = 100.0 * self.timestep  # Default: 100 timesteps
            lines = [f"fix 1 all nvt temp {T} {T} {taut}"]
            return "\n".join(lines)
        else:
            raise ValueError(f"Unknown thermostat: {self.thermostat}")

    def _write_input_files(self, iter_dir: str, steps: int) -> None:
        """Write data file, table files, and input script to iter_dir."""
        if self._restart_file is not None:
            # Copy restart file into working directory
            import shutil
            restart_dest = os.path.join(iter_dir, os.path.basename(self._restart_file))
            if os.path.abspath(self._restart_file) != os.path.abspath(restart_dest):
                shutil.copy2(self._restart_file, restart_dest)
        else:
            # Data file
            data_path = os.path.join(iter_dir, "system.data")
            write_lammps_data(self._system, self.topology, data_path, mol_ids=self.mol_ids)

        # Table files (always needed)
        self._write_table_files(iter_dir)

        # Input script
        script = self._generate_input_script(steps)
        input_path = os.path.join(iter_dir, "input.lmp")
        with open(input_path, "w") as f:
            f.write(script)

    def _write_table_files(self, iter_dir: str) -> None:
        """Write tabulated potential files."""
        if self._lammps_specs is not None:
            self._write_table_files_from_specs(iter_dir)
            return

        if self.energy_params is None:
            return

        if "pair" in self.energy_params:
            pair = self.energy_params["pair"]
            filepath = os.path.join(iter_dir, "pair_table.txt")
            open(filepath, "w").close()  # Clear file

            if "types" in pair and isinstance(pair["types"][0], (list, tuple)):
                # Multi-type: separate x/y per type
                for idx, _ in enumerate(pair["types"]):
                    x = pair["x"][idx] if isinstance(pair["x"], list) else pair["x"]
                    y = pair["y"][idx] if isinstance(pair["y"], list) else pair["y"]
                    write_lammps_table(filepath, x, y, f"PAIR_{idx}", "pair")
            else:
                write_lammps_table(filepath, pair["x"], pair["y"], "PAIR_0", "pair")

        if "bond" in self.energy_params:
            bond = self.energy_params["bond"]
            filepath = os.path.join(iter_dir, "bond_table.txt")
            open(filepath, "w").close()

            if "types" in bond:
                for idx in range(len(bond["types"])):
                    x = bond["x"][idx] if isinstance(bond["x"], list) else bond["x"]
                    y = bond["y"][idx] if isinstance(bond["y"], list) else bond["y"]
                    write_lammps_table(filepath, x, y, f"BOND_{idx}", "bond")
            else:
                bond_types_arr = self.topology.get("bond_types", np.empty(0, dtype=int))
                n_bt = int(bond_types_arr.max()) + 1 if len(bond_types_arr) > 0 else 1
                for bt in range(n_bt):
                    write_lammps_table(filepath, bond["x"], bond["y"], f"BOND_{bt}", "bond")

        if "angle" in self.energy_params:
            angle = self.energy_params["angle"]
            filepath = os.path.join(iter_dir, "angle_table.txt")
            open(filepath, "w").close()

            if "types" in angle:
                for idx in range(len(angle["types"])):
                    x = angle["x"][idx] if isinstance(angle["x"], list) else angle["x"]
                    y = angle["y"][idx] if isinstance(angle["y"], list) else angle["y"]
                    write_lammps_table(filepath, x, y, f"ANGLE_{idx}", "angle")
            else:
                angle_types_arr = self.topology.get("angle_types", np.empty(0, dtype=int))
                n_at = int(angle_types_arr.max()) + 1 if len(angle_types_arr) > 0 else 1
                for at in range(n_at):
                    write_lammps_table(filepath, angle["x"], angle["y"], f"ANGLE_{at}", "angle")

        if "dihedral" in self.energy_params:
            dihedral = self.energy_params["dihedral"]
            filepath = os.path.join(iter_dir, "dihedral_table.txt")
            open(filepath, "w").close()

            if "types" in dihedral:
                for idx in range(len(dihedral["types"])):
                    x = dihedral["x"][idx] if isinstance(dihedral["x"], list) else dihedral["x"]
                    y = dihedral["y"][idx] if isinstance(dihedral["y"], list) else dihedral["y"]
                    write_lammps_table(filepath, x, y, f"DIHEDRAL_{idx}", "dihedral")
            else:
                dihedral_types_arr = self.topology.get("dihedral_types", np.empty(0, dtype=int))
                n_dt = int(dihedral_types_arr.max()) + 1 if len(dihedral_types_arr) > 0 else 1
                for dt in range(n_dt):
                    write_lammps_table(filepath, dihedral["x"], dihedral["y"], f"DIHEDRAL_{dt}", "dihedral")

    def _write_table_files_from_specs(self, iter_dir: str) -> None:
        """Write table files only for table-style interactions in _lammps_specs."""
        for itype in ("pair", "bond", "angle", "dihedral"):
            if itype not in self._lammps_specs:
                continue
            spec = self._lammps_specs[itype]
            style = spec["lammps_style"]

            # Collect tables to write
            tables_to_write = []
            if style == "table":
                tables_to_write = spec.get("tables", [])
            elif style in ("hybrid/overlay", "hybrid"):
                for item_style, item in spec.get("items", []):
                    if item_style == "table":
                        tables_to_write.append(item)

            if not tables_to_write:
                continue

            table_file = os.path.join(iter_dir, f"{itype}_table.txt")
            open(table_file, "w").close()

            pot_type = itype if itype != "pair" else "pair"
            for table in tables_to_write:
                write_lammps_table(
                    table_file, table["x"], table["y"],
                    table["keyword"], potential_type=pot_type,
                )

    def _read_trajectory(self, dump_file: str) -> Trajectory:
        """Parse LAMMPS dump file and convert to Trajectory (positions in nm)."""
        import jax.numpy as jnp

        reader = LAMMPSDumpReader(dump_file)
        data = reader.read()

        coords_nm = positions_from_lammps(data["coords"])  # (T, N, 3)
        cell_nm = cell_from_lammps(data["cells"][0])  # Use first frame cell

        return Trajectory(
            positions=jnp.array(coords_nm, dtype=jnp.float32),
            Z=self._system.Z,
            cell=jnp.array(cell_nm, dtype=jnp.float32),
            masses=self._system.masses,
            pbc=self._system.pbc,
        )

    @staticmethod
    def _parse_thermo_log(stdout: str) -> Dict:
        """Parse LAMMPS thermo output from stdout.

        Expects ``thermo_style custom step temp press pe ke etotal vol``.
        PE and KE are converted from kcal/mol to kJ/mol.
        """
        thermo = {"step": [], "temp": [], "press": [], "pe": [], "ke": [], "etotal": [], "vol": []}

        in_thermo = False
        for line in stdout.split("\n"):
            stripped = line.strip()

            # Detect start of thermo output
            if stripped.startswith("Step "):
                in_thermo = True
                continue

            # Detect end of thermo output
            if in_thermo and (stripped.startswith("Loop time") or stripped == "" or not stripped[0].isdigit()):
                if stripped.startswith("Loop time"):
                    in_thermo = False
                    continue
                if stripped == "":
                    continue
                if not stripped[0].isdigit() and not stripped.startswith("-"):
                    in_thermo = False
                    continue

            if in_thermo:
                parts = stripped.split()
                if len(parts) >= 7:
                    try:
                        thermo["step"].append(int(parts[0]))
                        thermo["temp"].append(float(parts[1]))
                        thermo["press"].append(float(parts[2]))
                        thermo["pe"].append(float(parts[3]) * KCALMOL_TO_KJMOL)
                        thermo["ke"].append(float(parts[4]) * KCALMOL_TO_KJMOL)
                        thermo["etotal"].append(float(parts[5]) * KCALMOL_TO_KJMOL)
                        thermo["vol"].append(float(parts[6]))
                    except (ValueError, IndexError):
                        continue

        return thermo

    @staticmethod
    def _save_thermo_csv(thermo: Dict, filepath: str) -> None:
        """Save parsed thermo data as CSV."""
        if not thermo or not thermo.get("step"):
            return
        keys = ["step", "temp", "press", "pe", "ke", "etotal", "vol"]
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            for i in range(len(thermo["step"])):
                writer.writerow([thermo[k][i] for k in keys])
