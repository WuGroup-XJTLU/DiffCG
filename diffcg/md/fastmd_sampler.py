"""fastMD-based molecular dynamics sampler.

Runs the fastMD CUDA MD engine via subprocess. Uses the same public
interface as LAMMPSSampler for interchangeable use in DiffSim workflows.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import jax.numpy as jnp

from diffcg._core.logger import get_logger
from diffcg._core.constants import BOLTZMANN_KJMOLK
from diffcg.system import AtomicSystem, Trajectory
from diffcg.io.fastmd_traj import read_fastmd_trajectory

logger = get_logger(__name__)

# Unit conversions: diffcg (fs, K) -> fastMD (ps, kJ/mol)
FS_TO_PS = 0.001
KB = BOLTZMANN_KJMOLK

_DEFAULT_CONFIG_TEMPLATE = """\
natoms {natoms}
ntypes {ntypes}
rc {rc:.6f}
skin {skin:.6f}
dt {dt:.6f}
nsteps {nsteps}
dump_freq {dump_freq}
thermo 1 {thermo_freq} thermo_fastmd.dat
restart 0 restart.data
{ensemble_line}
{table_lines}
lammps_data_file system.data
"""


def _write_fastmd_table(filepath: str, x_vals, y_vals, keyword: str, force_vals=None):
    """Write a single table section for fastMD (nm, kJ/mol — no unit conversion).

    fastMD expects the same text format as LAMMPS TABLE but uses nm/kJ/mol
    internally, so we skip the Angstrom/kcal conversion applied by
    write_lammps_table.

    Args:
        filepath: Output path; appends if file exists.
        x_vals: Independent variable in nm.
        y_vals: Energy in kJ/mol.
        keyword: Section keyword (e.g. "PAIR_0").
        force_vals: Precomputed f_div_r values. If None, computed via np.gradient.
    """
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    N = len(x)
    if force_vals is not None:
        force = np.asarray(force_vals, dtype=float)
    else:
        dydx = np.gradient(y, x)
        # fastMD table kernel computes fx = f_scalar * dx (no /r division),
        # so it expects f_scalar = (-dE/dr) / r, not -dE/dr.
        with np.errstate(divide='ignore', invalid='ignore'):
            force = -dydx / np.where(x > 0, x, np.inf)
        force = np.nan_to_num(force, nan=0.0, posinf=0.0, neginf=0.0)

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a") as f:
        f.write(f"{keyword}\n")
        f.write(f"N {N} R {x[0]:.10e} {x[-1]:.10e}\n\n")
        for i in range(N):
            f.write(f"{i + 1} {x[i]:.10e} {y[i]:.10e} {force[i]:.10e}\n")
        f.write("\n")


def _parse_restart_velocities(filepath: str) -> Optional[np.ndarray]:
    """Parse Velocities section from a fastMD restart LAMMPS data file.

    Args:
        filepath: Path to restart_final.data

    Returns:
        (N,3) float32 array in nm/ps, or None if no Velocities section found.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    in_velocities = False
    velocities = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "Velocities":
            in_velocities = True
            continue
        if in_velocities:
            if stripped[0].isupper():  # next section header (Bonds, Angles, etc.)
                break
            parts = stripped.split()
            if len(parts) >= 4:
                velocities.append([float(parts[1]), float(parts[2]), float(parts[3])])

    if not velocities:
        return None
    return np.array(velocities, dtype=np.float32)


class FastMDSampler:
    """Molecular dynamics sampler that runs fastMD via subprocess.

    Public API (mirrors LAMMPSSampler):
        run(steps) -> Trajectory
        get_trajectory() -> Trajectory
        get_final_system() -> AtomicSystem
        set_system(system) -> None
        update_potentials(energy_params, energy_objects) -> None
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
        fastmd_exe: Optional[str] = None,
        work_dir: Optional[str] = None,
        random_seed: int = 0,
        skin: float = 0.3,
    ) -> None:
        self._system = system
        self.energy_params = energy_params
        self.energy_objects = energy_objects
        self.topology = topology
        self.ensemble = ensemble.lower()
        self.thermostat = thermostat.lower()
        self.temperature = temperature       # K
        self.timestep = timestep             # fs
        self.friction = friction             # fs (Langevin damping)
        self.cutoff = cutoff                 # nm
        self.r_onset = r_onset               # nm
        self.mol_ids = mol_ids
        self.trajectory_path = trajectory
        self.logfile = logfile
        self.loginterval = loginterval
        if fastmd_exe is None:
            from fastmd import get_binary_path

            fastmd_exe = get_binary_path()
        self.fastmd_exe = fastmd_exe
        self.random_seed = random_seed if random_seed != 0 else 12345
        self.skin = skin                     # nm

        # Convert to fastMD units
        self._kT = temperature * KB                 # K -> kJ/mol
        self._dt_ps = timestep * FS_TO_PS           # fs -> ps
        self._friction_ps = friction * FS_TO_PS     # fs -> ps

        if energy_params is None and energy_objects is None:
            raise ValueError(
                "Either energy_params or energy_objects must be provided"
            )

        if work_dir is None:
            self._work_dir_obj = tempfile.TemporaryDirectory(
                prefix="diffcg_fastmd_"
            )
            self._work_dir = self._work_dir_obj.name
        else:
            self._work_dir = work_dir
            os.makedirs(work_dir, exist_ok=True)
            self._work_dir_obj = None

        self._last_trajectory: Optional[Trajectory] = None
        self._final_velocities: Optional[jnp.ndarray] = None

        logger.debug(
            "FastMDSampler: ensemble=%s thermostat=%s T=%.1fK (kT=%.4f) "
            "dt=%.4fps friction=%.4fps exe=%s",
            self.ensemble, self.thermostat, temperature,
            self._kT, self._dt_ps, self._friction_ps, fastmd_exe,
        )

    def _generate_config(self, steps: int) -> str:
        return _DEFAULT_CONFIG_TEMPLATE.format(
            natoms=self._system.n_atoms,
            ntypes=int(np.asarray(self._system.Z).max()) + 1,
            rc=self.cutoff,
            skin=self.skin,
            dt=self._dt_ps,
            nsteps=steps,
            dump_freq=self.loginterval,
            thermo_freq=self.loginterval,
            ensemble_line=self._make_ensemble_line(),
            table_lines=self._make_table_config_lines(),
        )

    def _make_ensemble_line(self) -> str:
        if self.ensemble == "nve":
            raise ValueError("fastMD does not support NVE ensemble")
        if self.ensemble == "nvt":
            if self.thermostat == "langevin":
                kT = self._kT
                damp = self._friction_ps
                return (
                    f"nvt_langevin {kT:.6f} {kT:.6f} "
                    f"{damp:.6f} {self.random_seed}"
                )
            if self.thermostat in ("nose-hoover", "nosehoover", "nh"):
                kT = self._kT
                damp = 100.0 * self._dt_ps
                return f"nvt_nh {kT:.6f} {kT:.6f} {damp:.6f}"
            raise ValueError(
                f"Unsupported thermostat: {self.thermostat}"
            )
        raise ValueError(f"Unsupported ensemble: {self.ensemble}")

    def _make_table_config_lines(self) -> str:
        """Generate 'table ti tj filename keyword' config lines.

        Merges tables from all pair energy objects that share the same
        type pair (e.g. TabulatedPairEnergy + GenericRepulsionEnergy).
        """
        if self.energy_objects is None:
            return "# No table potentials"

        pair_tables = []
        for obj in self.energy_objects:
            spec = obj.to_lammps()
            if spec.get("interaction_type") != "pair":
                continue
            if spec.get("lammps_style") == "table":
                pair_tables.extend(spec["tables"])

        if not pair_tables:
            return "# No table potentials"

        # Deduplicate by keyword
        seen = set()
        lines = []
        for t in pair_tables:
            kw = t["keyword"]
            if kw in seen:
                continue
            seen.add(kw)
            types = t.get("types")
            ti = tj = 0
            if types is not None:
                ti, tj = types
            lines.append(f"table {ti} {tj} pair_table.txt {kw}")
        return "\n".join(lines)

    def _write_input_files(self, iter_dir: str, steps: int) -> None:
        """Write system.data, pair_table.txt, and fastmd.conf."""
        self._write_data_file(os.path.join(iter_dir, "system.data"))
        self._write_table_files(iter_dir)
        config = self._generate_config(steps)
        with open(os.path.join(iter_dir, "fastmd.conf"), "w") as f:
            f.write(config)

    def _write_data_file(self, filepath: str) -> None:
        """Write LAMMPS-format data file (atom_style full) in nm.

        Unlike write_lammps_data() which converts to Angstroms, this
        writes positions directly in fastMD-native nm units.
        """
        R = np.asarray(self._system.R)
        n_atoms = R.shape[0]
        atom_types = np.asarray(self._system.Z)
        n_atom_types = int(atom_types.max()) + 1

        masses_arr = (
            np.asarray(self._system.masses)
            if self._system.masses is not None
            else np.ones(n_atoms)
        )
        type_masses = {}
        for i in range(n_atoms):
            t = int(atom_types[i])
            if t not in type_masses:
                type_masses[t] = float(masses_arr[i])

        mol_ids_1 = (
            np.asarray(self.mol_ids, dtype=int) + 1
            if self.mol_ids is not None
            else np.ones(n_atoms, dtype=int)
        )

        if self._system.cell is not None:
            cell = np.asarray(self._system.cell)
            lx = float(cell[0, 0])
            ly = float(cell[1, 1])
            lz = float(cell[2, 2])
        else:
            padding = 1.0
            lx = float(R[:, 0].max() - R[:, 0].min()) + 2 * padding
            ly = float(R[:, 1].max() - R[:, 1].min()) + 2 * padding
            lz = float(R[:, 2].max() - R[:, 2].min()) + 2 * padding

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            f.write("LAMMPS data file via DiffCG (fastMD, nm)\n\n")
            f.write(f"{n_atoms} atoms\n")
            f.write(f"{n_atom_types} atom types\n")
            f.write(f"\n0.0 {lx:.6f} xlo xhi\n")
            f.write(f"0.0 {ly:.6f} ylo yhi\n")
            f.write(f"0.0 {lz:.6f} zlo zhi\n")
            f.write("\nMasses\n\n")
            for t in sorted(type_masses.keys()):
                f.write(f"{t + 1} {type_masses[t]:.4f}\n")
            f.write("\nAtoms # full\n\n")
            for i in range(n_atoms):
                f.write(
                    f"{i + 1} {int(mol_ids_1[i])} "
                    f"{int(atom_types[i]) + 1} 0.0 "
                    f"{R[i, 0]:.6f} {R[i, 1]:.6f} {R[i, 2]:.6f}\n"
                )

            if self._system.velocities is not None:
                vel = np.asarray(self._system.velocities)
                f.write("\nVelocities\n\n")
                for i in range(n_atoms):
                    f.write(
                        f"{i + 1} {vel[i, 0]:.6f} {vel[i, 1]:.6f} {vel[i, 2]:.6f}\n"
                    )

    def _write_table_files(self, iter_dir: str) -> None:
        """Write pair_table.txt in nm/kJ/mol.

        Extracts table data from energy_objects via their to_lammps()
        method (which returns kJ/mol values), then merges tables with
        the same keyword by summing y-values. This handles the water
        model case where both TabulatedPairEnergy and
        GenericRepulsionEnergy contribute to the same pair type.
        """
        if self.energy_objects is None:
            return

        # Collect all pair tables from energy objects
        merged = {}  # keyword -> {"x": array, "y": array, "force": array or None}
        for obj in self.energy_objects:
            spec = obj.to_lammps()
            if spec.get("interaction_type") != "pair":
                continue
            if spec.get("lammps_style") != "table":
                continue
            for t in spec["tables"]:
                kw = t["keyword"]
                if kw not in merged:
                    merged[kw] = {"x": t["x"], "y": np.zeros_like(t["y"]), "force": None}
                merged[kw]["y"] = merged[kw]["y"] + np.asarray(t["y"])
                if "force_vals" in t:
                    fv = np.asarray(t["force_vals"])
                    if merged[kw]["force"] is None:
                        merged[kw]["force"] = np.zeros_like(fv)
                    merged[kw]["force"] = merged[kw]["force"] + fv
                else:
                    merged[kw]["force"] = None  # at least one contributor missing analytic forces

        if not merged:
            return

        filepath = os.path.join(iter_dir, "pair_table.txt")
        open(filepath, "w").close()  # clear
        for kw, m in merged.items():
            _write_fastmd_table(filepath, m["x"], m["y"], kw, force_vals=m["force"])

    def run(self, steps: int) -> Trajectory:
        """Run fastMD for *steps* MD steps.

        Returns the trajectory read from traj.bin.
        """
        self._write_input_files(self._work_dir, steps)

        log_path = os.path.join(self._work_dir, "fastmd.log")
        result = subprocess.run(
            [self.fastmd_exe, "fastmd.conf"],
            cwd=self._work_dir,
            capture_output=True,
            text=True,
        )

        with open(log_path, "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n--- STDERR ---\n")
                f.write(result.stderr)

        if result.returncode != 0:
            raise RuntimeError(
                f"fastMD failed (return code {result.returncode}).\n"
                f"Working directory: {self._work_dir}\n"
                f"STDERR: {result.stderr[-2000:] if result.stderr else '(empty)'}"
            )

        traj_bin = os.path.join(self._work_dir, "traj.bin")
        if not os.path.exists(traj_bin):
            raise RuntimeError(
                f"fastMD did not produce traj.bin. "
                f"Check {log_path} for errors."
            )

        self._last_trajectory = read_fastmd_trajectory(
            traj_bin,
            Z=self._system.Z,
            masses=self._system.masses,
            pbc=self._system.pbc,
        )

        restart_final = os.path.join(self._work_dir, "restart_final.data")
        if os.path.exists(restart_final):
            self._final_velocities = jnp.array(
                _parse_restart_velocities(restart_final), dtype=jnp.float32
            )
            logger.debug("Parsed velocities from %s", restart_final)
        else:
            self._final_velocities = None

        if self.trajectory_path is not None:
            self._last_trajectory.save(self.trajectory_path)

        logger.debug(
            "fastMD completed, %d frames read",
            len(self._last_trajectory),
        )
        return self._last_trajectory

    def get_trajectory(self) -> Trajectory:
        if self._last_trajectory is None:
            return Trajectory(
                positions=jnp.zeros((0, self._system.n_atoms, 3)),
                Z=self._system.Z,
                cell=self._system.cell,
                masses=self._system.masses,
                pbc=self._system.pbc,
            )
        return self._last_trajectory

    def get_final_system(self) -> AtomicSystem:
        """Return the system at the final frame of the last trajectory."""
        if self._last_trajectory is not None and len(self._last_trajectory) > 0:
            system = self._last_trajectory[-1]
            if self._final_velocities is not None:
                system = replace(system, velocities=self._final_velocities)
            return system
        return self._system

    def set_system(self, system: AtomicSystem) -> None:
        self._system = system

    def update_potentials(
        self,
        energy_params: Optional[Dict] = None,
        energy_objects: Optional[List] = None,
    ) -> None:
        """Update potentials for next run (used in iterative learning)."""
        if energy_objects is not None:
            self.energy_objects = energy_objects
        if energy_params is not None:
            self.energy_params = energy_params
