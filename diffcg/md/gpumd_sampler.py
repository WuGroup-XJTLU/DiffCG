"""GPUMD-based molecular dynamics sampler.

Runs the GPUMD CUDA MD engine via subprocess. Uses the same public
interface as LAMMPSSampler for interchangeable use in DiffSim workflows.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import jax.numpy as jnp

from diffcg._core.logger import get_logger
from diffcg._core.units import NM_TO_ANGSTROM, ANGSTROM_TO_NM, KJMOL_TO_EV
from diffcg.system import AtomicSystem, Trajectory
from diffcg.io.gpumd_writer import write_xyz_in
from diffcg.io.gpumd_reader import read_dump_xyz
from diffcg.io.nep import write_nep

logger = get_logger(__name__)


class GPUMDSampler:
    """Molecular dynamics sampler that runs GPUMD via subprocess.

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
        nep_params: Optional[Dict] = None,
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
        gpumd_exe: Optional[str] = None,
        work_dir: Optional[str] = None,
        random_seed: int = 0,
        restart_system: Optional[AtomicSystem] = None,
    ) -> None:
        self._system = system
        self.energy_params = energy_params
        self.energy_objects = energy_objects
        self.topology = topology
        self.nep_params = nep_params
        self.ensemble = ensemble.lower()
        self.thermostat = thermostat.lower()
        self.temperature = temperature       # K
        self.timestep = timestep             # fs
        self.friction = friction
        self.cutoff = cutoff                 # nm
        self.r_onset = r_onset
        self.mol_ids = mol_ids
        self.trajectory_path = trajectory
        self.logfile = logfile
        self.loginterval = loginterval
        if gpumd_exe is None:
            from diffcg._binaries import get_gpumd_path

            gpumd_exe = get_gpumd_path()
        self.gpumd_exe = gpumd_exe
        self.random_seed = random_seed if random_seed != 0 else 12345
        self._restart_system = restart_system

        if nep_params is None:
            raise ValueError("nep_params dict is required for GPUMDSampler")

        if work_dir is None:
            self._work_dir_obj = tempfile.TemporaryDirectory(prefix="diffcg_gpumd_")
            self._work_dir = self._work_dir_obj.name
        else:
            self._work_dir = work_dir
            os.makedirs(work_dir, exist_ok=True)
            self._work_dir_obj = None

        self._last_trajectory: Optional[Trajectory] = None

        logger.debug(
            "GPUMDSampler: ensemble=%s thermostat=%s T=%.1fK dt=%.1ffs exe=%s",
            self.ensemble, self.thermostat, temperature, timestep, gpumd_exe,
        )

    def _generate_run_in(self, steps: int) -> str:
        """Generate the GPUMD run.in control file."""
        lines = []

        # Potential
        lines.append("potential nep.txt")

        # Velocity initialization
        if self._restart_system is not None and self._restart_system.velocities is not None:
            lines.append("velocity 0")
        else:
            lines.append(f"velocity {self.temperature}")

        # Ensemble
        if self.ensemble == "nve":
            lines.append("ensemble nve")
        elif self.ensemble == "nvt":
            if self.thermostat == "langevin":
                lines.append(f"ensemble nvt_lan {self.temperature} {self.temperature} {self.friction}")
            elif self.thermostat in ("nose-hoover", "nosehoover", "nh"):
                taut = 100.0 * self.timestep
                lines.append(f"ensemble nvt_nh {self.temperature} {self.temperature} {taut}")
            else:
                raise ValueError(f"Unknown thermostat: {self.thermostat}")
        else:
            raise ValueError(f"Unsupported ensemble: {self.ensemble}")

        lines.append(f"time_step {self.timestep}")
        lines.append(f"dump_thermo {self.loginterval}")
        lines.append(f"dump_xyz -1 0 {self.loginterval} dump_xyz.xyz")
        lines.append(f"run {steps}")

        return "\n".join(lines) + "\n"

    def _write_input_files(self, steps: int) -> None:
        """Write xyz.in, nep.txt, and run.in to work directory."""
        # GPUMD reads the structure from model.xyz by default
        system = self._restart_system if self._restart_system is not None else self._system
        write_xyz_in(system, os.path.join(self._work_dir, "model.xyz"))

        # nep.txt — convert cutoffs from nm (diffcg internal) to Angstroms (GPUMD)
        import copy
        nep_params_ang = copy.deepcopy(self.nep_params)
        nm_to_a = NM_TO_ANGSTROM
        nep_params_ang["rc_radial"] = [r * nm_to_a for r in nep_params_ang["rc_radial"]]
        nep_params_ang["rc_angular"] = [r * nm_to_a for r in nep_params_ang["rc_angular"]]
        if "soft_repulsion" in nep_params_ang and nep_params_ang["soft_repulsion"] is not None:
            sr = nep_params_ang["soft_repulsion"]
            sr["sigma"] *= nm_to_a
            sr["r_onset"] *= nm_to_a
            sr["r_cutoff"] *= nm_to_a
            sr["epsilon"] *= KJMOL_TO_EV
        write_nep(os.path.join(self._work_dir, "nep.txt"), nep_params_ang)

        # run.in
        run_in = self._generate_run_in(steps)
        with open(os.path.join(self._work_dir, "run.in"), "w") as f:
            f.write(run_in)

    def run(self, steps: int) -> Trajectory:
        """Run GPUMD for *steps* MD steps."""
        self._write_input_files(steps)

        log_path = os.path.join(self._work_dir, "gpumd.log")
        result = subprocess.run(
            [self.gpumd_exe],
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
                f"GPUMD failed (return code {result.returncode}).\n"
                f"Working directory: {self._work_dir}\n"
                f"STDERR: {result.stderr[-2000:] if result.stderr else '(empty)'}"
            )

        dump_file = os.path.join(self._work_dir, "dump_xyz.xyz")
        if not os.path.exists(dump_file):
            raise RuntimeError(
                f"GPUMD did not produce dump_xyz.xyz. Check {log_path}."
            )

        self._last_trajectory = read_dump_xyz(
            dump_file,
            Z=self._system.Z,
            masses=self._system.masses,
            pbc=self._system.pbc,
        )

        if self.trajectory_path is not None:
            self._last_trajectory.save(self.trajectory_path)

        logger.debug("GPUMD completed, %d frames read", len(self._last_trajectory))
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
        if self._last_trajectory is not None and len(self._last_trajectory) > 0:
            return self._last_trajectory[-1]
        return self._system

    def set_system(self, system: AtomicSystem) -> None:
        self._system = system

    def update_potentials(
        self,
        energy_params: Optional[Dict] = None,
        energy_objects: Optional[List] = None,
    ) -> None:
        """Update potentials for next run.

        For GPUMD, the nep_params dict (passed at init) is the primary
        potential. This method exists for interface compatibility.
        """
        if energy_params is not None:
            if "nep_params" in energy_params:
                self.nep_params = energy_params["nep_params"]
            self.energy_params = energy_params
        if energy_objects is not None:
            self.energy_objects = energy_objects
