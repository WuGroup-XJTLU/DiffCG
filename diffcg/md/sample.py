# SPDX-License-Identifier: MIT
# Copyright (c) 2025 WuResearchGroup

"""Molecular dynamics sampling module using JAX-MD.

This module provides the MolecularDynamics class that wraps JAX-MD integrators
for NVE, NVT (Langevin/Nose-Hoover), and NPT (Nose-Hoover) ensembles.

Note: Berendsen thermostat is not supported as JAX-MD does not provide it.
Use 'langevin' or 'nose-hoover' thermostats instead.
"""

from typing import Callable, Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from diffcg.system import AtomicSystem, Trajectory, System
from diffcg.md.jaxmd_sampler import JAXMDSampler, MDResult
from diffcg.md.fastmd_sampler import FastMDSampler
from diffcg.md.gpumd_sampler import GPUMDSampler
from diffcg._core.logger import get_logger
from diffcg._core.constants import BOLTZMANN_KJMOLK

logger = get_logger(__name__)


def maxwell_boltzmann_velocities(
    masses: jnp.ndarray,
    temperature: float,
    key: jnp.ndarray,
    kB: float = BOLTZMANN_KJMOLK,
) -> jnp.ndarray:
    """Sample Maxwell-Boltzmann velocities in pure JAX.

    Args:
        masses: (N,) masses in g/mol
        temperature: target temperature in K
        key: JAX PRNG key
        kB: Boltzmann constant in kJ/(mol*K)

    Returns:
        velocities: (N, 3) in nm/ps (DiffCG internal velocity units)
    """
    N = masses.shape[0]
    # sigma_v = sqrt(kB * T / m) for each atom
    sigma = jnp.sqrt(kB * temperature / masses)  # (N,)
    v = random.normal(key, shape=(N, 3)) * sigma[:, None]  # (N, 3)

    # Zero center-of-mass momentum: v -= sum(m*v) / sum(m)
    total_mass = jnp.sum(masses)
    com_v = jnp.sum(masses[:, None] * v, axis=0) / total_mass
    v = v - com_v[None, :]

    # Rescale to exact target temperature:
    # T_actual = sum(m * v^2) / (3 * N * kB)
    KE2 = jnp.sum(masses[:, None] * v ** 2)
    T_actual = KE2 / (3.0 * N * kB)
    v = v * jnp.sqrt(temperature / jnp.maximum(T_actual, 1e-30))

    return v


class MolecularDynamics:
    """Molecular dynamics class using JAX-MD integrators.

    Note: Berendsen thermostats are not supported. Use 'langevin' or 'nose-hoover'.
    """

    def __init__(
        self,
        system: AtomicSystem,
        *,
        energy_fn: Callable,
        ensemble: str = "nvt",
        thermostat: str = "langevin",
        temperature: float = 300.0,
        starting_temperature: Optional[float] = None,
        timestep: float = 2.0,
        cutoff: float = 1.0,
        pressure: float = 1.01325,
        taut: Optional[float] = None,
        taup: Optional[float] = None,
        trajectory: Optional[str] = None,
        logfile: Optional[str] = None,
        loginterval: int = 1,
        capacity_multiplier: float = 1.25,
        random_seed: int = 0,
        friction: float = 1.0,
        custom_mask_function: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        # Validate thermostat
        if thermostat.lower().startswith('berendsen'):
            raise ValueError(
                "Berendsen thermostat is not supported in JAX-MD. "
                "Use 'langevin' or 'nose-hoover' instead."
            )

        self.ensemble = ensemble.lower()
        self.thermostat = thermostat.lower()
        self.temperature = temperature
        self.timestep = timestep
        self.cutoff = cutoff
        self.trajectory_path = trajectory
        self.logfile = logfile
        self.loginterval = loginterval if loginterval is not None else 1
        self.random_seed = random_seed

        # Store the AtomicSystem
        self._system = system
        self.positions = system.R
        self.atomic_numbers = system.Z
        self.cell = system.cell
        self._masses_jax = system.masses if system.masses is not None else jnp.ones(system.n_atoms, dtype=jnp.float32)

        if starting_temperature is not None:
            self._key_init = random.PRNGKey(random_seed + 9999)
            # velocities are informational; JAX-MD sampler handles its own init
            logger.debug("Initialized velocities at T=%s K", starting_temperature)

        masses = self._masses_jax.astype(jnp.float32)

        self.sampler = JAXMDSampler(
            energy_fn=energy_fn,
            Z=self.atomic_numbers,
            cell=self.cell,
            cutoff=cutoff,
            ensemble=ensemble,
            thermostat=thermostat,
            temperature=temperature,
            timestep=timestep,
            pressure=pressure,
            taut=taut,
            taup=taup,
            friction=friction,
            capacity_multiplier=capacity_multiplier,
            mass=masses,
            custom_mask_function=custom_mask_function,
            **kwargs,
        )

        self._trajectory_positions: Optional[jnp.ndarray] = None
        self._thermo: Optional[dict] = None
        self._md_result: Optional[MDResult] = None
        self._key = random.PRNGKey(random_seed)
        self._neighbor = None
        self._state = None
        self._initial_state = None
        self._initial_neighbor = None

        logger.debug(
            "Created MolecularDynamics: ensemble=%s, thermostat=%s, T=%s K, dt=%s fs",
            self.ensemble,
            self.thermostat if self.ensemble == 'nvt' else 'n/a',
            temperature,
            timestep,
        )

    def run(self, steps: int) -> Trajectory:
        logger.debug("Running MD for %s steps", steps)

        self._key, subkey = random.split(self._key)

        # On first call, use stored initial state if available (restart support)
        init_state = self._initial_state
        init_neighbor = self._initial_neighbor
        if init_state is not None:
            self._initial_state = None
            self._initial_neighbor = None

        result = self.sampler.run(
            R=self.positions,
            steps=steps,
            key=subkey,
            neighbor=init_neighbor if init_neighbor is not None else self._neighbor,
            save_frequency=self.loginterval,
            initial_state=init_state,
        )

        self._md_result = result
        self._neighbor = result.final_neighbors
        self._state = result.final_state
        self.positions = result.final_state.position
        self._trajectory_positions = result.trajectory
        self._thermo = result.thermo

        traj = Trajectory.from_positions(result.trajectory, self._system)

        if self.trajectory_path is not None:
            self._save_trajectory(traj)

        if self.logfile is not None:
            self._write_log(steps)

        logger.debug("MD completed, %s frames saved", len(traj))
        return traj

    def _save_trajectory(self, traj: Trajectory) -> None:
        if self.trajectory_path.endswith('.traj'):
            # Save as ASE .traj for backward compat
            from diffcg.system import trajectory_to_ase
            from ase.io.trajectory import Trajectory as AseTrajectory
            atoms_list = trajectory_to_ase(traj)
            with AseTrajectory(self.trajectory_path, 'w') as ase_traj:
                for atoms in atoms_list:
                    ase_traj.write(atoms)
        else:
            # Default: NPZ format
            traj.save(self.trajectory_path)
        logger.debug("Saved trajectory to %s", self.trajectory_path)

    def _write_log(self, steps: int) -> None:
        thermo = getattr(self, '_thermo', None)
        n_frames = len(self._trajectory_positions) if self._trajectory_positions is not None else 0
        with open(self.logfile, 'w') as f:
            f.write(f"# JAX-MD Simulation Log\n")
            f.write(f"# Ensemble: {self.ensemble} | Thermostat: {self.thermostat} | Temperature: {self.temperature} K\n")
            f.write(f"# Timestep: {self.timestep} fs | Steps: {steps} | Frames: {n_frames}\n")
            if thermo is not None:
                ke = np.asarray(thermo['kinetic_energy'])
                pe = np.asarray(thermo['potential_energy'])
                temp = np.asarray(thermo['temperature'])
                etotal = np.asarray(thermo['total_energy'])
                press = np.asarray(thermo.get('pressure', np.zeros_like(ke)))
                has_pressure = np.any(press != 0.0)
                if has_pressure:
                    f.write("frame,KE(kJ/mol),PE(kJ/mol),E_total(kJ/mol),T(K),P(kJ/mol*nm^3)\n")
                    for i in range(len(ke)):
                        f.write(f"{i},{ke[i]:.6f},{pe[i]:.6f},{etotal[i]:.6f},{temp[i]:.2f},{press[i]:.6f}\n")
                else:
                    f.write("frame,KE(kJ/mol),PE(kJ/mol),E_total(kJ/mol),T(K)\n")
                    for i in range(len(ke)):
                        f.write(f"{i},{ke[i]:.6f},{pe[i]:.6f},{etotal[i]:.6f},{temp[i]:.2f}\n")

    def set_system(self, system: AtomicSystem) -> None:
        self._system = system
        self.positions = system.R

        if system.cell is not None:
            if self.cell is None or not jnp.allclose(system.cell, self.cell):
                self.cell = system.cell
                logger.warning("Cell changed, sampler may need to be recreated for NPT")
        else:
            self.cell = None

        self._neighbor = None

    # Backward compat
    def set_atoms(self, atoms) -> None:
        from diffcg.system import from_ase_atoms
        self.set_system(from_ase_atoms(atoms))

    def get_trajectory(self) -> Trajectory:
        if self._trajectory_positions is None or len(self._trajectory_positions) == 0:
            return Trajectory(
                positions=jnp.zeros((0, self._system.n_atoms, 3)),
                Z=self.atomic_numbers,
                cell=self.cell,
                masses=self._masses_jax,
                pbc=self._system.pbc,
            )
        return Trajectory(
            positions=self._trajectory_positions,
            Z=self.atomic_numbers,
            cell=self.cell,
            masses=self._masses_jax,
            pbc=self._system.pbc,
        )

    def get_trajectory_as_atoms(self):
        """Backward compat: return list of ASE Atoms."""
        from diffcg.system import trajectory_to_ase
        return trajectory_to_ase(self.get_trajectory())

    def get_final_system(self) -> AtomicSystem:
        velocities = None
        if self._state is not None and hasattr(self._state, 'momentum'):
            velocities = self._state.momentum / self._state.mass
        return AtomicSystem(
            R=self.positions,
            Z=self.atomic_numbers,
            cell=self.cell,
            masses=self._masses_jax,
            pbc=self._system.pbc,
            velocities=velocities,
        )

    def get_final_state(self):
        """Return the final JAX-MD integrator state (for restart)."""
        return self._state

    def get_final_neighbors(self):
        """Return the final neighbor list (for restart)."""
        return self._neighbor


def create_molecular_dynamics(
    system: AtomicSystem,
    energy_fn: Callable,
    ensemble: str = "nvt",
    thermostat: str = "langevin",
    temperature: float = 300.0,
    timestep: float = 2.0,
    cutoff: float = 1.0,
    **kwargs,
) -> MolecularDynamics:
    return MolecularDynamics(
        system,
        energy_fn=energy_fn,
        ensemble=ensemble,
        thermostat=thermostat,
        temperature=temperature,
        timestep=timestep,
        cutoff=cutoff,
        **kwargs,
    )


def create_equilibration_run(
    system: AtomicSystem,
    energy_fn: Callable,
    sampler_params: dict,
    cutoff: float,
    custom_mask_function: Optional[Callable] = None,
    sampler_backend: str = "jaxmd",
    lammps_config: Optional[dict] = None,
):
    """Create an MD run configured for equilibration (no trajectory output).

    Args:
        system: Starting atomic system
        energy_fn: Energy function (ignored when sampler_backend="lammps")
        sampler_params: Dict with keys: ensemble, thermostat, temperature,
            starting_temperature (optional), timestep, friction (optional)
        cutoff: Neighbor list cutoff
        sampler_backend: "jaxmd" (default) or "lammps"
        lammps_config: Dict with LAMMPS-specific config (required when backend="lammps").
            Keys: energy_params, topology, r_onset, mol_ids, lammps_exe, work_dir,
            extra_lammps_commands, input_template
    """
    _loginterval = sampler_params.get("loginterval", 100)

    if sampler_backend == "lammps":
        from diffcg.md.lammps_sampler import LAMMPSSampler
        lc = lammps_config or {}
        return LAMMPSSampler(
            system,
            energy_params=lc.get("energy_params"),
            energy_objects=lc.get("energy_objects"),
            topology=lc["topology"],
            ensemble=sampler_params["ensemble"],
            thermostat=sampler_params["thermostat"],
            temperature=sampler_params["temperature"],
            timestep=sampler_params["timestep"],
            friction=sampler_params.get("friction", 1.0),
            cutoff=cutoff,
            r_onset=lc.get("r_onset", cutoff * 0.8),
            mol_ids=lc.get("mol_ids"),
            trajectory=None,
            logfile=None,
            loginterval=_loginterval,
            lammps_exe=lc.get("lammps_exe", "lmp"),
            work_dir=lc.get("work_dir"),
            extra_lammps_commands=lc.get("extra_lammps_commands"),
            input_template=lc.get("input_template"),
            special_bonds=lc.get("special_bonds", "lj 0.0 0.0 0.0"),
        )

    if sampler_backend == "fastmd":
        fc = lammps_config or {}
        return FastMDSampler(
            system,
            energy_params=fc.get("energy_params"),
            energy_objects=fc.get("energy_objects"),
            topology=fc["topology"],
            ensemble=sampler_params["ensemble"],
            thermostat=sampler_params["thermostat"],
            temperature=sampler_params["temperature"],
            timestep=sampler_params["timestep"],
            friction=sampler_params.get("friction", 1.0),
            cutoff=cutoff,
            r_onset=fc.get("r_onset", cutoff * 0.8),
            mol_ids=fc.get("mol_ids"),
            trajectory=None,
            logfile=None,
            loginterval=_loginterval,
            fastmd_exe=fc.get("fastmd_exe"),
            work_dir=fc.get("work_dir"),
            random_seed=sampler_params.get("seed", 0),
        )

    if sampler_backend == "gpumd":
        gc = lammps_config or {}
        return GPUMDSampler(
            system,
            energy_params=gc.get("energy_params"),
            energy_objects=gc.get("energy_objects"),
            topology=gc.get("topology", {}),
            nep_params=gc["nep_params"],
            ensemble=sampler_params["ensemble"],
            thermostat=sampler_params["thermostat"],
            temperature=sampler_params["temperature"],
            timestep=sampler_params["timestep"],
            friction=sampler_params.get("friction", 1.0),
            cutoff=cutoff,
            r_onset=gc.get("r_onset", cutoff * 0.8),
            mol_ids=gc.get("mol_ids"),
            trajectory=None,
            logfile=None,
            loginterval=_loginterval,
            gpumd_exe=gc.get("gpumd_exe", "gpumd"),
            work_dir=gc.get("work_dir"),
            random_seed=sampler_params.get("seed", 0),
        )

    return MolecularDynamics(
        system,
        energy_fn=energy_fn,
        ensemble=sampler_params["ensemble"],
        thermostat=sampler_params["thermostat"],
        temperature=sampler_params["temperature"],
        starting_temperature=sampler_params.get(
            "starting_temperature", sampler_params["temperature"]
        ),
        timestep=sampler_params["timestep"],
        cutoff=cutoff,
        friction=sampler_params.get("friction", 1.0),
        trajectory=None,
        logfile=None,
        loginterval=_loginterval,
        custom_mask_function=custom_mask_function,
    )


def create_production_run(
    system: AtomicSystem,
    energy_fn: Callable,
    sampler_params: dict,
    cutoff: float,
    trajectory: Optional[str] = None,
    logfile: Optional[str] = None,
    loginterval: Optional[int] = None,
    custom_mask_function: Optional[Callable] = None,
    sampler_backend: str = "jaxmd",
    lammps_config: Optional[dict] = None,
    restart_state: Optional[dict] = None,
):
    """Create an MD run configured for production (with trajectory output).

    Args:
        system: Starting atomic system
        energy_fn: Energy function (ignored when sampler_backend="lammps")
        sampler_params: Dict with keys: ensemble, thermostat, temperature,
            starting_temperature (optional), timestep, friction (optional),
            loginterval (optional, fallback)
        cutoff: Neighbor list cutoff
        trajectory: Path for trajectory output
        logfile: Path for log output
        loginterval: Save frequency (overrides sampler_params if given)
        sampler_backend: "jaxmd" (default) or "lammps"
        lammps_config: Dict with LAMMPS-specific config (required when backend="lammps")
        restart_state: Required dict carrying state from equilibration.
            For jaxmd: {'state': <JAX-MD state>, 'neighbor': <neighbor list>}
            For lammps: {'restart_file': <path to restart.lmp>}
    """
    if restart_state is None:
        raise ValueError(
            "restart_state is required for create_production_run(). "
            "Pass the equilibrated state to avoid re-initializing velocities."
        )

    _loginterval = loginterval or sampler_params.get("loginterval", 100)

    if sampler_backend == "lammps":
        from diffcg.md.lammps_sampler import LAMMPSSampler
        lc = lammps_config or {}
        return LAMMPSSampler(
            system,
            energy_params=lc.get("energy_params"),
            energy_objects=lc.get("energy_objects"),
            topology=lc["topology"],
            ensemble=sampler_params["ensemble"],
            thermostat=sampler_params["thermostat"],
            temperature=sampler_params["temperature"],
            timestep=sampler_params["timestep"],
            friction=sampler_params.get("friction", 1.0),
            cutoff=cutoff,
            r_onset=lc.get("r_onset", cutoff * 0.8),
            mol_ids=lc.get("mol_ids"),
            trajectory=trajectory,
            logfile=logfile,
            loginterval=_loginterval,
            lammps_exe=lc.get("lammps_exe", "lmp"),
            work_dir=lc.get("work_dir"),
            extra_lammps_commands=lc.get("extra_lammps_commands"),
            input_template=lc.get("input_template"),
            restart_file=restart_state.get("restart_file"),
            special_bonds=lc.get("special_bonds", "lj 0.0 0.0 0.0"),
        )

    if sampler_backend == "fastmd":
        fc = lammps_config or {}
        return FastMDSampler(
            system,
            energy_params=fc.get("energy_params"),
            energy_objects=fc.get("energy_objects"),
            topology=fc["topology"],
            ensemble=sampler_params["ensemble"],
            thermostat=sampler_params["thermostat"],
            temperature=sampler_params["temperature"],
            timestep=sampler_params["timestep"],
            friction=sampler_params.get("friction", 1.0),
            cutoff=cutoff,
            r_onset=fc.get("r_onset", cutoff * 0.8),
            mol_ids=fc.get("mol_ids"),
            trajectory=trajectory,
            logfile=logfile,
            loginterval=_loginterval,
            fastmd_exe=fc.get("fastmd_exe"),
            work_dir=fc.get("work_dir"),
            random_seed=sampler_params.get("seed", 0),
        )

    if sampler_backend == "gpumd":
        gc = lammps_config or {}
        return GPUMDSampler(
            system,
            energy_params=gc.get("energy_params"),
            energy_objects=gc.get("energy_objects"),
            topology=gc.get("topology", {}),
            nep_params=gc["nep_params"],
            ensemble=sampler_params["ensemble"],
            thermostat=sampler_params["thermostat"],
            temperature=sampler_params["temperature"],
            timestep=sampler_params["timestep"],
            friction=sampler_params.get("friction", 1.0),
            cutoff=cutoff,
            r_onset=gc.get("r_onset", cutoff * 0.8),
            mol_ids=gc.get("mol_ids"),
            trajectory=trajectory,
            logfile=logfile,
            loginterval=_loginterval,
            gpumd_exe=gc.get("gpumd_exe", "gpumd"),
            work_dir=gc.get("work_dir"),
            random_seed=sampler_params.get("seed", 0),
            restart_system=restart_state.get("system") if restart_state else None,
        )

    md = MolecularDynamics(
        system,
        energy_fn=energy_fn,
        ensemble=sampler_params["ensemble"],
        thermostat=sampler_params["thermostat"],
        temperature=sampler_params["temperature"],
        starting_temperature=sampler_params.get(
            "starting_temperature", sampler_params["temperature"]
        ),
        timestep=sampler_params["timestep"],
        cutoff=cutoff,
        friction=sampler_params.get("friction", 1.0),
        trajectory=trajectory,
        logfile=logfile,
        loginterval=_loginterval,
        custom_mask_function=custom_mask_function,
    )
    md._initial_state = restart_state.get("state")
    md._initial_neighbor = restart_state.get("neighbor")
    return md
