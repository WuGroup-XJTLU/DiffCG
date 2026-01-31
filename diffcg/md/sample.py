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
        pressure: float = 1.01325e-4,
        taut: Optional[float] = None,
        taup: Optional[float] = None,
        trajectory: Optional[str] = None,
        logfile: Optional[str] = None,
        loginterval: int = 1,
        capacity_multiplier: float = 1.25,
        random_seed: int = 0,
        friction: float = 1.0,
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
            **kwargs,
        )

        self._trajectory_positions: Optional[jnp.ndarray] = None
        self._md_result: Optional[MDResult] = None
        self._key = random.PRNGKey(random_seed)
        self._neighbor = None
        self._state = None

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
        result = self.sampler.run(
            R=self.positions,
            steps=steps,
            key=subkey,
            neighbor=self._neighbor,
            save_frequency=self.loginterval,
        )

        self._md_result = result
        self._neighbor = result.final_neighbors
        self._state = result.final_state
        self.positions = result.final_state.position
        self._trajectory_positions = result.trajectory

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
        with open(self.logfile, 'a') as f:
            f.write(f"# JAX-MD Simulation Log\n")
            f.write(f"# Ensemble: {self.ensemble}\n")
            f.write(f"# Thermostat: {self.thermostat}\n")
            f.write(f"# Temperature: {self.temperature} K\n")
            f.write(f"# Timestep: {self.timestep} fs\n")
            f.write(f"# Steps: {steps}\n")
            f.write(f"# Frames saved: {len(self._trajectory_positions) if self._trajectory_positions is not None else 0}\n")

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
        return AtomicSystem(
            R=self.positions,
            Z=self.atomic_numbers,
            cell=self.cell,
            masses=self._masses_jax,
            pbc=self._system.pbc,
        )


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
) -> MolecularDynamics:
    """Create an MD run configured for equilibration (no trajectory output).

    Args:
        system: Starting atomic system
        energy_fn: Energy function
        sampler_params: Dict with keys: ensemble, thermostat, temperature,
            starting_temperature (optional), timestep, friction (optional)
        cutoff: Neighbor list cutoff
    """
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
        loginterval=1,
    )


def create_production_run(
    system: AtomicSystem,
    energy_fn: Callable,
    sampler_params: dict,
    cutoff: float,
    trajectory: Optional[str] = None,
    logfile: Optional[str] = None,
    loginterval: Optional[int] = None,
) -> MolecularDynamics:
    """Create an MD run configured for production (with trajectory output).

    Args:
        system: Starting atomic system
        energy_fn: Energy function
        sampler_params: Dict with keys: ensemble, thermostat, temperature,
            starting_temperature (optional), timestep, friction (optional),
            loginterval (optional, fallback)
        cutoff: Neighbor list cutoff
        trajectory: Path for trajectory output
        logfile: Path for log output
        loginterval: Save frequency (overrides sampler_params if given)
    """
    _loginterval = loginterval or sampler_params.get("loginterval", 100)
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
        trajectory=trajectory,
        logfile=logfile,
        loginterval=_loginterval,
    )
