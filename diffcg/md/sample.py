# SPDX-License-Identifier: MIT
# Copyright (c) 2025 WuResearchGroup

"""Molecular dynamics sampling module using JAX-MD.

This module provides the MolecularDynamics class that wraps JAX-MD integrators
for NVE, NVT (Langevin/Nose-Hoover), and NPT (Nose-Hoover) ensembles.

Note: Berendsen thermostat is not supported as JAX-MD does not provide it.
Use 'langevin' or 'nose-hoover' thermostats instead.
"""

from typing import Callable, Optional, List
import pickle

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from ase import Atoms
from ase.io.trajectory import Trajectory

from diffcg.system import System
from diffcg.md.jaxmd_sampler import JAXMDSampler, MDResult
from diffcg.util.logger import get_logger

logger = get_logger(__name__)


class TrajectoryObserver:
    """Trajectory observer that saves intermediate structures during MD."""

    def __init__(self, atoms: Atoms) -> None:
        self.atoms = atoms
        self.energies: List[float] = []
        self.forces: List[np.ndarray] = []
        self.stresses: List[np.ndarray] = []
        self.magmoms: List[np.ndarray] = []
        self.atom_positions: List[np.ndarray] = []
        self.cells: List[np.ndarray] = []

    def __call__(self) -> None:
        self.energies.append(self.compute_energy())
        self.forces.append(self.atoms.get_forces())
        self.stresses.append(self.atoms.get_stress())
        self.magmoms.append(self.atoms.get_magnetic_moments())
        self.atom_positions.append(self.atoms.get_positions())
        self.cells.append(self.atoms.get_cell()[:])

    def __len__(self) -> int:
        return len(self.energies)

    def compute_energy(self) -> float:
        return self.atoms.get_potential_energy()

    def save(self, filename: str) -> None:
        out_pkl = {
            "energy": self.energies,
            "forces": self.forces,
            "stresses": self.stresses,
            "magmoms": self.magmoms,
            "atom_positions": self.atom_positions,
            "cell": self.cells,
            "atomic_number": self.atoms.get_atomic_numbers(),
        }
        with open(filename, "wb") as file:
            pickle.dump(out_pkl, file)


def trajectory_to_atoms(
    positions_trajectory: jnp.ndarray,
    atomic_numbers: jnp.ndarray,
    cell: Optional[jnp.ndarray],
    pbc: bool = True,
) -> List[Atoms]:
    """Convert JAX trajectory array to list of ASE Atoms."""
    atoms_list = []
    positions_np = np.array(positions_trajectory)
    numbers_np = np.array(atomic_numbers)

    if cell is not None:
        cell_np = np.array(cell).T
    else:
        cell_np = None
        pbc = False

    for pos in positions_np:
        atoms = Atoms(positions=pos, numbers=numbers_np, cell=cell_np, pbc=pbc)
        atoms_list.append(atoms)

    return atoms_list


def atoms_to_trajectory(atoms_list: List[Atoms]) -> tuple:
    """Convert list of ASE Atoms to JAX arrays."""
    positions = jnp.stack([jnp.array(a.get_positions()) for a in atoms_list])
    atomic_numbers = jnp.array(atoms_list[0].get_atomic_numbers())

    if atoms_list[0].get_pbc().any():
        cell = jnp.array(atoms_list[0].get_cell()[:].T)
    else:
        cell = None

    return positions, atomic_numbers, cell


class MolecularDynamics:
    """Molecular dynamics class using JAX-MD integrators.

    Note: Berendsen thermostats are not supported. Use 'langevin' or 'nose-hoover'.
    """

    def __init__(
        self,
        atoms: Atoms,
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

        self._original_atoms = atoms.copy()
        self.atoms = atoms

        self.positions = jnp.array(atoms.get_positions())
        self.atomic_numbers = jnp.array(atoms.get_atomic_numbers())

        if atoms.get_pbc().any():
            self.cell = jnp.array(atoms.get_cell()[:].T)
        else:
            self.cell = None

        if starting_temperature is not None:
            self._initialize_velocities(starting_temperature)

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
            **kwargs,
        )

        self.trajectory: List[jnp.ndarray] = []
        self._md_result: Optional[MDResult] = None
        self._key = random.PRNGKey(random_seed)
        self._neighbor = None
        self._state = None

        logger.info(
            "Created MolecularDynamics: ensemble=%s, thermostat=%s, T=%s K, dt=%s fs",
            self.ensemble,
            self.thermostat if self.ensemble == 'nvt' else 'n/a',
            temperature,
            timestep,
        )

    def _initialize_velocities(self, temperature: float) -> None:
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
        MaxwellBoltzmannDistribution(self.atoms, temperature_K=temperature, force_temp=True)
        Stationary(self.atoms)
        logger.debug("Initialized velocities at T=%s K", temperature)

    def run(self, steps: int) -> jnp.ndarray:
        logger.info("Running MD for %s steps", steps)

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
        self.atoms.set_positions(np.array(self.positions))
        self.trajectory = result.trajectory

        if self.trajectory_path is not None:
            self._save_trajectory(result.trajectory)

        if self.logfile is not None:
            self._write_log(steps)

        logger.info("MD completed, %s frames saved", len(result.trajectory))
        return result.trajectory

    def _save_trajectory(self, trajectory: jnp.ndarray) -> None:
        atoms_list = trajectory_to_atoms(
            trajectory,
            self.atomic_numbers,
            self.cell,
            pbc=self._original_atoms.get_pbc().any(),
        )

        with Trajectory(self.trajectory_path, 'w') as traj:
            for atoms in atoms_list:
                traj.write(atoms)

        logger.debug("Saved trajectory to %s", self.trajectory_path)

    def _write_log(self, steps: int) -> None:
        with open(self.logfile, 'a') as f:
            f.write(f"# JAX-MD Simulation Log\n")
            f.write(f"# Ensemble: {self.ensemble}\n")
            f.write(f"# Thermostat: {self.thermostat}\n")
            f.write(f"# Temperature: {self.temperature} K\n")
            f.write(f"# Timestep: {self.timestep} fs\n")
            f.write(f"# Steps: {steps}\n")
            f.write(f"# Frames saved: {len(self.trajectory)}\n")

    def set_atoms(self, atoms: Atoms) -> None:
        self.atoms = atoms
        self.positions = jnp.array(atoms.get_positions())

        if atoms.get_pbc().any():
            new_cell = jnp.array(atoms.get_cell()[:].T)
            if self.cell is None or not jnp.allclose(new_cell, self.cell):
                self.cell = new_cell
                logger.warning("Cell changed, sampler may need to be recreated for NPT")
        else:
            self.cell = None

        self._neighbor = None

    def get_trajectory_as_atoms(self) -> List[Atoms]:
        if len(self.trajectory) == 0:
            return []

        return trajectory_to_atoms(
            self.trajectory,
            self.atomic_numbers,
            self.cell,
            pbc=self._original_atoms.get_pbc().any(),
        )

    def get_final_system(self) -> System:
        return System(R=self.positions, Z=self.atomic_numbers, cell=self.cell)


def create_molecular_dynamics(
    atoms: Atoms,
    energy_fn: Callable,
    ensemble: str = "nvt",
    thermostat: str = "langevin",
    temperature: float = 300.0,
    timestep: float = 2.0,
    cutoff: float = 1.0,
    **kwargs,
) -> MolecularDynamics:
    return MolecularDynamics(
        atoms=atoms,
        energy_fn=energy_fn,
        ensemble=ensemble,
        thermostat=thermostat,
        temperature=temperature,
        timestep=timestep,
        cutoff=cutoff,
        **kwargs,
    )
