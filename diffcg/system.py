# SPDX-License-Identifier: MIT
# Copyright (c) 2025 WuResearchGroup

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, List

import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

from diffcg._core.periodic import make_displacement


# ---------------------------------------------------------------------------
# AtomicSystem — frozen dataclass, registered as JAX pytree
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AtomicSystem:
    """Single-frame atomic system with explicit units.

    Fields:
        R: (N, 3) positions in nm
        Z: (N,) atomic numbers, int32
        cell: (3, 3) column-major in nm (or None for non-periodic)
        masses: (N,) in g/mol (or None)
        pbc: bool
        velocities: optional (N, 3) in nm/ps
    """
    R: jnp.ndarray
    Z: jnp.ndarray
    cell: Optional[jnp.ndarray] = None
    masses: Optional[jnp.ndarray] = None
    pbc: bool = True
    velocities: Optional[jnp.ndarray] = None

    @property
    def n_atoms(self) -> int:
        return self.R.shape[0]

    @property
    def volume(self) -> Optional[float]:
        if self.cell is None:
            return None
        return float(jnp.abs(jnp.linalg.det(self.cell)))


# Register AtomicSystem as JAX pytree
def _atomic_system_flatten(system: AtomicSystem):
    children = (system.R, system.Z, system.cell, system.masses, system.velocities)
    aux = (system.pbc,)
    return children, aux


def _atomic_system_unflatten(aux, children):
    R, Z, cell, masses, velocities = children
    (pbc,) = aux
    return AtomicSystem(R=R, Z=Z, cell=cell, masses=masses, pbc=pbc, velocities=velocities)


jax.tree_util.register_pytree_node(
    AtomicSystem,
    _atomic_system_flatten,
    _atomic_system_unflatten,
)

# Backward-compatible alias — energy functions access system.R, system.Z, system.cell
System = AtomicSystem


# ---------------------------------------------------------------------------
# Trajectory — stores (T, N, 3) positions + shared metadata
# ---------------------------------------------------------------------------

class Trajectory:
    """Multi-frame trajectory with shared metadata.

    Fields:
        positions: (T, N, 3) in nm
        Z: (N,) atomic numbers, int32
        cell: (3, 3) column-major in nm (or None)
        masses: (N,) in g/mol (or None)
        pbc: bool
    """

    def __init__(
        self,
        positions: jnp.ndarray,
        Z: jnp.ndarray,
        cell: Optional[jnp.ndarray] = None,
        masses: Optional[jnp.ndarray] = None,
        pbc: bool = True,
    ):
        self.positions = positions
        self.Z = Z
        self.cell = cell
        self.masses = masses
        self.pbc = pbc

    def __len__(self) -> int:
        return self.positions.shape[0]

    def __getitem__(self, idx) -> Union[AtomicSystem, "Trajectory"]:
        if isinstance(idx, (int, np.integer)):
            return AtomicSystem(
                R=self.positions[idx],
                Z=self.Z,
                cell=self.cell,
                masses=self.masses,
                pbc=self.pbc,
            )
        elif isinstance(idx, slice):
            return Trajectory(
                positions=self.positions[idx],
                Z=self.Z,
                cell=self.cell,
                masses=self.masses,
                pbc=self.pbc,
            )
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

    def to_batched_system(self) -> AtomicSystem:
        """Return a batched AtomicSystem suitable for vmap/scan.

        R: (T, N, 3), Z: (T, N), cell: (T, 3, 3)
        """
        T = len(self)
        Z_batched = jnp.broadcast_to(self.Z, (T, self.Z.shape[0]))
        cell_batched = (
            jnp.broadcast_to(self.cell, (T, 3, 3)) if self.cell is not None else None
        )
        masses_batched = (
            jnp.broadcast_to(self.masses, (T, self.masses.shape[0]))
            if self.masses is not None
            else None
        )
        return AtomicSystem(
            R=self.positions,
            Z=Z_batched,
            cell=cell_batched,
            masses=masses_batched,
            pbc=self.pbc,
        )

    @staticmethod
    def from_positions(positions: jnp.ndarray, template_system: AtomicSystem) -> "Trajectory":
        return Trajectory(
            positions=positions,
            Z=template_system.Z,
            cell=template_system.cell,
            masses=template_system.masses,
            pbc=template_system.pbc,
        )

    def save(self, path: str) -> None:
        data = {
            "positions": np.array(self.positions),
            "Z": np.array(self.Z),
            "pbc": np.array(self.pbc),
        }
        if self.cell is not None:
            data["cell"] = np.array(self.cell)
        if self.masses is not None:
            data["masses"] = np.array(self.masses)
        np.savez(path, **data)

    @staticmethod
    def load(path: str) -> "Trajectory":
        data = np.load(path, allow_pickle=False)
        cell = jnp.array(data["cell"]) if "cell" in data else None
        masses = jnp.array(data["masses"]) if "masses" in data else None
        pbc = bool(data["pbc"]) if "pbc" in data else True
        return Trajectory(
            positions=jnp.array(data["positions"]),
            Z=jnp.array(data["Z"]),
            cell=cell,
            masses=masses,
            pbc=pbc,
        )


# ---------------------------------------------------------------------------
# ASE interop (optional — only used at I/O boundaries)
# ---------------------------------------------------------------------------

def from_ase_atoms(atoms, dtype=jnp.float32) -> AtomicSystem:
    """Convert an ASE Atoms object to an AtomicSystem."""
    R = jnp.array(atoms.get_positions(), dtype=dtype)
    Z = jnp.array(atoms.get_atomic_numbers(), dtype=jnp.int32)

    if atoms.get_pbc().any():
        cell = jnp.array(atoms.get_cell().array.T, dtype=dtype)  # column-major
    else:
        cell = None

    masses = jnp.array(atoms.get_masses(), dtype=dtype)
    pbc = bool(atoms.get_pbc().any())

    return AtomicSystem(R=R, Z=Z, cell=cell, masses=masses, pbc=pbc)


def to_ase_atoms(system: AtomicSystem):
    """Convert an AtomicSystem to an ASE Atoms object."""
    from ase import Atoms as AseAtoms

    positions = np.array(system.R)
    numbers = np.array(system.Z)

    if system.cell is not None:
        cell = np.array(system.cell).T  # column-major → row-major for ASE
    else:
        cell = None

    atoms = AseAtoms(positions=positions, numbers=numbers, cell=cell, pbc=system.pbc)
    if system.masses is not None:
        atoms.set_masses(np.array(system.masses))
    return atoms


def trajectory_from_ase(atoms_list: list, dtype=jnp.float32) -> Trajectory:
    """Convert a list of ASE Atoms to a Trajectory."""
    positions = jnp.stack([jnp.array(a.get_positions(), dtype=dtype) for a in atoms_list])
    Z = jnp.array(atoms_list[0].get_atomic_numbers(), dtype=jnp.int32)

    if atoms_list[0].get_pbc().any():
        cell = jnp.array(atoms_list[0].get_cell().array.T, dtype=dtype)
    else:
        cell = None

    masses = jnp.array(atoms_list[0].get_masses(), dtype=dtype)
    pbc = bool(atoms_list[0].get_pbc().any())

    return Trajectory(positions=positions, Z=Z, cell=cell, masses=masses, pbc=pbc)


def trajectory_to_ase(traj: Trajectory) -> list:
    """Convert a Trajectory to a list of ASE Atoms."""
    return [to_ase_atoms(traj[i]) for i in range(len(traj))]


# ---------------------------------------------------------------------------
# Backward-compatible wrappers (redirect to new functions)
# ---------------------------------------------------------------------------

def atoms_to_system(atoms, dtype=jnp.float32) -> AtomicSystem:
    """Backward-compatible wrapper. Prefer from_ase_atoms()."""
    return from_ase_atoms(atoms, dtype=dtype)


def trj_atom_to_system(trj_atoms, dtype=jnp.float32) -> AtomicSystem:
    """Backward-compatible wrapper. Returns a batched AtomicSystem from ASE atoms list."""
    traj = trajectory_from_ase(trj_atoms, dtype=dtype)
    return traj.to_batched_system()


# ---------------------------------------------------------------------------
# Wrap / Unwrap utilities
# ---------------------------------------------------------------------------

def wrap_positions(R: jnp.ndarray, cell: jnp.ndarray) -> jnp.ndarray:
    """Wrap positions into the periodic cell.

    Args:
        R: (N, 3) positions
        cell: (3, 3) column-major cell

    Returns:
        R_wrapped: (N, 3) wrapped positions
    """
    cell_inv = jnp.linalg.inv(cell)
    frac = jnp.einsum("Aa,ia->iA", cell_inv, R)
    frac_wrapped = frac % 1.0
    return jnp.einsum("aA,iA->ia", cell, frac_wrapped)


def unwrap_positions(R_wrapped: jnp.ndarray, R_ref: jnp.ndarray, cell: jnp.ndarray) -> jnp.ndarray:
    """Unwrap positions relative to reference positions.

    Args:
        R_wrapped: (N, 3) wrapped positions
        R_ref: (N, 3) reference (unwrapped) positions
        cell: (3, 3) column-major cell

    Returns:
        R_unwrapped: (N, 3) unwrapped positions closest to R_ref
    """
    cell_inv = jnp.linalg.inv(cell)
    delta = R_wrapped - R_ref
    delta_frac = jnp.einsum("Aa,ia->iA", cell_inv, delta)
    delta_frac = delta_frac - jnp.round(delta_frac)
    delta_real = jnp.einsum("aA,iA->ia", cell, delta_frac)
    return R_ref + delta_real


# ---------------------------------------------------------------------------
# Displacement (unchanged — only uses system.cell)
# ---------------------------------------------------------------------------

def to_displacement(system):
    return make_displacement(system.cell)
