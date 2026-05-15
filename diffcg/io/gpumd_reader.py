"""Parse GPUMD dump_xyz output into DiffCG Trajectory."""

from typing import Optional
import numpy as np
import jax.numpy as jnp
from diffcg.system import Trajectory
from diffcg._core.units import ANGSTROM_TO_NM


def read_dump_xyz(
    filepath: str,
    Z: Optional[jnp.ndarray] = None,
    masses: Optional[jnp.ndarray] = None,
    pbc: bool = True,
) -> Trajectory:
    """Read a GPUMD dump_xyz (extended XYZ) file into a Trajectory.

    Each frame consists of:
      Line 1: natoms
      Line 2: comment with Lattice="h0 h3 h6 h1 h4 h7 h2 h5 h8"
      Lines 3-(natoms+2): species x y z [extra...]

    Returns positions in nm, cell in nm.
    """
    frames_positions = []
    cells = []

    with open(filepath, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue

            natoms = int(line)
            comment = f.readline().strip()

            # Parse box from comment: Lattice="h0 h1 h2 ... h8"
            cell = None
            if 'Lattice="' in comment:
                lattice_start = comment.index('Lattice="') + len('Lattice="')
                lattice_end = comment.index('"', lattice_start)
                lattice_str = comment[lattice_start:lattice_end]
                parts = [float(x) for x in lattice_str.split()]
                if len(parts) == 9:
                    cell = np.array([
                        [parts[0], parts[3], parts[6]],
                        [parts[1], parts[4], parts[7]],
                        [parts[2], parts[5], parts[8]],
                    ]) * ANGSTROM_TO_NM

            positions = np.zeros((natoms, 3))
            for i in range(natoms):
                atom_line = f.readline().strip()
                parts = atom_line.split()
                positions[i, 0] = float(parts[1])
                positions[i, 1] = float(parts[2])
                positions[i, 2] = float(parts[3])

            frames_positions.append(positions * ANGSTROM_TO_NM)
            if cell is not None:
                cells.append(cell)

    if not frames_positions:
        raise RuntimeError(f"No frames found in {filepath}")

    positions = np.stack(frames_positions, axis=0)
    cell = cells[0] if cells else jnp.eye(3)

    if Z is None:
        Z = jnp.zeros(positions.shape[1], dtype=jnp.int32)
    if masses is None:
        masses = jnp.ones(positions.shape[1], dtype=jnp.float32)

    return Trajectory(
        positions=jnp.array(positions, dtype=jnp.float32),
        Z=Z,
        cell=jnp.array(cell, dtype=jnp.float32),
        masses=jnp.array(masses, dtype=jnp.float32),
        pbc=pbc,
    )
