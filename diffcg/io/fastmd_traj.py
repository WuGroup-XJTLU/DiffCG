"""Reader for fastMD binary trajectory format (traj.bin).

Format (all little-endian):
  Header: int32 magic (0x4D444247), int32 natoms, int32 ntypes
  Per frame: int64 step, int32 n, float box_L, float[n][3] positions

Positions and box_L are in fastMD units (nm, matching diffcg).
"""

from typing import Optional
import numpy as np
import jax.numpy as jnp
from diffcg.system import Trajectory

_MAGIC = 0x4D444247


def read_fastmd_trajectory(
    path: str,
    Z: Optional[jnp.ndarray] = None,
    masses: Optional[jnp.ndarray] = None,
    pbc: bool = True,
) -> Trajectory:
    """Read a fastMD binary trajectory file.

    Args:
        path: Path to traj.bin
        Z: Atomic numbers array (N,). If None, defaults to zeros.
        masses: Masses array (N,). If None, defaults to ones.
        pbc: Whether system is periodic.

    Returns:
        Trajectory with positions in nm.
    """
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=3)
        if header[0] != _MAGIC:
            raise ValueError(
                f"Bad magic number in {path}: "
                f"expected {_MAGIC:#x}, got {header[0]:#x}"
            )
        natoms = int(header[1])

        frames_data = []
        box_L = 0.0
        while True:
            frame_header = f.read(16)  # int64 step + int32 n + float box_L
            if not frame_header or len(frame_header) < 16:
                break
            n = int(np.frombuffer(frame_header[8:12], dtype=np.int32)[0])
            box_L = float(np.frombuffer(frame_header[12:16], dtype=np.float32)[0])

            raw = f.read(n * 12)  # n * float3
            if len(raw) < n * 12:
                break
            pos = np.frombuffer(raw, dtype=np.float32).reshape(n, 3).copy()
            frames_data.append(pos)

    if not frames_data:
        raise RuntimeError(f"No frames found in {path}")

    positions = np.stack(frames_data, axis=0)

    cell = jnp.eye(3) * box_L
    if Z is None:
        Z = jnp.zeros(natoms, dtype=jnp.int32)
    if masses is None:
        masses = jnp.ones(natoms, dtype=jnp.float32)

    return Trajectory(
        positions=jnp.array(positions, dtype=jnp.float32),
        Z=Z,
        cell=jnp.array(cell, dtype=jnp.float32),
        masses=jnp.array(masses, dtype=jnp.float32),
        pbc=pbc,
    )
