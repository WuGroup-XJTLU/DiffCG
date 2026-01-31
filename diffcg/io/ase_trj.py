# SPDX-License-Identifier: MIT
# Copyright (c) 2025 WuResearchGroup

import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Optional, Union
from pathlib import Path

from diffcg.system import AtomicSystem, Trajectory, from_ase_atoms, trajectory_from_ase
from diffcg._core.logger import get_logger

logger = get_logger(__name__)


def read_ase_trj(filename: str) -> Trajectory:
    """Read an ASE trajectory file and return a Trajectory.

    Args:
        filename: Path to the ASE trajectory file (.traj, .xyz, .pdb, etc.)

    Returns:
        Trajectory object with all frames.
    """
    from ase import io as ase_io
    logger.info("Reading ASE trajectory from %s", filename)
    frames = ase_io.read(filename, index=':')

    if not isinstance(frames, list):
        frames = [frames]

    if not frames:
        raise ValueError(f"No frames found in trajectory file: {filename}")

    traj = trajectory_from_ase(frames)
    logger.debug("Loaded %s frames from %s", len(traj), filename)
    return traj


def ase2xyz(ase_traj, xyz_traj):
    """Convert an ASE trajectory to individual XYZ files.

    Args:
        ase_traj: Path to ASE trajectory file
        xyz_traj: Prefix for output XYZ files (writes {prefix}{i}.xyz)
    """
    from ase.io import read, write
    traj = read(ase_traj, index=':')
    for i, frame in enumerate(traj):
        write(f'{xyz_traj}{i}.xyz', frame)


def read_ase_single_frame(filename: str, index: int = 0) -> AtomicSystem:
    """Read a single frame from an ASE trajectory file.

    Args:
        filename: Path to the ASE trajectory file
        index: Frame index to read (default: 0)

    Returns:
        AtomicSystem for the requested frame.
    """
    from ase import io as ase_io
    frame = ase_io.read(filename, index=index)
    return from_ase_atoms(frame)
