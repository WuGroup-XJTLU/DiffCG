# SPDX-License-Identifier: MIT
# Copyright (c) 2025 WuResearchGroup

from diffcg.system import Trajectory


def save_trajectory(traj: Trajectory, path: str) -> None:
    """Save a Trajectory to NPZ format."""
    traj.save(path)


def load_trajectory(path: str) -> Trajectory:
    """Load a Trajectory from NPZ format."""
    return Trajectory.load(path)
