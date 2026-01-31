# SPDX-License-Identifier: MIT
# Copyright (c) 2025 WuResearchGroup

from functools import partial
from typing import Callable, Dict
import jax.numpy as jnp
import numpy as np
from jax import grad
import jax

from diffcg.system import AtomicSystem, System
from diffcg._core.neighborlist import jaxmd_neighbor_list
from diffcg._core.logger import get_logger

logger = get_logger(__name__)


def force(energy_fn: Callable) -> Callable:
    """Computes the force as the negative gradient of an energy."""
    return grad(lambda atoms, *args, **kwargs: -energy_fn(atoms, *args, **kwargs))


def compute_energy(
    system: AtomicSystem,
    energy_fn: Callable,
    cutoff: float = 1.0,
    capacity_multiplier: float = 1.25,
    neighbors=None,
    spatial_partitioning=None,
    **kwargs,
) -> jnp.ndarray:
    """Compute energy for an AtomicSystem.

    Args:
        system: AtomicSystem
        energy_fn: energy function(system, neighbors, **kwargs)
        cutoff: neighbor list cutoff
        capacity_multiplier: neighbor list capacity multiplier
        neighbors: pre-computed neighbors (optional)
        spatial_partitioning: pre-computed spatial partitioning (optional)

    Returns:
        scalar energy
    """
    if neighbors is None or spatial_partitioning is None:
        neighbors, spatial_partitioning = jaxmd_neighbor_list(
            positions=system.R,
            cell=system.cell,
            cutoff=cutoff,
            capacity_multiplier=capacity_multiplier,
        )

    neighbors = spatial_partitioning.neighbor_fn.update(system.R, neighbors)
    if neighbors.did_buffer_overflow:
        logger.error('Neighbor list overflow detected')
        raise RuntimeError('Spatial overflow.')

    return energy_fn(system, neighbors, **kwargs)


def compute_energy_and_forces(
    system: AtomicSystem,
    energy_fn: Callable,
    cutoff: float = 1.0,
    capacity_multiplier: float = 1.25,
    neighbors=None,
    spatial_partitioning=None,
    **kwargs,
) -> Dict[str, jnp.ndarray]:
    """Compute energy and forces for an AtomicSystem.

    Args:
        system: AtomicSystem
        energy_fn: energy function(system, neighbors, **kwargs)
        cutoff: neighbor list cutoff
        capacity_multiplier: neighbor list capacity multiplier
        neighbors: pre-computed neighbors (optional)
        spatial_partitioning: pre-computed spatial partitioning (optional)

    Returns:
        dict with 'energy' (scalar) and 'forces' (N, 3) arrays
    """
    if neighbors is None or spatial_partitioning is None:
        neighbors, spatial_partitioning = jaxmd_neighbor_list(
            positions=system.R,
            cell=system.cell,
            cutoff=cutoff,
            capacity_multiplier=capacity_multiplier,
        )

    neighbors = spatial_partitioning.neighbor_fn.update(system.R, neighbors)
    if neighbors.did_buffer_overflow:
        logger.error('Neighbor list overflow detected')
        raise RuntimeError('Spatial overflow.')

    def _energy(sys, nbrs):
        return energy_fn(sys, nbrs, **kwargs)

    energy, grads = jax.value_and_grad(_energy, allow_int=True)(system, neighbors)
    forces = -grads.R

    if jnp.isnan(energy):
        logger.error('NaN energy encountered')
        raise RuntimeError('Energy is NaN.')

    return {'energy': energy, 'forces': forces}


def init_energy_calculator(energy_fn, cutoff=1.0, capacity_multiplier=1.25, dtype=jnp.float64):
    """Create a function that computes energy for a System (no ASE dependency)."""

    def calculate_fn(system, **kwargs):
        R = system.R
        cell = system.cell
        neighbors, spatial_partitioning = jaxmd_neighbor_list(
            positions=R,
            cell=cell,
            cutoff=cutoff,
            capacity_multiplier=capacity_multiplier,
        )
        return energy_fn(system, neighbors, **kwargs)

    return calculate_fn
