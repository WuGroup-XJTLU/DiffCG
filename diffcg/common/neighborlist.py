# SPDX-License-Identifier: MIT
# Copyright (c) 2025 WuResearchGroup

from collections import namedtuple
from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from jax_md import space, partition

SpatialPartitioning = namedtuple(
    "SpatialPartitioning", ("allocate_fn", "update_fn", "cutoff", "skin", "capacity_multiplier")
)

# JAX-MD based spatial partitioning
JAXMDSpatialPartitioning = namedtuple(
    "JAXMDSpatialPartitioning",
    ("neighbor_fn", "displacement_fn", "shift_fn", "cutoff", "capacity_multiplier", "format")
)

@jax.jit
def add_batch_dim(tree):
    return jax.tree_map(lambda x: x[None], tree)


@jax.jit
def apply_neighbor_convention(tree):
    idx_i = jnp.where(tree['idx_i'] < len(tree['z']), tree['idx_i'], -1)
    idx_j = jnp.where(tree['idx_j'] < len(tree['z']), tree['idx_j'], -1)
    tree['idx_i'] = idx_i
    tree['idx_j'] = idx_j
    return tree


def jaxmd_neighbor_list(
    positions: jnp.ndarray,
    cell: Optional[jnp.ndarray],
    cutoff: float,
    capacity_multiplier: float = 1.25,
    format: partition.NeighborListFormat = partition.Sparse,
    fractional_coordinates: bool = False,
) -> Tuple:
    """JAX-MD based neighbor list.

    Args:
        positions: Particle positions, shape (N, 3).
        cell: Unit cell matrix, shape (3, 3) in column-major format.
              None for free boundary conditions.
        cutoff: Neighbor list cutoff distance.
        capacity_multiplier: Factor to multiply estimated neighbor count for buffer.
        format: JAX-MD neighbor list format (Sparse, Dense, or OrderedSparse).
        fractional_coordinates: Whether positions are in fractional coordinates.

    Returns:
        Tuple of (neighbors, spatial_partitioning).
    """
    if cell is not None:
        cell_jaxmd = cell.T  # Convert to row-major format
        displacement_fn, shift_fn = space.periodic_general(cell_jaxmd, fractional_coordinates=fractional_coordinates)
        box = cell_jaxmd
    else:
        displacement_fn, shift_fn = space.free()
        box = None

    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        box=box,
        r_cutoff=cutoff,
        capacity_multiplier=capacity_multiplier,
        format=format,
        fractional_coordinates=fractional_coordinates,
    )

    neighbors = neighbor_fn.allocate(positions)

    return neighbors, JAXMDSpatialPartitioning(
        neighbor_fn=neighbor_fn,
        displacement_fn=displacement_fn,
        shift_fn=shift_fn,
        cutoff=cutoff,
        capacity_multiplier=capacity_multiplier,
        format=format,
    )


def jaxmd_update_neighbor_list(
    positions: jnp.ndarray,
    neighbors,
    spatial_partitioning: JAXMDSpatialPartitioning,
    new_cell: Optional[jnp.ndarray] = None,
):
    """Update JAX-MD neighbor list."""
    return spatial_partitioning.neighbor_fn.update(positions, neighbors)


def neighbor_list(positions: jnp.ndarray, cutoff: float, skin: float, cell: jnp.ndarray = None,
                  capacity_multiplier: float = 1.4):
    """GLP-based neighbor list (legacy interface).

    Args:
        positions: Particle positions.
        cutoff: Cutoff distance.
        skin: Skin distance for neighbor list.
        cell: Unit cell matrix (column-major format).
        capacity_multiplier: Buffer multiplier.

    Returns:
        Tuple of (neighbors, SpatialPartitioning).
    """
    try:
        from glp.neighborlist import quadratic_neighbor_list
    except ImportError:
        raise ImportError('For neighborhood list, please install the glp package from ...')

    allocate, update = quadratic_neighbor_list(
        cell, cutoff, skin, capacity_multiplier=capacity_multiplier
    )

    neighbors = allocate(positions)

    return neighbors, SpatialPartitioning(allocate_fn=allocate,
                                          update_fn=jax.jit(update),
                                          cutoff=cutoff,
                                          skin=skin,
                                          capacity_multiplier=capacity_multiplier)

