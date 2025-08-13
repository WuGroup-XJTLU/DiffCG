# SPDX-License-Identifier: MIT
# Copyright (c) 2025 WuResearchGroup

from collections import namedtuple
import jax
import jax.numpy as jnp

SpatialPartitioning = namedtuple(
    "SpatialPartitioning", ("allocate_fn", "update_fn", "cutoff", "skin", "capacity_multiplier")
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

def neighbor_list(positions: jnp.ndarray, cutoff: float, skin: float, cell: jnp.ndarray = None,
                  capacity_multiplier: float = 1.4):
    """

    Args:
        positions ():
        cutoff ():
        skin ():
        cell (): ASE cell.
        capacity_multiplier ():

    Returns:

    """
    try:
        from glp.neighborlist import quadratic_neighbor_list
    except ImportError:
        raise ImportError('For neighborhood list, please install the glp package from ...')
    # Convenience interface
    # if cell is not None:
    #     cell_T = cell.T
    # else:
    #     cell_T = None

    allocate, update = quadratic_neighbor_list(
        cell, cutoff, skin, capacity_multiplier=capacity_multiplier
    )

    neighbors = allocate(positions)

    return neighbors, SpatialPartitioning(allocate_fn=allocate,
                                          update_fn=jax.jit(update),
                                          cutoff=cutoff,
                                          skin=skin,
                                          capacity_multiplier=capacity_multiplier)

