# SPDX-License-Identifier: MIT
# Copyright (c) 2025 WuResearchGroup

from collections import namedtuple
from typing import Callable, Optional, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from jax_md import space, partition

# JAX-MD based spatial partitioning
JAXMDSpatialPartitioning = namedtuple(
    "JAXMDSpatialPartitioning",
    ("neighbor_fn", "displacement_fn", "shift_fn", "cutoff", "capacity_multiplier", "format")
)

def make_exclusion_mask(topology, exclusion_level=3):
    """Create custom_mask_function for JAX-MD neighbor_list that excludes bonded pairs.

    Args:
        topology: dict with keys 'bond' (N,2), 'angle' (N,3), 'dihedral' (N,4)
        exclusion_level: 2 = exclude 1-2 only
                         3 = exclude 1-2 + 1-3
                         4 = exclude 1-2 + 1-3 + 1-4

    Returns:
        mask_fn compatible with partition.neighbor_list(custom_mask_function=...)
    """
    pairs_list = []

    if exclusion_level >= 2 and 'bond' in topology:
        bonds = np.asarray(topology['bond'])
        pairs_list.append(bonds[:, [0, 1]])

    if exclusion_level >= 3 and 'angle' in topology:
        angles = np.asarray(topology['angle'])
        # 1-3 pairs: end atoms of angles
        pairs_list.append(angles[:, [0, 2]])
        # Also include 1-2 pairs from angles (bond-like)
        pairs_list.append(angles[:, [0, 1]])
        pairs_list.append(angles[:, [1, 2]])

    if exclusion_level >= 4 and 'dihedral' in topology:
        dihedrals = np.asarray(topology['dihedral'])
        # 1-4 pairs: end atoms of dihedrals
        pairs_list.append(dihedrals[:, [0, 3]])
        # Also include intermediate pairs
        pairs_list.append(dihedrals[:, [0, 1]])
        pairs_list.append(dihedrals[:, [0, 2]])
        pairs_list.append(dihedrals[:, [1, 2]])
        pairs_list.append(dihedrals[:, [1, 3]])
        pairs_list.append(dihedrals[:, [2, 3]])

    if not pairs_list:
        return None

    all_pairs = np.concatenate(pairs_list, axis=0)
    # Normalize: smaller index first
    pair_min = np.minimum(all_pairs[:, 0], all_pairs[:, 1])
    pair_max = np.maximum(all_pairs[:, 0], all_pairs[:, 1])
    all_pairs = np.stack([pair_min, pair_max], axis=1)

    # Deduplicate
    max_atom = int(np.max(all_pairs)) + 1
    pair_keys = all_pairs[:, 0] * max_atom + all_pairs[:, 1]
    unique_keys = np.unique(pair_keys)
    sorted_keys = jnp.array(unique_keys, dtype=jnp.int32)

    def mask_fn(idx):
        """Mask bonded pairs in JAX-MD Dense neighbor list.

        Args:
            idx: (n_particles, max_neighbors) array of neighbor indices.

        Returns:
            Modified idx with excluded pairs set to n_particles.
        """
        n_particles = idx.shape[0]

        # Build center indices: for each (i, j_slot), center is i
        centers = jnp.arange(n_particles)[:, None] * jnp.ones(idx.shape[1], dtype=jnp.int32)[None, :]
        centers = centers.astype(jnp.int32)

        # Normalize pairs (smaller first)
        p_min = jnp.minimum(centers, idx)
        p_max = jnp.maximum(centers, idx)

        # Encode as single key
        query_keys = p_min * max_atom + p_max

        # Use searchsorted to check membership
        insert_idx = jnp.searchsorted(sorted_keys, query_keys)
        insert_idx_safe = jnp.minimum(insert_idx, len(sorted_keys) - 1)
        is_bonded = sorted_keys[insert_idx_safe] == query_keys

        # Mask bonded pairs by setting to n_particles (JAX-MD padding value)
        return jnp.where(is_bonded, n_particles, idx)

    return mask_fn


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
    format: partition.NeighborListFormat = partition.Dense,  # Fixed: was Sparse, but energy functions expect Dense
    fractional_coordinates: bool = False,
    custom_mask_function: Optional[Callable] = None,
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

    nl_kwargs = dict(
        box=box,
        r_cutoff=cutoff,
        capacity_multiplier=capacity_multiplier,
        format=format,
        fractional_coordinates=fractional_coordinates,
    )
    if custom_mask_function is not None:
        nl_kwargs['custom_mask_function'] = custom_mask_function

    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        **nl_kwargs,
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
