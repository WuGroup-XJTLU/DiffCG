# SPDX-License-Identifier: MIT
# Copyright (c) 2025 WuResearchGroup

"""Tests for bonded-pair exclusion via make_exclusion_mask."""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from jax_md import space, partition

from diffcg._core.neighborlist import make_exclusion_mask, jaxmd_neighbor_list


def _build_small_system():
    """Create a small 5-particle linear chain for testing.

    Topology:  0 -- 1 -- 2 -- 3 -- 4
    Bonds: (0,1), (1,2), (2,3), (3,4)
    Angles: (0,1,2), (1,2,3), (2,3,4)
    Dihedrals: (0,1,2,3), (1,2,3,4)
    """
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [0.3, 0.0, 0.0],
        [0.6, 0.0, 0.0],
        [0.9, 0.0, 0.0],
        [1.2, 0.0, 0.0],
    ], dtype=jnp.float64)

    topology = {
        'bond': np.array([[0, 1], [1, 2], [2, 3], [3, 4]]),
        'angle': np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]]),
        'dihedral': np.array([[0, 1, 2, 3], [1, 2, 3, 4]]),
    }
    return positions, topology


def _get_neighbor_pairs(neighbors, n_particles):
    """Extract active (non-padding) neighbor pairs from Dense neighbor list."""
    idx = neighbors.idx
    pairs = set()
    for i in range(n_particles):
        for j_slot in range(idx.shape[1]):
            j = int(idx[i, j_slot])
            if j < n_particles:
                pairs.add((min(i, j), max(i, j)))
    return pairs


def _build_neighbors(positions, mask_fn):
    """Build neighbor list with exclusion mask applied."""
    # Use a large periodic box to simulate free BC (JAX-MD requires box != None)
    cell = jnp.eye(3) * 10.0  # column-major, 10 nm box
    neighbors, sp = jaxmd_neighbor_list(
        positions=positions, cell=cell, cutoff=2.0,
        capacity_multiplier=2.0,
        custom_mask_function=mask_fn,
    )
    return neighbors


class TestMakeExclusionMask:
    def test_returns_none_for_empty_topology(self):
        mask_fn = make_exclusion_mask({}, exclusion_level=3)
        assert mask_fn is None

    def test_level2_excludes_bonds_only(self):
        positions, topology = _build_small_system()
        mask_fn = make_exclusion_mask(topology, exclusion_level=2)
        assert mask_fn is not None

        neighbors = _build_neighbors(positions, mask_fn)
        pairs = _get_neighbor_pairs(neighbors, 5)

        # 1-2 pairs (bonds) should be excluded
        assert (0, 1) not in pairs
        assert (1, 2) not in pairs
        assert (2, 3) not in pairs
        assert (3, 4) not in pairs

        # 1-3 pairs should still be present (not excluded at level 2)
        assert (0, 2) in pairs
        assert (1, 3) in pairs
        assert (2, 4) in pairs

    def test_level3_excludes_bonds_and_angles(self):
        positions, topology = _build_small_system()
        mask_fn = make_exclusion_mask(topology, exclusion_level=3)

        neighbors = _build_neighbors(positions, mask_fn)
        pairs = _get_neighbor_pairs(neighbors, 5)

        # 1-2 pairs (bonds) excluded
        assert (0, 1) not in pairs
        assert (1, 2) not in pairs

        # 1-3 pairs (angle ends) excluded
        assert (0, 2) not in pairs
        assert (1, 3) not in pairs
        assert (2, 4) not in pairs

        # 1-4 pairs should still be present
        assert (0, 3) in pairs
        assert (1, 4) in pairs

    def test_level4_excludes_all(self):
        positions, topology = _build_small_system()
        mask_fn = make_exclusion_mask(topology, exclusion_level=4)

        neighbors = _build_neighbors(positions, mask_fn)
        pairs = _get_neighbor_pairs(neighbors, 5)

        # 1-2 excluded
        assert (0, 1) not in pairs
        # 1-3 excluded
        assert (0, 2) not in pairs
        # 1-4 excluded
        assert (0, 3) not in pairs
        assert (1, 4) not in pairs

        # Only (0,4) should remain — not bonded through any topology
        assert (0, 4) in pairs


class TestJaxmdNeighborListWithMask:
    def test_custom_mask_function_threaded(self):
        """Verify that custom_mask_function flows through jaxmd_neighbor_list."""
        positions, topology = _build_small_system()
        mask_fn = make_exclusion_mask(topology, exclusion_level=3)

        cell = jnp.eye(3) * 10.0
        neighbors, sp = jaxmd_neighbor_list(
            positions=positions, cell=cell, cutoff=2.0,
            capacity_multiplier=2.0,
            custom_mask_function=mask_fn,
        )
        pairs = _get_neighbor_pairs(neighbors, 5)

        # Bonds excluded
        assert (0, 1) not in pairs
        # 1-3 excluded
        assert (0, 2) not in pairs
        # 1-4 still present
        assert (0, 3) in pairs
