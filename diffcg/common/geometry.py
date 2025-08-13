"""Low-level geometry helpers for distances, angles, and dihedrals.

SPDX-License-Identifier: MIT
Copyright (c) 2025 WuResearchGroup
"""

import jax.numpy as jnp
from jax import vmap

def distance(R_ij):
    return jnp.linalg.norm(R_ij)

def angle(R_ij, R_kj):
    """
    Computes the angle (kj, ij) from vectors R_kj and R_ij, correctly selecting the quadrant.

    Based on tan(theta) = |(R_ji x R_kj)| / (R_ji . R_kj). Beware non-differentability of arctan2(0,0).

    Args:
        R_ij: Vector pointing to i from j
        R_kj: Vector pointing to k from j

    Returns:
        Angle between vectors

    """
    cross = jnp.linalg.norm(jnp.cross(R_ij, R_kj))
    dot = jnp.dot(R_ij, R_kj)
    theta = jnp.arctan2(cross, dot)
    return theta

def vectorized_angle_fn(R_ij, R_kj):
    return vmap(angle)(R_ij, R_kj)


def dihedral(R_ab, R_bc, R_cd):
    """
    Computes the angle (kj, ij) from vectors R_kj and R_ij, correctly selecting the quadrant.

    Based on tan(theta) = |(R_ji x R_kj)| / (R_ji . R_kj). Beware non-differentability of arctan2(0,0).

    Args:
        R_ij: Vector pointing to i from j
        R_kj: Vector pointing to k from j

    Returns:
        Angle between vectors

    """
    cross_ab_bc = jnp.cross(R_ab, R_bc)
    cross_bc_cd = jnp.cross(R_bc, R_cd)
    bc_norm = jnp.linalg.norm(R_bc)
    cross = jnp.dot(jnp.cross(cross_ab_bc, cross_bc_cd), R_bc / bc_norm)
    dot = jnp.dot(cross_ab_bc, cross_bc_cd)
    theta = jnp.arctan2(cross,dot)
    return theta

def vectorized_dihedral_fn(R_ab, R_bc, R_cd):
    return vmap(dihedral)(R_ab, R_bc, R_cd)