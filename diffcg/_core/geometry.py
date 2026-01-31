"""Low-level geometry helpers for distances, angles, and dihedrals.

SPDX-License-Identifier: MIT
Copyright (c) 2025 WuResearchGroup
"""

import jax.numpy as jnp
from jax import vmap, jit


@jit
def distance(R_ij, eps=1e-8):
    """Compute distance with gradient-safe norm.

    Uses sqrt(sum(x^2) + eps^2) to ensure gradient is defined at zero.
    Forward: distance(0) ≈ eps (tiny but non-zero)
    Gradient: d/dx sqrt(x^2 + eps^2) = x / sqrt(x^2 + eps^2) (well-defined at x=0)
    """
    return jnp.sqrt(jnp.sum(R_ij**2) + eps**2)

@jit
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
    # Return 0 for degenerate cases (both cross and dot near zero)
    is_degenerate = (cross < 1e-10) & (jnp.abs(dot) < 1e-10)
    theta = jnp.where(is_degenerate, 0.0, jnp.arctan2(cross, dot))
    return theta

def vectorized_angle_fn(R_ij, R_kj):
    return vmap(angle)(R_ij, R_kj)


@jit
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
    bc_norm_safe = jnp.where(bc_norm < 1e-8, 1e-8, bc_norm)
    cross = jnp.dot(jnp.cross(cross_ab_bc, cross_bc_cd), R_bc / bc_norm_safe)
    dot = jnp.dot(cross_ab_bc, cross_bc_cd)
    # Return 0 for degenerate cases (both cross and dot near zero)
    is_degenerate = (jnp.abs(cross) < 1e-10) & (jnp.abs(dot) < 1e-10)
    theta = jnp.where(is_degenerate, 0.0, jnp.arctan2(cross, dot))
    return theta

def vectorized_dihedral_fn(R_ab, R_bc, R_cd):
    return vmap(dihedral)(R_ab, R_bc, R_cd)