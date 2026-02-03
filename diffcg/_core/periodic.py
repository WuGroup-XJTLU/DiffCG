# SPDX-License-Identifier: MIT
# Copyright (c) 2025 WuResearchGroup

"""tools for dealing with periodicity

Conventions:
- `a,b,c,...` indicate real-space cartesian directions
- `A,B,C,...` indicate lattice vectors or inverse lattice vectors
- `R` are real-space vectors
- `X` are fractional-coordinate vectors

"""

from jax import vmap
import jax.numpy as jnp
from functools import partial
from jax_md import space


def cast(x):
    """Cast number literal to jnp.ndarray.

    This avoids jit recompiles, as native python types
    are "weak" types in jax. This makes everything explicit.
    In high-precision situations, jax type promotion should
    do the right thing.
    """

    if type(x) == int:
        return jnp.array(x, dtype=jnp.int32)
    elif type(x) == float:
        return jnp.array(x, dtype=jnp.float32)
    else:
        raise ValueError(f"cannot cast {x} of as type {type(x)} is unknown to me")


def inverse(cell):
    return jnp.linalg.inv(cell)


def _to_frac(cell, R):
    return jnp.einsum("Aa,a->A", inverse(cell), R)



def to_frac(cell, R):
    return vmap(partial(_to_frac, cell))(R)


def _from_frac(cell, X):
    return jnp.einsum("aA,A->a", cell, X)


def from_frac(cell, X):
    return vmap(partial(_from_frac, cell))(X)


def get_displacement_fn(cell):
    """Create JAX-MD displacement function from cell matrix.

    NOTE: Returns JAX-MD convention: disp_fn(Ra, Rb) = Ra - Rb.
    This is opposite to the old DiffCG convention (Rb - Ra).
    Call sites using directional displacements (angles, dihedrals)
    must swap argument order to preserve behavior.

    Args:
        cell: (D, D) cell matrix (lattice vectors as columns) or None for non-periodic.

    Returns:
        displacement_fn(Ra, Rb) -> Ra - Rb (with periodic wrapping if cell given)
    """
    if cell is None:
        disp_fn, _ = space.free()
    else:
        disp_fn, _ = space.periodic_general(cell.T, fractional_coordinates=False)
    return disp_fn



def wrap(cell, R):
    return from_frac(cell, to_frac(cell, R) % cast(1.0))


def project_on(normals, R):
    return jnp.einsum("Aa,ia->iA", normals, R)


def get_heights(cell):
    normals = get_normals(cell)
    return jnp.diag(project_on(normals, cell.T))


def get_normals(cell):
    # surface normals of cell boundaries
    # (i.e. normalised lattice vectors of reciprocal lattice)
    # convention: indexed by the lattice vector they're not orthogonal to
    inv = inverse(cell)  # rows: inverse lattice vectors
    normals = inv / jnp.linalg.norm(inv, axis=1)[:, None]
    return normals


def project_on_normals(cell, R):
    return project_on(get_normals(cell), R)
