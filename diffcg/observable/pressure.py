# SPDX-License-Identifier: MIT
# Copyright (c) 2025 WuResearchGroup

"""Virial pressure via box-scaling for coarse-grained systems.

The instantaneous pressure is computed using the box-scaling virial:

    P = (N * kBT  -  (1/d) * dU/deps) / V

where eps is a uniform isotropic scaling parameter:
    R -> R * (1 + eps),  cell -> cell * (1 + eps)
evaluated at eps = 0.

This preserves fractional coordinates so neighbor list indices remain valid,
handles all interaction types (pair, bond, angle, dihedral) via autodiff,
and is differentiable w.r.t. energy function parameters.

Units: returns pressure in kJ/(mol*nm^3).
"""

import jax
import jax.numpy as jnp
from jax import lax

from diffcg.system import AtomicSystem, System
from diffcg._core.neighborlist import jaxmd_neighbor_list


def compute_instantaneous_pressure(energy_fn, system, neighbors, kT):
    """Compute virial pressure for a single frame via box-scaling.

    Args:
        energy_fn: Energy function with signature (System, neighbors) -> float.
        system: AtomicSystem with R (N, d) and cell (d, d).
        neighbors: JAX-MD neighbor list object.
        kT: Thermal energy in kJ/mol (= kB * T).

    Returns:
        Scalar pressure in kJ/(mol*nm^3).
    """
    N, d = system.R.shape
    V = jnp.abs(jnp.linalg.det(system.cell))

    def energy_at_eps(eps):
        s = 1.0 + eps
        scaled_system = System(R=system.R * s, Z=system.Z, cell=system.cell * s)
        return energy_fn(scaled_system, neighbors)

    dU_deps = jax.grad(energy_at_eps)(0.0)
    return (N * kT - dU_deps / d) / V


def make_pressure_compute_fn(kT):
    """Return a compute_fn with unified signature (system, energy_fn=None, neighbors=None).

    Args:
        kT: Thermal energy in kJ/mol (= kB * T).

    Returns:
        A function compatible with quantity_dict['compute_fn'] that computes
        instantaneous virial pressure as a (1,) array.
    """
    def pressure_fn(system, energy_fn=None, neighbors=None, **unused_kwargs):
        p = compute_instantaneous_pressure(energy_fn, system, neighbors, kT)
        return jnp.atleast_1d(p)
    return pressure_fn


def compute_pressure_trajectory(energy_fn, batched_systems, cell, cutoff, kT,
                                capacity_multiplier=1.25,
                                custom_mask_function=None):
    """Compute per-frame virial pressure for a trajectory.

    Uses jax.lax.scan for memory-efficient per-frame computation.

    Args:
        energy_fn: Energy function with signature (System, neighbors) -> float.
        batched_systems: AtomicSystem with R of shape (B, N, d).
        cell: (d, d) unit cell matrix (shared across frames).
        cutoff: Neighbor list cutoff distance.
        kT: Thermal energy in kJ/mol.
        capacity_multiplier: Neighbor list capacity multiplier.
        custom_mask_function: Optional exclusion mask function for neighbor list.

    Returns:
        (B,) array of pressures in kJ/(mol*nm^3).
    """
    all_R = batched_systems.R  # (B, N, d)
    Z = batched_systems.Z
    # Handle both batched Z (B, N) and unbatched Z (N,)
    z = Z[0] if Z.ndim == 2 else Z

    nbrs, sp = jaxmd_neighbor_list(
        positions=all_R[0], cell=cell, cutoff=cutoff,
        capacity_multiplier=capacity_multiplier,
        custom_mask_function=custom_mask_function,
    )

    def body_fn(nbrs_carry, R_i):
        nbrs_i = sp.neighbor_fn.update(R_i, nbrs_carry)
        sys_i = System(R=R_i, Z=z, cell=cell)
        p_i = compute_instantaneous_pressure(energy_fn, sys_i, nbrs_i, kT)
        return nbrs_i, p_i

    _, pressures = lax.scan(body_fn, nbrs, all_R)
    return pressures
