# SPDX-License-Identifier: MIT
# Copyright (c) 2025 WuResearchGroup

"""Custom integrators that capture potential energy via value_and_grad.

These integrators mirror JAX-MD's NVE (velocity Verlet) and NVT Langevin
(BAOAB) integrators exactly, but replace ``grad(energy_fn)`` with
``value_and_grad(energy_fn)`` so that the potential energy is obtained for
free alongside the forces.  The PE is stored in the state dataclass and
can be read out at every step for thermodynamic logging.
"""

import dataclasses
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad

from jax_md import quantity, simulate

f32 = jnp.float32


# ---------------------------------------------------------------------------
# JAX pytree registration helper (compatible with JAX < 0.4.27)
# ---------------------------------------------------------------------------
def _register_dataclass_pytree(cls):
    """Register a dataclass as a JAX pytree node."""
    field_names = [f.name for f in dataclasses.fields(cls)]

    def _flatten(obj):
        children = [getattr(obj, name) for name in field_names]
        return children, None

    def _unflatten(aux_data, children):
        return cls(**dict(zip(field_names, children)))

    jax.tree_util.register_pytree_node(cls, _flatten, _unflatten)
    return cls


# ---------------------------------------------------------------------------
# State dataclasses (mirror JAX-MD but add potential_energy)
# ---------------------------------------------------------------------------

@_register_dataclass_pytree
@dataclasses.dataclass
class NVEState:
    """NVE state with potential energy."""
    position: jax.Array
    momentum: jax.Array
    force: jax.Array
    mass: jax.Array
    potential_energy: jax.Array

    @property
    def velocity(self) -> jax.Array:
        return self.momentum / self.mass

    def set(self, **kwargs):
        return dataclasses.replace(self, **kwargs)


@_register_dataclass_pytree
@dataclasses.dataclass
class NVTLangevinState:
    """NVT Langevin state with potential energy."""
    position: jax.Array
    momentum: jax.Array
    force: jax.Array
    mass: jax.Array
    rng: jax.Array
    potential_energy: jax.Array

    @property
    def velocity(self) -> jax.Array:
        return self.momentum / self.mass

    def set(self, **kwargs):
        return dataclasses.replace(self, **kwargs)


# ---------------------------------------------------------------------------
# Helper: canonicalize mass for broadcasting (N,) -> (N, 1)
# ---------------------------------------------------------------------------

def _canonicalize_mass(mass):
    if isinstance(mass, float):
        return mass
    if mass.ndim == 1:
        return jnp.reshape(mass, (mass.shape[0], 1))
    if mass.ndim == 2 and mass.shape[1] == 1:
        return mass
    if mass.ndim == 0:
        return mass
    raise ValueError(f"Unexpected mass shape: {mass.shape}")


def _initialize_momenta(state, key, kT):
    """Initialize momenta from Maxwell-Boltzmann distribution.

    Matches JAX-MD's initialize_momenta exactly: tree_flatten the position,
    split key by number of leaves, generate momenta per leaf.
    """
    R = state.position
    mass = state.mass
    keys = random.split(key, 1)
    p = jnp.sqrt(mass * kT) * random.normal(keys[0], R.shape, dtype=R.dtype)
    if R.shape[0] > 1:
        p = p - jnp.mean(p, axis=0, keepdims=True)
    return state.set(momentum=p)


# ---------------------------------------------------------------------------
# force_and_energy helper
# ---------------------------------------------------------------------------

def _force_and_energy_fn(energy_fn):
    """Return a function that computes (force, potential_energy) in one pass."""
    def fn(R, **kwargs):
        pe, neg_grad = value_and_grad(energy_fn)(R, **kwargs)
        return -neg_grad, pe
    return fn


# ---------------------------------------------------------------------------
# NVE (velocity Verlet) with PE capture
# ---------------------------------------------------------------------------

def nve(
    energy_fn: Callable,
    shift_fn: Callable,
    dt: float,
) -> Tuple[Callable, Callable]:
    """NVE integrator (velocity Verlet) that stores potential energy in state.

    Args:
        energy_fn: Energy function ``(R, **kwargs) -> scalar``.
        shift_fn: JAX-MD shift function.
        dt: Timestep in internal units.

    Returns:
        ``(init_fn, apply_fn)`` pair.
    """
    fe_fn = _force_and_energy_fn(energy_fn)

    @jit
    def init_fn(key, R, kT=None, mass=f32(1.0), **kwargs):
        force, pe = fe_fn(R, **kwargs)
        mass_c = _canonicalize_mass(mass)
        state = NVEState(R, None, force, mass_c, pe)
        if kT is not None:
            state = _initialize_momenta(state, key, kT)
        else:
            state = state.set(momentum=jnp.zeros_like(R))
        return state

    @jit
    def apply_fn(state, **kwargs):
        _dt = kwargs.pop('dt', dt)
        dt_2 = _dt / 2

        # Half-kick
        new_mom = state.momentum + dt_2 * state.force
        # Drift
        new_pos = shift_fn(state.position, _dt * new_mom / state.mass, **kwargs)
        # Force + energy at new position
        new_force, pe = fe_fn(new_pos, **kwargs)
        # Half-kick
        new_mom = new_mom + dt_2 * new_force

        return NVEState(new_pos, new_mom, new_force, state.mass, pe)

    return init_fn, apply_fn


# ---------------------------------------------------------------------------
# NVT Langevin (BAOAB) with PE capture
# ---------------------------------------------------------------------------

def nvt_langevin(
    energy_fn: Callable,
    shift_fn: Callable,
    dt: float,
    kT: float,
    gamma: float = 0.1,
) -> Tuple[Callable, Callable]:
    """NVT Langevin (BAOAB) integrator that stores potential energy in state.

    Mirrors JAX-MD's ``simulate.nvt_langevin`` exactly, but captures PE from
    the ``value_and_grad`` call that computes forces.

    Args:
        energy_fn: Energy function ``(R, **kwargs) -> scalar``.
        shift_fn: JAX-MD shift function.
        dt: Timestep in internal units.
        kT: Thermal energy in energy units (kJ/mol).
        gamma: Friction coefficient in 1/ps.

    Returns:
        ``(init_fn, apply_fn)`` pair.
    """
    fe_fn = _force_and_energy_fn(energy_fn)

    @jit
    def init_fn(key, R, mass=f32(1.0), **kwargs):
        _kT = kwargs.pop('kT', kT)
        key, split = random.split(key)
        force, pe = fe_fn(R, **kwargs)
        mass_c = _canonicalize_mass(mass)
        state = NVTLangevinState(R, None, force, mass_c, key, pe)
        return _initialize_momenta(state, split, _kT)

    @jit
    def apply_fn(state, **kwargs):
        _dt = kwargs.pop('dt', dt)
        _kT = kwargs.pop('kT', kT)
        dt_2 = _dt / 2

        # B: half momentum kick
        mom = state.momentum + dt_2 * state.force
        # A: half position drift
        pos = shift_fn(state.position, dt_2 * mom / state.mass, **kwargs)
        # O: stochastic (Ornstein-Uhlenbeck) step
        c1 = jnp.exp(-gamma * _dt)
        c2 = jnp.sqrt(_kT * (1 - c1 ** 2))
        key, split = random.split(state.rng)
        noise = random.normal(split, mom.shape, dtype=mom.dtype)
        mom = c1 * mom + c2 * jnp.sqrt(state.mass) * noise
        # A: half position drift
        pos = shift_fn(pos, dt_2 * mom / state.mass, **kwargs)
        # Force + Energy
        force, pe = fe_fn(pos, **kwargs)
        # B: half momentum kick
        mom = mom + dt_2 * force

        return NVTLangevinState(pos, mom, force, state.mass, key, pe)

    return init_fn, apply_fn
