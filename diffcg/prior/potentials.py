from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Dict

import jax
import jax.numpy as jnp

from diffcg import energy
from diffcg.util import custom_interpolate


@dataclass
class GridConfig:
    pair_start: float = 0.4
    pair_end: float = 2.0
    pair_n: int = 80
    bond_start: float = 0.1
    bond_end: float = 1.0
    bond_n: int = 45
    angle_start: float = 0.1
    angle_end: float = 3.14
    angle_n: int = 55
    dihedral_start: float = -3.14
    dihedral_end: float = 3.14
    dihedral_n: int = 100


def build_pair_fn(params: jnp.ndarray, x_vals: jnp.ndarray):
    spline = custom_interpolate.MonotonicInterpolate(x_vals, params)
    tabulated_partial = partial(energy.tabulated, spline=spline)
    generic_repulsion = partial(energy.generic_repulsion, sigma=0.6, epsilon=1.0, exp=8)

    def U(r):
        return tabulated_partial(r) + generic_repulsion(r)

    return U


def build_bond_fn(params: jnp.ndarray, x_vals: jnp.ndarray):
    spline = custom_interpolate.MonotonicInterpolate(x_vals, params)
    tabulated_partial = partial(energy.tabulated, spline=spline)
    harmonic = partial(energy.simple_spring, length=0.45, epsilon=5000)

    def U(r):
        return tabulated_partial(r) + harmonic(r)

    return U


def build_angle_fn(params: jnp.ndarray, x_vals: jnp.ndarray):
    spline = custom_interpolate.MonotonicInterpolate(x_vals, params)
    tabulated_partial = partial(energy.tabulated, spline=spline)
    harmonic = partial(energy.harmonic_angle, angle_0=1.5, epsilon=50)

    def U(theta):
        return tabulated_partial(theta) + harmonic(theta)

    return U


def build_dihedral_fn(params: jnp.ndarray, x_vals: jnp.ndarray):
    spline = custom_interpolate.MonotonicInterpolate(x_vals, params)
    tabulated_partial = partial(energy.tabulated, spline=spline)
    harmonic = partial(energy.harmonic_dihedral, angle_0=1.5, epsilon=50)

    def U(phi):
        return tabulated_partial(phi) + harmonic(phi)

    return U


def make_grids(cfg: GridConfig) -> Dict[str, jnp.ndarray]:
    return {
        'pair': jnp.linspace(cfg.pair_start, cfg.pair_end, cfg.pair_n),
        'bond': jnp.linspace(cfg.bond_start, cfg.bond_end, cfg.bond_n),
        'angle': jnp.linspace(cfg.angle_start, cfg.angle_end, cfg.angle_n),
        'dihedral': jnp.linspace(cfg.dihedral_start, cfg.dihedral_end, cfg.dihedral_n),
    }


