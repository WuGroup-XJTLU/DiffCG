# SPDX-License-Identifier: MIT
# Copyright (c) 2025 WuResearchGroup

"""Custom definition of some potential energy functions."""
from typing import Callable, Dict, List, Optional

import numpy as np
import jax.numpy as jnp
from functools import partial,wraps
from jax import vmap
import jax
from diffcg._core import interpolate as custom_interpolate
from diffcg._core.geometry import distance, vectorized_angle_fn, vectorized_dihedral_fn
from diffcg._core.periodic import get_displacement_fn
from diffcg._core.math import high_precision_sum
from diffcg._core.units import NM_TO_ANGSTROM, KJMOL_TO_KCALMOL, RAD_TO_DEG
from jax_md import space, partition

def tabulated(dr: jnp.ndarray, spline: Callable[[jnp.ndarray], jnp.ndarray], **unused_kwargs
              ) -> jnp.ndarray:
    """
    Tabulated radial potential between particles given a spline function.

    Args:
        dr: An ndarray of pairwise distances between particles
        spline: A function computing the spline values at a given pairwise
                distance.

    Returns:
        Array of energies
    """

    return spline(dr)

def _smooth_cutoff_factor(dr, r_onset, r_cutoff):
  """Compute C1-smooth cutoff factor S(r) in [0, 1].

  S(r) = 1 for r < r_onset, S(r) = 0 for r > r_cutoff, and smoothly
  interpolated in between (HOOMD Blue scheme).

  Args:
    dr: An ndarray of pairwise distances.
    r_onset: Distance marking the onset of deformation.
    r_cutoff: Cutoff distance.

  Returns:
    Array of cutoff factors in [0, 1], same shape as dr.
  """
  r_c = r_cutoff ** 2
  r_o = r_onset ** 2
  r = dr ** 2

  inner = jnp.where(dr < r_cutoff,
                   (r_c - r)**2 * (r_c + 2 * r - 3 * r_o) / (r_c - r_o)**3,
                   0)
  return jnp.where(dr < r_onset, 1, inner)


def _smooth_cutoff_derivative(r, r_onset, r_cutoff):
    """Derivative of _smooth_cutoff_factor with respect to r (numpy)."""
    r = np.asarray(r, dtype=float)
    rc2 = r_cutoff ** 2
    ro2 = r_onset ** 2
    r2 = r ** 2
    result = np.zeros_like(r)
    mask = (r >= r_onset) & (r <= r_cutoff)
    r_m = r[mask]
    r2_m = r_m ** 2
    result[mask] = 12 * r_m * (rc2 - r2_m) * (ro2 - r2_m) / (rc2 - ro2) ** 3
    return result


def multiplicative_isotropic_cutoff(fn: Callable[..., jnp.ndarray],
                                    r_onset: float,
                                    r_cutoff: float) -> Callable[..., jnp.ndarray]:
  """Takes an isotropic function and constructs a truncated function.

  Given a function `f:R -> R`, we construct a new function `f':R -> R` such
  that `f'(r) = f(r)` for `r < r_onset`, `f'(r) = 0` for `r > r_cutoff`, and
  `f(r)` is :math:`C^1` everywhere. To do this, we follow the approach outlined
  in HOOMD Blue  [#hoomd]_ (thanks to Carl Goodrich for the pointer). We
  construct a function `S(r)` such that `S(r) = 1` for `r < r_onset`,
  `S(r) = 0` for `r > r_cutoff`, and `S(r)` is :math:`C^1`. Then
  `f'(r) = S(r)f(r)`.

  Args:
    fn: A function that takes an ndarray of distances of shape `[n, m]` as well
      as varargs.
    r_onset: A float specifying the distance marking the onset of deformation.
    r_cutoff: A float specifying the cutoff distance.

  Returns:
    A new function with the same signature as fn, with the properties outlined
    above.

  .. rubric:: References
  .. [#hoomd] HOOMD Blue documentation. Accessed on 05/31/2019.
      https://hoomd-blue.readthedocs.io/en/stable/module-md-pair.html#hoomd.md.pair.pair
  """

  @wraps(fn)
  def cutoff_fn(dr, *args, **kwargs):
    return _smooth_cutoff_factor(dr, r_onset, r_cutoff) * fn(dr, *args, **kwargs)

  return cutoff_fn

def generic_repulsion(dr,
                      sigma=1.,
                      epsilon=1.,
                      exp=12.,
                      **dynamic_kwargs):
    """
    Repulsive interaction between soft sphere particles: U = epsilon * (sigma / r)**exp.

    Args:
      dr: An ndarray of pairwise distances between particles.
      sigma: Repulsion length scale
      epsilon: Interaction energy scale
      exp: Exponent specifying interaction stiffness

    Returns:
      Array of energies
    """

    dr = jnp.where(dr > 1.e-8, dr, 1.e8)  # save masks dividing by 0
    idr = (sigma / dr)
    U = epsilon * idr ** exp
    return U


def lj(dr, sigma=1.0, epsilon=1.0, **dynamic_kwargs):
    """Lennard-Jones 12-6 potential: U = 4*epsilon*[(sigma/r)^12 - (sigma/r)^6].

    Ref: lammps/src/pair_lj_cut.cpp lines 116-129.

    Args:
      dr: Pairwise distances.
      sigma: Length scale where potential crosses zero.
      epsilon: Depth of the potential well.

    Returns:
      Array of energies.
    """
    dr = jnp.where(dr > 1.e-8, dr, 1.e8)
    idr = sigma / dr
    idr6 = idr ** 6
    return 4.0 * epsilon * (idr6 * idr6 - idr6)


def soft(dr, A=1.0, r_cutoff=1.0, **dynamic_kwargs):
    """Soft cosine potential: U = A*(1 + cos(pi*r/r_cutoff)).

    Ref: lammps/src/pair_soft.cpp lines 102-116.

    Args:
      dr: Pairwise distances.
      A: Prefactor (energy scale).
      r_cutoff: Cutoff distance (potential is zero at r_cutoff).

    Returns:
      Array of energies.
    """
    return A * (1.0 + jnp.cos(jnp.pi * dr / r_cutoff))


def morse(dr, D=1.0, alpha=1.0, r0=1.0, **dynamic_kwargs):
    """Morse potential: U = D*(exp(-2*alpha*(r-r0)) - 2*exp(-alpha*(r-r0))).

    Ref: lammps/src/pair_morse.cpp lines 105-117.
    Minimum = -D at r = r0.

    Args:
      dr: Pairwise distances.
      D: Well depth.
      alpha: Width parameter.
      r0: Equilibrium distance.

    Returns:
      Array of energies.
    """
    dexp = jnp.exp(-alpha * (dr - r0))
    return D * (dexp * dexp - 2.0 * dexp)


def mie(dr, sigma=1.0, epsilon=1.0, gamma_r=12.0, gamma_a=6.0, **dynamic_kwargs):
    """Mie (generalized LJ) potential.

    U = C*epsilon*[(sigma/r)^gamma_r - (sigma/r)^gamma_a]
    where C = (gamma_r/(gamma_r-gamma_a)) * (gamma_r/gamma_a)^(gamma_a/(gamma_r-gamma_a)).

    Ref: lammps/src/EXTRA-PAIR/pair_mie_cut.cpp lines 118-132, 536-541.
    Reduces to LJ when gamma_r=12, gamma_a=6 (C=4).

    Args:
      dr: Pairwise distances.
      sigma: Length scale.
      epsilon: Energy scale (well depth).
      gamma_r: Repulsive exponent.
      gamma_a: Attractive exponent.

    Returns:
      Array of energies.
    """
    dr = jnp.where(dr > 1.e-8, dr, 1.e8)
    C = (gamma_r / (gamma_r - gamma_a)) * (gamma_r / gamma_a) ** (gamma_a / (gamma_r - gamma_a))
    idr = sigma / dr
    return C * epsilon * (idr ** gamma_r - idr ** gamma_a)


def simple_spring(dr,
                  length=1,
                  epsilon=1,
                  alpha=2,
                  **unused_kwargs):
  """Isotropic spring potential with a given rest length.

  We define `simple_spring` to be a generalized Hookean spring with
  agreement when `alpha = 2`.
  """
  return epsilon / alpha * jnp.abs(dr - length) ** alpha

def harmonic_angle(angle,
                  angle_0=1,
                  epsilon=1,
                  alpha=2,
                  **unused_kwargs):
  """Isotropic harmonic angle potential with a given rest angle.
  We define `harmonic_angle` to be a generalized Hookian bending with
  agreement when alpha = 2.
  """
  return epsilon / alpha * (angle - angle_0) ** alpha

def harmonic_dihedral(angle,
                  angle_0=1,
                  epsilon=1,
                  alpha=2,
                  **unused_kwargs):
  """Isotropic harmonic angle potential with a given rest angle.
  We define `harmonic_angle` to be a generalized Hookian bending with
  agreement when alpha = 2.
  """
  return epsilon / alpha * (angle - angle_0) ** alpha

def build_pair_type_map(n_atom_types):
    """Build symmetric (type_i, type_j) -> pair_type_index mapping.

    Upper-triangle ordering: (0,0)=0, (0,1)=1, ..., (1,1)=n, ...
    Symmetric: map[i,j] == map[j,i].

    Args:
        n_atom_types: Number of distinct atom types.

    Returns:
        pair_type_map: (n_atom_types, n_atom_types) int32 array
        n_pair_types: int, total number of unique pair types = n*(n+1)//2
    """
    n = n_atom_types
    n_pair_types = n * (n + 1) // 2
    pair_map = jnp.zeros((n, n), dtype=jnp.int32)
    idx = 0
    for i in range(n):
        for j in range(i, n):
            pair_map = pair_map.at[i, j].set(idx)
            pair_map = pair_map.at[j, i].set(idx)
            idx += 1
    return pair_map, n_pair_types


def compute_pair_distances(positions, neighbors, disp_fn):
    """Compute pairwise distances from neighbor list (Dense or Sparse).

    Args:
        positions: (N, D) particle positions.
        neighbors: JAX-MD neighbor list object.
        disp_fn: JAX-MD displacement function.

    Returns:
        Tuple of (dr, dR, mask, normalization, type_i_idx, type_j_idx) where:
          - dr: pairwise distances (eps-safe)
          - dR: displacement vectors
          - mask: boolean mask for valid (non-padding) entries
          - normalization: 2.0 for Dense/Sparse (double-counted), 1.0 for OrderedSparse
          - type_i_idx: center atom indices (for per-type lookups)
          - type_j_idx: neighbor atom indices (for per-type lookups)
    """
    N = positions.shape[0]

    if partition.is_sparse(neighbors.format):
        # Sparse/OrderedSparse: neighbors.idx has shape [2, max_pairs]
        d = space.map_bond(disp_fn)
        safe_0 = jnp.minimum(neighbors.idx[0], N - 1)
        safe_1 = jnp.minimum(neighbors.idx[1], N - 1)
        dR = d(positions[safe_0], positions[safe_1])
        dr = vmap(distance)(dR)
        mask = neighbors.idx[0] < N
        normalization = 1.0 if neighbors.format is partition.NeighborListFormat.OrderedSparse else 2.0
        type_i_idx = safe_0
        type_j_idx = safe_1
    else:
        # Dense: neighbors.idx has shape [N, max_nbrs]
        idx = neighbors.idx
        safe_idx = jnp.minimum(idx, N - 1)
        R_neigh = positions[safe_idx]          # [N, max_nbrs, D]
        d = space.map_neighbor(disp_fn)
        dR = d(positions, R_neigh)             # [N, max_nbrs, D]
        dr = vmap(vmap(distance))(dR)          # [N, max_nbrs]
        mask = idx < N
        normalization = 2.0
        type_i_idx = jnp.arange(N)[:, None]   # [N, 1] broadcasts with [N, max_nbrs]
        type_j_idx = safe_idx                  # [N, max_nbrs]

    return dr, dR, mask, normalization, type_i_idx, type_j_idx


class TabulatedBondEnergy:
    interaction_type = "bond"

    def __init__(self, x_vals, y_vals, bonds, bond_types=None):
        self.x_vals = x_vals
        self.y_vals = jnp.atleast_2d(jnp.asarray(y_vals))  # (N_types, N_grid)
        self.bonds = bonds
        self.bond_types = bond_types
        self.n_types = self.y_vals.shape[0]

    def to_lammps(self, n_points=500):
        x = np.linspace(float(self.x_vals[0]), float(self.x_vals[-1]), n_points)
        tables = []
        for i in range(self.n_types):
            sp = custom_interpolate.MonotonicInterpolate(self.x_vals, self.y_vals[i])
            y = np.asarray(sp(x))
            tables.append({"keyword": f"BOND_{i}", "x": x, "y": y, "type_idx": i})
        return {
            "interaction_type": "bond",
            "lammps_style": "table",
            "n_points": n_points,
            "tables": tables,
        }

    def get_energy_fn(self):
        # Create spline for each type
        splines = [custom_interpolate.MonotonicInterpolate(self.x_vals, self.y_vals[i])
                   for i in range(self.n_types)]

        def energy_fn(system, neighbors, **dynamic_kwargs):
            positions = system.R
            disp_fn = get_displacement_fn(system.cell)
            Rb = positions[self.bonds[:, 1]]
            Ra = positions[self.bonds[:, 0]]
            edges = vmap(disp_fn)(Rb, Ra)  # Rb - Ra (JAX-MD: first - second)
            dr = vmap(distance)(edges)

            if self.bond_types is not None:
                def eval_spline(type_idx, d):
                    branches = [lambda d, s=s: s(d) for s in splines]
                    return jax.lax.switch(type_idx, branches, d)
                energies = vmap(eval_spline)(self.bond_types, dr)
            else:
                energies = splines[0](dr)

            return high_precision_sum(energies)
        return energy_fn

class HarmonicBondEnergy:
    interaction_type = "bond"

    def __init__(self, bonds, length=0.45, epsilon=5000, bond_types=None):
        self.bonds = bonds
        self.length = jnp.atleast_1d(jnp.asarray(length))
        self.epsilon = jnp.atleast_1d(jnp.asarray(epsilon))
        self.bond_types = bond_types

    def to_lammps(self):
        # simple_spring: epsilon/alpha * |r - length|^alpha, alpha=2
        # LAMMPS harmonic: K*(r - r0)^2, so K = epsilon/2
        n = len(self.length)
        coeffs = []
        for i in range(n):
            K_kjmol = float(self.epsilon[i]) / 2.0  # kJ/(mol*nm^2)
            r0_nm = float(self.length[i])
            # Convert: K in kcal/(mol*A^2), r0 in A
            K_kcal = K_kjmol * KJMOL_TO_KCALMOL / (NM_TO_ANGSTROM ** 2)
            r0_ang = r0_nm * NM_TO_ANGSTROM
            coeffs.append({"type_idx": i, "params": [K_kcal, r0_ang]})
        return {
            "interaction_type": "bond",
            "lammps_style": "harmonic",
            "coeffs": coeffs,
        }

    def get_energy_fn(self):

        def energy_fn(system, neighbors, **dynamic_kwargs):
            positions = system.R
            disp_fn = get_displacement_fn(system.cell)
            Rb = positions[self.bonds[:, 1]]
            Ra = positions[self.bonds[:, 0]]
            edges = vmap(disp_fn)(Rb, Ra)  # Rb - Ra (JAX-MD: first - second)
            dr = vmap(distance)(edges)

            if self.bond_types is not None:
                lengths = self.length[self.bond_types]
                epsilons = self.epsilon[self.bond_types]
            else:
                lengths = self.length[0]
                epsilons = self.epsilon[0]

            return high_precision_sum(simple_spring(dr, lengths, epsilons))
        return energy_fn


class TabulatedAngleEnergy:
    interaction_type = "angle"

    def __init__(self, x_vals, y_vals, angles, angle_types=None):
        self.x_vals = x_vals
        self.y_vals = jnp.atleast_2d(jnp.asarray(y_vals))  # (N_types, N_grid)
        self.angles = angles
        self.angle_types = angle_types
        self.n_types = self.y_vals.shape[0]

    def to_lammps(self, n_points=500):
        x = np.linspace(float(self.x_vals[0]), float(self.x_vals[-1]), n_points)
        tables = []
        for i in range(self.n_types):
            sp = custom_interpolate.MonotonicInterpolate(self.x_vals, self.y_vals[i])
            y = np.asarray(sp(x))
            tables.append({"keyword": f"ANGLE_{i}", "x": x, "y": y, "type_idx": i})
        return {
            "interaction_type": "angle",
            "lammps_style": "table",
            "n_points": n_points,
            "tables": tables,
        }

    def get_energy_fn(self):
        # Create spline for each type
        splines = [custom_interpolate.MonotonicInterpolate(self.x_vals, self.y_vals[i])
                   for i in range(self.n_types)]

        def energy_fn(system, neighbors, **dynamic_kwargs):
            positions = system.R
            disp_fn = get_displacement_fn(system.cell)
            # Swap args: JAX-MD disp(A,B) = A-B, so disp(j,k) = j-k = R_kj
            R_kj = vmap(disp_fn)(positions[self.angles[:,1]], positions[self.angles[:,2]])
            R_ij = vmap(disp_fn)(positions[self.angles[:,1]], positions[self.angles[:,0]])

            angles = vectorized_angle_fn(R_ij, R_kj)

            if self.angle_types is not None:
                def eval_spline(type_idx, a):
                    branches = [lambda a, s=s: s(a) for s in splines]
                    return jax.lax.switch(type_idx, branches, a)
                energies = vmap(eval_spline)(self.angle_types, angles)
            else:
                energies = splines[0](angles)

            return high_precision_sum(energies)
        return energy_fn

class HarmonicAngleEnergy:
    interaction_type = "angle"

    def __init__(self, angles, angle_0=1.5, epsilon=50, angle_types=None):
        self.angles = angles
        self.angle_0 = jnp.atleast_1d(jnp.asarray(angle_0))
        self.epsilon = jnp.atleast_1d(jnp.asarray(epsilon))
        self.angle_types = angle_types

    def to_lammps(self):
        # harmonic_angle: epsilon/alpha * (angle - angle_0)^alpha, alpha=2
        # LAMMPS harmonic: K*(theta - theta0)^2, so K = epsilon/2
        n = len(self.angle_0)
        coeffs = []
        for i in range(n):
            K_kjmol = float(self.epsilon[i]) / 2.0  # kJ/(mol*rad^2)
            theta0_rad = float(self.angle_0[i])
            # Convert: K in kcal/(mol*rad^2), theta0 in degrees
            K_kcal = K_kjmol * KJMOL_TO_KCALMOL / (RAD_TO_DEG ** 2)
            # Note: LAMMPS harmonic angle uses K*(theta-theta0)^2 with theta in degrees,
            # so K must be in kcal/(mol*deg^2)
            theta0_deg = theta0_rad * RAD_TO_DEG
            coeffs.append({"type_idx": i, "params": [K_kcal, theta0_deg]})
        return {
            "interaction_type": "angle",
            "lammps_style": "harmonic",
            "coeffs": coeffs,
        }

    def get_energy_fn(self):
        def energy_fn(system, neighbors, **dynamic_kwargs):
            positions = system.R
            disp_fn = get_displacement_fn(system.cell)
            R_kj = vmap(disp_fn)(positions[self.angles[:,1]], positions[self.angles[:,2]])
            R_ij = vmap(disp_fn)(positions[self.angles[:,1]], positions[self.angles[:,0]])

            angles = vectorized_angle_fn(R_ij, R_kj)

            if self.angle_types is not None:
                angle_0s = self.angle_0[self.angle_types]
                epsilons = self.epsilon[self.angle_types]
            else:
                angle_0s = self.angle_0[0]
                epsilons = self.epsilon[0]

            return high_precision_sum(harmonic_angle(angles, angle_0s, epsilons))
        return energy_fn

class TabulatedDihedralEnergy:
    interaction_type = "dihedral"

    def __init__(self, x_vals, y_vals, dihedrals, dihedral_types=None):
        self.x_vals = x_vals
        self.y_vals = jnp.atleast_2d(jnp.asarray(y_vals))  # (N_types, N_grid)
        self.dihedrals = dihedrals
        self.dihedral_types = dihedral_types
        self.n_types = self.y_vals.shape[0]

    def to_lammps(self, n_points=500):
        x = np.linspace(float(self.x_vals[0]), float(self.x_vals[-1]), n_points)
        tables = []
        for i in range(self.n_types):
            sp = custom_interpolate.MonotonicInterpolate(self.x_vals, self.y_vals[i])
            y = np.asarray(sp(x))
            tables.append({"keyword": f"DIHEDRAL_{i}", "x": x, "y": y, "type_idx": i})
        return {
            "interaction_type": "dihedral",
            "lammps_style": "table",
            "n_points": n_points,
            "tables": tables,
        }

    def get_energy_fn(self):
        # Create spline for each type
        splines = [custom_interpolate.MonotonicInterpolate(self.x_vals, self.y_vals[i])
                   for i in range(self.n_types)]

        def energy_fn(system, neighbors, **dynamic_kwargs):
            positions = system.R
            disp_fn = get_displacement_fn(system.cell)
            # disp_fn(A,B) = A-B, so swap args to get forward vectors (a→b, b→c, c→d)
            R_ab = vmap(disp_fn)(positions[self.dihedrals[:,1]], positions[self.dihedrals[:,0]])
            R_bc = vmap(disp_fn)(positions[self.dihedrals[:,2]], positions[self.dihedrals[:,1]])
            R_cd = vmap(disp_fn)(positions[self.dihedrals[:,3]], positions[self.dihedrals[:,2]])

            dihedrals = vectorized_dihedral_fn(R_ab, R_bc, R_cd)

            if self.dihedral_types is not None:
                def eval_spline(type_idx, d):
                    branches = [lambda d, s=s: s(d) for s in splines]
                    return jax.lax.switch(type_idx, branches, d)
                energies = vmap(eval_spline)(self.dihedral_types, dihedrals)
            else:
                energies = splines[0](dihedrals)

            return high_precision_sum(energies)
        return energy_fn

class HarmonicDihedralEnergy:
    interaction_type = "dihedral"

    def __init__(self, dihedrals, angle_0=1.5, epsilon=50, dihedral_types=None):
        self.dihedrals = dihedrals
        self.angle_0 = jnp.atleast_1d(jnp.asarray(angle_0))
        self.epsilon = jnp.atleast_1d(jnp.asarray(epsilon))
        self.dihedral_types = dihedral_types

    def to_lammps(self, n_points=500):
        # LAMMPS "harmonic" dihedral is cosine-based, not quadratic.
        # Always use table style for quadratic harmonic dihedrals.
        x = np.linspace(-np.pi + 1e-4, np.pi - 1e-4, n_points)
        tables = []
        n = len(self.angle_0)
        for i in range(n):
            y = np.asarray(harmonic_dihedral(
                x, angle_0=float(self.angle_0[i]), epsilon=float(self.epsilon[i])
            ))
            tables.append({"keyword": f"DIHEDRAL_{i}", "x": x, "y": y, "type_idx": i})
        return {
            "interaction_type": "dihedral",
            "lammps_style": "table",
            "n_points": n_points,
            "tables": tables,
        }

    def get_energy_fn(self):
        def energy_fn(system, neighbors, **dynamic_kwargs):
            positions = system.R
            disp_fn = get_displacement_fn(system.cell)
            # disp_fn(A,B) = A-B, so swap args to get forward vectors (a→b, b→c, c→d)
            R_ab = vmap(disp_fn)(positions[self.dihedrals[:,1]], positions[self.dihedrals[:,0]])
            R_bc = vmap(disp_fn)(positions[self.dihedrals[:,2]], positions[self.dihedrals[:,1]])
            R_cd = vmap(disp_fn)(positions[self.dihedrals[:,3]], positions[self.dihedrals[:,2]])
            dihedrals = vectorized_dihedral_fn(R_ab, R_bc, R_cd)

            if self.dihedral_types is not None:
                angle_0s = self.angle_0[self.dihedral_types]
                epsilons = self.epsilon[self.dihedral_types]
            else:
                angle_0s = self.angle_0[0]
                epsilons = self.epsilon[0]

            return high_precision_sum(harmonic_dihedral(dihedrals, angle_0s, epsilons))
        return energy_fn

class GenericRepulsionEnergy:
    interaction_type = "pair"

    def __init__(self, sigma=0.6, epsilon=1., exp=8, r_onset=0.9, r_cutoff=1.0,
                 atom_types=None, pair_type_map=None):
        self.r_onset = r_onset
        self.r_cutoff = r_cutoff
        self.atom_types = None if atom_types is None else jnp.asarray(atom_types, dtype=jnp.int32)
        self.pair_type_map = None if pair_type_map is None else jnp.asarray(pair_type_map, dtype=jnp.int32)

        if self.atom_types is not None:
            self.sigma = jnp.atleast_1d(jnp.asarray(sigma))
            self.epsilon = jnp.atleast_1d(jnp.asarray(epsilon))
            self.exp = jnp.atleast_1d(jnp.asarray(exp, dtype=jnp.float64))
        else:
            self.sigma = sigma
            self.epsilon = epsilon
            self.exp = exp

    def to_lammps(self, n_points=5000, r_min=0.05):
        """No native LAMMPS style for generic repulsion — always tabulated."""
        x = np.linspace(r_min, float(self.r_cutoff), n_points)
        S = np.asarray(_smooth_cutoff_factor(x, self.r_onset, self.r_cutoff))
        dS = _smooth_cutoff_derivative(x, self.r_onset, self.r_cutoff)
        tables = []
        if self.atom_types is not None:
            ptm = np.asarray(self.pair_type_map)
            n_at = ptm.shape[0]
            seen = set()
            for i in range(n_at):
                for j in range(i, n_at):
                    pt = int(ptm[i, j])
                    if pt in seen:
                        continue
                    seen.add(pt)
                    sig = float(self.sigma[pt])
                    eps = float(self.epsilon[pt])
                    exp_v = float(self.exp[pt])
                    U = generic_repulsion(x, sigma=sig, epsilon=eps, exp=exp_v)
                    U = np.asarray(U)
                    dU = -exp_v * U / np.where(x > 0, x, np.inf)
                    dE_dr = dS * U + S * dU
                    with np.errstate(divide='ignore', invalid='ignore'):
                        force_vals = -dE_dr / np.where(x > 0, x, np.inf)
                    force_vals = np.nan_to_num(force_vals, nan=0.0, posinf=0.0, neginf=0.0)
                    y = S * U
                    tables.append({"keyword": f"PAIR_{pt}", "x": x, "y": y,
                                   "force_vals": force_vals, "types": (i, j)})
        else:
            U = generic_repulsion(x, sigma=self.sigma, epsilon=self.epsilon, exp=self.exp)
            U = np.asarray(U)
            dU = -self.exp * U / np.where(x > 0, x, np.inf)
            dE_dr = dS * U + S * dU
            with np.errstate(divide='ignore', invalid='ignore'):
                force_vals = -dE_dr / np.where(x > 0, x, np.inf)
            force_vals = np.nan_to_num(force_vals, nan=0.0, posinf=0.0, neginf=0.0)
            y = S * U
            tables.append({"keyword": "PAIR_0", "x": x, "y": y,
                           "force_vals": force_vals, "types": None})
        return {
            "interaction_type": "pair",
            "lammps_style": "table",
            "n_points": n_points,
            "cutoff": float(self.r_cutoff) * NM_TO_ANGSTROM,
            "tables": tables,
        }

    def get_energy_fn(self):
        def energy_fn(system, neighbors, **dynamic_kwargs):
            positions = system.R
            disp_fn = get_displacement_fn(system.cell)
            dr, dR, mask, normalization, type_i_idx, type_j_idx = compute_pair_distances(
                positions, neighbors, disp_fn)

            if self.atom_types is not None:
                type_i = self.atom_types[type_i_idx]
                type_j = self.atom_types[type_j_idx]
                pair_types = self.pair_type_map[type_i, type_j]
                sigma_vals = self.sigma[pair_types]
                epsilon_vals = self.epsilon[pair_types]
                exp_vals = self.exp[pair_types]
            else:
                sigma_vals, epsilon_vals, exp_vals = self.sigma, self.epsilon, self.exp

            _energy = _smooth_cutoff_factor(dr, self.r_onset, self.r_cutoff) * generic_repulsion(dr, sigma_vals, epsilon_vals, exp_vals)
            out = _energy * mask
            return high_precision_sum(out) / normalization

        return energy_fn


class LJEnergy:
    """Lennard-Jones 12-6 pair energy with smooth cutoff and per-type support."""
    interaction_type = "pair"

    def __init__(self, sigma=1.0, epsilon=1.0, r_onset=2.0, r_cutoff=2.5,
                 atom_types=None, pair_type_map=None):
        self.r_onset = r_onset
        self.r_cutoff = r_cutoff
        self.atom_types = None if atom_types is None else jnp.asarray(atom_types, dtype=jnp.int32)
        self.pair_type_map = None if pair_type_map is None else jnp.asarray(pair_type_map, dtype=jnp.int32)

        if self.atom_types is not None:
            self.sigma = jnp.atleast_1d(jnp.asarray(sigma))
            self.epsilon = jnp.atleast_1d(jnp.asarray(epsilon))
        else:
            self.sigma = sigma
            self.epsilon = epsilon

    def to_lammps(self):
        cutoff_ang = float(self.r_cutoff) * NM_TO_ANGSTROM
        coeffs = []
        if self.atom_types is not None:
            ptm = np.asarray(self.pair_type_map)
            n_at = ptm.shape[0]
            seen = set()
            for i in range(n_at):
                for j in range(i, n_at):
                    pt = int(ptm[i, j])
                    if pt in seen:
                        continue
                    seen.add(pt)
                    eps_kcal = float(self.epsilon[pt]) * KJMOL_TO_KCALMOL
                    sig_ang = float(self.sigma[pt]) * NM_TO_ANGSTROM
                    coeffs.append({"types": (i, j), "params": [eps_kcal, sig_ang]})
        else:
            eps_kcal = float(self.epsilon) * KJMOL_TO_KCALMOL
            sig_ang = float(self.sigma) * NM_TO_ANGSTROM
            coeffs.append({"types": None, "params": [eps_kcal, sig_ang]})
        return {
            "interaction_type": "pair",
            "lammps_style": "lj/cut",
            "cutoff": cutoff_ang,
            "coeffs": coeffs,
        }

    def get_energy_fn(self):
        def energy_fn(system, neighbors, **dynamic_kwargs):
            positions = system.R
            disp_fn = get_displacement_fn(system.cell)
            dr, dR, mask, normalization, type_i_idx, type_j_idx = compute_pair_distances(
                positions, neighbors, disp_fn)

            if self.atom_types is not None:
                type_i = self.atom_types[type_i_idx]
                type_j = self.atom_types[type_j_idx]
                pair_types = self.pair_type_map[type_i, type_j]
                sigma_vals = self.sigma[pair_types]
                epsilon_vals = self.epsilon[pair_types]
            else:
                sigma_vals, epsilon_vals = self.sigma, self.epsilon

            _energy = _smooth_cutoff_factor(dr, self.r_onset, self.r_cutoff) * lj(dr, sigma_vals, epsilon_vals)
            out = _energy * mask
            return high_precision_sum(out) / normalization

        return energy_fn


class SoftEnergy:
    """Soft cosine pair energy with per-type support.

    No r_onset needed — potential is naturally zero at r_cutoff.
    """
    interaction_type = "pair"

    def __init__(self, A=1.0, r_cutoff=1.0,
                 atom_types=None, pair_type_map=None):
        self.r_cutoff = r_cutoff
        self.atom_types = None if atom_types is None else jnp.asarray(atom_types, dtype=jnp.int32)
        self.pair_type_map = None if pair_type_map is None else jnp.asarray(pair_type_map, dtype=jnp.int32)

        if self.atom_types is not None:
            self.A = jnp.atleast_1d(jnp.asarray(A))
        else:
            self.A = A

    def to_lammps(self):
        cutoff_ang = float(self.r_cutoff) * NM_TO_ANGSTROM
        coeffs = []
        if self.atom_types is not None:
            ptm = np.asarray(self.pair_type_map)
            n_at = ptm.shape[0]
            seen = set()
            for i in range(n_at):
                for j in range(i, n_at):
                    pt = int(ptm[i, j])
                    if pt in seen:
                        continue
                    seen.add(pt)
                    A_kcal = float(self.A[pt]) * KJMOL_TO_KCALMOL
                    coeffs.append({"types": (i, j), "params": [A_kcal]})
        else:
            A_kcal = float(self.A) * KJMOL_TO_KCALMOL
            coeffs.append({"types": None, "params": [A_kcal]})
        return {
            "interaction_type": "pair",
            "lammps_style": "soft",
            "cutoff": cutoff_ang,
            "coeffs": coeffs,
        }

    def get_energy_fn(self):
        def energy_fn(system, neighbors, **dynamic_kwargs):
            positions = system.R
            disp_fn = get_displacement_fn(system.cell)
            dr, dR, mask, normalization, type_i_idx, type_j_idx = compute_pair_distances(
                positions, neighbors, disp_fn)

            if self.atom_types is not None:
                type_i = self.atom_types[type_i_idx]
                type_j = self.atom_types[type_j_idx]
                pair_types = self.pair_type_map[type_i, type_j]
                A_vals = self.A[pair_types]
            else:
                A_vals = self.A

            _energy = jnp.where(dr < self.r_cutoff, soft(dr, A_vals, self.r_cutoff), 0.0)
            out = _energy * mask
            return high_precision_sum(out) / normalization

        return energy_fn


class MorseEnergy:
    """Morse pair energy with smooth cutoff and per-type support."""
    interaction_type = "pair"

    def __init__(self, D=1.0, alpha=1.0, r0=1.0, r_onset=2.0, r_cutoff=2.5,
                 atom_types=None, pair_type_map=None):
        self.r_onset = r_onset
        self.r_cutoff = r_cutoff
        self.atom_types = None if atom_types is None else jnp.asarray(atom_types, dtype=jnp.int32)
        self.pair_type_map = None if pair_type_map is None else jnp.asarray(pair_type_map, dtype=jnp.int32)

        if self.atom_types is not None:
            self.D = jnp.atleast_1d(jnp.asarray(D))
            self.alpha = jnp.atleast_1d(jnp.asarray(alpha))
            self.r0 = jnp.atleast_1d(jnp.asarray(r0))
        else:
            self.D = D
            self.alpha = alpha
            self.r0 = r0

    def to_lammps(self):
        cutoff_ang = float(self.r_cutoff) * NM_TO_ANGSTROM
        coeffs = []
        if self.atom_types is not None:
            ptm = np.asarray(self.pair_type_map)
            n_at = ptm.shape[0]
            seen = set()
            for i in range(n_at):
                for j in range(i, n_at):
                    pt = int(ptm[i, j])
                    if pt in seen:
                        continue
                    seen.add(pt)
                    D_kcal = float(self.D[pt]) * KJMOL_TO_KCALMOL
                    alpha_inv_ang = float(self.alpha[pt]) / NM_TO_ANGSTROM  # 1/nm -> 1/A
                    r0_ang = float(self.r0[pt]) * NM_TO_ANGSTROM
                    coeffs.append({"types": (i, j), "params": [D_kcal, alpha_inv_ang, r0_ang]})
        else:
            D_kcal = float(self.D) * KJMOL_TO_KCALMOL
            alpha_inv_ang = float(self.alpha) / NM_TO_ANGSTROM
            r0_ang = float(self.r0) * NM_TO_ANGSTROM
            coeffs.append({"types": None, "params": [D_kcal, alpha_inv_ang, r0_ang]})
        return {
            "interaction_type": "pair",
            "lammps_style": "morse",
            "cutoff": cutoff_ang,
            "coeffs": coeffs,
        }

    def get_energy_fn(self):
        def energy_fn(system, neighbors, **dynamic_kwargs):
            positions = system.R
            disp_fn = get_displacement_fn(system.cell)
            dr, dR, mask, normalization, type_i_idx, type_j_idx = compute_pair_distances(
                positions, neighbors, disp_fn)

            if self.atom_types is not None:
                type_i = self.atom_types[type_i_idx]
                type_j = self.atom_types[type_j_idx]
                pair_types = self.pair_type_map[type_i, type_j]
                D_vals = self.D[pair_types]
                alpha_vals = self.alpha[pair_types]
                r0_vals = self.r0[pair_types]
            else:
                D_vals, alpha_vals, r0_vals = self.D, self.alpha, self.r0

            _energy = _smooth_cutoff_factor(dr, self.r_onset, self.r_cutoff) * morse(dr, D_vals, alpha_vals, r0_vals)
            out = _energy * mask
            return high_precision_sum(out) / normalization

        return energy_fn


class MieEnergy:
    """Mie (generalized LJ) pair energy with smooth cutoff and per-type support."""
    interaction_type = "pair"

    def __init__(self, sigma=1.0, epsilon=1.0, gamma_r=12.0, gamma_a=6.0,
                 r_onset=2.0, r_cutoff=2.5,
                 atom_types=None, pair_type_map=None):
        self.r_onset = r_onset
        self.r_cutoff = r_cutoff
        self.atom_types = None if atom_types is None else jnp.asarray(atom_types, dtype=jnp.int32)
        self.pair_type_map = None if pair_type_map is None else jnp.asarray(pair_type_map, dtype=jnp.int32)

        if self.atom_types is not None:
            self.sigma = jnp.atleast_1d(jnp.asarray(sigma))
            self.epsilon = jnp.atleast_1d(jnp.asarray(epsilon))
            self.gamma_r = jnp.atleast_1d(jnp.asarray(gamma_r))
            self.gamma_a = jnp.atleast_1d(jnp.asarray(gamma_a))
        else:
            self.sigma = sigma
            self.epsilon = epsilon
            self.gamma_r = gamma_r
            self.gamma_a = gamma_a

    def to_lammps(self):
        cutoff_ang = float(self.r_cutoff) * NM_TO_ANGSTROM
        coeffs = []
        if self.atom_types is not None:
            ptm = np.asarray(self.pair_type_map)
            n_at = ptm.shape[0]
            seen = set()
            for i in range(n_at):
                for j in range(i, n_at):
                    pt = int(ptm[i, j])
                    if pt in seen:
                        continue
                    seen.add(pt)
                    eps_kcal = float(self.epsilon[pt]) * KJMOL_TO_KCALMOL
                    sig_ang = float(self.sigma[pt]) * NM_TO_ANGSTROM
                    gr = float(self.gamma_r[pt])
                    ga = float(self.gamma_a[pt])
                    coeffs.append({"types": (i, j), "params": [eps_kcal, sig_ang, gr, ga]})
        else:
            eps_kcal = float(self.epsilon) * KJMOL_TO_KCALMOL
            sig_ang = float(self.sigma) * NM_TO_ANGSTROM
            gr = float(self.gamma_r)
            ga = float(self.gamma_a)
            coeffs.append({"types": None, "params": [eps_kcal, sig_ang, gr, ga]})
        return {
            "interaction_type": "pair",
            "lammps_style": "mie/cut",
            "cutoff": cutoff_ang,
            "coeffs": coeffs,
        }

    def get_energy_fn(self):
        def energy_fn(system, neighbors, **dynamic_kwargs):
            positions = system.R
            disp_fn = get_displacement_fn(system.cell)
            dr, dR, mask, normalization, type_i_idx, type_j_idx = compute_pair_distances(
                positions, neighbors, disp_fn)

            if self.atom_types is not None:
                type_i = self.atom_types[type_i_idx]
                type_j = self.atom_types[type_j_idx]
                pair_types = self.pair_type_map[type_i, type_j]
                sigma_vals = self.sigma[pair_types]
                epsilon_vals = self.epsilon[pair_types]
                gamma_r_vals = self.gamma_r[pair_types]
                gamma_a_vals = self.gamma_a[pair_types]
            else:
                sigma_vals, epsilon_vals = self.sigma, self.epsilon
                gamma_r_vals, gamma_a_vals = self.gamma_r, self.gamma_a

            _energy = _smooth_cutoff_factor(dr, self.r_onset, self.r_cutoff) * mie(dr, sigma_vals, epsilon_vals, gamma_r_vals, gamma_a_vals)
            out = _energy * mask
            return high_precision_sum(out) / normalization

        return energy_fn


class TabulatedPairEnergy:
    interaction_type = "pair"

    def __init__(self, x_vals, y_vals, r_onset, r_cutoff,
                 atom_types=None, pair_type_map=None):
        self.x_vals = x_vals
        self.y_vals = jnp.atleast_2d(jnp.asarray(y_vals))  # (n_pair_types, N_grid) or (1, N_grid)
        self.r_onset = r_onset
        self.r_cutoff = r_cutoff
        self.atom_types = None if atom_types is None else jnp.asarray(atom_types, dtype=jnp.int32)
        self.pair_type_map = None if pair_type_map is None else jnp.asarray(pair_type_map, dtype=jnp.int32)
        self.n_pair_types = self.y_vals.shape[0]

    def to_lammps(self, n_points=5000, r_min=0.05):
        x = np.linspace(r_min, float(self.r_cutoff), n_points)
        cutoff_ang = float(self.r_cutoff) * NM_TO_ANGSTROM
        S = np.asarray(_smooth_cutoff_factor(x, self.r_onset, self.r_cutoff))
        dS = _smooth_cutoff_derivative(x, self.r_onset, self.r_cutoff)
        tables = []
        if self.atom_types is not None:
            ptm = np.asarray(self.pair_type_map)
            n_at = ptm.shape[0]
            seen = set()
            for i in range(n_at):
                for j in range(i, n_at):
                    pt = int(ptm[i, j])
                    if pt in seen:
                        continue
                    seen.add(pt)
                    sp = custom_interpolate.MonotonicInterpolate(self.x_vals, self.y_vals[pt])
                    U = np.asarray(sp(x))
                    dU = np.asarray(sp.derivative(x))
                    dE_dr = dS * U + S * dU
                    with np.errstate(divide='ignore', invalid='ignore'):
                        force_vals = -dE_dr / np.where(x > 0, x, np.inf)
                    force_vals = np.nan_to_num(force_vals, nan=0.0, posinf=0.0, neginf=0.0)
                    y = S * U
                    tables.append({"keyword": f"PAIR_{pt}", "x": x, "y": y,
                                   "force_vals": force_vals, "types": (i, j)})
        else:
            sp = custom_interpolate.MonotonicInterpolate(self.x_vals, self.y_vals[0])
            U = np.asarray(sp(x))
            dU = np.asarray(sp.derivative(x))
            dE_dr = dS * U + S * dU
            with np.errstate(divide='ignore', invalid='ignore'):
                force_vals = -dE_dr / np.where(x > 0, x, np.inf)
            force_vals = np.nan_to_num(force_vals, nan=0.0, posinf=0.0, neginf=0.0)
            y = S * U
            tables.append({"keyword": "PAIR_0", "x": x, "y": y,
                           "force_vals": force_vals, "types": None})
        return {
            "interaction_type": "pair",
            "lammps_style": "table",
            "n_points": n_points,
            "cutoff": cutoff_ang,
            "tables": tables,
        }

    def get_energy_fn(self):
        # Create one spline per pair type
        splines = [custom_interpolate.MonotonicInterpolate(self.x_vals, self.y_vals[i])
                    for i in range(self.n_pair_types)]

        if self.atom_types is None:
            # Single-type path: use first (only) spline with multiplicative cutoff
            tabulated_partial = partial(tabulated, spline=splines[0])
            truncated_fn = multiplicative_isotropic_cutoff(tabulated_partial, self.r_onset, self.r_cutoff)
        else:
            truncated_fn = None  # not used in per-type path

        def energy_fn(system, neighbors, **dynamic_kwargs):
            positions = system.R
            disp_fn = get_displacement_fn(system.cell)
            dr, dR, mask, normalization, type_i_idx, type_j_idx = compute_pair_distances(
                positions, neighbors, disp_fn)

            if self.atom_types is not None:
                type_i = self.atom_types[type_i_idx]
                type_j = self.atom_types[type_j_idx]
                pair_types = self.pair_type_map[type_i, type_j]
                # Flatten for spline evaluation, then reshape
                flat_pair_types = pair_types.ravel()
                flat_dr = dr.ravel()
                def eval_spline(type_idx, d):
                    return jax.lax.switch(type_idx, [lambda d, s=s: s(d) for s in splines], d)
                raw_energies = vmap(eval_spline)(flat_pair_types, flat_dr)
                raw_energies = raw_energies.reshape(dr.shape)
                _energy = _smooth_cutoff_factor(dr, self.r_onset, self.r_cutoff) * raw_energies
            else:
                _energy = truncated_fn(dr)

            out = _energy * mask
            return high_precision_sum(out) / normalization

        return energy_fn


# ---------------------------------------------------------------------------
# LAMMPS spec combiner
# ---------------------------------------------------------------------------

def build_lammps_specs(*energy_objects):
    """Combine energy objects into LAMMPS style/coeff specifications.

    Returns dict with keys 'pair', 'bond', 'angle', 'dihedral'.
    Each value describes the style(s) and coefficients needed.
    Handles hybrid/overlay when multiple pair styles are needed.

    Args:
        *energy_objects: Energy class instances that have ``to_lammps()`` methods.

    Returns:
        Dict mapping interaction type to LAMMPS specification dict.
    """
    groups: Dict[str, list] = {"pair": [], "bond": [], "angle": [], "dihedral": []}
    for obj in energy_objects:
        itype = obj.interaction_type
        spec = obj.to_lammps()
        groups[itype].append(spec)

    result = {}

    for itype in ("pair", "bond", "angle", "dihedral"):
        specs = groups[itype]
        if not specs:
            continue

        styles = set(s["lammps_style"] for s in specs)

        if len(styles) == 1:
            style = styles.pop()
            if style == "table":
                result[itype] = _merge_table_specs(specs, itype)
            else:
                all_coeffs = []
                cutoff = max(s.get("cutoff", 0) for s in specs)
                for s in specs:
                    all_coeffs.extend(s.get("coeffs", []))
                result[itype] = {
                    "lammps_style": style,
                    "cutoff": cutoff,
                    "coeffs": all_coeffs,
                }
        else:
            if itype == "pair":
                result[itype] = _build_hybrid_pair_spec(specs)
            else:
                result[itype] = _build_hybrid_bonded_spec(specs, itype)

    return result


def _merge_table_specs(specs, itype):
    """Merge multiple table-style specs, summing y-values for same type indices."""
    merged_tables = {}

    for spec in specs:
        for table in spec["tables"]:
            if "type_idx" in table:
                key = table["type_idx"]
            elif "types" in table:
                key = table["types"]
            else:
                key = 0

            if key in merged_tables:
                existing = merged_tables[key]
                if len(existing["x"]) == len(table["x"]) and np.allclose(existing["x"], table["x"]):
                    existing["y"] = existing["y"] + table["y"]
                else:
                    x_min = max(existing["x"][0], table["x"][0])
                    x_max = min(existing["x"][-1], table["x"][-1])
                    n = max(len(existing["x"]), len(table["x"]))
                    x_new = np.linspace(x_min, x_max, n)
                    y1 = np.interp(x_new, existing["x"], existing["y"])
                    y2 = np.interp(x_new, table["x"], table["y"])
                    existing["x"] = x_new
                    existing["y"] = y1 + y2
            else:
                merged_tables[key] = {
                    "keyword": table["keyword"],
                    "x": np.asarray(table["x"]),
                    "y": np.asarray(table["y"]),
                }
                if "types" in table:
                    merged_tables[key]["types"] = table["types"]
                if "type_idx" in table:
                    merged_tables[key]["type_idx"] = table["type_idx"]

    n_points = max(len(t["x"]) for t in merged_tables.values())
    cutoff = max((s.get("cutoff", 0) for s in specs), default=0)

    return {
        "lammps_style": "table",
        "n_points": n_points,
        "cutoff": cutoff,
        "tables": list(merged_tables.values()),
    }


def _build_hybrid_pair_spec(specs):
    """Build a hybrid/overlay pair style spec from mixed native + table specs."""
    style_parts = []
    seen_styles = {}
    all_items = []

    for spec in specs:
        style = spec["lammps_style"]
        cutoff = spec.get("cutoff", 0)

        if style not in seen_styles:
            if style == "table":
                n_points = spec.get("n_points", 500)
                # Table style: no cutoff in pair_style declaration
                style_parts.append((f"table linear {n_points}", None))
            else:
                style_parts.append((style, cutoff))
            seen_styles[style] = True

        if style == "table":
            for table in spec["tables"]:
                all_items.append(("table", table))
        else:
            for coeff in spec.get("coeffs", []):
                all_items.append((style, coeff))

    return {
        "lammps_style": "hybrid/overlay",
        "style_parts": style_parts,
        "items": all_items,
    }


def _build_hybrid_bonded_spec(specs, itype):
    """Build a hybrid bonded style spec from mixed native + table specs."""
    style_parts = []
    seen_styles = {}
    all_items = []

    for spec in specs:
        style = spec["lammps_style"]
        if style not in seen_styles:
            if style == "table":
                n_points = spec.get("n_points", 500)
                style_parts.append(f"table linear {n_points}")
            else:
                style_parts.append(style)
            seen_styles[style] = True

        if style == "table":
            for table in spec["tables"]:
                all_items.append(("table", table))
        else:
            for coeff in spec.get("coeffs", []):
                all_items.append((style, coeff))

    return {
        "lammps_style": "hybrid",
        "style_parts": style_parts,
        "items": all_items,
    }