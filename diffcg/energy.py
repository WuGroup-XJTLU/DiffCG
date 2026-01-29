# SPDX-License-Identifier: MIT
# Copyright (c) 2025 WuResearchGroup

"""Custom definition of some potential energy functions."""
from typing import Callable

import jax.numpy as jnp
from functools import partial,wraps
from jax import vmap
import jax
from diffcg.util import custom_interpolate
from diffcg.common.geometry import distance, vectorized_angle_fn, vectorized_dihedral_fn
from diffcg.common.periodic import displacement, make_displacement_with_cached_inverse
from diffcg.util.math import high_precision_sum

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

  r_c = r_cutoff ** 2
  r_o = r_onset ** 2

  def smooth_fn(dr):
    r = dr ** 2

    inner = jnp.where(dr < r_cutoff,
                     (r_c - r)**2 * (r_c + 2 * r - 3 * r_o) / (r_c - r_o)**3,
                     0)

    return jnp.where(dr < r_onset, 1, inner)

  @wraps(fn)
  def cutoff_fn(dr, *args, **kwargs):
    return smooth_fn(dr) * fn(dr, *args, **kwargs)

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

def build_bonded_pair_set(topology):
    """Pre-compute set of bonded atom pairs for O(1) lookup.

    Extracts all unique pairs of atoms that are bonded through the topology
    (bonds, angles, or dihedrals) and returns them as a sorted array for
    efficient lookup via searchsorted.

    Args:
        topology: (num_entries, n_cols) array where n_cols is 2, 3, or 4
                  for bonds, angles, or dihedrals respectively.

    Returns:
        bonded_pairs: (num_pairs, 2) int32 array of sorted (min, max) pairs
    """
    n_cols = topology.shape[1]
    pairs_list = []

    # Extract all pairs within each topology entry
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            # Stack pairs as (atom_i, atom_j) for each topology row
            col_i = topology[:, i]
            col_j = topology[:, j]
            # Normalize: smaller index first
            pair_min = jnp.minimum(col_i, col_j)
            pair_max = jnp.maximum(col_i, col_j)
            pairs_list.append(jnp.stack([pair_min, pair_max], axis=1))

    # Concatenate all pairs
    all_pairs = jnp.concatenate(pairs_list, axis=0)

    # Remove duplicates by using unique on a combined key
    # Encode pairs as single integers for unique operation
    max_atom = jnp.max(all_pairs) + 1
    pair_keys = all_pairs[:, 0] * max_atom + all_pairs[:, 1]
    unique_keys = jnp.unique(pair_keys)

    # Decode back to pairs
    unique_pairs = jnp.stack([unique_keys // max_atom, unique_keys % max_atom], axis=1)
    return unique_pairs.astype(jnp.int32)


def mask_bonded_neighbors(idx, topology, max_num_atoms, bonded_pairs=None):
    """
    Mask neighbor pairs that are bonded through topology interactions.

    This function identifies pairs of atoms that are bonded through topology
    interactions and masks them by setting their indices to max_num_atoms.
    This prevents double-counting of interactions between bonded atoms.

    Args:
        idx: Tuple of (i_indices, j_indices) for neighbor pairs
        topology: (num_entries, n_cols) array of topology definitions
        max_num_atoms (int): Maximum number of atoms, used as mask value
        bonded_pairs: Optional pre-computed bonded pairs from build_bonded_pair_set().
                      If None, will be computed on-the-fly.

    Returns:
        Tuple of (i_masked, j_masked) with bonded pairs masked to max_num_atoms
    """
    i = idx[0]  # (num_pairs,) - first atom in each pair
    j = idx[1]  # (num_pairs,) - second atom in each pair

    # Build bonded pairs if not provided
    if bonded_pairs is None:
        bonded_pairs = build_bonded_pair_set(topology)

    # Normalize input pairs (smaller index first)
    pair_min = jnp.minimum(i, j)
    pair_max = jnp.maximum(i, j)

    # Encode pairs as single integers for lookup
    max_atom = jnp.maximum(jnp.max(bonded_pairs) + 1, max_num_atoms + 1)
    query_keys = pair_min * max_atom + pair_max
    bonded_keys = bonded_pairs[:, 0] * max_atom + bonded_pairs[:, 1]

    # Sort bonded_keys for searchsorted
    sorted_indices = jnp.argsort(bonded_keys)
    sorted_bonded_keys = bonded_keys[sorted_indices]

    # Use searchsorted to find potential matches
    insert_indices = jnp.searchsorted(sorted_bonded_keys, query_keys)

    # Check if the found index actually matches (handles boundary cases)
    insert_indices_safe = jnp.minimum(insert_indices, len(sorted_bonded_keys) - 1)
    bonded_mask = sorted_bonded_keys[insert_indices_safe] == query_keys

    # Set both i and j to max_num_atoms if bonded, else keep original
    i_masked = jnp.where(bonded_mask, max_num_atoms, i)
    j_masked = jnp.where(bonded_mask, max_num_atoms, j)
    return i_masked, j_masked


def _get_edge_list_from_jaxmd_neighbors(neighbors):
    """Convert JAX-MD Sparse neighbor list to edge-list format.

    JAX-MD Sparse format: neighbors.idx with shape [N, max_neighbors]
      - idx[i, j] = j-th neighbor of particle i
      - Padding value = N for unused slots

    Args:
        neighbors: JAX-MD neighbor list object with .idx attribute

    Returns:
        Tuple of (centers, others, N) where:
          - centers: 1D array of center atom indices
          - others: 1D array of neighbor atom indices
          - N: Number of particles (from neighbor list)
    """
    idx = neighbors.idx
    N = idx.shape[0]
    max_neighbors = idx.shape[1]
    centers = jnp.repeat(jnp.arange(N), max_neighbors)
    others = idx.flatten()
    return centers, others, N


class TabulatedBondEnergy:
    def __init__(self, x_vals, y_vals, bonds, bond_types=None):
        self.x_vals = x_vals
        self.y_vals = jnp.atleast_2d(jnp.asarray(y_vals))  # (N_types, N_grid)
        self.bonds = bonds
        self.bond_types = bond_types
        self.n_types = self.y_vals.shape[0]

    def get_energy_fn(self):
        # Create spline for each type
        splines = [custom_interpolate.MonotonicInterpolate(self.x_vals, self.y_vals[i])
                   for i in range(self.n_types)]

        def energy_fn(system, neighbors, **dynamic_kwargs):
            positions = system.R
            # Use cached displacement function with pre-computed cell inverse
            disp_fn = make_displacement_with_cached_inverse(system.cell)
            Ra = positions[self.bonds[:, 0]]
            Rb = positions[self.bonds[:, 1]]
            edges = vmap(disp_fn)(Ra, Rb)
            dr = vmap(distance)(edges)

            if self.bond_types is not None:
                # Use lax.switch to select spline per bond - more efficient for many types
                def eval_spline(type_idx, d):
                    branches = [lambda d, s=s: s(d) for s in splines]
                    return jax.lax.switch(type_idx, branches, d)
                energies = vmap(eval_spline)(self.bond_types, dr)
            else:
                energies = splines[0](dr)

            return high_precision_sum(energies)
        return energy_fn

class HarmonicBondEnergy:
    def __init__(self, bonds, length=0.45, epsilon=5000, bond_types=None):
        self.bonds = bonds
        self.length = jnp.atleast_1d(jnp.asarray(length))
        self.epsilon = jnp.atleast_1d(jnp.asarray(epsilon))
        self.bond_types = bond_types

    def get_energy_fn(self):

        def energy_fn(system, neighbors, **dynamic_kwargs):
            positions = system.R
            # Use cached displacement function with pre-computed cell inverse
            disp_fn = make_displacement_with_cached_inverse(system.cell)
            Ra = positions[self.bonds[:, 0]]
            Rb = positions[self.bonds[:, 1]]
            edges = vmap(disp_fn)(Ra, Rb)
            dr = vmap(distance)(edges)

            if self.bond_types is not None:
                # Per-type: index parameters by type
                lengths = self.length[self.bond_types]
                epsilons = self.epsilon[self.bond_types]
            else:
                # Backward compatible: broadcast scalar
                lengths = self.length[0]
                epsilons = self.epsilon[0]

            return high_precision_sum(simple_spring(dr, lengths, epsilons))
        return energy_fn


class TabulatedAngleEnergy:
    def __init__(self, x_vals, y_vals, angles, angle_types=None):
        self.x_vals = x_vals
        self.y_vals = jnp.atleast_2d(jnp.asarray(y_vals))  # (N_types, N_grid)
        self.angles = angles
        self.angle_types = angle_types
        self.n_types = self.y_vals.shape[0]

    def get_energy_fn(self):
        # Create spline for each type
        splines = [custom_interpolate.MonotonicInterpolate(self.x_vals, self.y_vals[i])
                   for i in range(self.n_types)]

        def energy_fn(system, neighbors, **dynamic_kwargs):
            positions = system.R
            # Use cached displacement function with pre-computed cell inverse
            disp_fn = make_displacement_with_cached_inverse(system.cell)
            R_kj = vmap(disp_fn)(positions[self.angles[:,2]], positions[self.angles[:,1]])
            R_ij = vmap(disp_fn)(positions[self.angles[:,0]], positions[self.angles[:,1]])

            angles = vectorized_angle_fn(R_ij, R_kj)

            if self.angle_types is not None:
                # Use lax.switch to select spline per angle - more efficient for many types
                def eval_spline(type_idx, a):
                    branches = [lambda a, s=s: s(a) for s in splines]
                    return jax.lax.switch(type_idx, branches, a)
                energies = vmap(eval_spline)(self.angle_types, angles)
            else:
                energies = splines[0](angles)

            return high_precision_sum(energies)
        return energy_fn

class HarmonicAngleEnergy:
    def __init__(self, angles, angle_0=1.5, epsilon=50, angle_types=None):
        self.angles = angles
        self.angle_0 = jnp.atleast_1d(jnp.asarray(angle_0))
        self.epsilon = jnp.atleast_1d(jnp.asarray(epsilon))
        self.angle_types = angle_types

    def get_energy_fn(self):
        def energy_fn(system, neighbors, **dynamic_kwargs):
            positions = system.R
            # Use cached displacement function with pre-computed cell inverse
            disp_fn = make_displacement_with_cached_inverse(system.cell)
            R_kj = vmap(disp_fn)(positions[self.angles[:,2]], positions[self.angles[:,1]])
            R_ij = vmap(disp_fn)(positions[self.angles[:,0]], positions[self.angles[:,1]])

            angles = vectorized_angle_fn(R_ij, R_kj)

            if self.angle_types is not None:
                # Per-type: index parameters by type
                angle_0s = self.angle_0[self.angle_types]
                epsilons = self.epsilon[self.angle_types]
            else:
                # Backward compatible: broadcast scalar
                angle_0s = self.angle_0[0]
                epsilons = self.epsilon[0]

            return high_precision_sum(harmonic_angle(angles, angle_0s, epsilons))
        return energy_fn

class TabulatedDihedralEnergy:
    def __init__(self, x_vals, y_vals, dihedrals, dihedral_types=None):
        self.x_vals = x_vals
        self.y_vals = jnp.atleast_2d(jnp.asarray(y_vals))  # (N_types, N_grid)
        self.dihedrals = dihedrals
        self.dihedral_types = dihedral_types
        self.n_types = self.y_vals.shape[0]

    def get_energy_fn(self):
        # Create spline for each type
        splines = [custom_interpolate.MonotonicInterpolate(self.x_vals, self.y_vals[i])
                   for i in range(self.n_types)]

        def energy_fn(system, neighbors, **dynamic_kwargs):
            positions = system.R
            # Use cached displacement function with pre-computed cell inverse
            disp_fn = make_displacement_with_cached_inverse(system.cell)
            R_cd = vmap(disp_fn)(positions[self.dihedrals[:,3]], positions[self.dihedrals[:,2]])
            R_bc = vmap(disp_fn)(positions[self.dihedrals[:,2]], positions[self.dihedrals[:,1]])
            R_ab = vmap(disp_fn)(positions[self.dihedrals[:,1]], positions[self.dihedrals[:,0]])

            dihedrals = vectorized_dihedral_fn(R_ab, R_bc, R_cd)

            if self.dihedral_types is not None:
                # Use lax.switch to select spline per dihedral - more efficient for many types
                def eval_spline(type_idx, d):
                    branches = [lambda d, s=s: s(d) for s in splines]
                    return jax.lax.switch(type_idx, branches, d)
                energies = vmap(eval_spline)(self.dihedral_types, dihedrals)
            else:
                energies = splines[0](dihedrals)

            return high_precision_sum(energies)
        return energy_fn

class HarmonicDihedralEnergy:
    def __init__(self, dihedrals, angle_0=1.5, epsilon=50, dihedral_types=None):
        self.dihedrals = dihedrals
        self.angle_0 = jnp.atleast_1d(jnp.asarray(angle_0))
        self.epsilon = jnp.atleast_1d(jnp.asarray(epsilon))
        self.dihedral_types = dihedral_types

    def get_energy_fn(self):
        def energy_fn(system, neighbors, **dynamic_kwargs):
            positions = system.R
            # Use cached displacement function with pre-computed cell inverse
            disp_fn = make_displacement_with_cached_inverse(system.cell)
            R_cd = vmap(disp_fn)(positions[self.dihedrals[:,3]], positions[self.dihedrals[:,2]])
            R_bc = vmap(disp_fn)(positions[self.dihedrals[:,2]], positions[self.dihedrals[:,1]])
            R_ab = vmap(disp_fn)(positions[self.dihedrals[:,1]], positions[self.dihedrals[:,0]])
            dihedrals = vectorized_dihedral_fn(R_ab, R_bc, R_cd)

            if self.dihedral_types is not None:
                # Per-type: index parameters by type
                angle_0s = self.angle_0[self.dihedral_types]
                epsilons = self.epsilon[self.dihedral_types]
            else:
                # Backward compatible: broadcast scalar
                angle_0s = self.angle_0[0]
                epsilons = self.epsilon[0]

            return high_precision_sum(harmonic_dihedral(dihedrals, angle_0s, epsilons))
        return energy_fn

class GenericRepulsionEnergy:
    def __init__(self, sigma=0.6, epsilon=1., exp=8,mask_topology=None,max_num_atoms=None,r_onset=0.9,r_cutoff=1.0):
        self.sigma = sigma
        self.epsilon = epsilon
        self.exp = exp
        self.mask_topology = mask_topology
        self.max_num_atoms = max_num_atoms
        self.r_onset = r_onset
        self.r_cutoff = r_cutoff
        # Pre-compute bonded pairs for O(n log m) lookup
        self.bonded_pairs = None
        if mask_topology is not None:
            self.bonded_pairs = build_bonded_pair_set(mask_topology)

    def get_energy_fn(self):
        if self.mask_topology is None:
            def energy_fn(system, neighbors, **dynamic_kwargs):
                positions = system.R

                # Convert JAX-MD neighbor list to edge list
                centers, others, N = _get_edge_list_from_jaxmd_neighbors(neighbors)

                # Clamp indices to valid range to prevent OOB access from padding entries
                safe_others = jnp.minimum(others, N - 1)

                # Use cached displacement function with pre-computed cell inverse
                disp_fn = make_displacement_with_cached_inverse(system.cell)
                edges = vmap(disp_fn)(positions[centers], positions[safe_others])
                dr = vmap(distance)(edges)

                _energy = multiplicative_isotropic_cutoff(generic_repulsion, self.r_onset, self.r_cutoff)(dr, self.sigma, self.epsilon, self.exp)
                mask = others < N  # Filter padding (JAX-MD uses N as padding value for unused neighbor slots)

                out = _energy * mask
                return high_precision_sum(out) * 0.5

            return energy_fn
        else:
            # Capture pre-computed bonded pairs in closure
            bonded_pairs = self.bonded_pairs

            def energy_fn(system, neighbors, **dynamic_kwargs):
                positions = system.R

                # Convert JAX-MD neighbor list to edge list
                centers, others, N = _get_edge_list_from_jaxmd_neighbors(neighbors)

                mask_centers, mask_others = mask_bonded_neighbors(
                    (centers, others), self.mask_topology, self.max_num_atoms,
                    bonded_pairs=bonded_pairs
                )
                mask = mask_centers < N

                # Clamp indices to valid range to prevent OOB access from padding entries
                safe_others = jnp.minimum(others, N - 1)

                # Use cached displacement function with pre-computed cell inverse
                disp_fn = make_displacement_with_cached_inverse(system.cell)
                edges = vmap(disp_fn)(positions[centers], positions[safe_others])
                dr = vmap(distance)(edges)

                _energy = multiplicative_isotropic_cutoff(generic_repulsion, self.r_onset, self.r_cutoff)(dr, self.sigma, self.epsilon, self.exp)
                out = _energy * mask
                return high_precision_sum(out) * 0.5

            return energy_fn


class TabulatedPairEnergy:
    def __init__(self, x_vals, y_vals,r_onset,r_cutoff,mask_topology=None,max_num_atoms=None):
        self.x_vals = x_vals
        self.y_vals = y_vals
        self.r_onset = r_onset
        self.r_cutoff = r_cutoff
        self.mask_topology = mask_topology
        self.max_num_atoms = max_num_atoms
        # Pre-compute bonded pairs for O(n log m) lookup
        self.bonded_pairs = None
        if mask_topology is not None:
            self.bonded_pairs = build_bonded_pair_set(mask_topology)

    def get_energy_fn(self):
        spline = custom_interpolate.MonotonicInterpolate(self.x_vals, self.y_vals)
        tabulated_partial = partial(tabulated, spline=spline)

        if self.mask_topology is None:
            def energy_fn(system, neighbors, **dynamic_kwargs):
                positions = system.R

                # Convert JAX-MD neighbor list to edge list
                centers, others, N = _get_edge_list_from_jaxmd_neighbors(neighbors)

                # Clamp indices to valid range to prevent OOB access from padding entries
                safe_others = jnp.minimum(others, N - 1)

                # Use cached displacement function with pre-computed cell inverse
                disp_fn = make_displacement_with_cached_inverse(system.cell)
                edges = vmap(disp_fn)(positions[centers], positions[safe_others])
                dr = vmap(distance)(edges)

                truncated_fn = multiplicative_isotropic_cutoff(tabulated_partial, self.r_onset, self.r_cutoff)
                mask = others < N  # Filter padding

                _energy = truncated_fn(dr)
                out = _energy * mask
                return high_precision_sum(out) * 0.5

            return energy_fn
        else:
            # Capture pre-computed bonded pairs in closure
            bonded_pairs = self.bonded_pairs

            def energy_fn(system, neighbors, **dynamic_kwargs):
                positions = system.R

                # Convert JAX-MD neighbor list to edge list
                centers, others, N = _get_edge_list_from_jaxmd_neighbors(neighbors)

                mask_centers, mask_others = mask_bonded_neighbors(
                    (centers, others), self.mask_topology, self.max_num_atoms,
                    bonded_pairs=bonded_pairs
                )
                mask = mask_centers < N

                # Clamp indices to valid range to prevent OOB access from padding entries
                safe_others = jnp.minimum(others, N - 1)

                # Use cached displacement function with pre-computed cell inverse
                disp_fn = make_displacement_with_cached_inverse(system.cell)
                edges = vmap(disp_fn)(positions[centers], positions[safe_others])
                dr = vmap(distance)(edges)

                truncated_fn = multiplicative_isotropic_cutoff(tabulated_partial, self.r_onset, self.r_cutoff)

                _energy = truncated_fn(dr)
                out = _energy * mask
                return high_precision_sum(out) * 0.5

            return energy_fn