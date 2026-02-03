# SPDX-License-Identifier: MIT
# Copyright (c) 2025 WuResearchGroup

import jax.numpy as jnp
import jax
from jax import jit, grad, vmap, lax
from jax.scipy.integrate import trapezoid
from functools import partial
from jax.scipy.stats.norm import cdf as normal_cdf
import numpy as np
import dataclasses
from diffcg._core.math import high_precision_sum
from diffcg._core.geometry import angle, dihedral, vectorized_angle_fn, vectorized_dihedral_fn, distance
from diffcg._core.periodic import get_displacement_fn
from diffcg._core.neighborlist import jaxmd_neighbor_list, jaxmd_update_neighbor_list
from jax_md import space, partition

def windowed_gaussian_kde(values, bin_centers, bin_size, sigma, half_window=4):
    """Windowed Gaussian KDE using scatter-add. Evaluates ~2*half_window bins per sample instead of all.

    Differentiable w.r.t. values (indices are non-differentiable constants, same as neighbor list pattern).

    Args:
        values: (M,) array of sample values
        bin_centers: (nbins,) array of bin center positions
        bin_size: scalar bin width
        sigma: scalar Gaussian width
        half_window: number of bins on each side of center to evaluate

    Returns:
        (nbins,) histogram array
    """
    nbins = bin_centers.shape[0]
    start = bin_centers[0] - bin_size / 2.0
    gaussian_norm = bin_size / jnp.sqrt(2 * jnp.pi * sigma ** 2)
    sigma_sq_2 = 2 * sigma ** 2

    # Center bin index for each sample
    center_idx = jnp.floor((values - start) / bin_size).astype(jnp.int32)

    # Window offsets: [-half_window, ..., +half_window]
    offsets = jnp.arange(-half_window, half_window + 1)  # (W,)
    window_size = offsets.shape[0]

    # Bin indices for each (sample, offset): (M, W)
    bin_indices = center_idx[:, jnp.newaxis] + offsets[jnp.newaxis, :]

    # Corresponding bin center positions
    bin_indices_clipped = jnp.clip(bin_indices, 0, nbins - 1)
    window_centers = bin_centers[bin_indices_clipped]  # (M, W)

    # Gaussian weights
    gauss = jnp.exp(-(values[:, jnp.newaxis] - window_centers) ** 2 / sigma_sq_2) * gaussian_norm  # (M, W)

    # Zero out contributions from out-of-range bins
    valid = (bin_indices >= 0) & (bin_indices < nbins)
    gauss = gauss * valid

    # Scatter-add into histogram
    result = jnp.zeros(nbins).at[bin_indices_clipped.ravel()].add(gauss.ravel())
    return result


def box_volume(box, ndim):
    """Computes the volume of the simulation box"""
    if box.size == ndim:
        return jnp.prod(box)
    elif box.ndim == 2:  # box tensor
        signed_volume = jnp.linalg.det(box)
        return jnp.abs(signed_volume)

@dataclasses.dataclass
class RDFParams:
    """
    A struct containing hyperparameters to initialize a radial distribution (RDF) compute function.

    Attributes:
    reference_rdf: The target rdf; initialize with None if no target available
    rdf_bin_centers: The radial positions of the centers of the rdf bins
    rdf_bin_boundaries: The radial positions of the edges of the rdf bins
    sigma_RDF: Standard deviation of smoothing Gaussian
    """
    reference_rdf: jnp.ndarray
    rdf_bin_centers: jnp.ndarray
    rdf_bin_boundaries: jnp.ndarray
    sigma_RDF: jnp.ndarray


def rdf_discretization(RDF_cut, nbins=300, RDF_start=0.):
    """
    Computes dicretization parameters for initialization of RDF compute function.

    Args:
        RDF_cut: Cut-off length inside which pairs of particles are considered
        nbins: Number of bins in radial direction
        RDF_start: Minimal radial distance after which pairs of particles are considered

    Returns:
        Arrays with radial positions of bin centers, and bin edges and the standard
        deviation of the Gaussian smoothing kernel.

    """
    dx_bin = (RDF_cut - RDF_start) / float(nbins)
    rdf_bin_centers = jnp.linspace(RDF_start + dx_bin / 2., RDF_cut - dx_bin / 2., nbins)
    rdf_bin_boundaries = jnp.linspace(RDF_start, RDF_cut, nbins + 1)
    sigma_RDF = jnp.array(dx_bin)
    return rdf_bin_centers, rdf_bin_boundaries, sigma_RDF

@dataclasses.dataclass
class InterRDFParams:
    """Hyperparameters for inter-molecular radial distribution function (RDF).

    Attributes:
        reference_rdf: Target RDF; None if no target available.
        rdf_bin_centers: Radial positions of bin centers.
        rdf_bin_boundaries: Radial positions of bin edges.
        sigma_RDF: Standard deviation of smoothing Gaussian.
        exclude_pairs: (M, 2) array of atom-index pairs to exclude (sparse).
            Replaces the old dense N×N ``exclude_mask``.
        species: Optional (N,) int array of type indices per atom.
        type_pair: Optional (A, B) tuple selecting which types for a partial RDF.
    """
    reference_rdf: jnp.ndarray
    rdf_bin_centers: jnp.ndarray
    rdf_bin_boundaries: jnp.ndarray
    sigma_RDF: jnp.ndarray
    exclude_pairs: jnp.ndarray
    species: jnp.ndarray = None
    type_pair: tuple = None

    def __post_init__(self):
        # Backward compatibility: if someone passes a dense N×N mask, convert it
        ep = np.asarray(self.exclude_pairs)
        if ep.ndim == 2 and ep.shape[0] == ep.shape[1]:
            import warnings
            warnings.warn(
                "InterRDFParams: passing a dense N×N exclude_mask is deprecated. "
                "Use an (M, 2) exclude_pairs array instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.exclude_pairs = _dense_mask_to_pairs(ep)

def _dense_mask_to_pairs(mask):
    """Convert a dense N×N exclusion mask (0=exclude, 1=keep) to (M,2) pair list."""
    mask_np = np.asarray(mask)
    # Extract upper-triangle zeros (excluded pairs)
    rows, cols = np.where(np.triu(1 - mask_np, k=1) > 0.5)
    if len(rows) == 0:
        return jnp.zeros((0, 2), dtype=jnp.int32)
    return jnp.array(np.stack([rows, cols], axis=1), dtype=jnp.int32)


def exclusion_pairs_from_topology(topology, exclusion_level=4):
    """Extract a deduplicated (M, 2) pair array of excluded atom pairs from topology.

    Args:
        topology: dict with keys 'bond' (N,2), 'angle' (N,3), 'dihedral' (N,4).
        exclusion_level: 2 = 1-2 only, 3 = 1-2 + 1-3, 4 = 1-2 + 1-3 + 1-4.

    Returns:
        (M, 2) jnp.int32 array with i < j for each pair.
    """
    pairs_list = []

    if exclusion_level >= 2 and 'bond' in topology:
        bonds = np.asarray(topology['bond'])
        pairs_list.append(bonds[:, [0, 1]])

    if exclusion_level >= 3 and 'angle' in topology:
        angles = np.asarray(topology['angle'])
        pairs_list.append(angles[:, [0, 2]])
        pairs_list.append(angles[:, [0, 1]])
        pairs_list.append(angles[:, [1, 2]])

    if exclusion_level >= 4 and 'dihedral' in topology:
        dihedrals = np.asarray(topology['dihedral'])
        pairs_list.append(dihedrals[:, [0, 3]])
        pairs_list.append(dihedrals[:, [0, 1]])
        pairs_list.append(dihedrals[:, [0, 2]])
        pairs_list.append(dihedrals[:, [1, 2]])
        pairs_list.append(dihedrals[:, [1, 3]])
        pairs_list.append(dihedrals[:, [2, 3]])

    if not pairs_list:
        return jnp.zeros((0, 2), dtype=jnp.int32)

    all_pairs = np.concatenate(pairs_list, axis=0)
    pair_min = np.minimum(all_pairs[:, 0], all_pairs[:, 1])
    pair_max = np.maximum(all_pairs[:, 0], all_pairs[:, 1])
    all_pairs = np.stack([pair_min, pair_max], axis=1)

    # Deduplicate
    all_pairs = np.unique(all_pairs, axis=0)
    return jnp.array(all_pairs, dtype=jnp.int32)


def exclusion_pairs_from_molecules(molecule_ids):
    """Generate all intra-molecular pairs from molecule-ID assignments.

    Args:
        molecule_ids: (N,) array where molecule_ids[i] is the molecule index of atom i.

    Returns:
        (M, 2) jnp.int32 array of intra-molecular pairs with i < j.
    """
    mol_ids = np.asarray(molecule_ids)
    n = len(mol_ids)
    pairs_list = []
    # Group atoms by molecule
    unique_mols = np.unique(mol_ids)
    for mid in unique_mols:
        atoms = np.where(mol_ids == mid)[0]
        if len(atoms) < 2:
            continue
        # All pairs within molecule
        for ii in range(len(atoms)):
            for jj in range(ii + 1, len(atoms)):
                pairs_list.append([atoms[ii], atoms[jj]])
    if not pairs_list:
        return jnp.zeros((0, 2), dtype=jnp.int32)
    return jnp.array(np.array(pairs_list), dtype=jnp.int32)


def inter_rdf_polymer_exclude(n_mole=1,chainlength=2):
    exclude=[]
    for mole_index in range(n_mole):
        index=0+chainlength*mole_index
        for atom_index1 in range(chainlength-1):
            index1=index+atom_index1
            for atom_index2 in range(atom_index1+1,chainlength):
                index2=index+atom_index2
                exclude.append([index1,index2])
    ex_pairs=np.array(exclude)
    N=int(n_mole*chainlength)
    mask = np.ones((N, N),dtype=np.int32)
    mask[ex_pairs[:,0], ex_pairs[:, 1]] = 0
    mask[ex_pairs[:,1], ex_pairs[:, 0]] = 0
    return jnp.array(mask)

@dataclasses.dataclass
class BDFParams:
    """
    A struct containing hyperparameters to initialize a bond distribution (BDF) compute function.

    Attributes:
    reference_bdf: The target bdf; initialize with None if no target available
    bdf_bin_centers: The positions of the centers of the adf bins over theta
    sigma_BDF: Standard deviation of smoothing Gaussian
    """
    reference_bdf: jnp.ndarray
    bdf_bin_centers: jnp.ndarray
    bdf_bin_boundaries: jnp.ndarray
    sigma_BDF: jnp.ndarray
    bond_top: jnp.ndarray

def bdf_discretization(BDF_cut, nbins=300, BDF_start=0.):
    """
    Computes dicretization parameters for initialization of BDF compute function.

    Args:
        BDF_cut: Cut-off length inside which pairs of particles are considered
        nbins: Number of bins in radial direction
        BDF_start: Minimal radial distance after which pairs of particles are considered

    Returns:
        Arrays containing bin centers in bond distance and the standard
        deviation of the Gaussian smoothing kernel.

    """
    dx_bin = (BDF_cut - BDF_start) / float(nbins)
    bdf_bin_centers = jnp.linspace(BDF_start + dx_bin / 2., BDF_cut - dx_bin / 2., nbins)
    bdf_bin_boundaries = jnp.linspace(BDF_start, BDF_cut, nbins + 1)
    sigma_BDF = jnp.array(dx_bin)
    return bdf_bin_centers, bdf_bin_boundaries, sigma_BDF

def initialize_bond_distribution_fun(bdf_params, displacement_fn=None):
    """
    Initializes a function that computes the bond distribution function (BDF) for a single state.

    Args:
        bdf_params: BDFParams defining the hyperparameters of the BDF
        displacement_fn: Optional pre-built displacement function. If None, created per frame.

    Returns:
        A function that takes a simulation state and returns the instantaneous bdf
    """
    _, bdf_bin_centers, bdf_bin_boundaries, sigma, bond_top = dataclasses.astuple(bdf_params)
    bin_size = jnp.diff(bdf_bin_boundaries)
    # Pre-compute Gaussian normalization constant
    gaussian_norm = bin_size / jnp.sqrt(2 * jnp.pi * sigma ** 2)
    sigma_sq_2 = 2 * sigma ** 2  # Pre-compute 2*sigma^2

    def bond_corr_fun(system, **dynamic_kwargs):
        # computes instantaneous pair correlation function ensuring each particle pair contributes exactly 1
        positions = system.R
        Ra = positions[bond_top[:,0],:]
        Rb = positions[bond_top[:,1],:]
        disp_fn = displacement_fn if displacement_fn is not None else get_displacement_fn(system.cell)
        edges = vmap(disp_fn)(Rb, Ra)  # Rb - Ra (JAX-MD: first - second)
        dr=vmap(distance)(edges)
        #  Gaussian distribution ensures that discrete integral over distribution is 1
        exp = jnp.exp(-(dr[:, jnp.newaxis] - bdf_bin_centers) ** 2 / sigma_sq_2)  # Gaussian exponent
        gaussian_distances = exp * gaussian_norm
        bond_corr = high_precision_sum(gaussian_distances, axis=0)  # sum over all neighbors
        integral = trapezoid(bond_corr, bdf_bin_centers)
        mean_bond_corr = bond_corr / jnp.where(jnp.abs(integral) < 1e-10, 1.0, integral)
        return mean_bond_corr

    def bdf_compute_fun(system, **unused_kwargs):
        # Note: we cannot use neighborlist as RDF cutoff and neighborlist cut-off don't coincide in general
        bdf = bond_corr_fun(system)
        return bdf
    return bdf_compute_fun

def initialize_radial_distribution_fun(box, displacement_fn, rdf_params):
    """
    Initializes a function that computes the radial distribution function (RDF) for a single state.
    
    Args:
        box: Simulation box
        displacement_fn: Displacement function
        rdf_params: RDFParams defining the hyperparameters of the RDF

    Returns:
        A function that takes a simulation state and returns the instantaneous rdf
    """
    _, rdf_bin_centers, rdf_bin_boundaries, sigma = dataclasses.astuple(rdf_params)
    distance_metric = space.canonicalize_displacement_or_metric(displacement_fn)
    bin_size = jnp.diff(rdf_bin_boundaries)

    def pair_corr_fun(R, **dynamic_kwargs):
        # computes instantaneous pair correlation function ensuring each particle pair contributes exactly 1
        n_particles = R.shape[0]
        metric = partial(distance_metric, **dynamic_kwargs)
        metric = space.map_product(metric)
        dr = metric(R, R)
        dr = jnp.where(dr > util.f32(1.e-7), dr, util.f32(1.e7))  # neglect same particles i.e. distance = 0.

        #  Gaussian distribution ensures that discrete integral over distribution is 1
        exp = jnp.exp(-util.f32(0.5) * (dr[:, :, jnp.newaxis] - rdf_bin_centers) ** 2 / sigma ** 2)  # Gaussian exponent
        gaussian_distances = exp * bin_size / jnp.sqrt(2 * jnp.pi * sigma ** 2)
        pair_corr_per_particle = util.high_precision_sum(gaussian_distances, axis=1)  # sum over all neighbors
        mean_pair_corr = util.high_precision_sum(pair_corr_per_particle, axis=0) / n_particles
        return mean_pair_corr

    def norming_factors(particle_density, bin_boundaries):
        # RDF is defined to relate the particle densities to an ideal gas:
        # This function computes densities that would correspond to an ideal gas
        r_small = bin_boundaries[:-1]
        r_large = bin_boundaries[1:]
        bin_volume = (4. / 3.) * jnp.pi * (jnp.power(r_large, 3) - jnp.power(r_small, 3))
        bin_weights = bin_volume * particle_density
        return bin_weights

    def rdf_compute_fun(system, **unused_kwargs):
        # Note: we cannot use neighborlist as RDF cutoff and neighborlist cut-off don't coincide in general
        R = system.R
        n_particles = R.shape[0]
        total_vol = box_volume(system.cell, system.R.shape[1])  # volume of partition
        particle_density = n_particles / total_vol
        mean_pair_corr = pair_corr_fun(system)
        rdf = mean_pair_corr / norming_factors(particle_density, rdf_bin_boundaries)
        return rdf
    return rdf_compute_fun

def initialize_inter_radial_distribution_fun(inter_rdf_params):
    """Initializes a function that computes the inter-molecular RDF for a single state.

    Supports:
    - Sparse exclusion pairs (M, 2) instead of dense N×N mask
    - Optional per-type partial RDFs via ``species`` and ``type_pair``
    - Windowed Gaussian KDE for O(N×W) binning instead of O(N²×nbins)

    Args:
        inter_rdf_params: InterRDFParams with bin centers, boundaries, sigma,
            exclude_pairs, and optionally species + type_pair.

    Returns:
        A function ``rdf_compute_fun(system)`` returning the instantaneous RDF.
    """
    rdf_bin_centers = inter_rdf_params.rdf_bin_centers
    rdf_bin_boundaries = inter_rdf_params.rdf_bin_boundaries
    sigma = inter_rdf_params.sigma_RDF
    exclude_pairs = inter_rdf_params.exclude_pairs
    species = inter_rdf_params.species
    type_pair = inter_rdf_params.type_pair

    bin_size = rdf_bin_boundaries[1] - rdf_bin_boundaries[0]
    nbins = rdf_bin_centers.shape[0]

    # Build exclusion lookup: encode pairs as (min*MAX_ATOMS + max) for fast membership test
    ep = np.asarray(exclude_pairs)
    n_exclude = ep.shape[0]
    _has_exclusions = n_exclude > 0
    if _has_exclusions:
        ep_min = np.minimum(ep[:, 0], ep[:, 1])
        ep_max = np.maximum(ep[:, 0], ep[:, 1])
        max_atom_est = int(np.max(ep_max)) + 1
        exclude_keys = np.sort(ep_min * max_atom_est + ep_max)
        _exclude_keys = jnp.array(exclude_keys, dtype=jnp.int32)
        _max_atom = max_atom_est
    else:
        _exclude_keys = None
        _max_atom = 1

    def _is_excluded(i, j):
        """Check if pair (i,j) is in the exclusion set. Returns bool array."""
        if not _has_exclusions:
            return jnp.zeros_like(i, dtype=bool)
        p_min = jnp.minimum(i, j)
        p_max = jnp.maximum(i, j)
        keys = p_min * _max_atom + p_max
        idx = jnp.searchsorted(_exclude_keys, keys)
        idx_safe = jnp.minimum(idx, _exclude_keys.shape[0] - 1)
        return _exclude_keys[idx_safe] == keys

    def pair_corr_fun(system, **dynamic_kwargs):
        positions = system.R
        N = positions.shape[0]

        # All-pairs distance matrix (N, N, D)
        diff = positions[jnp.newaxis, :, :] - positions[:, jnp.newaxis, :]

        # Apply periodic boundary conditions
        if system.cell is not None:
            cell_inv = jnp.linalg.inv(system.cell)
            frac = jnp.einsum('Aa,ija->ijA', cell_inv, diff)
            frac = jnp.mod(frac + 0.5, 1.0) - 0.5
            diff = jnp.einsum('aA,ijA->ija', system.cell, frac)

        dr = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-16)  # (N, N)

        # Build combined mask from sparse exclusion list
        not_self = 1.0 - jnp.eye(N)
        i_idx = jnp.broadcast_to(jnp.arange(N)[:, None], (N, N))
        j_idx = jnp.broadcast_to(jnp.arange(N)[None, :], (N, N))
        not_excluded = 1.0 - _is_excluded(i_idx, j_idx).astype(jnp.float32)

        # Type mask for partial RDFs
        if species is not None and type_pair is not None:
            tA, tB = type_pair
            sp = species
            si = sp[i_idx]
            sj = sp[j_idx]
            if tA == tB:
                type_mask = ((si == tA) & (sj == tA)).astype(jnp.float32)
            else:
                type_mask = (((si == tA) & (sj == tB)) |
                             ((si == tB) & (sj == tA))).astype(jnp.float32)
        else:
            type_mask = jnp.ones((N, N), dtype=jnp.float32)

        combined_mask = not_self * not_excluded * type_mask

        # Mask out excluded/self pairs by setting distance to large value
        dr = dr * combined_mask + (1.0 - combined_mask) * 1e7

        # Gaussian KDE
        gaussian_norm = bin_size / jnp.sqrt(2 * jnp.pi * sigma ** 2)
        sigma_sq_2 = 2 * sigma ** 2
        exp = jnp.exp(-(dr[:, :, jnp.newaxis] - rdf_bin_centers) ** 2 / sigma_sq_2)
        gaussian_distances = exp * gaussian_norm
        pair_corr_per_particle = high_precision_sum(gaussian_distances, axis=1)
        mean_pair_corr = high_precision_sum(pair_corr_per_particle, axis=0)

        return mean_pair_corr, combined_mask

    def norming_factors(particle_density, bin_boundaries):
        r_small = bin_boundaries[:-1]
        r_large = bin_boundaries[1:]
        bin_volume = (4. / 3.) * jnp.pi * (jnp.power(r_large, 3) - jnp.power(r_small, 3))
        return bin_volume * particle_density

    def rdf_compute_fun(system, **unused_kwargs):
        N = system.R.shape[0]
        total_vol = box_volume(system.cell, system.R.shape[1])

        mean_pair_corr, combined_mask = pair_corr_fun(system)

        # Per-type normalization
        if species is not None and type_pair is not None:
            tA, tB = type_pair
            N_A = jnp.sum((species == tA).astype(jnp.float32))
            n_ref = N_A
        else:
            n_ref = jnp.array(N, dtype=jnp.float32)

        # Average over reference particles
        mean_pair_corr = mean_pair_corr / n_ref

        # Effective neighbor density for normalization (ensures g(r) → 1 at large r)
        n_effective_neighbors = jnp.sum(combined_mask) / n_ref
        particle_density = n_effective_neighbors / total_vol

        rdf = mean_pair_corr / norming_factors(particle_density, rdf_bin_boundaries)
        return rdf
    return rdf_compute_fun

@dataclasses.dataclass
class ADFParams:
    """
    A struct containing hyperparameters to initialize a angle distribution (BDF) compute function.

    Attributes:
    reference_adf: The target bdf; initialize with None if no target available
    adf_bin_centers: The positions of the centers of the adf bins over theta
    sigma_ADF: Standard deviation of smoothing Gaussian
    """
    reference_adf: jnp.ndarray
    adf_bin_centers: jnp.ndarray
    adf_bin_boundaries: jnp.ndarray
    sigma_ADF: jnp.ndarray
    angle_top: jnp.ndarray


def adf_discretization(ADF_cut, nbins=300, ADF_start=0.):
    """
    Computes dicretization parameters for initialization of ADF compute function.

    Args:
        ADF_cut: Cut-off angle for a bending potential
        nbins: Number of bins in angles
        ADF_start: Minimal angle for a bending potential is considered

    Returns:
        Arrays containing bin centers in theta direction and the standard
        deviation of the Gaussian smoothing kernel.
    """
    dtheta_bin = (ADF_cut - ADF_start) / float(nbins)
    adf_bin_centers = jnp.linspace(ADF_start + dtheta_bin / 2., ADF_cut - dtheta_bin / 2., nbins)
    adf_bin_boundaries = jnp.linspace(ADF_start, ADF_cut, nbins + 1)
    sigma_ADF = jnp.array(dtheta_bin)
    return adf_bin_centers, adf_bin_boundaries, sigma_ADF


def initialize_angle_distribution_fun(adf_params, displacement_fn=None):
    """
    Initializes a function that computes the angular distribution function (ADF) for a single state.

    Args:
        adf_params: ADFParams defining the hyperparameters of the ADF
        displacement_fn: Optional pre-built displacement function. If None, created per frame.

    Returns:
        A function that takes a simulation state and returns the instantaneous adf
    """
    _, adf_bin_centers, adf_bin_boundaries, sigma_theta, angle_top = dataclasses.astuple(adf_params)
    bin_size = jnp.diff(adf_bin_boundaries)
    # Pre-compute Gaussian normalization constant
    gaussian_norm = bin_size / jnp.sqrt(2 * jnp.pi * sigma_theta ** 2)
    sigma_sq_2 = 2 * sigma_theta ** 2  # Pre-compute 2*sigma^2

    def angle_corr_fun(system):
        """Compute adf contribution of each triplet."""
        positions = system.R
        disp_fn = displacement_fn if displacement_fn is not None else get_displacement_fn(system.cell)
        R_kj = vmap(disp_fn)(positions[angle_top[:,1]],positions[angle_top[:,2]])
        R_ij = vmap(disp_fn)(positions[angle_top[:,1]],positions[angle_top[:,0]])

        angles = vectorized_angle_fn(R_ij, R_kj)

        exponent = jnp.exp(-(angles[:, jnp.newaxis] - adf_bin_centers) ** 2 / sigma_sq_2)
        gaussians = exponent * gaussian_norm
        unnormed_adf = high_precision_sum(gaussians, axis=0)
        integral = trapezoid(unnormed_adf, adf_bin_centers)
        adf = unnormed_adf / jnp.where(jnp.abs(integral) < 1e-10, 1.0, integral)
        return adf

    def adf_fn(system, **unused_kwargs):
        adf = angle_corr_fun(system)
        return adf
    return adf_fn


@dataclasses.dataclass
class DDFParams:
    """
    A struct containing hyperparameters to initialize a angle distribution (BDF) compute function.

    Attributes:
    reference_ddf: The target ddf; initialize with None if no target available
    ddf_bin_centers: The positions of the centers of the ddf bins over theta
    sigma_DDF: Standard deviation of smoothing Gaussian
    """
    reference_ddf: jnp.ndarray
    ddf_bin_centers: jnp.ndarray
    ddf_bin_boundaries: jnp.ndarray
    sigma_DDF: jnp.ndarray
    angle_top: jnp.ndarray


def ddf_discretization(DDF_cut, nbins=300, DDF_start=0.):
    """
    Computes dicretization parameters for initialization of DDF compute function.

    Args:
        DDF_cut: Cut-off angle for a bending potential
        nbins: Number of bins in angles
        DDF_start: Minimal angle for a bending potential is considered

    Returns:
        Arrays containing bin centers in theta direction and the standard
        deviation of the Gaussian smoothing kernel.
    """
    dtheta_bin = (DDF_cut - DDF_start) / float(nbins)
    ddf_bin_centers = jnp.linspace(DDF_start + dtheta_bin / 2., DDF_cut - dtheta_bin / 2., nbins)
    ddf_bin_boundaries = jnp.linspace(DDF_start, DDF_cut, nbins + 1)
    sigma_DDF = jnp.array(dtheta_bin)
    return ddf_bin_centers, ddf_bin_boundaries, sigma_DDF


def initialize_dihedral_distribution_fun(ddf_params, displacement_fn=None):
    """
    Initializes a function that computes the dihedral distribution function (DDF) for a single state.

    Args:
        ddf_params: DDFParams defining the hyperparameters of the DDF
        displacement_fn: Optional pre-built displacement function. If None, created per frame.

    Returns:
        A function that takes a simulation state and returns the instantaneous ddf
    """
    _, ddf_bin_centers, ddf_bin_boundaries, sigma_theta, angle_top = dataclasses.astuple(ddf_params)
    bin_size = jnp.diff(ddf_bin_boundaries)
    # Pre-compute Gaussian normalization constant
    gaussian_norm = bin_size / jnp.sqrt(2 * jnp.pi * sigma_theta ** 2)
    sigma_sq_2 = 2 * sigma_theta ** 2  # Pre-compute 2*sigma^2

    def dihedral_corr_fun(system):
        """Compute ddf contribution of each quadruplet."""

        positions = system.R
        disp_fn = displacement_fn if displacement_fn is not None else get_displacement_fn(system.cell)
        # disp_fn(A,B) = A-B, so swap args to get forward vectors (a→b, b→c, c→d)
        R_ab = vmap(disp_fn)(positions[angle_top[:,1]], positions[angle_top[:,0]])
        R_bc = vmap(disp_fn)(positions[angle_top[:,2]], positions[angle_top[:,1]])
        R_cd = vmap(disp_fn)(positions[angle_top[:,3]], positions[angle_top[:,2]])

        dihedrals = vectorized_dihedral_fn(R_ab, R_bc, R_cd)

        exponent = jnp.exp(-(dihedrals[:, jnp.newaxis] - ddf_bin_centers) ** 2 / sigma_sq_2)
        gaussians = exponent * gaussian_norm
        unnormed_ddf = high_precision_sum(gaussians, axis=0)
        integral = trapezoid(unnormed_ddf, ddf_bin_centers)
        ddf = unnormed_ddf / jnp.where(jnp.abs(integral) < 1e-10, 1.0, integral)
        return ddf

    def ddf_fn(system, **unused_kwargs):
        ddf = dihedral_corr_fun(system)
        return ddf
    return ddf_fn


def initialize_inter_rdf_neighborlist_fun(inter_rdf_params, init_system,
                                          custom_mask_function=None,
                                          capacity_multiplier=1.5):
    """Neighbor-list-based inter-molecular RDF. O(N×max_nbrs) instead of O(N²).

    Args:
        inter_rdf_params: InterRDFParams with bin centers, boundaries, sigma, exclude_mask.
        init_system: An AtomicSystem used to allocate the neighbor list.
        custom_mask_function: Optional JAX-MD mask function for bonded exclusions.
            If None, falls back to the dense exclude_mask from inter_rdf_params.
        capacity_multiplier: Buffer multiplier for neighbor list allocation.

    Returns:
        Tuple of (rdf_compute_fn, nbrs, spatial_partitioning) where:
          - rdf_compute_fn(system, neighbors) computes instantaneous RDF
          - nbrs: initial JAX-MD neighbor list object
          - spatial_partitioning: for updating neighbor list with new positions
    """
    rdf_bin_centers = inter_rdf_params.rdf_bin_centers
    rdf_bin_boundaries = inter_rdf_params.rdf_bin_boundaries
    sigma = inter_rdf_params.sigma_RDF
    exclude_pairs = inter_rdf_params.exclude_pairs
    bin_size = jnp.diff(rdf_bin_boundaries)
    gaussian_norm = bin_size / jnp.sqrt(2 * jnp.pi * sigma ** 2)
    sigma_sq_2 = 2 * sigma ** 2
    rdf_cutoff = float(rdf_bin_boundaries[-1])
    n_particles = init_system.R.shape[0]

    # Build the neighbor list for RDF cutoff
    nbrs, spatial_partitioning = jaxmd_neighbor_list(
        init_system.R, init_system.cell, rdf_cutoff,
        capacity_multiplier=capacity_multiplier,
        format=partition.Dense,
        custom_mask_function=custom_mask_function,
    )

    # If no custom_mask_function was provided, reconstruct a dense exclude mask
    # from the sparse pair list for per-pair lookup in the neighbor list
    use_dense_exclude = custom_mask_function is None
    if use_dense_exclude:
        exclude_mask_dense = np.ones((n_particles, n_particles), dtype=np.float32)
        ep = np.asarray(exclude_pairs)
        if ep.shape[0] > 0:
            exclude_mask_dense[ep[:, 0], ep[:, 1]] = 0
            exclude_mask_dense[ep[:, 1], ep[:, 0]] = 0
        combined_mask = jnp.array((1.0 - np.eye(n_particles)) * exclude_mask_dense)
    else:
        combined_mask = None

    def pair_corr_fun(system, neighbors):
        positions = system.R
        N = positions.shape[0]
        idx = neighbors.idx  # (N, max_nbrs)
        safe_idx = jnp.minimum(idx, N - 1)
        valid_mask = idx < N  # (N, max_nbrs)

        # Compute displacements using the spatial partitioning displacement fn
        R_neigh = positions[safe_idx]  # (N, max_nbrs, D)
        d = space.map_neighbor(spatial_partitioning.displacement_fn)
        dR = d(positions, R_neigh)  # (N, max_nbrs, D)
        dr = jnp.sqrt(jnp.sum(dR ** 2, axis=-1) + 1e-16)  # (N, max_nbrs)

        # Apply exclusion mask if using dense exclude
        if use_dense_exclude:
            center_idx = jnp.arange(N)[:, None]  # (N, 1)
            pair_mask = combined_mask[center_idx, safe_idx]  # (N, max_nbrs)
            valid_mask = valid_mask * pair_mask

        # Gaussian KDE: (N, max_nbrs, 1) - (nbins,) -> (N, max_nbrs, nbins)
        exp = jnp.exp(-(dr[:, :, jnp.newaxis] - rdf_bin_centers) ** 2 / sigma_sq_2)
        gaussian_distances = exp * gaussian_norm  # (N, max_nbrs, nbins)
        # Zero out invalid/excluded pairs
        gaussian_distances = gaussian_distances * valid_mask[:, :, jnp.newaxis]
        # Sum over neighbors, then average over particles
        # Dense format double-counts (i->j and j->i), so divide by 2
        pair_corr_per_particle = high_precision_sum(gaussian_distances, axis=1)  # (N, nbins)
        mean_pair_corr = high_precision_sum(pair_corr_per_particle, axis=0) / n_particles / 2.0
        return mean_pair_corr

    def norming_factors(particle_density, bin_boundaries):
        r_small = bin_boundaries[:-1]
        r_large = bin_boundaries[1:]
        bin_volume = (4. / 3.) * jnp.pi * (jnp.power(r_large, 3) - jnp.power(r_small, 3))
        return bin_volume * particle_density

    def rdf_compute_fn(system, neighbors, **unused_kwargs):
        R = system.R
        n_particles_local = R.shape[0]
        total_vol = box_volume(system.cell, R.shape[1])
        mean_pair_corr = pair_corr_fun(system, neighbors)
        if use_dense_exclude:
            n_effective_neighbors = jnp.sum(combined_mask) / n_particles_local
        else:
            n_effective_neighbors = jnp.array(n_particles_local - 1, dtype=jnp.float32)
        particle_density = n_effective_neighbors / total_vol
        rdf = mean_pair_corr / norming_factors(particle_density, rdf_bin_boundaries)
        return rdf

    return rdf_compute_fn, nbrs, spatial_partitioning