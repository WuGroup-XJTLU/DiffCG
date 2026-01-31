# SPDX-License-Identifier: MIT
# Copyright (c) 2025 WuResearchGroup

import jax.numpy as jnp
from jax import jit, grad, vmap, lax
from jax.scipy.integrate import trapezoid
from functools import partial
from jax.scipy.stats.norm import cdf as normal_cdf
import numpy as np
import dataclasses
from diffcg._core.math import high_precision_sum
from diffcg._core.geometry import angle, dihedral, vectorized_angle_fn, vectorized_dihedral_fn, distance
from diffcg._core.periodic import make_displacement_with_cached_inverse

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
    exclude_mask: jnp.ndarray

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

def initialize_bond_distribution_fun(bdf_params):
    """
    Initializes a function that computes the radial distribution function (RDF) for a single state.
    
    Args:
        box: Simulation box
        displacement_fn: Displacement function
        bdf_params: RDFParams defining the hyperparameters of the BDF

    Returns:
        A function that takes a simulation state and returns the instantaneous rdf
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
        # Use cached displacement function with pre-computed cell inverse
        disp_fn = make_displacement_with_cached_inverse(system.cell)
        edges = vmap(disp_fn)(Ra, Rb)
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
    """
    Initializes a function that computes the radial distribution function (RDF) for a single state.
    
    Args:
        box: Simulation box
        displacement_fn: Displacement function
        inter_rdf_params: RDFParams defining the hyperparameters of the RDF

    Returns:
        A function that takes a simulation state and returns the instantaneous rdf
    """
    _, rdf_bin_centers, rdf_bin_boundaries, sigma, exclude_mask = dataclasses.astuple(inter_rdf_params)
    bin_size = jnp.diff(rdf_bin_boundaries)
    # Pre-compute Gaussian normalization constant
    gaussian_norm = bin_size / jnp.sqrt(2 * jnp.pi * sigma ** 2)
    sigma_sq_2 = 2 * sigma ** 2  # Pre-compute 2*sigma^2
    # Pre-compute combined exclusion mask (self-pairs + user-specified exclusions)
    n_particles = exclude_mask.shape[0]
    combined_mask = (1.0 - jnp.eye(n_particles)) * exclude_mask

    def pair_corr_fun(system, **dynamic_kwargs):
        # computes instantaneous pair correlation function ensuring each particle pair contributes exactly 1
        positions = system.R

        # Vectorized pairwise distance computation using broadcasting
        # diff[i,j] = positions[j] - positions[i], shape (N, N, D)
        diff = positions[jnp.newaxis, :, :] - positions[:, jnp.newaxis, :]

        # Apply periodic boundary conditions if cell exists
        if system.cell is not None:
            cell_inv = jnp.linalg.inv(system.cell)
            # Convert to fractional coordinates
            frac = jnp.einsum('Aa,ija->ijA', cell_inv, diff)
            # Apply minimum image convention
            frac = jnp.mod(frac + 0.5, 1.0) - 0.5
            # Convert back to Cartesian
            diff = jnp.einsum('aA,ijA->ija', system.cell, frac)

        # Compute distances using vectorized norm
        dr = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-16)  # (N, N) with small eps for gradient safety

        # Apply pre-computed exclusion mask
        dr = dr * combined_mask + (1.0 - combined_mask) * 1e7  # set excluded/self pairs to large value

        # Gaussian distribution ensures that discrete integral over distribution is 1
        exp = jnp.exp(-(dr[:, :, jnp.newaxis] - rdf_bin_centers) ** 2 / sigma_sq_2)  # Gaussian exponent
        gaussian_distances = exp * gaussian_norm
        pair_corr_per_particle = high_precision_sum(gaussian_distances, axis=1)  # sum over all neighbors
        mean_pair_corr = high_precision_sum(pair_corr_per_particle, axis=0) / n_particles
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
        mean_pair_corr = pair_corr_fun(system)
        # Use effective neighbor count to account for excluded pairs,
        # ensuring g(r) → 1.0 at large r
        n_effective_neighbors = jnp.sum(combined_mask) / n_particles
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


def initialize_angle_distribution_fun( adf_params):
    """
    Initializes a function that computes the angular distribution function (ADF) for a single state.

    Angles are smoothed in radial direction via a Gaussian kernel (compare RDF function). In radial
    direction, triplets are weighted according to a Gaussian cumulative distribution function, such that
    triplets with both radii inside the cut-off band are weighted approximately 1 and the weights of
    triplets towards the band edges are soomthly reduced to 0.
    For computational speed-up and reduced memory needs, R_init and nbrs_init can be provided
    to estmate the maximum number of triplets - similarly to the maximum capacity of neighbors
    in the neighbor list. Caution: currrently the user does not receive information if overflow occured.
    This function assumes that r_outer is smaller than the neighborlist cut-off. If this is not the case,
    a function computing all pairwise distances is necessary.

    Args:
        displacement_fn: Displacement functions
        adf_params: ADFParams defining the hyperparameters of the RDF
        smoothing_dr: Standard deviation of Gaussian smoothing in radial direction
        R_init: Initial position to estimate maximum number of triplets
        nbrs_init: Initial neighborlist to estimate maximum number of triplets
        max_weights_multiplier: Multiplier for estimate of number of triplets

    Returns:
        A function that takes a simulation state with neighborlist and returns the instantaneous adf
    """
    _, adf_bin_centers, adf_bin_boundaries, sigma_theta, angle_top = dataclasses.astuple(adf_params)
    bin_size = jnp.diff(adf_bin_boundaries)
    # Pre-compute Gaussian normalization constant
    gaussian_norm = bin_size / jnp.sqrt(2 * jnp.pi * sigma_theta ** 2)
    sigma_sq_2 = 2 * sigma_theta ** 2  # Pre-compute 2*sigma^2

    def angle_corr_fun(system):
        """Compute adf contribution of each triplet."""
        positions = system.R
        # Use cached displacement function with pre-computed cell inverse
        disp_fn = make_displacement_with_cached_inverse(system.cell)
        R_kj = vmap(disp_fn)(positions[angle_top[:,2]],positions[angle_top[:,1]])
        R_ij = vmap(disp_fn)(positions[angle_top[:,0]],positions[angle_top[:,1]])

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


def initialize_dihedral_distribution_fun(ddf_params):
    """
    Initializes a function that computes the angular distribution function (DDF) for a single state.

    Angles are smoothed in radial direction via a Gaussian kernel (compare RDF function). In radial
    direction, triplets are weighted according to a Gaussian cumulative distribution function, such that
    triplets with both radii inside the cut-off band are weighted approximately 1 and the weights of
    triplets towards the band edges are soomthly reduced to 0.
    For computational speed-up and reduced memory needs, R_init and nbrs_init can be provided
    to estmate the maximum number of triplets - similarly to the maximum capacity of neighbors
    in the neighbor list. Caution: currrently the user does not receive information if overflow occured.
    This function assumes that r_outer is smaller than the neighborlist cut-off. If this is not the case,
    a function computing all pairwise distances is necessary.

    Args:
        displacement_fn: Displacement functions
        ddf_params: DDFParams defining the hyperparameters of the RDF
        smoothing_dr: Standard deviation of Gaussian smoothing in radial direction
        R_init: Initial position to estimate maximum number of triplets
        nbrs_init: Initial neighborlist to estimate maximum number of triplets
        max_weights_multiplier: Multiplier for estimate of number of triplets

    Returns:
        A function that takes a simulation state with neighborlist and returns the instantaneous adf
    """
    _, ddf_bin_centers, ddf_bin_boundaries, sigma_theta, angle_top = dataclasses.astuple(ddf_params)
    bin_size = jnp.diff(ddf_bin_boundaries)
    # Pre-compute Gaussian normalization constant
    gaussian_norm = bin_size / jnp.sqrt(2 * jnp.pi * sigma_theta ** 2)
    sigma_sq_2 = 2 * sigma_theta ** 2  # Pre-compute 2*sigma^2

    def dihedral_corr_fun(system):
        """Compute adf contribution of each triplet."""

        positions = system.R
        # Use cached displacement function with pre-computed cell inverse
        disp_fn = make_displacement_with_cached_inverse(system.cell)
        R_cd = vmap(disp_fn)(positions[angle_top[:,3]],positions[angle_top[:,2]])
        R_bc = vmap(disp_fn)(positions[angle_top[:,2]],positions[angle_top[:,1]])
        R_ab = vmap(disp_fn)(positions[angle_top[:,1]],positions[angle_top[:,0]])

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