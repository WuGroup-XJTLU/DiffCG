import jax.numpy as jnp
from jax import jit, grad, vmap, lax, jacrev, jacfwd, ops
from functools import partial
from jax.scipy.stats.norm import cdf as normal_cdf
import numpy as np
import dataclasses
from diffcg.util.math import high_precision_sum
from diffcg.common.geometry import angle, dihedral, vectorized_angle_fn, vectorized_dihedral_fn, distance
from diffcg.common.periodic import displacement

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

    def bond_corr_fun(system, **dynamic_kwargs):
        # computes instantaneous pair correlation function ensuring each particle pair contributes exactly 1
        positions = system.R
        Ra = positions[bond_top[:,0],:]
        Rb = positions[bond_top[:,1],:]
        edges = vmap(partial(displacement, system.cell))(Ra, Rb)
        dr=vmap(distance)(edges)
        #  Gaussian distribution ensures that discrete integral over distribution is 1
        exp = jnp.exp(-0.5 * (dr[:, jnp.newaxis] - bdf_bin_centers) ** 2 / sigma ** 2)  # Gaussian exponent
        gaussian_distances = exp * bin_size / jnp.sqrt(2 * jnp.pi * sigma ** 2)
        bond_corr = high_precision_sum(gaussian_distances, axis=0)  # sum over all neighbors
        mean_bond_corr = bond_corr / jnp.trapz(bond_corr, bdf_bin_centers)
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

    def pair_corr_fun(system, **dynamic_kwargs):
        # computes instantaneous pair correlation function ensuring each particle pair contributes exactly 1
        positions = system.R
        n_particles = positions.shape[0]

        # Compute all pairwise displacement vectors
        disp_fn = partial(displacement, system.cell)
        # (N, N, D)
        disp_matrix = vmap(lambda x: vmap(lambda y: disp_fn(x, y))(positions))(positions)
        # Compute all pairwise distances (N, N)
        dr = vmap(lambda row: vmap(distance)(row))(disp_matrix)

        # Exclude self-pairs and pairs to be masked
        mask = (1.0 - jnp.eye(n_particles)) * exclude_mask  # 0 for self, 1 for valid pairs, 0 for excluded
        dr = dr * mask + (1.0 - mask) * 1e7  # set excluded/self pairs to large value

        # Gaussian distribution ensures that discrete integral over distribution is 1
        exp = jnp.exp(-0.5 * (dr[:, :, jnp.newaxis] - rdf_bin_centers) ** 2 / sigma ** 2)  # Gaussian exponent
        gaussian_distances = exp * bin_size / jnp.sqrt(2 * jnp.pi * sigma ** 2)
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
        particle_density = n_particles / total_vol
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

    def angle_corr_fun(system):
        """Compute adf contribution of each triplet."""
        positions = system.R
        R_kj = vmap(partial(displacement, system.cell))(positions[angle_top[:,2]],positions[angle_top[:,1]])
        R_ij = vmap(partial(displacement, system.cell))(positions[angle_top[:,0]],positions[angle_top[:,1]])

        angles = vectorized_angle_fn(R_ij, R_kj)
        
        exponent = jnp.exp(-0.5 * (angles[:, jnp.newaxis] - adf_bin_centers) ** 2 / sigma_theta ** 2)
        gaussians = exponent * bin_size / jnp.sqrt(2 * jnp.pi * sigma_theta ** 2)
        unnormed_adf = high_precision_sum(gaussians, axis=0)
        adf = unnormed_adf / jnp.trapz(unnormed_adf, adf_bin_centers)
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
    reference_adf: The target bdf; initialize with None if no target available
    adf_bin_centers: The positions of the centers of the adf bins over theta
    sigma_ADF: Standard deviation of smoothing Gaussian
    """
    reference_adf: jnp.ndarray
    adf_bin_centers: jnp.ndarray
    adf_bin_boundaries: jnp.ndarray
    sigma_ADF: jnp.ndarray
    angle_top: jnp.ndarray


def ddf_discretization(DDF_cut, nbins=300, DDF_start=0.):
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
    dtheta_bin = (DDF_cut - DDF_start) / float(nbins)
    ddf_bin_centers = jnp.linspace(DDF_start + dtheta_bin / 2., DDF_cut - dtheta_bin / 2., nbins)
    ddf_bin_boundaries = jnp.linspace(DDF_start, DDF_cut, nbins + 1)
    sigma_ADF = jnp.array(dtheta_bin)
    return ddf_bin_centers, ddf_bin_boundaries, sigma_ADF


def initialize_dihedral_distribution_fun(ddf_params):
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
    _, adf_bin_centers, adf_bin_boundaries, sigma_theta, angle_top = dataclasses.astuple(ddf_params)
    bin_size = jnp.diff(adf_bin_boundaries)

    def dihedral_corr_fun(system):
        """Compute adf contribution of each triplet."""

        positions = system.R
        R_cd = vmap(partial(displacement, system.cell))(positions[angle_top[:,3]],positions[angle_top[:,2]])
        R_bc = vmap(partial(displacement, system.cell))(positions[angle_top[:,2]],positions[angle_top[:,1]])
        R_ab = vmap(partial(displacement, system.cell))(positions[angle_top[:,1]],positions[angle_top[:,0]])

        dihedrals = vectorized_dihedral_fn(R_ab, R_bc, R_cd)
        
        exponent = jnp.exp(-0.5 * (dihedrals[:, jnp.newaxis] - adf_bin_centers) ** 2 / sigma_theta ** 2)
        gaussians = exponent * bin_size / jnp.sqrt(2 * jnp.pi * sigma_theta ** 2)
        unnormed_adf = high_precision_sum(gaussians, axis=0)
        ddf = unnormed_adf / jnp.trapz(unnormed_adf, adf_bin_centers)
        return ddf

    def ddf_fn(system, **unused_kwargs):
        ddf = dihedral_corr_fun(system)
        return ddf
    return ddf_fn