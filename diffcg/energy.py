"""Custom definition of some potential energy functions."""
from typing import Callable

import jax.numpy as jnp
from functools import partial,wraps
from jax import vmap
import jax
from diffcg.util import custom_interpolate
from diffcg.common.geometry import distance, vectorized_angle_fn, vectorized_dihedral_fn
from diffcg.common.periodic import displacement
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

def mask_bonded_neighbors(idx, topology, max_num_atoms):
    """
    Mask neighbor pairs that are bonded through dihedral interactions.

    This function identifies pairs of atoms that are bonded through dihedral
    interactions and masks them by setting their indices to max_num_atoms.
    This prevents double-counting of interactions between bonded atoms.

    Args:
        idx (jnp.ndarray): (2, num_pairs) int32 array of neighbor pair indices
        dihedrals (jnp.ndarray): (num_dihedrals, 4) int32 array of dihedral definitions
        max_num_atoms (int): Maximum number of atoms, used as mask value

    Returns:
        jnp.ndarray: (2, num_pairs) int32 array with bonded pairs masked
    """
    i = idx[0]  # (num_pairs,) - first atom in each pair
    j = idx[1]  # (num_pairs,) - second atom in each pair

    # For each pair (i, j), check if it is in dihedrals (in either order)
    def is_bonded(pair_i, pair_j):
        """
        Check if two atoms are bonded through any dihedral interaction.

        Args:
            pair_i (int): Index of first atom
            pair_j (int): Index of second atom

        Returns:
            bool: True if atoms are bonded through dihedral
        """
        if topology.shape[1]==4:
            return jnp.any(
            # Check all possible pairs within dihedrals
            ((topology[:, 0] == pair_i) & (topology[:, 1] == pair_j)) |
            ((topology[:, 0] == pair_j) & (topology[:, 1] == pair_i)) |
            ((topology[:, 0] == pair_i) & (topology[:, 2] == pair_j)) |
            ((topology[:, 0] == pair_j) & (topology[:, 2] == pair_i)) |
            ((topology[:, 0] == pair_i) & (topology[:, 3] == pair_j)) |
            ((topology[:, 0] == pair_j) & (topology[:, 3] == pair_i)) |
            ((topology[:, 1] == pair_i) & (topology[:, 2] == pair_j)) |
            ((topology[:, 1] == pair_j) & (topology[:, 2] == pair_i)) |
            ((topology[:, 1] == pair_i) & (topology[:, 3] == pair_j)) |
            ((topology[:, 1] == pair_j) & (topology[:, 3] == pair_i)) |
            ((topology[:, 2] == pair_i) & (topology[:, 3] == pair_j)) |
            ((topology[:, 2] == pair_j) & (topology[:, 3] == pair_i))
        )
        elif topology.shape[1]==3:
            return jnp.any(
            # Check all possible pairs within dihedrals
            ((topology[:, 0] == pair_i) & (topology[:, 1] == pair_j)) |
            ((topology[:, 0] == pair_j) & (topology[:, 1] == pair_i)) |
            ((topology[:, 0] == pair_i) & (topology[:, 2] == pair_j)) |
            ((topology[:, 0] == pair_j) & (topology[:, 2] == pair_i)) |
            ((topology[:, 1] == pair_i) & (topology[:, 2] == pair_j)) |
            ((topology[:, 1] == pair_j) & (topology[:, 2] == pair_i)) 
            )
        elif topology.shape[1]==2:
            return jnp.any(
            ((topology[:, 0] == pair_i) & (topology[:, 1] == pair_j)) |
            ((topology[:, 0] == pair_j) & (topology[:, 1] == pair_i))
            )
        else:
            raise ValueError(f"Topology shape {topology.shape} not supported")  
        
    # Vectorize the bonded check over all pairs
    v_is_bonded = vmap(is_bonded)
    bonded_mask = v_is_bonded(i, j)

    # Set both i and j to max_num_atoms if bonded, else keep original
    i_masked = jnp.where(bonded_mask, max_num_atoms, i)
    j_masked = jnp.where(bonded_mask, max_num_atoms, j)
    return i_masked,j_masked



class TabulatedBondEnergy:
    def __init__(self, x_vals, y_vals,bonds):
        self.x_vals = x_vals
        self.y_vals = y_vals
        self.bonds = bonds

    def get_energy_fn(self):
        spline = custom_interpolate.MonotonicInterpolate(self.x_vals, self.y_vals)
        tabulated_partial = partial(tabulated, spline=spline)
      
        def energy_fn(system, neighbors, **dynamic_kwargs):
            positions = system.R
            nodes = system.Z
            Ra = positions[self.bonds[:, 0]]
            Rb = positions[self.bonds[:, 1]]
            edges = vmap(partial(displacement, system.cell))(Ra, Rb)
            dr=vmap(distance)(edges)
            return high_precision_sum(tabulated_partial(dr))
        return energy_fn
    
class HarmonicBondEnergy:
    def __init__(self,bonds,length=0.45,epsilon=5000):
        self.bonds = bonds
        self.length = length
        self.epsilon = epsilon

    def get_energy_fn(self):
      
        def energy_fn(system, neighbors, **dynamic_kwargs):
            positions = system.R
            nodes = system.Z
            Ra = positions[self.bonds[:, 0]]
            Rb = positions[self.bonds[:, 1]]
            edges = vmap(partial(displacement, system.cell))(Ra, Rb)
            dr=vmap(distance)(edges)
            return high_precision_sum(simple_spring(dr, self.length, self.epsilon))
        return energy_fn
    
  
class TabulatedAngleEnergy:
    def __init__(self, x_vals, y_vals,angles):
        self.x_vals = x_vals
        self.y_vals = y_vals
        self.angles = angles

    def get_energy_fn(self):
        spline = custom_interpolate.MonotonicInterpolate(self.x_vals, self.y_vals)
        tabulated_partial = partial(tabulated, spline=spline)
      
        def energy_fn(system, neighbors, **dynamic_kwargs):
            positions = system.R
            nodes = system.Z
            R_kj = vmap(partial(displacement, system.cell))(positions[self.angles[:,2]],positions[self.angles[:,1]])
            R_ij = vmap(partial(displacement, system.cell))(positions[self.angles[:,0]],positions[self.angles[:,1]])

            angles = vectorized_angle_fn(R_ij, R_kj)
            return high_precision_sum(tabulated_partial(angles))
        return energy_fn
    
class HarmonicAngleEnergy:
    def __init__(self,angles,angle_0=1.5,epsilon=50):
        self.angles = angles
        self.angle_0 = angle_0
        self.epsilon = epsilon

    def get_energy_fn(self):
        def energy_fn(system, neighbors, **dynamic_kwargs):
            positions = system.R
            nodes = system.Z
            R_kj = vmap(partial(displacement, system.cell))(positions[self.angles[:,2]],positions[self.angles[:,1]])
            R_ij = vmap(partial(displacement, system.cell))(positions[self.angles[:,0]],positions[self.angles[:,1]])

            angles = vectorized_angle_fn(R_ij, R_kj)
            return high_precision_sum(harmonic_angle(angles, self.angle_0, self.epsilon))
        return energy_fn
  
class TabulatedDihedralEnergy:
    def __init__(self, x_vals, y_vals,dihedrals):
        self.x_vals = x_vals
        self.y_vals = y_vals
        self.dihedrals = dihedrals

    def get_energy_fn(self):
        spline = custom_interpolate.MonotonicInterpolate(self.x_vals, self.y_vals)
        tabulated_partial = partial(tabulated, spline=spline)
      
        def energy_fn(system, neighbors, **dynamic_kwargs):
            positions = system.R
            nodes = system.Z
            R_cd = vmap(partial(displacement, system.cell))(positions[self.dihedrals[:,3]],positions[self.dihedrals[:,2]])
            R_bc = vmap(partial(displacement, system.cell))(positions[self.dihedrals[:,2]],positions[self.dihedrals[:,1]])
            R_ab = vmap(partial(displacement, system.cell))(positions[self.dihedrals[:,1]],positions[self.dihedrals[:,0]])

            dihedrals = vectorized_dihedral_fn(R_ab, R_bc, R_cd)
            return high_precision_sum(tabulated_partial(dihedrals))
        return energy_fn
    
class HarmonicDihedralEnergy:
    def __init__(self,dihedrals,angle_0=1.5,epsilon=50):
        self.dihedrals = dihedrals
        self.angle_0 = angle_0
        self.epsilon = epsilon

    def get_energy_fn(self):
        def energy_fn(system, neighbors, **dynamic_kwargs):
            positions = system.R
            nodes = system.Z
            R_cd = vmap(partial(displacement, system.cell))(positions[self.dihedrals[:,3]],positions[self.dihedrals[:,2]])
            R_bc = vmap(partial(displacement, system.cell))(positions[self.dihedrals[:,2]],positions[self.dihedrals[:,1]])
            R_ab = vmap(partial(displacement, system.cell))(positions[self.dihedrals[:,1]],positions[self.dihedrals[:,0]])
            dihedrals = vectorized_dihedral_fn(R_ab, R_bc, R_cd)
            return high_precision_sum(harmonic_dihedral(dihedrals, self.angle_0, self.epsilon))
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

    def get_energy_fn(self):
        if self.mask_topology is None:
            def energy_fn(system, neighbors, **dynamic_kwargs):
                positions = system.R
                nodes = system.Z
                
                edges = vmap(partial(displacement, system.cell))(
                    positions[neighbors.centers], positions[neighbors.others]
                )
                dr = vmap(distance)(edges)
                
                _energy = multiplicative_isotropic_cutoff(generic_repulsion, self.r_onset, self.r_cutoff)(dr, self.sigma, self.epsilon, self.exp)
                mask = neighbors.centers != positions.shape[0]
                
                out = _energy * mask
                return high_precision_sum(out) * 0.5

            return energy_fn
        else:
            def energy_fn(system, neighbors, **dynamic_kwargs):
                positions = system.R
                nodes = system.Z

                mask_centers,mask_others = mask_bonded_neighbors((neighbors.centers,neighbors.others),self.mask_topology,self.max_num_atoms)
                mask = mask_centers < positions.shape[0]
                #for i in range(len(mask_centers)):
                #    print(mask_centers[i],mask_others[i],neighbors.centers[i],neighbors.others[i])

                edges = vmap(partial(displacement, system.cell))(
                    positions[neighbors.centers], positions[neighbors.others]
                )
                #for i in range(edges.shape[0]):
                #    print(edges[i])

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

    def get_energy_fn(self):
        spline = custom_interpolate.MonotonicInterpolate(self.x_vals, self.y_vals)
        tabulated_partial = partial(tabulated, spline=spline)

        if self.mask_topology is None:
            def energy_fn(system, neighbors, **dynamic_kwargs):
                positions = system.R
                nodes = system.Z
                
                edges = vmap(partial(displacement, system.cell))(
                    positions[neighbors.centers], positions[neighbors.others]
                )
                dr = vmap(distance)(edges)
                
                truncated_fn=multiplicative_isotropic_cutoff(tabulated_partial, self.r_onset,
                                                self.r_cutoff)
                mask = neighbors.centers != positions.shape[0]
                
                _energy = truncated_fn(dr)
                out = _energy * mask
                return high_precision_sum(out) * 0.5

            return energy_fn
        else:
            def energy_fn(system, neighbors, **dynamic_kwargs):
                positions = system.R
                nodes = system.Z

                mask_centers,mask_others = mask_bonded_neighbors((neighbors.centers,neighbors.others),self.mask_topology,self.max_num_atoms)
                mask = mask_centers < positions.shape[0]
                
                edges = vmap(partial(displacement, system.cell))(
                    positions[neighbors.centers], positions[neighbors.others]
                )
                dr = vmap(distance)(edges)
                
                truncated_fn=multiplicative_isotropic_cutoff(tabulated_partial, self.r_onset,
                                                self.r_cutoff)
                
                _energy = truncated_fn(dr)
                out = _energy * mask
                return high_precision_sum(out) * 0.5

            return energy_fn