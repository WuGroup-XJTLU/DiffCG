# define a general ASE calculator class that can use custom energy function to calculate the energy and force
from functools import partial
from typing import Callable, Union
import jax.numpy as jnp
import numpy as np
from jax import grad, vmap
import jax
from ase.calculators.calculator import Calculator

from collections import namedtuple
from typing import Any

from ase.calculators.calculator import Calculator

try:
    from glp.calculators.utils import strain_graph, get_strain
    from glp import System, atoms_to_system
    from glp.graph import system_to_graph
except ImportError:
    raise ImportError('Please install GLP package for running MD.')

SpatialPartitioning = namedtuple(
    "SpatialPartitioning", ("allocate_fn", "update_fn", "cutoff", "skin", "capacity_multiplier")
)


def cast(x):
    """Cast number literal to jnp.ndarray.

    This avoids jit recompiles, as native python types
    are "weak" types in jax. This makes everything explicit.
    In high-precision situations, jax type promotion should™
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

def displacement(cell, Ra, Rb):
    if cell is None:
        return Rb - Ra

    else:
        R = Rb - Ra
        X = _to_frac(cell, R)
        X = jnp.mod(X + cast(0.5), cast(1.0)) - cast(0.5)

        return _from_frac(cell, X)

def force(energy_fn: Callable) -> Callable:
  """Computes the force as the negative gradient of an energy."""
  return grad(lambda atoms, *args, **kwargs: -energy_fn(atoms, *args, **kwargs))


class CustomCalculator(Calculator):
    def __init__(self, potentials, calculate_stress=False, capacity_multiplier=1.25, dtype=jnp.float64, cutoff=1.0,skin=0.0):
        super().__init__()

    
        if calculate_stress:
            def energy_fn(system, strain: jnp.ndarray, neighbors):
                graph = system_to_graph(system, neighbors)
                graph = strain_graph(graph, strain)
                return potentials(graph).sum()

            @jax.jit
            def calculate_fn(system: System, neighbors):
                strain = get_strain()
                energy, grads = jax.value_and_grad(energy_fn, argnums=(0, 1), allow_int=True)(system, strain, neighbors)
                forces = - grads[0].R
                stress = grads[1]/system.cell[0][0]**3 #only work for cubic box
                return {'energy': energy, 'forces': forces, 'stress': stress}
        else:
            def energy_fn(system, neighbors):
                return potentials(system, neighbors)

            @jax.jit
            def calculate_fn(system, neighbors):
                energy, grads = jax.value_and_grad(energy_fn, allow_int=True)(system, neighbors)
                forces = - grads.R
                return {'energy': energy, 'forces': forces}

        self.calculate_fn = calculate_fn

        self.neighbors = None
        self.spatial_partitioning = None
        self.capacity_multiplier = capacity_multiplier

        self.cutoff = cutoff

        self.dtype = dtype
        self.potential_energy = 0.0

        self.implemented_properties = [
                "energy",
                "forces"
            ]



    def calculate(self, atoms=None, *args, **kwargs):
        super(CustomCalculator, self).calculate(atoms, *args, **kwargs)

        R = jnp.array(atoms.get_positions(), dtype=self.dtype)  # shape: (n,3)
        z = jnp.array(atoms.get_atomic_numbers(), dtype=jnp.int16)  # shape: (n)

        if atoms.get_pbc().any():
            cell = jnp.array(np.array(atoms.get_cell()), dtype=self.dtype).T  # (3,3)
        else:
            cell = None
        if self.spatial_partitioning is None:
            self.neighbors, self.spatial_partitioning = neighbor_list(positions=R,
                                                                      cell=cell,
                                                                      cutoff=self.cutoff,
                                                                      skin=0.,
                                                                      capacity_multiplier=self.capacity_multiplier)

        neighbors = self.spatial_partitioning.update_fn(R, self.neighbors,new_cell=cell)
        if neighbors.overflow:
            raise RuntimeError('Spatial overflow.')
        else:
            self.neighbors = neighbors

        output = self.calculate_fn(System(R=R, Z=z, cell=cell), neighbors=neighbors)  # note different cell convention
        
        self.results = jax.tree_map(lambda x: np.array(x, dtype=self.dtype), output)


def to_displacement(cell):
    """
    Returns function to calculate replacement. Returned function takes Ra and Rb as input and return Ra - Rb

    Args:
        cell ():

    Returns:

    """
    from glp.periodic import make_displacement

    displacement = make_displacement(cell)
    # displacement(Ra, Rb) calculates Rb - Ra

    # reverse sign convention bc feels more natural
    return lambda Ra, Rb: displacement(Rb, Ra)

@jax.jit
def add_batch_dim(tree):
    return jax.tree_map(lambda x: x[None], tree)


@jax.jit
def apply_neighbor_convention(tree):
    idx_i = jnp.where(tree['idx_i'] < len(tree['z']), tree['idx_i'], -1)
    idx_j = jnp.where(tree['idx_j'] < len(tree['z']), tree['idx_j'], -1)
    tree['idx_i'] = idx_i
    tree['idx_j'] = idx_j
    return tree


def neighbor_list(positions: jnp.ndarray, cutoff: float, skin: float, cell: jnp.ndarray = None,
                  capacity_multiplier: float = 1.4):
    """

    Args:
        positions ():
        cutoff ():
        skin ():
        cell (): ASE cell.
        capacity_multiplier ():

    Returns:

    """
    try:
        from glp.neighborlist import quadratic_neighbor_list
    except ImportError:
        raise ImportError('For neighborhood list, please install the glp package from ...')
    # Convenience interface
    # if cell is not None:
    #     cell_T = cell.T
    # else:
    #     cell_T = None

    allocate, update = quadratic_neighbor_list(
        cell, cutoff, skin, capacity_multiplier=capacity_multiplier
    )

    neighbors = allocate(positions)

    return neighbors, SpatialPartitioning(allocate_fn=allocate,
                                          update_fn=jax.jit(update),
                                          cutoff=cutoff,
                                          skin=skin,
                                          capacity_multiplier=capacity_multiplier)

