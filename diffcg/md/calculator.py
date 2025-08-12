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

from diffcg.system import System, atoms_to_system
from diffcg.common.neighborlist import neighbor_list
from diffcg.util.logger import get_logger

logger = get_logger(__name__)


def force(energy_fn: Callable) -> Callable:
  """Computes the force as the negative gradient of an energy."""
  return grad(lambda atoms, *args, **kwargs: -energy_fn(atoms, *args, **kwargs))


class CustomCalculator(Calculator):
    def __init__(self, potentials, calculate_stress=False, capacity_multiplier=1.25, dtype=jnp.float64, cutoff=1.0,skin=0.0):
        super().__init__()

        if calculate_stress:
            # TODO: implement stress calculation
            pass
            """
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
            """
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
        self.skin = skin
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
            logger.debug("Building neighbor list: cutoff=%s, skin=%s", self.cutoff, self.skin)
            self.neighbors, self.spatial_partitioning = neighbor_list(positions=R,
                                                                      cell=cell,
                                                                      cutoff=self.cutoff,
                                                                      skin=self.skin,
                                                                      capacity_multiplier=self.capacity_multiplier)

        neighbors = self.spatial_partitioning.update_fn(R, self.neighbors, new_cell=cell)
        if neighbors.overflow:
            logger.error('Neighbor list overflow detected')
            raise RuntimeError('Spatial overflow.')
        else:
            self.neighbors = neighbors

        output = self.calculate_fn(System(R=R, Z=z, cell=cell), neighbors=neighbors)  # note different cell convention
        
        self.results = jax.tree_map(lambda x: np.array(x, dtype=self.dtype), output)
        if jnp.isnan(self.results['energy']):
            logger.error('NaN energy encountered')
            raise RuntimeError('Energy is NaN.')


class CustomEnergyCalculator(Calculator):
    def __init__(self, potentials, calculate_stress=False, capacity_multiplier=1.25, dtype=jnp.float64, cutoff=1.0,skin=0.0):
        super().__init__()

        if calculate_stress:
            # TODO: implement stress calculation
            pass
            """
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
            """
        else:
            def energy_fn(system, neighbors):
                return potentials(system, neighbors)

            @jax.jit
            def calculate_fn(system, neighbors):
                energy,_ = jax.value_and_grad(energy_fn, allow_int=True)(system, neighbors)
                return energy

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
        super(CustomEnergyCalculator, self).calculate(atoms, *args, **kwargs)

        R = jnp.array(atoms.get_positions(), dtype=self.dtype)  # shape: (n,3)
        z = jnp.array(atoms.get_atomic_numbers(), dtype=jnp.int16)  # shape: (n)

        if atoms.get_pbc().any():
            cell = jnp.array(np.array(atoms.get_cell()), dtype=self.dtype).T  # (3,3)
        else:
            cell = None
        if self.spatial_partitioning is None:
            logger.debug("Building neighbor list (energy-only): cutoff=%s", self.cutoff)
            self.neighbors, self.spatial_partitioning = neighbor_list(positions=R,
                                                                      cell=cell,
                                                                      cutoff=self.cutoff,
                                                                      skin=0.,
                                                                      capacity_multiplier=self.capacity_multiplier)

        neighbors = self.spatial_partitioning.update_fn(R, self.neighbors,new_cell=cell)
        if neighbors.overflow:
            logger.error('Neighbor list overflow detected')
            raise RuntimeError('Spatial overflow.')
        else:
            self.neighbors = neighbors

        output = self.calculate_fn(System(R=R, Z=z, cell=cell), neighbors=neighbors)  # note different cell convention
        self.results = output
        
def init_energy_calculator(energy_fn,cutoff=1.0,capacity_multiplier=1.25,dtype=jnp.float64):

    def calculate_fn(system,**kwargs):
        R = system.R
        z = system.Z
        cell = system.cell
        neighbors, spatial_partitioning = neighbor_list(positions=R,
                                                        cell=cell,
                                                        cutoff=cutoff,
                                                        skin=0.,
                                                        capacity_multiplier=capacity_multiplier)
        
        return energy_fn(system, neighbors,**kwargs)
    return calculate_fn