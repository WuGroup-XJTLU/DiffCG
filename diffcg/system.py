from collections import namedtuple

from jax import numpy as jnp
from jax.tree_util import tree_map
from diffcg.common.periodic import make_displacement

System = namedtuple("System", ("R", "Z", "cell"))

UnfoldedSystem = namedtuple(
    "System", ("R", "Z", "cell", "mask", "replica_idx", "padding_mask", "updated")
)


def trj_atom_to_system(trj_atoms, dtype=jnp.float32):
    systems = []
    for atoms in trj_atoms:
        R = jnp.array(atoms.get_positions(), dtype=dtype)
        Z = jnp.array(
            atoms.get_atomic_numbers(), dtype=jnp.int32
        )  # we will infer this type
        cell = jnp.array(atoms.get_cell().array.T, dtype=dtype)
        systems.append(System(R, Z, cell))
    batched_systems = tree_map(lambda *xs: jnp.stack(xs), *systems)  # R: (B,500,3), Z: (B,500), cell: (B,3,3)
    return batched_systems

def atoms_to_system(atoms, dtype=jnp.float32):
    R = jnp.array(atoms.get_positions(), dtype=dtype)
    Z = jnp.array(
        atoms.get_atomic_numbers(), dtype=jnp.int32
    )  # we will infer this type
    cell = jnp.array(atoms.get_cell().array.T, dtype=dtype)
    return System(R, Z, cell)


def unfold_system(system, unfolding):
    from glp.unfold import unfold

    N = system.R.shape[0]

    wrapped, unfolded = unfold(system.R, system.cell, unfolding)
    all_R = jnp.concatenate((wrapped, unfolded), axis=0)
    all_idx = jnp.concatenate((jnp.arange(N), unfolding.replica_idx), axis=0)
    all_Z = system.Z[all_idx]

    mask = jnp.arange(all_R.shape[0]) < N
    padding_mask = jnp.concatenate((jnp.ones(N, dtype=bool), unfolding.padding_mask))

    return UnfoldedSystem(all_R, all_Z, None, mask, all_idx, padding_mask, unfolding.updated)


def to_displacement(system):
    return make_displacement(system.cell)