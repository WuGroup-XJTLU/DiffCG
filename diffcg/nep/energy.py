"""Top-level NEP energy function builder for DiffSim integration."""

import jax.numpy as jnp
from diffcg.nep.descriptor import compute_nep_descriptor
from diffcg.nep.network import apply_nep_network_batch
from diffcg._core.units import EV_TO_KJMOL


def build_nep_energy_fn(nep_params: dict):
    """Build a JAX energy function from NEP parameters.

    Args:
        nep_params: dict from read_nep() containing version, num_types, elements,
            n_max_radial, n_max_angular, basis_size_radial, basis_size_angular,
            L_max, has_q_222, has_q_1111, has_q_112, has_q_1122,
            num_neurons, descriptor_params, ann_params, b1, q_scaler, etc.

    Returns:
        energy_fn(system, neighbors) -> energy in kJ/mol
    """
    num_types = nep_params["num_types"]
    rc_radial = jnp.array(nep_params["rc_radial"], dtype=jnp.float32)
    rc_angular = jnp.array(nep_params["rc_angular"], dtype=jnp.float32)
    n_max_radial = nep_params["n_max_radial"]
    n_max_angular = nep_params["n_max_angular"]
    basis_size_radial = nep_params["basis_size_radial"]
    basis_size_angular = nep_params["basis_size_angular"]
    L_max = nep_params["L_max"]
    has_q_222 = nep_params["has_q_222"]
    has_q_1111 = nep_params["has_q_1111"]
    has_q_112 = nep_params.get("has_q_112", 0)
    has_q_1122 = nep_params.get("has_q_1122", 0)
    num_L = nep_params["num_L"]
    c_descriptor = nep_params["descriptor_params"]
    ann_params = nep_params["ann_params"]
    b1 = nep_params["b1"]
    q_scaler = nep_params["q_scaler"]

    def energy_fn(system, neighbors):
        """Compute total energy. system: System(R, Z, cell), neighbors: JAX-MD nbrs.
        Returns: scalar energy in kJ/mol.
        """
        q = compute_nep_descriptor(
            system.R, system.Z, system.cell, neighbors,
            c_descriptor, rc_radial, rc_angular,
            n_max_radial, n_max_angular,
            basis_size_radial, basis_size_angular,
            num_L, L_max,
            has_q_222, has_q_1111, has_q_112, has_q_1122,
            q_scaler,
        )
        e_per_atom_ev = apply_nep_network_batch(q, system.Z, ann_params, b1)
        return jnp.sum(e_per_atom_ev) * EV_TO_KJMOL

    return energy_fn
