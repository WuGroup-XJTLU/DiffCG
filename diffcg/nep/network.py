"""JAX NEP neural network — single hidden layer per atom type."""

import jax.numpy as jnp


def apply_nep_network(
    q: jnp.ndarray,
    t: int,
    w0: jnp.ndarray,
    b0: jnp.ndarray,
    w1: jnp.ndarray,
    b1: float,
) -> jnp.ndarray:
    """Compute energy of one atom. q: (dim,), returns scalar energy in eV."""
    hidden = jnp.tanh(w0 @ q + b0)
    return jnp.dot(w1, hidden) + b1


def apply_nep_network_batch(
    descriptors: jnp.ndarray,
    types: jnp.ndarray,
    ann_params: dict,
    b1: float,
) -> jnp.ndarray:
    """Compute per-atom energies.
    descriptors: (N, dim), types: (N,), ann_params: per-type {w0, b0, w1}.
    Returns: (N,) per-atom energies in eV.
    """
    N = descriptors.shape[0]
    energies = jnp.zeros(N)
    for t in ann_params:
        mask = (types == t)
        if not jnp.any(mask):
            continue
        ap = ann_params[t]
        q_masked = descriptors[mask]
        hidden = jnp.tanh(q_masked @ ap["w0"].T + ap["b0"])
        e_masked = hidden @ ap["w1"] + b1
        energies = energies.at[mask].set(e_masked)
    return energies
