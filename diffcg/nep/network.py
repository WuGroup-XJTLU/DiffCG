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
    # GPUMD convention: tanh(w0@q - b0), energy = sum(w1*tanh) - b1
    hidden = jnp.tanh(w0 @ q - b0)
    return jnp.dot(w1, hidden) - b1


def apply_nep_network_batch(
    descriptors: jnp.ndarray,
    types: jnp.ndarray,
    ann_params: dict,
    b1: float,
) -> jnp.ndarray:
    """Compute per-atom energies (JAX-traceable).

    descriptors: (N, dim), types: (N,), ann_params: per-type {w0, b0, w1}.
    Returns: (N,) per-atom energies in eV.

    Uses mask-based accumulation instead of boolean indexing so the
    function is compatible with JAX tracing (jax.jit, jax.grad).
    """
    N = descriptors.shape[0]
    energies = jnp.zeros(N)
    for t in ann_params:
        ap = ann_params[t]
        mask = (types == t)  # (N,) bool
        mask_f = mask.astype(descriptors.dtype)  # (N,) float

        # Apply network to all atoms, then mask by type
        # GPUMD convention: tanh(w0@q - b0), energy = sum(w1*tanh) - b1
        hidden = jnp.tanh(descriptors @ ap["w0"].T - ap["b0"])  # (N, neurons)
        e_all = hidden @ ap["w1"] - b1  # (N,)
        energies = energies + mask_f * e_all
    return energies
