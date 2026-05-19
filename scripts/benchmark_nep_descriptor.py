#!/usr/bin/env python3
"""Benchmark NEP descriptor computation before/after optimization."""
import os
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from diffcg.nep.descriptor import compute_nep_descriptor
from diffcg.system import AtomicSystem
from diffcg._core.neighborlist import jaxmd_neighbor_list

# NEP model dimensions (match golden test)
N_MAX_RADIAL = 2
N_MAX_ANGULAR = 2
BASIS_SIZE_RADIAL = 3
BASIS_SIZE_ANGULAR = 2
L_MAX = 2
HAS_Q_222 = 1
HAS_Q_1111 = 0
HAS_Q_112 = 1
HAS_Q_1122 = 0
NUM_TYPES = 1

FLAG_COUNT = HAS_Q_222 + HAS_Q_1111 + HAS_Q_112 + HAS_Q_1122
NUM_L = L_MAX + FLAG_COUNT
DIM = (N_MAX_RADIAL + 1) + (N_MAX_ANGULAR + 1) * L_MAX + FLAG_COUNT * (N_MAX_ANGULAR + 1)

RADIAL_PARAM_SIZE = (N_MAX_RADIAL + 1) * (BASIS_SIZE_RADIAL + 1)
ANGULAR_PARAM_SIZE = (N_MAX_ANGULAR + 1) * (BASIS_SIZE_ANGULAR + 1)
BLOCK_SIZE = RADIAL_PARAM_SIZE + ANGULAR_PARAM_SIZE
NUM_DESCRIPTOR = NUM_TYPES * NUM_TYPES * BLOCK_SIZE


def build_system(n_atoms=30):
    """Build a synthetic periodic system with reasonable neighbor counts."""
    key = jax.random.PRNGKey(42)
    box_size = 1.2  # nm, tight box ensures neighbors within 0.6 nm cutoff
    positions = jax.random.uniform(key, (n_atoms, 3), dtype=jnp.float32) * box_size
    cell = jnp.eye(3, dtype=jnp.float32) * box_size
    Z = jnp.zeros(n_atoms, dtype=jnp.int32)
    masses = jnp.ones(n_atoms, dtype=jnp.float32)
    return AtomicSystem(R=positions, Z=Z, cell=cell, masses=masses, pbc=True)


def build_params():
    c_descriptor = jnp.ones(NUM_DESCRIPTOR, dtype=jnp.float32)
    rc_radial = jnp.full((NUM_TYPES,), 0.6, dtype=jnp.float32)
    rc_angular = jnp.full((NUM_TYPES,), 0.6, dtype=jnp.float32)
    q_scaler = jnp.ones(DIM, dtype=jnp.float32)
    return c_descriptor, rc_radial, rc_angular, q_scaler


def main():
    system = build_system(n_atoms=30)
    c_descriptor, rc_radial, rc_angular, q_scaler = build_params()
    cutoff = float(jnp.maximum(jnp.max(rc_radial), jnp.max(rc_angular)))

    neighbors, _ = jaxmd_neighbor_list(
        positions=system.R, cell=system.cell, cutoff=cutoff,
        capacity_multiplier=2.0,
    )

    avg_nbrs = int(jnp.mean(jnp.sum(neighbors.idx < system.R.shape[0], axis=1)))
    print(f"System: {system.R.shape[0]} atoms, ~{avg_nbrs} neighbors/atom, dim={DIM}")

    @jax.jit
    def compute_fn(R):
        return compute_nep_descriptor(
            R, system.Z, system.cell, neighbors,
            c_descriptor, rc_radial, rc_angular,
            N_MAX_RADIAL, N_MAX_ANGULAR,
            BASIS_SIZE_RADIAL, BASIS_SIZE_ANGULAR,
            NUM_L, L_MAX,
            HAS_Q_222, HAS_Q_1111, HAS_Q_112, HAS_Q_1122,
            q_scaler,
        )

    # Warm-up
    _ = compute_fn(system.R).block_until_ready()

    # Benchmark forward pass
    n_runs = 20
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = compute_fn(system.R).block_until_ready()
        times.append(time.perf_counter() - t0)

    print(f"\nForward pass ({n_runs} runs):")
    print(f"  mean: {sum(times)/len(times)*1000:.2f} ms")
    print(f"  min:  {min(times)*1000:.2f} ms")
    print(f"  max:  {max(times)*1000:.2f} ms")

    # Benchmark forward + backward
    grad_fn = jax.jit(jax.grad(lambda R: jnp.sum(compute_fn(R))))
    _ = grad_fn(system.R).block_until_ready()

    grad_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = grad_fn(system.R).block_until_ready()
        grad_times.append(time.perf_counter() - t0)

    print(f"\nForward + backward ({n_runs} runs):")
    print(f"  mean: {sum(grad_times)/len(grad_times)*1000:.2f} ms")
    print(f"  min:  {min(grad_times)*1000:.2f} ms")
    print(f"  max:  {max(grad_times)*1000:.2f} ms")


if __name__ == "__main__":
    main()
