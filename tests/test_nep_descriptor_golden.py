"""Golden reference test: capture current NEP descriptor output before optimization.

Uses a small synthetic system — large enough to exercise all pathways
(radial, 3-body, 4-body, 5-body) but small enough to complete quickly.
"""
import jax
import jax.numpy as jnp
import numpy as np

from diffcg.nep.descriptor import compute_nep_descriptor
from diffcg.system import AtomicSystem
from diffcg._core.neighborlist import jaxmd_neighbor_list

GOLDEN_PATH = "/tmp/golden_descriptors.npz"

# Small model dimensions to keep things fast
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


def _build_synthetic_params():
    """All-ones descriptor params and q_scaler, cutoff=0.6 nm."""
    c_descriptor = jnp.ones(NUM_DESCRIPTOR, dtype=jnp.float32)
    rc_radial = jnp.full((NUM_TYPES,), 0.6, dtype=jnp.float32)  # 6 Angstrom
    rc_angular = jnp.full((NUM_TYPES,), 0.6, dtype=jnp.float32)
    q_scaler = jnp.ones(DIM, dtype=jnp.float32)
    return c_descriptor, rc_radial, rc_angular, q_scaler


def test_nep_descriptor_golden_values():
    """Capture reference descriptor values from a small synthetic system."""
    # 6 atoms in a 1.5 nm box: distances within 0.6 nm cutoff
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [0.15, 0.0, 0.0],
        [0.0, 0.15, 0.0],
        [0.0, 0.0, 0.15],
        [0.1, 0.1, 0.0],
        [0.0, 0.1, 0.1],
    ], dtype=jnp.float32)

    cell = jnp.eye(3, dtype=jnp.float32) * 1.5
    Z = jnp.zeros(6, dtype=jnp.int32)
    masses = jnp.ones(6, dtype=jnp.float32)

    system = AtomicSystem(R=positions, Z=Z, cell=cell, masses=masses, pbc=True)

    c_descriptor, rc_radial, rc_angular, q_scaler = _build_synthetic_params()
    cutoff = float(jnp.maximum(jnp.max(rc_radial), jnp.max(rc_angular)))

    neighbors, _ = jaxmd_neighbor_list(
        positions=system.R, cell=system.cell, cutoff=cutoff,
        capacity_multiplier=2.0,
    )

    descriptors = compute_nep_descriptor(
        system.R, system.Z, system.cell, neighbors,
        c_descriptor, rc_radial, rc_angular,
        N_MAX_RADIAL, N_MAX_ANGULAR,
        BASIS_SIZE_RADIAL, BASIS_SIZE_ANGULAR,
        NUM_L, L_MAX,
        HAS_Q_222, HAS_Q_1111, HAS_Q_112, HAS_Q_1122,
        q_scaler,
    )

    np.savez(GOLDEN_PATH, descriptors=np.array(descriptors))

    assert descriptors.shape[0] == 6
    assert descriptors.shape[1] == DIM
    assert not jnp.any(jnp.isnan(descriptors))
    assert not jnp.any(jnp.isinf(descriptors))

    print(f"DIM={DIM}, shape={descriptors.shape}")
    print(f"mean={float(jnp.mean(descriptors)):.6f}, std={float(jnp.std(descriptors)):.6f}")


def test_nep_descriptor_matches_golden():
    """After optimization, descriptor output must match golden values exactly."""
    import os
    if not os.path.exists(GOLDEN_PATH):
        pytest.skip("Golden file not generated yet — run test_nep_descriptor_golden_values first")

    golden = np.load(GOLDEN_PATH)
    golden_descriptors = jnp.array(golden["descriptors"])

    # Rebuild same system
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [0.15, 0.0, 0.0],
        [0.0, 0.15, 0.0],
        [0.0, 0.0, 0.15],
        [0.1, 0.1, 0.0],
        [0.0, 0.1, 0.1],
    ], dtype=jnp.float32)
    cell = jnp.eye(3, dtype=jnp.float32) * 1.5
    Z = jnp.zeros(6, dtype=jnp.int32)
    masses = jnp.ones(6, dtype=jnp.float32)
    system = AtomicSystem(R=positions, Z=Z, cell=cell, masses=masses, pbc=True)

    c_descriptor, rc_radial, rc_angular, q_scaler = _build_synthetic_params()
    cutoff = float(jnp.maximum(jnp.max(rc_radial), jnp.max(rc_angular)))

    neighbors, _ = jaxmd_neighbor_list(
        positions=system.R, cell=system.cell, cutoff=cutoff,
        capacity_multiplier=2.0,
    )

    descriptors = compute_nep_descriptor(
        system.R, system.Z, system.cell, neighbors,
        c_descriptor, rc_radial, rc_angular,
        N_MAX_RADIAL, N_MAX_ANGULAR,
        BASIS_SIZE_RADIAL, BASIS_SIZE_ANGULAR,
        NUM_L, L_MAX,
        HAS_Q_222, HAS_Q_1111, HAS_Q_112, HAS_Q_1122,
        q_scaler,
    )

    assert jnp.allclose(descriptors, golden_descriptors, atol=0, rtol=0), \
        "Descriptor output diverged from golden — optimization broke bitwise-identical guarantee"
