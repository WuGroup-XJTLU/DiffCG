import jax.numpy as jnp
from diffcg.nep.network import apply_nep_network, apply_nep_network_batch


def test_nep_network_single_atom():
    dim = 5
    neurons = 3
    w0 = jnp.ones((neurons, dim), dtype=jnp.float32) * 0.1
    b0 = jnp.zeros(neurons, dtype=jnp.float32)
    w1 = jnp.ones(neurons, dtype=jnp.float32) * 0.1
    q = jnp.ones(dim, dtype=jnp.float32) * 0.5
    b1 = jnp.float32(0.0)

    e = apply_nep_network(q, 0, w0, b0, w1, b1)
    assert e.shape == ()
    assert float(e) != 0.0  # non-zero input should give non-zero output


def test_nep_network_batch():
    dim = 5
    neurons = 3
    ann_params = {
        0: {"w0": jnp.ones((neurons, dim)) * 0.1,
            "b0": jnp.zeros(neurons),
            "w1": jnp.ones(neurons) * 0.1},
        1: {"w0": jnp.ones((neurons, dim)) * 0.2,
            "b0": jnp.zeros(neurons),
            "w1": jnp.ones(neurons) * 0.2},
    }
    descriptors = jnp.ones((4, dim)) * 0.5
    types = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
    b1 = jnp.float32(0.0)

    e = apply_nep_network_batch(descriptors, types, ann_params, b1)
    assert e.shape == (4,)
    # Type 0 and type 1 should have different energies (different weights)
    assert e[0] != e[2]
