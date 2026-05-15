import jax
import jax.numpy as jnp
from diffcg.nep.descriptor import compute_radial_descriptor


def test_radial_descriptor_two_atoms():
    """Two atoms of same type, descriptor should be non-zero."""
    c_radial = jnp.ones((1, 3, 5))  # 1 type, n_max=2, basis_size=4
    R_i = jnp.array([0.0, 0.0, 0.0])
    R_nbr = jnp.array([[2.0, 0.0, 0.0]])  # 2 Angstrom away
    types_nbr = jnp.array([0])

    q_radial = compute_radial_descriptor(
        R_i, R_nbr, types_nbr, 0,
        c_radial, rc_radial=5.0,
        n_max=2, basis_size=4,
    )
    assert q_radial.shape == (3,)  # n_max+1 = 3
    # All positive params + positive cutoff => positive descriptor
    assert jnp.all(q_radial > 0.0)


def test_radial_descriptor_beyond_cutoff():
    """Neighbor beyond cutoff should contribute zero."""
    c_radial = jnp.ones((1, 3, 5))
    R_i = jnp.array([0.0, 0.0, 0.0])
    R_nbr = jnp.array([[10.0, 0.0, 0.0]])  # 10 A away, cutoff is 5 A

    q_radial = compute_radial_descriptor(
        R_i, R_nbr, jnp.array([0]), 0,
        c_radial, rc_radial=5.0,
        n_max=2, basis_size=4,
    )
    assert jnp.allclose(q_radial, 0.0, atol=1e-6)
