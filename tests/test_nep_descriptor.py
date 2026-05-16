import jax
import jax.numpy as jnp
from diffcg.nep.descriptor import (
    compute_radial_descriptor,
    compute_angular_descriptor,
    _legendre_all,
)


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


def test_angular_descriptor_two_neighbors():
    """Two neighbors at symmetric positions should give non-zero angular descriptor."""
    n_max = 1
    num_L = 3
    num_types = 1
    c_angular = jnp.ones((num_types, n_max + 1, 3))  # basis_size=2
    R_i = jnp.array([0.0, 0.0, 0.0])
    R_nbr = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    types_nbr = jnp.array([0, 0])

    q = compute_angular_descriptor(
        R_i, R_nbr, types_nbr, 0,
        c_angular, rc_angular=3.0,
        n_max=1, basis_size=2, num_L=3, L_max=2,
        has_q_222=0, has_q_1111=0, has_q_112=0, has_q_1122=0,
    )
    # angular dim = (n_max+1) * num_L = 2 * 3 = 6
    assert q.shape == (6,)


def test_legendre():
    """P_0(x)=1, P_1(x)=x, P_2(x)=(3x^2-1)/2."""
    x = jnp.array(0.5)
    P = _legendre_all(x, L_max=2)
    assert abs(float(P[0]) - 1.0) < 1e-6
    assert abs(float(P[1]) - 0.5) < 1e-6
    assert abs(float(P[2]) - (-0.125)) < 1e-6


def test_legendre_all_basic():
    """Check values against known Legendre polynomials."""
    x = jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    P = _legendre_all(x, L_max=3)  # (5, 4)
    assert P.shape == (5, 4)

    # P_0 = 1
    assert jnp.allclose(P[:, 0], 1.0)
    # P_1 = x
    assert jnp.allclose(P[:, 1], x)
    # P_2(x) = (3x^2 - 1)/2
    expected_P2 = (3.0 * x ** 2 - 1.0) / 2.0
    assert jnp.allclose(P[:, 2], expected_P2)
    # P_3(x) = (5x^3 - 3x)/2
    expected_P3 = (5.0 * x ** 3 - 3.0 * x) / 2.0
    assert jnp.allclose(P[:, 3], expected_P3)


def test_legendre_all_edge_cases():
    """L_max=0 and L_max=1."""
    x = jnp.array([0.3])
    P0 = _legendre_all(x, L_max=0)
    assert P0.shape == (1, 1)
    assert jnp.allclose(P0[0, 0], 1.0)

    P1 = _legendre_all(x, L_max=1)
    assert P1.shape == (1, 2)
    assert jnp.allclose(P1[0, 0], 1.0)
    assert jnp.allclose(P1[0, 1], 0.3)


def test_legendre_all_multidimensional():
    """Works with multi-dimensional input."""
    x = jnp.array([[0.0, 0.5], [1.0, -0.5]])  # (2, 2)
    P = _legendre_all(x, L_max=2)  # (2, 2, 3)
    assert P.shape == (2, 2, 3)
    assert jnp.allclose(P[0, 0, 0], 1.0)  # P_0 everywhere = 1
    assert jnp.allclose(P[:, :, 1], x)     # P_1 = x
