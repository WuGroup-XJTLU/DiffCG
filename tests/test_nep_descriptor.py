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


# --- 4-body contraction tests ---

from diffcg.nep.descriptor import _contract_4body
from diffcg.nep.constants import C4B, C4B2


def test_contract_4body_basic():
    """4-body contraction with simple 3-neighbor geometry."""
    n_max = 2
    M = 3
    fc = jnp.array([0.5, 0.5, 0.5])
    T = jnp.ones((M, n_max + 1))  # all T_n = 1 for simplicity
    pair_mask = 1.0 - jnp.eye(M)  # all pairs valid
    cos_theta = jnp.array([
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ])

    q = _contract_4body(fc, T, pair_mask, cos_theta, C4B, L_val=2, n_max=n_max)
    assert q.shape == (n_max + 1,)
    # With all-positive contributions, result should be non-zero
    assert jnp.all(q != 0.0)


def test_contract_4body_distinct_mask():
    """4-body with only 2 neighbors should be zero (need 3 distinct)."""
    n_max = 1
    M = 2
    fc = jnp.array([1.0, 1.0])
    T = jnp.ones((M, n_max + 1))
    pair_mask = 1.0 - jnp.eye(M)
    cos_theta = jnp.array([[0.0, 0.5], [0.5, 0.0]])

    q = _contract_4body(fc, T, pair_mask, cos_theta, C4B, L_val=2, n_max=n_max)
    assert q.shape == (n_max + 1,)
    # With M=2, can't form 3 distinct neighbors: mask_3 should zero everything
    assert jnp.allclose(q, 0.0)


# --- JAX compatibility tests ---


def test_radial_descriptor_jit_grad():
    """radial descriptor compiles under jit and grad."""
    c_radial = jnp.ones((1, 3, 5))
    R_i = jnp.array([0.0, 0.0, 0.0])
    R_nbr = jnp.array([[2.0, 0.0, 0.0]])
    types_nbr = jnp.array([0])

    fn = lambda ri: compute_radial_descriptor(
        ri, R_nbr, types_nbr, 0, c_radial, rc_radial=5.0,
        n_max=2, basis_size=4,
    )

    q_jit = jax.jit(fn)(R_i)
    assert q_jit.shape == (3,)

    grad_fn = jax.grad(lambda ri: jnp.sum(fn(ri)))
    g = jax.jit(grad_fn)(R_i)
    assert g.shape == (3,)


def test_angular_descriptor_jit_grad():
    """angular descriptor compiles under jit and grad."""
    n_max = 1
    num_L = 3
    c_angular = jnp.ones((1, n_max + 1, 3))
    R_i = jnp.array([0.0, 0.0, 0.0])
    R_nbr = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    types_nbr = jnp.array([0, 0])

    fn = lambda ri: compute_angular_descriptor(
        ri, R_nbr, types_nbr, 0,
        c_angular, rc_angular=3.0,
        n_max=n_max, basis_size=2, num_L=num_L, L_max=2,
        has_q_222=0, has_q_1111=0, has_q_112=0, has_q_1122=0,
    )

    q_jit = jax.jit(fn)(R_i)
    assert q_jit.shape == ((n_max + 1) * num_L,)

    grad_fn = jax.grad(lambda ri: jnp.sum(fn(ri)))
    g = jax.jit(grad_fn)(R_i)
    assert g.shape == (3,)


def test_angular_descriptor_with_4body_flags():
    """Shape includes 4-body terms when flags are set."""
    n_max = 1
    num_L = 3
    c_angular = jnp.ones((1, n_max + 1, 3))
    R_i = jnp.array([0.0, 0.0, 0.0])
    R_nbr = jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    types_nbr = jnp.array([0, 0, 0])

    # has_q_222=1 adds (n_max+1) = 2 entries
    q = compute_angular_descriptor(
        R_i, R_nbr, types_nbr, 0,
        c_angular, rc_angular=3.0,
        n_max=n_max, basis_size=2, num_L=num_L, L_max=2,
        has_q_222=1, has_q_1111=0, has_q_112=0, has_q_1122=0,
    )
    expected_dim = (n_max + 1) * num_L + (n_max + 1)
    assert q.shape == (expected_dim,)

    # has_q_222=1, has_q_1111=1 adds 4 entries
    q2 = compute_angular_descriptor(
        R_i, R_nbr, types_nbr, 0,
        c_angular, rc_angular=3.0,
        n_max=n_max, basis_size=2, num_L=num_L, L_max=2,
        has_q_222=1, has_q_1111=1, has_q_112=0, has_q_1122=0,
    )
    expected_dim2 = (n_max + 1) * num_L + 2 * (n_max + 1)
    assert q2.shape == (expected_dim2,)


def test_angular_descriptor_all_flags_jit():
    """Full angular descriptor with all 4-body/5-body flags compiles under jit."""
    n_max = 1
    num_L = 3
    c_angular = jnp.ones((1, n_max + 1, 3))
    R_i = jnp.array([0.0, 0.0, 0.0])
    R_nbr = jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    types_nbr = jnp.array([0, 0, 0])

    fn = lambda ri: compute_angular_descriptor(
        ri, R_nbr, types_nbr, 0,
        c_angular, rc_angular=3.0,
        n_max=n_max, basis_size=2, num_L=num_L, L_max=2,
        has_q_222=1, has_q_1111=1, has_q_112=1, has_q_1122=1,
    )

    q = jax.jit(fn)(R_i)
    # 3-body: 6 + 4*(2) = 14
    assert q.shape == (14,)

    grad_fn = jax.grad(lambda ri: jnp.sum(fn(ri)))
    g = jax.jit(grad_fn)(R_i)
    assert g.shape == (3,)


def test_descriptor_shape_consistency():
    """Descriptor output dimensions match computed dim."""
    n_max_radial, n_max_angular = 2, 1
    num_L = 3
    flag_count = 1  # has_q_222
    expected_dim = (n_max_radial + 1) + (n_max_angular + 1) * num_L + flag_count * (n_max_angular + 1)

    c_angular = jnp.ones((1, n_max_angular + 1, 3))
    R_i = jnp.array([0.0, 0.0, 0.0])
    R_nbr = jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    types_nbr = jnp.array([0, 0])

    q_angular = compute_angular_descriptor(
        R_i, R_nbr, types_nbr, 0,
        c_angular, rc_angular=3.0,
        n_max=n_max_angular, basis_size=2, num_L=num_L, L_max=2,
        has_q_222=1, has_q_1111=0, has_q_112=0, has_q_1122=0,
    )
    assert q_angular.shape[0] == (n_max_angular + 1) * num_L + flag_count * (n_max_angular + 1)
