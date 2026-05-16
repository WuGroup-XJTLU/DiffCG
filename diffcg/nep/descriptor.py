"""JAX implementation of NEP radial and angular descriptors.

Based on GPUMD's NEP4 descriptor (Phys. Rev. B 104, 104309, 2021).
Matches the CUDA kernels in nep_small_box.cuh.
"""

import jax
import jax.numpy as jnp
from diffcg.nep.constants import C3B, C4B, C5B


def cosine_cutoff(r: jnp.ndarray, rc: jnp.ndarray) -> jnp.ndarray:
    """f_c(r) = 0.5 * (1 + cos(pi * r / rc)), zero beyond rc."""
    ratio = r / rc
    return jnp.where(r < rc, 0.5 * (1.0 + jnp.cos(jnp.pi * ratio)), 0.0)


def chebyshev_polynomials(x: jnp.ndarray, n_max: int) -> jnp.ndarray:
    """Compute T_0(x) through T_{n_max}(x) using recurrence.
    Args: x: (...) scaled to [-1, 1]. Returns: (..., n_max+1).
    """
    T = [jnp.ones_like(x)]
    if n_max >= 1:
        T.append(x)
    for _ in range(2, n_max + 1):
        T.append(2.0 * x * T[-1] - T[-2])
    return jnp.stack(T, axis=-1)


def compute_radial_descriptor(
    R_i: jnp.ndarray,
    R_neighbors: jnp.ndarray,
    types_neighbors: jnp.ndarray,
    t_i: int,
    c_radial: jnp.ndarray,
    rc_radial: float,
    n_max: int,
    basis_size: int,
) -> jnp.ndarray:
    """Compute radial descriptor q^{n}_i for a single atom.

    q^n_i = sum_{j in nbrs} sum_k c^{n,k}_{ti,tj} * T_k(s_ij) * fc(r_ij)
    where s_ij = 2*r_ij/rc - 1 maps r in [0, rc] to [-1, 1].

    Args:
        R_i: (3,) position of center atom
        R_neighbors: (M, 3) positions of neighbors
        types_neighbors: (M,) types of neighbors
        t_i: type of center atom
        c_radial: (num_types, n_max+1, basis_size+1) radial params
        rc_radial: cutoff radius for this center type
        n_max: radial expansion order
        basis_size: number of Chebyshev basis functions (0-indexed)

    Returns: (n_max+1,) radial descriptor
    """
    r_ij = jnp.linalg.norm(R_neighbors - R_i, axis=-1)  # (M,)
    s_ij = 2.0 * r_ij / rc_radial - 1.0  # scale to [-1, 1]
    fc = cosine_cutoff(r_ij, rc_radial)  # (M,)
    T = chebyshev_polynomials(s_ij, basis_size)  # (M, basis_size+1)

    result = jnp.zeros(n_max + 1)
    for tj in range(c_radial.shape[0]):
        mask_tj = (types_neighbors == tj)
        T_masked = jnp.where(mask_tj[:, None], T, 0.0)  # (M, basis_size+1)
        fc_masked = jnp.where(mask_tj, fc, 0.0)  # (M,)
        # c_radial[tj]: (n_max+1, basis_size+1)
        # T_masked: (M, basis_size+1) -> broadcast to (n_max+1, M, basis_size+1)
        weighted = fc_masked[None, :, None] * T_masked[None, :, :]  # (1, M, basis_size+1)
        c_t = c_radial[tj]  # (n_max+1, basis_size+1)
        result += jnp.sum(c_t[:, None, :] * weighted, axis=(1, 2))  # (n_max+1,)
    return result


def _legendre(l: int, x):
    """Legendre polynomial P_l(x)."""
    if l == 0:
        return jnp.ones_like(x)
    if l == 1:
        return x
    P_prev2 = jnp.ones_like(x)
    P_prev = x
    for n in range(2, l + 1):
        P = ((2 * n - 1) * x * P_prev - (n - 1) * P_prev2) / n
        P_prev2 = P_prev
        P_prev = P
    return P_prev


def compute_angular_descriptor(
    R_i: jnp.ndarray,
    R_neighbors: jnp.ndarray,
    types_neighbors: jnp.ndarray,
    t_i: int,
    c_angular: jnp.ndarray,
    rc_angular: float,
    n_max: int,
    basis_size: int,
    num_L: int,
    L_max: int,
    has_q_222: int,
    has_q_1111: int,
    has_q_112: int,
    has_q_1122: int,
) -> jnp.ndarray:
    """Compute angular descriptor q^{nl}_i for a single atom (vectorized).

    Fully JAX-compatible: no Python if/for on traced values.
    Uses MxM pair matrices for cos_theta and masking.

    Returns: (n_max+1) * num_L angular descriptors (flattened).
    """
    M = R_neighbors.shape[0]
    dim_angular = (n_max + 1) * num_L
    rc = rc_angular

    # Per-neighbor quantities: (M,)
    r_ij_vec = R_neighbors - R_i  # (M, 3)
    r_ij = jnp.linalg.norm(r_ij_vec, axis=-1)  # (M,)
    valid = (r_ij < rc) & (r_ij >= 1e-10)  # (M,)

    s_ij = 2.0 * r_ij / rc - 1.0
    T_all = chebyshev_polynomials(s_ij, n_max)  # (M, n_max+1)
    fc_all = cosine_cutoff(r_ij, rc)  # (M,)

    # Pairwise quantities: (M, M)
    dot_prods = jnp.einsum("jd,kd->jk", r_ij_vec, r_ij_vec)  # (M, M)
    r_prods = r_ij[:, None] * r_ij[None, :]  # (M, M)
    cos_theta = jnp.where(r_prods > 1e-20, dot_prods / r_prods, 0.0)
    cos_theta = jnp.clip(cos_theta, -1.0, 1.0)

    # Pair mask: both valid and j != k
    pair_mask = valid[:, None] * valid[None, :] * (1.0 - jnp.eye(M))  # (M, M)
    fc_pair = fc_all[:, None] * fc_all[None, :]  # (M, M)

    q_angular = jnp.zeros(dim_angular)
    l_offset = 0
    for l in range(num_L):
        P_l = _legendre(l, cos_theta)  # (M, M)
        for n in range(n_max + 1):
            T_pair = T_all[:, n, None] * T_all[None, :, n]  # (M, M)
            contrib = pair_mask * fc_pair * T_pair * P_l
            idx = l_offset + n
            q_angular = q_angular.at[idx].set(jnp.sum(contrib))
        l_offset += n_max + 1

    return q_angular


def compute_nep_descriptor(
    positions: jnp.ndarray,
    Z: jnp.ndarray,
    cell: jnp.ndarray,
    nbrs,
    c_descriptor: jnp.ndarray,
    rc_radial: jnp.ndarray,
    rc_angular: jnp.ndarray,
    n_max_radial: int,
    n_max_angular: int,
    basis_size_radial: int,
    basis_size_angular: int,
    num_L: int,
    L_max: int,
    has_q_222: int,
    has_q_1111: int,
    has_q_112: int,
    has_q_1122: int,
    q_scaler: jnp.ndarray,
) -> jnp.ndarray:
    """Compute full NEP descriptor for all atoms.

    Returns: (N, dim) normalized descriptor,
        where dim = (n_max_radial+1) + (n_max_angular+1)*num_L.
    """
    N = positions.shape[0]
    num_types = rc_radial.shape[0]
    dim = (n_max_radial + 1) + (n_max_angular + 1) * num_L

    # Reshape c_descriptor into per-type-pair arrays
    radial_param_size = (n_max_radial + 1) * (basis_size_radial + 1)
    angular_param_size = (n_max_angular + 1) * (basis_size_angular + 1)
    block_size = radial_param_size + angular_param_size

    c_radial_params = jnp.zeros((num_types, num_types, n_max_radial + 1, basis_size_radial + 1))
    c_angular_params = jnp.zeros((num_types, num_types, n_max_angular + 1, basis_size_angular + 1))

    for ti in range(num_types):
        for tj in range(num_types):
            offset = (ti * num_types + tj) * block_size
            c_radial_params = c_radial_params.at[ti, tj].set(
                c_descriptor[offset:offset + radial_param_size].reshape(n_max_radial + 1, basis_size_radial + 1)
            )
            c_angular_params = c_angular_params.at[ti, tj].set(
                c_descriptor[offset + radial_param_size:offset + block_size].reshape(n_max_angular + 1, basis_size_angular + 1)
            )

    # Ghost atom at "infinite" distance so padded neighbors contribute zero
    # via the cutoff function. Avoids boolean indexing on traced arrays.
    ghost_pos = jnp.array([1e10, 1e10, 1e10], dtype=positions.dtype)
    R_extended = jnp.vstack([positions, ghost_pos])
    Z_extended = jnp.append(Z, jnp.int32(0))

    descriptors = jnp.zeros((N, dim))
    for i in range(N):
        t_i = Z[i]  # JAX scalar; avoid int() for trace compatibility
        nbr_idx = nbrs.idx[i]
        nbr_mask = nbr_idx < N
        nbr_idx = jnp.where(nbr_mask, nbr_idx, N)  # pad with ghost atom index

        R_nbr = R_extended[nbr_idx]
        types_nbr = Z_extended[nbr_idx]

        q_radial = compute_radial_descriptor(
            positions[i], R_nbr, types_nbr, t_i,
            c_radial_params[t_i], rc_radial[t_i],
            n_max_radial, basis_size_radial,
        )
        q_angular = compute_angular_descriptor(
            positions[i], R_nbr, types_nbr, t_i,
            c_angular_params[t_i], rc_angular[t_i],
            n_max_angular, basis_size_angular, num_L, L_max,
            has_q_222, has_q_1111, has_q_112, has_q_1122,
        )

        q_i = jnp.concatenate([q_radial, q_angular])
        descriptors = descriptors.at[i].set(q_i / q_scaler)

    return descriptors
