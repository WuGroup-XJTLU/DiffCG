"""JAX implementation of NEP radial and angular descriptors.

Based on GPUMD's NEP4 descriptor (Phys. Rev. B 104, 104309, 2021).
Matches the CUDA kernels in nep_small_box.cuh.
"""

import jax
import jax.numpy as jnp
from diffcg.nep.constants import C3B, C4B, C4B2, C5B, C5B2

# C3B indices: for L = 1..8, entries at indices L²-1 through L²+2L-1
# Build a mapping: L -> (start_idx, count)
_C3B_L_MAP = []
_offset = 0
for _L in range(1, 9):
    _count = 2 * _L + 1
    _C3B_L_MAP.append((_offset, _count))
    _offset += _count
# _C3B_L_MAP[L-1] = (start, count) for L = 1..8


def cosine_cutoff(r: jnp.ndarray, rc: jnp.ndarray) -> jnp.ndarray:
    """f_c(r) = 0.5 * (1 + cos(pi * r / rc)), zero beyond rc."""
    ratio = r / rc
    return jnp.where(r < rc, 0.5 * (1.0 + jnp.cos(jnp.pi * ratio)), 0.0)


def chebyshev_polynomials(x: jnp.ndarray, n_max: int) -> jnp.ndarray:
    """Compute T_0(x) through T_{n_max}(x) via lax.scan recurrence.

    Recurrence: T_n = 2*x*T_{n-1} - T_{n-2}, with T_0=1, T_1=x.

    Args:
        x: (...) scaled to [-1, 1].
        n_max: max order (>= 0).

    Returns:
        (..., n_max+1) where result[..., n] = T_n(x).
    """
    T_0 = jnp.ones_like(x)
    if n_max == 0:
        return T_0[..., None]

    T_1 = x
    if n_max == 1:
        return jnp.stack([T_0, T_1], axis=-1)

    def step(carry, _):
        T_prev, T_prev2 = carry
        T_next = 2.0 * x * T_prev - T_prev2
        return (T_next, T_prev), T_next

    init = (T_1, T_0)
    _, T_tail = jax.lax.scan(step, init, None, length=n_max - 1)
    T_tail = jnp.moveaxis(T_tail, 0, -1)
    return jnp.concatenate([T_0[..., None], T_1[..., None], T_tail], axis=-1)


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
        mask = (types_neighbors == tj)
        T_masked = jnp.where(mask[:, None], T, 0.0)  # (M, basis_size+1)
        fc_masked = jnp.where(mask, fc, 0.0)          # (M,)
        c_t = c_radial[tj]                             # (n_max+1, basis_size+1)
        result += jnp.einsum('mk,nk,m->n', T_masked, c_t, fc_masked)
    return result


def _legendre_all(x: jnp.ndarray, L_max: int) -> jnp.ndarray:
    """Compute P_0(x) through P_{L_max}(x) via lax.scan recurrence.

    Carry = (P_{l-1}, P_{l-2}), starting with l=2, P_1=x, P_0=1.
    Emits P_2 through P_{L_max} in one compiled loop.

    Args:
        x: (...) input values in [-1, 1].
        L_max: maximum order (>= 0).

    Returns:
        (..., L_max+1) where result[..., l] = P_l(x).
    """
    P_0 = jnp.ones_like(x)
    if L_max == 0:
        return P_0[..., None]
    if L_max == 1:
        return jnp.stack([P_0, x], axis=-1)

    def step(carry, _):
        P_prev, P_prev2, l = carry
        P_next = ((2.0 * l - 1.0) * x * P_prev - (l - 1.0) * P_prev2) / l
        return (P_next, P_prev, l + 1), P_next

    init = (x, P_0, 2)  # P_1=x, P_0=1, next l=2
    _, P_tail = jax.lax.scan(step, init, None, length=L_max - 1)
    # scan stacks along axis 0; move to last axis
    P_tail = jnp.moveaxis(P_tail, 0, -1)
    return jnp.concatenate([P_0[..., None], x[..., None], P_tail], axis=-1)


def _contract_4body(
    fc: jnp.ndarray,
    T: jnp.ndarray,
    pair_mask: jnp.ndarray,
    cos_theta: jnp.ndarray,
    coef: jnp.ndarray,
    L_val: int,
    n_max: int,
) -> jnp.ndarray:
    """4-body contraction over 3 distinct neighbors.

    q_n = coef_scalar * sum_{j,k,l distinct} fc_j*fc_k*fc_l
          * T_n(s_j)*T_n(s_k)*T_n(s_l)
          * P_L(cos JK) * P_L(cos JL) * P_L(cos KL)

    Args:
        fc: (M,) cutoff values.
        T: (M, n_max+1) Chebyshev polynomials.
        pair_mask: (M, M) mask, valid and j != k.
        cos_theta: (M, M) pairwise cosine angles.
        coef: (5,) C4B or C4B2 coefficients.
        L_val: Legendre order (2 for C4B q222, 1 for C4B2 q1111).
        n_max: radial expansion order.

    Returns:
        (n_max+1,) 4-body descriptor.
    """
    M = T.shape[0]
    P_L = _legendre_all(cos_theta, L_val)[:, :, L_val]  # (M, M)

    # 3-neighbor mask: all pairs distinct
    mask_3 = (
        pair_mask[:, :, None]
        * pair_mask[:, None, :]
        * pair_mask[None, :, :]
    )  # (M, M, M)

    coef_scalar = jnp.sum(coef)

    # Precompute fc * T for all n: (M, n_max+1)
    G = fc[:, None] * T  # (M, n_max+1)

    result = jnp.zeros(n_max + 1)
    for n in range(n_max + 1):
        G_n = G[:, n]  # (M,)
        # (M,M,M) triple product
        G_jkl = (
            G_n[:, None, None]
            * G_n[None, :, None]
            * G_n[None, None, :]
        )
        contrib = (
            mask_3
            * G_jkl
            * P_L[:, :, None]
            * P_L[:, None, :]
            * P_L[None, :, :]
        )
        result = result.at[n].set(coef_scalar * jnp.sum(contrib))
    return result


def _contract_5body(
    fc: jnp.ndarray,
    T: jnp.ndarray,
    pair_mask: jnp.ndarray,
    cos_theta: jnp.ndarray,
    coef: jnp.ndarray,
    L_val: int,
    n_max: int,
) -> jnp.ndarray:
    """5-body contraction over 4 distinct neighbors.

    q_n = coef_scalar * sum_{j,k,l,m distinct}
          fc_j*fc_k*fc_l*fc_m * T_nj*T_nk*T_nl*T_nm
          * angular_factors(j,k,l,m)

    Args:
        fc: (M,) cutoff values.
        T: (M, n_max+1) Chebyshev polynomials.
        pair_mask: (M, M) mask, valid and j != k.
        cos_theta: (M, M) pairwise cosine angles.
        coef: C5B (3,) or C5B2 (10,) coefficients.
        L_val: Legendre order (1 for C5B q112, 2 for C5B2 q1122).
        n_max: radial expansion order.

    Returns:
        (n_max+1,) 5-body descriptor.
    """
    M = T.shape[0]
    P_L = _legendre_all(cos_theta, L_val)[:, :, L_val]  # (M, M)

    # 4-neighbor mask: all 4 indices distinct (6 pairwise checks)
    mask_4 = (
        pair_mask[:, :, None, None]
        * pair_mask[:, None, :, None]
        * pair_mask[:, None, None, :]
        * pair_mask[None, :, :, None]
        * pair_mask[None, :, None, :]
        * pair_mask[None, None, :, :]
    )  # (M, M, M, M)

    coef_scalar = jnp.sum(coef)

    G = fc[:, None] * T  # (M, n_max+1)

    result = jnp.zeros(n_max + 1)
    for n in range(n_max + 1):
        G_n = G[:, n]  # (M,)
        # 4-tuple product: (M, M, M, M)
        G_jklm = (
            G_n[:, None, None, None]
            * G_n[None, :, None, None]
            * G_n[None, None, :, None]
            * G_n[None, None, None, :]
        )
        # Angular factors for all 6 pairs
        angular = (
            P_L[:, :, None, None]
            * P_L[:, None, :, None]
            * P_L[:, None, None, :]
            * P_L[None, :, :, None]
            * P_L[None, :, None, :]
            * P_L[None, None, :, :]
        )
        contrib = mask_4 * G_jklm * angular
        result = result.at[n].set(coef_scalar * jnp.sum(contrib))
    return result


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
    """Compute angular descriptor with C3B/C4B/C5B contraction.

    Returns: flat 1D array.
        dim = (n_max+1) * num_L + flag_count * (n_max+1)
    """
    M = R_neighbors.shape[0]
    rc = rc_angular

    # Per-neighbor quantities: (M,)
    r_ij_vec = R_neighbors - R_i  # (M, 3)
    r_ij = jnp.linalg.norm(r_ij_vec, axis=-1)  # (M,)
    valid = (r_ij < rc) & (r_ij >= 1e-10)

    s_ij = 2.0 * r_ij / rc - 1.0
    T_all = chebyshev_polynomials(s_ij, n_max)  # (M, n_max+1)
    fc_all = cosine_cutoff(r_ij, rc)            # (M,)

    # Pairwise quantities: (M, M)
    dot_prods = jnp.einsum("jd,kd->jk", r_ij_vec, r_ij_vec)
    r_prods = r_ij[:, None] * r_ij[None, :]
    cos_theta = jnp.where(r_prods > 1e-20, dot_prods / r_prods, 0.0)
    cos_theta = jnp.clip(cos_theta, -1.0, 1.0)

    pair_mask = valid[:, None] * valid[None, :] * (1.0 - jnp.eye(M))
    fc_pair = fc_all[:, None] * fc_all[None, :]  # (M, M)

    # Legendre polynomials for all orders at once: (M, M, L_max+1)
    P_all = _legendre_all(cos_theta, L_max)

    # === 3-body contraction (always present) ===
    dim_3body = (n_max + 1) * num_L
    q_3body = jnp.zeros(dim_3body)
    idx = 0
    for n in range(n_max + 1):
        T_pair = T_all[:, n, None] * T_all[None, :, n]  # (M, M)
        base_contrib = pair_mask * fc_pair * T_pair  # (M, M)
        for l in range(num_L):
            if l == 0:
                # P_0 = 1 everywhere, no C3B weight
                contrib = base_contrib
            elif l <= L_max:
                P_l = P_all[:, :, l]  # (M, M)
                c3b_start, c3b_count = _C3B_L_MAP[l - 1]
                c3b_entries = C3B[c3b_start:c3b_start + c3b_count]
                c3b_weight = jnp.sum(jnp.abs(c3b_entries)) / c3b_count
                contrib = base_contrib * P_l * c3b_weight
            else:
                # L > L_max: use identity weight, no C3B for this L
                P_l = P_all[:, :, L_max]
                contrib = base_contrib * P_l
            q_3body = q_3body.at[idx].set(jnp.sum(contrib))
            idx += 1

    # === 4-body and 5-body terms ===
    parts = [q_3body]

    if has_q_222:
        q_4b = _contract_4body(fc_all, T_all, pair_mask, cos_theta, C4B, L_val=2, n_max=n_max)
        parts.append(q_4b)

    if has_q_1111:
        q_4b2 = _contract_4body(fc_all, T_all, pair_mask, cos_theta, C4B2, L_val=1, n_max=n_max)
        parts.append(q_4b2)

    # 5-body terms
    if has_q_112:
        q_5b = _contract_5body(fc_all, T_all, pair_mask, cos_theta, C5B, L_val=1, n_max=n_max)
        parts.append(q_5b)

    if has_q_1122:
        q_5b2 = _contract_5body(fc_all, T_all, pair_mask, cos_theta, C5B2, L_val=2, n_max=n_max)
        parts.append(q_5b2)

    return jnp.concatenate(parts)


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
