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
    x = jnp.clip(x, -1.0, 1.0)  # prevent overflow when fc=0 zeroes result anyway
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
    y = r_ij / rc_radial
    x = 2.0 * (y - 1.0) ** 2 - 1.0
    T = chebyshev_polynomials(x, basis_size)
    T = (T + 1.0) / 2.0
    fc = cosine_cutoff(r_ij, rc_radial)  # (M,)

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


def _accumulate_s_one_jax(x, y, z, gn, L, Z_COEFF_L):
    """JAX equivalent of GPUMD's accumulate_s_one.

    Args:
        x, y, z: (M,) normalized direction vectors.
        gn: (M,) neighbor weights for a single n.
        L: int, spherical harmonic order (1..8).
        Z_COEFF_L: (L+1, L+1) Z-coefficient array.

    Returns:
        s: (2*L+1,) accumulated real spherical harmonic components.
    """
    M = x.shape[0]
    z_pow = jnp.stack([z ** p for p in range(L + 1)], axis=0)  # (L+1, M)

    # Complex powers (x + i*y)^p for p=1..L
    rp = [x]
    ip = [y]
    for p in range(1, L):
        rp_prev, ip_prev = rp[-1], ip[-1]
        rp.append(rp_prev * x - ip_prev * y)
        ip.append(rp_prev * y + ip_prev * x)
    rp = jnp.stack(rp, axis=0)  # (L, M)
    ip = jnp.stack(ip, axis=0)  # (L, M)

    s = jnp.zeros(2 * L + 1)
    for n1 in range(L + 1):
        n2_start = (L + n1) % 2
        n2 = jnp.arange(n2_start, L - n1 + 1, 2)
        z_factor = jnp.sum(Z_COEFF_L[n1, n2][:, None] * z_pow[n2, :], axis=0)  # (M,)

        if n1 == 0:
            contrib = jnp.sum(z_factor * gn)
            s = s.at[0].add(contrib)
        else:
            contrib_real = jnp.sum(z_factor * gn * rp[n1 - 1, :])
            contrib_imag = jnp.sum(z_factor * gn * ip[n1 - 1, :])
            s = s.at[2 * n1 - 1].add(contrib_real)
            s = s.at[2 * n1].add(contrib_imag)
    return s


def _find_q_one_jax(s, L):
    """JAX equivalent of GPUMD's find_q_one.

    Computes: q = 2*sum_{k=1}^{2L} C3B[start+k]*s[start+k]^2 + C3B[start]*s[start]^2
    """
    start = L * L - 1
    num_terms = 2 * L + 1
    q_nonzero = jnp.sum(C3B[start + 1:start + num_terms] * s[start + 1:start + num_terms] ** 2)
    q_zero = C3B[start] * s[start] ** 2
    return 2.0 * q_nonzero + q_zero


def _contract_4body(
    gn: jnp.ndarray,
    pair_mask: jnp.ndarray,
    cos_theta: jnp.ndarray,
    coef: jnp.ndarray,
    L_val: int,
    n_max: int,
) -> jnp.ndarray:
    """4-body contraction over 3 distinct neighbors.

    q_n = coef_scalar * sum_{j,k,l distinct} gn_jn * gn_kn * gn_ln
          * P_L(cos JK) * P_L(cos JL) * P_L(cos KL)

    Args:
        gn: (M, n_max+1) angular weights.
        pair_mask: (M, M) mask, valid and j != k.
        cos_theta: (M, M) pairwise cosine angles.
        coef: (5,) C4B or C4B2 coefficients.
        L_val: Legendre order (2 for C4B q222, 1 for C4B2 q1111).
        n_max: radial expansion order.

    Returns:
        (n_max+1,) 4-body descriptor.
    """
    M = gn.shape[0]
    P_L = _legendre_all(cos_theta, L_val)[:, :, L_val]  # (M, M)

    # 3-neighbor mask: all pairs distinct
    mask_3 = (
        pair_mask[:, :, None]
        * pair_mask[:, None, :]
        * pair_mask[None, :, :]
    )  # (M, M, M)

    coef_scalar = jnp.sum(coef)

    G = gn  # (M, n_max+1)

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
    gn: jnp.ndarray,
    pair_mask: jnp.ndarray,
    cos_theta: jnp.ndarray,
    coef: jnp.ndarray,
    L_val: int,
    n_max: int,
) -> jnp.ndarray:
    """5-body contraction over 4 distinct neighbors.

    q_n = coef_scalar * sum_{j,k,l,m distinct}
          gn_jn * gn_kn * gn_ln * gn_mn
          * angular_factors(j,k,l,m)

    Args:
        gn: (M, n_max+1) angular weights.
        pair_mask: (M, M) mask, valid and j != k.
        cos_theta: (M, M) pairwise cosine angles.
        coef: C5B (3,) or C5B2 (10,) coefficients.
        L_val: Legendre order (1 for C5B q112, 2 for C5B2 q1122).
        n_max: radial expansion order.

    Returns:
        (n_max+1,) 5-body descriptor.
    """
    M = gn.shape[0]
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

    G = gn  # (M, n_max+1)

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
    """Compute angular descriptor with real spherical-harmonic accumulation.

    Returns: flat 1D array.
        dim = (n_max+1) * L_max + flag_count * (n_max+1)
    """
    M = R_neighbors.shape[0]
    rc = rc_angular

    r_ij_vec = R_neighbors - R_i
    r_ij = jnp.linalg.norm(r_ij_vec, axis=-1)
    valid = (r_ij < rc) & (r_ij >= 1e-10)

    # Shifted Chebyshev basis
    y = r_ij / rc
    x = 2.0 * (y - 1.0) ** 2 - 1.0
    T = chebyshev_polynomials(x, basis_size)
    basis = (T + 1.0) / 2.0 * cosine_cutoff(r_ij, rc)[:, None]

    # Angular weights gn (incorporate c_angular, iterating over neighbor types)
    gn = jnp.zeros((M, n_max + 1))
    for tj in range(c_angular.shape[0]):
        mask = (types_neighbors == tj)
        basis_masked = jnp.where(mask[:, None], basis, 0.0)
        c_t = c_angular[tj]
        gn += jnp.einsum('mk,nk->mn', basis_masked, c_t)
    gn = jnp.where(valid[:, None], gn, 0.0)

    # Normalized directions
    x_dir = jnp.where(valid, r_ij_vec[:, 0] / r_ij, 0.0)
    y_dir = jnp.where(valid, r_ij_vec[:, 1] / r_ij, 0.0)
    z_dir = jnp.where(valid, r_ij_vec[:, 2] / r_ij, 0.0)

    from diffcg.nep.constants import Z_COEFFICIENTS

    num_s_terms = (L_max + 1) ** 2 - 1
    s_all = jnp.zeros((n_max + 1, num_s_terms))

    for L in range(1, L_max + 1):
        Z_L = Z_COEFFICIENTS[L - 1]
        start = L * L - 1
        count = 2 * L + 1
        for n in range(n_max + 1):
            s_slice = _accumulate_s_one_jax(x_dir, y_dir, z_dir, gn[:, n], L, Z_L)
            s_all = s_all.at[n, start:start + count].add(s_slice)

    # 3-body invariants (GPUMD ordering: outer L, inner n)
    q_3body = jnp.zeros((n_max + 1) * L_max)
    idx = 0
    for L in range(1, L_max + 1):
        for n in range(n_max + 1):
            q_val = _find_q_one_jax(s_all[n], L)
            q_3body = q_3body.at[idx].set(q_val)
            idx += 1

    parts = [q_3body]

    # Pairwise quantities for 4-body/5-body terms
    dot_prods = jnp.einsum("jd,kd->jk", r_ij_vec, r_ij_vec)
    r_prods = r_ij[:, None] * r_ij[None, :]
    cos_theta = jnp.where(r_prods > 1e-20, dot_prods / r_prods, 0.0)
    cos_theta = jnp.clip(cos_theta, -1.0, 1.0)
    pair_mask = valid[:, None] * valid[None, :] * (1.0 - jnp.eye(M))

    if has_q_222:
        q_4b = _contract_4body(gn, pair_mask, cos_theta, C4B, L_val=2, n_max=n_max)
        parts.append(q_4b)
    if has_q_1111:
        q_4b2 = _contract_4body(gn, pair_mask, cos_theta, C4B2, L_val=1, n_max=n_max)
        parts.append(q_4b2)
    if has_q_112:
        q_5b = _contract_5body(gn, pair_mask, cos_theta, C5B, L_val=1, n_max=n_max)
        parts.append(q_5b)
    if has_q_1122:
        q_5b2 = _contract_5body(gn, pair_mask, cos_theta, C5B2, L_val=2, n_max=n_max)
        parts.append(q_5b2)

    return jnp.concatenate(parts)


def _per_atom_descriptor(
    R_i: jnp.ndarray,
    t_i: jnp.ndarray,
    nbr_idx: jnp.ndarray,
    nbr_mask: jnp.ndarray,
    R_extended: jnp.ndarray,
    Z_extended: jnp.ndarray,
    c_radial_params: jnp.ndarray,
    c_angular_params: jnp.ndarray,
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
    cell: jnp.ndarray,
    N: int,
) -> jnp.ndarray:
    """Compute descriptor for a single atom. Compatible with vmap."""
    nbr_idx = jnp.where(nbr_mask, nbr_idx, N)  # pad with ghost
    R_nbr = R_extended[nbr_idx]
    types_nbr = Z_extended[nbr_idx]

    # Apply MIC only to valid neighbors; skip ghost
    dr = R_nbr - R_i
    cell_inv = jnp.linalg.inv(cell)
    dr_frac = dr @ cell_inv.T
    dr_frac = dr_frac - jnp.round(dr_frac)
    dr_mic = dr_frac @ cell.T
    R_nbr_mic = jnp.where(nbr_mask[:, None], R_i + dr_mic, R_nbr)

    q_radial = compute_radial_descriptor(
        R_i, R_nbr_mic, types_nbr, t_i,
        c_radial_params[t_i], rc_radial[t_i],
        n_max_radial, basis_size_radial,
    )
    q_angular = compute_angular_descriptor(
        R_i, R_nbr_mic, types_nbr, t_i,
        c_angular_params[t_i], rc_angular[t_i],
        n_max_angular, basis_size_angular, num_L, L_max,
        has_q_222, has_q_1111, has_q_112, has_q_1122,
    )
    return jnp.concatenate([q_radial, q_angular]) * q_scaler


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
    """Compute full NEP descriptor for all atoms via vmap.

    Returns:
        (N, dim) normalized descriptor,
        dim = (n_max_radial+1) + (n_max_angular+1)*num_L
              + (has_q_222+has_q_1111+has_q_112+has_q_1122)*(n_max_angular+1)
    """
    N = positions.shape[0]
    num_types = rc_radial.shape[0]
    flag_count = has_q_222 + has_q_1111 + has_q_112 + has_q_1122
    # GPUMD: 3-body angular uses l=1..L_max (L_max terms per n), not l=0..L_max
    angular_3body_dim = (n_max_angular + 1) * L_max
    angular_extra_dim = flag_count * (n_max_angular + 1)
    dim = (n_max_radial + 1) + angular_3body_dim + angular_extra_dim

    # Reshape c_descriptor into per-type-pair arrays (static Python loop)
    radial_param_size = (n_max_radial + 1) * (basis_size_radial + 1)
    angular_param_size = (n_max_angular + 1) * (basis_size_angular + 1)
    block_size = radial_param_size + angular_param_size

    c_radial_params = jnp.zeros((num_types, num_types, n_max_radial + 1, basis_size_radial + 1))
    c_angular_params = jnp.zeros((num_types, num_types, n_max_angular + 1, basis_size_angular + 1))

    for ti in range(num_types):
        for tj in range(num_types):
            offset = (ti * num_types + tj) * block_size
            c_radial_params = c_radial_params.at[ti, tj].set(
                c_descriptor[offset:offset + radial_param_size].reshape(
                    n_max_radial + 1, basis_size_radial + 1
                )
            )
            c_angular_params = c_angular_params.at[ti, tj].set(
                c_descriptor[offset + radial_param_size:offset + block_size].reshape(
                    n_max_angular + 1, basis_size_angular + 1
                )
            )

    # Ghost atom at "infinite" distance
    ghost_pos = jnp.array([1e10, 1e10, 1e10], dtype=positions.dtype)
    R_extended = jnp.vstack([positions, ghost_pos])
    Z_extended = jnp.append(Z, jnp.int32(0))

    # Per-atom inputs: stack along leading axis for scan
    t_all = Z  # (N,)
    nbr_idx_all = nbrs.idx  # (N, max_nbrs)
    nbr_mask_all = nbr_idx_all < N  # (N, max_nbrs)

    def per_atom_scan_fn(carry, inputs):
        R_i, t_i, nbr_idx, nbr_mask = inputs
        desc_i = _per_atom_descriptor(
            R_i, t_i, nbr_idx, nbr_mask,
            R_extended, Z_extended,
            c_radial_params, c_angular_params,
            rc_radial, rc_angular,
            n_max_radial, n_max_angular,
            basis_size_radial, basis_size_angular,
            num_L, L_max,
            has_q_222, has_q_1111, has_q_112, has_q_1122,
            q_scaler, cell, N,
        )
        return carry, desc_i

    _, descriptors = jax.lax.scan(
        per_atom_scan_fn, None,
        (positions, t_all, nbr_idx_all, nbr_mask_all),
    )
    return descriptors
