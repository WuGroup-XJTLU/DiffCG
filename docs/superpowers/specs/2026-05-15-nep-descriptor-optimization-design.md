# NEP Descriptor Optimization — JAX Efficiency + Full Contraction

**Date:** 2026-05-15
**Scope:** `diffcg/nep/descriptor.py`

## Summary

Optimize `descriptor.py` following JAX best practices (vmap, lax.scan, einsum) and implement the full C3B/C4B/C5B angular descriptor contraction matching GPUMD's NEP4. The two goals are complementary: vectorized code compiles better under `jax.jit` and traces cleanly under `jax.grad`.

## Architecture Overview

Same public API, same signatures. Internal rewrites:

| Function | Change | Why |
|---|---|---|
| `cosine_cutoff` | None | Already pure JAX |
| `chebyshev_polynomials` | list-append + `jnp.stack` → `jax.lax.scan` | Eliminate Python list tracing; compiles under vmap/grad |
| `_legendre` | Python for-loop → `jax.lax.scan` recurrence, emit all orders 0…L_max in one pass | Same as above |
| `compute_radial_descriptor` | Keep `for tj in range(num_types)`; fuse inner broadcast into single einsum | num_types is small (1–3); einsum avoids intermediate (n_max+1, M, basis_size+1) allocation |
| `compute_angular_descriptor` | Full rewrite: C3B/C4B/C5B/C4B2/C5B2 contraction with `has_q_*` flags; vectorize L/n loops via einsum; `_legendre` returns all orders from one scan | Correct physics + JAX efficiency |
| `compute_nep_descriptor` | `vmap` over atom index; remove `for i in range(N)` with `.at[i].set()` | Fixes #1 bottleneck — tracing cost scales with N without vmap |

## Radial Descriptor Changes

Minimal. The core logic is correct.

**`chebyshev_polynomials` via `lax.scan`:**
- Carry = `(T_{k-1}, T_{k-2})`, emit each `T_k`
- Removes Python list tracking through JAX tracing

**`compute_radial_descriptor` inner loop:**
- Replace 3-step (where + broadcast + sum) with single `einsum` combining masked T, fc, and coefficients
- Signature unchanged, returns `(n_max+1,)`

## Angular Descriptor Changes

### Output Structure

Flattened `(n_max+1) * num_L` → contracted, concatenated 1D array:

| Component | Dimension | Condition |
|---|---|---|
| 3-body (C3B) | `num_L_3body` (80 entries in C3B, mapped per-L) | Always |
| 4-body q²²² (C4B) | `n_max_angular + 1` | `has_q_222 == 1` |
| 4-body q¹¹¹¹ (C4B2) | `n_max_angular + 1` | `has_q_1111 == 1` |
| 5-body q¹¹² (C5B) | `n_max_angular + 1` | `has_q_112 == 1` |
| 5-body q¹¹²² (C5B2) | `n_max_angular + 1` | `has_q_1122 == 1` |

`dim_angular = num_L_3body + (flag_count) * (n_max_angular + 1)`

### 3-Body Contraction

- Compute Legendre polynomials `P_1` to `P_{L_max}` via single `lax.scan` on `cos_theta`
- Einsum over (neighbors, n, L, coefficients) dimensions — collapses L and n loops into one operation
- Uses C3B (80 entries) mapped to (l1, l2, L) triples

### 4-Body Terms (q²²², q¹¹¹¹)

Pattern: sum over 3 distinct neighbors (j, k, l) of `fc_j*fc_k*fc_l * T_n(s_j)*T_n(s_k)*T_n(s_l) * P_L(angles) * C4B_contraction`.

- Precompute pairwise `P_2` or `P_1` matrix `(M, M)` as scalar
- C4B contraction reduces 5 entries → scalar factor applied to `fc * T_n` triples
- Einsum: `"j,k,l,jn,kn,ln->n"` with masking to exclude diagonal terms

Implement as helper: `_contract_4body(fc, T, P_L_pair, coef_C4, n_max) -> (n_max+1,)`

### 5-Body Terms (q¹¹², q¹¹²²)

Pattern: sum over 4 distinct neighbors (j, k, l, m), same radial structure but 4-tuple.

- C5B (3 entries) / C5B2 (10 entries) contraction mixed with C3B/C4B2 for angular part
- Einsum: `"j,k,l,m,jn,kn,ln,mn->n"` with masking
- Main cost is 4-tuple summation; acceptable for typical GPUMD neighbor counts (M < 100)

Implement as helper: `_contract_5body(fc, T, P_L_pairs, coef_C5, n_max) -> (n_max+1,)`

### `_legendre` via `lax.scan`

Carry = `(P_{l-1}, P_{l-2})`, emit `P_0` through `P_{L_max}`. All orders computed in one compiled loop.

## compute_nep_descriptor — vmap

Current Python loop:
```python
for i in range(N):
    ...
    descriptors = descriptors.at[i].set(q_i / q_scaler)
```

Replaced with a `vmap`-compatible per-atom function and one `jax.vmap` call. Parameter reshaping (from flat `c_descriptor` to per-type-pair arrays) stays in Python since `num_types` is static — the reshape runs once at trace time, not per-atom.

## Testing

1. **Regression**: Radial descriptor values remain bitwise-identical for same inputs
2. **Contraction correctness**: Simple geometries (equilateral triangle → q²²² nonzero; collinear → known angle; far distance → all zero)
3. **JAX compatibility**: `jax.jit`, `jax.grad`, `jax.jit(jax.grad(...))` all compile without error
4. **Shape verification**: Output dimension matches expected `dim = (n_max_radial+1) + dim_angular` for each `has_q_*` flag combination

## Error Handling

No new validation. Inputs come from trusted internal caller (`build_nep_energy_fn`). Boundary conditions handled by existing `jnp.where` patterns (r >= rc → cutoff zero, r ≈ 0 → valid mask).
