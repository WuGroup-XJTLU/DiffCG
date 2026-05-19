# Fix diffcg NEP to Match GPUMD 5.3 — Design

**Date:** 2026-05-19  
**Status:** Approved

## Goal

Bring `diffcg`'s JAX NEP implementation into numerical consistency with GPUMD 5.3 by correcting the descriptor computation and I/O unit conversion.

## Background

`diffcg`'s NEP descriptor diverges from GPUMD in four ways:

1. **Radial basis mapping:** `diffcg` uses a linear Chebyshev argument `s = 2r/rc - 1`; GPUMD uses a quadratic mapping `x = 2*(r/rc - 1)^2 - 1` and a shifted basis `(T_k(x) + 1) / 2`.
2. **Angular descriptor structure:** `diffcg` computes pairwise `P_l(cos_theta)` contractions (Behler-Parrinello-style) and includes an extra `l=0` term. GPUMD accumulates real spherical harmonics (`accumulate_s` + `find_q`) and omits `l=0`.
3. **Missing `c_angular` usage:** `diffcg` ignores the angular descriptor coefficients `c_angular`; GPUMD weights neighbor contributions by `gn = sum_k c[n,k] * basis_k(r)`.
4. **`q_scaler` convention:** `diffcg` divides by `q_scaler`; GPUMD multiplies.
5. **`write_nep` units:** `diffcg` writes cutoffs in nm; GPUMD expects Angstroms.

## Approach

In-place fixes to `diffcg/nep/descriptor.py` and `diffcg/io/nep.py`, plus test updates.

## Files to Modify

- `diffcg/nep/descriptor.py` — radial basis, angular descriptor rewrite, `q_scaler` fix
- `diffcg/io/nep.py` — cutoff unit conversion
- `tests/test_nep_descriptor.py` — updated shapes and assertions
- `tests/test_gpumd_integration.py` — corrected `MINIMAL_NEP` dimensions
- `tests/test_nep_gpumd_consistency.py` — new regression test (from comparison script)

## Detailed Changes

### `diffcg/nep/descriptor.py`

#### Radial Basis
- In `compute_radial_descriptor`:
  - Replace `s_ij = 2r/rc - 1` with `x_ij = 2*(r_ij/rc - 1)^2 - 1`.
  - Replace `T = chebyshev_polynomials(s, basis_size)` with shifted basis `basis = (chebyshev_polynomials(x, basis_size) + 1) / 2`.
  - Continue multiplying by `fc` as before.

#### Angular Basis
- In `compute_angular_descriptor`:
  - Compute the same shifted radial basis `basis_j = (T_n(x_j) + 1)/2 * fc_j` for each neighbor.
  - Weight by `c_angular` to get `gn_j = sum_k c[n,k] * basis_k(r_j)`.
  - Accumulate real spherical harmonic components `s[abc] = sum_j Y_{l,abc}(r_j) * gn_j`.
  - Compute 3-body invariants `q[n,l] = sum_{abc} C3B[abc] * s[abc]^2`.
  - Loop `l = 1 ... L_max` (no `l=0`).
  - Pass the same `gn` to 4-body and 5-body contraction helpers.

#### `q_scaler`
- In `_per_atom_descriptor`, change `return jnp.concatenate([q_radial, q_angular]) / q_scaler` to `* q_scaler`.

### `diffcg/io/nep.py`

- In `write_nep`, before writing the `cutoff` line, convert `rc_radial` and `rc_angular` from nm to Angstrom using `NM_TO_ANGSTROM`.

### Tests

- `test_nep_descriptor.py`: update expected descriptor dimensions (e.g., `L_max=2` gives `dim=9`, not 12). Update any hardcoded reference values.
- `test_gpumd_integration.py`: update `MINIMAL_NEP` so `dim` matches corrected formula.
- `test_nep_gpumd_consistency.py`: add the comparison script as a pytest test, skipped if GPUMD binary is missing.

## Success Criteria

1. `scripts/compare_nep_jax_gpumd.py` (or the equivalent test) reports `PASS` for energy and forces within tolerance.
2. All existing pytest tests pass after updates.
3. `write_nep` produces a `nep.txt` that GPUMD interprets with the correct cutoff in Angstroms.

## Backward Compatibility

Breaking. Any existing trained NEP weights will produce different energies/forces because the descriptor vector changes structurally. This is acceptable per user approval.
