# diffcg packaging finalization and push

## Goal

Commit pending changes and push to origin so users can `pip install diffcg` and use all samplers (JAX-MD, fastMD, GPUMD) without manual setup.

## Architecture decision

fastMD and GPUMD are **vendored** inside diffcg (not separate packages). The single `pip install diffcg` compiles everything via scikit-build-core + CMake. `_binaries.py` resolves compiled binary paths at runtime by looking next to the package.

## Changes to commit

### Commit 1: NEP descriptor and energy table improvements

- `diffcg/_core/interpolate.py` — add `MonotonicInterpolate.derivative()` for analytic spline derivatives
- `diffcg/energy.py` — add `_smooth_cutoff_derivative()`, analytic force computation in `to_lammps()` for `GenericRepulsionEnergy` and `TabulatedPairEnergy`, increase table resolution to 5000 points
- `diffcg/nep/descriptor.py` — NEP descriptor optimizations
- `diffcg/nep/network.py` — NEP network optimizations

### Commit 2: Packaging metadata

- `diffcg.egg-info/requires.txt` — add `jax>=0.4` and `jax-md>=0.2.7`
- `diffcg.egg-info/SOURCES.txt` — updated file listing
- `diffcg.egg-info/PKG-INFO` — updated package info

### Push

Push all commits to `origin/main` at `https://github.com/WuGroup-XJTLU/DiffCG.git`.

## Verification

- `pip install -e .` succeeds (CMake compiles vendored CUDA source)
- `from diffcg._binaries import get_fastmd_path; print(get_fastmd_path())` resolves correctly
- No regressions in existing tests
