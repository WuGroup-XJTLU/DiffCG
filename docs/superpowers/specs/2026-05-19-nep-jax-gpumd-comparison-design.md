# NEP JAX vs GPUMD Consistency Checker Design

**Date:** 2026-05-19  
**Status:** Approved

## Goal

Verify that `diffcg`'s JAX NEP implementation produces the same total energy and per-atom forces as the GPUMD reference implementation for identical NEP parameters and atomic configurations.

## Approach

Standalone comparison script (Option A from brainstorming).

## Script Location

`scripts/compare_nep_jax_gpumd.py`

## Workflow

1. **Setup**  
   Generate a small random `AtomicSystem` (single atom type, e.g. 20 atoms in a cubic box). Build a `nep_params` dict with random trainable weights using the hyperparameters defined in `water_nep_common.py`.

2. **JAX Calculation**  
   - Build the NEP energy function via `diffcg.nep.build_nep_energy_fn(nep_params)`.  
   - Compute total energy: `energy = energy_fn(system, neighbors)`.  
   - Compute per-atom forces via JAX autodiff: `forces = -jax.grad(...)(system.R)`.

3. **GPUMD Calculation**  
   - Write `xyz.in` using `diffcg.io.gpumd_writer.write_xyz_in`.  
   - Write `nep.txt` using `diffcg.io.nep.write_nep`.  
   - Write a minimal `run.in` with `potential nep.txt`, `dump_force 1`, `dump_thermo 1`, `run 1`.  
   - Execute the GPUMD binary at `/home/zhenghaowu/gpumd/src/gpumd`.  
   - Read `thermo.out` (total energy in eV) and `force.out` (forces in eV/Å).

4. **Unit Conversion**  
   Convert GPUMD outputs to `diffcg` internal units (kJ/mol and kJ/mol/nm) using conversion constants from `diffcg._core.units`, or convert JAX outputs to eV and eV/Å.

5. **Comparison**  
   - Absolute and relative energy difference.  
   - Max absolute difference in force components.  
   - Print a concise summary table.

6. **Pass/Fail**  
   Define tolerances and report PASS/FAIL:  
   - Energy absolute error < 1e-4 eV/atom  
   - Force absolute error < 1e-3 eV/Å  
   Optionally loop over multiple random configurations to increase confidence.

## Error Handling

- If the GPUMD binary is missing or the subprocess returns a non-zero exit code, print a clear error message and exit.
- If output files (`thermo.out`, `force.out`) are not produced, print a diagnostic message.

## Success Criteria

The script runs without errors and reports PASS when JAX and GPUMD energies/forces agree within the specified tolerances.
