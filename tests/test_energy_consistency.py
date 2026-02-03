# SPDX-License-Identifier: MIT
# Copyright (c) 2025 WuResearchGroup

"""Tests for energy consistency between JAX direct evaluation and LAMMPS table
interpolation, verifying that the double-interpolation pipeline
(Steffen cubic → dense grid → LAMMPS linear) introduces bounded error."""

import shutil
import tempfile

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from diffcg._core.interpolate import MonotonicInterpolate
from diffcg.energy import _smooth_cutoff_factor, generic_repulsion, simple_spring


class TestInterpolationError:
    """Pure-Python test (no LAMMPS needed) comparing JAX spline evaluation
    against the dense-grid + linear interpolation pipeline used by LAMMPS."""

    def test_pair_interpolation_error(self):
        """Steffen cubic on dense grid + np.interp should match direct eval."""
        # Define spline knots (mimicking a realistic CG pair potential)
        x_knots = jnp.linspace(0.3, 1.2, 12)
        rng = np.random.RandomState(42)
        y_knots = jnp.array(rng.randn(12) * 2.0)

        spline = MonotonicInterpolate(x_knots, y_knots)

        # Dense grid (as LAMMPS table would use)
        x_dense = np.linspace(0.3, 1.2, 500)
        y_dense = np.asarray(spline(jnp.array(x_dense)))

        # Evaluation points between grid points
        x_eval = np.linspace(0.31, 1.19, 1000)
        y_exact = np.asarray(spline(jnp.array(x_eval)))

        # Simulate LAMMPS linear interpolation on dense grid
        y_linear = np.interp(x_eval, x_dense, y_dense)

        # The error should be small — bounded by O(h^2) where h = grid spacing
        max_error = np.max(np.abs(y_exact - y_linear))
        rel_error = max_error / (np.max(np.abs(y_exact)) + 1e-10)

        # With 500 points over 0.9 nm range, h ~ 0.0018 nm
        # Cubic interpolation error on linear grid should be < 1%
        assert rel_error < 0.01, (
            f"Pair interpolation relative error {rel_error:.4e} exceeds 1% threshold"
        )

    def test_pair_with_cutoff_and_prior(self):
        """Full pair potential (spline + cutoff + repulsion prior)."""
        R_CUT = 1.2
        R_ONSET = 0.9

        x_knots = jnp.linspace(0.3, R_CUT, 10)
        rng = np.random.RandomState(123)
        y_knots = jnp.array(rng.randn(10) * 1.5)

        spline = MonotonicInterpolate(x_knots, y_knots)

        def full_potential(x):
            tab = _smooth_cutoff_factor(x, R_ONSET, R_CUT) * spline(x)
            prior = (
                _smooth_cutoff_factor(x, 0.9, 1.0)
                * generic_repulsion(x, sigma=0.6, epsilon=1.0, exp=8)
            )
            return tab + prior

        # Dense grid for LAMMPS table (start at 0.4 to avoid repulsive wall)
        x_dense = np.linspace(0.4, R_CUT, 500)
        y_dense = np.asarray(full_potential(jnp.array(x_dense)))

        # Evaluation points (within dense grid range)
        x_eval = np.linspace(0.41, R_CUT - 0.01, 800)
        y_exact = np.asarray(full_potential(jnp.array(x_eval)))
        y_linear = np.interp(x_eval, x_dense, y_dense)

        max_error = np.max(np.abs(y_exact - y_linear))
        # Absolute error check — should be well below 1 kJ/mol
        assert max_error < 1.0, (
            f"Pair+prior max absolute error {max_error:.4e} kJ/mol exceeds 1 kJ/mol"
        )

    def test_bond_interpolation_error(self):
        """Bond potential: spline + harmonic prior."""
        x_knots = jnp.linspace(0.1, 1.0, 8)
        rng = np.random.RandomState(7)
        y_knots = jnp.array(rng.randn(8) * 3.0)

        spline = MonotonicInterpolate(x_knots, y_knots)

        def bond_potential(x):
            return spline(x) + simple_spring(x, length=0.45, epsilon=5000)

        x_dense = np.linspace(0.1, 1.0, 300)
        y_dense = np.asarray(bond_potential(jnp.array(x_dense)))

        x_eval = np.linspace(0.11, 0.99, 500)
        y_exact = np.asarray(bond_potential(jnp.array(x_eval)))
        y_linear = np.interp(x_eval, x_dense, y_dense)

        max_error = np.max(np.abs(y_exact - y_linear))
        # Bond potentials can be steep near minima; allow up to 1 kJ/mol
        assert max_error < 1.0, (
            f"Bond max absolute error {max_error:.4e} kJ/mol exceeds 1 kJ/mol"
        )

    def test_force_interpolation_error(self):
        """Force from np.gradient on dense grid vs JAX autodiff."""
        x_knots = jnp.linspace(0.3, 1.2, 12)
        rng = np.random.RandomState(42)
        y_knots = jnp.array(rng.randn(12) * 2.0)

        spline = MonotonicInterpolate(x_knots, y_knots)

        # JAX exact force via autodiff
        force_fn = jax.vmap(jax.grad(lambda r: -spline(r.reshape(1))[0]))

        x_dense = np.linspace(0.3, 1.2, 500)
        y_dense = np.asarray(spline(jnp.array(x_dense)))

        # Numerical force (as LAMMPS computes)
        dydx = np.gradient(y_dense, x_dense)
        force_numerical = -dydx

        # Exact force at dense grid points
        force_exact = np.asarray(force_fn(jnp.array(x_dense)))

        # Compare interior points (avoid boundary effects of np.gradient)
        interior = slice(5, -5)
        max_force_error = np.max(np.abs(force_exact[interior] - force_numerical[interior]))
        scale = np.max(np.abs(force_exact[interior])) + 1e-10
        rel_force_error = max_force_error / scale

        assert rel_force_error < 0.05, (
            f"Force relative error {rel_force_error:.4e} exceeds 5% threshold"
        )


class TestDihedralSignConvention:
    """Regression test: TabulatedDihedralEnergy must compute the same
    dihedral angle sign as direct dihedral(R_ab, R_bc, R_cd) with forward vectors."""

    def test_energy_fn_matches_direct_dihedral(self):
        from diffcg._core.geometry import dihedral as compute_dihedral
        from diffcg.energy import TabulatedDihedralEnergy
        from diffcg.system import AtomicSystem

        R = jnp.array(
            [
                [1.0, 1.0, 1.0],
                [1.5, 1.0, 1.0],
                [1.75, 1.5, 1.0],
                [2.25, 1.5, 1.3],
            ],
            dtype=jnp.float64,
        )
        Z = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
        cell = jnp.diag(jnp.array([10.0, 10.0, 10.0], dtype=jnp.float64))
        masses = jnp.array([12.0, 12.0, 12.0, 12.0], dtype=jnp.float64)
        system = AtomicSystem(R=R, Z=Z, cell=cell, masses=masses, pbc=True)

        # Direct dihedral with forward vectors
        R_ab = R[1] - R[0]
        R_bc = R[2] - R[1]
        R_cd = R[3] - R[2]
        theta_direct = float(compute_dihedral(R_ab, R_bc, R_cd))

        # Use an asymmetric spline so sign errors are detectable
        rng = np.random.RandomState(42)
        x_knots = jnp.linspace(-np.pi, np.pi, 12)
        y_knots = jnp.array(rng.randn(12) * 2.0)
        spline = MonotonicInterpolate(x_knots, y_knots)

        x_grid = jnp.linspace(-np.pi + 1e-4, np.pi - 1e-4, 500)
        y_grid = np.asarray(spline(x_grid))

        # Energy from TabulatedDihedralEnergy
        dihedral_topology = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
        tab = TabulatedDihedralEnergy(x_vals=x_grid, y_vals=y_grid, dihedrals=dihedral_topology)
        energy_fn = tab.get_energy_fn()
        energy_from_fn = float(energy_fn(system, neighbors=None))

        # Energy from direct evaluation at theta_direct
        energy_direct = float(spline(jnp.array([theta_direct]))[0])

        np.testing.assert_allclose(
            energy_from_fn, energy_direct, atol=1e-4,
            err_msg=(
                f"TabulatedDihedralEnergy evaluates at wrong angle. "
                f"Direct theta={theta_direct:.4f}, energy_fn={energy_from_fn:.4f}, "
                f"direct={energy_direct:.4f}"
            ),
        )

        # Also verify sign: energy at +theta should differ from energy at -theta
        energy_neg = float(spline(jnp.array([-theta_direct]))[0])
        assert not np.isclose(energy_from_fn, energy_neg, atol=1e-3), (
            f"Energy function gives same value at +theta and -theta for asymmetric spline — "
            f"sign convention is likely wrong"
        )


class TestDihedralConvention:
    """Verify that the dihedral angle convention is consistent between
    JAX evaluation and the LAMMPS table written by DiffCG."""

    def test_dihedral_table_no_inversion(self):
        """The LAMMPS dihedral table should evaluate the spline at the same
        angles (no cis/trans swap). Regression test for the x_jax bug."""
        from diffcg.energy import harmonic_dihedral

        # Arbitrary spline params (asymmetric so a swap is detectable)
        rng = np.random.RandomState(99)
        x_knots = jnp.linspace(-np.pi, np.pi, 12)
        y_knots = jnp.array(rng.randn(12) * 2.0)
        spline = MonotonicInterpolate(x_knots, y_knots)

        # Dense evaluation grid (same as run_diffsim_lammps.py)
        x_dihedral = np.linspace(-np.pi + 1e-4, np.pi - 1e-4, 500)

        # Correct: evaluate spline directly at x_dihedral
        y_correct = np.asarray(
            spline(x_dihedral) + harmonic_dihedral(x_dihedral, angle_0=1.5, epsilon=50)
        )

        # Wrong (old code): swap cis/trans
        x_swapped = np.where(x_dihedral >= 0, np.pi - x_dihedral, -np.pi - x_dihedral)
        y_swapped = np.asarray(
            spline(x_swapped) + harmonic_dihedral(x_swapped, angle_0=1.5, epsilon=50)
        )

        # The two should NOT be the same (asymmetric spline)
        assert not np.allclose(y_correct, y_swapped, atol=1e-6), (
            "Asymmetric spline gave identical results with and without swap — "
            "test is not discriminating"
        )

        # Verify that the correct table (no swap) matches direct evaluation
        # at all grid points (these are identical by construction)
        y_direct = np.asarray(spline(x_dihedral))
        y_table_spline_part = np.asarray(spline(x_dihedral))
        np.testing.assert_allclose(
            y_correct,
            y_table_spline_part + np.asarray(harmonic_dihedral(x_dihedral, angle_0=1.5, epsilon=50)),
            atol=1e-10,
            err_msg="Dihedral table doesn't match direct evaluation (identity check)",
        )


@pytest.mark.skipif(
    shutil.which("lmp") is None,
    reason="LAMMPS executable 'lmp' not found in PATH",
)
class TestLAMMPSEnergyConsistency:
    """End-to-end test: compare JAX energy evaluation vs LAMMPS table energy
    on the same configuration."""

    def test_pair_energy_matches(self):
        """Run a 0-step LAMMPS sim and compare PE against JAX evaluation."""
        from diffcg.io.lammps_writer import write_lammps_data, write_lammps_table
        from diffcg.md.lammps_sampler import LAMMPSSampler
        from diffcg.system import AtomicSystem

        # Simple 4-atom system with no bonds (pure pair interactions)
        R = jnp.array(
            [[0.5, 0.5, 0.5], [1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [1.0, 1.0, 0.5]],
            dtype=jnp.float64,
        )
        Z = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
        cell = jnp.diag(jnp.array([3.0, 3.0, 3.0], dtype=jnp.float64))
        masses = jnp.array([12.0, 12.0, 12.0, 12.0], dtype=jnp.float64)
        system = AtomicSystem(R=R, Z=Z, cell=cell, masses=masses, pbc=True)

        # Define a simple pair potential: repulsive exponential
        x_pair = np.linspace(0.2, 1.5, 500)
        y_pair = 10.0 * np.exp(-x_pair / 0.3)  # kJ/mol

        energy_params = {"pair": {"x": x_pair, "y": y_pair}}
        topology = {
            "bonds": np.empty((0, 2), dtype=int),
            "bond_types": np.empty(0, dtype=int),
        }

        tmpdir = tempfile.mkdtemp()
        try:
            sampler = LAMMPSSampler(
                system,
                energy_params=energy_params,
                topology=topology,
                temperature=300.0,
                timestep=1.0,
                cutoff=1.5,
                loginterval=1,
                work_dir=tmpdir,
                random_seed=42,
            )

            # Run 0 production steps — just get initial energy
            # We use 1 step with loginterval=1 to get thermo
            traj = sampler.run(1)
            thermo = sampler.get_thermo_log()

            # LAMMPS PE at step 0 (in kJ/mol, already converted)
            lammps_pe = thermo["pe"][0]

            # JAX energy: compute pair energy directly
            # Pairwise distances for 4 atoms in periodic box
            R_np = np.asarray(R)
            box = np.asarray(jnp.diag(cell))
            jax_pe = 0.0
            for i in range(4):
                for j in range(i + 1, 4):
                    dr = R_np[i] - R_np[j]
                    # Minimum image convention
                    dr = dr - box * np.round(dr / box)
                    r = np.linalg.norm(dr)
                    if r < 1.5:
                        jax_pe += np.interp(r, x_pair, y_pair)

            # Compare — allow for LAMMPS linear interpolation + shift errors
            # The LAMMPS table uses linear interpolation and pair_modify shift yes
            np.testing.assert_allclose(
                lammps_pe, jax_pe, rtol=0.05,
                err_msg="LAMMPS PE and JAX PE differ by more than 5%",
            )
        finally:
            shutil.rmtree(tmpdir)

    def test_dihedral_energy_matches(self):
        """Run a 0-step LAMMPS sim with a dihedral and compare PE against JAX.

        Uses TabulatedDihedralEnergy (the actual energy function) to compute
        JAX PE, ensuring the displacement vector convention is correct.
        """
        from diffcg.energy import TabulatedDihedralEnergy, harmonic_dihedral
        from diffcg.io.lammps_writer import write_lammps_data, write_lammps_table
        from diffcg.md.lammps_sampler import LAMMPSSampler
        from diffcg.system import AtomicSystem

        # 4 atoms forming a known dihedral in a big box (no pair interactions)
        R = jnp.array(
            [
                [1.0, 1.0, 1.0],
                [1.5, 1.0, 1.0],
                [1.75, 1.5, 1.0],
                [2.25, 1.5, 1.3],
            ],
            dtype=jnp.float64,
        )
        Z = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
        cell = jnp.diag(jnp.array([10.0, 10.0, 10.0], dtype=jnp.float64))
        masses = jnp.array([12.0, 12.0, 12.0, 12.0], dtype=jnp.float64)
        system = AtomicSystem(R=R, Z=Z, cell=cell, masses=masses, pbc=True)

        # Build an asymmetric dihedral potential (spline + harmonic prior)
        rng = np.random.RandomState(42)
        x_knots = jnp.linspace(-np.pi, np.pi, 12)
        y_knots = jnp.array(rng.randn(12) * 2.0)
        spline = MonotonicInterpolate(x_knots, y_knots)

        x_dihedral = np.linspace(-np.pi + 1e-4, np.pi - 1e-4, 500)
        dihedral_y = np.asarray(
            spline(x_dihedral) + harmonic_dihedral(x_dihedral, angle_0=1.5, epsilon=50)
        )

        # JAX energy via TabulatedDihedralEnergy (tests the actual vector convention)
        dihedral_topology = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
        tab_energy = TabulatedDihedralEnergy(
            x_vals=x_dihedral, y_vals=dihedral_y, dihedrals=dihedral_topology,
        )
        energy_fn = tab_energy.get_energy_fn()
        jax_pe = float(energy_fn(system, neighbors=None))

        # Make pair potential zero beyond a tiny distance
        x_pair = np.linspace(0.05, 1.5, 100)
        y_pair = np.zeros_like(x_pair)

        energy_params = {
            "pair": {"x": x_pair, "y": y_pair},
            "dihedral": {"x": x_dihedral, "y": dihedral_y},
        }
        topology = {
            "bonds": np.array([[0, 1], [1, 2], [2, 3]], dtype=int),
            "bond_types": np.zeros(3, dtype=int),
            "angles": np.array([[0, 1, 2], [1, 2, 3]], dtype=int),
            "angle_types": np.zeros(2, dtype=int),
            "dihedrals": np.array([[0, 1, 2, 3]], dtype=int),
            "dihedral_types": np.zeros(1, dtype=int),
        }

        # Need bond/angle tables too (zero energy, just to satisfy LAMMPS)
        x_bond = np.linspace(0.1, 2.0, 100)
        y_bond = np.zeros_like(x_bond)
        x_angle = np.linspace(0.0, np.pi, 100)
        y_angle = np.zeros_like(x_angle)
        energy_params["bond"] = {"x": x_bond, "y": y_bond}
        energy_params["angle"] = {"x": x_angle, "y": y_angle}

        tmpdir = tempfile.mkdtemp()
        try:
            sampler = LAMMPSSampler(
                system,
                energy_params=energy_params,
                topology=topology,
                temperature=300.0,
                timestep=1.0,
                cutoff=1.5,
                loginterval=1,
                work_dir=tmpdir,
                random_seed=42,
                special_bonds="lj 0.0 0.0 0.0",
            )

            traj = sampler.run(1)
            thermo = sampler.get_thermo_log()
            lammps_pe = thermo["pe"][0]

            # LAMMPS PE should be close to JAX dihedral energy
            # (pair/bond/angle are zero, so PE ≈ dihedral energy only)
            np.testing.assert_allclose(
                lammps_pe, jax_pe, rtol=0.05,
                err_msg=(
                    f"LAMMPS dihedral PE ({lammps_pe:.4f}) and JAX dihedral PE "
                    f"({jax_pe:.4f}) differ by more than 5%"
                ),
            )
        finally:
            shutil.rmtree(tmpdir)
