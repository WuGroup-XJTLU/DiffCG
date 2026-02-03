"""Tests for thermodynamic output from custom integrators.

Verifies that:
1. Custom Langevin integrator reproduces JAX-MD Langevin trajectories
2. Custom NVE integrator reproduces JAX-MD NVE trajectories
3. NVE total energy is conserved
4. NVT mean temperature matches target
5. Thermo dict has correct keys and shapes
6. CSV log file is written correctly
"""

import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from jax_md import space, partition, simulate, energy as jaxmd_energy, quantity

from diffcg.md.integrators import (
    nve as custom_nve,
    nvt_langevin as custom_nvt_langevin,
    NVEState,
    NVTLangevinState,
)
from diffcg.md.jaxmd_sampler import JAXMDSampler, MDResult, KB_KJ_MOL
from diffcg.md.sample import MolecularDynamics
from diffcg.system import System, AtomicSystem


# ---------------------------------------------------------------------------
# Helpers: simple Lennard-Jones system for testing
# ---------------------------------------------------------------------------

def _default_dtype():
    """Return the default float dtype (respects jax_enable_x64)."""
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def _make_lj_system(N=16, box_size=3.0, seed=42):
    """Create a small periodic LJ system on a grid to avoid overlaps."""
    dtype = _default_dtype()
    n_side = int(np.ceil(N ** (1.0 / 3.0)))
    spacing = box_size / n_side
    coords = []
    for ix in range(n_side):
        for iy in range(n_side):
            for iz in range(n_side):
                if len(coords) < N:
                    coords.append([ix * spacing + 0.5 * spacing,
                                   iy * spacing + 0.5 * spacing,
                                   iz * spacing + 0.5 * spacing])
    R = jnp.array(coords[:N], dtype=dtype)
    Z = jnp.ones(N, dtype=jnp.int32)
    cell = jnp.eye(3, dtype=dtype) * box_size
    return R, Z, cell, box_size


def _make_jaxmd_lj_energy(box_size):
    """Create a JAX-MD LJ energy function (pair-wise, for testing)."""
    dtype = _default_dtype()
    box = jnp.eye(3, dtype=dtype) * box_size
    displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=False)

    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        box=box,
        r_cutoff=1.5,
        capacity_multiplier=1.5,
        format=partition.Dense,
    )

    energy_fn = jaxmd_energy.lennard_jones_pair(displacement_fn, sigma=0.5, epsilon=0.5, r_cutoff=1.5)

    return energy_fn, shift_fn, neighbor_fn, displacement_fn


# ---------------------------------------------------------------------------
# Test: custom NVE reproduces JAX-MD NVE trajectory
# ---------------------------------------------------------------------------

class TestNVETrajectoryEquivalence:
    """Custom NVE should produce physically correct velocity Verlet dynamics."""

    def test_nve_positions_match_when_same_initial_state(self):
        """Given identical initial conditions, custom NVE should match JAX-MD NVE."""
        R, Z, cell, box_size = _make_lj_system(box_size=5.0)
        energy_fn, shift_fn, neighbor_fn, _ = _make_jaxmd_lj_energy(box_size)
        neighbor = neighbor_fn.allocate(R)

        dt = 0.001
        kT = 0.5
        key = random.PRNGKey(0)
        steps = 20

        # Initialize via JAX-MD to get a reference state
        jaxmd_init, jaxmd_apply = simulate.nve(energy_fn, shift_fn, dt)
        jaxmd_state = jaxmd_init(key, R, kT, neighbor=neighbor)

        # Create custom NVE integrator and manually set same initial conditions
        custom_init, custom_apply = custom_nve(energy_fn, shift_fn, dt)
        custom_state = custom_init(key, R, kT=kT, neighbor=neighbor)
        # Override with JAX-MD's initial momentum so both start identically
        custom_state = custom_state.set(momentum=jaxmd_state.momentum)

        # Run both side-by-side with synchronized neighbor lists
        jaxmd_nb = neighbor_fn.allocate(R)
        custom_nb = neighbor_fn.allocate(R)
        for _ in range(steps):
            jaxmd_state = jaxmd_apply(jaxmd_state, neighbor=jaxmd_nb)
            jaxmd_nb = neighbor_fn.update(jaxmd_state.position, jaxmd_nb)

            custom_state = custom_apply(custom_state, neighbor=custom_nb)
            custom_nb = neighbor_fn.update(custom_state.position, custom_nb)

        np.testing.assert_allclose(
            custom_state.position, jaxmd_state.position, atol=1e-4,
            err_msg="Custom NVE trajectory diverged from JAX-MD NVE"
        )

    def test_nve_has_potential_energy(self):
        """Custom NVE state should carry potential energy."""
        R, Z, cell, box_size = _make_lj_system()
        energy_fn, shift_fn, neighbor_fn, _ = _make_jaxmd_lj_energy(box_size)
        neighbor = neighbor_fn.allocate(R)

        init_fn, apply_fn = custom_nve(energy_fn, shift_fn, 0.001)
        state = init_fn(random.PRNGKey(0), R, kT=0.5, neighbor=neighbor)

        assert hasattr(state, 'potential_energy')
        assert state.potential_energy.shape == ()
        # PE should be finite
        assert np.isfinite(float(state.potential_energy))


# ---------------------------------------------------------------------------
# Test: custom Langevin reproduces JAX-MD Langevin trajectory
# ---------------------------------------------------------------------------

class TestLangevinTrajectoryEquivalence:
    """Custom Langevin should produce same dynamics given same initial state."""

    def test_langevin_positions_match_when_same_initial_state(self):
        """Given identical initial conditions and RNG, trajectories match."""
        R, Z, cell, box_size = _make_lj_system(box_size=5.0)
        energy_fn, shift_fn, neighbor_fn, _ = _make_jaxmd_lj_energy(box_size)
        neighbor = neighbor_fn.allocate(R)

        dt = 0.001
        kT = 0.5
        gamma = 1.0
        key = random.PRNGKey(0)
        steps = 20

        # Initialize via JAX-MD
        jaxmd_init, jaxmd_apply = simulate.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma)
        jaxmd_state = jaxmd_init(key, R, neighbor=neighbor)

        # Initialize custom with same momentum and rng
        custom_init, custom_apply = custom_nvt_langevin(energy_fn, shift_fn, dt, kT, gamma)
        custom_state = custom_init(key, R, neighbor=neighbor)
        custom_state = custom_state.set(
            momentum=jaxmd_state.momentum,
            rng=jaxmd_state.rng,
        )

        # Run both with synchronized neighbor lists
        jaxmd_nb = neighbor_fn.allocate(R)
        custom_nb = neighbor_fn.allocate(R)
        for _ in range(steps):
            jaxmd_state = jaxmd_apply(jaxmd_state, neighbor=jaxmd_nb)
            jaxmd_nb = neighbor_fn.update(jaxmd_state.position, jaxmd_nb)

            custom_state = custom_apply(custom_state, neighbor=custom_nb)
            custom_nb = neighbor_fn.update(custom_state.position, custom_nb)

        np.testing.assert_allclose(
            custom_state.position, jaxmd_state.position, atol=1e-4,
            err_msg="Custom Langevin trajectory diverged from JAX-MD Langevin"
        )

    def test_langevin_has_potential_energy(self):
        """Custom Langevin state should carry potential energy."""
        R, Z, cell, box_size = _make_lj_system()
        energy_fn, shift_fn, neighbor_fn, _ = _make_jaxmd_lj_energy(box_size)
        neighbor = neighbor_fn.allocate(R)

        init_fn, apply_fn = custom_nvt_langevin(energy_fn, shift_fn, 0.001, 0.5, 1.0)
        state = init_fn(random.PRNGKey(0), R, neighbor=neighbor)

        assert hasattr(state, 'potential_energy')
        assert state.potential_energy.shape == ()
        assert np.isfinite(float(state.potential_energy))


# ---------------------------------------------------------------------------
# Test: NVE energy conservation
# ---------------------------------------------------------------------------

class TestNVEEnergyConservation:
    """NVE total energy should be well conserved."""

    def test_total_energy_conserved(self):
        R, Z, cell, box_size = _make_lj_system(box_size=5.0)
        energy_fn, shift_fn, neighbor_fn, _ = _make_jaxmd_lj_energy(box_size)
        neighbor = neighbor_fn.allocate(R)

        dt = 0.0005  # Small timestep for good conservation
        kT = 0.3
        key = random.PRNGKey(1)
        steps = 200

        init_fn, apply_fn = custom_nve(energy_fn, shift_fn, dt)
        state = init_fn(key, R, kT=kT, neighbor=neighbor)

        etotals = []
        for _ in range(steps):
            state = apply_fn(state, neighbor=neighbor)
            neighbor = neighbor_fn.update(state.position, neighbor)
            ke = quantity.kinetic_energy(momentum=state.momentum, mass=state.mass)
            pe = state.potential_energy
            etotals.append(float(ke + pe))

        etotals = np.array(etotals)
        # Relative fluctuation should be small
        std_e = np.std(etotals)
        mean_e = np.abs(np.mean(etotals))
        if mean_e > 1e-6:
            rel_fluct = std_e / mean_e
            assert rel_fluct < 0.01, (
                f"NVE energy not conserved: relative fluctuation = {rel_fluct:.4f}"
            )
        else:
            assert std_e < 0.01, f"NVE energy std too large: {std_e:.6f}"


# ---------------------------------------------------------------------------
# Test: NVT mean temperature
# ---------------------------------------------------------------------------

class TestNVTTemperature:
    """NVT Langevin should sample near the target temperature."""

    def test_mean_temperature_near_target(self):
        N = 16
        box_size = 5.0  # Large box to avoid overlaps
        key = random.PRNGKey(2)
        R, Z, cell, box_size = _make_lj_system(N=N, box_size=box_size)

        energy_fn, shift_fn, neighbor_fn, _ = _make_jaxmd_lj_energy(box_size)
        neighbor = neighbor_fn.allocate(R)

        dt = 0.001
        kT = 1.0
        gamma = 1.0

        init_fn, apply_fn = custom_nvt_langevin(energy_fn, shift_fn, dt, kT, gamma)
        state = init_fn(key, R, neighbor=neighbor)

        temps = []
        # Equilibrate
        for _ in range(500):
            state = apply_fn(state, neighbor=neighbor)
            neighbor = neighbor_fn.update(state.position, neighbor)

        # Collect
        for _ in range(500):
            state = apply_fn(state, neighbor=neighbor)
            neighbor = neighbor_fn.update(state.position, neighbor)
            kT_measured = float(quantity.temperature(momentum=state.momentum, mass=state.mass))
            temps.append(kT_measured)

        mean_kT = np.mean(temps)
        # Mean kinetic temperature should be within 40% of target kT (small system)
        assert abs(mean_kT - kT) / kT < 0.4, (
            f"Mean kT {mean_kT:.4f} too far from target {kT}"
        )


# ---------------------------------------------------------------------------
# Test: MDResult thermo dict shape and keys
# ---------------------------------------------------------------------------

class TestThermoOutput:
    """JAXMDSampler.run() should return thermo with correct keys/shapes."""

    def _make_sampler_and_run(self, ensemble='nvt', thermostat='langevin', steps=50, save_frequency=1):
        N = 8
        R, Z, cell, box_size = _make_lj_system(N=N, box_size=3.0)

        # DiffCG-style energy function: (System, neighbors) -> scalar
        box = jnp.eye(3) * box_size
        displacement_fn, _ = space.periodic_general(box, fractional_coordinates=False)
        lj_energy = jaxmd_energy.lennard_jones_pair(displacement_fn, sigma=0.5, epsilon=0.5, r_cutoff=1.5)

        def diffcg_energy_fn(system, neighbors, **kwargs):
            return lj_energy(system.R, neighbor=neighbors, **kwargs)

        sampler = JAXMDSampler(
            energy_fn=diffcg_energy_fn,
            Z=Z,
            cell=cell,
            cutoff=1.5,
            ensemble=ensemble,
            thermostat=thermostat,
            temperature=300.0,
            timestep=1.0,
            capacity_multiplier=1.5,
        )
        result = sampler.run(R, steps=steps, save_frequency=save_frequency)
        return result, steps, save_frequency

    def test_thermo_keys(self):
        result, _, _ = self._make_sampler_and_run()
        assert result.thermo is not None
        for key in ['kinetic_energy', 'potential_energy', 'temperature', 'total_energy']:
            assert key in result.thermo, f"Missing thermo key: {key}"

    def test_thermo_shapes(self):
        steps = 50
        result, _, _ = self._make_sampler_and_run(steps=steps)
        for key in ['kinetic_energy', 'potential_energy', 'temperature', 'total_energy']:
            assert result.thermo[key].shape == (steps,), (
                f"Thermo {key} shape {result.thermo[key].shape} != ({steps},)"
            )

    def test_thermo_shapes_with_save_frequency(self):
        steps = 100
        save_freq = 10
        result, _, _ = self._make_sampler_and_run(steps=steps, save_frequency=save_freq)
        expected = steps // save_freq
        for key in ['kinetic_energy', 'potential_energy', 'temperature', 'total_energy']:
            assert result.thermo[key].shape == (expected,), (
                f"Thermo {key} shape {result.thermo[key].shape} != ({expected},)"
            )

    def test_nve_thermo(self):
        result, _, _ = self._make_sampler_and_run(ensemble='nve')
        assert result.thermo is not None
        assert 'total_energy' in result.thermo


# ---------------------------------------------------------------------------
# Test: CSV log output
# ---------------------------------------------------------------------------

class TestCSVLog:
    """MolecularDynamics should write per-frame CSV logs."""

    def test_csv_log_written(self):
        N = 8
        R, Z, cell, box_size = _make_lj_system(N=N, box_size=3.0)

        box = jnp.eye(3) * box_size
        displacement_fn, _ = space.periodic_general(box, fractional_coordinates=False)
        lj_energy = jaxmd_energy.lennard_jones_pair(displacement_fn, sigma=0.5, epsilon=0.5, r_cutoff=1.5)

        def diffcg_energy_fn(system, neighbors, **kwargs):
            return lj_energy(system.R, neighbor=neighbors, **kwargs)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            logfile = f.name

        try:
            system = AtomicSystem(
                R=R, Z=Z, cell=cell,
                masses=jnp.ones(N),
                pbc=jnp.array([True, True, True]),
            )
            md = MolecularDynamics(
                system,
                energy_fn=diffcg_energy_fn,
                ensemble='nvt',
                thermostat='langevin',
                temperature=300.0,
                timestep=1.0,
                cutoff=1.5,
                logfile=logfile,
                loginterval=5,
                capacity_multiplier=1.5,
            )
            md.run(50)

            with open(logfile) as f:
                lines = f.readlines()

            # Should have header comments + CSV header + data rows
            assert any(line.startswith('# JAX-MD') for line in lines)
            # Find CSV header
            csv_header_idx = None
            for i, line in enumerate(lines):
                if line.startswith('frame,'):
                    csv_header_idx = i
                    break
            assert csv_header_idx is not None, "CSV header not found"

            # Check data lines (50 steps / 5 save_freq = 10 frames)
            data_lines = [l for l in lines[csv_header_idx + 1:] if l.strip()]
            assert len(data_lines) == 10, f"Expected 10 data lines, got {len(data_lines)}"

            # Check first data line parses correctly
            parts = data_lines[0].strip().split(',')
            assert len(parts) == 5, f"Expected 5 CSV columns, got {len(parts)}"
            assert parts[0] == '0'  # First frame index
        finally:
            os.unlink(logfile)
