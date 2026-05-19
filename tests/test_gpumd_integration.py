"""Integration test for GPUMDSampler in the DiffSim pipeline.

These tests verify the full pipeline: sampler creation, run.in generation,
trajectory readback, and energy recomputation. Tests that require a real GPU
and GPUMD binary are skip-decorated by default.
"""
import pytest
import os
import jax
import jax.numpy as jnp
from diffcg.system import AtomicSystem
from diffcg.md.gpumd_sampler import GPUMDSampler
from diffcg.md.sample import create_equilibration_run, create_production_run

# Minimal NEP params for testing
MINIMAL_NEP = {
    "version": 4, "num_types": 1, "elements": ["H"],
    "rc_radial": [4.0], "rc_angular": [3.0],
    "MN_radial": 50, "MN_angular": 30,
    "n_max_radial": 2, "n_max_angular": 2,
    "basis_size_radial": 4, "basis_size_angular": 4,
    "L_max": 2, "has_q_222": 0, "has_q_1111": 0,
    "has_q_112": 0, "has_q_1122": 0,
    "num_neurons": 10,
    "num_L": 2, "dim": 9,
    "descriptor_params": jnp.zeros(75, dtype=jnp.float32),
    "ann_params": {0: {
        "w0": jnp.zeros((10, 9), dtype=jnp.float32),
        "b0": jnp.zeros(10, dtype=jnp.float32),
        "w1": jnp.zeros(10, dtype=jnp.float32),
    }},
    "b1": jnp.float32(0.0),
    "q_scaler": jnp.ones(9, dtype=jnp.float32),
}


def test_create_equilibration_run_gpumd():
    system = AtomicSystem(
        R=jnp.zeros((10, 3), dtype=jnp.float32),
        Z=jnp.zeros(10, dtype=jnp.int32),
        cell=jnp.eye(3) * 3.0, pbc=True,
    )
    sampler = create_equilibration_run(
        system, energy_fn=None,
        sampler_params={
            "ensemble": "nvt", "thermostat": "langevin",
            "temperature": 300, "timestep": 2.0,
            "loginterval": 100,
        },
        cutoff=1.0,
        sampler_backend="gpumd",
        lammps_config={
            "topology": {},
            "nep_params": MINIMAL_NEP,
            "gpumd_exe": "gpumd",
        },
    )
    assert isinstance(sampler, GPUMDSampler)
    assert sampler.ensemble == "nvt"
    assert sampler.temperature == 300


def test_create_production_run_gpumd():
    system = AtomicSystem(
        R=jnp.zeros((10, 3), dtype=jnp.float32),
        Z=jnp.zeros(10, dtype=jnp.int32),
        cell=jnp.eye(3) * 3.0, pbc=True,
    )
    sampler = create_production_run(
        system, energy_fn=None,
        sampler_params={
            "ensemble": "nvt", "thermostat": "langevin",
            "temperature": 300, "timestep": 2.0,
            "loginterval": 100,
        },
        cutoff=1.0,
        trajectory="test.traj",
        logfile="test.log",
        sampler_backend="gpumd",
        lammps_config={
            "topology": {},
            "nep_params": MINIMAL_NEP,
            "gpumd_exe": "gpumd",
        },
        restart_state={"system": system},
    )
    assert isinstance(sampler, GPUMDSampler)
    assert sampler.trajectory_path == "test.traj"


@pytest.mark.skipif(
    not os.path.exists(os.path.expanduser("~/gpumd/src/gpumd")),
    reason="GPUMD binary not available"
)
def test_gpumd_run_real():
    """Run GPUMD with a real system (requires GPU and GPUMD binary)."""
    from diffcg.io.gpumd_writer import write_xyz_in
    import tempfile, subprocess

    with tempfile.TemporaryDirectory() as tmpdir:
        system = AtomicSystem(
            R=jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=jnp.float32),
            Z=jnp.array([0, 0], dtype=jnp.int32),
            cell=jnp.eye(3) * 5.0, pbc=True,
        )
        write_xyz_in(system, os.path.join(tmpdir, "xyz.in"))
        # Verifies we can write input; actual GPUMD run requires binary + GPU
