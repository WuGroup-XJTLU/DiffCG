import jax.numpy as jnp
from diffcg.system import AtomicSystem
from diffcg.md.gpumd_sampler import GPUMDSampler

MINIMAL_NEP = {
    "version": 4, "num_types": 1, "elements": ["H"],
    "rc_radial": [4.0], "rc_angular": [3.0],
    "MN_radial": 50, "MN_angular": 30,
    "n_max_radial": 2, "n_max_angular": 2,
    "basis_size_radial": 4, "basis_size_angular": 4,
    "L_max": 2, "has_q_222": 0, "has_q_1111": 0,
    "has_q_112": 0, "has_q_1122": 0,
    "num_neurons": 10,
    "num_L": 3, "dim": 9,
    "descriptor_params": jnp.zeros(75, dtype=jnp.float32),
    "ann_params": {0: {
        "w0": jnp.zeros((10, 9), dtype=jnp.float32),
        "b0": jnp.zeros(10, dtype=jnp.float32),
        "w1": jnp.zeros(10, dtype=jnp.float32),
    }},
    "b1": jnp.float32(0.0),
    "q_scaler": jnp.ones(9, dtype=jnp.float32),
}


def test_generate_run_in():
    system = AtomicSystem(
        R=jnp.zeros((10, 3), dtype=jnp.float32),
        Z=jnp.zeros(10, dtype=jnp.int32),
        cell=jnp.eye(3) * 3.0,
        pbc=True,
    )
    sampler = GPUMDSampler(
        system,
        topology={},
        nep_params=MINIMAL_NEP,
        ensemble="nvt",
        thermostat="langevin",
        temperature=300.0,
        timestep=2.0,
        friction=100.0,
        gpumd_exe="gpumd",
    )
    run_in = sampler._generate_run_in(1000)
    assert "potential nep.txt" in run_in
    assert "velocity 300" in run_in
    assert "nvt_lan 300.0 300.0 100.0" in run_in
    assert "time_step 2.0" in run_in
    assert "dump_thermo 100" in run_in
    assert "run 1000" in run_in


def test_gpumd_sampler_requires_nep_params():
    system = AtomicSystem(
        R=jnp.zeros((10, 3), dtype=jnp.float32),
        Z=jnp.zeros(10, dtype=jnp.int32),
    )
    try:
        GPUMDSampler(system, topology={})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "nep_params" in str(e)
