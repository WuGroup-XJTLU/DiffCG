"""Regression test: diffcg NEP energy/forces must match GPUMD 5.3."""

import os
import subprocess
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffcg.nep import build_nep_energy_fn
from diffcg.system import AtomicSystem
from diffcg.io.gpumd_writer import write_xyz_in
from diffcg.io.nep import write_nep
from diffcg._core.neighborlist import jaxmd_neighbor_list
from diffcg._core.units import EV_TO_KJMOL, NM_TO_ANGSTROM

GPUMD_EXE = "/home/zhenghaowu/gpumd/src/gpumd"


@pytest.mark.skipif(not os.path.exists(GPUMD_EXE), reason="GPUMD binary not available")
def test_nep_energy_force_consistency():
    NUM_ATOMS = 20
    BOX_SIZE_NM = 2.0
    CUTOFF_NM = 0.6

    NEP_ARCH = {
        "version": 4,
        "num_types": 1,
        "elements": ["H"],
        "rc_radial": [CUTOFF_NM],
        "rc_angular": [CUTOFF_NM],
        "MN_radial": 50,
        "MN_angular": 30,
        "n_max_radial": 2,
        "n_max_angular": 2,
        "basis_size_radial": 4,
        "basis_size_angular": 4,
        "L_max": 2,
        "has_q_222": 0,
        "has_q_1111": 0,
        "has_q_112": 0,
        "has_q_1122": 0,
        "num_neurons": 10,
        "num_L": 2,
        "dim": 9,
    }

    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    n_desc = (
        (NEP_ARCH["n_max_radial"] + 1) * (NEP_ARCH["basis_size_radial"] + 1)
        + (NEP_ARCH["n_max_angular"] + 1) * (NEP_ARCH["basis_size_angular"] + 1)
    )

    nep_params = {
        **NEP_ARCH,
        "descriptor_params": jax.random.normal(k1, (n_desc,), dtype=jnp.float32) * 0.1,
        "ann_params": {
            0: {
                "w0": jax.random.normal(k2, (NEP_ARCH["num_neurons"], NEP_ARCH["dim"]), dtype=jnp.float32) * 0.1,
                "b0": jnp.zeros(NEP_ARCH["num_neurons"], dtype=jnp.float32),
                "w1": jax.random.normal(k3, (NEP_ARCH["num_neurons"],), dtype=jnp.float32) * 0.1,
            }
        },
        "b1": jnp.float32(0.0),
        "q_scaler": jnp.ones(NEP_ARCH["dim"], dtype=jnp.float32),
    }

    positions = jax.random.uniform(k4, (NUM_ATOMS, 3)) * BOX_SIZE_NM
    system = AtomicSystem(
        R=positions.astype(jnp.float32),
        Z=jnp.zeros(NUM_ATOMS, dtype=jnp.int32),
        cell=jnp.eye(3) * BOX_SIZE_NM,
        masses=jnp.full(NUM_ATOMS, 18.01528, dtype=jnp.float32),
        pbc=True,
    )

    neighbors, _sp = jaxmd_neighbor_list(
        positions=system.R,
        cell=system.cell,
        cutoff=CUTOFF_NM,
        capacity_multiplier=1.5,
    )

    energy_fn = build_nep_energy_fn(nep_params)
    jax_energy = float(energy_fn(system, neighbors))

    def energy_at_pos(pos):
        s = AtomicSystem(
            R=pos, Z=system.Z, cell=system.cell, masses=system.masses, pbc=True
        )
        return energy_fn(s, neighbors)

    jax_forces = -jax.grad(energy_at_pos)(system.R)

    with tempfile.TemporaryDirectory() as tmpdir:
        xyz_path = os.path.join(tmpdir, "model.xyz")
        nep_path = os.path.join(tmpdir, "nep.txt")
        runin_path = os.path.join(tmpdir, "run.in")

        write_xyz_in(system, xyz_path)
        write_nep(nep_path, nep_params)

        with open(runin_path, "w") as f:
            f.write(f"potential {nep_path}\n")
            f.write("velocity 1\n")
            f.write("ensemble nve\n")
            f.write("time_step 0\n")
            f.write("dump_force 1\n")
            f.write("dump_thermo 1\n")
            f.write("run 1\n")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        result = subprocess.run(
            [GPUMD_EXE],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            env=env,
        )

        assert result.returncode == 0, f"GPUMD exited with code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"

        thermo_path = os.path.join(tmpdir, "thermo.out")
        force_path = os.path.join(tmpdir, "force.out")

        assert os.path.exists(thermo_path), "thermo.out not produced by GPUMD"
        assert os.path.exists(force_path), "force.out not produced by GPUMD"

        thermo = np.loadtxt(thermo_path)
        if thermo.ndim == 1:
            thermo = thermo.reshape(1, -1)
        gpumd_energy_ev = float(thermo[0, 2])

        gpumd_forces = np.loadtxt(force_path).reshape(NUM_ATOMS, 3)

    gpumd_energy_kjmol = gpumd_energy_ev * EV_TO_KJMOL
    gpumd_forces_kjmol_nm = gpumd_forces * EV_TO_KJMOL * NM_TO_ANGSTROM

    energy_diff = abs(jax_energy - gpumd_energy_kjmol)
    force_diff = np.abs(np.array(jax_forces) - gpumd_forces_kjmol_nm)

    energy_tol_ev_per_atom = 1e-4
    force_tol_ev_per_angstrom = 1e-3

    energy_tol_kjmol_per_atom = energy_tol_ev_per_atom * EV_TO_KJMOL
    force_tol_kjmol_per_nm = force_tol_ev_per_angstrom * EV_TO_KJMOL * NM_TO_ANGSTROM

    pass_energy = (energy_diff / NUM_ATOMS) < energy_tol_kjmol_per_atom
    pass_force = np.max(force_diff) < force_tol_kjmol_per_nm

    assert pass_energy, (
        f"Energy per-atom error {energy_diff / NUM_ATOMS:.6e} exceeds tolerance "
        f"{energy_tol_kjmol_per_atom:.6e}"
    )
    assert pass_force, (
        f"Max force error {np.max(force_diff):.6e} exceeds tolerance "
        f"{force_tol_kjmol_per_nm:.6e}"
    )
