#!/usr/bin/env python
"""Compare JAX NEP energy/forces against GPUMD reference."""

import os
import subprocess
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from diffcg.nep import build_nep_energy_fn
from diffcg.system import AtomicSystem
from diffcg.io.gpumd_writer import write_xyz_in
from diffcg.io.nep import write_nep
from diffcg._core.neighborlist import jaxmd_neighbor_list
from diffcg._core.units import EV_TO_KJMOL, NM_TO_ANGSTROM

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GPUMD_EXE = "/home/zhenghaowu/gpumd/src/gpumd"
NUM_ATOMS = 20
BOX_SIZE_NM = 2.0
CUTOFF_NM = 0.6

# Minimal NEP architecture (must be compatible with GPUMD nep.txt format)
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
    "num_L": 3,
    "dim": 12,
}

# Random NEP parameters
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

# ---------------------------------------------------------------------------
# Build random atomic system
# ---------------------------------------------------------------------------
positions = jax.random.uniform(k4, (NUM_ATOMS, 3)) * BOX_SIZE_NM
system = AtomicSystem(
    R=positions.astype(jnp.float32),
    Z=jnp.zeros(NUM_ATOMS, dtype=jnp.int32),
    cell=jnp.eye(3) * BOX_SIZE_NM,
    masses=jnp.full(NUM_ATOMS, 18.01528, dtype=jnp.float32),
    pbc=True,
)

print(f"System: {NUM_ATOMS} atoms in {BOX_SIZE_NM} nm box")

# ---------------------------------------------------------------------------
# JAX: energy and forces
# ---------------------------------------------------------------------------
neighbors, _sp = jaxmd_neighbor_list(
    positions=system.R,
    cell=system.cell,
    cutoff=CUTOFF_NM,
    capacity_multiplier=1.5,
)

energy_fn = build_nep_energy_fn(nep_params)
jax_energy = float(energy_fn(system, neighbors))  # kJ/mol


def energy_at_pos(pos):
    s = AtomicSystem(
        R=pos, Z=system.Z, cell=system.cell, masses=system.masses, pbc=True
    )
    return energy_fn(s, neighbors)


jax_forces = -jax.grad(energy_at_pos)(system.R)  # kJ/mol/nm
print(f"JAX energy: {jax_energy:.6f} kJ/mol")

# ---------------------------------------------------------------------------
# GPUMD: write inputs and run
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmpdir:
    xyz_path = os.path.join(tmpdir, "model.xyz")
    nep_path = os.path.join(tmpdir, "nep.txt")
    runin_path = os.path.join(tmpdir, "run.in")

    write_xyz_in(system, xyz_path)
    # GPUMD expects cutoffs in Angstrom; diffcg uses nm
    nep_params_gpumd = dict(nep_params)
    nep_params_gpumd["rc_radial"] = [c * NM_TO_ANGSTROM for c in nep_params["rc_radial"]]
    nep_params_gpumd["rc_angular"] = [c * NM_TO_ANGSTROM for c in nep_params["rc_angular"]]
    write_nep(nep_path, nep_params_gpumd)

    # Debug: print first few lines of generated nep.txt
    with open(nep_path) as f:
        for _ in range(8):
            print("NEP:", f.readline().strip())

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

    if result.returncode != 0:
        print("GPUMD stdout:")
        print(result.stdout)
        print("GPUMD stderr:")
        print(result.stderr)
        raise RuntimeError(f"GPUMD exited with code {result.returncode}")

    # -----------------------------------------------------------------------
    # Read GPUMD outputs
    # -----------------------------------------------------------------------
    thermo_path = os.path.join(tmpdir, "thermo.out")
    force_path = os.path.join(tmpdir, "force.out")

    if not os.path.exists(thermo_path):
        raise FileNotFoundError("thermo.out not produced by GPUMD")
    if not os.path.exists(force_path):
        raise FileNotFoundError("force.out not produced by GPUMD")

    # thermo.out columns: T, KE, PE, ... (PE is 3rd column, index 2)
    thermo = np.loadtxt(thermo_path)
    if thermo.ndim == 1:
        thermo = thermo.reshape(1, -1)
    gpumd_energy_ev = float(thermo[0, 2])

    # force.out: one line per atom, fx fy fz in eV/Angstrom
    gpumd_forces = np.loadtxt(force_path).reshape(NUM_ATOMS, 3)

# ---------------------------------------------------------------------------
# Convert GPUMD outputs to diffcg units
# ---------------------------------------------------------------------------
# Energy: eV -> kJ/mol
gpumd_energy_kjmol = gpumd_energy_ev * EV_TO_KJMOL

# Force: eV/Angstrom -> kJ/mol/nm
# 1 eV/Angstrom = EV_TO_KJMOL * NM_TO_ANGSTROM kJ/mol/nm
gpumd_forces_kjmol_nm = gpumd_forces * EV_TO_KJMOL * NM_TO_ANGSTROM

# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------
energy_diff = abs(jax_energy - gpumd_energy_kjmol)
force_diff = np.abs(np.array(jax_forces) - gpumd_forces_kjmol_nm)

print(f"GPUMD energy:    {gpumd_energy_kjmol:.6f} kJ/mol")
print(f"Energy diff:     {energy_diff:.6e} kJ/mol")
print(f"Energy diff/atom:{energy_diff / NUM_ATOMS:.6e} kJ/mol")
print(f"Max force diff:  {np.max(force_diff):.6e} kJ/mol/nm")

# Tolerances
energy_tol_ev_per_atom = 1e-4
force_tol_ev_per_angstrom = 1e-3

energy_tol_kjmol_per_atom = energy_tol_ev_per_atom * EV_TO_KJMOL
force_tol_kjmol_per_nm = force_tol_ev_per_angstrom * EV_TO_KJMOL * NM_TO_ANGSTROM

pass_energy = (energy_diff / NUM_ATOMS) < energy_tol_kjmol_per_atom
pass_force = np.max(force_diff) < force_tol_kjmol_per_nm

if pass_energy and pass_force:
    print("\nPASS")
else:
    print("\nFAIL")
    if not pass_energy:
        print(f"  Energy per-atom error {energy_diff / NUM_ATOMS:.6e} exceeds tolerance {energy_tol_kjmol_per_atom:.6e}")
    if not pass_force:
        print(f"  Max force error {np.max(force_diff):.6e} exceeds tolerance {force_tol_kjmol_per_nm:.6e}")
