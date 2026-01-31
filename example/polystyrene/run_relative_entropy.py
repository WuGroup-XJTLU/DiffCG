"""Relative Entropy minimisation for a polystyrene CG model.

Requires a reference ASE trajectory (``sample.traj``) in the working
directory, e.g. produced by a prior DiffSim run.
"""

import os

import jax
jax.config.update("jax_enable_x64", True)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import optax
from ase.io import read as ase_read

from diffcg import configure_logging
from diffcg.system import from_ase_atoms
from diffcg.learning.relative_entropy import init_relative_entropy, optimize_relative_entropy

from common import (
    load_targets, load_pretrained_params, build_energy_fn,
    R_CUT, BOLTZMANN_CONSTANT,
)

configure_logging(level="INFO")

# ── Configuration ──────────────────────────────────────────────────────
Temperature = 600
script_dir = os.path.dirname(os.path.abspath(__file__))
ref_traj = os.path.join(script_dir, "sample.traj")
assert os.path.exists(ref_traj), f"Reference trajectory not found: {ref_traj}"

# ── Load data ──────────────────────────────────────────────────────────
_, topology = load_targets(Temperature)
init_system = from_ase_atoms(ase_read(ref_traj, index=0))
pretrained_params = load_pretrained_params()


# ── Energy builder ─────────────────────────────────────────────────────
def build_energy_fn_with_params(params, max_num_atoms=1):
    return build_energy_fn(params, topology, max_num_atoms=max_num_atoms)


# ── Optimizer ──────────────────────────────────────────────────────────
initial_lr = 0.01
lr_schedule = optax.exponential_decay(-initial_lr, 200, 0.01)
optimizer = optax.chain(
    optax.scale_by_adam(0.9, 0.99),
    optax.scale_by_schedule(lr_schedule),
)

sim_time_scheme = {"equilibration_steps": 10000, "production_steps": 100000}
sampler_params = {
    "ensemble": "nvt",
    "thermostat": "berendsen",
    "temperature": Temperature,
    "starting_temperature": Temperature,
    "timestep": 4.0,
    "trajectory": os.path.join(script_dir, "re_sample"),
    "logfile": os.path.join(script_dir, "re_sample"),
    "loginterval": 20,
}

# ── Run Relative Entropy ──────────────────────────────────────────────
state = {
    "init_system": init_system,
    "r_cut": R_CUT,
    "sampler_params": sampler_params,
    "sim_time_scheme": sim_time_scheme,
}

update_fn = init_relative_entropy(
    ref_traj_path=ref_traj,
    state=state,
    build_energy_fn_with_params_fn=build_energy_fn_with_params,
    optimizer=optimizer,
    reweight_ratio=0.9,
    Boltzmann_constant=BOLTZMANN_CONSTANT,
)

loss_history, times_per_update, metrics_history, params_set = optimize_relative_entropy(
    update_fn, pretrained_params, total_iterations=10
)

print(f"Relative Entropy optimisation finished. Final loss: {loss_history[-1]:.6f}")
