"""Multi-state DiffSim optimisation of a polystyrene CG model.

Trains a single set of CG potentials against multiple temperatures
simultaneously using the coweighting algorithm.
"""

import jax
jax.config.update("jax_enable_x64", True)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import optax

from diffcg import configure_logging
from diffcg.learning.diffsim import init_multistate_diffsim, optimize_multistate_diffsim

from common import (
    load_targets, load_system, load_pretrained_params,
    build_quantity_dict, build_energy_fn, make_calculate_observables_fn,
    R_CUT, BOLTZMANN_CONSTANT,
)

configure_logging(level="DEBUG")

# ── Configuration ──────────────────────────────────────────────────────
temperatures = [500, 600]

# ── Load per-temperature data ──────────────────────────────────────────
target_dicts, topologies, init_systems, quantity_dicts = {}, {}, {}, {}
for temp in temperatures:
    target_dicts[temp], topologies[temp] = load_targets(temp)
    init_systems[temp], _ = load_system(temp)
    quantity_dicts[temp] = build_quantity_dict(target_dicts[temp])

# Use topology from any temperature (they share the same polymer)
topology = topologies[temperatures[0]]
pretrained_params = load_pretrained_params()


# ── Energy builder (no per-type pairs) ─────────────────────────────────
def build_energy_fn_with_params(params, max_num_atoms=1):
    return build_energy_fn(params, topology, max_num_atoms=max_num_atoms)


# ── Optimizer ──────────────────────────────────────────────────────────
initial_lr = 0.1
lr_schedule = optax.exponential_decay(-initial_lr, 200, 0.005)
optimizer = optax.chain(
    optax.scale_by_adam(0.9, 0.99),
    optax.scale_by_schedule(lr_schedule),
)

sim_time_scheme = {"production_steps": 6000, "equilibration_steps": 1000}

# ── Build multi-state configuration ───────────────────────────────────
states = {}
state_weights = {}

for temp in temperatures:
    state_id = f"T{temp}"
    sampler_params = {
        "ensemble": "nvt",
        "thermostat": "berendsen",
        "temperature": temp,
        "starting_temperature": temp,
        "timestep": 4,
        "trajectory": f"sample_T{temp}_",
        "logfile": f"sample_T{temp}_",
        "loginterval": 12,
    }
    calculate_observables_fn = make_calculate_observables_fn(
        quantity_dicts[temp], init_systems[temp]
    )
    states[state_id] = {
        "init_system": init_systems[temp],
        "r_cut": R_CUT,
        "quantity_dict": quantity_dicts[temp],
        "calculate_observables_fn": calculate_observables_fn,
        "sampler_params": sampler_params,
        "sim_time_scheme": sim_time_scheme,
    }
    state_weights[state_id] = 1.0

# ── Run multi-state DiffSim ───────────────────────────────────────────
update_fn = init_multistate_diffsim(
    reweight_ratio=0.9,
    states=states,
    build_energy_fn_with_params_fn=build_energy_fn_with_params,
    optimizer=optimizer,
    Boltzmann_constant=BOLTZMANN_CONSTANT,
    state_weights=state_weights,
    multiobj="coweighting",
)

loss_history, times_per_update, predictions_history, params_set, per_state_loss_history = (
    optimize_multistate_diffsim(update_fn, pretrained_params, total_iterations=1)
)

print("Final per-state losses:")
for state_id, loss in per_state_loss_history[-1].items():
    print(f"  {state_id}: {loss}")
print(f"Total loss: {loss_history[-1]}")
