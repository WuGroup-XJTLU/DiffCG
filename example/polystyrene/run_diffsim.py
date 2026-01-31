"""Single-state DiffSim optimisation of a polystyrene CG model.

Uses per-type pair interactions and a Langevin thermostat.
"""

import jax
jax.config.update("jax_enable_x64", True)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import jax.numpy as jnp
import numpy as np
import optax

from diffcg import energy, configure_logging
from diffcg.learning.diffsim import init_diffsim, optimize_diffsim, init_independent_mse_loss_fn
from diffcg._core.visualization import plot_potentials, create_iteration_folder, save_potentials_data
from diffcg._core.interpolate import MonotonicInterpolate
from diffcg.energy import simple_spring, harmonic_angle, harmonic_dihedral, generic_repulsion, _smooth_cutoff_factor

from common import (
    load_targets, load_system, load_pretrained_params,
    build_quantity_dict, build_energy_fn, make_calculate_observables_fn,
    R_CUT, R_ONSET, BOLTZMANN_CONSTANT,
    SPLINE_GRID_PAIR, SPLINE_GRID_BOND, SPLINE_GRID_ANGLE, SPLINE_GRID_DIHEDRAL,
)

configure_logging(level="DEBUG")

# ── Configuration ──────────────────────────────────────────────────────
Temperature = 600

# ── Load data ──────────────────────────────────────────────────────────
target_dict, topology = load_targets(Temperature)
init_system, atom_types = load_system(Temperature)

# Per-type pair expansion
n_atom_types = int(atom_types.max()) + 1
pair_type_map, n_pair_types = energy.build_pair_type_map(n_atom_types)

pretrained_params = load_pretrained_params()
pair_vals = jnp.asarray(pretrained_params["pair"])
if pair_vals.ndim == 1:
    pretrained_params["pair"] = jnp.tile(pair_vals, (n_pair_types, 1))
elif pair_vals.shape[0] != n_pair_types:
    raise ValueError(f"pair params shape {pair_vals.shape} incompatible with n_pair_types={n_pair_types}")

# ── Quantity dict ──────────────────────────────────────────────────────
quantity_dict = build_quantity_dict(target_dict)
calculate_observables = make_calculate_observables_fn(quantity_dict, init_system)
loss_fn = init_independent_mse_loss_fn(quantity_dict)


# ── Energy builder (per-type pairs) ───────────────────────────────────
def build_energy_fn_with_params(params, max_num_atoms=1):
    return build_energy_fn(
        params, topology, max_num_atoms=max_num_atoms,
        atom_types=atom_types, pair_type_map=pair_type_map,
    )


# ── Potential plotting ─────────────────────────────────────────────────
def plot_potentials_for_iteration(params, step, output_dir="output"):
    x_bond = jnp.linspace(0.1, 1.0, 200)
    x_angle = jnp.linspace(0.1, 3.14, 200)
    x_dihedral = jnp.linspace(-3.14, 3.14, 200)
    x_pair = jnp.linspace(0.4, 2.0, 200)

    bond_spline = MonotonicInterpolate(SPLINE_GRID_BOND, jnp.asarray(params["bond"]).ravel())
    angle_spline = MonotonicInterpolate(SPLINE_GRID_ANGLE, jnp.asarray(params["angle"]).ravel())
    dihedral_spline = MonotonicInterpolate(SPLINE_GRID_DIHEDRAL, jnp.asarray(params["dihedral"]).ravel())

    pair_params = jnp.asarray(params["pair"])
    tab_pair_list, prior_pair_list, pair_labels = [], [], []
    for pt in range(pair_params.shape[0]):
        sp = MonotonicInterpolate(SPLINE_GRID_PAIR, pair_params[pt])
        tab_pair_list.append(_smooth_cutoff_factor(x_pair, R_ONSET, R_CUT) * sp(x_pair))
        prior_pair_list.append(
            _smooth_cutoff_factor(x_pair, 0.9, 1.0) * generic_repulsion(x_pair, sigma=0.6, epsilon=1.0, exp=8)
        )
        pair_labels.append(f"pair {pt}")

    potentials_data = [
        {"name": "Bond", "x": x_bond,
         "tabulated": bond_spline(x_bond), "prior": simple_spring(x_bond, length=0.45, epsilon=5000),
         "xlabel": "r (nm)", "ylabel": "Energy (kJ/mol)"},
        {"name": "Angle", "x": x_angle,
         "tabulated": angle_spline(x_angle), "prior": harmonic_angle(x_angle, angle_0=1.5, epsilon=50),
         "xlabel": "Angle (rad)", "ylabel": "Energy (kJ/mol)"},
        {"name": "Dihedral", "x": x_dihedral,
         "tabulated": dihedral_spline(x_dihedral), "prior": harmonic_dihedral(x_dihedral, angle_0=1.5, epsilon=50),
         "xlabel": "Dihedral (rad)", "ylabel": "Energy (kJ/mol)"},
        {"name": "Pair", "x": x_pair,
         "tabulated": tab_pair_list, "prior": prior_pair_list,
         "xlabel": "r (nm)", "ylabel": "Energy (kJ/mol)", "labels": pair_labels},
    ]

    iteration_folder = create_iteration_folder(output_dir, step)
    plot_potentials(potentials_data, os.path.join(iteration_folder, "potentials.png"))
    save_potentials_data(potentials_data, iteration_folder)


import os  # noqa: E402 (used in plot_potentials_for_iteration)

# ── Optimizer ──────────────────────────────────────────────────────────
initial_lr = 0.1
lr_schedule = optax.exponential_decay(-initial_lr, 200, 0.005)
optimizer = optax.chain(
    optax.scale_by_adam(0.9, 0.99),
    optax.scale_by_schedule(lr_schedule),
)

# ── Sampler settings ───────────────────────────────────────────────────
sim_time_scheme = {"production_steps": 6000, "equilibration_steps": 6000}
sampler_params = {
    "ensemble": "nvt",
    "thermostat": "langevin",
    "temperature": Temperature,
    "starting_temperature": Temperature,
    "timestep": 4,
    "trajectory": "sample",
    "logfile": "sample",
    "loginterval": 25,
}

# ── Run DiffSim ────────────────────────────────────────────────────────
state = {
    "init_system": init_system,
    "r_cut": R_CUT,
    "quantity_dict": quantity_dict,
    "calculate_observables_fn": calculate_observables,
    "sampler_params": sampler_params,
    "sim_time_scheme": sim_time_scheme,
    "output_dir": "output",
}

generate_trajectory_fn, update_fn = init_diffsim(
    reweight_ratio=0.9,
    state=state,
    build_energy_fn_with_params_fn=build_energy_fn_with_params,
    optimizer=optimizer,
    Boltzmann_constant=BOLTZMANN_CONSTANT,
)

loss_history, times_per_update, predictions_history, params_set = optimize_diffsim(
    generate_trajectory_fn, update_fn, pretrained_params, total_iterations=5,
    quantity_dict=quantity_dict,
    output_dir="output",
    save_figures=True,
    optimizer=optimizer,
)

for step, p in enumerate(params_set):
    plot_potentials_for_iteration(p, step, output_dir="output")
