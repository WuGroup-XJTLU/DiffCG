#!/usr/bin/env python
"""Single-state DiffSim optimization of a coarse-grained water model.

Implements a single-site water model using only O-O RDF as the target observable.
The model uses tabulated pair potential with a generic repulsion prior.
"""

import jax
jax.config.update("jax_enable_x64", True)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import jax.numpy as jnp
import numpy as np
import optax
import os

from diffcg import energy, configure_logging
from diffcg.learning.diffsim import init_diffsim, optimize_diffsim, init_independent_mse_loss_fn
from diffcg._core.visualization import plot_potentials, create_iteration_folder, save_potentials_data
from diffcg._core.interpolate import MonotonicInterpolate
from diffcg.energy import generic_repulsion, _smooth_cutoff_factor

from water_common import (
    load_targets, load_water_system, get_initial_params,
    build_quantity_dict, build_energy_fn,
    R_CUT, R_ONSET, BOLTZMANN_CONSTANT,
    SPLINE_GRID_PAIR, NUM_WATER, WATER_MASS,
)

configure_logging(level="DEBUG")

# ── Configuration ──────────────────────────────────────────────────────
Temperature = 300  # K

# ── Load data ──────────────────────────────────────────────────────────
# Load target RDF
rdf_path = "water_oo_rdf.dat"
if not os.path.exists(rdf_path):
    raise FileNotFoundError(
        f"Target RDF file not found: {rdf_path}\n"
        "Run 'python generate_water_rdf.py' first to generate the target RDF."
    )

target_dict = load_targets(rdf_path, r_cut=R_CUT, num_water=NUM_WATER)

# Load initial system
init_system = load_water_system("SPCE_water_oxygen.data", num_water=NUM_WATER)

print(f"Loaded system with {NUM_WATER} water molecules")
print(f"Box size: {init_system.cell[0, 0]:.3f} nm")
print(f"Number density: {NUM_WATER / jnp.linalg.det(init_system.cell):.3f} nm^-3")

# ── Quantity dict ──────────────────────────────────────────────────────
quantity_dict = build_quantity_dict(target_dict)
loss_fn = init_independent_mse_loss_fn(quantity_dict)


# ── Energy builder ─────────────────────────────────────────────────────
def build_energy_fn_with_params(params, max_num_atoms=NUM_WATER):
    return build_energy_fn(params, max_num_atoms=max_num_atoms)


# ── Potential plotting ─────────────────────────────────────────────────
def plot_potentials_for_iteration(params, step, output_dir="water_output"):
    x_pair = jnp.linspace(0.2, R_CUT, 200)

    pair_spline = MonotonicInterpolate(SPLINE_GRID_PAIR, jnp.asarray(params["pair"]).ravel())

    tab_pair = _smooth_cutoff_factor(x_pair, R_ONSET, R_CUT) * pair_spline(x_pair)
    prior_pair = _smooth_cutoff_factor(x_pair, 0.9, 1.0) * generic_repulsion(x_pair, sigma=0.3, epsilon=1.0, exp=8)

    potentials_data = [
        {
            "name": "Pair",
            "x": x_pair,
            "tabulated": [tab_pair],
            "prior": [prior_pair],
            "xlabel": "r (nm)",
            "ylabel": "Energy (kJ/mol)",
            "labels": ["O-O"],
        },
    ]

    iteration_folder = create_iteration_folder(output_dir, step)
    plot_potentials(potentials_data, os.path.join(iteration_folder, "potentials.png"))
    save_potentials_data(potentials_data, iteration_folder)


# ── Initial parameters ──────────────────────────────────────────────────
initial_params = get_initial_params()
print(f"Initial parameters: pair potential with {len(initial_params['pair'])} grid points")


# ── Optimizer ──────────────────────────────────────────────────────────
initial_lr = 0.5
lr_schedule = optax.exponential_decay(-initial_lr, 100, 0.01)
optimizer = optax.chain(
    optax.scale_by_adam(0.9, 0.99),
    optax.scale_by_schedule(lr_schedule),
)


# ── Sampler settings ───────────────────────────────────────────────────
sim_time_scheme = {
    "production_steps": 10000,
    "equilibration_steps": 10000,
}

sampler_params = {
    "ensemble": "nvt",
    "thermostat": "langevin",
    "temperature": Temperature,
    "starting_temperature": Temperature,
    "timestep": 2,  # fs - smaller timestep for water due to faster dynamics
    "trajectory": "sample",
    "logfile": "sample",
    "loginterval": 25,
    'friction':1.0
}


# ── Run DiffSim ────────────────────────────────────────────────────────
state = {
    "init_system": init_system,
    "r_cut": R_CUT,
    "quantity_dict": quantity_dict,
    "sampler_params": sampler_params,
    "sim_time_scheme": sim_time_scheme,
    "output_dir": "water_output",
}

print("\n" + "=" * 60)
print("Starting DiffSim optimization for coarse-grained water model")
print("=" * 60)
print(f"Temperature: {Temperature} K")
print(f"R_CUT: {R_CUT} nm")
print(f"R_ONSET: {R_ONSET} nm")
print(f"Number of water molecules: {NUM_WATER}")
print(f"Timestep: {sampler_params['timestep']} fs")
print(f"Production steps: {sim_time_scheme['production_steps']}")
print(f"Equilibration steps: {sim_time_scheme['equilibration_steps']}")
print(f"Total iterations: 20")
print("=" * 60 + "\n")

generate_trajectory_fn, update_fn, compute_observables_fn = init_diffsim(
    reweight_ratio=0.9,
    state=state,
    build_energy_fn_with_params_fn=build_energy_fn_with_params,
    optimizer=optimizer,
    Boltzmann_constant=BOLTZMANN_CONSTANT,
)

loss_history, times_per_update, predictions_history, params_set = optimize_diffsim(
    generate_trajectory_fn, update_fn, initial_params, total_iterations=20,
    quantity_dict=quantity_dict,
    output_dir="water_output",
    save_figures=True,
    optimizer=optimizer,
    compute_observables_fn=compute_observables_fn,
    loss_fn=loss_fn,
)

# Plot potentials for each iteration
for step, p in enumerate(params_set):
    plot_potentials_for_iteration(p, step, output_dir="water_output")

print("\n" + "=" * 60)
print("Optimization complete!")
print("=" * 60)
print(f"Final loss: {loss_history[-1]:.6f}")
print(f"Loss reduction: {(loss_history[0] - loss_history[-1]) / loss_history[0] * 100:.1f}%")
print(f"Results saved to: water_output/")
print("=" * 60)
