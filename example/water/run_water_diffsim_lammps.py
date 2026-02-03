#!/usr/bin/env python
"""Single-state DiffSim optimization of a CG water model (LAMMPS backend).

Uses the LAMMPS sampler for trajectory generation while keeping JAX-based
reweighting for gradient optimization. Targets both O-O RDF and pressure (1 bar).
"""

import jax
jax.config.update("jax_enable_x64", True)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os

import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt

from diffcg import energy, configure_logging
from diffcg.learning.diffsim import init_diffsim, optimize_diffsim, init_independent_mse_loss_fn
from diffcg._core.visualization import plot_potentials, create_iteration_folder, save_potentials_data
from diffcg._core.interpolate import MonotonicInterpolate
from diffcg._core.constants import PRESSURE_CONVERSION_KJMOL_NM3_TO_BAR
from diffcg.energy import (
    generic_repulsion, _smooth_cutoff_factor,
    TabulatedPairEnergy, GenericRepulsionEnergy,
)
from water_common import (
    load_targets, load_water_system, get_initial_params,
    build_quantity_dict, build_energy_fn,
    R_CUT, R_ONSET, BOLTZMANN_CONSTANT,
    SPLINE_GRID_PAIR, NUM_WATER, WATER_MASS,
)

configure_logging(level="DEBUG")

# -- Configuration ----------------------------------------------------------
Temperature = 300  # K
kBT = BOLTZMANN_CONSTANT * Temperature
OUTPUT_DIR = "water_lammps_output"

# -- Load data --------------------------------------------------------------
rdf_path = "water_oo_rdf.dat"
if not os.path.exists(rdf_path):
    raise FileNotFoundError(
        f"Target RDF file not found: {rdf_path}\n"
        "Run 'python generate_water_rdf.py' first to generate the target RDF."
    )

target_dict = load_targets(rdf_path, r_cut=R_CUT, num_water=NUM_WATER)
init_system = load_water_system("SPCE_water_oxygen.data", num_water=NUM_WATER)
initial_params = get_initial_params()

print(f"Loaded system with {NUM_WATER} water molecules")
print(f"Box size: {init_system.cell[0, 0]:.3f} nm")
print(f"Number density: {NUM_WATER / jnp.linalg.det(init_system.cell):.3f} nm^-3")
print(f"Initial parameters: pair potential with {len(initial_params['pair'])} grid points")


# -- Build energy objects from params for LAMMPS ----------------------------
def params_to_energy_objects(params):
    """Build energy objects from DiffCG spline params for LAMMPS auto-coupling."""
    return [
        TabulatedPairEnergy(SPLINE_GRID_PAIR, params["pair"], R_ONSET, R_CUT),
        GenericRepulsionEnergy(sigma=0.3, epsilon=1.0, exp=8),
    ]


# -- LAMMPS config ----------------------------------------------------------
lammps_config = {
    "energy_objects": params_to_energy_objects(initial_params),
    "topology": {},
    "r_onset": R_ONSET,
    "lammps_exe": "lmp",
}


# -- Quantity dict ----------------------------------------------------------
GAMMAS = {"rdf": 1.0, "pressure": 1e-6}
quantity_dict = build_quantity_dict(target_dict, gammas=GAMMAS, kT=kBT)

loss_fn = init_independent_mse_loss_fn(quantity_dict)


# -- Energy builder (also updates LAMMPS config) ----------------------------
def build_energy_fn_with_params(params, max_num_atoms=NUM_WATER):
    """Build energy_fn and update LAMMPS config."""
    try:
        lammps_config["energy_objects"] = params_to_energy_objects(params)
    except jax.errors.TracerArrayConversionError:
        pass  # Skip during JAX tracing (gradient computation)
    return build_energy_fn(params, max_num_atoms=max_num_atoms)


# -- Potential plotting -----------------------------------------------------
def plot_potentials_for_iteration(params, step, output_dir=OUTPUT_DIR):
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


# -- Pressure plotting ------------------------------------------------------
def plot_pressure_history(predictions_history, output_dir=OUTPUT_DIR):
    """Plot averaged pressure vs iteration with a target line."""
    if "pressure" not in predictions_history[0]:
        return

    mean_pressures_bar = []
    for preds in predictions_history:
        p_internal = np.asarray(preds["pressure"]).mean()
        mean_pressures_bar.append(p_internal * PRESSURE_CONVERSION_KJMOL_NM3_TO_BAR)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(mean_pressures_bar)), mean_pressures_bar, "o-", label="Predicted pressure")
    ax.axhline(y=1.0, color="r", linestyle="--", label="Target (1 bar)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Pressure (bar)")
    ax.set_title("Mean Pressure vs Iteration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "pressure_history.png"), dpi=150)
    plt.close(fig)
    print(f"Pressure history plot saved to {output_dir}/pressure_history.png")


# -- Optimizer --------------------------------------------------------------
initial_lr = 0.5
lr_schedule = optax.exponential_decay(-initial_lr, 100, 0.01)
optimizer = optax.chain(
    optax.scale_by_adam(0.9, 0.99),
    optax.scale_by_schedule(lr_schedule),
)


# -- Sampler settings -------------------------------------------------------
sim_time_scheme = {
    "production_steps": 10000,
    "equilibration_steps": 10000,
}

sampler_params = {
    "ensemble": "nvt",
    "thermostat": "langevin",
    "temperature": Temperature,
    "starting_temperature": Temperature,
    "timestep": 2,  # fs
    "friction": 100.0,  # Langevin damping in fs (LAMMPS real units)
    "trajectory": "sample",
    "logfile": "sample",
    "loginterval": 25,
}


# -- Run DiffSim (LAMMPS backend) ------------------------------------------
print("\n" + "=" * 60)
print("Starting DiffSim optimization (LAMMPS) for CG water model")
print("=" * 60)
print(f"Temperature: {Temperature} K")
print(f"kBT: {kBT:.6f} kJ/mol")
print(f"R_CUT: {R_CUT} nm, R_ONSET: {R_ONSET} nm")
print(f"Number of water molecules: {NUM_WATER}")
print(f"Targets: RDF + Pressure (1 bar)")
print(f"Timestep: {sampler_params['timestep']} fs")
print(f"Production steps: {sim_time_scheme['production_steps']}")
print(f"Equilibration steps: {sim_time_scheme['equilibration_steps']}")
print(f"Total iterations: 50")
print("=" * 60 + "\n")

state = {
    "init_system": init_system,
    "r_cut": R_CUT,
    "quantity_dict": quantity_dict,
    "sampler_params": sampler_params,
    "sim_time_scheme": sim_time_scheme,
    "output_dir": OUTPUT_DIR,
    "sampler_backend": "lammps",
    "lammps_config": lammps_config,
}

generate_trajectory_fn, update_fn, compute_observables_fn = init_diffsim(
    reweight_ratio=0.9,
    state=state,
    build_energy_fn_with_params_fn=build_energy_fn_with_params,
    optimizer=optimizer,
    Boltzmann_constant=BOLTZMANN_CONSTANT,
)

loss_history, times_per_update, predictions_history, params_set = optimize_diffsim(
    generate_trajectory_fn, update_fn, initial_params, total_iterations=50,
    quantity_dict=quantity_dict,
    output_dir=OUTPUT_DIR,
    save_figures=True,
    optimizer=optimizer,
    compute_observables_fn=compute_observables_fn,
    loss_fn=loss_fn,
)

# -- Post-processing -------------------------------------------------------
for step, p in enumerate(params_set):
    plot_potentials_for_iteration(p, step, output_dir=OUTPUT_DIR)

plot_pressure_history(predictions_history, output_dir=OUTPUT_DIR)

print("\n" + "=" * 60)
print("Optimization complete!")
print("=" * 60)
print(f"Final loss: {loss_history[-1]:.6f}")
print(f"Loss reduction: {(loss_history[0] - loss_history[-1]) / loss_history[0] * 100:.1f}%")
print(f"Results saved to: {OUTPUT_DIR}/")
print("=" * 60)
