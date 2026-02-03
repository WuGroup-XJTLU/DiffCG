"""Single-state DiffSim optimisation of a polystyrene CG model (LAMMPS backend).

Uses the LAMMPS sampler for trajectory generation while keeping JAX-based
reweighting for gradient optimization. Per-type pair interactions and a
Langevin thermostat.
"""

import jax
jax.config.update("jax_enable_x64", True)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os

import jax.numpy as jnp
import numpy as np
import optax

from diffcg import energy, configure_logging
from diffcg.learning.diffsim import init_diffsim, optimize_diffsim, init_independent_mse_loss_fn
from diffcg._core.visualization import plot_potentials, create_iteration_folder, save_potentials_data
from diffcg._core.interpolate import MonotonicInterpolate
from diffcg.energy import (
    simple_spring, harmonic_angle, harmonic_dihedral, generic_repulsion, _smooth_cutoff_factor,
    TabulatedPairEnergy, GenericRepulsionEnergy,
    TabulatedBondEnergy, HarmonicBondEnergy,
    TabulatedAngleEnergy, HarmonicAngleEnergy,
    TabulatedDihedralEnergy, HarmonicDihedralEnergy,
)

from common import (
    load_targets, load_system, load_pretrained_params,
    build_quantity_dict, build_energy_fn, build_exclusion_mask,
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

# ── Exclusion mask ────────────────────────────────────────────────────
custom_mask_function = build_exclusion_mask(topology, exclusion_level=3)

# ── Quantity dict ──────────────────────────────────────────────────────
quantity_dict = build_quantity_dict(target_dict)
loss_fn = init_independent_mse_loss_fn(quantity_dict)


# ── Build energy objects from params ──────────────────────────────────
def params_to_energy_objects(params):
    """Build energy objects from DiffCG spline params for LAMMPS auto-coupling."""
    pair_kw = dict(atom_types=atom_types, pair_type_map=pair_type_map)

    return [
        TabulatedPairEnergy(SPLINE_GRID_PAIR, params["pair"], R_ONSET, R_CUT, **pair_kw),
        GenericRepulsionEnergy(sigma=0.6, epsilon=1.0, exp=8, r_onset=0.9, r_cutoff=1.0, **pair_kw),
        TabulatedBondEnergy(SPLINE_GRID_BOND, params["bond"], topology["bond"]),
        HarmonicBondEnergy(bonds=topology["bond"], length=0.45, epsilon=5000),
        TabulatedAngleEnergy(SPLINE_GRID_ANGLE, params["angle"], topology["angle"]),
        HarmonicAngleEnergy(angles=topology["angle"], angle_0=1.5, epsilon=50),
        TabulatedDihedralEnergy(SPLINE_GRID_DIHEDRAL, params["dihedral"], topology["dihedral"]),
        HarmonicDihedralEnergy(dihedrals=topology["dihedral"], angle_0=1.5, epsilon=50),
    ]


# ── LAMMPS-compatible topology (plural keys + type arrays) ───────────
lammps_topology = {
    "bonds": topology["bond"],
    "bond_types": np.zeros(len(topology["bond"]), dtype=int),
    "angles": topology["angle"],
    "angle_types": np.zeros(len(topology["angle"]), dtype=int),
    "dihedrals": topology["dihedral"],
    "dihedral_types": np.zeros(len(topology["dihedral"]), dtype=int),
}

# ── LAMMPS config ────────────────────────────────────────────────────
lammps_config = {
    "energy_objects": params_to_energy_objects(pretrained_params),
    "topology": lammps_topology,
    "r_onset": R_ONSET,
    "lammps_exe": "lmp",
    "special_bonds": "lj 0.0 0.0 1.0",
}


# ── Energy builder (per-type pairs) — also updates LAMMPS config ─────
def build_energy_fn_with_params(params, max_num_atoms=1):
    # Only update LAMMPS config with concrete (non-traced) params
    try:
        lammps_config["energy_objects"] = params_to_energy_objects(params)
    except jax.errors.TracerArrayConversionError:
        pass  # Skip during JAX tracing (gradient computation)
    return build_energy_fn(
        params, topology,
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


# ── Optimizer ──────────────────────────────────────────────────────────
initial_lr = 0.5
lr_schedule = optax.exponential_decay(-initial_lr, 300, 0.001)
optimizer = optax.chain(
    optax.scale_by_adam(0.9, 0.99),
    optax.scale_by_schedule(lr_schedule),
)

# ── Sampler settings ───────────────────────────────────────────────────
sim_time_scheme = {"production_steps": 60000, "equilibration_steps": 60000}
sampler_params = {
    "ensemble": "nvt",
    "thermostat": "langevin",
    "temperature": Temperature,
    "starting_temperature": Temperature,
    "timestep": 4,
    "friction": 100.0,  # Langevin damping in fs (LAMMPS real units)
    "trajectory": "sample",
    "logfile": "sample",
    "loginterval": 250,
}

# ── Run DiffSim (LAMMPS backend) ─────────────────────────────────────
state = {
    "init_system": init_system,
    "r_cut": R_CUT,
    "quantity_dict": quantity_dict,
    "sampler_params": sampler_params,
    "sim_time_scheme": sim_time_scheme,
    "output_dir": "output",
    "custom_mask_function": custom_mask_function,
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
    generate_trajectory_fn, update_fn, pretrained_params, total_iterations=300,
    quantity_dict=quantity_dict,
    output_dir="output",
    save_figures=True,
    optimizer=optimizer,
    compute_observables_fn=compute_observables_fn,
    loss_fn=loss_fn,
)

for step, p in enumerate(params_set):
    plot_potentials_for_iteration(p, step, output_dir="output")
