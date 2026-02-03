"""Single-state DiffSim optimisation of a polystyrene CG model (LAMMPS backend).

Simplified version using only bond, angle, and pair potentials (no dihedral).
Targets: RDF, BDF, ADF. Dihedral topology is still loaded for the RDF
exclusion mask.
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
from diffcg.energy import simple_spring, harmonic_angle, generic_repulsion, _smooth_cutoff_factor

from common import (
    load_system, load_pretrained_params,
    build_quantity_dict, build_exclusion_mask,
    R_CUT, R_ONSET, BOLTZMANN_CONSTANT, PRESSURE_TARGET,
    SPLINE_GRID_PAIR, SPLINE_GRID_BOND, SPLINE_GRID_ANGLE,
    load_curve_csv, load_topology_csv, DATA_DIR, NUM_BEADS,
)
from diffcg.observable.structure import (
    bdf_discretization, adf_discretization, rdf_discretization,
    BDFParams, ADFParams, InterRDFParams, exclusion_pairs_from_topology,
)

configure_logging(level="DEBUG")

# ── Configuration ──────────────────────────────────────────────────────
Temperature = 600
OUTPUT_DIR = "output_lammps_bap"


# ── Custom target loader (no DDF) ─────────────────────────────────────
def load_targets_bap(temp, num_beads=NUM_BEADS):
    """Load BDF, ADF, RDF targets only — no DDF.

    Dihedral topology is still loaded for the RDF exclusion mask.
    """
    # BDF
    bdf_bin_centers, bdf_bin_boundaries, sigma_BDF = bdf_discretization(
        1.0, nbins=200, BDF_start=0.0
    )
    bdf_path = os.path.join(DATA_DIR, f"T{temp}", "bondAA_smooth.dist.tgt")
    reference_bdf = load_curve_csv(bdf_path, bdf_bin_centers)
    bond_top = load_topology_csv(os.path.join(DATA_DIR, "polymer", "bond.csv"))
    bdf_struct = BDFParams(reference_bdf, bdf_bin_centers, bdf_bin_boundaries, sigma_BDF, bond_top)

    # ADF
    adf_bin_centers, adf_bin_boundaries, sigma_ADF = adf_discretization(
        np.pi, nbins=200, ADF_start=0.00
    )
    adf_path = os.path.join(DATA_DIR, f"T{temp}", "angleAAA.dist.tgt")
    reference_adf = load_curve_csv(adf_path, adf_bin_centers)
    angle_top = load_topology_csv(os.path.join(DATA_DIR, "polymer", "angle.csv"))
    adf_struct = ADFParams(reference_adf, adf_bin_centers, adf_bin_boundaries, sigma_ADF, angle_top)

    # Dihedral topology (for RDF exclusion mask only)
    dihedral_top = load_topology_csv(os.path.join(DATA_DIR, "polymer", "dihedral.csv"))

    # RDF (with exclusion pairs from topology — excludes 1-2, 1-3, 1-4 interactions)
    rdf_bin_centers, rdf_bin_boundaries, sigma_RDF = rdf_discretization(RDF_cut=R_CUT)
    topology = {"bond": bond_top, "angle": angle_top, "dihedral": dihedral_top}
    exclude_pairs = exclusion_pairs_from_topology(topology, exclusion_level=4)

    rdf_path = os.path.join(DATA_DIR, f"T{temp}", "nb_smoothed.dist.tgt")
    reference_rdf = load_curve_csv(rdf_path, rdf_bin_centers, zero_eps=0.0)
    rdf_struct = InterRDFParams(reference_rdf, rdf_bin_centers, rdf_bin_boundaries, sigma_RDF, exclude_pairs)

    target_dict = {
        "rdf": rdf_struct,
        "bdf": bdf_struct,
        "adf": adf_struct,
        "pressure": PRESSURE_TARGET,
    }
    topology = {"bond": bond_top, "angle": angle_top, "dihedral": dihedral_top}
    return target_dict, topology


# ── Custom energy builder (no dihedral terms) ─────────────────────────
def build_energy_fn_bap(params, topology, atom_types=None, pair_type_map=None,
                        **_unused):
    """Build total energy from bond + angle + pair (no dihedral)."""
    bond_top = topology["bond"]
    angle_top = topology["angle"]

    pair_kw = {}
    if atom_types is not None and pair_type_map is not None:
        pair_kw.update(atom_types=atom_types, pair_type_map=pair_type_map)

    pair_energy_fn = energy.TabulatedPairEnergy(
        SPLINE_GRID_PAIR, params["pair"], R_ONSET, R_CUT, **pair_kw
    ).get_energy_fn()
    bond_energy_fn = energy.TabulatedBondEnergy(
        SPLINE_GRID_BOND, params["bond"], bond_top
    ).get_energy_fn()
    angle_energy_fn = energy.TabulatedAngleEnergy(
        SPLINE_GRID_ANGLE, params["angle"], angle_top
    ).get_energy_fn()

    prior_fn = energy.GenericRepulsionEnergy(
        sigma=0.6, epsilon=1.0, exp=8, **pair_kw
    ).get_energy_fn()
    bond_pot_fn = energy.HarmonicBondEnergy(bonds=bond_top, length=0.45, epsilon=5000).get_energy_fn()
    angle_pot_fn = energy.HarmonicAngleEnergy(angles=angle_top, angle_0=1.5, epsilon=50).get_energy_fn()

    fns = [pair_energy_fn, bond_energy_fn, angle_energy_fn,
           bond_pot_fn, angle_pot_fn, prior_fn]

    def total_energy_fn(system, neighbors, **dynamic_kwargs):
        return sum(f(system, neighbors, **dynamic_kwargs) for f in fns)

    return total_energy_fn


# ── Load data ──────────────────────────────────────────────────────────
target_dict, topology = load_targets_bap(Temperature)
init_system, atom_types = load_system(Temperature)

# Per-type pair expansion
n_atom_types = int(atom_types.max()) + 1
pair_type_map, n_pair_types = energy.build_pair_type_map(n_atom_types)

pretrained_params = load_pretrained_params()
# Filter to bond/angle/pair only
pretrained_params = {k: pretrained_params[k] for k in ("pair", "bond", "angle")}
pair_vals = jnp.asarray(pretrained_params["pair"])
if pair_vals.ndim == 1:
    pretrained_params["pair"] = jnp.tile(pair_vals, (n_pair_types, 1))
elif pair_vals.shape[0] != n_pair_types:
    raise ValueError(f"pair params shape {pair_vals.shape} incompatible with n_pair_types={n_pair_types}")

# ── Exclusion mask ────────────────────────────────────────────────────
custom_mask_function = build_exclusion_mask(topology, exclusion_level=4)

# ── Quantity dict ──────────────────────────────────────────────────────
kBT = BOLTZMANN_CONSTANT * Temperature
quantity_dict = build_quantity_dict(target_dict, kT=kBT)
loss_fn = init_independent_mse_loss_fn(quantity_dict)


# ── LAMMPS energy params conversion (no dihedral) ────────────────────
def params_to_lammps_energy_params(params):
    """Convert DiffCG spline params to LAMMPS tabulated energy format."""
    x_pair = np.linspace(0.05, R_CUT, 500)
    x_bond = np.linspace(0.1, 1.0, 300)
    x_angle = np.linspace(0.0, np.pi, 300)

    # Pair: tabulated spline + smooth cutoff + repulsion prior
    pair_params = jnp.asarray(params["pair"])
    pair_y_list = []
    for pt in range(pair_params.shape[0]):
        sp = MonotonicInterpolate(SPLINE_GRID_PAIR, pair_params[pt])
        tab = _smooth_cutoff_factor(x_pair, R_ONSET, R_CUT) * sp(x_pair)
        prior = (
            _smooth_cutoff_factor(x_pair, 0.9, 1.0)
            * generic_repulsion(x_pair, sigma=0.6, epsilon=1.0, exp=8)
        )
        pair_y_list.append(np.asarray(tab + prior))

    # Bond: tabulated + harmonic prior
    bond_sp = MonotonicInterpolate(SPLINE_GRID_BOND, jnp.asarray(params["bond"]).ravel())
    bond_y = np.asarray(bond_sp(x_bond) + simple_spring(x_bond, length=0.45, epsilon=5000))

    # Angle: tabulated + harmonic prior
    angle_sp = MonotonicInterpolate(SPLINE_GRID_ANGLE, jnp.asarray(params["angle"]).ravel())
    angle_y = np.asarray(angle_sp(x_angle) + harmonic_angle(x_angle, angle_0=1.5, epsilon=50))

    # Build pair type list: upper-triangle ordering matching build_pair_type_map
    _n = n_atom_types
    pair_types = []
    for i in range(_n):
        for j in range(i, _n):
            pair_types.append((i, j))

    return {
        "pair": {"x": x_pair, "y": pair_y_list, "types": pair_types},
        "bond": {"x": x_bond, "y": bond_y},
        "angle": {"x": x_angle, "y": angle_y},
    }


# ── LAMMPS-compatible topology (no dihedral) ─────────────────────────
lammps_topology = {
    "bonds": topology["bond"],
    "bond_types": np.zeros(len(topology["bond"]), dtype=int),
    "angles": topology["angle"],
    "angle_types": np.zeros(len(topology["angle"]), dtype=int),
}

# ── LAMMPS config ────────────────────────────────────────────────────
lammps_config = {
    "energy_params": params_to_lammps_energy_params(pretrained_params),
    "topology": lammps_topology,
    "r_onset": R_ONSET,
    "lammps_exe": "lmp",
}


# ── Energy builder — also updates LAMMPS config ─────────────────────
def build_energy_fn_with_params(params, max_num_atoms=1):
    try:
        lammps_config["energy_params"] = params_to_lammps_energy_params(params)
    except jax.errors.TracerArrayConversionError:
        pass
    return build_energy_fn_bap(
        params, topology,
        atom_types=atom_types, pair_type_map=pair_type_map,
    )


# ── Potential plotting (3 panels: bond, angle, pair) ──────────────────
def plot_potentials_for_iteration(params, step, output_dir=OUTPUT_DIR):
    x_bond = jnp.linspace(0.1, 1.0, 200)
    x_angle = jnp.linspace(0.1, 3.14, 200)
    x_pair = jnp.linspace(0.4, 2.0, 200)

    bond_spline = MonotonicInterpolate(SPLINE_GRID_BOND, jnp.asarray(params["bond"]).ravel())
    angle_spline = MonotonicInterpolate(SPLINE_GRID_ANGLE, jnp.asarray(params["angle"]).ravel())

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
        {"name": "Pair", "x": x_pair,
         "tabulated": tab_pair_list, "prior": prior_pair_list,
         "xlabel": "r (nm)", "ylabel": "Energy (kJ/mol)", "labels": pair_labels},
    ]

    iteration_folder = create_iteration_folder(output_dir, step)
    plot_potentials(potentials_data, os.path.join(iteration_folder, "potentials.png"))
    save_potentials_data(potentials_data, iteration_folder)


# ── Optimizer ──────────────────────────────────────────────────────────
initial_lr = 0.5
lr_schedule = optax.exponential_decay(-initial_lr, 50, 0.01)
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
    "friction": 100.0,
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
    "output_dir": OUTPUT_DIR,
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
    generate_trajectory_fn, update_fn, pretrained_params, total_iterations=50,
    quantity_dict=quantity_dict,
    output_dir=OUTPUT_DIR,
    save_figures=True,
    optimizer=optimizer,
    compute_observables_fn=compute_observables_fn,
    loss_fn=loss_fn,
)

for step, p in enumerate(params_set):
    plot_potentials_for_iteration(p, step, output_dir=OUTPUT_DIR)
