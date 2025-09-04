import os

# Configure GPU settings
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Disable memory preallocation

from jax.tree_util import tree_map
from ase.io import read
import numpy as np
import jax.numpy as jnp

import pandas as pd
from scipy import interpolate as sci_interpolate
from ase import units

import sys
sys.path.append('/home/zhenghaowu/development/diffCG')

from diffcg import energy
from diffcg.observable.structure import (
    initialize_inter_radial_distribution_fun,
    initialize_angle_distribution_fun,
    initialize_bond_distribution_fun,
    initialize_dihedral_distribution_fun,
    InterRDFParams,
    BDFParams,
    ADFParams,
    DDFParams,
    bdf_discretization,
    adf_discretization,
    ddf_discretization,
    rdf_discretization,
)
from diffcg.io.ase_trj import read_ase_trj
import warnings
from diffcg.learning.diffsim import init_independent_mse_loss_fn, init_multistate_diffsim, optimize_multistate_diffsim
import optax
from diffcg.io.lammps import read_lammps_data
from ase import Atoms
from diffcg.observable.analyze import analyze

# Suppress all UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

from diffcg import configure_logging

configure_logging(level="DEBUG")

# Or suppress specific warning messages
warnings.filterwarnings("ignore", message="specific warning text")
import os


DATASET_ROOT = "/home/zhenghaowu/development/diffCG/examples/test_gradCG_polystyrene/datasets"


def _load_curve_csv(path: str, x_grid: np.ndarray, zero_eps: float = 1e-7):
    data = pd.read_csv(path, sep=r"\s+", header=None)
    xy = data[[0, 1]].values
    spline = sci_interpolate.interp1d(
        xy[:, 0], xy[:, 1], kind="cubic", bounds_error=False, fill_value="extrapolate"
    )
    y = spline(x_grid)
    y[y < zero_eps] = 0
    return y


def _load_topology_csv(path: str):
    return pd.read_csv(path, header=None, sep=r"\s+").values - 1


def get_target_dict(temp):
    pressure_conversion = 16.6054  # from kJ/mol nm^-3 to bar
    pressure_target = 1.0 / pressure_conversion  # 1 bar in kJ / mol nm^3

    # BDF
    bdf_bin_centers, bdf_bin_boundaries, sigma_BDF = bdf_discretization(
        1.0, nbins=200, BDF_start=0.0
    )
    bdf_path = os.path.join(DATASET_ROOT, f"T{temp}", "bondAA_smooth.dist.tgt")
    reference_bdf = _load_curve_csv(bdf_path, bdf_bin_centers)
    bond_top = _load_topology_csv(os.path.join(DATASET_ROOT, "polymer", "bond.csv"))
    bdf_struct = BDFParams(
        reference_bdf, bdf_bin_centers, bdf_bin_boundaries, sigma_BDF, bond_top
    )

    # ADF
    adf_bin_centers, adf_bin_boundaries, sigma_ADF = adf_discretization(
        np.pi, nbins=200, ADF_start=0.00
    )
    adf_path = os.path.join(DATASET_ROOT, f"T{temp}", "angleAAA.dist.tgt")
    reference_adf = _load_curve_csv(adf_path, adf_bin_centers)
    angle_top = _load_topology_csv(os.path.join(DATASET_ROOT, "polymer", "angle.csv"))
    adf_struct = ADFParams(
        reference_adf, adf_bin_centers, adf_bin_boundaries, sigma_ADF, angle_top
    )

    # DDF
    ddf_bin_centers, ddf_bin_boundaries, sigma_DDF = ddf_discretization(
        3.14, nbins=200, DDF_start=-3.14
    )
    ddf_path = os.path.join(DATASET_ROOT, f"T{temp}", "dihedralAAAA.dist.tgt")
    reference_ddf = _load_curve_csv(ddf_path, ddf_bin_centers)
    dihedral_top = _load_topology_csv(os.path.join(DATASET_ROOT, "polymer", "dihedral.csv"))
    ddf_struct = DDFParams(
        reference_ddf, ddf_bin_centers, ddf_bin_boundaries, sigma_DDF, dihedral_top
    )

    # RDF
    rdf_bin_centers, rdf_bin_boundaries, sigma_RDF = rdf_discretization(RDF_cut=2.0)
    mask = np.ones((500, 500))
    mask[dihedral_top[:, 0], dihedral_top[:, 1]] = 0
    mask[dihedral_top[:, 1], dihedral_top[:, 0]] = 0
    mask[dihedral_top[:, 2], dihedral_top[:, 0]] = 0
    mask[dihedral_top[:, 0], dihedral_top[:, 2]] = 0
    mask[dihedral_top[:, 3], dihedral_top[:, 0]] = 0
    mask[dihedral_top[:, 0], dihedral_top[:, 3]] = 0
    mask[dihedral_top[:, 2], dihedral_top[:, 1]] = 0
    mask[dihedral_top[:, 1], dihedral_top[:, 2]] = 0
    mask[dihedral_top[:, 3], dihedral_top[:, 1]] = 0
    mask[dihedral_top[:, 1], dihedral_top[:, 3]] = 0
    mask[dihedral_top[:, 3], dihedral_top[:, 2]] = 0
    mask[dihedral_top[:, 2], dihedral_top[:, 3]] = 0
    polymer_exclude = jnp.array(mask)

    rdf_path = os.path.join(DATASET_ROOT, f"T{temp}", "nb_smoothed.dist.tgt")
    reference_rdf = _load_curve_csv(rdf_path, rdf_bin_centers, zero_eps=0.0)
    rdf_struct = InterRDFParams(
        reference_rdf, rdf_bin_centers, rdf_bin_boundaries, sigma_RDF, polymer_exclude
    )

    target_dict = {
        'rdf': rdf_struct,
        'bdf': bdf_struct,
        'adf': adf_struct,
        'ddf': ddf_struct,
        'pressure': pressure_target,
    }
    return target_dict, bond_top, angle_top, dihedral_top


# Get target dictionaries for different temperatures
temperatures = [500, 600]
target_dicts = {}
for temp in temperatures:
    target_dicts[temp], bond_top, angle_top, dihedral_top = get_target_dict(temp)



r_cut = 2.0
r_onset = 1.5
Boltzmann_constant = 0.0083145107  # in kJ / mol K
spline_grid_pair = jnp.linspace(0.4, r_cut, 80)

angle_limit = [0.1, 3.14, 55]
bond_limit = [0.1, 1.0, 45]
dihedral_limit = [-3.14, 3.14, 100]

spline_grid_angle = jnp.linspace(angle_limit[0], angle_limit[1], angle_limit[2])
spline_grid_bond = jnp.linspace(bond_limit[0], bond_limit[1], bond_limit[2])
spline_grid_dihedral = jnp.linspace(dihedral_limit[0], dihedral_limit[1], dihedral_limit[2])


def build_energy_fn_with_params(params, max_num_atoms=1):
    pair_energy_fn = energy.TabulatedPairEnergy(
        spline_grid_pair, params['pair'], r_onset, r_cut, mask_topology=angle_top, max_num_atoms=max_num_atoms
    ).get_energy_fn()
    bond_energy_fn = energy.TabulatedBondEnergy(spline_grid_bond, params['bond'], bond_top).get_energy_fn()
    angle_energy_fn = energy.TabulatedAngleEnergy(spline_grid_angle, params['angle'], angle_top).get_energy_fn()
    dihedral_energy_fn = energy.TabulatedDihedralEnergy(
        spline_grid_dihedral, params['dihedral'], dihedral_top
    ).get_energy_fn()

    prior_fn = energy.GenericRepulsionEnergy(
        sigma=0.6, epsilon=1.0, exp=8, mask_topology=angle_top, max_num_atoms=max_num_atoms
    ).get_energy_fn()
    bond_pot_fn = energy.HarmonicBondEnergy(bonds=bond_top, length=0.45, epsilon=5000).get_energy_fn()
    angle_pot_fn = energy.HarmonicAngleEnergy(angles=angle_top, angle_0=1.5, epsilon=50).get_energy_fn()
    dihedral_pot_fn = energy.HarmonicDihedralEnergy(dihedrals=dihedral_top, angle_0=1.5, epsilon=50).get_energy_fn()

    energy_fn_list = [
        pair_energy_fn,
        bond_energy_fn,
        angle_energy_fn,
        dihedral_energy_fn,
        bond_pot_fn,
        angle_pot_fn,
        dihedral_pot_fn,
        prior_fn,
    ]

    def energy_fn(system, neighbors, **dynamic_kwargs):
        return sum(energy_fn(system, neighbors, **dynamic_kwargs) for energy_fn in energy_fn_list) * units.kJ / units.mol

    return energy_fn


pretrained_params = np.load('/home/zhenghaowu/development/diffCG/tests/pretrained_params.npy', allow_pickle=True).item()

def create_quantity_dict(target_dict):
    """Create quantity dictionary for a given target dictionary."""
    quantity_dict = {}
    if 'rdf' in target_dict:
        rdf_struct = target_dict['rdf']
        rdf_fn = initialize_inter_radial_distribution_fun(rdf_struct)
        rdf_dict = {'compute_fn': rdf_fn, 'target': rdf_struct.reference_rdf, 'gamma': 1.0}
        quantity_dict['rdf'] = rdf_dict

    if 'bdf' in target_dict:
        bdf_struct = target_dict['bdf']
        bdf_fn = initialize_bond_distribution_fun(bdf_struct)
        bdf_dict = {'compute_fn': bdf_fn, 'target': bdf_struct.reference_bdf, 'gamma': 1.0 / 1000}
        quantity_dict['bdf'] = bdf_dict

    if 'adf' in target_dict:
        adf_struct = target_dict['adf']
        adf_fn = initialize_angle_distribution_fun(adf_struct)
        adf_dict = {'compute_fn': adf_fn, 'target': adf_struct.reference_adf, 'gamma': 1.0 / 10}
        quantity_dict['adf'] = adf_dict

    if 'ddf' in target_dict:
        ddf_struct = target_dict['ddf']
        ddf_fn = initialize_dihedral_distribution_fun(ddf_struct)
        ddf_dict = {'compute_fn': ddf_fn, 'target': ddf_struct.reference_ddf, 'gamma': 1.0}
        quantity_dict['ddf'] = ddf_dict
    
    return quantity_dict

# Create quantity dictionaries for each temperature
quantity_dicts = {}
for temp in temperatures:
    quantity_dicts[temp] = create_quantity_dict(target_dicts[temp])


def create_calculate_observables_fn(quantity_dict, init_atoms):
    """Create a calculate_observables function for a specific quantity_dict."""
    def calculate_observables(traj_file='sample.traj'):
        systems = read_ase_trj(traj_file)
        batched_systems = tree_map(lambda *xs: jnp.stack(xs), *systems)  # R: (B,500,3), Z: (B,500), cell: (B,3,3)
        
        observables = {}
        if 'rdf' in quantity_dict:
            rdf_analyzer = analyze(quantity_dict['rdf']['compute_fn'], init_atoms)
            observables['rdf'] = rdf_analyzer.analyze(batched_systems)
        
        if 'bdf' in quantity_dict:
            bdf_analyzer = analyze(quantity_dict['bdf']['compute_fn'], init_atoms)
            observables['bdf'] = bdf_analyzer.analyze(batched_systems)
        
        if 'adf' in quantity_dict:
            adf_analyzer = analyze(quantity_dict['adf']['compute_fn'], init_atoms)
            observables['adf'] = adf_analyzer.analyze(batched_systems)
        
        if 'ddf' in quantity_dict:
            ddf_analyzer = analyze(quantity_dict['ddf']['compute_fn'], init_atoms)
            observables['ddf'] = ddf_analyzer.analyze(batched_systems)

        return observables
    return calculate_observables


# Load initial structure
lammpsdata_file = {
    'T600': '/home/zhenghaowu/development/diffCG/examples/test_gradCG_polystyrene/datasets/T600/PS.data',
    'T500': '/home/zhenghaowu/development/diffCG/examples/test_gradCG_polystyrene/datasets/T500/PS.data',
    'T400': '/home/zhenghaowu/development/diffCG/examples/test_gradCG_polystyrene/datasets/T400/PS.data',
    'T300': '/home/zhenghaowu/development/diffCG/examples/test_gradCG_polystyrene/datasets/T300/PS.data',
}

init_atoms = {} 
for temp in temperatures:
    sys_data = read_lammps_data(lammpsdata_file[f'T{temp}'])
    # Mapping of atom types
    atom_types_mapping = {0: 'C', 1: 'C', 2: 'H'}
    data_element_figurelist = sys_data["atom_types"]
    data_element = [atom_types_mapping[i] for i in data_element_figurelist]

    # Extract coordinates and cell information
    data_coord = sys_data["coords"] / 10.0  # A to nm
    cell = sys_data["cells"][0] / 10.0  # A to nm

    data_coord = data_coord[0]

    # Create ASE Atoms object
    init_atoms[temp] = Atoms(positions=data_coord, symbols=data_element, cell=cell, pbc=jnp.array([True, True, True]))
    _masses = [104.0 for i in range(500)]
    init_atoms[temp].set_masses(_masses)

initial_lr = 0.1
lr_schedule = optax.exponential_decay(-initial_lr, 200, 0.005)
optimizer = optax.chain(
    optax.scale_by_adam(0.9, 0.99),
    optax.scale_by_schedule(lr_schedule),
)

sim_time_scheme = {'production_steps': 6 * 1000, 'equilibration_steps': 1000}

params = pretrained_params

# Create multistate configuration
states = {}
state_weights = {}

for temp in temperatures:
    state_id = f"T{temp}"
    
    # Create unique trajectory and log file names for each state
    sampler_params = {
        'ensemble': "nvt",
        'thermostat': "berendsen",
        'temperature': temp,
        'starting_temperature': temp,
        'timestep': 4,
        'trajectory': f"sample_T{temp}_",  # Unique trajectory prefix
        'logfile': f"sample_T{temp}_",     # Unique log prefix
        'loginterval': 12,
    }
    
    # Create calculate_observables function for this temperature
    calculate_observables_fn = create_calculate_observables_fn(quantity_dicts[temp], init_atoms[temp])
    
    states[state_id] = {
        'init_atoms': init_atoms[temp],
        'r_cut': r_cut,
        'quantity_dict': quantity_dicts[temp],
        'calculate_observables_fn': calculate_observables_fn,
        'sampler_params': sampler_params,
        'sim_time_scheme': sim_time_scheme,
    }
    
    # Equal weights for all states
    state_weights[state_id] = 1.0

# Initialize multistate DiffSim with coweighting
update_fn = init_multistate_diffsim(
    reweight_ratio=0.9,
    states=states,
    build_energy_fn_with_params_fn=build_energy_fn_with_params,
    optimizer=optimizer,
    Boltzmann_constant=Boltzmann_constant,
    state_weights=state_weights,
    multiobj='coweighting'  # Use coweighting algorithm
)

# Run multistate optimization
loss_history, times_per_update, predictions_history, params_set, per_state_loss_history = optimize_multistate_diffsim(
    update_fn, params, total_iterations=1
)

# Print results for each state
print("Final per-state losses:")
for state_id, loss in per_state_loss_history[-1].items():
    print(f"  {state_id}: {loss}")

print(f"Total loss: {loss_history[-1]}")


