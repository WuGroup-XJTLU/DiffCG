import os
import numpy as np
import jax.numpy as jnp
import optax

from ase import units
from ase.io import read

import sys
sys.path.append('/home/zhenghaowu/development/diffCG')

from diffcg import energy, configure_logging
from diffcg.learning.relative_entropy import (
    init_relative_entropy,
    optimize_relative_entropy,
)

from diffcg.observable.structure import (
    InterRDFParams,
    BDFParams,
    ADFParams,
    DDFParams,
    bdf_discretization,
    adf_discretization,
    ddf_discretization,
    rdf_discretization,
)
import pandas as pd
from scipy import interpolate as sci_interpolate
import warnings

# Suppress all UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)


Temperature = 600


def get_target_dict(temp):
    pressure_conversion = 16.6054  # from kJ/mol nm^-3 to bar
    pressure_target = 1.0 / pressure_conversion  # 1 bar in kJ / mol nm^3

    bdf_bin_centers, bdf_bin_boundaries, sigma_BDF = bdf_discretization(
        1.0, nbins=200, BDF_start=0.0
    )  # cut RDF at 1nm
    bond_data = pd.read_csv(
        
        '/home/zhenghaowu/development/diffCG/examples/test_gradCG_polystyrene/datasets/T'
        + str(temp)
        + '/bondAA_smooth.dist.tgt',
        sep="\s+",
        header=None,
    )
    reference_bdf = bond_data[[0, 1]].values
    bdf_spline = sci_interpolate.interp1d(reference_bdf[:, 0], reference_bdf[:, 1], kind='cubic')
    reference_bdf = bdf_spline(bdf_bin_centers)
    reference_bdf[reference_bdf < 1e-7] = 0
    bond_top = pd.read_csv(
        '/home/zhenghaowu/development/diffCG/examples/test_gradCG_polystyrene/datasets/polymer/bond.csv',
        header=None,
        sep='\s+',
    )
    bond_top = bond_top.values - 1
    bdf_struct = BDFParams(
        reference_bdf, bdf_bin_centers, bdf_bin_boundaries, sigma_BDF, bond_top
    )

    adf_bin_centers, adf_bin_boundaries, sigma_ADF = adf_discretization(
        np.pi, nbins=200, ADF_start=0.00
    )  # cut RDF at 1nm
    angle_data = pd.read_csv(
        '/home/zhenghaowu/development/diffCG/examples/test_gradCG_polystyrene/datasets/T'
        + str(temp)
        + '/angleAAA.dist.tgt',
        sep="\s+",
        header=None,
    )
    reference_adf = angle_data[[0, 1]].values
    adf_spline = sci_interpolate.interp1d(reference_adf[:, 0], reference_adf[:, 1], kind='cubic')
    reference_adf = adf_spline(adf_bin_centers)
    reference_adf[reference_adf < 1e-7] = 0
    angle_top = pd.read_csv(
        '/home/zhenghaowu/development/diffCG/examples/test_gradCG_polystyrene/datasets/polymer/angle.csv',
        header=None,
        sep='\s+',
    )
    angle_top = angle_top.values - 1
    adf_struct = ADFParams(
        reference_adf, adf_bin_centers, adf_bin_boundaries, sigma_ADF, angle_top
    )

    ddf_bin_centers, ddf_bin_boundaries, sigma_DDF = ddf_discretization(
        3.14, nbins=200, DDF_start=-3.14
    )  # cut RDF at 1nm
    dihedral_data = pd.read_csv(
        '/home/zhenghaowu/development/diffCG/examples/test_gradCG_polystyrene/datasets/T'
        + str(temp)
        + '/dihedralAAAA.dist.tgt',
        sep="\s+",
        header=None,
    )
    reference_ddf = dihedral_data[[0, 1]].values
    ddf_spline = sci_interpolate.interp1d(reference_ddf[:, 0], reference_ddf[:, 1], kind='cubic')
    reference_ddf = ddf_spline(ddf_bin_centers)
    reference_ddf[reference_ddf < 1e-7] = 0
    dihedral_top = pd.read_csv(
        '/home/zhenghaowu/development/diffCG/examples/test_gradCG_polystyrene/datasets/polymer/dihedral.csv',
        header=None,
        sep='\s+',
    )
    dihedral_top = dihedral_top.values - 1
    ddf_struct = DDFParams(
        reference_ddf, ddf_bin_centers, ddf_bin_boundaries, sigma_DDF, dihedral_top
    )

    rdf_bin_centers, rdf_bin_boundaries, sigma_RDF = rdf_discretization(
        RDF_cut=2.0
    )  # cut RDF at 1nm
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
    reference_rdf = pd.read_csv(
        '/home/zhenghaowu/development/diffCG/examples/test_gradCG_polystyrene/datasets/T'
        + str(temp)
        + '/nb_smoothed.dist.tgt',
        header=None,
        sep='\s+',
    )
    reference_rdf = reference_rdf[[0, 1]].values
    rdf_spline = sci_interpolate.interp1d(reference_rdf[:, 0], reference_rdf[:, 1], kind='cubic')
    reference_rdf = rdf_spline(rdf_bin_centers)
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


target_dict, bond_top, angle_top, dihedral_top = get_target_dict(Temperature)

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

def build_energy_fn_with_params_factory():

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

    return build_energy_fn_with_params


def test_relative_entropy():
    configure_logging(level="INFO")

    tests_dir = os.path.dirname(os.path.abspath(__file__))
    ref_traj = os.path.join(tests_dir, 'sample.traj')
    assert os.path.exists(ref_traj), "Reference trajectory tests/sample.traj not found"

    # Initial atoms from first frame of reference trajectory
    init_atoms = read(ref_traj, index=0)

    # Parameters and energy builder
    build_energy_fn_with_params = build_energy_fn_with_params_factory()

    # Load pretrained parameters for initialization from tests assets
    params_path = os.path.join(tests_dir, 'pretrained_params.npy')
    pretrained_params = np.load(params_path, allow_pickle=True).item()

    # Optimizer (small step)
    initial_lr = 0.01
    lr_schedule = optax.exponential_decay(-initial_lr, 200, 0.01)
    optimizer = optax.chain(
        optax.scale_by_adam(0.9, 0.99),
        optax.scale_by_schedule(lr_schedule),
    )

    # Minimal MD config for generating a short CG proposal trajectory
    sim_time_scheme = {'equilibration_steps': 10000, 'production_steps': 100000}
    sampler_params = {
        'ensemble': 'nvt',
        'thermostat': 'berendsen',
        'temperature': 600,
        'starting_temperature': 600,
        'timestep': 4.0,  # fs
        'trajectory': os.path.join(tests_dir, 're_sample'),
        'logfile': os.path.join(tests_dir, 're_sample'),
        'loginterval': 20,
    }

    state = {
        'init_atoms': init_atoms,
        'r_cut': r_cut,
        'sampler_params': sampler_params,
        'sim_time_scheme': sim_time_scheme,
    }

    # Initialize RE update
    update_fn = init_relative_entropy(
        ref_traj_path=ref_traj,
        state=state,
        build_energy_fn_with_params_fn=build_energy_fn_with_params,
        optimizer=optimizer,
        reweight_ratio=0.9,
        Boltzmann_constant=0.0083145107,
    )

    # Run a single RE iteration
    from diffcg.learning.relative_entropy import optimize_relative_entropy as _opt

    loss_history, times_per_update, metrics_history, params_set = _opt(
        update_fn, pretrained_params, total_iterations=10
    )

test_relative_entropy()
