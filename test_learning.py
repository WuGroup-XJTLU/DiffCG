from diffcg import energy
from jax.tree_util import tree_map
import ase
from ase.io import read, write
import numpy as np
import jax.numpy as jnp
from jax import vmap
from jax import lax
from jax import grad
from jax import value_and_grad
from diffcg.util import custom_quantity
import pandas as pd
from scipy import interpolate as sci_interpolate
from diffcg.md.calculator import CustomEnergyCalculator,CustomCalculator
from ase import units
from diffcg.learning.reweighting import ReweightEstimator
import jax
from diffcg.observable.structure import initialize_inter_radial_distribution_fun, initialize_angle_distribution_fun, initialize_bond_distribution_fun, initialize_dihedral_distribution_fun, initialize_radial_distribution_fun
from jax import lax
from diffcg.io.ase_trj import read_ase_trj
import warnings
from diffcg.common.error import MSE
from diffcg.learning.reweighting import init_independent_mse_loss_fn
import optax
from diffcg.md.sample import MolecularDynamics
from diffcg.io.lammps import read_lammps_data
from ase import Atoms
from diffcg.util import custom_interpolate
# Suppress all UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# Or suppress specific warning messages
warnings.filterwarnings("ignore", message="specific warning text")
Temperature=600

def get_target_dict(temp):
  pressure_conversion = 16.6054  # from kJ/mol nm^-3 to bar
  pressure_target = 1. / pressure_conversion  # 1 bar in kJ / mol nm^3

  bdf_bin_centers, bdf_bin_boundaries, sigma_BDF = custom_quantity.bdf_discretization(1.0, nbins=200, BDF_start=0.)  # cut RDF at 1nm
  bond_data=pd.read_csv('/home/zhenghaowu/development/diffCG/examples/test_gradCG_polystyrene/datasets/T'+str(temp)+'/bondAA_smooth.dist.tgt',sep="\s+",header=None)
  reference_bdf=bond_data[[0,1]].values
  bdf_spline = sci_interpolate.interp1d(reference_bdf[:, 0], reference_bdf[:, 1], kind='cubic')
  reference_bdf = bdf_spline(bdf_bin_centers)
  reference_bdf[reference_bdf<1e-7]=0
  bond_top=pd.read_csv('/home/zhenghaowu/development/diffCG/examples/test_gradCG_polystyrene/datasets/polymer/bond.csv',header=None,sep='\s+')
  bond_top=bond_top.values-1
  bdf_struct = custom_quantity.BDFParams(reference_bdf, bdf_bin_centers, bdf_bin_boundaries, sigma_BDF,bond_top)

  adf_bin_centers, adf_bin_boundaries, sigma_ADF = custom_quantity.adf_discretization(np.pi, nbins=200, ADF_start=0.00)  # cut RDF at 1nm
  angle_data=pd.read_csv('/home/zhenghaowu/development/diffCG/examples/test_gradCG_polystyrene/datasets/T'+str(temp)+'/angleAAA.dist.tgt',sep="\s+",header=None)
  reference_adf=angle_data[[0,1]].values
  adf_spline = sci_interpolate.interp1d(reference_adf[:, 0], reference_adf[:, 1], kind='cubic')
  reference_adf = adf_spline(adf_bin_centers)
  reference_adf[reference_adf<1e-7]=0
  angle_top=pd.read_csv('/home/zhenghaowu/development/diffCG/examples/test_gradCG_polystyrene/datasets/polymer/angle.csv',header=None,sep='\s+')
  angle_top=angle_top.values-1
  adf_struct = custom_quantity.ADFParams(reference_adf, adf_bin_centers, adf_bin_boundaries, sigma_ADF,angle_top)


  ddf_bin_centers, ddf_bin_boundaries, sigma_DDF = custom_quantity.ddf_discretization(3.14, nbins=200, DDF_start=-3.14)  # cut RDF at 1nm
  dihedral_data=pd.read_csv('/home/zhenghaowu/development/diffCG/examples/test_gradCG_polystyrene/datasets/T'+str(temp)+'/dihedralAAAA.dist.tgt',sep="\s+",header=None)
  reference_ddf=dihedral_data[[0,1]].values
  ddf_spline = sci_interpolate.interp1d(reference_ddf[:, 0], reference_ddf[:, 1], kind='cubic')
  reference_ddf = ddf_spline(ddf_bin_centers)
  reference_ddf[reference_ddf<1e-7]=0
  dihedral_top=pd.read_csv('/home/zhenghaowu/development/diffCG/examples/test_gradCG_polystyrene/datasets/polymer/dihedral.csv',header=None,sep='\s+')
  dihedral_top=dihedral_top.values-1
  ddf_struct = custom_quantity.DDFParams(reference_ddf, ddf_bin_centers, ddf_bin_boundaries, sigma_DDF,dihedral_top)

  
  rdf_bin_centers, rdf_bin_boundaries, sigma_RDF = custom_quantity.rdf_discretization(RDF_cut=2.0)  # cut RDF at 1nm
  mask = np.ones((500, 500))
  mask[dihedral_top[:,0], dihedral_top[:, 1]] = 0
  mask[dihedral_top[:,1], dihedral_top[:, 0]] = 0

  mask[dihedral_top[:,2], dihedral_top[:, 0]] = 0
  mask[dihedral_top[:,0], dihedral_top[:, 2]] = 0

  mask[dihedral_top[:,3], dihedral_top[:, 0]] = 0
  mask[dihedral_top[:,0], dihedral_top[:, 3]] = 0

  mask[dihedral_top[:,2], dihedral_top[:, 1]] = 0
  mask[dihedral_top[:,1], dihedral_top[:, 2]] = 0

  mask[dihedral_top[:,3], dihedral_top[:, 1]] = 0
  mask[dihedral_top[:,1], dihedral_top[:, 3]] = 0

  mask[dihedral_top[:,3], dihedral_top[:, 2]] = 0
  mask[dihedral_top[:,2], dihedral_top[:, 3]] = 0
  polymer_exclude=jnp.array(mask)
  reference_rdf = pd.read_csv('/home/zhenghaowu/development/diffCG/examples/test_gradCG_polystyrene/datasets/T'+str(temp)+'/nb_smoothed.dist.tgt',header=None,sep='\s+')
  reference_rdf=reference_rdf[[0,1]].values
  rdf_spline = sci_interpolate.interp1d(reference_rdf[:, 0], reference_rdf[:, 1], kind='cubic')
  reference_rdf = rdf_spline(rdf_bin_centers)
  rdf_struct = custom_quantity.InterRDFParams(reference_rdf, rdf_bin_centers, rdf_bin_boundaries, sigma_RDF,polymer_exclude)

  target_dict = {'rdf': rdf_struct, 'bdf': bdf_struct, 'adf': adf_struct, 'ddf':ddf_struct, 'pressure': pressure_target}
  return target_dict,bond_top,angle_top,dihedral_top

target_dict,bond_top,angle_top,dihedral_top=get_target_dict(Temperature)

r_cut=2.0
r_onset=1.5
spline_grid_pair = jnp.linspace(0.4, r_cut, 80)

angle_limit=[0.1,3.14, 55]
bond_limit=[0.1,1.0,45]
dihedral_limit=[-3.14,3.14,100]

spline_grid_angle = jnp.linspace(angle_limit[0],angle_limit[1],angle_limit[2])
spline_grid_bond = jnp.linspace(bond_limit[0],bond_limit[1],bond_limit[2])
spline_grid_dihedral = jnp.linspace(dihedral_limit[0],dihedral_limit[1],dihedral_limit[2])

def build_energy_fn_with_params(params,max_num_atoms=1):
    pair_energy_fn = energy.TabulatedPairEnergy(spline_grid_pair, params['pair'], r_onset, r_cut,mask_topology=angle_top,max_num_atoms=max_num_atoms).get_energy_fn()
    bond_energy_fn = energy.TabulatedBondEnergy(spline_grid_bond, params['bond'],bond_top).get_energy_fn()
    angle_energy_fn = energy.TabulatedAngleEnergy(spline_grid_angle, params['angle'],angle_top).get_energy_fn()
    dihedral_energy_fn = energy.TabulatedDihedralEnergy(spline_grid_dihedral, params['dihedral'],dihedral_top).get_energy_fn()

    prior_fn = energy.GenericRepulsionEnergy(sigma=0.6, epsilon=1., exp=8,mask_topology=angle_top,max_num_atoms=max_num_atoms).get_energy_fn()
    bond_pot_fn=energy.HarmonicBondEnergy(bonds=bond_top,length=0.45,epsilon=5000).get_energy_fn()
    angle_pot_fn=energy.HarmonicAngleEnergy(angles=angle_top,angle_0=1.5,epsilon=50).get_energy_fn()
    dihedral_pot_fn=energy.HarmonicDihedralEnergy(dihedrals=dihedral_top,angle_0=1.5,epsilon=50).get_energy_fn()


    energy_fn_list = [pair_energy_fn,bond_energy_fn,angle_energy_fn,dihedral_energy_fn,bond_pot_fn,angle_pot_fn,dihedral_pot_fn,prior_fn]
    def energy_fn(system, neighbors, **dynamic_kwargs):
        return sum(energy_fn(system, neighbors, **dynamic_kwargs) for energy_fn in energy_fn_list) * units.kJ/ units.mol 
    
    return energy_fn

def rerun_energy(params, traj):
    results = []
    energy_fn = build_energy_fn_with_params(params,max_num_atoms=500)
    calculator = CustomEnergyCalculator(energy_fn,cutoff=r_cut)
    for atoms in traj:
        calculator.calculate(atoms)
        results.append(calculator.results)
        #print(results)
    return jnp.stack(results)

pretrained_params = np.load('pretrained_params.npy',allow_pickle=True).item()

quantity_dict = {}
if 'rdf' in target_dict:
    rdf_struct = target_dict['rdf']
    rdf_fn = initialize_inter_radial_distribution_fun(rdf_struct)
    rdf_dict = {'compute_fn': rdf_fn, 'target': rdf_struct.reference_rdf, 'gamma': 1.}
    quantity_dict['rdf'] = rdf_dict
    
if 'bdf' in target_dict:
    bdf_struct = target_dict['bdf']
    bdf_fn = initialize_bond_distribution_fun(bdf_struct)
    bdf_dict = {'compute_fn': bdf_fn, 'target': bdf_struct.reference_bdf, 'gamma': 1.0/1000}
    quantity_dict['bdf'] = bdf_dict
    
if 'adf' in target_dict:
    adf_struct = target_dict['adf']
    adf_fn = initialize_angle_distribution_fun(adf_struct)
    adf_dict = {'compute_fn': adf_fn, 'target': adf_struct.reference_adf, 'gamma': 1./10}
    quantity_dict['adf'] = adf_dict

if 'ddf' in target_dict:
    ddf_struct = target_dict['ddf']
    ddf_fn = initialize_dihedral_distribution_fun(ddf_struct)
    ddf_dict = {'compute_fn': ddf_fn, 'target': ddf_struct.reference_adf, 'gamma': 1.}
    quantity_dict['ddf'] = ddf_dict


def calculate_observables(traj_file='sample.traj'):
    systems = read_ase_trj(traj_file)
    batched_systems = tree_map(lambda *xs: jnp.stack(xs), *systems)  # R: (B,500,3), Z: (B,500), cell: (B,3,3)

    B = batched_systems.R.shape[0]
    proto = quantity_dict['rdf']['compute_fn'](systems[0])
    proto_bdf = quantity_dict['bdf']['compute_fn'](systems[0])
    proto_adf = quantity_dict['adf']['compute_fn'](systems[0])
    proto_ddf = quantity_dict['ddf']['compute_fn'](systems[0])

    def body(carry, i):
        sys_i = tree_map(lambda x: x[i], batched_systems)
        val = quantity_dict['rdf']['compute_fn'](sys_i)
        return carry + val, val

    _, rdfs = lax.scan(body, jnp.zeros_like(proto), jnp.arange(B))

    def body_bdf(carry, i):
        sys_i = tree_map(lambda x: x[i], batched_systems)
        val = quantity_dict['bdf']['compute_fn'](sys_i)
        return carry + val, val
    _, bdfs = lax.scan(body_bdf, jnp.zeros_like(proto_bdf), jnp.arange(B))  

    def body_adf(carry, i):
        sys_i = tree_map(lambda x: x[i], batched_systems)
        val = quantity_dict['adf']['compute_fn'](sys_i)
        return carry + val, val
    _, adfs = lax.scan(body_adf, jnp.zeros_like(proto_adf), jnp.arange(B))  

    def body_ddf(carry, i):
        sys_i = tree_map(lambda x: x[i], batched_systems)
        val = quantity_dict['ddf']['compute_fn'](sys_i)
        return carry + val, val
    _, ddfs = lax.scan(body_ddf, jnp.zeros_like(proto_ddf), jnp.arange(B))  

    observables={'rdf':rdfs,'bdf':bdfs,'adf':adfs,'ddf':ddfs}
    return observables

loss_fn = init_independent_mse_loss_fn(quantity_dict)

lammpsdata_file = '/home/zhenghaowu/development/diffCG/examples/test_gradCG_polystyrene/datasets/T600/PS.data'
sys_data = read_lammps_data(lammpsdata_file)
# Mapping of atom types
atom_types_mapping = {0: 'C', 1: 'C', 2: 'H'}
data_element_figurelist = sys_data["atom_types"]
data_element = [atom_types_mapping[i] for i in data_element_figurelist]

# Extract coordinates and cell information
data_coord = sys_data["coords"]/10.0 #A to nm
cell = sys_data["cells"][0]/10.0 #A to nm

data_coord = data_coord[0]

# Create ASE Atoms object
init_atoms = Atoms(positions=data_coord, symbols=data_element, cell=cell, pbc=jnp.array([True,True,True]))
_masses=[104.0 for i in range(500)]
init_atoms.set_masses(_masses)

initial_lr = 0.1
lr_schedule = optax.exponential_decay(-initial_lr, 200, 0.005)
optimizer = optax.chain(
    optax.scale_by_adam(0.9, 0.99),
    optax.scale_by_schedule(lr_schedule)
)
opt_state = optimizer.init(pretrained_params)

def update(step, params, old_params, opt_state):
    sample_energy_fn = build_energy_fn_with_params(params,max_num_atoms=500)
    calculator = CustomCalculator(sample_energy_fn,cutoff=r_cut)
    sample_md = MolecularDynamics(init_atoms, 
                            custom_calculator=calculator, 
                            ensemble="nvt", 
                            thermostat="berendsen", 
                            temperature=Temperature, 
                            starting_temperature=Temperature,
                            timestep=1, 
                            trajectory=f"sample{step}.traj",
                            logfile=f"sample{step}.log",
                            loginterval=12)
    sample_md.run(60*1000)
    observables = calculate_observables(f"sample{step}.traj")
    trajs = read(f"sample{step}.traj",index=':')

    results = rerun_energy(old_params, trajs)
    estimator = ReweightEstimator(results, base_energies=None, volume=None)

    def loss(params):
        energies = rerun_energy(params, trajs)
        weight, n_eff = estimator.estimate_weight(energies)
        loss, predictions = loss_fn(observables, weight)
        return loss, predictions

    v_and_g = jax.value_and_grad(loss,has_aux=True)
    outputs, curr_grad = v_and_g(params)
    loss_val, predictions = outputs

    scaled_grad, opt_state = optimizer.update(curr_grad, opt_state, params)
    new_params = optax.apply_updates(params, scaled_grad)   
    return new_params, params, opt_state, loss_val, predictions

params = pretrained_params
new_params = params
import time
import matplotlib.pyplot as plt

rdf_bin_centers, rdf_bin_boundaries, sigma_RDF = custom_quantity.rdf_discretization(RDF_cut=2.0)  # cut RDF at 1nm
bdf_bin_centers, bdf_bin_boundaries, sigma_BDF = custom_quantity.bdf_discretization(1.0, nbins=200, BDF_start=0.)  # cut RDF at 1nm
adf_bin_centers, adf_bin_boundaries, sigma_ADF = custom_quantity.adf_discretization(np.pi, nbins=200, ADF_start=0.00)  # cut RDF at 1nm
ddf_bin_centers, ddf_bin_boundaries, sigma_DDF = custom_quantity.ddf_discretization(3.14, nbins=200, DDF_start=-3.14)  # cut RDF at 1nm


r_cut=2.0
r_onset=1.5

r_cutoff=r_cut
r_c = r_cutoff ** 2
r_o = r_onset ** 2

def smooth_fn(dr):
    r = dr ** 2

    inner = jnp.where(dr < r_cutoff,
                     (r_c - r)**2 * (r_c + 2 * r - 3 * r_o) / (r_c - r_o)**3,
                     0)

    return jnp.where(dr < r_onset, 1, inner)


loss_history, times_per_update, predicted_quantities, params_set = [], [], [], []
for step in range(10): 
    start_time = time.time()
    new_params, params,opt_state, loss_val, predictions = update(step, new_params, params, opt_state)
    print(loss_val)

    step_time = time.time() - start_time
    print("Step {} in {:0.2f} sec".format(step, step_time), 'Loss = ', loss_val, '\n')
    if jnp.isnan(loss_val):  # stop learning when optimization diverged
        print('Loss is NaN. This was likely caused by divergence of the optimization or a bad model setup '
              'causing a NaN trajectory.')
    params_set.append(params)
    times_per_update.append(step_time)
    loss_history.append(loss_val)
    predicted_quantities.append(predictions)
    
    if step%1==0:
        
        fig, ax = plt.subplots(2, 5, figsize=(24, 8))
        
        rdf_series = [prediction_dict['rdf'] for prediction_dict in predicted_quantities]

        ax[0][0].set_xlabel('$r$ in nm')
        ax[0][0].set_ylabel('RDF')
        ax[0][0].plot(rdf_bin_centers, rdf_series[0], label='initial guess', color='#00a087ff')
        ax[0][0].plot(rdf_bin_centers, rdf_series[-1], label='predicted', color='#3c5488ff')
        ax[0][0].plot(rdf_bin_centers, quantity_dict['rdf']['target'], label='target', linestyle='--', color='k')
        ax[0][0].legend()

        bdf_series = [prediction_dict['bdf'] for prediction_dict in predicted_quantities]

        ax[0][1].set_xlabel('$r$ in nm')
        ax[0][1].set_ylabel('BDF')
        ax[0][1].plot(bdf_bin_centers, bdf_series[0], label='initial guess', color='#00a087ff')
        ax[0][1].plot(bdf_bin_centers, bdf_series[-1], label='predicted', color='#3c5488ff')
        ax[0][1].plot(bdf_bin_centers, quantity_dict['bdf']['target'], label='target', linestyle='--', color='k')
        ax[0][1].legend()

        adf_series = [prediction_dict['adf'] for prediction_dict in predicted_quantities]

        ax[0][2].set_xlabel('$r$ in nm')
        ax[0][2].set_ylabel('ADF')
        ax[0][2].plot(adf_bin_centers, adf_series[0], label='initial guess', color='#00a087ff')
        ax[0][2].plot(adf_bin_centers, adf_series[-1], label='predicted', color='#3c5488ff')
        ax[0][2].plot(adf_bin_centers, quantity_dict['adf']['target'], label='target', linestyle='--', color='k')
        ax[0][2].legend()

        ddf_series = [prediction_dict['ddf'] for prediction_dict in predicted_quantities]

        ax[0][3].set_xlabel('$r$ in nm')
        ax[0][3].set_ylabel('DDF')
        ax[0][3].plot(ddf_bin_centers, ddf_series[0], label='initial guess', color='#00a087ff')
        ax[0][3].plot(ddf_bin_centers, ddf_series[-1], label='predicted', color='#3c5488ff')
        ax[0][3].plot(ddf_bin_centers, quantity_dict['ddf']['target'], label='target', linestyle='--', color='k')
        ax[0][3].legend()

        spline_grid_pair = jnp.linspace(0.4, r_cut, 80)
        spline_grid_angle = jnp.linspace(angle_limit[0],angle_limit[1],angle_limit[2])
        spline_grid_bond = jnp.linspace(bond_limit[0],bond_limit[1],bond_limit[2])


        spline_energy = custom_interpolate.MonotonicInterpolate(spline_grid_pair, params['pair'])
        ax[1][0].set_xlabel('$r$ in nm')
        ax[1][0].set_ylabel('potential')
        ax[1][0].set_ylim(-2,3)
        ax[1][0].set_xlim(0.4,2)
        ax[1][0].plot(spline_grid_pair, smooth_fn(spline_grid_pair)*(spline_energy(spline_grid_pair)+energy.generic_repulsion(spline_grid_pair,sigma=0.6,epsilon=1,exp=8)), label='current', color='#3c5488ff')
        ax[1][0].legend()

        spline_energy = custom_interpolate.MonotonicInterpolate(spline_grid_bond, params['bond'])
        ax[1][1].set_xlabel('$r$ in nm')
        ax[1][1].set_ylabel('potential')
        ax[1][1].set_ylim(-2,10)
        ax[1][1].set_xlim(0.0,1.0)
        ax[1][1].plot(spline_grid_bond, spline_energy(spline_grid_bond)+energy.simple_spring(spline_grid_bond, length=0.45, epsilon=5000), label='current', color='#3c5488ff')
        ax[1][1].legend()

        spline_energy = custom_interpolate.MonotonicInterpolate(spline_grid_angle, params['angle'])
        ax[1][2].set_xlabel('$r$ in nm')
        ax[1][2].set_ylabel('potential')
        ax[1][2].set_ylim(-2,10)
        ax[1][2].set_xlim(0.0,3.14)
        ax[1][2].plot(spline_grid_angle, spline_energy(spline_grid_angle)+energy.harmonic_angle(spline_grid_angle,angle_0=1.5, epsilon=50), label='current', color='#3c5488ff')
        ax[1][2].legend()

        spline_energy = custom_interpolate.MonotonicInterpolate(spline_grid_dihedral, params['dihedral'])
        ax[1][3].set_xlabel('$r$ in nm')
        ax[1][3].set_ylabel('potential')
        ax[1][3].set_ylim(-2,10)
        ax[1][3].set_xlim(-3.14,3.14)
        ax[1][3].plot(spline_grid_dihedral, spline_energy(spline_grid_dihedral)+energy.harmonic_dihedral(spline_grid_dihedral,angle_0=1.5, epsilon=50), label='current', color='#3c5488ff')
        ax[1][3].legend()

        ax[1][4].set_xlabel('step')
        ax[1][4].set_ylabel('loss')
        ax[1][4].plot(jnp.arange(0,step+1,1),loss_history, label='predicted', color='#3c5488ff')
        ax[1][4].legend()

        plt.savefig('training_'+str(step)+'.png')
        plt.close()
