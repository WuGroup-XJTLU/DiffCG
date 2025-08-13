
import dpdata
from ase import Atoms
from ase import units
from ase.units import bar
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.trajectory import Trajectory
from ase.io import read, write
from ase.md.npt import NPT
import matplotlib.pyplot as plt
import jax.numpy as jnp
from chemfiles import Trajectory, Frame
import jax
from scipy import interpolate as sci_interpolate
import numpy as np
import pandas as pd
import optax
from jax import jit, random
import time

import sys
sys.path.append('/home/zhenghaowu/development/diffCG')
from diffcg.md.calculator import CustomCalculator
from diffcg import energy
from diffcg.io.lammps import read_lammps_data
from diffcg.util import custom_interpolate
from diffcg.util import custom_quantity
from functools import partial
# Load the system data from the LAMMPS data file
lammpsdata_file = '/home/zhenghaowu/development/diffCG/examples/test_gradCG_polystyrene/datasets/T600/PS.data'
sys_data = read_lammps_data(lammpsdata_file)
# Mapping of atom types
atom_types_mapping = {0: 'C', 1: 'C', 2: 'H'}
data_element_figurelist = sys_data["atom_types"]
data_element = [atom_types_mapping[i] for i in data_element_figurelist]

# Extract coordinates and cell information
data_coord = sys_data["coords"]
cell = sys_data["cells"][0]

data_coord = data_coord[0]

# Create ASE Atoms object
atoms = Atoms(positions=data_coord, symbols=data_element, cell=cell, pbc=jnp.array([True,True,True]))

r_cut=2.0
spline_grid_pair = jnp.linspace(0.4, r_cut, 80)

angle_limit=[0.1,3.14, 55]
bond_limit=[0.1,1.0,45]
dihedral_limit=[-3.14,3.14,100]

Temperature=600
random_key1=2
random_key2=3

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

def pair_pot(r,params,x_vals):
  spline = custom_interpolate.MonotonicInterpolate(x_vals, params)
  tabulated_partial = partial(energy.tabulated, spline=spline)
  generic_repulsion=partial(energy.generic_repulsion, sigma=0.6, epsilon=1.0, exp=8)
  U=tabulated_partial(r)+generic_repulsion(r)
  return U

def bond_pot(r,params,x_vals):
  spline = custom_interpolate.MonotonicInterpolate(x_vals, params)
  tabulated_partial = partial(energy.tabulated, spline=spline)
  harmonic=partial(energy.simple_spring, length=0.45, epsilon=5000)
  U=tabulated_partial(r)+harmonic(r)
  return U

def angle_pot(r,params,x_vals):
  spline = custom_interpolate.MonotonicInterpolate(x_vals, params)
  tabulated_partial = partial(energy.tabulated, spline=spline)

  harmonic=partial(energy.harmonic_angle, angle_0=1.5, epsilon=50)
  U=tabulated_partial(r)+harmonic(r)
  return U

def dihedral_pot(r,params,x_vals):
  spline = custom_interpolate.MonotonicInterpolate(x_vals, params)
  tabulated_partial = partial(energy.tabulated, spline=spline)

  harmonic=partial(energy.harmonic_dihedral, angle_0=1.5, epsilon=50)
  U=tabulated_partial(r)+harmonic(r)
  return U

def Boltzmann_Inversion(kbT,dist):
  U_BI=-kbT*jnp.log(dist)
  return U_BI

def calc_kBT(real_T):
  system_temperature = real_T  # Kelvin = 23 deg. Celsius
  Boltzmann_constant = 0.0083145107  # in kJ / mol K
  kbT = system_temperature * Boltzmann_constant
  return kbT

Temperature = 600

target={}
x_range={}
U_BI=Boltzmann_Inversion(calc_kBT(Temperature),target_dict['rdf'].reference_rdf)-Boltzmann_Inversion(calc_kBT(Temperature),target_dict['rdf'].reference_rdf)[-1]
x_range['pair']=target_dict['rdf'].rdf_bin_centers[target_dict['rdf'].rdf_bin_centers>0.4]
target['pair']=U_BI[target_dict['rdf'].rdf_bin_centers>0.4]

U_BI=Boltzmann_Inversion(calc_kBT(Temperature),target_dict['bdf'].reference_bdf)-min(Boltzmann_Inversion(calc_kBT(Temperature),target_dict['bdf'].reference_bdf))
x_range['bond']=target_dict['bdf'].bdf_bin_centers[U_BI!=np.inf]
target['bond']=U_BI[U_BI!=np.inf]
spline_bond=sci_interpolate.interp1d(x_range['bond'],target['bond'],fill_value="extrapolate")

U_BI=Boltzmann_Inversion(calc_kBT(Temperature),target_dict['adf'].reference_adf)-min(Boltzmann_Inversion(calc_kBT(Temperature),target_dict['adf'].reference_adf))
x_range['angle']=target_dict['adf'].adf_bin_centers[U_BI!=np.inf]
target['angle']=U_BI[U_BI!=np.inf]
spline_angle=sci_interpolate.interp1d(x_range['angle'],target['angle'],fill_value="extrapolate")

U_BI=Boltzmann_Inversion(calc_kBT(Temperature),target_dict['ddf'].reference_adf)-min(Boltzmann_Inversion(calc_kBT(Temperature),target_dict['ddf'].reference_adf))
x_range['dihedral']=target_dict['ddf'].adf_bin_centers[U_BI!=np.inf]
target['dihedral']=U_BI[U_BI!=np.inf]
spline_dihedal=sci_interpolate.interp1d(x_range['dihedral'],target['dihedral'],fill_value="extrapolate")

r_cut=2.0
spline_grid_pair = jnp.linspace(0.4, r_cut, 80)

angle_limit=[0.1,3.14, 55]
bond_limit=[0.1,1.0,45]
dihedral_limit=[-3.14,3.14,100]
spline_grid_angle = jnp.linspace(angle_limit[0],angle_limit[1],angle_limit[2])
spline_grid_bond = jnp.linspace(bond_limit[0],bond_limit[1],bond_limit[2])
spline_grid_dihedral = jnp.linspace(dihedral_limit[0],dihedral_limit[1],dihedral_limit[2])


def loss_fn(param):
  energy_fn=partial(pair_pot,params=param[0],x_vals=spline_grid_pair)
  pair=jax.vmap(energy_fn)
  energy=pair(x_range['pair'])
  loss_values=jnp.sum((energy-target['pair'])**2)

  energy_fn=partial(bond_pot,params=param[1],x_vals=spline_grid_bond)
  pair=jax.vmap(energy_fn)
  energy=pair(x_range['bond'])
  loss_values+=jnp.sum((energy-spline_bond(x_range['bond']))**2)
  
  energy_fn=partial(angle_pot,params=param[2],x_vals=spline_grid_angle)
  pair=jax.vmap(energy_fn)
  energy=pair(spline_grid_angle)
  loss_values+=jnp.sum((energy-spline_angle(spline_grid_angle))**2)

  energy_fn=partial(dihedral_pot,params=param[3],x_vals=spline_grid_dihedral)
  pair=jax.vmap(energy_fn)
  energy=pair(spline_grid_dihedral)
  loss_values+=jnp.sum((energy-spline_dihedal(spline_grid_dihedral))**2)

  return loss_values

def pretrain():
  initial_lr = 1e-1
  lr_schedule = optax.exponential_decay(-initial_lr, 100000, 0.001)
  #lr_schedule = optax.warmup_exponential_decay_schedule(-initial_lr,-initial_lr,50, 950, 0.01, 50)
  optimizer = optax.chain(
      optax.scale_by_adam(0.9, 0.99),
      optax.scale_by_schedule(lr_schedule)
  )

  @jit
  def update(params, opt_state):
    loss_value, curr_grad=jax.value_and_grad(loss_fn)(params)
    scaled_grad, opt_state = optimizer.update(curr_grad, opt_state, params)
    new_params = optax.apply_updates(params, scaled_grad)
    return new_params, opt_state, curr_grad, loss_value

  loss_history, predicted_densities, gradients, times_per_epoch, learned_potentials = [], [], [], [], []

  key = random.PRNGKey(random_key2)  # define random seed for initialization of model and simulation
  model_init_key, simuation_init_key = random.split(key, 2)
  init_params_energy=0.001 * random.normal(model_init_key, spline_grid_pair.shape)
  init_params_bond=0.001 * random.normal(model_init_key, spline_grid_bond.shape)
  init_params_angle=0.001 * random.normal(model_init_key, spline_grid_angle.shape)
  init_params_dihedral=0.001 * random.normal(model_init_key, spline_grid_dihedral.shape)

  params=(init_params_energy,init_params_bond,init_params_angle,init_params_dihedral)
  opt_state = optimizer.init(params)
  
  num_updates = 10000
  for step in range(num_updates):
      start_time = time.time()
      params, opt_state, curr_grad, loss_val = update(params, opt_state)
      step_time = time.time() - start_time
      if step%10000==0:
        print("Step {} in {:0.2f} sec".format(step, step_time), 'loss = ', loss_val, '\n')
  print("Step {} in {:0.2f} sec".format(step, step_time), 'loss = ', loss_val, '\n')
      
  plt.plot(target_dict['rdf'].rdf_bin_centers,Boltzmann_Inversion(calc_kBT(Temperature),target_dict['rdf'].reference_rdf)-Boltzmann_Inversion(calc_kBT(Temperature),target_dict['rdf'].reference_rdf)[-1],'ro')
  plt.plot(target_dict['rdf'].rdf_bin_centers,pair_pot(target_dict['rdf'].rdf_bin_centers,params[0],spline_grid_pair),'k--')
  plt.ylim(-1,10)
  plt.savefig('rdf_pretrain.png')
  plt.close()
  plt.plot(target_dict['bdf'].bdf_bin_centers,Boltzmann_Inversion(calc_kBT(Temperature),target_dict['bdf'].reference_bdf)-min(Boltzmann_Inversion(calc_kBT(Temperature),target_dict['bdf'].reference_bdf)),'ro')
  plt.plot(target_dict['bdf'].bdf_bin_centers,bond_pot(target_dict['bdf'].bdf_bin_centers,params[1],spline_grid_bond),'k--')
  #plt.plot(target_dict['bdf'].bdf_bin_centers,spline_bond(target_dict['bdf'].bdf_bin_centers),'b--')
  plt.ylim(-1,20)
  plt.savefig('bdf_pretrain.png')
  plt.close()
  plt.plot(target_dict['adf'].adf_bin_centers,Boltzmann_Inversion(calc_kBT(Temperature),target_dict['adf'].reference_adf)-min(Boltzmann_Inversion(calc_kBT(Temperature),target_dict['adf'].reference_adf)),'ro')
  plt.plot(target_dict['adf'].adf_bin_centers,angle_pot(target_dict['adf'].adf_bin_centers,params[2],spline_grid_angle),'k--')
  #plt.plot(target_dict['adf'].adf_bin_centers,spline_angle(target_dict['adf'].adf_bin_centers),'b--')
  plt.ylim(-1,60)
  plt.savefig('adf_pretrain.png')
  plt.close()
  plt.plot(target_dict['ddf'].adf_bin_centers,Boltzmann_Inversion(calc_kBT(Temperature),target_dict['ddf'].reference_adf)-min(Boltzmann_Inversion(calc_kBT(Temperature),target_dict['ddf'].reference_adf)),'ro')
  plt.plot(target_dict['ddf'].adf_bin_centers,dihedral_pot(target_dict['ddf'].adf_bin_centers,params[3],spline_grid_dihedral),'k--')
  #plt.plot(target_dict['ddf'].adf_bin_centers,spline_dihedal(target_dict['ddf'].adf_bin_centers),'b--')
  plt.ylim(-1,20)
  plt.savefig('ddf_pretrain.png')
  plt.close()
  
  return params,angle_limit,bond_limit,dihedral_limit

pretrained_params,angle_limit,bond_limit,dihedral_limit=pretrain()

pretrained_params_dict={'pair':pretrained_params[0],'bond':pretrained_params[1],'angle':pretrained_params[2],'dihedral':pretrained_params[3]}

np.save('pretrained_params.npy',pretrained_params_dict)