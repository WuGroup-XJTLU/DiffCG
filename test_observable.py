from ase import Atoms
from ase import units
from ase.units import bar
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.trajectory import Trajectory as ase_trajectory
from ase.io import read, write
from ase.md.langevin import Langevin
import matplotlib.pyplot as plt
import jax.numpy as jnp
from chemfiles import Trajectory, Frame
import jax
from jax import vmap
from scipy import interpolate as sci_interpolate
import numpy as np
import pandas as pd
import optax
from jax import jit, random
import time
import ase

import sys
sys.path.append('/home/zhenghaowu/development/diffCG')
from diffcg.md.calculator import CustomCalculator
from diffcg import energy
from diffcg.io.lammps import read_lammps_data
from diffcg.io.ase_trj import read_ase_trj
from diffcg.util import custom_interpolate
from diffcg.util import custom_quantity
from functools import partial
from diffcg.system import atoms_to_system
from diffcg.observable.structure import initialize_inter_radial_distribution_fun, initialize_angle_distribution_fun, initialize_bond_distribution_fun, initialize_dihedral_distribution_fun, initialize_radial_distribution_fun


Temperature = 600

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

systems = read_ase_trj('sample.traj')
print(systems[0])

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

from jax.tree_util import tree_map

batched_systems = tree_map(lambda *xs: jnp.stack(xs), *systems)  # R: (B,500,3), Z: (B,500), cell: (B,3,3)
bdf = jnp.mean(vmap(quantity_dict['bdf']['compute_fn'])(batched_systems), axis=0)
bdf_series = [bdf]
bdf_bin_centers = target_dict['bdf'].bdf_bin_centers

adf = jnp.mean(vmap(quantity_dict['adf']['compute_fn'])(batched_systems), axis=0)
adf_series = [adf]
adf_bin_centers = target_dict['adf'].adf_bin_centers

from jax import lax
B = batched_systems.R.shape[0]
proto = quantity_dict['rdf']['compute_fn'](systems[0])
def body(carry, i):
    sys_i = tree_map(lambda x: x[i], batched_systems)
    val = quantity_dict['rdf']['compute_fn'](sys_i)
    return carry + val, None

total, _ = lax.scan(body, jnp.zeros_like(proto), jnp.arange(B))
rdf = total / B
rdf_series = [rdf]
rdf_bin_centers = target_dict['rdf'].rdf_bin_centers

fig, ax = plt.subplots(3, 1, figsize=(12, 12))
ax[0].set_xlabel('$r$ in nm')
ax[0].set_ylabel('BDF')
ax[0].plot(bdf_bin_centers, bdf_series[0], label='predicted', color='#00a087ff')
ax[0].plot(bdf_bin_centers, quantity_dict['bdf']['target'], label='target', linestyle='--', color='k')
ax[1].set_xlabel('$\theta$ in rad')
ax[1].set_ylabel('ADF')
ax[1].plot(adf_bin_centers, adf_series[0], label='predicted', color='#00a087ff')
ax[1].plot(adf_bin_centers, quantity_dict['adf']['target'], label='target', linestyle='--', color='k')
ax[2].set_xlabel('$r$ in nm')
ax[2].set_ylabel('RDF')
ax[2].plot(rdf_bin_centers, rdf_series[0], label='predicted', color='#00a087ff')
ax[2].plot(rdf_bin_centers, quantity_dict['rdf']['target'], label='target', linestyle='--', color='k')
ax[2].legend()
plt.savefig('structure_test.png')
plt.close()