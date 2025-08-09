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
from diffcg.md.sample import MolecularDynamics

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


# Load the system data from the LAMMPS data file
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
atoms = Atoms(positions=data_coord, symbols=data_element, cell=cell, pbc=jnp.array([True,True,True]))
max_num_atoms=data_coord.shape[0]
r_cut=2.0
r_onset=1.5
spline_grid_pair = jnp.linspace(0.4, r_cut, 80)

angle_limit=[0.1,3.14, 55]
bond_limit=[0.1,1.0,45]
dihedral_limit=[-3.14,3.14,100]

spline_grid_angle = jnp.linspace(angle_limit[0],angle_limit[1],angle_limit[2])
spline_grid_bond = jnp.linspace(bond_limit[0],bond_limit[1],bond_limit[2])
spline_grid_dihedral = jnp.linspace(dihedral_limit[0],dihedral_limit[1],dihedral_limit[2])

pretrained_params = np.load('pretrained_params.npy',allow_pickle=True).item()

pair_energy_fn = energy.TabulatedPairEnergy(spline_grid_pair, pretrained_params['pair'], r_onset, r_cut,mask_topology=angle_top,max_num_atoms=max_num_atoms).get_energy_fn()
bond_energy_fn = energy.TabulatedBondEnergy(spline_grid_bond, pretrained_params['bond'],bond_top).get_energy_fn()
angle_energy_fn = energy.TabulatedAngleEnergy(spline_grid_angle, pretrained_params['angle'],angle_top).get_energy_fn()
dihedral_energy_fn = energy.TabulatedDihedralEnergy(spline_grid_dihedral, pretrained_params['dihedral'],dihedral_top).get_energy_fn()

prior_fn = energy.GenericRepulsionEnergy(sigma=0.6, epsilon=1., exp=8,mask_topology=angle_top,max_num_atoms=max_num_atoms).get_energy_fn()
bond_pot_fn=energy.HarmonicBondEnergy(bonds=bond_top,length=0.45,epsilon=5000).get_energy_fn()
angle_pot_fn=energy.HarmonicAngleEnergy(angles=angle_top,angle_0=1.5,epsilon=50).get_energy_fn()
dihedral_pot_fn=energy.HarmonicDihedralEnergy(dihedrals=dihedral_top,angle_0=1.5,epsilon=50).get_energy_fn()

def get_sum_energy_fn(energy_fn_list):
    def sum_energy_fn(system, neighbors, **dynamic_kwargs):
        return sum(energy_fn(system, neighbors, **dynamic_kwargs) for energy_fn in energy_fn_list) * units.kJ/ units.mol 
    
    return sum_energy_fn

total_energy_fn = get_sum_energy_fn([pair_energy_fn,bond_energy_fn,angle_energy_fn,dihedral_energy_fn,bond_pot_fn,angle_pot_fn,dihedral_pot_fn,prior_fn])
# Set the calculator
calculator = CustomCalculator(total_energy_fn,cutoff=r_cut)

calculator.calculate(atoms)
print(calculator.results)

_masses=[104.0 for i in range(500)]
atoms.set_masses(_masses)


sample_md = MolecularDynamics(atoms, 
                            custom_calculator=calculator, 
                            ensemble="nvt", 
                            thermostat="langevin", 
                            temperature=Temperature, 
                            starting_temperature=Temperature,
                            timestep=1.0, 
                            friction=0.01,
                            trajectory="sample.traj",
                            logfile="sample.log",
                            loginterval=100)
sample_md.run(10000)