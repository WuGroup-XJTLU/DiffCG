import os

# Configure GPU settings
os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # Use GPU 0
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Disable memory preallocation


from jax.tree_util import tree_map
from ase.io import read
import numpy as np
import jax.numpy as jnp
import pandas as pd
from scipy import interpolate as sci_interpolate
from ase import units
import os

import sys
sys.path.append('/home/zhenghaowu/development/diffCG')

from diffcg.learning.ibi import (
    IBITargets,
    IBIConfig,
    IterativeBoltzmannInversion,
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
from diffcg.io.ase_trj import read_ase_trj
from diffcg.observable.analyze import analyze
from diffcg.io.lammps import read_lammps_data
from ase import Atoms
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from diffcg import configure_logging

configure_logging(level="DEBUG")

Temperature = 600
Boltzmann_constant = 0.0083145107  # kJ/mol/K

DATASET_ROOT = \
    "/home/zhenghaowu/development/diffCG/examples/test_gradCG_polystyrene/datasets"


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


def get_target_dict(temp,num_beads=500):
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
    rdf_bin_centers, rdf_bin_boundaries, sigma_RDF = rdf_discretization(
        RDF_cut=2.0
    )
    mask = np.ones((num_beads, num_beads))
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
        "rdf": rdf_struct,
        "bdf": bdf_struct,
        "adf": adf_struct,
        "ddf": ddf_struct,
    }
    return target_dict, bond_top, angle_top, dihedral_top


def _build_init_atoms(num_beads=500):
    lammpsdata_file = (
        "/home/zhenghaowu/development/diffCG/examples/test_gradCG_polystyrene/datasets/T600/PS.data"
    )
    sys_data = read_lammps_data(lammpsdata_file)
    atom_types_mapping = {0: "C", 1: "C", 2: "H"}
    data_element_figurelist = sys_data["atom_types"]
    data_element = [atom_types_mapping[i] for i in data_element_figurelist]
    data_coord = sys_data["coords"][0] / 10.0  # A to nm
    cell = sys_data["cells"][0] / 10.0  # A to nm
    init_atoms = Atoms(positions=data_coord, symbols=data_element, cell=cell, pbc=jnp.array([True, True, True]))
    _masses = [104.0 for _ in range(num_beads)]
    init_atoms.set_masses(_masses)
    return init_atoms


def test_ibi_ps_system_smoke():
    # Build targets and atoms as in functional tests
    target_dict, bond_top, angle_top, dihedral_top = get_target_dict(Temperature)
    init_atoms = _build_init_atoms()

    # Prepare IBI inputs
    targets = IBITargets(
        rdf=target_dict["rdf"],
        bdf=target_dict["bdf"],
        adf=target_dict["adf"],
        ddf=target_dict["ddf"],
    )

    r_cut = 2.0
    r_onset = 1.5
    sim_time_scheme = {"equilibration_steps": 1000, "production_steps": 5000}
    sampler_params = {
        "ensemble": "nvt",
        "thermostat": "berendsen",
        "temperature": Temperature,
        "starting_temperature": Temperature,
        "timestep": 4,
        "trajectory": "ibi_ps_",
        "logfile": "ibi_ps_",
        "loginterval": 10,
    }

    cfg = IBIConfig(
        r_cut=r_cut,
        r_onset=r_onset,
        sim_time_scheme=sim_time_scheme,
        sampler_params=sampler_params,
        alpha_pair=0.1,
        alpha_bond=0.1,
        alpha_angle=0.1,
        alpha_dihedral=0.1,
        smooth_window_pair=5,
        smooth_window_bond=3,
        smooth_window_angle=3,
        smooth_window_dihedral=3,
        max_iters=3,
        trajectory_prefix="ibi_ps_",
        logfile_prefix="ibi_ps_",
    )

    ibi = IterativeBoltzmannInversion(
        kBT=Temperature * Boltzmann_constant,
        init_atoms=init_atoms,
        targets=targets,
        config=cfg,
        mask_topology=angle_top,
        max_num_atoms=init_atoms.get_global_number_of_atoms(),
    )

    # Pre-run assertions on initialization
    x_pair, U_pair = ibi.x_pair, ibi.U_pair
    x_bond, U_bond = ibi.x_bond, ibi.U_bond
    x_angle, U_angle = ibi.x_angle, ibi.U_angle
    x_dihedral, U_dihedral = ibi.x_dihedral, ibi.U_dihedral

    assert x_pair is not None and U_pair is not None and x_pair.shape == U_pair.shape
    assert x_bond is not None and U_bond is not None and x_bond.shape == U_bond.shape
    assert x_angle is not None and U_angle is not None and x_angle.shape == U_angle.shape
    assert x_dihedral is not None and U_dihedral is not None and x_dihedral.shape == U_dihedral.shape

    # Run one IBI iteration (short MD)
    result = ibi.run()
    print(result)
    # Check outputs exist and have consistent shapes
    (xp, Up) = result["pair"]
    (xb, Ub) = result["bond"]
    (xa, Ua) = result["angle"]
    (xd, Ud) = result["dihedral"]
    assert xp.shape == Up.shape == x_pair.shape
    assert xb.shape == Ub.shape == x_bond.shape
    assert xa.shape == Ua.shape == x_angle.shape
    assert xd.shape == Ud.shape == x_dihedral.shape

    # Check history has at least one entry and contains rmse keys
    assert len(result["history"]) >= 1
    rmse = result["history"][0]["rmse"]
    assert "rdf" in rmse and "bdf" in rmse and "adf" in rmse and "ddf" in rmse


test_ibi_ps_system_smoke()