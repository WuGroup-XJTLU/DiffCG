"""Shared utilities for polystyrene coarse-graining examples.

Provides data loading, target construction, energy builder, and
observable computation helpers used across DiffSim, IBI, and
Relative Entropy workflows.
"""

import os

import numpy as np
import jax.numpy as jnp
import pandas as pd
from jax.tree_util import tree_map
from scipy import interpolate as sci_interpolate

from diffcg import energy
from diffcg.system import AtomicSystem
from diffcg.io.lammps import read_lammps_data
from diffcg.io.ase_trj import read_ase_trj
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
from diffcg.observable.analyze import TrajectoryAnalyzer
from diffcg._core.constants import BOLTZMANN_KJMOLK, PRESSURE_CONVERSION_KJMOL_NM3_TO_BAR

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ---------------------------------------------------------------------------
# Physics constants & grids
# ---------------------------------------------------------------------------
R_CUT = 2.0
R_ONSET = 1.5
BOLTZMANN_CONSTANT = BOLTZMANN_KJMOLK
PRESSURE_TARGET = 1.0 / PRESSURE_CONVERSION_KJMOL_NM3_TO_BAR  # 1 bar in kJ/(mol·nm³)

NUM_BEADS = 500
BEAD_MASS = 104.0
ATOM_TYPES_MAPPING = {0: "C", 1: "C", 2: "H"}

SPLINE_GRID_PAIR = jnp.linspace(0.4, R_CUT, 80)
SPLINE_GRID_BOND = jnp.linspace(0.1, 1.0, 45)
SPLINE_GRID_ANGLE = jnp.linspace(0.1, 3.14, 55)
SPLINE_GRID_DIHEDRAL = jnp.linspace(-3.14, 3.14, 100)


# ---------------------------------------------------------------------------
# Low-level loaders
# ---------------------------------------------------------------------------
def load_curve_csv(path: str, x_grid: np.ndarray, zero_eps: float = 1e-7):
    """Interpolate a two-column distribution file onto *x_grid*."""
    data = pd.read_csv(path, sep=r"\s+", header=None)
    xy = data[[0, 1]].values
    spline = sci_interpolate.interp1d(
        xy[:, 0], xy[:, 1], kind="cubic", bounds_error=False, fill_value="extrapolate"
    )
    y = spline(x_grid)
    y[y < zero_eps] = 0
    return y


def load_topology_csv(path: str):
    """Load topology CSV and return a zero-indexed array."""
    return pd.read_csv(path, header=None, sep=r"\s+").values - 1


# ---------------------------------------------------------------------------
# High-level loaders
# ---------------------------------------------------------------------------
def load_targets(temp, num_beads=NUM_BEADS):
    """Load target distributions and topology for a given temperature.

    Returns
    -------
    target_dict : dict
        Keys: 'rdf', 'bdf', 'adf', 'ddf', 'pressure'.
    topology : dict
        Keys: 'bond', 'angle', 'dihedral' (zero-indexed arrays).
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

    # DDF
    ddf_bin_centers, ddf_bin_boundaries, sigma_DDF = ddf_discretization(
        3.14, nbins=200, DDF_start=-3.14
    )
    ddf_path = os.path.join(DATA_DIR, f"T{temp}", "dihedralAAAA.dist.tgt")
    reference_ddf = load_curve_csv(ddf_path, ddf_bin_centers)
    dihedral_top = load_topology_csv(os.path.join(DATA_DIR, "polymer", "dihedral.csv"))
    ddf_struct = DDFParams(reference_ddf, ddf_bin_centers, ddf_bin_boundaries, sigma_DDF, dihedral_top)

    # RDF (with exclusion mask from dihedral topology)
    rdf_bin_centers, rdf_bin_boundaries, sigma_RDF = rdf_discretization(RDF_cut=R_CUT)
    mask = np.ones((num_beads, num_beads))
    for i, j in [(0, 1), (1, 0), (2, 0), (0, 2), (3, 0), (0, 3),
                 (2, 1), (1, 2), (3, 1), (1, 3), (3, 2), (2, 3)]:
        mask[dihedral_top[:, i], dihedral_top[:, j]] = 0
    polymer_exclude = jnp.array(mask)

    rdf_path = os.path.join(DATA_DIR, f"T{temp}", "nb_smoothed.dist.tgt")
    reference_rdf = load_curve_csv(rdf_path, rdf_bin_centers, zero_eps=0.0)
    rdf_struct = InterRDFParams(reference_rdf, rdf_bin_centers, rdf_bin_boundaries, sigma_RDF, polymer_exclude)

    target_dict = {
        "rdf": rdf_struct,
        "bdf": bdf_struct,
        "adf": adf_struct,
        "ddf": ddf_struct,
        "pressure": PRESSURE_TARGET,
    }
    topology = {"bond": bond_top, "angle": angle_top, "dihedral": dihedral_top}
    return target_dict, topology


def load_system(temp, num_beads=NUM_BEADS):
    """Load a LAMMPS data file and return an AtomicSystem plus atom-type array.

    Coordinates are converted from Angstroms to nanometers.

    Returns
    -------
    system : AtomicSystem
    atom_types : jnp.ndarray  (int32)
    """
    lammps_path = os.path.join(DATA_DIR, f"T{temp}", "PS.data")
    sys_data = read_lammps_data(lammps_path)
    atom_type_ids = sys_data["atom_types"]
    coords = sys_data["coords"][0] / 10.0  # Å → nm
    cell = sys_data["cells"][0] / 10.0

    system = AtomicSystem(
        R=jnp.array(coords, dtype=jnp.float32),
        Z=jnp.array(atom_type_ids, dtype=jnp.int32),
        cell=jnp.array(cell, dtype=jnp.float32),
        masses=jnp.array([BEAD_MASS] * num_beads, dtype=jnp.float32),
        pbc=True,
    )
    atom_types = jnp.array(atom_type_ids, dtype=jnp.int32)
    return system, atom_types


def load_pretrained_params():
    """Load the pretrained spline parameters from the example directory."""
    path = os.path.join(os.path.dirname(__file__), "pretrained_params.npy")
    return np.load(path, allow_pickle=True).item()


# ---------------------------------------------------------------------------
# Quantity dict & observable helpers
# ---------------------------------------------------------------------------
_DEFAULT_GAMMAS = {"rdf": 1.0, "bdf": 1e-3, "adf": 0.1, "ddf": 1.0}


def build_quantity_dict(target_dict, gammas=None):
    """Build quantity_dict from target distributions.

    Parameters
    ----------
    target_dict : dict  from ``load_targets``
    gammas : dict, optional
        Per-observable loss weights. Defaults to ``_DEFAULT_GAMMAS``.
    """
    gammas = gammas or _DEFAULT_GAMMAS
    quantity_dict = {}

    if "rdf" in target_dict:
        s = target_dict["rdf"]
        fn = initialize_inter_radial_distribution_fun(s)
        quantity_dict["rdf"] = {
            "compute_fn": fn, "target": s.reference_rdf,
            "gamma": gammas.get("rdf", 1.0), "bin_centers": s.rdf_bin_centers,
        }
    if "bdf" in target_dict:
        s = target_dict["bdf"]
        fn = initialize_bond_distribution_fun(s)
        quantity_dict["bdf"] = {
            "compute_fn": fn, "target": s.reference_bdf,
            "gamma": gammas.get("bdf", 1e-3), "bin_centers": s.bdf_bin_centers,
        }
    if "adf" in target_dict:
        s = target_dict["adf"]
        fn = initialize_angle_distribution_fun(s)
        quantity_dict["adf"] = {
            "compute_fn": fn, "target": s.reference_adf,
            "gamma": gammas.get("adf", 0.1), "bin_centers": s.adf_bin_centers,
        }
    if "ddf" in target_dict:
        s = target_dict["ddf"]
        fn = initialize_dihedral_distribution_fun(s)
        quantity_dict["ddf"] = {
            "compute_fn": fn, "target": s.reference_ddf,
            "gamma": gammas.get("ddf", 1.0), "bin_centers": s.ddf_bin_centers,
        }
    return quantity_dict


def build_energy_fn(params, topology, max_num_atoms=1,
                    atom_types=None, pair_type_map=None):
    """Construct a total energy function from spline parameters.

    When *atom_types* and *pair_type_map* are provided the pair (and
    repulsion prior) terms use per-type interactions.
    """
    bond_top = topology["bond"]
    angle_top = topology["angle"]
    dihedral_top = topology["dihedral"]

    pair_kw = dict(mask_topology=angle_top, max_num_atoms=max_num_atoms)
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
    dihedral_energy_fn = energy.TabulatedDihedralEnergy(
        SPLINE_GRID_DIHEDRAL, params["dihedral"], dihedral_top
    ).get_energy_fn()

    prior_fn = energy.GenericRepulsionEnergy(
        sigma=0.6, epsilon=1.0, exp=8, **pair_kw
    ).get_energy_fn()
    bond_pot_fn = energy.HarmonicBondEnergy(bonds=bond_top, length=0.45, epsilon=5000).get_energy_fn()
    angle_pot_fn = energy.HarmonicAngleEnergy(angles=angle_top, angle_0=1.5, epsilon=50).get_energy_fn()
    dihedral_pot_fn = energy.HarmonicDihedralEnergy(dihedrals=dihedral_top, angle_0=1.5, epsilon=50).get_energy_fn()

    fns = [pair_energy_fn, bond_energy_fn, angle_energy_fn, dihedral_energy_fn,
           bond_pot_fn, angle_pot_fn, dihedral_pot_fn, prior_fn]

    def total_energy_fn(system, neighbors, **dynamic_kwargs):
        return sum(f(system, neighbors, **dynamic_kwargs) for f in fns)

    return total_energy_fn


def make_calculate_observables_fn(quantity_dict, ref_system):
    """Return a callable that computes observables from a trajectory or batched systems."""

    def calculate_observables(traj_file="sample.traj", batched_systems=None):
        if batched_systems is None:
            systems = read_ase_trj(traj_file)
            batched_systems = tree_map(lambda *xs: jnp.stack(xs), *systems)

        observables = {}
        for key in ("rdf", "bdf", "adf", "ddf"):
            if key in quantity_dict:
                analyzer = TrajectoryAnalyzer(quantity_dict[key]["compute_fn"], ref_system)
                observables[key] = analyzer.analyze(batched_systems)
        return observables

    return calculate_observables
