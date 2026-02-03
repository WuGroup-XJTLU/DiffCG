"""Shared utilities for coarse-grained water model examples.

Provides data loading, target construction, energy builder, and
observable computation helpers for single-site water models.
"""

import os
from typing import Dict, Optional

import numpy as np
import jax.numpy as jnp
import pandas as pd
from scipy import interpolate as sci_interpolate

from diffcg import energy
from diffcg.system import AtomicSystem
from diffcg.io.lammps import read_lammps_data
from diffcg.observable.structure import (
    initialize_inter_radial_distribution_fun,
    InterRDFParams,
    rdf_discretization,
)
from diffcg._core.constants import BOLTZMANN_KJMOLK, PRESSURE_CONVERSION_KJMOL_NM3_TO_BAR

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Physics constants & grids
# ---------------------------------------------------------------------------
R_CUT = 1.0  # nm
R_ONSET = 0.8  # nm
BOLTZMANN_CONSTANT = BOLTZMANN_KJMOLK
PRESSURE_TARGET = 1.0 / PRESSURE_CONVERSION_KJMOL_NM3_TO_BAR  # 1 bar in kJ/(mol·nm³)

NUM_WATER = 300
WATER_MASS = 15.9994  # g/mol (oxygen mass)

SPLINE_GRID_PAIR = jnp.linspace(0.2, R_CUT, 80)


# ---------------------------------------------------------------------------
# Data loading functions
# ---------------------------------------------------------------------------

def load_curve_csv(path: str, x_grid: np.ndarray, zero_eps: float = 1e-7):
    """Interpolate a two-column distribution file onto *x_grid*."""
    data = pd.read_csv(path, sep=r"\s+", header=None, comment="#")
    xy = data[[0, 1]].values
    spline = sci_interpolate.interp1d(
        xy[:, 0], xy[:, 1], kind="cubic", bounds_error=False, fill_value="extrapolate"
    )
    y = spline(x_grid)
    y[y < zero_eps] = 0
    return y


def load_water_rdf(path: str, x_grid: jnp.ndarray) -> jnp.ndarray:
    """Load target O-O RDF from file.

    Args:
        path: Path to RDF file (two-column format: distance, g(r))
        x_grid: Grid to interpolate the RDF onto

    Returns:
        Interpolated RDF values on x_grid
    """
    data = np.loadtxt(path)
    distances = data[:, 0]
    rdf_values = data[:, 1]

    # Interpolate onto the provided grid
    spline = sci_interpolate.interp1d(
        distances, rdf_values, kind="cubic", bounds_error=False, fill_value="extrapolate"
    )
    rdf = spline(x_grid)
    rdf[rdf < 0] = 0  # Ensure non-negative
    return jnp.array(rdf, dtype=jnp.float64)


def load_water_system(path: str, num_water: int = NUM_WATER) -> AtomicSystem:
    """Load oxygen-only LAMMPS data file and return an AtomicSystem.

    Coordinates are converted from Angstroms to nanometers.

    Args:
        path: Path to LAMMPS data file
        num_water: Number of water molecules

    Returns:
        AtomicSystem with oxygen positions
    """
    sys_data = read_lammps_data(path)
    atom_type_ids = sys_data["atom_types"]
    coords = sys_data["coords"][0] / 10.0  # Å -> nm
    cell = sys_data["cells"][0] / 10.0

    system = AtomicSystem(
        R=jnp.array(coords, dtype=jnp.float32),
        Z=jnp.array(atom_type_ids, dtype=jnp.int32),
        cell=jnp.array(cell, dtype=jnp.float32),
        masses=jnp.array([WATER_MASS] * num_water, dtype=jnp.float32),
        pbc=True,
    )
    return system


# ---------------------------------------------------------------------------
# Target loading
# ---------------------------------------------------------------------------

def load_targets(rdf_path: str, r_cut: float = R_CUT, num_water: int = NUM_WATER):
    """Load target RDF for water model.

    Args:
        rdf_path: Path to target RDF file
        r_cut: RDF cutoff distance
        num_water: Number of water molecules

    Returns:
        target_dict: Dictionary with 'rdf' key containing InterRDFParams
    """
    # Setup RDF discretization
    rdf_bin_centers, rdf_bin_boundaries, sigma_RDF = rdf_discretization(
        RDF_cut=r_cut, nbins=300, RDF_start=0.0
    )

    # Load target RDF
    reference_rdf = load_water_rdf(rdf_path, rdf_bin_centers)

    # No exclusions needed for single-site water
    exclude_pairs = jnp.empty((0, 2), dtype=jnp.int32)

    # Create InterRDFParams
    rdf_struct = InterRDFParams(
        reference_rdf=reference_rdf,
        rdf_bin_centers=rdf_bin_centers,
        rdf_bin_boundaries=rdf_bin_boundaries,
        sigma_RDF=sigma_RDF,
        exclude_pairs=exclude_pairs,
    )

    target_dict = {"rdf": rdf_struct, "pressure": PRESSURE_TARGET}
    return target_dict


# ---------------------------------------------------------------------------
# Quantity dict & observable helpers
# ---------------------------------------------------------------------------

_DEFAULT_GAMMAS = {"rdf": 1.0}


def build_quantity_dict(target_dict: Dict, gammas: Optional[Dict] = None, kT: float = None) -> Dict:
    """Build quantity_dict from target distributions.

    For single-site water, only RDF is needed (plus optionally pressure).

    Args:
        target_dict: Dictionary from load_targets with 'rdf' key
        gammas: Optional per-observable loss weights. Defaults to RDF=1.0.
        kT: Thermal energy in kJ/mol. Required when 'pressure' is in target_dict.

    Returns:
        quantity_dict with compute_fn, target, gamma, bin_centers for each observable
    """
    gammas = gammas or _DEFAULT_GAMMAS
    quantity_dict = {}

    if "rdf" in target_dict:
        s = target_dict["rdf"]
        fn = initialize_inter_radial_distribution_fun(s)
        quantity_dict["rdf"] = {
            "compute_fn": fn,
            "target": s.reference_rdf,
            "gamma": gammas.get("rdf", 1.0),
            "bin_centers": s.rdf_bin_centers,
        }

    if "pressure" in target_dict and "pressure" in gammas:
        from diffcg.observable.pressure import make_pressure_compute_fn
        if kT is None:
            raise ValueError("kT is required when 'pressure' is in target_dict")
        quantity_dict["pressure"] = {
            "compute_fn": make_pressure_compute_fn(kT),
            "target": jnp.atleast_1d(jnp.array(target_dict["pressure"])),
            "gamma": gammas.get("pressure", 10.0),
        }

    return quantity_dict


# ---------------------------------------------------------------------------
# Energy function builder
# ---------------------------------------------------------------------------

def build_energy_fn(params: Dict, **kwargs) -> callable:
    """Construct total energy function for single-site water model.

    Uses only tabulated pair potential plus a generic repulsion prior.
    No bonds, angles, or dihedrals for single-site water.

    Args:
        params: Dictionary with 'pair' key containing spline parameters

    Returns:
        Total energy function: energy_fn(system, neighbors, **kwargs) -> energy
    """
    # Tabulated pair energy
    pair_energy_fn = energy.TabulatedPairEnergy(
        SPLINE_GRID_PAIR,
        params["pair"],
        R_ONSET,
        R_CUT,
    ).get_energy_fn()

    # Generic repulsion prior (soft repulsion to prevent overlap)
    prior_fn = energy.GenericRepulsionEnergy(
        sigma=0.3, epsilon=1.0, exp=8,
    ).get_energy_fn()

    def total_energy_fn(system, neighbors, **kwargs):
        return pair_energy_fn(system, neighbors, **kwargs) + prior_fn(system, neighbors, **kwargs)

    return total_energy_fn


def get_initial_params() -> Dict:
    """Get initial parameters for water model optimization.

    Uses generic repulsion as initial guess for the pair potential.

    Returns:
        Dictionary with 'pair' key containing initial spline values
    """
    # Evaluate generic repulsion on spline grid as initial guess
    r_grid = np.asarray(SPLINE_GRID_PAIR)

    def generic_repulsion(r, sigma=0.3, epsilon=1.0, exp=8):
        """Soft repulsion: U = epsilon * (sigma / r)**exp"""
        r = np.where(r > 1e-8, r, 1e8)
        return epsilon * (sigma / r) ** exp

    initial_pair = generic_repulsion(r_grid)

    return {"pair": jnp.array(initial_pair, dtype=jnp.float64)}
