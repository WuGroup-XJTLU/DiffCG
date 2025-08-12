from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from scipy import interpolate as sci_interpolate

from diffcg.util import custom_quantity
from diffcg.common.constants import PRESSURE_CONVERSION_KJMOL_NM3_TO_BAR


@dataclass
class TargetDict:
    rdf: custom_quantity.InterRDFParams
    bdf: custom_quantity.BDFParams
    adf: custom_quantity.ADFParams
    ddf: custom_quantity.DDFParams
    pressure: float


def _interp_spline(xy: np.ndarray, xq: np.ndarray) -> np.ndarray:
    spline = sci_interpolate.interp1d(xy[:, 0], xy[:, 1], kind='cubic', fill_value="extrapolate")
    return spline(xq)


def load_polymer_targets(base_dir: str, temperature: int) -> TargetDict:
    # pressure target: 1 bar in kJ / (mol * nm^3)
    pressure_target = 1.0 / PRESSURE_CONVERSION_KJMOL_NM3_TO_BAR

    # BDF
    bdf_bin_centers, bdf_bin_boundaries, sigma_BDF = custom_quantity.bdf_discretization(1.0, nbins=200, BDF_start=0.)
    bdf_df = pd.read_csv(f"{base_dir}/T{temperature}/bondAA_smooth.dist.tgt", sep="\s+", header=None)
    ref_bdf = _interp_spline(bdf_df[[0, 1]].values, bdf_bin_centers)
    ref_bdf[ref_bdf < 1e-7] = 0
    bond_top = pd.read_csv(f"{base_dir}/polymer/bond.csv", header=None, sep='\s+').values - 1
    bdf_struct = custom_quantity.BDFParams(ref_bdf, bdf_bin_centers, bdf_bin_boundaries, sigma_BDF, bond_top)

    # ADF
    adf_bin_centers, adf_bin_boundaries, sigma_ADF = custom_quantity.adf_discretization(np.pi, nbins=200, ADF_start=0.00)
    adf_df = pd.read_csv(f"{base_dir}/T{temperature}/angleAAA.dist.tgt", sep="\s+", header=None)
    ref_adf = _interp_spline(adf_df[[0, 1]].values, adf_bin_centers)
    ref_adf[ref_adf < 1e-7] = 0
    angle_top = pd.read_csv(f"{base_dir}/polymer/angle.csv", header=None, sep='\s+').values - 1
    adf_struct = custom_quantity.ADFParams(ref_adf, adf_bin_centers, adf_bin_boundaries, sigma_ADF, angle_top)

    # DDF
    ddf_bin_centers, ddf_bin_boundaries, sigma_DDF = custom_quantity.ddf_discretization(3.14, nbins=200, DDF_start=-3.14)
    ddf_df = pd.read_csv(f"{base_dir}/T{temperature}/dihedralAAAA.dist.tgt", sep="\s+", header=None)
    ref_ddf = _interp_spline(ddf_df[[0, 1]].values, ddf_bin_centers)
    ref_ddf[ref_ddf < 1e-7] = 0
    dihedral_top = pd.read_csv(f"{base_dir}/polymer/dihedral.csv", header=None, sep='\s+').values - 1
    ddf_struct = custom_quantity.DDFParams(ref_ddf, ddf_bin_centers, ddf_bin_boundaries, sigma_DDF, dihedral_top)

    # RDF with exclusions from topology
    rdf_bin_centers, rdf_bin_boundaries, sigma_RDF = custom_quantity.rdf_discretization(RDF_cut=2.0)
    max_idx = int(max(dihedral_top.max(), angle_top.max(), bond_top.max())) + 1
    mask = np.ones((max_idx, max_idx))
    # Exclude any bonded pairs covered by dihedral quadruplets
    for row in dihedral_top:
        for a in range(4):
            for b in range(a + 1, 4):
                i, j = int(row[a]), int(row[b])
                if 0 <= i < max_idx and 0 <= j < max_idx:
                    mask[i, j] = 0
                    mask[j, i] = 0
    rdf_df = pd.read_csv(f"{base_dir}/T{temperature}/nb_smoothed.dist.tgt", header=None, sep='\s+')
    ref_rdf = _interp_spline(rdf_df[[0, 1]].values, rdf_bin_centers)
    rdf_struct = custom_quantity.InterRDFParams(ref_rdf, rdf_bin_centers, rdf_bin_boundaries, sigma_RDF, mask)

    return TargetDict(rdf=rdf_struct, bdf=bdf_struct, adf=adf_struct, ddf=ddf_struct, pressure=pressure_target)


