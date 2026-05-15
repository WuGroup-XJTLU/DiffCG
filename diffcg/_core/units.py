# SPDX-License-Identifier: MIT
# Copyright (c) 2025 WuResearchGroup

"""Unit conversion constants and helpers for DiffCG <-> LAMMPS real units.

DiffCG internal units:
    Length: nm
    Energy: kJ/mol
    Time: fs
    Mass: g/mol
    Angle: radians

LAMMPS real units:
    Length: Angstroms
    Energy: kcal/mol
    Time: fs
    Mass: g/mol
    Angle: degrees (for some commands, but tables use radians internally)
"""

import numpy as np

# --- Conversion constants ---

NM_TO_ANGSTROM = 10.0
ANGSTROM_TO_NM = 1.0 / NM_TO_ANGSTROM

KJMOL_TO_KCALMOL = 1.0 / 4.184
KCALMOL_TO_KJMOL = 4.184

EV_TO_KJMOL = 96.48533212
KJMOL_TO_EV = 1.0 / EV_TO_KJMOL

RAD_TO_DEG = 180.0 / np.pi
DEG_TO_RAD = np.pi / 180.0


# --- Helper functions ---

def positions_to_lammps(R_nm):
    """Convert positions from nm to Angstroms."""
    return np.asarray(R_nm) * NM_TO_ANGSTROM


def positions_from_lammps(R_ang):
    """Convert positions from Angstroms to nm."""
    return np.asarray(R_ang) * ANGSTROM_TO_NM


def cell_to_lammps(cell_nm):
    """Convert cell vectors from nm to Angstroms."""
    return np.asarray(cell_nm) * NM_TO_ANGSTROM


def cell_from_lammps(cell_ang):
    """Convert cell vectors from Angstroms to nm."""
    return np.asarray(cell_ang) * ANGSTROM_TO_NM


def energy_to_lammps(E_kjmol):
    """Convert energy from kJ/mol to kcal/mol."""
    return np.asarray(E_kjmol) * KJMOL_TO_KCALMOL


def energy_from_lammps(E_kcalmol):
    """Convert energy from kcal/mol to kJ/mol."""
    return np.asarray(E_kcalmol) * KCALMOL_TO_KJMOL
