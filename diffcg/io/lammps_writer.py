# SPDX-License-Identifier: MIT
# Copyright (c) 2025 WuResearchGroup

"""Functions to write LAMMPS input files from DiffCG objects.

Handles unit conversions from DiffCG internal units (nm, kJ/mol) to
LAMMPS real units (Angstroms, kcal/mol).
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Union

from diffcg._core.units import (
    NM_TO_ANGSTROM,
    KJMOL_TO_KCALMOL,
    RAD_TO_DEG,
    positions_to_lammps,
    cell_to_lammps,
    energy_to_lammps,
)
from diffcg._core.logger import get_logger

logger = get_logger(__name__)


def write_lammps_data(
    system,
    topology: Dict,
    filepath: str,
    mol_ids: Optional[np.ndarray] = None,
) -> None:
    """Write a LAMMPS data file (atom_style full) from a DiffCG AtomicSystem.

    Args:
        system: AtomicSystem with positions in nm
        topology: Dict with keys: bonds, angles, dihedrals, bond_types,
            angle_types, dihedral_types (all 0-indexed numpy arrays)
        filepath: Output file path
        mol_ids: Optional (N,) array of molecule IDs (0-indexed). If None,
            all atoms are assigned to molecule 1.
    """
    R_ang = positions_to_lammps(np.asarray(system.R))
    n_atoms = R_ang.shape[0]

    # Atom types (0-indexed in DiffCG)
    atom_types = np.asarray(system.Z)
    n_atom_types = int(atom_types.max()) + 1

    # Masses
    masses_arr = np.asarray(system.masses) if system.masses is not None else np.ones(n_atoms)
    # Build per-type mass dict
    type_masses = {}
    for i in range(n_atoms):
        t = int(atom_types[i])
        if t not in type_masses:
            type_masses[t] = float(masses_arr[i])

    # Molecule IDs (1-indexed for LAMMPS)
    if mol_ids is None:
        mol_ids_1 = np.ones(n_atoms, dtype=int)
    else:
        mol_ids_1 = np.asarray(mol_ids, dtype=int) + 1

    # Cell
    if system.cell is not None:
        cell_ang = cell_to_lammps(np.asarray(system.cell))
        # Assume orthorhombic (diagonal cell matrix, column-major)
        lx = float(cell_ang[0, 0])
        ly = float(cell_ang[1, 1])
        lz = float(cell_ang[2, 2])
    else:
        # Non-periodic: use bounding box with padding
        padding = 10.0  # Angstroms
        lx = float(R_ang[:, 0].max() - R_ang[:, 0].min()) + 2 * padding
        ly = float(R_ang[:, 1].max() - R_ang[:, 1].min()) + 2 * padding
        lz = float(R_ang[:, 2].max() - R_ang[:, 2].min()) + 2 * padding

    # Topology arrays
    bonds = np.asarray(topology.get("bonds", np.empty((0, 2), dtype=int)))
    bond_types = np.asarray(topology.get("bond_types", np.empty(0, dtype=int)))
    angles = np.asarray(topology.get("angles", np.empty((0, 3), dtype=int)))
    angle_types = np.asarray(topology.get("angle_types", np.empty(0, dtype=int)))
    dihedrals = np.asarray(topology.get("dihedrals", np.empty((0, 4), dtype=int)))
    dihedral_types = np.asarray(topology.get("dihedral_types", np.empty(0, dtype=int)))

    n_bonds = len(bonds)
    n_angles = len(angles)
    n_dihedrals = len(dihedrals)
    n_bond_types = int(bond_types.max()) + 1 if n_bonds > 0 else 0
    n_angle_types = int(angle_types.max()) + 1 if n_angles > 0 else 0
    n_dihedral_types = int(dihedral_types.max()) + 1 if n_dihedrals > 0 else 0

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        f.write("LAMMPS data file via DiffCG\n\n")

        f.write(f"{n_atoms} atoms\n")
        f.write(f"{n_atom_types} atom types\n")
        if n_bonds > 0:
            f.write(f"{n_bonds} bonds\n")
            f.write(f"{n_bond_types} bond types\n")
        if n_angles > 0:
            f.write(f"{n_angles} angles\n")
            f.write(f"{n_angle_types} angle types\n")
        if n_dihedrals > 0:
            f.write(f"{n_dihedrals} dihedrals\n")
            f.write(f"{n_dihedral_types} dihedral types\n")

        f.write(f"\n0.0 {lx:.6f} xlo xhi\n")
        f.write(f"0.0 {ly:.6f} ylo yhi\n")
        f.write(f"0.0 {lz:.6f} zlo zhi\n")

        # Masses section
        f.write("\nMasses\n\n")
        for t in sorted(type_masses.keys()):
            f.write(f"{t + 1} {type_masses[t]:.4f}\n")

        # Atoms section (atom_style full: atom_id mol_id atom_type charge x y z)
        f.write("\nAtoms # full\n\n")
        for i in range(n_atoms):
            atom_id = i + 1
            mol_id = int(mol_ids_1[i])
            atype = int(atom_types[i]) + 1
            charge = 0.0
            x, y, z = R_ang[i]
            f.write(f"{atom_id} {mol_id} {atype} {charge:.4f} {x:.6f} {y:.6f} {z:.6f}\n")

        # Bonds section
        if n_bonds > 0:
            f.write("\nBonds\n\n")
            for j in range(n_bonds):
                bond_id = j + 1
                btype = int(bond_types[j]) + 1
                a1 = int(bonds[j, 0]) + 1
                a2 = int(bonds[j, 1]) + 1
                f.write(f"{bond_id} {btype} {a1} {a2}\n")

        # Angles section
        if n_angles > 0:
            f.write("\nAngles\n\n")
            for j in range(n_angles):
                angle_id = j + 1
                atype = int(angle_types[j]) + 1
                a1 = int(angles[j, 0]) + 1
                a2 = int(angles[j, 1]) + 1
                a3 = int(angles[j, 2]) + 1
                f.write(f"{angle_id} {atype} {a1} {a2} {a3}\n")

        # Dihedrals section
        if n_dihedrals > 0:
            f.write("\nDihedrals\n\n")
            for j in range(n_dihedrals):
                dih_id = j + 1
                dtype = int(dihedral_types[j]) + 1
                a1 = int(dihedrals[j, 0]) + 1
                a2 = int(dihedrals[j, 1]) + 1
                a3 = int(dihedrals[j, 2]) + 1
                a4 = int(dihedrals[j, 3]) + 1
                f.write(f"{dih_id} {dtype} {a1} {a2} {a3} {a4}\n")

    logger.debug("Wrote LAMMPS data file to %s (%d atoms)", filepath, n_atoms)


def write_lammps_table(
    filepath: str,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    keyword: str,
    potential_type: str = "pair",
) -> None:
    """Write a LAMMPS TABLE format file for pair/bond/angle/dihedral potentials.

    Supports multi-type tables by calling this function multiple times with
    mode='a' (append). Use write_lammps_tables() for convenience.

    Args:
        filepath: Output file path
        x_vals: Independent variable (nm for pair/bond, radians for angle/dihedral)
        y_vals: Energy values in kJ/mol
        keyword: Section keyword (e.g., "PAIR_1", "BOND_1")
        potential_type: One of "pair", "bond", "angle", "dihedral"
    """
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    N = len(x)

    if potential_type in ("pair", "bond"):
        # Convert nm -> Angstroms
        x_lammps = x * NM_TO_ANGSTROM
        y_lammps = y * KJMOL_TO_KCALMOL
        # Force = -dE/dr, convert consistently
        dydx = np.gradient(y, x)  # dE/dr in kJ/(mol*nm)
        force = -dydx * KJMOL_TO_KCALMOL / NM_TO_ANGSTROM  # kcal/(mol*Ang)
    elif potential_type in ("angle", "dihedral"):
        # LAMMPS angle/dihedral tables use degrees for x
        x_lammps = x * RAD_TO_DEG
        y_lammps = y * KJMOL_TO_KCALMOL
        # Force = -dE/d(angle), in kcal/(mol*degree)
        dydx = np.gradient(y, x)  # dE/d(rad) in kJ/(mol*rad)
        force = -dydx * KJMOL_TO_KCALMOL * (np.pi / 180.0)  # kcal/(mol*deg)
    else:
        raise ValueError(f"Unknown potential_type: {potential_type}")

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "a") as f:
        f.write(f"# LAMMPS table generated by DiffCG ({potential_type})\n\n")
        f.write(f"{keyword}\n")
        f.write(f"N {N}\n\n")

        for i in range(N):
            f.write(f"{i + 1} {x_lammps[i]:.10e} {y_lammps[i]:.10e} {force[i]:.10e}\n")
        f.write("\n")

    logger.debug("Wrote LAMMPS table '%s' to %s (%d points)", keyword, filepath, N)


def write_lammps_tables(
    filepath: str,
    tables: List[Dict],
    potential_type: str = "pair",
) -> None:
    """Write multiple table sections to a single LAMMPS table file.

    Args:
        filepath: Output file path (will be overwritten)
        tables: List of dicts with keys: x, y, keyword
        potential_type: One of "pair", "bond", "angle", "dihedral"
    """
    # Clear file first
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    open(filepath, "w").close()

    for table in tables:
        write_lammps_table(
            filepath,
            table["x"],
            table["y"],
            table["keyword"],
            potential_type=potential_type,
        )
