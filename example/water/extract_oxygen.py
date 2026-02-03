#!/usr/bin/env python3
"""
Extract oxygen atoms from SPCE_water.data and create a new LAMMPS data file.

This script filters the original water data file to extract only oxygen atoms
(type 2 in the original file), renumbers them sequentially, and writes a new
data file suitable for simulations with only oxygen atoms.
"""

import os

def extract_oxygen_atoms(input_file, output_file):
    """
    Extract oxygen atoms from LAMMPS data file and create new data file.

    Args:
        input_file: Path to input SPCE_water.data file
        output_file: Path to output SPCE_water_oxygen.data file
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Parse header and find key sections
    num_atoms = 0
    xlo = xhi = ylo = yhi = zlo = zhi = 0.0
    masses_start = None
    atoms_start = None
    atoms_end = None

    for i, line in enumerate(lines):
        line = line.strip()
        if ' atoms' in line and not line.startswith('#'):
            num_atoms = int(line.split()[0])
        elif 'xlo xhi' in line:
            parts = line.split()
            xlo, xhi = float(parts[0]), float(parts[1])
        elif 'ylo yhi' in line:
            parts = line.split()
            ylo, yhi = float(parts[0]), float(parts[1])
        elif 'zlo zhi' in line:
            parts = line.split()
            zlo, zhi = float(parts[0]), float(parts[1])
        elif line == 'Masses':
            masses_start = i + 2  # Skip the "Masses" line and blank line
        elif line == 'Atoms  # full':
            atoms_start = i + 2  # Skip the "Atoms # full" line and blank line

    # Find where atoms section ends (look for next section or end of file)
    if atoms_start is not None:
        for i in range(atoms_start, len(lines)):
            line = lines[i].strip()
            if line and not line.split()[0].isdigit():  # Not an atom line
                atoms_end = i
                break
        if atoms_end is None:
            atoms_end = len(lines)

    # Read atoms and filter for oxygen (type 2)
    oxygen_atoms = []
    for i in range(atoms_start, atoms_end):
        parts = lines[i].strip().split()
        if len(parts) >= 6:
            atom_id = int(parts[0])
            molecule_id = int(parts[1])
            atom_type = int(parts[2])
            charge = float(parts[3])
            x = float(parts[4])
            y = float(parts[5])
            z = float(parts[6]) if len(parts) > 6 else 0.0

            # Filter for oxygen atoms (type 2)
            if atom_type == 2:
                oxygen_atoms.append({
                    'atom_id': atom_id,
                    'molecule_id': molecule_id,
                    'atom_type': atom_type,
                    'charge': charge,
                    'x': x,
                    'y': y,
                    'z': z
                })

    # Count oxygen atoms
    num_oxygen = len(oxygen_atoms)

    # Write output file
    with open(output_file, 'w') as f:
        # Header
        f.write("# LAMMPS data file - Oxygen atoms only from SPCE water\n")
        f.write(f"{num_oxygen} atoms\n")
        f.write("0 bonds\n")
        f.write("0 angles\n")
        f.write("1 atom types\n")
        f.write(f"{xlo} {xhi} xlo xhi\n")
        f.write(f"{ylo} {yhi} ylo yhi\n")
        f.write(f"{zlo} {zhi} zlo zhi\n")
        f.write("\n")

        # Masses section
        f.write("Masses\n")
        f.write("\n")
        f.write("1 15.9994\n")
        f.write("\n")

        # Atoms section
        f.write("Atoms  # full\n")
        f.write("\n")

        # Write oxygen atoms with renumbered IDs
        for new_id, atom in enumerate(oxygen_atoms, start=1):
            # Change type from 2 to 1 (now only one atom type)
            f.write(f"{new_id} {atom['molecule_id']} 1 {atom['charge']} "
                   f"{atom['x']:.10f} {atom['y']:.10f} {atom['z']:.10f}\n")

    print(f"Successfully extracted {num_oxygen} oxygen atoms")
    print(f"Output written to: {output_file}")
    print(f"\nOutput file statistics:")
    print(f"  - Number of atoms: {num_oxygen}")
    print(f"  - Atom types: 1 (oxygen only)")
    print(f"  - Mass: 15.9994")
    print(f"  - Box dimensions: ({xlo}, {xhi}) x ({ylo}, {yhi}) x ({zlo}, {zhi})")

if __name__ == "__main__":
    # Define file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "SPCE_water.data")
    output_file = os.path.join(script_dir, "SPCE_water_oxygen.data")

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        exit(1)

    # Extract oxygen atoms
    extract_oxygen_atoms(input_file, output_file)
