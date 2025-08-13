# SPDX-License-Identifier: MIT
# Copyright (c) 2025 WuResearchGroup

# this file implement the function to read the lammps data file and lammps trajectory file following the dpdata format (https://github.com/deepmodeling/dpdata/tree/master)

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import re
from pathlib import Path
from diffcg.util.logger import get_logger

logger = get_logger(__name__)


class LAMMPSDataReader:
    """Reader for LAMMPS data files following dpdata format."""
    
    def __init__(self, filename: str):
        """
        Initialize the LAMMPS data file reader.
        
        Args:
            filename: Path to the LAMMPS data file
        """
        self.filename = filename
        self.data = {}
        
    def read(self) -> Dict:
        """
        Read the LAMMPS data file and return data in dpdata format.
        
        Returns:
            Dictionary containing the system data in dpdata format:
            - atom_types: numpy array of atom types
            - coords: numpy array of coordinates (shape: (1, n_atoms, 3))
            - cells: numpy array of cell vectors (shape: (1, 3, 3))
            - atom_names: list of atom type names
            - atom_numbs: list of atom counts per type
            - masses: dictionary mapping atom types to masses
            - bonds: numpy array of bond connections (shape: (n_bonds, 2))
            - bond_types: numpy array of bond types
            - angles: numpy array of angle connections (shape: (n_angles, 3))
            - angle_types: numpy array of angle types
            - dihedrals: numpy array of dihedral connections (shape: (n_dihedrals, 4))
            - dihedral_types: numpy array of dihedral types
        """
        logger.info("Reading LAMMPS data from %s", self.filename)
        with open(self.filename, 'r') as f:
            lines = f.readlines()
        
        # Parse header information
        header_info = self._parse_header(lines)
        
        # Parse masses
        masses = self._parse_masses(lines, header_info['mass_start'], header_info['mass_end'])
        
        # Parse atoms
        atoms_data = self._parse_atoms(lines, header_info['atoms_start'], header_info['atoms_end'])
        
        # Parse bonds (if present)
        bonds_data = self._parse_bonds(lines, header_info['bonds_start'], header_info['bonds_end'])
        
        # Parse angles (if present)
        angles_data = self._parse_angles(lines, header_info['angles_start'], header_info['angles_end'])
        
        # Parse dihedrals (if present)
        dihedrals_data = self._parse_dihedrals(lines, header_info['dihedrals_start'], header_info['dihedrals_end'])
        
        # Parse cell information
        cell = self._parse_cell(lines, header_info)
        
        # Organize data in dpdata format
        self.data = {
            'atom_types': atoms_data['types'],
            'coords': np.array([atoms_data['coords']]),  # Add frame dimension
            'cells': np.array([cell]),  # Add frame dimension
            'atom_names': header_info['atom_names'],
            'atom_numbs': header_info['atom_numbs'],
            'masses': masses
        }
        
        # Add bonds, angles, and dihedrals if they exist
        if header_info['n_bonds'] > 0:
            self.data['bonds'] = bonds_data['bonds']
            self.data['bond_types'] = bonds_data['bond_types']
        
        if header_info['n_angles'] > 0:
            self.data['angles'] = angles_data['angles']
            self.data['angle_types'] = angles_data['angle_types']
        
        if header_info['n_dihedrals'] > 0:
            self.data['dihedrals'] = dihedrals_data['dihedrals']
            self.data['dihedral_types'] = dihedrals_data['dihedral_types']
        
        logger.debug("Parsed data: %s atoms, %s bonds, %s angles, %s dihedrals",
                     header_info.get('n_atoms'), header_info.get('n_bonds'),
                     header_info.get('n_angles'), header_info.get('n_dihedrals'))
        return self.data
    
    def _parse_header(self, lines: List[str]) -> Dict:
        """Parse the header section of the LAMMPS data file."""
        header_info = {
            'n_atoms': 0,
            'n_atom_types': 0,
            'n_bonds': 0,
            'n_bond_types': 0,
            'n_angles': 0,
            'n_angle_types': 0,
            'n_dihedrals': 0,
            'n_dihedral_types': 0,
            'atom_names': [],
            'atom_numbs': [],
            'mass_start': 0,
            'mass_end': 0,
            'atoms_start': 0,
            'atoms_end': 0,
            'bonds_start': 0,
            'bonds_end': 0,
            'angles_start': 0,
            'angles_end': 0,
            'dihedrals_start': 0,
            'dihedrals_end': 0,
            'xlo': 0.0, 'xhi': 0.0,
            'ylo': 0.0, 'yhi': 0.0,
            'zlo': 0.0, 'zhi': 0.0
        }
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('#'):
                continue
                
            if 'atoms' in line and 'atom types' not in line and line.split()[1] == 'atoms':
                header_info['n_atoms'] = int(line.split()[0])
            elif 'atom types' in line:
                header_info['n_atom_types'] = int(line.split()[0])
            elif 'bonds' in line and 'bond types' not in line and line.split()[1] == 'bonds':
                header_info['n_bonds'] = int(line.split()[0])
            elif 'bond types' in line:
                header_info['n_bond_types'] = int(line.split()[0])
            elif 'angles' in line and 'angle types' not in line and line.split()[1] == 'angles':
                header_info['n_angles'] = int(line.split()[0])
            elif 'angle types' in line:
                header_info['n_angle_types'] = int(line.split()[0])
            elif 'dihedrals' in line and 'dihedral types' not in line and line.split()[1] == 'dihedrals':
                header_info['n_dihedrals'] = int(line.split()[0])
            elif 'dihedral types' in line:
                header_info['n_dihedral_types'] = int(line.split()[0])
            elif 'xlo' in line and 'xhi' in line:
                parts = line.split()
                header_info['xlo'] = float(parts[0])
                header_info['xhi'] = float(parts[1])
            elif 'ylo' in line and 'yhi' in line:
                parts = line.split()
                header_info['ylo'] = float(parts[0])
                header_info['yhi'] = float(parts[1])
            elif 'zlo' in line and 'zhi' in line:
                parts = line.split()
                header_info['zlo'] = float(parts[0])
                header_info['zhi'] = float(parts[1])
            elif 'Masses' in line:
                header_info['mass_start'] = i + 2  # Skip the "Masses" line and empty line
            elif 'Atoms' in line:
                header_info['mass_end'] = i - 1
                header_info['atoms_start'] = i + 2  # Skip the "Atoms" line and empty line
            elif 'Bonds' in line:
                header_info['atoms_end'] = i - 1
                header_info['bonds_start'] = i + 2  # Skip the "Bonds" line and empty line
            elif 'Angles' in line:
                header_info['bonds_end'] = i - 1
                header_info['angles_start'] = i + 2  # Skip the "Angles" line and empty line
            elif 'Dihedrals' in line:
                header_info['angles_end'] = i - 1
                header_info['dihedrals_start'] = i + 2  # Skip the "Dihedrals" line and empty line
                # Set the end to the end of the file since dihedrals is the last section
                header_info['dihedrals_end'] = len(lines)
                break
        
        # If no bonds/angles/dihedrals sections were found, set atoms_end to end of file
        if header_info['atoms_end'] == 0:
            header_info['atoms_end'] = len(lines)
        
        # Generate atom names and counts (will be updated when parsing atoms)
        header_info['atom_names'] = [f'Type_{i+1}' for i in range(header_info['n_atom_types'])]
        header_info['atom_numbs'] = [0] * header_info['n_atom_types']
        
        return header_info
    
    def _parse_masses(self, lines: List[str], start: int, end: int) -> Dict[int, float]:
        """Parse the masses section."""
        masses = {}
        for i in range(start, end):
            line = lines[i].strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    atom_type = int(parts[0])
                    mass = float(parts[1])
                    masses[atom_type] = mass
        return masses
    
    def _parse_atoms(self, lines: List[str], start: int, end: int) -> Dict:
        """Parse the atoms section."""
        coords = []
        types = []
        atom_type_count = {}
        
        for i in range(start, end):
            line = lines[i].strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) == 7:  # atom_id, mol_id, atom_type, x, y, z
                    atom_type = int(parts[2])
                    x, y, z = float(parts[4]), float(parts[5]), float(parts[6])
                    
                    coords.append([x, y, z])
                    types.append(atom_type - 1)  # Convert to 0-based indexing
                    
                    atom_type_count[atom_type] = atom_type_count.get(atom_type, 0) + 1
        
        return {
            'coords': np.array(coords),
            'types': np.array(types),
            'atom_type_count': atom_type_count
        }
    
    def _parse_cell(self, lines: List[str], header_info: Dict) -> np.ndarray:
        """Parse cell information from header."""
        lx = header_info['xhi'] - header_info['xlo']
        ly = header_info['yhi'] - header_info['ylo']
        lz = header_info['zhi'] - header_info['zlo']
        
        # Create diagonal cell matrix
        cell = np.array([[lx, 0.0, 0.0],
                        [0.0, ly, 0.0],
                        [0.0, 0.0, lz]])
        
        return cell

    def _parse_bonds(self, lines: List[str], start: int, end: int) -> Dict:
        """Parse the bonds section."""
        bonds = []
        bond_types = []
        
        for i in range(start, end):
            line = lines[i].strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 4:  # bond_id, bond_type, atom1, atom2
                    bond_id = int(parts[0])
                    bond_type = int(parts[1])
                    atom1 = int(parts[2]) - 1  # Convert to 0-based indexing
                    atom2 = int(parts[3]) - 1  # Convert to 0-based indexing
                    
                    bonds.append([atom1, atom2])
                    bond_types.append(bond_type - 1)  # Convert to 0-based indexing
        
        return {
            'bonds': np.array(bonds) if bonds else np.empty((0, 2), dtype=int),
            'bond_types': np.array(bond_types) if bond_types else np.empty(0, dtype=int)
        }

    def _parse_angles(self, lines: List[str], start: int, end: int) -> Dict:
        """Parse the angles section."""
        angles = []
        angle_types = []
        
        for i in range(start, end):
            line = lines[i].strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) == 5:  # angle_id, angle_type, atom1, atom2, atom3
                    angle_id = int(parts[0])
                    angle_type = int(parts[1])
                    atom1 = int(parts[2]) - 1  # Convert to 0-based indexing
                    atom2 = int(parts[3]) - 1  # Convert to 0-based indexing
                    atom3 = int(parts[4]) - 1  # Convert to 0-based indexing
                    
                    angles.append([atom1, atom2, atom3])
                    angle_types.append(angle_type - 1)  # Convert to 0-based indexing
        
        return {
            'angles': np.array(angles) if angles else np.empty((0, 3), dtype=int),
            'angle_types': np.array(angle_types) if angle_types else np.empty(0, dtype=int)
        }

    def _parse_dihedrals(self, lines: List[str], start: int, end: int) -> Dict:
        """Parse the dihedrals section."""
        dihedrals = []
        dihedral_types = []
        
        for i in range(start, end):
            line = lines[i].strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 6:  # dihedral_id, dihedral_type, atom1, atom2, atom3, atom4
                    dihedral_id = int(parts[0])
                    dihedral_type = int(parts[1])
                    atom1 = int(parts[2]) - 1  # Convert to 0-based indexing
                    atom2 = int(parts[3]) - 1  # Convert to 0-based indexing
                    atom3 = int(parts[4]) - 1  # Convert to 0-based indexing
                    atom4 = int(parts[5]) - 1  # Convert to 0-based indexing
                    
                    dihedrals.append([atom1, atom2, atom3, atom4])
                    dihedral_types.append(dihedral_type - 1)  # Convert to 0-based indexing
        
        return {
            'dihedrals': np.array(dihedrals) if dihedrals else np.empty((0, 4), dtype=int),
            'dihedral_types': np.array(dihedral_types) if dihedral_types else np.empty(0, dtype=int)
        }


class LAMMPSDumpReader:
    """Reader for LAMMPS dump trajectory files following dpdata format."""
    
    def __init__(self, filename: str):
        """
        Initialize the LAMMPS dump file reader.
        
        Args:
            filename: Path to the LAMMPS dump file
        """
        self.filename = filename
        self.data = {}
        
    def read(self) -> Dict:
        """
        Read the LAMMPS dump file and return data in dpdata format.
        
        Returns:
            Dictionary containing the trajectory data in dpdata format:
            - atom_types: numpy array of atom types
            - coords: numpy array of coordinates (shape: (n_frames, n_atoms, 3))
            - cells: numpy array of cell vectors (shape: (n_frames, 3, 3))
            - atom_names: list of atom type names
            - atom_numbs: list of atom counts per type
            - timesteps: list of timesteps
        """
        logger.info("Reading LAMMPS dump from %s", self.filename)
        with open(self.filename, 'r') as f:
            content = f.read()
        
        # Split content into frames
        frames = self._split_frames(content)
        
        # Parse each frame
        coords_list = []
        cells_list = []
        timesteps = []
        atom_types = None
        atom_names = None
        atom_numbs = None
        
        for frame_content in frames:
            frame_data = self._parse_frame(frame_content)
            coords_list.append(frame_data['coords'])
            cells_list.append(frame_data['cell'])
            timesteps.append(frame_data['timestep'])
            
            if atom_types is None:
                atom_types = frame_data['atom_types']
                atom_names = frame_data['atom_names']
                atom_numbs = frame_data['atom_numbs']
        
        self.data = {
            'atom_types': atom_types,
            'coords': np.array(coords_list),
            'cells': np.array(cells_list),
            'atom_names': atom_names,
            'atom_numbs': atom_numbs,
            'timesteps': timesteps
        }
        
        logger.debug("Parsed dump: %s frames", len(timesteps))
        return self.data
    
    def _split_frames(self, content: str) -> List[str]:
        """Split the dump file content into individual frames."""
        # Split by "ITEM: TIMESTEP"
        frames = re.split(r'ITEM: TIMESTEP\n', content)
        # Remove empty first frame if it exists
        if frames[0].strip() == '':
            frames = frames[1:]
        return frames
    
    def _parse_frame(self, frame_content: str) -> Dict:
        """Parse a single frame from the dump file."""
        lines = frame_content.strip().split('\n')
        
        # Parse timestep
        timestep = int(lines[0])
        
        # Parse number of atoms
        n_atoms_line = None
        for line in lines:
            if 'ITEM: NUMBER OF ATOMS' in line:
                n_atoms_line = line
                break
        
        if n_atoms_line is None:
            raise ValueError("Could not find number of atoms in frame")
        
        n_atoms = int(lines[lines.index(n_atoms_line) + 1])
        
        # Parse box bounds
        box_bounds = self._parse_box_bounds(lines)
        
        # Parse atoms
        atoms_data = self._parse_frame_atoms(lines, n_atoms)
        
        return {
            'timestep': timestep,
            'coords': atoms_data['coords'],
            'cell': box_bounds,
            'atom_types': atoms_data['types'],
            'atom_names': atoms_data['atom_names'],
            'atom_numbs': atoms_data['atom_numbs']
        }
    
    def _parse_box_bounds(self, lines: List[str]) -> np.ndarray:
        """Parse box bounds from frame."""
        xlo, xhi = 0.0, 0.0
        ylo, yhi = 0.0, 0.0
        zlo, zhi = 0.0, 0.0
        
        for line in lines:
            if 'ITEM: BOX BOUNDS' in line:
                # Find the next 3 lines containing box bounds
                start_idx = lines.index(line) + 1
                for i in range(3):
                    bounds_line = lines[start_idx + i].split()
                    if i == 0:  # x bounds
                        xlo, xhi = float(bounds_line[0]), float(bounds_line[1])
                    elif i == 1:  # y bounds
                        ylo, yhi = float(bounds_line[0]), float(bounds_line[1])
                    elif i == 2:  # z bounds
                        zlo, zhi = float(bounds_line[0]), float(bounds_line[1])
                break
        
        lx = xhi - xlo
        ly = yhi - ylo
        lz = zhi - zlo
        
        # Create diagonal cell matrix
        cell = np.array([[lx, 0.0, 0.0],
                        [0.0, ly, 0.0],
                        [0.0, 0.0, lz]])
        
        return cell
    
    def _parse_frame_atoms(self, lines: List[str], n_atoms: int) -> Dict:
        """Parse atoms section from frame."""
        coords = []
        types = []
        atom_type_count = {}
        
        # Find the atoms section
        atoms_start = None
        for i, line in enumerate(lines):
            if 'ITEM: ATOMS' in line:
                atoms_start = i + 1
                break
        
        if atoms_start is None:
            raise ValueError("Could not find atoms section in frame")
        
        # Parse atom lines
        for i in range(atoms_start, atoms_start + n_atoms):
            if i < len(lines):
                line = lines[i].strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 4:  # atom_id, atom_type, x, y, z
                        atom_id = int(parts[0])
                        atom_type = int(parts[1])
                        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                        
                        coords.append([x, y, z])
                        types.append(atom_type - 1)  # Convert to 0-based indexing
                        
                        atom_type_count[atom_type] = atom_type_count.get(atom_type, 0) + 1
        
        # Generate atom names and counts
        atom_names = [f'Type_{i+1}' for i in range(max(atom_type_count.keys()) if atom_type_count else 0)]
        atom_numbs = [atom_type_count.get(i+1, 0) for i in range(len(atom_names))]
        
        return {
            'coords': np.array(coords),
            'types': np.array(types),
            'atom_names': atom_names,
            'atom_numbs': atom_numbs
        }


def read_lammps_data(filename: str) -> Dict:
    """
    Read a LAMMPS data file and return data in dpdata format.
    
    Args:
        filename: Path to the LAMMPS data file
        
    Returns:
        Dictionary containing the system data in dpdata format
    """
    reader = LAMMPSDataReader(filename)
    return reader.read()


def read_lammps_dump(filename: str) -> Dict:
    """
    Read a LAMMPS dump trajectory file and return data in dpdata format.
    
    Args:
        filename: Path to the LAMMPS dump file
        
    Returns:
        Dictionary containing the trajectory data in dpdata format
    """
    reader = LAMMPSDumpReader(filename)
    return reader.read()


def read_lammps_traj(filename: str) -> Dict:
    """
    Read a LAMMPS trajectory file (data or dump) and return data in dpdata format.
    Automatically detects the file type.
    
    Args:
        filename: Path to the LAMMPS file
        
    Returns:
        Dictionary containing the data in dpdata format
    """
    file_path = Path(filename)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File {filename} not found")
    
    # Try to detect file type based on content
    with open(filename, 'r') as f:
        first_line = f.readline().strip()
    
    if 'LAMMPS' in first_line or 'atoms' in first_line:
        # This looks like a LAMMPS data file
        logger.debug("Detected LAMMPS data format for %s", filename)
        return read_lammps_data(filename)
    elif 'ITEM: TIMESTEP' in first_line:
        # This looks like a LAMMPS dump file
        logger.debug("Detected LAMMPS dump format for %s", filename)
        return read_lammps_dump(filename)
    else:
        logger.error("Unknown LAMMPS file type for %s", filename)
        raise ValueError(f"Could not determine file type for {filename}")


# Convenience functions for backward compatibility
def load_lammps_data(filename: str) -> Dict:
    """Alias for read_lammps_data for backward compatibility."""
    return read_lammps_data(filename)


def load_lammps_dump(filename: str) -> Dict:
    """Alias for read_lammps_dump for backward compatibility."""
    return read_lammps_dump(filename)


def load_lammps_traj(filename: str) -> Dict:
    """Alias for read_lammps_traj for backward compatibility."""
    return read_lammps_traj(filename)