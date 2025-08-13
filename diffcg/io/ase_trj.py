# SPDX-License-Identifier: MIT
# Copyright (c) 2025 WuResearchGroup

# this file implement the function to read the ase trajectory following the dpdata format (https://github.com/deepmodeling/dpdata/tree/master)

import ase
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path

from diffcg.system import atoms_to_system
from diffcg.util.logger import get_logger

logger = get_logger(__name__)

def read_ase_trj(filename: str) -> Dict:
    """
    Read an ASE trajectory file and return data in dpdata format.
    
    Args:
        filename: Path to the ASE trajectory file (supports various formats like .traj, .xyz, .pdb, etc.)
        
    Returns:
        Dictionary containing the trajectory data in dpdata format:
        - atom_types: numpy array of atom types (0-based indexing)
        - coords: numpy array of coordinates (shape: (n_frames, n_atoms, 3))
        - cells: numpy array of cell vectors (shape: (n_frames, 3, 3))
        - atom_names: list of atom type names (element symbols)
        - atom_numbs: list of atom counts per type
        - masses: dictionary mapping atom types to masses (optional)
        - timesteps: list of timesteps (if available)
    """
    # Read the trajectory using ASE
    from ase import io as ase_io
    logger.info("Reading ASE trajectory from %s", filename)
    trajectory = ase_io.read(filename, index=':')
    
    # Handle single frame case
    if not isinstance(trajectory, list):
        trajectory = [trajectory]
    
    if not trajectory:
        raise ValueError(f"No frames found in trajectory file: {filename}")
    
    # Initialize lists to store frame data
    systems = []
    # Process each frame
    for i, frame in enumerate(trajectory):

        system = atoms_to_system(frame)
        systems.append(system)
    
    logger.debug("Loaded %s frames from %s", len(systems), filename)
    return systems


def _extract_cell(frame: ase.Atoms) -> np.ndarray:
    """
    Extract cell information from an ASE Atoms object.
    
    Args:
        frame: ASE Atoms object
        
    Returns:
        numpy array: 3x3 cell matrix
    """
    cell = frame.get_cell()
    
    if cell is None:
        # No cell defined, create default large cell
        return np.eye(3) * 1000.0
    
    # Check if cell is a Cell object
    if hasattr(cell, 'array'):
        # Get the cell array and transpose it to match dpdata format
        cell_array = cell.array
        if cell_array is None or np.allclose(cell_array, 0):
            # Zero or None cell, create default large cell
            return np.eye(3) * 1000.0
        return cell_array.T
    
    # If cell is already a numpy array or list
    if isinstance(cell, (list, np.ndarray)):
        cell_array = np.array(cell)
        if np.allclose(cell_array, 0):
            # Zero cell, create default large cell
            return np.eye(3) * 1000.0
        if cell_array.shape == (3,):
            # 1D cell (diagonal), convert to 3x3
            return np.diag(cell_array)
        elif cell_array.shape == (3, 3):
            # 3x3 cell, transpose to match dpdata format
            return cell_array.T
        else:
            # Unknown shape, create default cell
            return np.eye(3) * 1000.0
    
    # Fallback to default cell
    return np.eye(3) * 1000.0


def _extract_atom_info(frame: ase.Atoms) -> tuple:
    """
    Extract atom types, names, and counts from an ASE Atoms object.
    
    Args:
        frame: ASE Atoms object
        
    Returns:
        tuple: (atom_types, atom_names, atom_numbs)
    """
    # Get atomic numbers and symbols
    atomic_numbers = frame.get_atomic_numbers()
    symbols = frame.get_chemical_symbols()
    
    # Create mapping from atomic number to type index (maintain order of appearance)
    unique_numbers = []
    number_to_type = {}
    type_counter = 0
    
    for num in atomic_numbers:
        if num not in number_to_type:
            number_to_type[num] = type_counter
            unique_numbers.append(num)
            type_counter += 1
    
    # Convert atomic numbers to atom types (0-based indexing)
    atom_types = np.array([number_to_type[num] for num in atomic_numbers])
    
    # Get unique symbols in order of appearance
    atom_names = []
    for num in unique_numbers:
        # Find the first occurrence of this atomic number
        for i, anum in enumerate(atomic_numbers):
            if anum == num:
                atom_names.append(symbols[i])
                break
    
    # Count atoms of each type
    atom_numbs = []
    for num in unique_numbers:
        count = np.sum(atomic_numbers == num)
        atom_numbs.append(count)
    
    return atom_types, atom_names, atom_numbs


def _extract_masses(frame: ase.Atoms) -> Optional[Dict[int, float]]:
    """
    Extract atomic masses from an ASE Atoms object.
    
    Args:
        frame: ASE Atoms object
        
    Returns:
        Dictionary mapping atom types to masses, or None if not available
    """
    try:
        masses = frame.get_masses()
        if masses is None:
            return None
        
        # Create mapping from atom type to mass
        atomic_numbers = frame.get_atomic_numbers()
        unique_numbers = sorted(set(atomic_numbers))
        number_to_type = {num: i for i, num in enumerate(unique_numbers)}
        
        mass_dict = {}
        for num in unique_numbers:
            # Find the first occurrence of this atomic number
            for i, anum in enumerate(atomic_numbers):
                if anum == num:
                    mass_dict[number_to_type[num]] = masses[i]
                    break
        
        return mass_dict
    except:
        return None


def read_ase_single_frame(filename: str, index: int = 0) -> Dict:
    """
    Read a single frame from an ASE trajectory file and return data in dpdata format.
    
    Args:
        filename: Path to the ASE trajectory file
        index: Frame index to read (default: 0)
        
    Returns:
        Dictionary containing the frame data in dpdata format
    """
    # Read the specific frame
    from ase import io as ase_io
    frame = ase_io.read(filename, index=index)
    
    # Extract atom information
    atom_types, atom_names, atom_numbs = _extract_atom_info(frame)
    
    # Extract coordinates and cell
    coords = frame.get_positions()
    cell = _extract_cell(frame)
    
    # Create dpdata format dictionary
    dpdata_dict = {
        'atom_types': atom_types,
        'coords': np.array([coords]),  # Add frame dimension
        'cells': np.array([cell]),     # Add frame dimension
        'atom_names': atom_names,
        'atom_numbs': atom_numbs
    }
    
    # Add masses if available
    masses = _extract_masses(frame)
    if masses:
        dpdata_dict['masses'] = masses
    
    return dpdata_dict
