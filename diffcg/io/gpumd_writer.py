"""Write GPUMD input files (xyz.in) from DiffCG AtomicSystem."""

import numpy as np
from diffcg.system import AtomicSystem
from diffcg._core.units import NM_TO_ANGSTROM


def write_xyz_in(system: AtomicSystem, filepath: str) -> None:
    """Write GPUMD extended XYZ input file (xyz.in).

    Positions are converted from nm to Angstroms.
    Box vectors are converted from nm to Angstroms.
    """
    R = np.asarray(system.R) * NM_TO_ANGSTROM
    n_atoms = R.shape[0]
    atom_types = np.asarray(system.Z, dtype=int)

    n_types = int(atom_types.max()) + 1
    element_symbols = _ELEMENT_SYMBOLS[:n_types]

    if system.cell is not None:
        cell = np.asarray(system.cell) * NM_TO_ANGSTROM
        lattice_str = (
            f'Lattice="{cell[0,0]:.8f} {cell[1,0]:.8f} {cell[2,0]:.8f} '
            f'{cell[0,1]:.8f} {cell[1,1]:.8f} {cell[2,1]:.8f} '
            f'{cell[0,2]:.8f} {cell[1,2]:.8f} {cell[2,2]:.8f}"'
        )
    else:
        lattice_str = 'Lattice="1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0"'

    pbc_str = 'pbc="T T T"' if system.pbc else 'pbc="F F F"'

    with open(filepath, "w") as f:
        f.write(f"{n_atoms}\n")
        f.write(f'Time=0.0 {pbc_str} {lattice_str} Properties=species:S:1:pos:R:3\n')
        for i in range(n_atoms):
            t = int(atom_types[i])
            sym = element_symbols[t]
            f.write(f"{sym} {R[i,0]:.8f} {R[i,1]:.8f} {R[i,2]:.8f}")
            if system.velocities is not None:
                vel = np.asarray(system.velocities)
                vel_ang = vel * NM_TO_ANGSTROM
                f.write(f" {vel_ang[i,0]:.8f} {vel_ang[i,1]:.8f} {vel_ang[i,2]:.8f}")
            f.write("\n")


_ELEMENT_SYMBOLS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu",
]
