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

    # Determine whether we need to write explicit masses
    has_custom_mass = False
    if system.masses is not None:
        masses_arr = np.asarray(system.masses)
        for i in range(n_atoms):
            sym = element_symbols[int(atom_types[i])]
            standard_mass = _ATOMIC_MASSES.get(sym, 1.0)
            if abs(masses_arr[i] - standard_mass) > 1e-3:
                has_custom_mass = True
                break

    properties = "Properties=species:S:1:pos:R:3"
    if has_custom_mass:
        properties += ":mass:R:1"
    if system.velocities is not None:
        properties += ":vel:R:3"

    with open(filepath, "w") as f:
        f.write(f"{n_atoms}\n")
        f.write(f'Time=0.0 {pbc_str} {lattice_str} {properties}\n')
        for i in range(n_atoms):
            t = int(atom_types[i])
            sym = element_symbols[t]
            line_parts = [f"{sym} {R[i,0]:.8f} {R[i,1]:.8f} {R[i,2]:.8f}"]
            if has_custom_mass:
                line_parts.append(f"{masses_arr[i]:.8f}")
            if system.velocities is not None:
                vel = np.asarray(system.velocities)
                vel_ang = vel * NM_TO_ANGSTROM
                line_parts.append(f"{vel_ang[i,0]:.8f} {vel_ang[i,1]:.8f} {vel_ang[i,2]:.8f}")
            f.write(" ".join(line_parts) + "\n")


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

_ATOMIC_MASSES = {
    "H": 1.008,
    "He": 4.0026,
    "Li": 6.94,
    "Be": 9.0122,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Ne": 20.180,
    "Na": 22.990,
    "Mg": 24.305,
    "Al": 26.982,
    "Si": 28.085,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "Ar": 39.948,
    "K": 39.098,
    "Ca": 40.078,
    "Sc": 44.956,
    "Ti": 47.867,
    "V": 50.942,
    "Cr": 51.996,
    "Mn": 54.938,
    "Fe": 55.845,
    "Co": 58.933,
    "Ni": 58.693,
    "Cu": 63.546,
    "Zn": 65.38,
    "Ga": 69.723,
    "Ge": 72.63,
    "As": 74.922,
    "Se": 78.96,
    "Br": 79.904,
    "Kr": 83.798,
    "Rb": 85.468,
    "Sr": 87.62,
    "Y": 88.906,
    "Zr": 91.224,
    "Nb": 92.906,
    "Mo": 95.95,
    "Tc": 98,
    "Ru": 101.07,
    "Rh": 102.91,
    "Pd": 106.42,
    "Ag": 107.87,
    "Cd": 112.41,
    "In": 114.82,
    "Sn": 118.71,
    "Sb": 121.76,
    "Te": 127.60,
    "I": 126.90,
    "Xe": 131.29,
    "Cs": 132.91,
    "Ba": 137.33,
    "La": 138.91,
    "Ce": 140.12,
    "Pr": 140.91,
    "Nd": 144.24,
    "Pm": 145,
    "Sm": 150.36,
    "Eu": 151.96,
    "Gd": 157.25,
    "Tb": 158.93,
    "Dy": 162.50,
    "Ho": 164.93,
    "Er": 167.26,
    "Tm": 168.93,
    "Yb": 173.05,
    "Lu": 174.97,
    "Hf": 178.49,
    "Ta": 180.95,
    "W": 183.84,
    "Re": 186.21,
    "Os": 190.23,
    "Ir": 192.22,
    "Pt": 195.08,
    "Au": 196.97,
    "Hg": 200.59,
    "Tl": 204.38,
    "Pb": 207.2,
    "Bi": 208.98,
    "Po": 209,
    "At": 210,
    "Rn": 222,
    "Fr": 223,
    "Ra": 226,
    "Ac": 227,
    "Th": 232.04,
    "Pa": 231.04,
    "U": 238.03,
    "Np": 237,
    "Pu": 244,
}
