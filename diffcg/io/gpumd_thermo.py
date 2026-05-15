"""Parse GPUMD thermo.out into a dict of arrays."""

import numpy as np
from diffcg._core.units import EV_TO_KJMOL


def read_thermo(filepath: str) -> dict:
    """Read GPUMD thermo.out file.

    Each line has 18 space-separated float values:
      temperature[K], ke[eV], pe[eV], s_xx[GPa], s_yy[GPa], s_zz[GPa],
      s_yz[GPa], s_xz[GPa], s_xy[GPa],
      h[0], h[3], h[6], h[1], h[4], h[7], h[2], h[5], h[8]

    Returns dict with keys:
      temperature (K), ke (kJ/mol), pe (kJ/mol),
      stress_xx, stress_yy, stress_zz, stress_yz, stress_xz, stress_xy (GPa),
      box (3,3) in Angstroms
    """
    data = np.loadtxt(filepath)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    return {
        "temperature": data[:, 0],          # K
        "ke": data[:, 1] * EV_TO_KJMOL,     # eV -> kJ/mol
        "pe": data[:, 2] * EV_TO_KJMOL,     # eV -> kJ/mol
        "stress_xx": data[:, 3],            # GPa
        "stress_yy": data[:, 4],
        "stress_zz": data[:, 5],
        "stress_yz": data[:, 6],
        "stress_xz": data[:, 7],
        "stress_xy": data[:, 8],
        "box_h": data[:, 9:18].reshape(-1, 3, 3),  # A
    }
