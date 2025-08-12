import numpy as np


def boltzmann_inversion(kbT: float, dist: np.ndarray) -> np.ndarray:
    """Compute potential of mean force via Boltzmann inversion.

    Returns -k_B T log(dist). The caller is responsible for any shifts.
    """
    with np.errstate(divide="ignore"):
        U_BI = -kbT * np.log(dist)
    return U_BI


def get_target_from_distribution(kbT: float, dist: dict):
    """Return bin centers and shifted PMF from a distribution dict.

    Expects keys 'dist' and 'bin_centers'.
    """
    U_BI = boltzmann_inversion(kbT, dist["dist"]) - boltzmann_inversion(kbT, dist["dist"])[-1]
    bin_centers = dist["bin_centers"]
    return bin_centers, U_BI


