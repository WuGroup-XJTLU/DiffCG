"""Deprecated module: kept for backward compatibility.

Use `diffcg.util.boltzmann_inversion` instead.
"""

from .boltzmann_inversion import boltzmann_inversion as Boltzmann_Inversion  # noqa: N813
from .boltzmann_inversion import get_target_from_distribution as get_target_dict  # noqa: F401