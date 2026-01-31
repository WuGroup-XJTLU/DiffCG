# SPDX-License-Identifier: MIT
# Copyright (c) 2025 WuResearchGroup

import os
from logging import NullHandler, getLogger

# Attach a NullHandler by default so importing users don't get noisy logs unless
# they configure logging explicitly.
_logger = getLogger("diffcg")
if not _logger.handlers:
    _logger.addHandler(NullHandler())

# Re-export logging configuration helpers for convenience and allow optional
# auto-configuration for terminal output via env vars.
try:
    from diffcg._core.logger import configure as configure_logging, get_logger

    # If DIFFCG_LOG_AUTO is set truthy, or DIFFCG_LOG_LEVEL is provided,
    # auto-configure a StreamHandler that prints to the terminal.
    _auto = os.getenv("DIFFCG_LOG_AUTO")
    _level_present = os.getenv("DIFFCG_LOG_LEVEL") is not None
    if (_auto and _auto.lower() not in ("0", "false", "no")) or _level_present:
        configure_logging()
except Exception:
    # Soft-fail if optional deps are missing during partial installs
    pass

# Public API
from diffcg.system import AtomicSystem, Trajectory, from_ase_atoms, to_ase_atoms
from diffcg.energy import (
    TabulatedPairEnergy,
    TabulatedBondEnergy,
    TabulatedAngleEnergy,
    TabulatedDihedralEnergy,
    HarmonicBondEnergy,
    HarmonicAngleEnergy,
    HarmonicDihedralEnergy,
    GenericRepulsionEnergy,
)
from diffcg.md import MolecularDynamics, compute_energy, compute_energy_and_forces
from diffcg.learning.diffsim import init_diffsim, optimize_diffsim
from diffcg.learning.ibi import IBIConfig, IBITargets
from diffcg.learning.relative_entropy import init_relative_entropy
from diffcg._core.constants import BOLTZMANN_KJMOLK

__all__ = [
    # System
    "AtomicSystem",
    "Trajectory",
    "from_ase_atoms",
    "to_ase_atoms",
    # Energy
    "TabulatedPairEnergy",
    "TabulatedBondEnergy",
    "TabulatedAngleEnergy",
    "TabulatedDihedralEnergy",
    "HarmonicBondEnergy",
    "HarmonicAngleEnergy",
    "HarmonicDihedralEnergy",
    "GenericRepulsionEnergy",
    # MD
    "MolecularDynamics",
    "compute_energy",
    "compute_energy_and_forces",
    # Learning
    "init_diffsim",
    "optimize_diffsim",
    "IBIConfig",
    "IBITargets",
    "init_relative_entropy",
    # Constants
    "BOLTZMANN_KJMOLK",
    # Logging
    "configure_logging",
    "get_logger",
]
