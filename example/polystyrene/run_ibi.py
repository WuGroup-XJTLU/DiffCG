"""Iterative Boltzmann Inversion for a polystyrene CG model.

IBI builds its own energy function internally — this script only
provides targets, system, and sampler configuration.
"""

import jax
jax.config.update("jax_enable_x64", True)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from diffcg import configure_logging
from diffcg.learning.ibi import IBITargets, IBIConfig, IterativeBoltzmannInversion

from common import (
    load_targets, load_system,
    BOLTZMANN_CONSTANT,
)

configure_logging(level="DEBUG")

# ── Configuration ──────────────────────────────────────────────────────
Temperature = 600

# ── Load data ──────────────────────────────────────────────────────────
target_dict, topology = load_targets(Temperature)
init_system, _ = load_system(Temperature)

# ── IBI setup ──────────────────────────────────────────────────────────
targets = IBITargets(
    rdf=target_dict["rdf"],
    bdf=target_dict["bdf"],
    adf=target_dict["adf"],
    ddf=target_dict["ddf"],
)

sim_time_scheme = {"equilibration_steps": 1000, "production_steps": 5000}
sampler_params = {
    "ensemble": "nvt",
    "thermostat": "berendsen",
    "temperature": Temperature,
    "starting_temperature": Temperature,
    "timestep": 4,
    "trajectory": "ibi_ps_",
    "logfile": "ibi_ps_",
    "loginterval": 10,
}

cfg = IBIConfig(
    r_cut=2.0,
    r_onset=1.5,
    sim_time_scheme=sim_time_scheme,
    sampler_params=sampler_params,
    alpha_pair=0.1,
    alpha_bond=0.1,
    alpha_angle=0.1,
    alpha_dihedral=0.1,
    smooth_window_pair=5,
    smooth_window_bond=3,
    smooth_window_angle=3,
    smooth_window_dihedral=3,
    max_iters=3,
    trajectory_prefix="ibi_ps_",
    logfile_prefix="ibi_ps_",
)

ibi = IterativeBoltzmannInversion(
    kBT=Temperature * BOLTZMANN_CONSTANT,
    system=init_system,
    targets=targets,
    config=cfg,
    mask_topology=topology["angle"],
    max_num_atoms=init_system.n_atoms,
)

# ── Run IBI ────────────────────────────────────────────────────────────
result = ibi.run()

print("IBI completed.")
for key in ("pair", "bond", "angle", "dihedral"):
    x, U = result[key]
    print(f"  {key}: {x.shape}")

for i, entry in enumerate(result["history"]):
    rmse = entry["rmse"]
    print(f"  iter {i}: rdf={rmse['rdf']:.4f}  bdf={rmse['bdf']:.4f}  "
          f"adf={rmse['adf']:.4f}  ddf={rmse['ddf']:.4f}")
