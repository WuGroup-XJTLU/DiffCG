# SPDX-License-Identifier: MIT
# Copyright (c) 2025 WuResearchGroup

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple, Any

import numpy as np
import jax.numpy as jnp

from ase import units

from diffcg import energy as energy_mod
# CustomCalculator removed - MD now uses energy_fn directly
from diffcg.md.sample import MolecularDynamics
from diffcg.observable.analyze import analyze
from diffcg.observable.structure import (
    initialize_inter_radial_distribution_fun,
    initialize_bond_distribution_fun,
    initialize_angle_distribution_fun,
    initialize_dihedral_distribution_fun,
)
from diffcg.io.ase_trj import read_ase_trj
from diffcg.util.boltzmann_inversion import boltzmann_inversion


# Small numerical stabilizer for logs and divisions
EPS = 1e-12


@dataclass
class IBIConfig:
    # MD and cutoff
    r_cut: float
    r_onset: float
    sim_time_scheme: Dict[str, int]
    sampler_params: Dict[str, Any]

    # IBI hyperparameters
    alpha_pair: float = 0.2
    alpha_bond: float = 0.2
    alpha_angle: float = 0.2
    alpha_dihedral: float = 0.2

    # Smoothing (moving average window in bins; 0 disables)
    smooth_window_pair: int = 3
    smooth_window_bond: int = 3
    smooth_window_angle: int = 3
    smooth_window_dihedral: int = 3

    # Stopping
    max_iters: int = 20
    rdf_tol: float = 2e-2
    bdf_tol: float = 2e-2
    adf_tol: float = 2e-2
    ddf_tol: float = 2e-2

    # Files
    trajectory_prefix: str = "ibi"
    logfile_prefix: str = "ibi"


@dataclass
class IBITargets:
    # Each field is optional; provide dataclass instances used elsewhere in repo
    # RDF: InterRDFParams with .reference_rdf, .rdf_bin_centers, .rdf_bin_boundaries, .sigma_RDF, .exclude_mask
    rdf: Optional[Any] = None
    # BDF/ADF/DDF params carry bin centers/boundaries and topology arrays
    bdf: Optional[Any] = None
    adf: Optional[Any] = None
    ddf: Optional[Any] = None


def _moving_average(y: jnp.ndarray, window: int) -> jnp.ndarray:
    if window is None or window <= 1:
        return y
    # Use numpy for convenience, then cast back
    kernel = np.ones(int(window), dtype=float)
    kernel /= kernel.sum()
    y_np = np.asarray(y)
    # pad reflect to keep length
    pad = window // 2
    y_pad = np.pad(y_np, (pad, pad), mode="edge")
    y_smooth = np.convolve(y_pad, kernel, mode="valid")
    return jnp.array(y_smooth[: y_np.shape[0]])


def _rmse(a: jnp.ndarray, b: jnp.ndarray) -> float:
    a_np = np.asarray(a)
    b_np = np.asarray(b)
    return float(np.sqrt(np.mean((a_np - b_np) ** 2)))


class IterativeBoltzmannInversion:
    """Iterative Boltzmann Inversion driver.

    Orchestrates: initialization from targets, short MD, observable measurement,
    tabulated potential updates with damping/smoothing, and convergence checks.
    """

    def __init__(
        self,
        *,
        kBT: float,
        init_atoms,
        targets: IBITargets,
        config: IBIConfig,
        mask_topology: Optional[jnp.ndarray] = None,
        max_num_atoms: Optional[int] = None,
        initial_tables: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.kBT = float(kBT)
        self.atoms = init_atoms
        self.targets = targets
        self.cfg = config
        self.mask_topology = mask_topology
        self.max_num_atoms = max_num_atoms

        # Initialize tabulated x-grids and potentials from targets
        self.x_pair: Optional[jnp.ndarray] = None
        self.U_pair: Optional[jnp.ndarray] = None
        self.x_bond: Optional[jnp.ndarray] = None
        self.U_bond: Optional[jnp.ndarray] = None
        self.x_angle: Optional[jnp.ndarray] = None
        self.U_angle: Optional[jnp.ndarray] = None
        self.x_dihedral: Optional[jnp.ndarray] = None
        self.U_dihedral: Optional[jnp.ndarray] = None

        self._init_from_targets()
        if initial_tables:
            self._apply_initial_tables(initial_tables)

    # Initialization utilities
    def _init_from_targets(self) -> None:
        if self.targets.rdf is not None:
            tgt = self.targets.rdf.reference_rdf
            centers = self.targets.rdf.rdf_bin_centers
            U0 = boltzmann_inversion(self.kBT, np.clip(np.asarray(tgt), EPS, None))
            U0 = U0 - U0[-1]  # shift to zero at tail
            self.x_pair = jnp.array(centers)
            self.U_pair = jnp.array(U0)

        if self.targets.bdf is not None:
            tgt = self.targets.bdf.reference_bdf
            centers = self.targets.bdf.bdf_bin_centers
            U0 = boltzmann_inversion(self.kBT, np.clip(np.asarray(tgt), EPS, None))
            U0 = U0 - U0[-1]
            self.x_bond = jnp.array(centers)
            self.U_bond = jnp.array(U0)

        if self.targets.adf is not None:
            tgt = self.targets.adf.reference_adf
            centers = self.targets.adf.adf_bin_centers
            U0 = boltzmann_inversion(self.kBT, np.clip(np.asarray(tgt), EPS, None))
            # angles are periodic; allow zero shift at first bin
            U0 = U0 - U0[0]
            self.x_angle = jnp.array(centers)
            self.U_angle = jnp.array(U0)

        if self.targets.ddf is not None:
            tgt = self.targets.ddf.reference_ddf
            centers = self.targets.ddf.ddf_bin_centers
            U0 = boltzmann_inversion(self.kBT, np.clip(np.asarray(tgt), EPS, None))
            U0 = U0 - U0[0]
            self.x_dihedral = jnp.array(centers)
            self.U_dihedral = jnp.array(U0)

    def _apply_initial_tables(self, tables: Dict[str, Any]) -> None:
        """Override initial potentials from user-provided tables.

        Accepts either a y-array matching the target bin centers length, or a
        (x, y) tuple which will be linearly resampled onto the target bin centers.
        Keys: 'pair', 'bond', 'angle', 'dihedral'.
        """
        def _maybe_resample(x_target, value):
            if isinstance(value, tuple) or isinstance(value, list):
                x_in, y_in = value
                y_resampled = np.interp(np.asarray(x_target), np.asarray(x_in), np.asarray(y_in))
                return jnp.array(y_resampled)
            else:
                y_in = np.asarray(value)
                if y_in.shape != x_target.shape:
                    raise ValueError("Initial table length must match target bin centers when providing y only.")
                return jnp.array(y_in)

        if "pair" in tables and self.x_pair is not None:
            self.U_pair = _maybe_resample(self.x_pair, tables["pair"]).astype(self.U_pair.dtype)
            self.U_pair = self._pair_y_with_cutoff_shift(self.U_pair)

        if "bond" in tables and self.x_bond is not None:
            self.U_bond = _maybe_resample(self.x_bond, tables["bond"]).astype(self.U_bond.dtype)

        if "angle" in tables and self.x_angle is not None:
            self.U_angle = _maybe_resample(self.x_angle, tables["angle"]).astype(self.U_angle.dtype)

        if "dihedral" in tables and self.x_dihedral is not None:
            self.U_dihedral = _maybe_resample(self.x_dihedral, tables["dihedral"]).astype(self.U_dihedral.dtype)

    # Energy assembly
    def _build_energy_fn(self) -> Callable:
        energy_fns = []

        # Pair
        if self.U_pair is not None and self.x_pair is not None:
            pair_term = energy_mod.TabulatedPairEnergy(
                self.x_pair,
                self._pair_y_with_cutoff_shift(self.U_pair),
                self.cfg.r_onset,
                self.cfg.r_cut,
                mask_topology=self.mask_topology,
                max_num_atoms=self.max_num_atoms,
            ).get_energy_fn()
            energy_fns.append(pair_term)

        # Bond
        if self.U_bond is not None and self.x_bond is not None and self.targets.bdf is not None:
            bond_term = energy_mod.TabulatedBondEnergy(
                self.x_bond,
                self.U_bond,
                self.targets.bdf.bond_top,
            ).get_energy_fn()
            energy_fns.append(bond_term)

        # Angle
        if self.U_angle is not None and self.x_angle is not None and self.targets.adf is not None:
            angle_term = energy_mod.TabulatedAngleEnergy(
                self.x_angle,
                self.U_angle,
                self.targets.adf.angle_top,
            ).get_energy_fn()
            energy_fns.append(angle_term)

        # Dihedral
        if (
            self.U_dihedral is not None
            and self.x_dihedral is not None
            and self.targets.ddf is not None
        ):
            dihedral_term = energy_mod.TabulatedDihedralEnergy(
                self.x_dihedral,
                self.U_dihedral,
                self.targets.ddf.angle_top,
            ).get_energy_fn()
            energy_fns.append(dihedral_term)

        def sum_energy(system, neighbors, **dynamic_kwargs):
            total = 0.0
            for fn in energy_fns:
                total = total + fn(system, neighbors, **dynamic_kwargs)
            # Convert to ASE expected units
            return total * units.kJ / units.mol

        return sum_energy

    def _pair_y_with_cutoff_shift(self, U: jnp.ndarray) -> jnp.ndarray:
        # Ensure tail goes to ~0 to cooperate with multiplicative cutoff
        if U is None:
            return U
        U_np = np.asarray(U)
        U_np = U_np - U_np[-1]
        return jnp.array(U_np)

    # MD helpers
    def _create_md_equ(self, step: int, init_atoms, energy_fn: Callable):
        params = self.cfg.sampler_params

        md_equ = MolecularDynamics(
            init_atoms,
            energy_fn=energy_fn,
            ensemble=params["ensemble"],
            thermostat=params["thermostat"],
            temperature=params["temperature"],
            starting_temperature=params.get("starting_temperature", params["temperature"]),
            timestep=params["timestep"],
            cutoff=self.cfg.r_cut,
            friction=params.get("friction", 1.0),
            trajectory=None,
            logfile=None,
            loginterval=1,
        )
        return md_equ

    def _create_md_prd(self, step: int, init_atoms, energy_fn: Callable):
        params = self.cfg.sampler_params

        md_prod = MolecularDynamics(
            init_atoms,
            energy_fn=energy_fn,
            ensemble=params["ensemble"],
            thermostat=params["thermostat"],
            temperature=params["temperature"],
            starting_temperature=params.get("starting_temperature", params["temperature"]),
            timestep=params["timestep"],
            cutoff=self.cfg.r_cut,
            friction=params.get("friction", 1.0),
            trajectory=f"{self.cfg.trajectory_prefix}{step}.traj",
            logfile=f"{self.cfg.logfile_prefix}{step}.log",
            loginterval=params.get("loginterval", 100),
        )
        return md_prod

    # Observables
    def _build_quantity_fns(self) -> Dict[str, Dict[str, Any]]:
        quantity: Dict[str, Dict[str, Any]] = {}

        if self.targets.rdf is not None:
            rdf_fn = initialize_inter_radial_distribution_fun(self.targets.rdf)
            quantity["rdf"] = {"compute_fn": rdf_fn, "target": self.targets.rdf.reference_rdf}

        if self.targets.bdf is not None:
            bdf_fn = initialize_bond_distribution_fun(self.targets.bdf)
            quantity["bdf"] = {"compute_fn": bdf_fn, "target": self.targets.bdf.reference_bdf}

        if self.targets.adf is not None:
            adf_fn = initialize_angle_distribution_fun(self.targets.adf)
            quantity["adf"] = {"compute_fn": adf_fn, "target": self.targets.adf.reference_adf}

        if self.targets.ddf is not None:
            ddf_fn = initialize_dihedral_distribution_fun(self.targets.ddf)
            quantity["ddf"] = {"compute_fn": ddf_fn, "target": self.targets.ddf.reference_ddf}

        return quantity

    def _compute_observables(self, traj_path: str) -> Dict[str, jnp.ndarray]:
        systems = read_ase_trj(traj_path)
        # Build analyzers per quantity
        quantities = self._build_quantity_fns()
        obs: Dict[str, jnp.ndarray] = {}
        for name, q in quantities.items():
            analyzer = analyze(q["compute_fn"], self.atoms)
            series = analyzer.analyze(
                # stack batched systems via tree_map inside analyze
                # analyze class handles batching internally with lax.scan.
                # We just pass the stacked systems produced by read_ase_trj
                # read_ase_trj already returns list of System; analyze expects batched structure.
                # The analyze implementation converts inside using tree_map.
                # So pass as-is: a tuple/list of Systems.
                # To avoid confusion, we rely on analyze to read from Systems.
                # Here we pass a small wrapper that matches expected structure.
                # However, analyze requires a batched System (stacked). Reuse its internal batching.
                # We can safely pass the list by constructing a simple container.
                # The analyze class, as implemented, expects a batched System (tree stacked) not list.
                # Convert to batched here:
                _stack_systems(systems)
            )
            # Average across frames
            obs[name] = jnp.mean(series, axis=0)
        return obs

    # Update rules
    def _update_table(self, U: jnp.ndarray, P_sim: jnp.ndarray, P_tgt: jnp.ndarray, alpha: float, smooth_window: int) -> jnp.ndarray:
        # ΔU = alpha * kBT * ln(P_sim/P_tgt)
        ratio = (jnp.clip(P_sim, EPS) / jnp.clip(P_tgt, EPS))
        dU = alpha * self.kBT * jnp.log(ratio)
        U_new = U + dU
        U_new = _moving_average(U_new, smooth_window)
        return U_new

    def step(self, step_idx: int) -> Dict[str, Any]:
        energy_fn = self._build_energy_fn()
        scheme = self.cfg.sim_time_scheme

        md_equ = self._create_md_equ(step_idx, self.atoms, energy_fn)
        md_equ.run(scheme["equilibration_steps"])
        md_prod = self._create_md_prd(step_idx, md_equ.atoms, energy_fn)
        md_prod.run(scheme["production_steps"])

        traj_path = f"{self.cfg.trajectory_prefix}{step_idx}.traj"
        observables = self._compute_observables(traj_path)

        diagnostics: Dict[str, Any] = {"rmse": {}}

        # Pair
        if self.U_pair is not None and "rdf" in observables:
            P_sim = observables["rdf"]
            P_tgt = jnp.array(self.targets.rdf.reference_rdf)
            self.U_pair = self._update_table(
                self.U_pair,
                P_sim,
                P_tgt,
                self.cfg.alpha_pair,
                self.cfg.smooth_window_pair,
            )
            # Tail shift to zero
            self.U_pair = self._pair_y_with_cutoff_shift(self.U_pair)
            diagnostics["rmse"]["rdf"] = _rmse(P_sim, P_tgt)

        # Bond
        if self.U_bond is not None and "bdf" in observables:
            P_sim = observables["bdf"]
            P_tgt = jnp.array(self.targets.bdf.reference_bdf)
            self.U_bond = self._update_table(
                self.U_bond,
                P_sim,
                P_tgt,
                self.cfg.alpha_bond,
                self.cfg.smooth_window_bond,
            )
            diagnostics["rmse"]["bdf"] = _rmse(P_sim, P_tgt)

        # Angle
        if self.U_angle is not None and "adf" in observables:
            P_sim = observables["adf"]
            P_tgt = jnp.array(self.targets.adf.reference_adf)
            self.U_angle = self._update_table(
                self.U_angle,
                P_sim,
                P_tgt,
                self.cfg.alpha_angle,
                self.cfg.smooth_window_angle,
            )
            diagnostics["rmse"]["adf"] = _rmse(P_sim, P_tgt)

        # Dihedral
        if self.U_dihedral is not None and "ddf" in observables:
            P_sim = observables["ddf"]
            P_tgt = jnp.array(self.targets.ddf.reference_ddf)
            self.U_dihedral = self._update_table(
                self.U_dihedral,
                P_sim,
                P_tgt,
                self.cfg.alpha_dihedral,
                self.cfg.smooth_window_dihedral,
            )
            diagnostics["rmse"]["ddf"] = _rmse(P_sim, P_tgt)

        diagnostics["observables"] = observables
        return diagnostics

    def run(self) -> Dict[str, Any]:
        history = []
        for step in range(self.cfg.max_iters):
            diags = self.step(step)
            history.append(diags)
            # Convergence check on available quantities
            conv = True
            if "rdf" in diags.get("rmse", {}):
                conv = conv and diags["rmse"]["rdf"] < self.cfg.rdf_tol
            if "bdf" in diags.get("rmse", {}):
                conv = conv and diags["rmse"]["bdf"] < self.cfg.bdf_tol
            if "adf" in diags.get("rmse", {}):
                conv = conv and diags["rmse"]["adf"] < self.cfg.adf_tol
            if "ddf" in diags.get("rmse", {}):
                conv = conv and diags["rmse"]["ddf"] < self.cfg.ddf_tol
            if conv:
                break

        return {
            "pair": (self.x_pair, self.U_pair),
            "bond": (self.x_bond, self.U_bond),
            "angle": (self.x_angle, self.U_angle),
            "dihedral": (self.x_dihedral, self.U_dihedral),
            "history": history,
        }


# Utility to stack a list of System objects into a batched System tree
def _stack_systems(systems):
    # Defer import to avoid cycles
    from jax.tree_util import tree_map
    return tree_map(lambda *xs: jnp.stack(xs), *systems)


