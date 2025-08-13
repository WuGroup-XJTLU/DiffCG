# SPDX-License-Identifier: MIT
# Copyright (c) 2025 WuResearchGroup

from typing import Dict

import jax
import jax.numpy as jnp
import optax
import time
import os
from shutil import copy2
from pathlib import Path
from jax import lax
from ase.io import read

from diffcg.util.logger import get_logger
from diffcg.md.calculator import CustomCalculator, CustomEnergyCalculator, init_energy_calculator
from diffcg.learning.reweighting import ReweightEstimator
from diffcg.system import trj_atom_to_system, System
from diffcg.common.neighborlist import neighbor_list

def _tree_sum_batch(tree, axis: int = 0):
    """Sum a batched pytree along a given axis for every leaf."""
    return jax.tree_map(lambda x: jnp.sum(x, axis=axis), tree)

def _tree_add(x, y):
    return jax.tree_map(lambda a, b: a + b, x, y)

def _tree_slice(tree, start: int, end: int):
    return jax.tree_map(lambda x: x[start:end], tree)

logger = get_logger(__name__)


def _logmeanexp(x: jnp.ndarray) -> jnp.ndarray:
    """Numerically stable log-mean-exp over the first axis.

    Args:
        x: Array of shape (n, ...)

    Returns:
        log(mean(exp(x))) along axis 0
    """
    x_max = jnp.max(x, axis=0)
    return jnp.log(jnp.mean(jnp.exp(x - x_max), axis=0)) + x_max


def init_relative_entropy(
    *,
    ref_traj_path: str,
    state: Dict,
    build_energy_fn_with_params_fn,
    optimizer,
    reweight_ratio: float,
    Boltzmann_constant: float = 0.0083145107,
):
    """Initialize a single-state Relative Entropy (RE) update function.

    The RE loss (up to an additive constant independent of parameters) is
        L_RE(θ) = β E_ref[U_θ] - log E_{p_base}[exp(-β (U_θ - U_base))],
    where p_base is the CG distribution at the previous parameters (old_params).

    Args:
        ref_traj_path: ASE trajectory path of the mapped AA→CG reference frames.
        state: Dict with keys:
            - 'init_atoms': ASE Atoms
            - 'r_cut': float (optional)
            - 'sampler_params': MD sampler params; must include 'trajectory' and 'logfile' prefixes
            - 'sim_time_scheme': dict with either
                {'equilibration_steps': int, 'production_steps': int}
                or {'total_simulation_steps': int}
        build_energy_fn_with_params_fn: Callable (params, max_num_atoms) -> energy_fn
        optimizer: optax optimizer
        reweight_ratio: Threshold for n_eff to decide whether to recompute the CG trajectory.
        Boltzmann_constant: float in kJ/(mol*K)

    Returns:
        update_fn(step, params, old_params, opt_state) ->
            (new_params, params, opt_state, loss_val, metrics)
    """

    # Load reference trajectory once
    ref_trajs = read(ref_traj_path, index=':')
    if len(ref_trajs) == 0:
        raise ValueError(f"No frames found in reference trajectory: {ref_traj_path}")

    # Prepare constants and helpers
    temperature = state['sampler_params']['temperature']
    beta = 1.0 / (temperature * Boltzmann_constant)
    max_num_atoms = state['init_atoms'].get_global_number_of_atoms()

    def create_md(step, sample_energy_fn):
        sampler_params = state['sampler_params']
        r_cut = state.get('r_cut', 1.0)
        init_atoms = state['init_atoms']
        calculator = CustomCalculator(sample_energy_fn, cutoff=r_cut)
        scheme = state['sim_time_scheme']
        if 'equilibration_steps' in scheme and 'production_steps' in scheme:
            from diffcg.md.sample import MolecularDynamics

            md_equ = MolecularDynamics(
                init_atoms,
                custom_calculator=calculator,
                ensemble=sampler_params['ensemble'],
                thermostat=sampler_params['thermostat'],
                temperature=sampler_params['temperature'],
                starting_temperature=sampler_params['starting_temperature'],
                timestep=sampler_params['timestep'] ,
                trajectory=None,
                logfile=None,
                loginterval=None,
            )
            md_prod = MolecularDynamics(
                init_atoms,
                custom_calculator=calculator,
                ensemble=sampler_params['ensemble'],
                thermostat=sampler_params['thermostat'],
                temperature=sampler_params['temperature'],
                starting_temperature=sampler_params['starting_temperature'],
                timestep=sampler_params['timestep'] ,
                trajectory=f"{sampler_params['trajectory']}{step}.traj",
                logfile=f"{sampler_params['logfile']}{step}.log",
                loginterval=sampler_params['loginterval'],
            )
            return md_equ, md_prod
        else:
            from diffcg.md.sample import MolecularDynamics
            md = MolecularDynamics(
                init_atoms,
                custom_calculator=calculator,
                ensemble=sampler_params['ensemble'],
                thermostat=sampler_params['thermostat'],
                temperature=sampler_params['temperature'],
                starting_temperature=sampler_params['starting_temperature'],
                timestep=sampler_params['timestep'] ,
                trajectory=f"{sampler_params['trajectory']}{step}.traj",
                logfile=f"{sampler_params['logfile']}{step}.log",
                loginterval=sampler_params['loginterval'],
            )
            return md

    def rerun_energy(params, traj):
        results = []
        energy_fn = build_energy_fn_with_params_fn(params, max_num_atoms=max_num_atoms)
        calculator = CustomEnergyCalculator(energy_fn, cutoff=state.get('r_cut', 1.0))
        for atoms in traj:
            calculator.calculate(atoms)
            results.append(calculator.results)
        return jnp.stack(results)

    def update_fn(step, params, old_params, opt_state):
        if opt_state is None:
            opt_state = optimizer.init(params)

        sampler_params = state['sampler_params']
        scheme = state['sim_time_scheme']

        sample_energy_fn = build_energy_fn_with_params_fn(params, max_num_atoms=max_num_atoms)

        # Prepare CG proposal trajectories: run or reuse
        if step == 0:
            md_objs = create_md(step, sample_energy_fn)
            if isinstance(md_objs, tuple):
                md_equ, md_prod = md_objs
                md_equ.run(scheme['equilibration_steps'])
                md_prod.set_atoms(md_equ.atoms)
                md_prod.run(scheme['production_steps'])
            else:
                md = md_objs
                md.run(scheme['total_simulation_steps'])
            trajs_cg = read(f"{sampler_params['trajectory']}{step}.traj", index=':')
        else:
            logger.info(f"Reusing trajectory {step-1}")
            copy2(f"{sampler_params['trajectory']}{step-1}.traj", f"{sampler_params['trajectory']}{step}.traj")
            copy2(f"{sampler_params['logfile']}{step-1}.log", f"{sampler_params['logfile']}{step}.log")
            trajs_cg = read(f"{sampler_params['trajectory']}{step}.traj", index=':')

        # Base energies on CG frames under old params (reference for FEP)
        U_base_cg = rerun_energy(old_params, trajs_cg)

        def energy_theta(params, system: System, neighbors, **kwargs):
            """Energy evaluated by the underlying model with precomputed neighbors."""
            _energy_fn = build_energy_fn_with_params_fn(params, max_num_atoms=max_num_atoms)
            return _energy_fn(system, neighbors, **kwargs)

        def weighted_gradient(system: System, neighbors, weight):
            snapshot_grad = jax.grad(energy_theta)(params, system, neighbors)
            return jax.tree_map(lambda g: weight * g, snapshot_grad)

        reweight_estimator = ReweightEstimator(ref_energies=U_base_cg, kBT=temperature*Boltzmann_constant)
        
        ref_trajs_system = trj_atom_to_system(ref_trajs)
        trajs_cg_system = trj_atom_to_system(trajs_cg)

        def precompute_neighbors_batched(batched_system: System):
            batch_size = batched_system.R.shape[0]
            # Allocate on first frame to fix capacities; reuse for others via update_fn
            sys0 = System(R=batched_system.R[0], Z=batched_system.Z[0], cell=batched_system.cell[0])
            neighbors0, spatial_partitioning = neighbor_list(
                positions=sys0.R,
                cell=sys0.cell,
                cutoff=state.get('r_cut', 1.0),
                skin=0.0,
                capacity_multiplier=1.5,
            )
            neighbors_list = [neighbors0]
            for i in range(1, batch_size):
                sys_i = System(R=batched_system.R[i], Z=batched_system.Z[i], cell=batched_system.cell[i])
                nbrs_i = spatial_partitioning.update_fn(sys_i.R, neighbors0, new_cell=sys_i.cell)
                neighbors_list.append(nbrs_i)
            neighbors_batched = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *neighbors_list)
            return neighbors_batched

        ref_neighbors = precompute_neighbors_batched(ref_trajs_system)
        cg_neighbors = precompute_neighbors_batched(trajs_cg_system)

        def re_gradient(p):
            # Reference contribution: β E_ref[U_θ]
            U_ref_theta = rerun_energy(p, ref_trajs)
            ref_weights = jnp.ones(len(ref_trajs)) / (len(ref_trajs))

            # Chunked vmap to control memory
            def accumulate_batched(system_b, neighbors_b, weights_b, chunk_size: int = 32):
                num = system_b.R.shape[0]
                acc = None
                for s in range(0, num, chunk_size):
                    e = min(s + chunk_size, num)
                    sys_chunk = _tree_slice(system_b, s, e)
                    nbrs_chunk = _tree_slice(neighbors_b, s, e)
                    w_chunk = weights_b[s:e]
                    grads_chunk = jax.vmap(weighted_gradient)(sys_chunk, nbrs_chunk, w_chunk)
                    grads_sum = _tree_sum_batch(grads_chunk, axis=0)
                    acc = grads_sum if acc is None else _tree_add(acc, grads_sum)
                return acc

            mean_ref_grad = accumulate_batched(ref_trajs_system, ref_neighbors, ref_weights)

            energies_cg = rerun_energy(p, trajs_cg)
            weights, n_eff = reweight_estimator.estimate_weight(energies_cg)
            mean_gen_grad = accumulate_batched(trajs_cg_system, cg_neighbors, weights)

            combine_grads = lambda x, y: beta * (x - y)
            grad_val = jax.tree_map(combine_grads, mean_ref_grad, mean_gen_grad)

            metrics = {
                'RE_grad': grad_val,
                'n_eff': n_eff,
                'num_cg_frames': jnp.array(len(trajs_cg), dtype=jnp.float32),
                'num_ref_frames': jnp.array(len(ref_trajs), dtype=jnp.float32),
            }
            return grad_val, metrics

        grad, metrics = re_gradient(params)
        scaled_grad, opt_state = optimizer.update(grad, opt_state, params)
        new_params = optax.apply_updates(params, scaled_grad)
        
        # Decide whether to recompute trajectory for next step using n_eff heuristic (keep convention in reweighting.py)
        n_eff_val = float(metrics['n_eff'])
        recompute = (n_eff_val > reweight_ratio * len(trajs_cg)) and (step > 0)
        if recompute:
            sample_energy_fn = build_energy_fn_with_params_fn(new_params, max_num_atoms=max_num_atoms)
            logger.info(
                f"Recomputing trajectory {step} because n_eff = {n_eff_val} > {reweight_ratio * len(trajs_cg)}"
            )
            try:
                Path(f"{sampler_params['logfile']}{step}.log").unlink(missing_ok=True)
            except TypeError:
                # Python <3.8 compatibility
                p = Path(f"{sampler_params['logfile']}{step}.log")
                if p.exists():
                    p.unlink()
            md_objs = create_md(step, sample_energy_fn)
            if isinstance(md_objs, tuple):
                md_equ, md_prod = md_objs
                md_equ.run(scheme['equilibration_steps'])
                md_prod.set_atoms(md_equ.atoms)
                md_prod.run(scheme['production_steps'])
            else:
                md = md_objs
                md.run(scheme['total_simulation_steps'])

        # Placeholder loss; compute actual RE loss if needed
        leaves = jax.tree_util.tree_leaves(grad)
        grad_norm = jnp.sqrt(sum(jnp.vdot(g, g) for g in leaves))
        return new_params, params, opt_state, grad_norm, metrics

    return update_fn


def optimize_relative_entropy(update_fn, params, total_iterations):
    """Convenience optimizer loop for the RE update function.

    Returns:
        gradient_norm_history, times_per_update, metrics_history, params_set
    """
    new_params = params
    opt_state = None
    gradient_norm_history = []
    times_per_update = []
    metrics_history = []
    params_set = []

    for step in range(total_iterations):
        start_time = time.time()
        new_params, params, opt_state, grad_norm, metrics = update_fn(
            step, new_params, params, opt_state
        )
        step_time = time.time() - start_time
        logger.info(
            f"Step {step} in {step_time:0.2f} sec. RE Grad Norm = {grad_norm} | n_eff = {metrics.get('n_eff', jnp.nan)}\n\n"
        )
        if jnp.isnan(grad_norm):
            logger.error(
                'Loss is NaN. This was likely caused by divergence of the optimization or a bad model setup causing a NaN trajectory.'
            )
        gradient_norm_history.append(grad_norm)
        times_per_update.append(step_time)
        metrics_history.append(metrics)
        params_set.append(params)

    return gradient_norm_history, times_per_update, metrics_history, params_set

