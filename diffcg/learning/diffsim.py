# SPDX-License-Identifier: MIT
# Copyright (c) 2025 WuResearchGroup

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax import value_and_grad
import optax
import time
import numpy as np
from collections import namedtuple
from typing import Optional, Dict
from diffcg.md.calculator import compute_energy, init_energy_calculator
from diffcg.learning.reweighting import ReweightEstimator
from diffcg.system import AtomicSystem, Trajectory, System
from diffcg._core.neighborlist import jaxmd_neighbor_list
from diffcg._core.logger import get_logger
from diffcg.md.sample import MolecularDynamics, create_equilibration_run, create_production_run
from diffcg._core.math import high_precision_sum
from diffcg._core.errors import MSE
from diffcg._core.constants import BOLTZMANN_KJMOLK
import os


logger = get_logger(__name__)

def init_independent_mse_loss_fn(quantities):
    """
    Initializes the default loss function, where MSE errors of destinct quantities are added.

    First, observables are computed via the reweighting scheme. These observables can be ndarray
    valued, e.g. vectors for RDF / ADF or matrices for stress. For each observable, the element-wise
    MSE error is computed wrt. the target provided in "quantities[quantity_key]['target']".
    This per-quantity loss is multiplied by gamma in "quantities[quantity_key]['gamma']". The final loss is
    then the sum over all of these weighted per-quantity MSE losses.
    A pre-requisite for using this function is that observables are simply ensemble averages of
    instantaneously fluctuating quantities. If this is not the case, a custom loss_fn needs to be defined.
    The custom loss_fn needs to have the same input-output signuture as the loss_fn implemented here.


    Args:
        quantities: The quantity dict with 'compute_fn', 'gamma' and 'target' for each observable

    Returns:
        The loss_fn taking trajectories of fluctuating properties, computing ensemble averages via the
        reweighting scheme and outputs the loss and predicted observables.

    """
    def loss_fn(quantity_trajs, weights):
        loss = 0.
        predictions = {}
        for quantity_key in quantities:
            quantity_snapshots = quantity_trajs[quantity_key]
            weighted_snapshots = (quantity_snapshots.T * weights).T
            ensemble_average = high_precision_sum(weighted_snapshots, axis=0)  # weights account for "averaging"
            predictions[quantity_key] = ensemble_average
            loss += quantities[quantity_key]['gamma'] * MSE(ensemble_average, quantities[quantity_key]['target'])
        return loss, predictions
    return loss_fn


def init_multistate_diffsim(
    *,
    reweight_ratio,
    states,
    build_energy_fn_with_params_fn,
    optimizer,
    Boltzmann_constant: float = BOLTZMANN_KJMOLK,
    state_weights: Optional[dict] = None,
    multiobj = None,
):
    """
    Initialize a multistate DiffSim trajectory generator and update function.

    All states share the same energy function (same parameters), but may have distinct
    quantities/targets, observable calculators, sampler settings and time schemes.

    Args:
        reweight_ratio: Threshold for n_eff to decide whether to recompute a state's trajectory.
        states: Dict mapping state_id -> dict with keys:
            - 'init_system': AtomicSystem
            - 'r_cut': float (optional; can be omitted if a global cutoff is encoded in energy fn)
            - 'quantity_dict': quantity spec for this state
            - 'calculate_observables_fn': function taking trajectory path -> dict of snapshots
            - 'sampler_params': MD sampler params; must include unique 'trajectory' and 'logfile' prefixes
            - 'sim_time_scheme': dict with either
                {'equilibration_steps': int, 'production_steps': int}
                or {'total_simulation_steps': int}
        build_energy_fn_with_params_fn: Callable (params, max_num_atoms) -> energy_fn
        optimizer: optax optimizer
        Boltzmann_constant: float in kJ/(mol*K)
        state_weights: Optional dict mapping state_id -> scalar weight (defaults to 1.0)

    Returns:
        (generate_trajectories_fn, update_fn) where:
            generate_trajectories_fn(params) -> traj_states dict (per state_id)
            update_fn(params, opt_state, traj_states) ->
                (new_params, opt_state, traj_states, total_loss, per_state_losses, predictions_by_state)
    """

    # Pre-build per-state helpers and metadata
    state_ids = list(states.keys())
    if state_weights is None:
        state_weights = {sid: 1.0 for sid in state_ids}

    # Output directory — read from any state or default to 'output'
    output_dir = states[state_ids[0]].get('output_dir', 'output')

    # Validate unique trajectory/logfile prefixes to avoid collisions across states
    traj_prefixes = []
    log_prefixes = []
    for sid in state_ids:
        sampler_params = states[sid]['sampler_params']
        traj_prefixes.append(sampler_params['trajectory'])
        log_prefixes.append(sampler_params['logfile'])
    if len(set(traj_prefixes)) != len(traj_prefixes):
        raise ValueError('All states must use unique sampler_params["trajectory"] prefixes.')
    if len(set(log_prefixes)) != len(log_prefixes):
        raise ValueError('All states must use unique sampler_params["logfile"] prefixes.')

    # Loss fns per state
    loss_fn_by_state = {
        sid: init_independent_mse_loss_fn(states[sid]['quantity_dict']) for sid in state_ids
    }

    # Precompute max atoms per state for energy fn templates
    max_atoms_by_state = {
        sid: states[sid]['init_system'].n_atoms for sid in state_ids
    }

    # Mutable counter for logging iteration directories
    _step_counter = [0]

    def build_rerun_energy_fn_for_state(state_id):
        r_cut = states[state_id].get('r_cut', 1.0)
        _state_nbrs = [None]
        _state_sp = [None]

        def rerun_energy(params, traj: Trajectory):
            energy_fn = build_energy_fn_with_params_fn(params, max_num_atoms=max_atoms_by_state[state_id])
            dtype = jnp.float64

            all_R = traj.positions.astype(dtype)
            z = traj.Z.astype(jnp.int16)
            cell = traj.cell.astype(dtype) if traj.cell is not None else None

            if _state_nbrs[0] is None or _state_sp[0] is None:
                _state_nbrs[0], _state_sp[0] = jaxmd_neighbor_list(
                    positions=all_R[0], cell=cell, cutoff=r_cut, capacity_multiplier=1.25
                )

            sp = _state_sp[0]

            @jax.jit
            def _scan_energies(all_positions, nbrs):
                def body_fn(carry, R_i):
                    nbrs_i = sp.neighbor_fn.update(R_i, carry)
                    system_i = System(R=R_i, Z=z, cell=cell)
                    e_i = energy_fn(system_i, nbrs_i)
                    overflow_i = nbrs_i.did_buffer_overflow
                    return nbrs_i, (e_i, overflow_i)
                final_nbrs, (energies, overflows) = jax.lax.scan(body_fn, nbrs, all_positions)
                return energies, overflows, final_nbrs

            energies, overflows, final_nbrs = _scan_energies(all_R, _state_nbrs[0])
            _state_nbrs[0] = final_nbrs

            if jnp.any(overflows):
                logger.warning(f"[state={state_id}] Neighbor list overflow — falling back to per-frame loop")
                results = []
                for i in range(len(traj)):
                    sys_i = traj[i]
                    sys_i_typed = AtomicSystem(R=sys_i.R.astype(dtype), Z=sys_i.Z.astype(jnp.int16), cell=sys_i.cell.astype(dtype) if sys_i.cell is not None else None)
                    e_i = compute_energy(sys_i_typed, energy_fn, cutoff=r_cut)
                    results.append(e_i)
                return jnp.stack(results)

            return energies

        return rerun_energy

    rerun_energy_by_state = {sid: build_rerun_energy_fn_for_state(sid) for sid in state_ids}

    def _create_md_for_state_equ(state_id, start_system, sample_energy_fn):
        """Create and return MD object for equilibration."""
        st = states[state_id]
        return create_equilibration_run(
            start_system, sample_energy_fn, st['sampler_params'],
            st.get('r_cut', 1.0),
        )

    def _create_md_for_state_prd(state_id, start_system, sample_energy_fn, step):
        """Create and return MD object for production."""
        st = states[state_id]
        sampler_params = st['sampler_params']
        r_cut = st.get('r_cut', 1.0)
        iter_dir = os.path.join(output_dir, f"iteration_{step}")
        os.makedirs(iter_dir, exist_ok=True)

        return create_production_run(
            start_system, sample_energy_fn, sampler_params, r_cut,
            trajectory=os.path.join(iter_dir, f"{sampler_params['trajectory']}{step}.traj"),
            logfile=os.path.join(iter_dir, f"{sampler_params['logfile']}{step}.log"),
            loginterval=sampler_params['loginterval'],
        )

    def _run_trajectory_for_state(state_id, params, start_system: AtomicSystem, step):
        """Run equilibration + production MD for a single state, return traj_state dict."""
        st = states[state_id]
        # Safety net: ensure CG masses are preserved across iterations
        init_masses = st['init_system'].masses
        if init_masses is not None and start_system.masses is not None:
            if not jnp.allclose(start_system.masses, init_masses):
                logger.warning(
                    f"[state={state_id}] Step {step}: masses mismatch — restoring CG masses"
                )
                start_system = AtomicSystem(
                    R=start_system.R, Z=start_system.Z, cell=start_system.cell,
                    masses=init_masses, pbc=start_system.pbc,
                )
        scheme = st['sim_time_scheme']
        sample_energy_fn = build_energy_fn_with_params_fn(params, max_num_atoms=max_atoms_by_state[state_id])

        md_equ = _create_md_for_state_equ(state_id, start_system, sample_energy_fn)
        md_equ.run(scheme['equilibration_steps'])
        md_prod = _create_md_for_state_prd(state_id, md_equ.get_final_system(), sample_energy_fn, step)
        md_prod.run(scheme['production_steps'])
        trajs = md_prod.get_trajectory()
        ref_energies = rerun_energy_by_state[state_id](params, trajs)
        return {'trajs': trajs, 'ref_energies': ref_energies}

    def generate_trajectories_fn(params):
        """Generate initial trajectories for all states before the optimization loop."""
        step = _step_counter[0]
        logger.debug(f"Generating initial trajectories for all states (step={step})")
        traj_states = {}
        for sid in state_ids:
            logger.debug(f"[state={sid}] Generating initial trajectory")
            traj_states[sid] = _run_trajectory_for_state(sid, params, states[sid]['init_system'], step)
        return traj_states

    def update_fn(params, opt_state, traj_states):
        """Single multistate reweighting optimization step."""
        step = _step_counter[0]

        per_state_context = {}
        for sid in state_ids:
            st = states[sid]
            sampler_params = st['sampler_params']
            trajs = traj_states[sid]['trajs']
            ref_energies = traj_states[sid]['ref_energies']

            # Build estimator from stored reference energies
            estimator = ReweightEstimator(
                ref_energies,
                kBT=sampler_params['temperature'] * Boltzmann_constant,
                base_energies=None,
                volume=None,
            )

            # Check n_eff with current params to decide recompute
            curr_energies = rerun_energy_by_state[sid](params, trajs)
            _, n_eff = estimator.estimate_weight(curr_energies)
            recompute = n_eff < reweight_ratio * len(trajs)

            if recompute:
                logger.debug(
                    f"[state={sid}] Recomputing trajectory (step={step}) because n_eff = {n_eff} < {reweight_ratio * len(trajs)}"
                )
                # Use last frame as starting point
                new_system = trajs[-1]
                traj_states[sid] = _run_trajectory_for_state(sid, params, new_system, step)
                trajs = traj_states[sid]['trajs']
                ref_energies = traj_states[sid]['ref_energies']
                estimator = ReweightEstimator(
                    ref_energies,
                    kBT=sampler_params['temperature'] * Boltzmann_constant,
                    base_energies=None,
                    volume=None,
                )

            per_state_context[sid] = {
                'trajs': trajs,
                'estimator': estimator,
                'loss_fn': loss_fn_by_state[sid],
                'weight': state_weights.get(sid, 1.0),
            }

        # Compute observables for each state from in-memory trajectories
        iter_dir = os.path.join(output_dir, f"iteration_{step}")
        observables_by_state = {
            sid: states[sid]['calculate_observables_fn'](
                os.path.join(iter_dir, f"{states[sid]['sampler_params']['trajectory']}{step}.traj"),
                batched_systems=per_state_context[sid]['trajs'].to_batched_system(),
            )
            for sid in state_ids
        }

        # Pre-compute per-state constants for inline weight computation (outside gradient path)
        grad_context_by_state = {}
        for sid in state_ids:
            ctx = per_state_context[sid]
            trajs_sid = ctx['trajs']
            sampler_params_sid = states[sid]['sampler_params']
            r_cut_sid = states[sid].get('r_cut', 1.0)
            dtype = jnp.float64

            all_R_sid = trajs_sid.positions.astype(dtype)
            z_sid = trajs_sid.Z.astype(jnp.int16)
            cell_sid = trajs_sid.cell.astype(dtype) if trajs_sid.cell is not None else None

            # Initialize neighbor lists for gradient computation
            nbrs_sid, sp_sid = jaxmd_neighbor_list(
                positions=all_R_sid[0], cell=cell_sid, cutoff=r_cut_sid, capacity_multiplier=1.25
            )

            kBT_sid = sampler_params_sid['temperature'] * Boltzmann_constant
            ref_energies_sid = traj_states[sid]['ref_energies']

            grad_context_by_state[sid] = {
                'all_R': all_R_sid,
                'z': z_sid,
                'cell': cell_sid,
                'nbrs': nbrs_sid,
                'sp': sp_sid,
                'kBT': kBT_sid,
                'ref_energies': ref_energies_sid,
                'max_num_atoms': max_atoms_by_state[sid],
            }

        if multiobj == 'coweighting':
            from diffcg.learning.multiobj import init_coweighting_stats, coweightingloss_init
            coweighting_stats = init_coweighting_stats(len(state_ids))
            coweighting_fn = coweightingloss_init()

        def wrapped_total_loss_fn(p):
            if multiobj == 'coweighting':
                nonlocal coweighting_stats  # Declare nonlocal at the beginning
            per_state_losses = {}
            predictions_by_state = {}
            for sid in state_ids:
                gctx = grad_context_by_state[sid]
                energy_fn = build_energy_fn_with_params_fn(p, max_num_atoms=gctx['max_num_atoms'])

                # Compute energies inline (no @jax.jit barrier)
                def body_fn(nbrs, R_i, _sp=gctx['sp'], _z=gctx['z'], _cell=gctx['cell']):
                    nbrs_i = _sp.neighbor_fn.update(R_i, nbrs)
                    system_i = System(R=R_i, Z=_z, cell=_cell)
                    e_i = energy_fn(system_i, nbrs_i)
                    return nbrs_i, e_i

                _, energies_new = jax.lax.scan(body_fn, gctx['nbrs'], gctx['all_R'])

                # Inline weight computation
                log_weights = -(1.0 / gctx['kBT']) * (energies_new - gctx['ref_energies'])
                log_weights = log_weights - jnp.max(log_weights)
                prob_ratios = jnp.exp(log_weights)
                weights = prob_ratios / jnp.sum(prob_ratios)

                loss_val, predictions = per_state_context[sid]['loss_fn'](observables_by_state[sid], weights)
                predictions_by_state[sid] = predictions
                per_state_losses[sid] = loss_val

            if multiobj == 'coweighting':
                # Apply coweighting algorithm
                total_loss, updated_stats = coweighting_fn(per_state_losses, coweighting_stats)
                # Update the coweighting_stats for next iteration
                coweighting_stats = updated_stats
            else:
                # Original simple weighted sum
                total_loss = sum(state_weights.get(sid, 1.0) * loss_val
                                for sid, loss_val in per_state_losses.items())
            return total_loss, (per_state_losses, predictions_by_state)

        v_and_g = value_and_grad(wrapped_total_loss_fn, has_aux=True)
        (total_loss, (per_state_losses, predictions_by_state)), grad = v_and_g(params)
        scaled_grad, opt_state_new = optimizer.update(grad, opt_state, params)
        new_params = optax.apply_updates(params, scaled_grad)

        _step_counter[0] += 1
        return new_params, opt_state_new, traj_states, total_loss, per_state_losses, predictions_by_state

    return generate_trajectories_fn, update_fn


def optimize_multistate_diffsim(generate_trajectories_fn, update_fn, params, total_iterations, *,
                                 states=None, quantity_dicts=None,
                                 output_dir="output", save_figures=False,
                                 optimizer=None):
    """
    Convenience optimizer loop for the multistate DiffSim update function.

    Phase 1: Generate initial trajectories for all states BEFORE the loop.
    Phase 2: Reweighting optimization loop.
    """
    # Phase 1: Generate initial trajectories before the optimization loop
    logger.debug("Phase 1: Generating initial trajectories for all states")
    traj_states = generate_trajectories_fn(params)

    if optimizer is not None:
        opt_state = optimizer.init(params)
    else:
        raise ValueError("optimizer is required for optimize_multistate_diffsim")

    loss_history = []
    times_per_update = []
    predictions_history = []
    params_set = []
    per_state_loss_history = []

    # Phase 2: Reweighting optimization loop
    logger.debug("Phase 2: Starting multistate reweighting optimization loop")
    for step in range(total_iterations):
        start_time = time.time()
        params_before = params
        params, opt_state, traj_states, total_loss, per_state_losses, predictions = update_fn(
            params, opt_state, traj_states
        )
        step_time = time.time() - start_time
        logger.info(
            f"Step {step} in {step_time:0.2f} sec. Total Loss = {total_loss} | per-state = {per_state_losses}\n\n"
        )
        if jnp.isnan(total_loss):
            logger.error(
                'Loss is NaN. This was likely caused by divergence of the optimization or a bad model setup '
                'causing a NaN trajectory.'
            )
        loss_history.append(total_loss)
        times_per_update.append(step_time)
        predictions_history.append(predictions)
        params_set.append(params_before)
        per_state_loss_history.append(per_state_losses)

        # Save figures if enabled
        if save_figures and quantity_dicts is not None:
            from diffcg._core.visualization import save_multistate_iteration_figures
            save_multistate_iteration_figures(
                step, predictions, states, quantity_dicts,
                loss_history, per_state_loss_history, output_dir
            )

    return loss_history, times_per_update, predictions_history, params_set, per_state_loss_history


def init_diffsim(
    *,
    reweight_ratio,
    state: Dict,
    build_energy_fn_with_params_fn,
    optimizer,
    Boltzmann_constant: float = BOLTZMANN_KJMOLK,
):
    """
    Initialize a single-state DiffSim trajectory generator and update function (functional API).

    Args:
        reweight_ratio: Threshold for n_eff to decide whether to recompute the trajectory.
        state: Dict with keys:
            - 'init_system': AtomicSystem
            - 'r_cut': float (optional)
            - 'quantity_dict': quantity spec for this state
            - 'calculate_observables_fn': function taking trajectory path -> dict of snapshots
            - 'sampler_params': MD sampler params; must include 'trajectory' and 'logfile' prefixes
            - 'sim_time_scheme': dict with either
                {'equilibration_steps': int, 'production_steps': int}
                or {'total_simulation_steps': int}
        build_energy_fn_with_params_fn: Callable (params, max_num_atoms) -> energy_fn
        optimizer: optax optimizer
        Boltzmann_constant: float in kJ/(mol*K)

    Returns:
        (generate_trajectory_fn, update_fn) where:
            generate_trajectory_fn(params) -> traj_state dict
            update_fn(params, opt_state, traj_state) ->
                (new_params, opt_state, traj_state, loss, predictions)
    """

    # Prepare reusable elements
    loss_fn = init_independent_mse_loss_fn(state['quantity_dict'])
    init_system = state['init_system']
    max_num_atoms = init_system.n_atoms
    output_dir = state.get('output_dir', 'output')

    # Mutable counter for logging iteration directories
    _step_counter = [0]

    _r_cut = state.get('r_cut', 1.0)

    def _create_md_equ(start_system, sample_energy_fn):
        return create_equilibration_run(
            start_system, sample_energy_fn, state['sampler_params'], _r_cut,
        )

    def _create_md_prd(start_system, sample_energy_fn, step):
        sampler_params = state['sampler_params']
        iter_dir = os.path.join(output_dir, f"iteration_{step}")
        os.makedirs(iter_dir, exist_ok=True)

        return create_production_run(
            start_system, sample_energy_fn, sampler_params, _r_cut,
            trajectory=os.path.join(iter_dir, f"{sampler_params['trajectory']}{step}.traj"),
            logfile=os.path.join(iter_dir, f"{sampler_params['logfile']}{step}.log"),
            loginterval=sampler_params['loginterval'],
        )

    _rerun_nbrs = None
    _rerun_sp = None

    def rerun_energy(params, traj: Trajectory):
        nonlocal _rerun_nbrs, _rerun_sp
        energy_fn = build_energy_fn_with_params_fn(params, max_num_atoms=max_num_atoms)
        r_cut = state.get('r_cut', 1.0)
        dtype = jnp.float64

        all_R = traj.positions.astype(dtype)  # (B, N, 3)
        z = traj.Z.astype(jnp.int16)  # (N,)
        cell = traj.cell.astype(dtype) if traj.cell is not None else None  # (3,3)

        # Allocate neighbor list once (reuse across calls if possible)
        if _rerun_nbrs is None or _rerun_sp is None:
            _rerun_nbrs, _rerun_sp = jaxmd_neighbor_list(
                positions=all_R[0], cell=cell, cutoff=r_cut, capacity_multiplier=1.25
            )

        @jax.jit
        def _scan_energies(all_positions, nbrs):
            def body_fn(carry, R_i):
                nbrs_i = _rerun_sp.neighbor_fn.update(R_i, carry)
                system_i = System(R=R_i, Z=z, cell=cell)
                e_i = energy_fn(system_i, nbrs_i)
                overflow_i = nbrs_i.did_buffer_overflow
                return nbrs_i, (e_i, overflow_i)
            final_nbrs, (energies, overflows) = jax.lax.scan(body_fn, nbrs, all_positions)
            return energies, overflows, final_nbrs

        energies, overflows, final_nbrs = _scan_energies(all_R, _rerun_nbrs)
        _rerun_nbrs = final_nbrs

        # Check for overflow — fall back to Python loop if any frame overflowed
        if jnp.any(overflows):
            logger.warning("Neighbor list overflow in scan — falling back to per-frame loop")
            results = []
            for i in range(len(traj)):
                sys_i = traj[i]
                sys_i_typed = AtomicSystem(R=sys_i.R.astype(dtype), Z=sys_i.Z.astype(jnp.int16), cell=sys_i.cell.astype(dtype) if sys_i.cell is not None else None)
                e_i = compute_energy(sys_i_typed, energy_fn, cutoff=r_cut)
                results.append(e_i)
            return jnp.stack(results)

        return energies

    def _run_trajectory(params, start_system: AtomicSystem, step):
        """Run equilibration + production MD and return traj_state dict."""
        # Safety net: ensure CG masses are preserved across iterations
        init_masses = init_system.masses
        if init_masses is not None and start_system.masses is not None:
            if not jnp.allclose(start_system.masses, init_masses):
                logger.warning(
                    f"Step {step}: masses mismatch detected — restoring CG masses"
                )
                start_system = AtomicSystem(
                    R=start_system.R, Z=start_system.Z, cell=start_system.cell,
                    masses=init_masses, pbc=start_system.pbc,
                )
        scheme = state['sim_time_scheme']
        sample_energy_fn = build_energy_fn_with_params_fn(params, max_num_atoms=max_num_atoms)

        md_equ = _create_md_equ(start_system, sample_energy_fn)
        md_equ.run(scheme['equilibration_steps'])
        md_prod = _create_md_prd(md_equ.get_final_system(), sample_energy_fn, step)
        md_prod.run(scheme['production_steps'])
        trajs = md_prod.get_trajectory()
        ref_energies = rerun_energy(params, trajs)

        return {'trajs': trajs, 'ref_energies': ref_energies}

    def generate_trajectory_fn(params):
        """Generate initial trajectory before the optimization loop."""
        step = _step_counter[0]
        logger.debug(f"Generating initial trajectory (step={step})")
        traj_state = _run_trajectory(params, init_system, step)
        return traj_state

    def update_fn(params, opt_state, traj_state):
        """Single reweighting optimization step."""
        nonlocal _rerun_nbrs, _rerun_sp
        step = _step_counter[0]
        sampler_params = state['sampler_params']
        trajs = traj_state['trajs']
        ref_energies = traj_state['ref_energies']

        # Build estimator from stored reference energies
        estimator = ReweightEstimator(
            ref_energies,
            kBT=sampler_params['temperature'] * Boltzmann_constant,
            base_energies=None,
            volume=None,
        )

        # Check n_eff with current params to decide recompute
        curr_energies = rerun_energy(params, trajs)
        _, n_eff = estimator.estimate_weight(curr_energies)
        recompute = n_eff < reweight_ratio * len(trajs)

        if recompute:
            logger.debug(
                f"Recomputing trajectory (step={step}) because n_eff = {n_eff} < {reweight_ratio * len(trajs)}"
            )
            # Use last frame as starting point
            new_system = trajs[-1]
            traj_state = _run_trajectory(params, new_system, step)
            trajs = traj_state['trajs']
            ref_energies = traj_state['ref_energies']
            estimator = ReweightEstimator(
                ref_energies,
                kBT=sampler_params['temperature'] * Boltzmann_constant,
                base_energies=None,
                volume=None,
            )

        # Build batched_systems from Trajectory for observables
        batched_systems = trajs.to_batched_system()

        # Compute observables — pass batched_systems to avoid disk read
        iter_dir = os.path.join(output_dir, f"iteration_{step}")
        traj_path = os.path.join(iter_dir, f"{sampler_params['trajectory']}{step}.traj")
        observables = state['calculate_observables_fn'](
            traj_path,
            batched_systems=batched_systems,
        )

        # Pre-compute constants for inline weight computation (outside gradient path)
        kBT = sampler_params['temperature'] * Boltzmann_constant
        dtype = jnp.float64
        all_R = trajs.positions.astype(dtype)
        z = trajs.Z.astype(jnp.int16)
        cell_arr = trajs.cell.astype(dtype) if trajs.cell is not None else None

        r_cut = state.get('r_cut', 1.0)
        if _rerun_nbrs is None or _rerun_sp is None:
            _rerun_nbrs, _rerun_sp = jaxmd_neighbor_list(
                positions=all_R[0], cell=cell_arr, cutoff=r_cut, capacity_multiplier=1.25
            )

        # Capture neighbor list init for use inside wrapped_loss (constant w.r.t. params)
        nbrs_for_grad = _rerun_nbrs
        sp_for_grad = _rerun_sp

        def _compute_energies_for_grad(energy_fn, positions, atomic_numbers, cell, nbrs_init):
            """Compute per-frame energies for use inside value_and_grad."""
            def body_fn(nbrs, R_i):
                nbrs_i = sp_for_grad.neighbor_fn.update(R_i, nbrs)
                system_i = System(R=R_i, Z=atomic_numbers, cell=cell)
                e_i = energy_fn(system_i, nbrs_i)
                return nbrs_i, e_i

            _, energies = jax.lax.scan(body_fn, nbrs_init, positions)
            return energies

        def wrapped_loss(p):
            energy_fn = build_energy_fn_with_params_fn(p, max_num_atoms=max_num_atoms)
            energies_new = _compute_energies_for_grad(energy_fn, all_R, z, cell_arr, nbrs_for_grad)
            # Inline weight computation (matching notebook pattern)
            log_weights = -(1.0 / kBT) * (energies_new - ref_energies)
            log_weights = log_weights - jnp.max(log_weights)
            prob_ratios = jnp.exp(log_weights)
            weights = prob_ratios / jnp.sum(prob_ratios)
            return loss_fn(observables, weights)

        v_and_g = value_and_grad(wrapped_loss, has_aux=True)
        (loss_val, predictions), grad = v_and_g(params)
        scaled_grad, opt_state_new = optimizer.update(grad, opt_state, params)
        new_params = optax.apply_updates(params, scaled_grad)

        _step_counter[0] += 1
        return new_params, opt_state_new, traj_state, loss_val, predictions

    return generate_trajectory_fn, update_fn


def optimize_diffsim(generate_trajectory_fn, update_fn, params, total_iterations, *,
                     quantity_dict=None, output_dir="output", save_figures=False,
                     optimizer=None):
    """
    Convenience optimizer loop for single-state DiffSim update function.

    Phase 1: Generate initial trajectory BEFORE the loop.
    Phase 2: Reweighting optimization loop — each step checks n_eff, possibly
    regenerates, then takes a gradient step.
    """
    # Phase 1: Generate initial trajectory before the optimization loop
    logger.debug("Phase 1: Generating initial trajectory")
    traj_state = generate_trajectory_fn(params)

    if optimizer is not None:
        opt_state = optimizer.init(params)
    else:
        # Fallback: optimizer must be passed
        raise ValueError("optimizer is required for optimize_diffsim")

    loss_history = []
    times_per_update = []
    predictions_history = []
    params_set = []

    # Phase 2: Reweighting optimization loop
    logger.debug("Phase 2: Starting reweighting optimization loop")
    for step in range(total_iterations):
        start_time = time.time()
        params_before = params
        params, opt_state, traj_state, loss_val, predictions = update_fn(
            params, opt_state, traj_state
        )
        step_time = time.time() - start_time
        logger.info('Step {} in {:0.2f} sec. Loss = {}\n\n'.format(step, step_time, loss_val))
        if jnp.isnan(loss_val):
            logger.error(
                'Loss is NaN. This was likely caused by divergence of the optimization or a bad model setup '
                'causing a NaN trajectory.'
            )
        loss_history.append(loss_val)
        times_per_update.append(step_time)
        predictions_history.append(predictions)
        params_set.append(params_before)

        # Save figures if enabled
        if save_figures and quantity_dict is not None:
            from diffcg._core.visualization import save_iteration_figures
            save_iteration_figures(step, predictions, quantity_dict, loss_history, output_dir)

    return loss_history, times_per_update, predictions_history, params_set
