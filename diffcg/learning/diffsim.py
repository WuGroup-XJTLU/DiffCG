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
    regularizer_fn=None,
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
            - 'quantity_dict': quantity spec for this state. Each entry must have a
              'compute_fn' with signature (system, energy_fn=None, neighbors=None) -> array.
            - 'sampler_params': MD sampler params; must include unique 'trajectory' and 'logfile' prefixes
            - 'sim_time_scheme': dict with either
                {'equilibration_steps': int, 'production_steps': int}
                or {'total_simulation_steps': int}
        build_energy_fn_with_params_fn: Callable (params, max_num_atoms) -> energy_fn
        optimizer: optax optimizer
        Boltzmann_constant: float in kJ/(mol*K)
        state_weights: Optional dict mapping state_id -> scalar weight (defaults to 1.0)

    Returns:
        (generate_trajectories_fn, update_fn, loss_fn_by_state, compute_observables_fn) where:
            generate_trajectories_fn(params) -> traj_states dict (per state_id)
            update_fn(params, opt_state, traj_states) ->
                (new_params, opt_state, traj_states, total_loss, per_state_losses, predictions_by_state)
            compute_observables_fn(params, traj_states) -> dict of per-state per-frame observables
    """

    # Pre-build per-state helpers and metadata
    state_ids = list(states.keys())
    if state_weights is None:
        state_weights = {sid: 1.0 for sid in state_ids}

    # Extract per-state custom_mask_function (if any)
    mask_fn_by_state = {sid: states[sid].get('custom_mask_function', None) for sid in state_ids}

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
        _state_mask_fn = mask_fn_by_state[state_id]
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
                    positions=all_R[0], cell=cell, cutoff=r_cut, capacity_multiplier=1.25,
                    custom_mask_function=_state_mask_fn,
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
                    sys_i_typed = AtomicSystem(R=sys_i.R.astype(jnp.float32), Z=sys_i.Z.astype(jnp.int16), cell=sys_i.cell.astype(jnp.float32) if sys_i.cell is not None else None)
                    e_i = compute_energy(sys_i_typed, energy_fn, cutoff=r_cut)
                    results.append(e_i)
                return jnp.stack(results)

            return energies

        return rerun_energy

    rerun_energy_by_state = {sid: build_rerun_energy_fn_for_state(sid) for sid in state_ids}

    def _create_md_for_state_equ(state_id, start_system, sample_energy_fn, step):
        """Create and return MD object for equilibration."""
        st = states[state_id]
        _lc = st.get('lammps_config', None)
        if _lc is not None:
            _lc = dict(_lc)
            _lc["work_dir"] = os.path.join(output_dir, f"iteration_{step}", f"lammps_equ_{state_id}")
        return create_equilibration_run(
            start_system, sample_energy_fn, st['sampler_params'],
            st.get('r_cut', 1.0),
            custom_mask_function=mask_fn_by_state[state_id],
            sampler_backend=st.get('sampler_backend', 'jaxmd'),
            lammps_config=_lc,
        )

    def _create_md_for_state_prd(state_id, start_system, sample_energy_fn, step, restart_state=None):
        """Create and return MD object for production."""
        st = states[state_id]
        sampler_params = st['sampler_params']
        r_cut = st.get('r_cut', 1.0)
        iter_dir = os.path.join(output_dir, f"iteration_{step}")
        os.makedirs(iter_dir, exist_ok=True)

        _lc = st.get('lammps_config', None)
        if _lc is not None:
            _lc = dict(_lc)
            _lc["work_dir"] = os.path.join(iter_dir, f"lammps_prd_{state_id}")
        return create_production_run(
            start_system, sample_energy_fn, sampler_params, r_cut,
            trajectory=os.path.join(iter_dir, f"{sampler_params['trajectory']}{step}.traj"),
            logfile=os.path.join(iter_dir, f"{sampler_params['logfile']}{step}.log"),
            loginterval=sampler_params['loginterval'],
            custom_mask_function=mask_fn_by_state[state_id],
            sampler_backend=st.get('sampler_backend', 'jaxmd'),
            lammps_config=_lc,
            restart_state=restart_state,
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

        md_equ = _create_md_for_state_equ(state_id, start_system, sample_energy_fn, step)
        md_equ.run(scheme['equilibration_steps'])

        _backend = st.get('sampler_backend', 'jaxmd')
        if _backend == 'lammps':
            restart_state = {'restart_file': md_equ.get_restart_file()}
        elif _backend == 'fastmd':
            restart_state = {}
        elif _backend == 'gpumd':
            restart_state = {'system': md_equ.get_final_system()}
        else:
            restart_state = {
                'state': md_equ.get_final_state(),
                'neighbor': md_equ.get_final_neighbors(),
            }

        md_prod = _create_md_for_state_prd(state_id, md_equ.get_final_system(), sample_energy_fn, step,
                                           restart_state=restart_state)
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
        _step_counter[0] += 1
        return traj_states

    def update_fn(params, opt_state, traj_states):
        """Single multistate reweighting optimization step.

        Algorithm (matching notebook and single-state update_fn):
          1. Check n_eff with current params → recompute trajectory if needed
          2. Compute observables from (possibly fresh) trajectory
          3. Compute weighted loss + gradient
          4. Update params
        """
        step = _step_counter[0]

        # --- Step 1: Check n_eff per state, recompute if needed ---
        for sid in state_ids:
            trajs = traj_states[sid]['trajs']
            ref_energies = traj_states[sid]['ref_energies']
            sampler_params = states[sid]['sampler_params']

            curr_energies = rerun_energy_by_state[sid](params, trajs)
            estimator = ReweightEstimator(
                ref_energies,
                kBT=sampler_params['temperature'] * Boltzmann_constant,
                base_energies=None,
                volume=None,
            )
            _, n_eff = estimator.estimate_weight(curr_energies)
            recompute = n_eff < reweight_ratio * len(trajs)

            if recompute:
                logger.debug(
                    f"[state={sid}] Recomputing trajectory (step={step}) because n_eff = {n_eff} < {reweight_ratio * len(trajs)}"
                )
                new_system = trajs[-1]
                traj_states[sid] = _run_trajectory_for_state(sid, params, new_system, step)

        # --- Step 2: Build per-state context from (possibly fresh) trajectories ---
        per_state_context = {}
        for sid in state_ids:
            trajs = traj_states[sid]['trajs']

            per_state_context[sid] = {
                'trajs': trajs,
                'loss_fn': loss_fn_by_state[sid],
                'weight': state_weights.get(sid, 1.0),
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
                positions=all_R_sid[0], cell=cell_sid, cutoff=r_cut_sid, capacity_multiplier=1.25,
                custom_mask_function=mask_fn_by_state[sid],
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
                _quantity_dict_sid = states[sid]['quantity_dict']

                # Compute energies + observables inline (no @jax.jit barrier)
                def body_fn(nbrs, R_i, _sp=gctx['sp'], _z=gctx['z'], _cell=gctx['cell'],
                            _qd=_quantity_dict_sid):
                    nbrs_i = _sp.neighbor_fn.update(R_i, nbrs)
                    system_i = System(R=R_i, Z=_z, cell=_cell)
                    e_i = energy_fn(system_i, nbrs_i)
                    obs_i = {}
                    for qkey, qspec in _qd.items():
                        obs_i[qkey] = qspec['compute_fn'](system_i, energy_fn=energy_fn, neighbors=nbrs_i)
                    return nbrs_i, (e_i, obs_i)

                body_fn_remat = jax.checkpoint(body_fn)
                _, (energies_new, obs_per_frame) = jax.lax.scan(body_fn_remat, gctx['nbrs'], gctx['all_R'])

                # Inline weight computation
                log_weights = -(1.0 / gctx['kBT']) * (energies_new - gctx['ref_energies'])
                log_weights = log_weights - jnp.max(log_weights)
                prob_ratios = jnp.exp(log_weights)
                weights = prob_ratios / jnp.sum(prob_ratios)

                loss_val, predictions = per_state_context[sid]['loss_fn'](obs_per_frame, weights)
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
            if regularizer_fn is not None:
                total_loss = total_loss + regularizer_fn(p)
            return total_loss, (per_state_losses, predictions_by_state)

        v_and_g = value_and_grad(wrapped_total_loss_fn, has_aux=True)
        (total_loss, (per_state_losses, predictions_by_state)), grad = v_and_g(params)
        scaled_grad, opt_state_new = optimizer.update(grad, opt_state, params)
        new_params = optax.apply_updates(params, scaled_grad)

        _step_counter[0] += 1
        return new_params, opt_state_new, traj_states, total_loss, per_state_losses, predictions_by_state

    def compute_observables_fn(params, traj_states):
        """Compute per-frame observables for all states (outside gradient tape).

        Used for initial predictions before the optimization loop.
        """
        result = {}
        for sid in state_ids:
            trajs = traj_states[sid]['trajs']
            r_cut_sid = states[sid].get('r_cut', 1.0)
            dtype = jnp.float64
            all_R = trajs.positions.astype(dtype)
            z = trajs.Z.astype(jnp.int16)
            cell = trajs.cell.astype(dtype) if trajs.cell is not None else None

            nbrs_sid, sp_sid = jaxmd_neighbor_list(
                positions=all_R[0], cell=cell, cutoff=r_cut_sid, capacity_multiplier=1.25,
                custom_mask_function=mask_fn_by_state[sid],
            )
            energy_fn = build_energy_fn_with_params_fn(params, max_num_atoms=max_atoms_by_state[sid])
            _qd = states[sid]['quantity_dict']

            def _scan_obs(all_positions, _energy_fn=energy_fn, _z=z, _cell=cell,
                          _sp=sp_sid, _nbrs=nbrs_sid, _qd_inner=_qd):
                def body_fn(nbrs_carry, R_i):
                    nbrs_i = _sp.neighbor_fn.update(R_i, nbrs_carry)
                    system_i = System(R=R_i, Z=_z, cell=_cell)
                    obs_i = {}
                    for qkey, qspec in _qd_inner.items():
                        obs_i[qkey] = qspec['compute_fn'](system_i, energy_fn=_energy_fn, neighbors=nbrs_i)
                    return nbrs_i, obs_i
                _, obs = jax.lax.scan(body_fn, _nbrs, all_positions)
                return obs

            result[sid] = jax.jit(_scan_obs)(all_R)
        return result

    return generate_trajectories_fn, update_fn, loss_fn_by_state, compute_observables_fn


def optimize_multistate_diffsim(generate_trajectories_fn, update_fn, params, total_iterations, *,
                                 states=None, quantity_dicts=None,
                                 output_dir="output", save_figures=False,
                                 optimizer=None, loss_fn_by_state=None,
                                 compute_observables_fn=None):
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

    # Compute and save initial observables (uniform weights since params == generating params)
    if save_figures and quantity_dicts is not None and loss_fn_by_state is not None and compute_observables_fn is not None:
        from diffcg._core.visualization import save_multistate_iteration_figures
        init_obs_by_state = compute_observables_fn(params, traj_states)
        init_predictions = {}
        for sid, traj_st in traj_states.items():
            n_frames = len(traj_st['trajs'])
            uniform_weights = jnp.ones(n_frames) / n_frames
            _, preds = loss_fn_by_state[sid](init_obs_by_state[sid], uniform_weights)
            init_predictions[sid] = preds
        save_multistate_iteration_figures(
            0, init_predictions, states, quantity_dicts, [], [], output_dir
        )
        logger.info("Saved initial multistate observable figures to iteration_0")

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
                step + 1, predictions, states, quantity_dicts,
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
    regularizer_fn=None,
    start_step=0,
    max_frames=2000,
):
    """
    Initialize a single-state DiffSim trajectory generator and update function (functional API).

    Args:
        reweight_ratio: Threshold for n_eff to decide whether to recompute the trajectory.
        state: Dict with keys:
            - 'init_system': AtomicSystem
            - 'r_cut': float (optional)
            - 'quantity_dict': quantity spec for this state. Each entry must have a
              'compute_fn' with signature (system, energy_fn=None, neighbors=None) -> array.
            - 'sampler_params': MD sampler params; must include 'trajectory' and 'logfile' prefixes
            - 'sim_time_scheme': dict with either
                {'equilibration_steps': int, 'production_steps': int}
                or {'total_simulation_steps': int}
        build_energy_fn_with_params_fn: Callable (params, max_num_atoms) -> energy_fn
        optimizer: optax optimizer
        Boltzmann_constant: float in kJ/(mol*K)
        start_step: Starting step number for iteration folder naming. Default 0.
            Set to N+1 when resuming from checkpoint at iteration N.
        max_frames: Maximum trajectory frames to process in JAX scans.
            If trajectory exceeds this, frames are uniformly subsampled.
            Default 2000. Set to None to disable subsampling.

    Returns:
        (generate_trajectory_fn, update_fn, compute_observables_fn) where:
            generate_trajectory_fn(params) -> traj_state dict
            update_fn(params, opt_state, traj_state) ->
                (new_params, opt_state, traj_state, loss, predictions)
            compute_observables_fn(params, traj) -> dict of per-frame observables
    """

    # Prepare reusable elements
    loss_fn = init_independent_mse_loss_fn(state['quantity_dict'])
    init_system = state['init_system']
    max_num_atoms = init_system.n_atoms
    output_dir = state.get('output_dir', 'output')
    _custom_mask_function = state.get('custom_mask_function', None)

    # Mutable counter for logging iteration directories
    _step_counter = [start_step]

    _r_cut = state.get('r_cut', 1.0)
    _sampler_backend = state.get('sampler_backend', 'jaxmd')
    _lammps_config = state.get('lammps_config', None)

    def _create_md_equ(start_system, sample_energy_fn, step):
        if _lammps_config is not None:
            lc = dict(_lammps_config)
            lc["work_dir"] = os.path.join(output_dir, f"iteration_{step}", "lammps_equ")
        else:
            lc = None
        return create_equilibration_run(
            start_system, sample_energy_fn, state['sampler_params'], _r_cut,
            custom_mask_function=_custom_mask_function,
            sampler_backend=_sampler_backend,
            lammps_config=lc,
        )

    def _create_md_prd(start_system, sample_energy_fn, step, restart_state=None):
        sampler_params = state['sampler_params']
        iter_dir = os.path.join(output_dir, f"iteration_{step}")
        os.makedirs(iter_dir, exist_ok=True)

        if _lammps_config is not None:
            lc = dict(_lammps_config)
            lc["work_dir"] = os.path.join(iter_dir, "lammps_prd")
        else:
            lc = None
        return create_production_run(
            start_system, sample_energy_fn, sampler_params, _r_cut,
            trajectory=os.path.join(iter_dir, f"{sampler_params['trajectory']}{step}.traj"),
            logfile=os.path.join(iter_dir, f"{sampler_params['logfile']}{step}.log"),
            loginterval=sampler_params['loginterval'],
            custom_mask_function=_custom_mask_function,
            sampler_backend=_sampler_backend,
            lammps_config=lc,
            restart_state=restart_state,
        )

    _rerun_nbrs = None
    _rerun_sp = None
    _obs_cache = None  # (traj_id, obs_per_frame) — invalidated when trajectory changes

    def _subsample_all_R(all_R):
        """Uniformly subsample frames if trajectory exceeds max_frames."""
        if max_frames is None or all_R.shape[0] <= max_frames:
            return all_R
        indices = np.random.choice(all_R.shape[0], max_frames, replace=False)
        return all_R[indices]

    def rerun_energy(params, traj: Trajectory):
        nonlocal _rerun_nbrs, _rerun_sp
        energy_fn = build_energy_fn_with_params_fn(params, max_num_atoms=max_num_atoms)
        r_cut = state.get('r_cut', 1.0)
        all_R = traj.positions.astype(jnp.float32)  # (B, N, 3)
        z = traj.Z.astype(jnp.int16)  # (N,)
        cell = traj.cell.astype(jnp.float32) if traj.cell is not None else None  # (3,3)

        all_R = _subsample_all_R(all_R)

        # Allocate neighbor list once (reuse across calls if possible)
        if _rerun_nbrs is None or _rerun_sp is None:
            _rerun_nbrs, _rerun_sp = jaxmd_neighbor_list(
                positions=all_R[0], cell=cell, cutoff=r_cut, capacity_multiplier=2.0,
                custom_mask_function=_custom_mask_function,
            )

        @jax.jit
        def _scan_energies(all_positions, nbrs):
            def body_fn(carry, R_i):
                nbrs_i = _rerun_sp.neighbor_fn.update(R_i, carry)
                system_i = System(R=R_i, Z=z, cell=cell)
                e_i = energy_fn(system_i, nbrs_i)
                overflow_i = nbrs_i.did_buffer_overflow
                return nbrs_i, (e_i, overflow_i)
            final_nbrs, (energies, overflows) = jax.lax.scan(body_fn, nbrs, all_positions, unroll=1)
            return energies, overflows, final_nbrs

        energies, overflows, final_nbrs = _scan_energies(all_R, _rerun_nbrs)
        _rerun_nbrs = final_nbrs

        # Check for overflow — fall back to Python loop if any frame overflowed
        if jnp.any(overflows):
            logger.warning("Neighbor list overflow in scan — falling back to per-frame loop")
            results = []
            for i in range(len(traj)):
                sys_i = traj[i]
                sys_i_typed = AtomicSystem(R=sys_i.R.astype(jnp.float32), Z=sys_i.Z.astype(jnp.int16), cell=sys_i.cell.astype(jnp.float32) if sys_i.cell is not None else None)
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

        md_equ = _create_md_equ(start_system, sample_energy_fn, step)
        md_equ.run(scheme['equilibration_steps'])

        if _sampler_backend == 'lammps':
            restart_state = {'restart_file': md_equ.get_restart_file()}
        elif _sampler_backend == 'fastmd':
            restart_state = {}
        elif _sampler_backend == 'gpumd':
            restart_state = {'system': md_equ.get_final_system()}
        else:
            restart_state = {
                'state': md_equ.get_final_state(),
                'neighbor': md_equ.get_final_neighbors(),
            }

        md_prod = _create_md_prd(md_equ.get_final_system(), sample_energy_fn, step,
                                 restart_state=restart_state)
        md_prod.run(scheme['production_steps'])
        trajs = md_prod.get_trajectory()
        ref_energies = rerun_energy(params, trajs)

        return {'trajs': trajs, 'ref_energies': ref_energies}

    def generate_trajectory_fn(params):
        """Generate initial trajectory before the optimization loop."""
        step = _step_counter[0]
        logger.debug(f"Generating initial trajectory (step={step})")
        traj_state = _run_trajectory(params, init_system, step)
        _step_counter[0] += 1
        return traj_state

    def _compute_observables(quantity_dict, energy_fn, all_R, z, cell, nbrs, sp):
        """Scan over frames computing all observables. Used for initial predictions."""
        def body_fn(nbrs_carry, R_i):
            nbrs_i = sp.neighbor_fn.update(R_i, nbrs_carry)
            system_i = System(R=R_i, Z=z, cell=cell)
            obs_i = {}
            for qkey, qspec in quantity_dict.items():
                obs_i[qkey] = qspec['compute_fn'](system_i, energy_fn=energy_fn, neighbors=nbrs_i)
            return nbrs_i, obs_i

        _, obs_per_frame = jax.lax.scan(body_fn, nbrs, all_R)
        return obs_per_frame

    def _precompute_observables(all_R, z, cell, nbrs, sp, energy_fn):
        """Scan over frames computing all observables. No gradient tracking.

        Called once per trajectory change (not JIT'd since sp contains function
        attributes that can't be traced as explicit parameters).
        """
        def body_fn(nbrs_carry, R_i):
            nbrs_i = sp.neighbor_fn.update(R_i, nbrs_carry)
            system_i = System(R=R_i, Z=z, cell=cell)
            obs_i = {}
            for qkey, qspec in state['quantity_dict'].items():
                obs_i[qkey] = qspec['compute_fn'](system_i, energy_fn=energy_fn, neighbors=nbrs_i)
            return nbrs_i, obs_i
        _, obs_per_frame = jax.lax.scan(body_fn, nbrs, all_R)
        return obs_per_frame

    def compute_observables_fn(params, traj: Trajectory):
        """Compute per-frame observables for a trajectory (outside gradient tape).

        Used for initial predictions before the optimization loop.
        """
        nonlocal _rerun_nbrs, _rerun_sp
        energy_fn = build_energy_fn_with_params_fn(params, max_num_atoms=max_num_atoms)
        r_cut = state.get('r_cut', 1.0)
        all_R = traj.positions.astype(jnp.float32)
        z = traj.Z.astype(jnp.int16)
        cell = traj.cell.astype(jnp.float32) if traj.cell is not None else None

        all_R = _subsample_all_R(all_R)

        if _rerun_nbrs is None or _rerun_sp is None:
            _rerun_nbrs, _rerun_sp = jaxmd_neighbor_list(
                positions=all_R[0], cell=cell, cutoff=r_cut, capacity_multiplier=2.0,
                custom_mask_function=_custom_mask_function,
            )

        return jax.jit(lambda aR: _compute_observables(
            state['quantity_dict'], energy_fn, aR, z, cell, _rerun_nbrs, _rerun_sp
        ))(all_R)

    def update_fn(params, opt_state, traj_state):
        """Single reweighting optimization step.

        Algorithm:
          1. Check n_eff with current params → recompute trajectory if too low
          2. Compute weighted loss + gradient (observables computed inside value_and_grad)
          3. Update params
        """
        nonlocal _rerun_nbrs, _rerun_sp, _obs_cache
        step = _step_counter[0]
        sampler_params = state['sampler_params']

        # --- Step 1: Check n_eff and possibly recompute trajectory ---
        trajs = traj_state['trajs']
        ref_energies = traj_state['ref_energies']
        curr_energies = rerun_energy(params, trajs)
        estimator = ReweightEstimator(
            ref_energies,
            kBT=sampler_params['temperature'] * Boltzmann_constant,
            base_energies=None,
            volume=None,
        )
        _, n_eff = estimator.estimate_weight(curr_energies)
        recompute = n_eff < reweight_ratio * len(trajs)

        if recompute:
            logger.debug(
                f"Recomputing trajectory (step={step}) because n_eff = {n_eff} < {reweight_ratio * len(trajs)}"
            )
            new_system = trajs[-1]
            traj_state = _run_trajectory(params, new_system, step)

        # Re-read (possibly updated) trajectory state
        trajs = traj_state['trajs']
        ref_energies = traj_state['ref_energies']

        # --- Precompute observables if trajectory changed or first call ---
        traj_id = id(trajs)
        if _obs_cache is None or _obs_cache[0] != traj_id:
            all_R_obs = trajs.positions.astype(jnp.float32)
            z_obs = trajs.Z.astype(jnp.int16)
            cell_obs = trajs.cell.astype(jnp.float32) if trajs.cell is not None else None
            r_cut = state.get('r_cut', 1.0)

            energy_fn_cache = build_energy_fn_with_params_fn(params, max_num_atoms=max_num_atoms)

            nbrs_obs, sp_obs = jaxmd_neighbor_list(
                positions=all_R_obs[0], cell=cell_obs, cutoff=r_cut,
                capacity_multiplier=2.0,
                custom_mask_function=_custom_mask_function,
            )
            obs_full = _precompute_observables(
                all_R_obs, z_obs, cell_obs, nbrs_obs, sp_obs, energy_fn_cache,
            )
            _obs_cache = (traj_id, obs_full)

        # --- Step 2: Compute weighted loss + gradient ---
        kBT = sampler_params['temperature'] * Boltzmann_constant
        # Apply frame subsampling (consistent indices for positions, energies, observables)
        all_R_full = trajs.positions.astype(jnp.float32)
        obs_full = _obs_cache[1]  # dict of {qkey: (num_frames, ...)}
        if max_frames is not None and all_R_full.shape[0] > max_frames:
            indices = np.random.choice(all_R_full.shape[0], max_frames, replace=False)
            indices = np.sort(indices)  # sort for scan stability
            all_R = all_R_full[indices]
            ref_energies_sub = ref_energies[indices]
            obs_per_frame = {k: v[indices] for k, v in obs_full.items()}
        else:
            all_R = all_R_full
            ref_energies_sub = ref_energies
            obs_per_frame = obs_full

        z = trajs.Z.astype(jnp.int16)
        cell_arr = trajs.cell.astype(jnp.float32) if trajs.cell is not None else None

        r_cut = state.get('r_cut', 1.0)
        if _rerun_nbrs is None or _rerun_sp is None:
            _rerun_nbrs, _rerun_sp = jaxmd_neighbor_list(
                positions=all_R[0], cell=cell_arr, cutoff=r_cut, capacity_multiplier=2.0,
                custom_mask_function=_custom_mask_function,
            )

        nbrs_for_grad = _rerun_nbrs
        sp_for_grad = _rerun_sp

        # Capture precomputed observables for use inside wrapped_loss
        _obs_cached = obs_per_frame

        def wrapped_loss(p):
            energy_fn = build_energy_fn_with_params_fn(p, max_num_atoms=max_num_atoms)

            def body_fn(nbrs, R_i):
                nbrs_i = sp_for_grad.neighbor_fn.update(R_i, nbrs)
                system_i = System(R=R_i, Z=z, cell=cell_arr)
                e_i = energy_fn(system_i, nbrs_i)
                return nbrs_i, e_i

            body_fn_remat = jax.checkpoint(body_fn)
            _, energies_new = jax.lax.scan(body_fn_remat, nbrs_for_grad, all_R)

            log_weights = -(1.0 / kBT) * (energies_new - ref_energies_sub)
            log_weights = log_weights - jnp.max(log_weights)
            prob_ratios = jnp.exp(log_weights)
            weights = prob_ratios / jnp.sum(prob_ratios)
            loss_val, predictions = loss_fn(_obs_cached, weights)
            if regularizer_fn is not None:
                loss_val = loss_val + regularizer_fn(p)
            return loss_val, predictions

        v_and_g = value_and_grad(wrapped_loss, has_aux=True)
        (loss_val, predictions), grad = v_and_g(params)

        # --- Diagnostic logging ---
        grad_leaves = jax.tree_util.tree_leaves(grad)
        grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in grad_leaves))
        logger.info(f"Step {step} | grad_norm = {grad_norm:.6f}")

        # --- Step 3: Update params ---
        scaled_grad, opt_state_new = optimizer.update(grad, opt_state, params)
        new_params = optax.apply_updates(params, scaled_grad)

        param_leaves_new = jax.tree_util.tree_leaves(new_params)
        param_leaves_old = jax.tree_util.tree_leaves(params)
        param_delta_norm = jnp.sqrt(sum(jnp.sum((n - o) ** 2) for n, o in zip(param_leaves_new, param_leaves_old)))
        logger.info(f"Step {step} | param_delta_norm = {param_delta_norm:.6f}")

        _step_counter[0] += 1
        return new_params, opt_state_new, traj_state, loss_val, predictions

    return generate_trajectory_fn, update_fn, compute_observables_fn


def _save_checkpoint(step, params, opt_state, traj_state, loss_history, output_dir):
    """Save checkpoint data to iteration_{step}/checkpoint.npz.

    Saves flattened optax optimizer state (momentum accumulators, schedule step
    counter), reference energies for reweighting, and loss history. Params and
    trajectory are NOT duplicated — they remain in their existing files
    (params.npz and *.traj.npz).
    """
    import numpy as np

    flat_opt, _tree_opt = jax.tree_util.tree_flatten(opt_state)
    save_dict = {
        'ref_energies': np.asarray(traj_state['ref_energies']),
        'step': np.array(step),
        'n_opt_leaves': np.array(len(flat_opt)),
    }
    for i, leaf in enumerate(flat_opt):
        save_dict[f'opt_state_{i}'] = np.asarray(leaf)
    if loss_history:
        save_dict['loss_history'] = np.array(loss_history, dtype=np.float64)

    iter_dir = os.path.join(output_dir, f"iteration_{step}")
    os.makedirs(iter_dir, exist_ok=True)
    ckpt_path = os.path.join(iter_dir, "checkpoint.npz")
    np.savez(ckpt_path, **save_dict)
    logger.debug(f"Saved checkpoint to {ckpt_path}")


def _load_checkpoint(resume_from, output_dir, optimizer, params):
    """Load checkpoint from iteration_{resume_from}/ and return all state.

    Reads checkpoint.npz (opt_state, ref_energies), params.npz, and the
    trajectory file (found via glob — exactly one *.traj.npz per folder).
    The optax optimizer tree structure is re-derived from a fresh init
    so no tree metadata needs to be serialized.

    Returns:
        (params, opt_state, traj_state, loss_history)
    """
    import glob
    import numpy as np

    ckpt_dir = os.path.join(output_dir, f"iteration_{resume_from}")
    ckpt_path = os.path.join(ckpt_dir, "checkpoint.npz")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. "
            f"Iteration {resume_from} was likely created by an older version "
            f"of diffcg that does not support checkpointing. "
            f"Run without resume_from to start fresh."
        )

    ckpt = np.load(ckpt_path, allow_pickle=False)

    # Load params
    params_path = os.path.join(ckpt_dir, "params.npz")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"params.npz not found in {ckpt_dir}")
    params_np = np.load(params_path, allow_pickle=True)
    if "treedef" in params_np:
        import pickle
        params_treedef = pickle.loads(params_np["treedef"])
        n_p_leaves = sum(1 for k in params_np.keys() if k.startswith("p_"))
        params_leaves = [jnp.asarray(params_np[f"p_{i}"]) for i in range(n_p_leaves)]
        loaded_params = jax.tree_util.tree_unflatten(params_treedef, params_leaves)
    else:
        # Backward compatibility: old format with single "pair" array
        loaded_params = {"pair": jnp.asarray(params_np["pair"])}

    # Load trajectory (find the .traj.npz file — exactly one per folder)
    traj_files = glob.glob(os.path.join(ckpt_dir, "*.traj.npz"))
    if not traj_files:
        raise FileNotFoundError(f"No .traj.npz file found in {ckpt_dir}")
    if len(traj_files) > 1:
        logger.warning(
            f"Multiple .traj.npz files found in {ckpt_dir}, using {traj_files[0]}"
        )
    traj = Trajectory.load(traj_files[0])

    # Load ref_energies
    ref_energies = jnp.asarray(ckpt["ref_energies"])
    traj_state = {"trajs": traj, "ref_energies": ref_energies}

    # Load opt_state — re-derive tree structure from optimizer
    n_leaves = int(ckpt["n_opt_leaves"])
    opt_leaves = [jnp.asarray(ckpt[f"opt_state_{i}"]) for i in range(n_leaves)]
    _, tree_opt = jax.tree_util.tree_flatten(optimizer.init(loaded_params))
    opt_state = jax.tree_util.tree_unflatten(tree_opt, opt_leaves)

    # Load loss history
    loss_history = []
    if "loss_history" in ckpt:
        loss_history = [float(x) for x in ckpt["loss_history"]]
    else:
        # Fallback: try CSV
        csv_path = os.path.join(ckpt_dir, "loss_history.csv")
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) == 2:
                        loss_history.append(float(parts[1]))

    logger.info(
        f"Loaded checkpoint from iteration {resume_from}: "
        f"{len(loss_history)} loss values, "
        f"{len(traj)} trajectory frames"
    )
    return loaded_params, opt_state, traj_state, loss_history


def optimize_diffsim(generate_trajectory_fn, update_fn, params, total_iterations, *,
                     quantity_dict=None, output_dir="output", save_figures=False,
                     optimizer=None, compute_observables_fn=None, loss_fn=None,
                     resume_from=None):
    """
    Convenience optimizer loop for single-state DiffSim update function.

    Phase 1: Generate initial trajectory BEFORE the loop.
    Phase 2: Reweighting optimization loop — each step checks n_eff, possibly
    regenerates, then takes a gradient step.

    Args:
        resume_from: If set to N, load checkpoint from iteration_{N}/ and resume
                     from step N+1. Requires that iteration_{N}/ contains a
                     checkpoint.npz file (produced by diffcg >= this version).
    """
    if resume_from is not None:
        # --- Resume from checkpoint ---
        if optimizer is None:
            raise ValueError("optimizer is required for resume")
        params, opt_state, traj_state, loss_history = _load_checkpoint(
            resume_from, output_dir, optimizer, params
        )

        times_per_update = []
        predictions_history = []
        params_set = []

        logger.info(f"Resumed from iteration {resume_from}. "
                     f"Running steps {resume_from + 1} to {total_iterations - 1}.")

        for step in range(resume_from + 1, total_iterations):
            start_time = time.time()
            params_before = params
            params, opt_state, traj_state, loss_val, predictions = update_fn(
                params, opt_state, traj_state
            )
            step_time = time.time() - start_time
            logger.info('Step {} in {:0.2f} sec. Loss = {}\n\n'.format(step, step_time, loss_val))
            if jnp.isnan(loss_val):
                logger.error(
                    'Loss is NaN. This was likely caused by divergence of the '
                    'optimization or a bad model setup causing a NaN trajectory.'
                )
            loss_history.append(loss_val)
            times_per_update.append(step_time)
            predictions_history.append(predictions)
            params_set.append(params_before)

            # Save parameters for post-run diagnostics
            iter_dir = os.path.join(output_dir, f"iteration_{step}")
            os.makedirs(iter_dir, exist_ok=True)
            import pickle
            flat_params, params_treedef = jax.tree_util.tree_flatten(params)
            params_save_dict = {f"p_{i}": np.asarray(leaf) for i, leaf in enumerate(flat_params)}
            params_save_dict["treedef"] = pickle.dumps(params_treedef)
            np.savez(os.path.join(iter_dir, "params.npz"), **params_save_dict)

            # Save checkpoint (opt_state + ref_energies + loss_history)
            _save_checkpoint(step, params, opt_state, traj_state, loss_history, output_dir)

            # Save figures if enabled
            if save_figures and quantity_dict is not None:
                from diffcg._core.visualization import save_iteration_figures
                save_iteration_figures(step, predictions, quantity_dict, loss_history, output_dir)

        return loss_history, times_per_update, predictions_history, params_set

    # --- Original path (no resume) ---
    # Phase 1: Generate initial trajectory before the optimization loop
    logger.debug("Phase 1: Generating initial trajectory")
    traj_state = generate_trajectory_fn(params)

    if optimizer is not None:
        opt_state = optimizer.init(params)
    else:
        raise ValueError("optimizer is required for optimize_diffsim")

    # Compute and save initial observables (uniform weights since params == generating params)
    if save_figures and quantity_dict is not None and compute_observables_fn is not None and loss_fn is not None:
        from diffcg._core.visualization import save_iteration_figures
        trajs_init = traj_state['trajs']
        init_observables = compute_observables_fn(params, trajs_init)
        n_frames = len(trajs_init)
        uniform_weights = jnp.ones(n_frames) / n_frames
        _, init_predictions = loss_fn(init_observables, uniform_weights)
        save_iteration_figures(0, init_predictions, quantity_dict, [], output_dir)
        logger.info("Saved initial observable figures to iteration_0")

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

        # Save parameters for post-run diagnostics
        iter_dir = os.path.join(output_dir, f"iteration_{step+1}")
        os.makedirs(iter_dir, exist_ok=True)
        import pickle
        flat_params, params_treedef = jax.tree_util.tree_flatten(params)
        params_save_dict = {f"p_{i}": np.asarray(leaf) for i, leaf in enumerate(flat_params)}
        params_save_dict["treedef"] = pickle.dumps(params_treedef)
        np.savez(os.path.join(iter_dir, "params.npz"), **params_save_dict)

        # Save checkpoint (opt_state + ref_energies + loss_history)
        _save_checkpoint(step + 1, params, opt_state, traj_state, loss_history, output_dir)

        # Save figures if enabled
        if save_figures and quantity_dict is not None:
            from diffcg._core.visualization import save_iteration_figures
            save_iteration_figures(step + 1, predictions, quantity_dict, loss_history, output_dir)

    return loss_history, times_per_update, predictions_history, params_set
