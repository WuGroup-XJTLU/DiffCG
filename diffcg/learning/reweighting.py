from collections import namedtuple
from typing import Optional, Dict
from diffcg.system import atoms_to_system
import jax.numpy as jnp
from diffcg.util.math import high_precision_sum
from diffcg.common.error import MSE
from diffcg.util.logger import get_logger
import jax
import optax
import time
from ase import units
from diffcg.md.calculator import CustomCalculator
from diffcg.md.sample import MolecularDynamics
from ase.io import read
from diffcg.md.calculator import CustomEnergyCalculator
import sys
import os
from jax.flatten_util import ravel_pytree

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

class ReweightEstimator():
    def __init__(
        self,
        ref_energies,
        base_energies=None,
        volume=None,
        kBT=1.0,
        pressure=1.0,
    ):
        self.beta = 1.0 / kBT
        self.ref_energies = jnp.array(ref_energies)
        if base_energies is None:
            self.base_energies = jnp.zeros(ref_energies.shape)
        else:
            self.base_energies = jnp.array(base_energies)
        if volume is not None:
            self.pv = jnp.array(volume * pressure * 0.06023)
        else:
            self.pv = jnp.zeros(ref_energies.shape)

    def estimate_effective_samples(self,weights):
        weights = jnp.where(weights > 1.e-10, weights, 1.e-10)  # mask to avoid NaN from log(0) if a few weights are 0.
        exponent = - jnp.sum(weights * jnp.log(weights))
        return jnp.exp(exponent)

    def estimate_weight(self, uinit):
        unew = uinit + self.base_energies + self.pv
        uref = self.ref_energies + self.pv
        exponent = (unew - uref) * self.beta
        exponent = exponent - exponent.max()
        prob_ratios = jnp.exp(-exponent)
        weight = prob_ratios / high_precision_sum(prob_ratios)
        n_eff = self.estimate_effective_samples(weight)
        return weight, n_eff


class DiffSim():
    def __init__(self,
                 reweight_ratio,
                 init_atoms,
                 r_cut,
                 quantity_dict,
                 calculate_observables_fn,
                 build_energy_fn_with_params_fn,
                 optimizer,
                 sim_time_scheme,
                 sampler_params,
                 Boltzmann_constant = 0.0083145107  # in kJ / mol K
                 ):
        # Deprecated OO API: keep for backward compatibility and delegate to functional API
        import warnings
        warnings.warn(
            "DiffSim class is deprecated. Use init_diffsim(...) and optimize_diffsim(...) instead.",
            category=UserWarning,
            stacklevel=2,
        )

        # Store only what is needed for wrapper methods
        self.quantity_dict = quantity_dict
        self.calculate_observables_fn = calculate_observables_fn
        self.build_energy_fn_with_params_fn = build_energy_fn_with_params_fn
        self.optimizer = optimizer
        self.sampler_params = sampler_params
        self.init_atoms = init_atoms
        self.r_cut = r_cut
        self.Boltzmann_constant = Boltzmann_constant
        self.reweight_ratio = reweight_ratio
        self.sim_time_scheme = sim_time_scheme

        # Build functional single-state update function for delegation
        state = {
            'init_atoms': init_atoms,
            'r_cut': r_cut,
            'quantity_dict': quantity_dict,
            'calculate_observables_fn': calculate_observables_fn,
            'sampler_params': sampler_params,
            'sim_time_scheme': sim_time_scheme,
        }
        self._update_fn = init_diffsim(
            reweight_ratio=reweight_ratio,
            state=state,
            build_energy_fn_with_params_fn=build_energy_fn_with_params_fn,
            optimizer=optimizer,
            Boltzmann_constant=Boltzmann_constant,
        )

    def obtain_sample_md(self,step,calculator):

        sample_md_equ = MolecularDynamics(self.init_atoms, 
                                    custom_calculator=calculator, 
                                    ensemble=self.sampler_params['ensemble'], 
                                    thermostat=self.sampler_params['thermostat'], 
                                    temperature=self.sampler_params['temperature'], 
                                    starting_temperature=self.sampler_params['starting_temperature'],
                                    timestep=self.sampler_params['timestep'], 
                                    trajectory=None,
                                    logfile=None,
                                    loginterval=None)

        sample_md_prod = MolecularDynamics(self.init_atoms, 
                                    custom_calculator=calculator, 
                                    ensemble=self.sampler_params['ensemble'], 
                                    thermostat=self.sampler_params['thermostat'], 
                                    temperature=self.sampler_params['temperature'], 
                                    starting_temperature=self.sampler_params['starting_temperature'],
                                    timestep=self.sampler_params['timestep'], 
                                    trajectory=f"{self.sampler_params['trajectory']}{step}.traj",
                                    logfile=f"{self.sampler_params['logfile']}{step}.log",
                                    loginterval=self.sampler_params['loginterval'])
        return sample_md_equ, sample_md_prod

    def obtain_update_fn(self):
        # Backward-compatible method: return the functional update fn
        return self._update_fn

    def obtain_rerun_energy_fn(self,max_num_atoms):
        def rerun_energy(params, traj):
            results = []
            energy_fn = self.build_energy_fn_with_params_fn(params,max_num_atoms=max_num_atoms)
            calculator = CustomEnergyCalculator(energy_fn,cutoff=self.r_cut)
            for atoms in traj:
                calculator.calculate(atoms)
                results.append(calculator.results)
            return jnp.stack(results)
        return rerun_energy
        
    def optimize(self, params, total_iteration):
        # Delegate to functional optimizer for single-state API
        update_fn = self.obtain_update_fn()
        return optimize_diffsim(update_fn, params, total_iteration)
    


coweighting_stats = namedtuple("coweighting_stats", 
                                ("current_iter", "num_losses", "mean_decay", "running_mean_L", "running_mean_l", "running_std_l", "running_S_l", "alphas"))

def init_coweighting_stats(num_losses):
    current_iter=-1
    mean_decay=False
    running_mean_L=jnp.zeros(num_losses)
    running_mean_l=jnp.zeros(num_losses)
    running_std_l=jnp.zeros(num_losses)
    running_S_l=jnp.zeros(num_losses)
    alphas=jnp.ones(num_losses)
    
    return coweighting_stats(current_iter,num_losses,mean_decay,running_mean_L,running_mean_l,running_std_l,running_S_l,alphas)

def coweightingloss_init():

    def coweightingloss(loss_dict,coweighting_stats):
        L, unravel = ravel_pytree(loss_dict)
        
        current_iter,num_losses,mean_decay,running_mean_L,running_mean_l,running_std_l,running_S_l,alphas = coweighting_stats

        # Increase the current iteration parameter.
        current_iter += 1

        L0=jnp.where(current_iter == 0, L, running_mean_L)

        l = L/L0 #L / L0  

        alphas=jnp.where(current_iter <=1, alphas / num_losses, (running_std_l / running_mean_L)/jnp.sum((running_std_l / running_mean_L)))

        mean_param=jnp.where(current_iter==0,0.0,(1. - 1 / (current_iter + 1)))

        x_l = l
        new_mean_l = mean_param * running_mean_L + (1 - mean_param) * x_l
        running_S_l += (x_l - running_mean_L) * (x_l - new_mean_l)
        running_mean_L = new_mean_l

        running_variance_l = running_S_l / (current_iter + 1)
        running_std_l = jnp.sqrt(running_variance_l + 1e-8)

        x_L = L
        running_mean_L = mean_param * running_mean_L + (1 - mean_param) * x_L

        weighted_losses = jnp.sum(L*alphas)

        return weighted_losses,coweighting_stats._replace(
                                                        current_iter=current_iter,
                                                        num_losses=num_losses,
                                                        mean_decay=mean_decay,
                                                        running_mean_L=running_mean_L,
                                                        running_mean_l=running_mean_l,
                                                        running_std_l=running_std_l,
                                                        running_S_l=running_S_l,
                                                        alphas=alphas)

    return coweightingloss

def init_multistate_diffsim(
    *,
    reweight_ratio,
    states,
    build_energy_fn_with_params_fn,
    optimizer,
    Boltzmann_constant: float = 0.0083145107,
    state_weights: Optional[dict] = None,
):
    """
    Initialize a multistate DiffSim update function.

    All states share the same energy function (same parameters), but may have distinct
    quantities/targets, observable calculators, sampler settings and time schemes.

    Args:
        reweight_ratio: Threshold for n_eff to decide whether to recompute a state's trajectory.
        states: Dict mapping state_id -> dict with keys:
            - 'init_atoms': ASE Atoms
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
        update_fn(step, params, old_params, opt_state) ->
            (new_params, params, opt_state, total_loss, per_state_losses, predictions_by_state)
    """

    # Pre-build per-state helpers and metadata
    state_ids = list(states.keys())
    if state_weights is None:
        state_weights = {sid: 1.0 for sid in state_ids}

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
        sid: states[sid]['init_atoms'].get_global_number_of_atoms() for sid in state_ids
    }

    def build_rerun_energy_fn_for_state(state_id):
        r_cut = states[state_id].get('r_cut', 1.0)

        def rerun_energy(params, traj):
            results = []
            energy_fn = build_energy_fn_with_params_fn(params, max_num_atoms=max_atoms_by_state[state_id])
            calculator = CustomEnergyCalculator(energy_fn, cutoff=r_cut)
            for atoms in traj:
                calculator.calculate(atoms)
                results.append(calculator.results)
            return jnp.stack(results)

        return rerun_energy

    rerun_energy_by_state = {sid: build_rerun_energy_fn_for_state(sid) for sid in state_ids}

    def create_md_for_state(state_id, step, sample_energy_fn):
        """Create and return MD object(s) for the given state and step.

        Returns either a tuple (md_equ, md_prod) for two-phase runs or a single md for one-phase.
        """
        state = states[state_id]
        sampler_params = state['sampler_params']
        r_cut = state.get('r_cut', 1.0)
        init_atoms = state['init_atoms']

        calculator = CustomCalculator(sample_energy_fn, cutoff=r_cut)

        scheme = state['sim_time_scheme']
        if 'equilibration_steps' in scheme and 'production_steps' in scheme:
            md_equ = MolecularDynamics(
                init_atoms,
                custom_calculator=calculator,
                ensemble=sampler_params['ensemble'],
                thermostat=sampler_params['thermostat'],
                temperature=sampler_params['temperature'],
                starting_temperature=sampler_params['starting_temperature'],
                timestep=sampler_params['timestep'],
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
            md = MolecularDynamics(
                init_atoms,
                custom_calculator=calculator,
                ensemble=sampler_params['ensemble'],
                thermostat=sampler_params['thermostat'],
                temperature=sampler_params['temperature'],
                starting_temperature=sampler_params['starting_temperature'],
                timestep=sampler_params['timestep'],
                trajectory=f"{sampler_params['trajectory']}{step}.traj",
                logfile=f"{sampler_params['logfile']}{step}.log",
                loginterval=sampler_params['loginterval'],
            )
            return md

    def update_fn(step, params, old_params, opt_state):
        if opt_state is None:
            opt_state = optimizer.init(params)
        # For each state, ensure trajectory for this step exists or reuse previous
        per_state_context = {}

        for sid in state_ids:
            state = states[sid]
            scheme = state['sim_time_scheme']
            sampler_params = state['sampler_params']

            sample_energy_fn = build_energy_fn_with_params_fn(
                params, max_num_atoms=max_atoms_by_state[sid]
            )

            # Prepare trajectories
            if step == 0:
                md_objs = create_md_for_state(sid, step, sample_energy_fn)
                if isinstance(md_objs, tuple):
                    md_equ, md_prod = md_objs
                    md_equ.run(scheme['equilibration_steps'])
                    md_prod.set_atoms(md_equ.atoms)
                    md_prod.run(scheme['production_steps'])
                else:
                    md = md_objs
                    md.run(scheme['total_simulation_steps'])
                trajs = read(f"{sampler_params['trajectory']}{step}.traj", index=':')
            else:
                logger.info(f"[state={sid}] Reusing trajectory {step-1}")
                os.system(
                    f"cp {sampler_params['trajectory']}{step-1}.traj {sampler_params['trajectory']}{step}.traj"
                )
                os.system(
                    f"cp {sampler_params['logfile']}{step-1}.log {sampler_params['logfile']}{step}.log"
                )
                trajs = read(f"{sampler_params['trajectory']}{step}.traj", index=':')

            # Build estimator from reference energies with old_params
            ref_energies = rerun_energy_by_state[sid](old_params, trajs)
            estimator = ReweightEstimator(
                ref_energies,
                kBT=state['sampler_params']['temperature'] * Boltzmann_constant,
                base_energies=None,
                volume=None,
            )

            # Check n_eff with current params to decide recompute
            curr_energies = rerun_energy_by_state[sid](params, trajs)
            _, n_eff = estimator.estimate_weight(curr_energies)
            recompute = (n_eff > reweight_ratio * len(trajs)) and (step > 0)

            if recompute:
                logger.info(
                    f"[state={sid}] Recomputing trajectory {step} because n_eff = {n_eff} > {reweight_ratio * len(trajs)}"
                )
                new_atoms = trajs[-1]
                os.system(f"rm {sampler_params['logfile']}{step}.log")
                md_objs = create_md_for_state(sid, step, sample_energy_fn)
                if isinstance(md_objs, tuple):
                    md_equ, md_prod = md_objs
                    md_equ.set_atoms(new_atoms)
                    md_equ.run(scheme['equilibration_steps'])
                    md_prod.set_atoms(md_equ.atoms)
                    md_prod.run(scheme['production_steps'])
                else:
                    md = md_objs
                    md.set_atoms(new_atoms)
                    md.run(scheme['total_simulation_steps'])
                trajs = read(f"{sampler_params['trajectory']}{step}.traj", index=':')
                # recompute ref energies and estimator
                ref_energies = rerun_energy_by_state[sid](old_params, trajs)
                estimator = ReweightEstimator(
                    ref_energies,
                    kBT=state['sampler_params']['temperature'] * Boltzmann_constant,
                    base_energies=None,
                    volume=None,
                )

            per_state_context[sid] = {
                'trajs': trajs,
                'estimator': estimator,
                'loss_fn': loss_fn_by_state[sid],
                'weight': state_weights.get(sid, 1.0),
            }

        # To supply state-specific observables inside the loss, rebuild a fast lookup
        observables_by_state = {
            sid: states[sid]['calculate_observables_fn'](
                f"{states[sid]['sampler_params']['trajectory']}{step}.traj"
            )
            for sid in state_ids
        }

        def wrapped_total_loss_fn(p):
            total = 0.0
            per_state_losses = {}
            predictions_by_state = {}
            for sid in state_ids:
                ctx = per_state_context[sid]
                energies = rerun_energy_by_state[sid](p, ctx['trajs'])
                weights, _ = ctx['estimator'].estimate_weight(energies)
                loss_val, predictions = ctx['loss_fn'](observables_by_state[sid], weights)
                predictions_by_state[sid] = predictions
                per_state_losses[sid] = loss_val
                total += state_weights.get(sid, 1.0) * loss_val
            return total, (per_state_losses, predictions_by_state)

        v_and_g = jax.value_and_grad(wrapped_total_loss_fn, has_aux=True)
        (total_loss, (per_state_losses, predictions_by_state)), grad = v_and_g(params)
        scaled_grad, opt_state = optimizer.update(grad, opt_state, params)
        new_params = optax.apply_updates(params, scaled_grad)
        return new_params, params, opt_state, total_loss, per_state_losses, predictions_by_state

    return update_fn


def optimize_multistate_diffsim(update_fn, params, total_iterations):
    """
    Convenience optimizer loop for the multistate DiffSim update function.

    Returns:
        loss_history, times_per_update, predictions_history, params_set, per_state_loss_history
    """
    new_params = params
    opt_state = None  # will be provided by closure via update_fn
    loss_history = []
    times_per_update = []
    predictions_history = []
    params_set = []
    per_state_loss_history = []

    # Initialize optimizer state using a dummy call (step=0) if needed
    for step in range(total_iterations):
        start_time = time.time()
        new_params, params, opt_state, total_loss, per_state_losses, predictions = update_fn(
            step, new_params, params, opt_state
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
        params_set.append(params)
        per_state_loss_history.append(per_state_losses)

    return loss_history, times_per_update, predictions_history, params_set, per_state_loss_history


def init_diffsim(
    *,
    reweight_ratio,
    state: Dict,
    build_energy_fn_with_params_fn,
    optimizer,
    Boltzmann_constant: float = 0.0083145107,
):
    """
    Initialize a single-state DiffSim update function (functional API).

    Args:
        reweight_ratio: Threshold for n_eff to decide whether to recompute the trajectory.
        state: Dict with keys:
            - 'init_atoms': ASE Atoms
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
        update_fn(step, params, old_params, opt_state) ->
            (new_params, params, opt_state, loss, predictions)
    """

    # Prepare reusable elements
    loss_fn = init_independent_mse_loss_fn(state['quantity_dict'])
    max_num_atoms = state['init_atoms'].get_global_number_of_atoms()

    def create_md(step, sample_energy_fn):
        sampler_params = state['sampler_params']
        r_cut = state.get('r_cut', 1.0)
        init_atoms = state['init_atoms']
        calculator = CustomCalculator(sample_energy_fn, cutoff=r_cut)
        scheme = state['sim_time_scheme']
        if 'equilibration_steps' in scheme and 'production_steps' in scheme:
            md_equ = MolecularDynamics(
                init_atoms,
                custom_calculator=calculator,
                ensemble=sampler_params['ensemble'],
                thermostat=sampler_params['thermostat'],
                temperature=sampler_params['temperature'],
                starting_temperature=sampler_params['starting_temperature'],
                timestep=sampler_params['timestep'],
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
                timestep=sampler_params['timestep'],
                trajectory=f"{sampler_params['trajectory']}{step}.traj",
                logfile=f"{sampler_params['logfile']}{step}.log",
                loginterval=sampler_params['loginterval'],
            )
            return md_equ, md_prod
        else:
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

        # Prepare trajectories: run or reuse
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
            trajs = read(f"{sampler_params['trajectory']}{step}.traj", index=':')
        else:
            logger.info(f"Reusing trajectory {step-1}")
            os.system(
                f"cp {sampler_params['trajectory']}{step-1}.traj {sampler_params['trajectory']}{step}.traj"
            )
            os.system(
                f"cp {sampler_params['logfile']}{step-1}.log {sampler_params['logfile']}{step}.log"
            )
            trajs = read(f"{sampler_params['trajectory']}{step}.traj", index=':')

        # Build estimator from reference energies with old_params
        ref_energies = rerun_energy(old_params, trajs)
        estimator = ReweightEstimator(
            ref_energies,
            kBT=sampler_params['temperature'] * Boltzmann_constant,
            base_energies=None,
            volume=None,
        )

        # Check n_eff with current params to decide recompute
        curr_energies = rerun_energy(params, trajs)
        _, n_eff = estimator.estimate_weight(curr_energies)
        recompute = (n_eff > reweight_ratio * len(trajs)) and (step > 0)

        if recompute:
            logger.info(
                f"Recomputing trajectory {step} because n_eff = {n_eff} > {reweight_ratio * len(trajs)}"
            )
            new_atoms = trajs[-1]
            os.system(f"rm {sampler_params['logfile']}{step}.log")
            md_objs = create_md(step, sample_energy_fn)
            if isinstance(md_objs, tuple):
                md_equ, md_prod = md_objs
                md_equ.set_atoms(new_atoms)
                md_equ.run(scheme['equilibration_steps'])
                md_prod.set_atoms(md_equ.atoms)
                md_prod.run(scheme['production_steps'])
            else:
                md = md_objs
                md.set_atoms(new_atoms)
                md.run(scheme['total_simulation_steps'])
            trajs = read(f"{sampler_params['trajectory']}{step}.traj", index=':')
            # recompute ref energies and estimator
            ref_energies = rerun_energy(old_params, trajs)
            estimator = ReweightEstimator(
                ref_energies,
                kBT=sampler_params['temperature'] * Boltzmann_constant,
                base_energies=None,
                volume=None,
            )

        # Compute observables once per step
        observables = state['calculate_observables_fn'](
            f"{sampler_params['trajectory']}{step}.traj"
        )

        def wrapped_loss(p):
            energies = rerun_energy(p, trajs)
            weights, _ = estimator.estimate_weight(energies)
            return loss_fn(observables, weights)

        v_and_g = jax.value_and_grad(wrapped_loss, has_aux=True)
        (loss_val, predictions), grad = v_and_g(params)
        scaled_grad, opt_state = optimizer.update(grad, opt_state, params)
        new_params = optax.apply_updates(params, scaled_grad)
        return new_params, params, opt_state, loss_val, predictions

    return update_fn


def optimize_diffsim(update_fn, params, total_iterations):
    """
    Convenience optimizer loop for single-state DiffSim update function.

    Returns:
        loss_history, times_per_update, predictions_history, params_set
    """
    new_params = params
    opt_state = None
    loss_history = []
    times_per_update = []
    predictions_history = []
    params_set = []

    for step in range(total_iterations):
        start_time = time.time()
        new_params, params, opt_state, loss_val, predictions = update_fn(
            step, new_params, params, opt_state
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
        params_set.append(params)

    return loss_history, times_per_update, predictions_history, params_set