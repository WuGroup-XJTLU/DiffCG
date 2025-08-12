"""
Relative Entropy (RE) training with importance reweighting (FEP).

Overview
--------
We fit CG parameters θ so that the model distribution

    p_θ(X) = exp(−β U_θ(X)) / Z_θ

matches the mapped atomistic (reference) distribution p_ref(X) in the sense of
minimizing the KL divergence D_KL(p_ref || p_θ). Up to a constant independent of θ,
this is equivalent to minimizing the convex objective

    L(θ) = β E_{p_ref}[U_θ(X)] + log Z_θ.

Directly differentiating log Z_θ requires expectations over p_θ. To avoid running a
new CG simulation at every update, we employ free-energy perturbation (Zwanzig): for
a base parameter set θ₀ with samples X ∼ p_{θ₀},

    log Z_θ − log Z_{θ₀} = log E_{p_{θ₀}}[exp(−β (U_θ(X) − U_{θ₀}(X)))].

Substituting gives the reweighting-based surrogate (up to constants):

    L_RE(θ) = β E_{p_ref}[U_θ(X)] − log E_{p_{θ₀}}[exp(−β (U_θ(X) − U_{θ₀}(X)))].

We estimate the two terms as follows:
  - Reference term: run the energy U_θ on the (fixed) reference frames and take the sample mean.
  - Normalizer term: run U_θ and U_{θ₀} on a short CG trajectory generated at θ₀,
    then compute a numerically stable log-mean-exp of −β ΔU where ΔU = U_θ − U_{θ₀}.

Gradient (conceptual)
---------------------
The population RE gradient is

    ∇_θ L(θ) = β ( E_{p_ref}[∂U_θ/∂θ] − E_{p_θ}[∂U_θ/∂θ] ).

In practice we evaluate this by autodiff of the scalar loss L_RE(θ). The second term is
implicitly estimated via the importance weights

    w(X) ∝ exp(−β (U_θ(X) − U_{θ₀}(X))),    X ∼ p_{θ₀},

which reweights θ₀-sampled frames towards p_θ. Effective sample size

    n_eff = exp( −∑_j w_j log w_j )

is tracked to detect weight degeneracy; when overlap between p_{θ₀} and p_θ is poor we
refresh the CG trajectory at the current parameters.

Step 0 behavior
---------------
At the first iteration we initialize the CG trajectory at θ₀ = θ. Then ΔU = 0 on CG frames,
the FEP term becomes log mean exp(0) = 0, weights are uniform, and the loss reduces to
β times the mean energy on the reference frames (up to constants).

Notes
-----
- The estimator is consistent but can be biased/variance-prone with poor overlap; the
  n_eff-triggered refresh mitigates this by periodically re-sampling at the current θ.
- Energies are evaluated via the provided JAX-differentiable energy builder, so gradients
  flow through spline parameters or other differentiable terms.
"""

from typing import Dict

import jax
import jax.numpy as jnp
import optax
import time
import os

from ase.io import read

from diffcg.util.logger import get_logger
from diffcg.md.calculator import CustomCalculator, CustomEnergyCalculator


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
                timestep=sampler_params['timestep'] * 0.1,
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
                timestep=sampler_params['timestep'] * 0.1,
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
                timestep=sampler_params['timestep'] * 0.1,
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
            os.system(
                f"cp {sampler_params['trajectory']}{step-1}.traj {sampler_params['trajectory']}{step}.traj"
            )
            os.system(
                f"cp {sampler_params['logfile']}{step-1}.log {sampler_params['logfile']}{step}.log"
            )
            trajs_cg = read(f"{sampler_params['trajectory']}{step}.traj", index=':')

        # Base energies on CG frames under old params (reference for FEP)
        U_base_cg = rerun_energy(old_params, trajs_cg)

        def wrapped_loss(p):
            # Reference contribution: β E_ref[U_θ]
            U_ref_theta = rerun_energy(p, ref_trajs)
            term_ref = beta * jnp.mean(U_ref_theta)

            # Normalizer via FEP on CG proposal frames from base (old_params)
            U_theta_cg = rerun_energy(p, trajs_cg)
            delta = U_theta_cg - U_base_cg
            term_norm = _logmeanexp(-beta * delta)

            loss_val = term_ref - term_norm

            # Diagnostics: effective sample size from FEP weights
            w_unnorm = jnp.exp(-beta * delta - jnp.max(-beta * delta))
            w = w_unnorm / jnp.sum(w_unnorm)
            n_eff = jnp.exp(-jnp.sum(jnp.where(w > 1e-12, w * jnp.log(w), 0.0)))

            metrics = {
                'RE_loss': loss_val,
                'term_ref': term_ref,
                'term_norm': term_norm,
                'n_eff': n_eff,
                'num_cg_frames': jnp.array(len(trajs_cg), dtype=jnp.float32),
                'num_ref_frames': jnp.array(len(ref_trajs), dtype=jnp.float32),
            }
            return loss_val, metrics

        v_and_g = jax.value_and_grad(wrapped_loss, has_aux=True)
        (loss_val, metrics), grad = v_and_g(params)

        scaled_grad, opt_state = optimizer.update(grad, opt_state, params)
        new_params = optax.apply_updates(params, scaled_grad)

        # Decide whether to recompute trajectory for next step using n_eff heuristic (keep convention in reweighting.py)
        n_eff_val = float(metrics['n_eff'])
        recompute = (n_eff_val > reweight_ratio * len(trajs_cg)) and (step > 0)
        if recompute:
            logger.info(
                f"Recomputing trajectory {step} because n_eff = {n_eff_val} > {reweight_ratio * len(trajs_cg)}"
            )
            new_atoms = trajs_cg[-1]
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

        return new_params, params, opt_state, loss_val, metrics

    return update_fn


def optimize_relative_entropy(update_fn, params, total_iterations):
    """Convenience optimizer loop for the RE update function.

    Returns:
        loss_history, times_per_update, metrics_history, params_set
    """
    new_params = params
    opt_state = None
    loss_history = []
    times_per_update = []
    metrics_history = []
    params_set = []

    for step in range(total_iterations):
        start_time = time.time()
        new_params, params, opt_state, loss_val, metrics = update_fn(
            step, new_params, params, opt_state
        )
        step_time = time.time() - start_time
        logger.info(
            f"Step {step} in {step_time:0.2f} sec. RE Loss = {loss_val} | n_eff = {metrics.get('n_eff', jnp.nan)}\n\n"
        )
        if jnp.isnan(loss_val):
            logger.error(
                'Loss is NaN. This was likely caused by divergence of the optimization or a bad model setup causing a NaN trajectory.'
            )
        loss_history.append(loss_val)
        times_per_update.append(step_time)
        metrics_history.append(metrics)
        params_set.append(params)

    return loss_history, times_per_update, metrics_history, params_set


