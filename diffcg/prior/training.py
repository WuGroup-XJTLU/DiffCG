from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import optax
from jax import jit, random

from diffcg.prior.targets import TargetDict
from diffcg.prior.potentials import GridConfig, make_grids, build_pair_fn, build_bond_fn, build_angle_fn, build_dihedral_fn


@dataclass
class PriorConfig:
    temperature: float
    init_scale: float = 1e-3
    learning_rate: float = 1e-1
    lr_decay_steps: int = 100_000
    lr_decay_factor: float = 0.001
    num_updates: int = 10_000
    seed: int = 3
    grid: GridConfig = GridConfig()


def kBT_from_temperature(T: float) -> float:
    from diffcg.common.constants import BOLTZMANN_KJMOLK

    return T * BOLTZMANN_KJMOLK


def build_loss_fn(targets: TargetDict, grids: Dict[str, jnp.ndarray]):
    kBT = kBT_from_temperature(config_temperature := 300.0)
    # The caller should pass in actual temperature. Keep function pure; we will close over real kBT later.

    def loss_fn(param, kBT_real: float):
        loss = 0.0

        # RDF pair term (shifted PMF target)
        pair = build_pair_fn(param[0], grids['pair'])
        energy_vals = jax.vmap(pair)(targets.rdf.rdf_bin_centers[targets.rdf.rdf_bin_centers > grids['pair'][0]])
        pmf = -kBT_real * jnp.log(targets.rdf.reference_rdf)
        pmf = pmf - pmf[-1]
        pmf = pmf[targets.rdf.rdf_bin_centers > grids['pair'][0]]
        loss += jnp.sum((energy_vals - pmf) ** 2)

        # Bond
        bond = build_bond_fn(param[1], grids['bond'])
        energy_vals = jax.vmap(bond)(grids['bond'])
        pmf = -kBT_real * jnp.log(targets.bdf.reference_bdf)
        pmf = pmf - jnp.min(pmf[jnp.isfinite(pmf)])
        pmf_interp = jnp.interp(grids['bond'], targets.bdf.bdf_bin_centers, pmf)
        loss += jnp.sum((energy_vals - pmf_interp) ** 2)

        # Angle
        angle = build_angle_fn(param[2], grids['angle'])
        energy_vals = jax.vmap(angle)(grids['angle'])
        pmf = -kBT_real * jnp.log(targets.adf.reference_adf)
        pmf = pmf - jnp.min(pmf[jnp.isfinite(pmf)])
        pmf_interp = jnp.interp(grids['angle'], targets.adf.adf_bin_centers, pmf)
        loss += jnp.sum((energy_vals - pmf_interp) ** 2)

        # Dihedral (custom_quantity.DDFParams uses adf-like field names)
        dihedral = build_dihedral_fn(param[3], grids['dihedral'])
        energy_vals = jax.vmap(dihedral)(grids['dihedral'])
        pmf = -kBT_real * jnp.log(targets.ddf.reference_adf)
        pmf = pmf - jnp.min(pmf[jnp.isfinite(pmf)])
        pmf_interp = jnp.interp(grids['dihedral'], targets.ddf.adf_bin_centers, pmf)
        loss += jnp.sum((energy_vals - pmf_interp) ** 2)

        return loss

    return loss_fn


def pretrain_potentials(targets: TargetDict, config: PriorConfig):
    grids = make_grids(config.grid)

    # Initialize parameters on each grid
    key = random.PRNGKey(config.seed)
    k1, k2, k3, k4 = random.split(key, 4)
    init_params = (
        config.init_scale * random.normal(k1, grids['pair'].shape),
        config.init_scale * random.normal(k2, grids['bond'].shape),
        config.init_scale * random.normal(k3, grids['angle'].shape),
        config.init_scale * random.normal(k4, grids['dihedral'].shape),
    )

    # Optimizer
    lr_schedule = optax.exponential_decay(-config.learning_rate, config.lr_decay_steps, config.lr_decay_factor)
    optimizer = optax.chain(optax.scale_by_adam(0.9, 0.99), optax.scale_by_schedule(lr_schedule))

    loss_fn = build_loss_fn(targets, grids)

    @jit
    def update(params, opt_state, kBT_real):
        loss_value, curr_grad = jax.value_and_grad(lambda p: loss_fn(p, kBT_real))(params)
        scaled_grad, opt_state = optimizer.update(curr_grad, opt_state, params)
        new_params = optax.apply_updates(params, scaled_grad)
        return new_params, opt_state, curr_grad, loss_value

    params = init_params
    opt_state = optimizer.init(params)

    kBT_real = kBT_from_temperature(config.temperature)

    for step in range(config.num_updates):
        params, opt_state, grad, loss_val = update(params, opt_state, kBT_real)
        if step % 10000 == 0:
            print(f"Step {step} loss = {loss_val}")

    params_dict = {
        'pair': params[0],
        'bond': params[1],
        'angle': params[2],
        'dihedral': params[3],
    }
    return params_dict, grids


