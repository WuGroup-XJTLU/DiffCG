# SPDX-License-Identifier: MIT
# Copyright (c) 2025 WuResearchGroup

"""Property test: jax.checkpoint on the per-frame scan body preserves gradient values.

This invariant is what diffsim.py relies on to reduce peak memory in the gradient
step. The test uses a body function that mirrors the diffsim shape (per-frame
positions -> energy + observable, MLP-like trainable params, weighted reweighting
loss) so it exercises the exact same gradient pattern as wrapped_loss and
wrapped_total_loss_fn."""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import value_and_grad
import pytest


def _build_loss_fn(use_checkpoint):
    """Return a loss_fn(params) -> (loss, predictions) modelled on diffsim's wrapped_loss."""

    n_frames = 8
    n_atoms = 5
    rng = jax.random.PRNGKey(0)
    all_R = jax.random.uniform(rng, (n_frames, n_atoms, 3), dtype=jnp.float64)
    ref_energies = jnp.zeros(n_frames, dtype=jnp.float64)
    kBT = 2.5

    def loss_fn(params):
        W1, b1, W2, b2 = params

        def body_fn(carry, R_i):
            # Mimic energy_fn: 2-layer MLP on flattened positions
            x = R_i.reshape(-1)
            h = jnp.tanh(x @ W1 + b1)
            e_i = jnp.sum(h @ W2 + b2)
            # Mimic an observable: mean pairwise distance
            diffs = R_i[:, None, :] - R_i[None, :, :]
            obs_i = jnp.mean(jnp.linalg.norm(diffs + 1e-12, axis=-1))
            return carry, (e_i, obs_i)

        if use_checkpoint:
            body_to_scan = jax.checkpoint(body_fn)
        else:
            body_to_scan = body_fn

        _, (energies_new, obs_per_frame) = jax.lax.scan(body_to_scan, None, all_R)

        log_weights = -(1.0 / kBT) * (energies_new - ref_energies)
        log_weights = log_weights - jnp.max(log_weights)
        prob_ratios = jnp.exp(log_weights)
        weights = prob_ratios / jnp.sum(prob_ratios)

        weighted_obs = jnp.sum(weights * obs_per_frame)
        target = 1.0
        loss = (weighted_obs - target) ** 2
        return loss, weighted_obs

    return loss_fn


def _init_params(seed=42):
    rng = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(rng, 4)
    W1 = jax.random.normal(k1, (15, 16), dtype=jnp.float64) * 0.1
    b1 = jax.random.normal(k2, (16,), dtype=jnp.float64) * 0.1
    W2 = jax.random.normal(k3, (16, 1), dtype=jnp.float64) * 0.1
    b2 = jax.random.normal(k4, (1,), dtype=jnp.float64) * 0.1
    return (W1, b1, W2, b2)


class TestCheckpointGradientParity:
    def test_loss_value_matches(self):
        """Forward loss values must match exactly between checkpoint and non-checkpoint paths."""
        params = _init_params()
        loss_baseline = _build_loss_fn(use_checkpoint=False)
        loss_ckpt = _build_loss_fn(use_checkpoint=True)

        (l_b, _) = loss_baseline(params)
        (l_c, _) = loss_ckpt(params)

        assert jnp.allclose(l_b, l_c, rtol=1e-12, atol=0.0), \
            f"Forward loss diverged: baseline={l_b}, checkpoint={l_c}"

    def test_gradient_matches(self):
        """Gradients must match within float64 reduction-order noise (rtol=1e-10)."""
        params = _init_params()
        loss_baseline = _build_loss_fn(use_checkpoint=False)
        loss_ckpt = _build_loss_fn(use_checkpoint=True)

        (_, _), grad_b = value_and_grad(loss_baseline, has_aux=True)(params)
        (_, _), grad_c = value_and_grad(loss_ckpt, has_aux=True)(params)

        for i, (gb, gc) in enumerate(zip(grad_b, grad_c)):
            assert jnp.allclose(gb, gc, rtol=1e-10, atol=1e-12), \
                f"Grad component {i} diverged: max abs diff = {jnp.max(jnp.abs(gb - gc))}"

    def test_aux_predictions_match(self):
        """has_aux predictions must match between paths."""
        params = _init_params()
        loss_baseline = _build_loss_fn(use_checkpoint=False)
        loss_ckpt = _build_loss_fn(use_checkpoint=True)

        (_, aux_b), _ = value_and_grad(loss_baseline, has_aux=True)(params)
        (_, aux_c), _ = value_and_grad(loss_ckpt, has_aux=True)(params)

        assert jnp.allclose(aux_b, aux_c, rtol=1e-12, atol=0.0)
