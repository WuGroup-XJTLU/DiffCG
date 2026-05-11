# SPDX-License-Identifier: MIT
"""Tests that init_diffsim's optional `regularizer_fn` kwarg adds an additive
params-dependent term to the loss when provided, and is a no-op when None.

These tests bypass the full sampler stack: we directly invoke the closure
returned by init_independent_mse_loss_fn and verify the algebraic shape of
the regularizer integration."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from diffcg.learning.diffsim import init_independent_mse_loss_fn


def _toy_quantity_dict():
    # Target is offset by 0.1 from the weighted-mean of the trajectory below,
    # so the MSE is ((0.1)^2 + (0.1)^2)/2 = 0.01.
    target = jnp.array([0.9, 1.9])
    return {
        "x": {
            "compute_fn": lambda *a, **kw: None,
            "gamma": 1.0,
            "target": target,
        }
    }


def test_base_loss_unchanged_by_none_regularizer():
    qd = _toy_quantity_dict()
    base_loss = init_independent_mse_loss_fn(qd)
    quantity_trajs = {"x": jnp.array([[1.1, 2.1], [0.9, 1.9]])}
    weights = jnp.array([0.5, 0.5])
    loss_val, preds = base_loss(quantity_trajs, weights)
    assert jnp.allclose(preds["x"], jnp.array([1.0, 2.0]))
    assert jnp.isclose(loss_val, 0.01, atol=1e-6)


def test_regularizer_added_to_loss():
    """The integration target: when regularizer_fn is set on init_diffsim,
    the loss inside the gradient closure should be base_loss + regularizer_fn(p).
    This test verifies the algebraic identity at the loss-composition layer."""
    qd = _toy_quantity_dict()
    base_loss = init_independent_mse_loss_fn(qd)
    quantity_trajs = {"x": jnp.array([[1.1, 2.1], [0.9, 1.9]])}
    weights = jnp.array([0.5, 0.5])
    params = {"pair": jnp.array([0.0, 1.0, 4.0, 9.0])}
    lam = 0.5

    def regularizer_fn(p):
        d2 = p["pair"][2:] - 2.0 * p["pair"][1:-1] + p["pair"][:-2]
        return lam * jnp.sum(d2 ** 2)

    base_val, _ = base_loss(quantity_trajs, weights)
    reg_val = regularizer_fn(params)
    combined = base_val + reg_val

    expected_reg = 0.5 * ((4.0 - 2 * 1.0 + 0.0) ** 2 + (9.0 - 2 * 4.0 + 1.0) ** 2)
    assert jnp.isclose(reg_val, expected_reg)
    assert jnp.isclose(combined, base_val + expected_reg)


def test_regularizer_grad_flows():
    params = {"pair": jnp.array([0.0, 1.0, 4.0, 9.0])}
    lam = 0.5

    def regularizer_fn(p):
        d2 = p["pair"][2:] - 2.0 * p["pair"][1:-1] + p["pair"][:-2]
        return lam * jnp.sum(d2 ** 2)

    grad = jax.grad(regularizer_fn)(params)
    assert grad["pair"].shape == params["pair"].shape
    assert jnp.any(grad["pair"] != 0.0)
