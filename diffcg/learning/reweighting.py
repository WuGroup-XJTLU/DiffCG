# SPDX-License-Identifier: MIT
# Copyright (c) 2025 WuResearchGroup

from collections import namedtuple
from typing import Optional, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
import time
import sys
import os

from diffcg.util.logger import get_logger
from diffcg.util.math import high_precision_sum

logger = get_logger(__name__)

# Numerical stability small constant
EPS: float = 1e-10


@jax.jit
def _estimate_effective_samples_jit(weights: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled effective sample size estimation."""
    safe_w = jnp.where(weights > EPS, weights, EPS)
    exponent = -jnp.sum(safe_w * jnp.log(safe_w))
    return jnp.exp(exponent)


@jax.jit
def _compute_weights_jit(
    energies_new: jnp.ndarray,
    ref_energies: jnp.ndarray,
    base_energies: jnp.ndarray,
    pv: jnp.ndarray,
    beta: float,
) -> jnp.ndarray:
    """JIT-compiled weight computation."""
    unew = energies_new + base_energies + pv
    uref = ref_energies + pv
    log_weights = -(unew - uref) * beta
    log_weights = log_weights - log_weights.max()
    prob_ratios = jnp.exp(log_weights)
    weights = prob_ratios / high_precision_sum(prob_ratios)
    return weights


class ReweightEstimator:
    """Estimator for reweighting factors and effective sample size.

    Computes normalized importance weights between a reference distribution
    and a new distribution based on potential energy differences (and optional PV work).
    """

    def __init__(
        self,
        ref_energies: jnp.ndarray,
        base_energies: Optional[jnp.ndarray] = None,
        volume: Optional[float] = None,
        kBT: float = 1.0,
        pressure: float = 1.0,
    ) -> None:
        self.beta: float = 1.0 / kBT
        self.ref_energies: jnp.ndarray = jnp.array(ref_energies)
        if base_energies is None:
            self.base_energies: jnp.ndarray = jnp.zeros(ref_energies.shape)
        else:
            self.base_energies = jnp.array(base_energies)
        if volume is not None:
            # Convert PV to kJ/mol (0.06023 factor per original code comment)
            self.pv: jnp.ndarray = jnp.array(volume * pressure * 0.06023)
        else:
            self.pv = jnp.zeros(ref_energies.shape)

    def estimate_effective_samples(self, weights: jnp.ndarray) -> jnp.ndarray:
        """Estimate the effective number of samples given normalized weights.

        Uses exp(-sum w_i log w_i) with clipping to avoid log(0).
        """
        return _estimate_effective_samples_jit(weights)

    def estimate_weight(self, energies_new: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return normalized weights and n_eff for new energies.

        Args:
            energies_new: Energies evaluated at the new parameters for the same frames
                          used to compute `ref_energies`.

        Returns:
            weights: Normalized importance weights per frame
            n_eff: Effective number of samples implied by the weights
        """
        weights = _compute_weights_jit(
            energies_new, self.ref_energies, self.base_energies, self.pv, self.beta
        )
        n_eff = _estimate_effective_samples_jit(weights)
        return weights, n_eff

    # Backward-compatibility alias used elsewhere in the codebase
    def compute_weights(self, energies_new: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Alias for estimate_weight for compatibility."""
        return self.estimate_weight(energies_new)

