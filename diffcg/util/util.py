# SPDX-License-Identifier: MIT
# Copyright (c) 2025 WuResearchGroup

import jax.numpy as jnp

def cast(x):
    """Cast number literal to jnp.ndarray.

    This avoids jit recompiles, as native python types
    are "weak" types in jax. This makes everything explicit.
    In high-precision situations, jax type promotion should™
    do the right thing.
    """

    if type(x) == int:
        return jnp.array(x, dtype=jnp.int32)
    elif type(x) == float:
        return jnp.array(x, dtype=jnp.float32)
    else:
        raise ValueError(f"cannot cast {x} of as type {type(x)} is unknown to me")
