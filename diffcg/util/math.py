import jax.numpy as jnp

def high_precision_sum(X,
                       axis=None,
                       keepdims=False):
  """Sums over axes at 64-bit precision then casts back to original dtype."""
  if jnp.issubdtype(X.dtype, jnp.integer):
    dtyp = jnp.int64
  elif jnp.issubdtype(X.dtype, jnp.complexfloating):
    dtyp = jnp.complex128
  else:
    dtyp = jnp.float64

  return jnp.array(
      jnp.sum(X, axis=axis, dtype=dtyp, keepdims=keepdims), dtype=X.dtype)