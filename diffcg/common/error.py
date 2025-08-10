import jax.numpy as jnp

def MSE(pred, ref):
    return jnp.mean((pred - ref)**2)

def RMSE(pred, ref):
    return jnp.sqrt(MSE(pred, ref))

def MAE(pred, ref):
    return jnp.mean(jnp.abs(pred - ref))