import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_map
from diffcg.util.logger import get_logger
from diffcg.system import atoms_to_system

logger = get_logger(__name__)

class analyze():
    def __init__(self,compute_fn,init_atoms):
        self.compute_fn = compute_fn
        self.system = atoms_to_system(init_atoms)
        

    def analyze(self,batched_systems):
        B = batched_systems.R.shape[0]
        proto = self.compute_fn(self.system)
        def body(carry, i):
            sys_i = tree_map(lambda x: x[i], batched_systems)
            val = self.compute_fn(sys_i)
            return carry + val, val

        _, observables = lax.scan(body, jnp.zeros_like(proto), jnp.arange(B))
        logger.debug("Analyzed %s frames", B)
        return observables
    
