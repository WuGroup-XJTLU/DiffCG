# SPDX-License-Identifier: MIT
# Copyright (c) 2025 WuResearchGroup

import jax
import jax.numpy as jnp
from jax import lax, vmap
from jax.tree_util import tree_map
from diffcg._core.logger import get_logger
from diffcg.system import AtomicSystem, from_ase_atoms

logger = get_logger(__name__)

class TrajectoryAnalyzer():
    def __init__(self, compute_fn, init_system):
        self.compute_fn = compute_fn
        # Accept both AtomicSystem and ASE Atoms (backward compat)
        if isinstance(init_system, AtomicSystem):
            self.system = init_system
        else:
            self.system = from_ase_atoms(init_system)

        # Pre-compile JIT'd versions (proto computation stays outside JIT)
        self._jit_scan = jax.jit(self._scan_impl)
        self._jit_vmap = jax.jit(self._vmap_impl)

    def _scan_impl(self, proto, batched_systems):
        B = batched_systems.R.shape[0]
        def body(carry, i):
            sys_i = tree_map(lambda x: x[i], batched_systems)
            val = self.compute_fn(sys_i)
            return carry + val, val
        _, observables = lax.scan(body, jnp.zeros_like(proto), jnp.arange(B))
        return observables

    def _vmap_impl(self, batched_systems):
        return vmap(self.compute_fn)(batched_systems)

    def analyze(self, batched_systems):
        """Scan-based analysis (constant memory). Best for large-memory-per-frame ops like RDF."""
        proto = self.compute_fn(self.system)
        observables = self._jit_scan(proto, batched_systems)
        logger.debug("Analyzed %s frames (scan)", batched_systems.R.shape[0])
        return observables

    def analyze_vmap(self, batched_systems):
        """vmap-based analysis (parallel across frames). Best for small-memory ops like BDF/ADF/DDF."""
        observables = self._jit_vmap(batched_systems)
        logger.debug("Analyzed %s frames (vmap)", batched_systems.R.shape[0])
        return observables
