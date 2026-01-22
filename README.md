# DiffCG

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**JAX-powered differentiable coarse-graining for molecular simulation**

## Introduction

DiffCG provides modular building blocks for differentiable coarse-graining workflows. It enables end-to-end gradient-based optimization of coarse-grained force fields by combining automatic differentiation with molecular dynamics simulation. The package supports both top-down and bottom-up parameterization approaches.

## Features

- **Differentiable Simulation (DiffSim)** - Backpropagate through MD trajectories for direct parameter optimization
- **Iterative Boltzmann Inversion (IBI)** - Classic structure-based coarse-graining
- **Relative Entropy Optimization** - Information-theoretic force field refinement
- **JAX-MD Integration** - Fully differentiable MD with JIT compilation and GPU acceleration
- **Structural Observables** - RDF, bond/angle/dihedral distributions with automatic differentiation
- **LAMMPS & ASE Compatibility** - Read/write standard simulation formats

## Installation

```bash
conda create -n diffcg python=3.8 && conda activate diffcg
pip install -e .
```

For GPU support, install JAX with CUDA following the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

## Quick Example

```python
import jax.numpy as jnp
from diffcg import energy
from diffcg.md.jaxmd_sampler import JAXMDSampler

# Define a tabulated pair potential
r_cut, r_onset = 2.0, 1.5
x_vals = jnp.linspace(0.4, r_cut, 80)
y_vals = jnp.zeros(80)  # learnable parameters

pair_energy = energy.TabulatedPairEnergy(x_vals, y_vals, r_onset, r_cut)

def total_energy(system, neighbors, **kwargs):
    return pair_energy.get_energy_fn()(system, neighbors)

# Run differentiable MD (fully differentiable via JAX-MD)
sampler = JAXMDSampler(
    energy_fn=total_energy,
    Z=jnp.ones(100, dtype=int),  # atom types
    cell=jnp.eye(3) * 5.0,       # box size in nm
    cutoff=r_cut,
    temperature=300.0,
    timestep=2.0,  # fs
)
result = sampler.run(R=initial_positions, steps=1000)
```

## Documentation

- See [`example/`](example/) for complete workflows including DiffSim optimization and multi-objective learning
- Reference papers:
  - [Learning pair potentials using differentiable simulations](https://pubs.aip.org/aip/jcp/article/158/4/044113/2876571)
  - [Structural Coarse-Graining via Multiobjective Optimization](https://pubs.acs.org/doi/10.1021/acs.jctc.3c01348)

## Contributing

Contributions welcome. Please open issues/PRs with minimal reproducible examples.

## License

MIT
