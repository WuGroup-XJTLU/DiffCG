## DiffCG: a JAX-based modular package for differentiable coarse-graining

DiffCG provides building blocks for differentiable coarse-graining workflows in molecular simulation. It supports both top-down and bottom-up approaches and leverages JAX for automatic differentiation and JIT compilation.

- Implemented methods: Iterative Boltzmann Inversion (IBI), Differentiable Simulation (DiffSim), Relative Entropy (RE) 


### Features

- **Differentiable energies** (JAX): tabulated pair/bond/angle/dihedral potentials, harmonic terms, smooth cutoffs, generic repulsions
- **Neighbor lists and PBC**: efficient neighbor list updates with periodic boundary conditions
- **ASE integration**: custom calculators and an MD convenience wrapper to run NVE/NVT/NPT
- **Observables**: RDF, inter-RDF (with polymer-exclusion masks), bond/angle/dihedral distributions (BDF/ADF/DDF)
- **I/O**: LAMMPS data/dump readers and ASE trajectory reader utilities
- **Learning utilities**: reweighting-based estimators for top-down learning workflows


### Installation

This repository is a monorepo. To use the `diffcg` package directly from source, either:

1) Add the repo to your Python path (recommended during development):

```bash
export PYTHONPATH=/home/zhenghaowu/development/diffCG:$PYTHONPATH
```

2) Or create a virtual environment and install dependencies manually:

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install jax jaxlib ase optax numpy scipy pandas matplotlib chemfiles
```

Note: JAX wheels are platform/accelerator specific. See the official JAX installation guide for your platform.


### Quickstart

#### 1) Define an energy and run MD with ASE

```python
import jax.numpy as jnp
import numpy as np
from ase import Atoms, units

from diffcg import energy
from diffcg.md.calculator import CustomCalculator
from diffcg.md.sample import MolecularDynamics

# Tabulated pair potential with smooth truncation
r_cut = 2.0
r_onset = 1.5
x_vals = jnp.linspace(0.4, r_cut, 80)
y_vals = jnp.zeros_like(x_vals)  # start from zero potential

pair_energy_fn = energy.TabulatedPairEnergy(x_vals, y_vals, r_onset, r_cut).get_energy_fn()

def total_energy(system, neighbors, **kwargs):
    return pair_energy_fn(system, neighbors)

calc = CustomCalculator(total_energy, cutoff=r_cut)

atoms = Atoms('Ar10', positions=np.random.rand(10, 3), cell=[3.0, 3.0, 3.0], pbc=True)
md = MolecularDynamics(
    atoms,
    custom_calculator=calc,
    ensemble="nvt",
    thermostat="berendsen",
    temperature=300,
    timestep=2.0 * units.fs,
    trajectory="out.traj",
    logfile="out.log",
    loginterval=50,
)
md.run(1000)
```

#### 2) Compute structural observables (RDF/BDF/ADF/DDF)

```python
import jax.numpy as jnp
from jax.tree_util import tree_map

from diffcg.io.ase_trj import read_ase_trj
from diffcg.observable.analyze import analyze
from diffcg.observable.structure import (
    rdf_discretization,
    InterRDFParams,
    initialize_inter_radial_distribution_fun,
)

# Read trajectory and batch systems
systems = read_ase_trj("out.traj")
batched = tree_map(lambda *xs: jnp.stack(xs), *systems)

# Inter-RDF (no exclusions here -> all ones mask)
centers, boundaries, sigma = rdf_discretization(RDF_cut=2.0)
num_atoms = batched.R.shape[1]
exclude_mask = jnp.ones((num_atoms, num_atoms))
inter_params = InterRDFParams(None, centers, boundaries, sigma, exclude_mask)
rdf_fn = initialize_inter_radial_distribution_fun(inter_params)

rdf_analyzer = analyze(rdf_fn)
rdf_series = rdf_analyzer.analyze(batched)  # shape: (n_frames, nbins)
```

#### 3) Read LAMMPS data/dump

```python
from diffcg.io.lammps import read_lammps_data, read_lammps_dump

data = read_lammps_data("PS.data")
traj = read_lammps_dump("dump.lammpstrj")
```

#### 4) Example top-down learning with reweighting

See `test_learning.py` for a full example of:

- assembling tabulated and harmonic terms into a total energy,
- running short MD to generate trajectories,
- computing observables (RDF/BDF/ADF/DDF), and
- optimizing spline parameters via a reweighting estimator.


### Main API surface

- Energies (`diffcg/energy.py`)
  - Tabulated pair/bond/angle/dihedral: `TabulatedPairEnergy`, `TabulatedBondEnergy`, `TabulatedAngleEnergy`, `TabulatedDihedralEnergy`
  - Harmonic terms: `HarmonicBondEnergy`, `HarmonicAngleEnergy`, `HarmonicDihedralEnergy`
  - Utilities: `tabulated`, `generic_repulsion`, `multiplicative_isotropic_cutoff`

- MD and ASE integration (`diffcg/md/`)
  - Calculators: `CustomCalculator`, `CustomEnergyCalculator`
  - MD wrapper: `MolecularDynamics`

- Observables (`diffcg/observable/`)
  - RDF: `rdf_discretization`, `initialize_radial_distribution_fun`, `initialize_inter_radial_distribution_fun`, `InterRDFParams`
  - BDF/ADF/DDF: `bdf_discretization`, `ADFParams`, `DDFParams`, `initialize_bond_distribution_fun`, `initialize_angle_distribution_fun`, `initialize_dihedral_distribution_fun`
  - Analysis helper: `analyze`

- I/O (`diffcg/io/`)
  - LAMMPS: `read_lammps_data`, `read_lammps_dump`, `read_lammps_traj`
  - ASE: `read_ase_trj`


### Notes and tips

- Units in examples commonly follow ASE conventions (e.g., positions in Å when using ASE `Atoms`), but many examples in this repo use nm and kJ/mol; ensure consistency when mixing data sources.
- Smooth cutoffs (`multiplicative_isotropic_cutoff`) ensure continuity at the cutoff; prefer them for stable MD.
- When building larger models, sum multiple energy terms inside a single callable and pass it to `CustomCalculator`.


### Contributing

Contributions are welcome. Please open issues/PRs with minimal reproducible examples when reporting bugs or proposing features.

### Acknowlodge
Difftre: https://github.com/tummfm/difftre
glp: https://github.com/sirmarcel/glp

