## DiffCG: a JAX-based modular package for differentiable coarse-graining

DiffCG provides building blocks for differentiable coarse-graining workflows in molecular simulation. It supports both top-down and bottom-up approaches and leverages JAX for automatic differentiation and JIT compilation.

Implemented methods: 
- Iterative Boltzmann Inversion (IBI) 
- Differentiable Simulation (DiffSim) 
- Relative Entropy (RE) 

### Installation

Create a conda virtual environment (recommended) and install from source:

```bash
conda create -n diffcg python=3.8 && conda activate diffcg
python -m pip install -e .
```

JAX note: JAX/JAXLIB wheels are accelerator- and platform-specific. If you
need GPU/TPU support, please follow the official JAX guide for your platform
([JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html)).


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

# Note: analyze class now requires init_atoms parameter
rdf_analyzer = analyze(rdf_fn, atoms)
rdf_series = rdf_analyzer.analyze(batched)  # shape: (n_frames, nbins)
```

#### 3) Read LAMMPS data/dump

```python
from diffcg.io.lammps import read_lammps_data, read_lammps_dump

data = read_lammps_data("PS.data")
traj = read_lammps_dump("dump.lammpstrj")
```

#### 4) Example learning workflows

See the `example/` directory for complete examples:

- `test_diffsim.py`: Differentiable simulation with single-objective optimization
- `test_diffsim_multiobj.py`: Multi-objective optimization with co-weighting
- `examples/test_gradCG_polystyrene/`: Comprehensive polystyrene coarse-graining workflow

These examples demonstrate:
- Assembling tabulated and harmonic terms into a total energy
- Running MD simulations to generate trajectories
- Computing observables (RDF/BDF/ADF/DDF)
- Optimizing parameters via differentiable simulation and reweighting


### Core concepts

- **System representation**: The package uses a `System` dataclass to represent molecular configurations with positions (`R`), atomic numbers (`Z`), and simulation cell information.
- **JAX integration**: All computations leverage JAX for automatic differentiation, JIT compilation, and vectorization.
- **Modular design**: Energy terms, observables, and learning algorithms are designed as composable building blocks.
- **ASE compatibility**: Seamless integration with the Atomic Simulation Environment (ASE) for structure manipulation and I/O.

### Main API surface

- **Energies** (`diffcg/energy.py`)
  - Tabulated potentials: `TabulatedPairEnergy`, `TabulatedBondEnergy`, `TabulatedAngleEnergy`, `TabulatedDihedralEnergy`
  - Harmonic terms: `HarmonicBondEnergy`, `HarmonicAngleEnergy`, `HarmonicDihedralEnergy`
  - Generic repulsion: `GenericRepulsionEnergy`
  - Utilities: `tabulated`, `multiplicative_isotropic_cutoff`

- **MD and ASE integration** (`diffcg/md/`)
  - Calculators: `CustomCalculator`, `CustomEnergyCalculator`
  - MD wrapper: `MolecularDynamics`
  - Trajectory observer: `TrajectoryObserver`

- **Learning methods** (`diffcg/learning/`)
  - IBI: `IterativeBoltzmannInversion`, `IBIConfig`, `IBITargets`
  - DiffSim: `init_diffsim`, `optimize_diffsim`, `init_independent_mse_loss_fn` (class `DiffSim` deprecated)
  - Relative entropy: `RelativeEntropy`
  - Multi-objective: `init_coweighting_stats`, `coweightingloss_init`
  - Reweighting: `ReweightEstimator`

- **Observables** (`diffcg/observable/`)
  - RDF: `rdf_discretization`, `initialize_radial_distribution_fun`, `initialize_inter_radial_distribution_fun`, `InterRDFParams`
  - BDF/ADF/DDF: `bdf_discretization`, `adf_discretization`, `ddf_discretization`, `BDFParams`, `ADFParams`, `DDFParams`
  - Initialization functions: `initialize_bond_distribution_fun`, `initialize_angle_distribution_fun`, `initialize_dihedral_distribution_fun`
  - Analysis helper: `analyze`
  - Thermodynamic observables: (`diffcg/observable/thermo.py`)

- **I/O** (`diffcg/io/`)
  - LAMMPS: `read_lammps_data`, `read_lammps_dump`, `read_lammps_traj`
  - ASE: `read_ase_trj`
  - Conversion utilities: (`diffcg/io/convert.py`)

- **Common utilities** (`diffcg/common/`)
  - Geometry: `distance`, `angle`, `dihedral`, `vectorized_angle_fn`, `vectorized_dihedral_fn`
  - Periodic boundary conditions: `displacement`
  - Neighbor lists: `neighbor_list`
  - Constants and error handling

- **Utilities** (`diffcg/util/`)
  - Logging: `configure_logging`, `get_logger`
  - Math utilities: `high_precision_sum`
  - Boltzmann inversion: `boltzmann_inversion`
  - Custom interpolation: `MonotonicInterpolate`


### Notes and tips

- **Units**: Examples commonly follow ASE conventions (e.g., positions in Å when using ASE `Atoms`), but many examples in this repo use nm and kJ/mol; ensure consistency when mixing data sources.
- **Smooth cutoffs**: Use `multiplicative_isotropic_cutoff` to ensure continuity at the cutoff; prefer them for stable MD.
- **Energy combination**: When building larger models, sum multiple energy terms inside a single callable and pass it to `CustomCalculator`.
- **Logging**: Use `from diffcg import configure_logging; configure_logging(level="DEBUG")` for detailed output during development.
- **Deprecated APIs**: The `DiffSim` class is deprecated; use the functional API with `init_diffsim` and `optimize_diffsim` instead.


### Contributing

Contributions are welcome. Please open issues/PRs with minimal reproducible examples when reporting bugs or proposing features.

### Reference
- [Learning pair potentials using differentiable simulations](https://pubs.aip.org/aip/jcp/article/158/4/044113/2876571)
- [Structural Coarse-Graining via Multiobjective Optimization with Differentiable Simulation](https://pubs.acs.org/doi/10.1021/acs.jctc.3c01348)
- [Learning neural network potentials from experimental data via Differentiable Trajectory Reweighting](https://www.nature.com/articles/s41467-021-27241-4)
- [Differentiable molecular simulation can learn all the parameters in a coarse-grained force field for proteins](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0256990)

### Acknowledgments

This package builds upon ideas and code from:
- [difftre](https://github.com/tummfm/difftre): Differentiable trajectory reweighting
- [glp](https://github.com/sirmarcel/glp): Gaussian linear potentials

