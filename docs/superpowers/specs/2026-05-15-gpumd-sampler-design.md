# GPUMD Sampler for DiffSim

Add GPUMD (`/home/zhenghaowu/gpumd`) as a 4th sampler backend in DiffCG's DiffSim
workflow, alongside jaxmd, lammps, and fastmd. The GPUMD sampler uses NEP (neuroevolution
potential) as the force field — both for GPU-accelerated MD sampling via the GPUMD
executable and for JAX-based reweighting/gradient computation in the DiffSim loop.

## Architecture

```
DiffSim Loop
┌─────────────────────────────────────────────────────────┐
│  ┌──────────────────┐       ┌──────────────────────┐    │
│  │ GPUMD (C++/CUDA) │       │ JAX NEP Energy Fn    │    │
│  │ - Sample traj     │       │ - Reweight traj      │    │
│  │ - dump_xyz.xyz    │◄──────│ - Compute gradients  │    │
│  │ - thermo.out      │ nep   │ - Optimize params    │    │
│  └──────────────────┘  .txt  └──────────────────────┘    │
│                         (shared parameters)              │
└─────────────────────────────────────────────────────────┘
```

GPUMD handles trajectory generation (expensive, GPU-accelerated). JAX NEP handles
reweighting and gradient computation (differentiable). Both use the same `nep.txt`
parameter file, ensuring physically consistent sampling and reweighting.

## Files

### New files

| File | Purpose | ~Lines |
|------|---------|--------|
| `diffcg/md/gpumd_sampler.py` | GPUMDSampler class | 350 |
| `diffcg/io/nep.py` | Read/write `nep.txt` format | 150 |
| `diffcg/io/gpumd_writer.py` | Write `xyz.in` for GPUMD | 80 |
| `diffcg/nep/__init__.py` | Public API exports | 20 |
| `diffcg/nep/descriptor.py` | JAX NEP descriptor computation | 250 |
| `diffcg/nep/network.py` | JAX NEP feedforward NN | 100 |
| `diffcg/nep/energy.py` | Top-level energy_fn builder | 80 |

### Modified files

| File | Change |
|------|--------|
| `diffcg/md/__init__.py` | Export `GPUMDSampler`, `create_gpumd_sampler` |
| `diffcg/md/sample.py` | Add `sampler_backend="gpumd"` branches in both factory functions |
| `diffcg/learning/diffsim.py` | Add "gpumd" restart state handler |

## GPUMDSampler

Follows the same public interface as `LAMMPSSampler` and `FastMDSampler`:

```
run(steps) -> Trajectory
get_trajectory() -> Trajectory
get_final_system() -> AtomicSystem
set_system(system) -> None
update_potentials(energy_params, energy_objects) -> None
```

### Internal flow

1. **`run(steps)`**: Write `xyz.in` (extended XYZ with atom positions and box),
   write `nep.txt` (NEP parameters), write `run.in` (GPUMD control script).
   Execute `gpumd` via subprocess. Parse `dump_xyz.xyz` for trajectory,
   `thermo.out` for thermodynamic data.

2. **Input format** (`xyz.in`): Extended XYZ with `Lattice` and `Properties`
   header line. Positions in Angstroms (converted from nm).

3. **Control file** (`run.in`): Keyword-based, similar to LAMMPS input.
   Keywords: `potential`, `velocity`, `ensemble`, `time_step`, `dump_thermo`,
   `dump_xyz`, `run`.

4. **Output parsing**: `dump_xyz.xyz` has per-frame XYZ blocks with box info
   in the comment line. `thermo.out` is columnar with temperature, potential
   energy, kinetic energy, pressure, volume, and box dimensions.

### Unit conversions

| Quantity | DiffCG internal | GPUMD internal |
|----------|----------------|----------------|
| Length   | nm             | Å (Angstrom)   |
| Energy   | kJ/mol         | eV             |
| Time     | fs             | fs             |

Conversions happen at the I/O boundary: `nm_to_angstrom` on write, `angstrom_to_nm`
on read; `ev_to_kjmol` on energy read.

## NEP Parameter I/O (`diffcg/io/nep.py`)

### nep.txt format

```
nep4 <num_types> <elem1> <elem2> ...
cutoff <rc_radial> <rc_angular> <MN_radial> <MN_angular>
n_max <n_max_radial> <n_max_angular>
basis_size <basis_size_radial> <basis_size_angular>
l_max <L_max> <has_q_222> <has_q_1111> [<has_q_112>] [<has_q_1122>]
ANN <num_neurons> 0
<float values...>     # descriptor params then NN weights
```

### Reader

`read_nep(filepath) -> dict` with keys: version, num_types, elements, rc_radial,
rc_angular, MN_radial, MN_angular, n_max_radial, n_max_angular, basis_size_radial,
basis_size_angular, L_max, has_q_222, has_q_1111, has_q_112, has_q_1122,
num_neurons, descriptor_params (jnp.array), ann_params (dict of per-type w0/b0/w1),
b1 (scalar), q_scaler (array dim).

### Writer

`write_nep(filepath, params_dict)` — writes exact format GPUMD expects.
Incremented per DiffSim iteration.

### Parameter layout (from GPUMD's `nep.cu`)

1. Descriptor params: `num_types^2 * [(n_max_radial+1)*(basis_size_radial+1) + (n_max_angular+1)*(basis_size_angular+1)]` floats
2. ANN params per type: `w0 (dim*neurons), b0 (neurons), w1 (neurons)`
3. Global `b1` (1 float)
4. `q_scaler` (dim floats)

## JAX NEP Energy Function

### Descriptor (`diffcg/nep/descriptor.py`)

For each atom i, compute descriptor qi of dimension `dim = (n_max_radial+1) + (n_max_angular+1) * num_L`.

**Radial** (n = 0, ..., n_max_radial):
```
q^n_i = sum_j sum_k c^{n,k}_{ti,tj} * T_k(rij) * fc(rij)
```
where T_k is the Chebyshev polynomial of order k, fc is the cosine cutoff function,
and the sum runs over neighbors j within the radial cutoff.

**Angular** (n = 0, ..., n_max_angular; l across num_L components):
```
q^{nl}_i = sum_{j!=k} sum_m c^{nl,m}_{ti,tj,tk} * T_m(rij) * T_m(rik) * Pl(cos_theta_ijk) * fc(rij) * fc(rik)
```
The num_L components are l = 0,1,...,L_max plus optional 3-body/4-body invariants
(has_q_222, has_q_1111, has_q_112, has_q_1122).

**Normalization**: `qi = qi / q_scaler` (element-wise division).

**Constants ported from GPUMD**: C3B (80), C4B (5), C5B (3), C4B2 (5), C5B2 (10)
arrays from `nep_utilities.cuh` → JAX constants.

### Neural Network (`diffcg/nep/network.py`)

Per atom i of type t:
```
hidden = tanh(w0_t @ qi + b0_t)
Ei = w1_t @ hidden + b1
```

NEP4: single hidden layer. NEP5: adds per-type output bias before final b1.

Total energy: `E = sum_i Ei` in eV, converted to kJ/mol.

### Energy function builder (`diffcg/nep/energy.py`)

```python
def build_nep_energy_fn(nep_params, num_types, elements, n_max_radial, n_max_angular,
                        basis_size_radial, basis_size_angular, L_max,
                        has_q_222, has_q_1111, has_q_112, has_q_1122,
                        num_neurons, version=4):
    """Returns energy_fn(system, neighbors) -> energy (kJ/mol)."""
```

Uses `jaxmd_neighbor_list` for neighbor computation. Entire forward pass is JAX
differentiable, enabling gradient-based optimization of all NEP parameters via optax.

## Integration Points

### sample.py

Add `sampler_backend="gpumd"` to `create_equilibration_run()` and
`create_production_run()`. Reads `gpumd_config` (reuses the `lammps_config` dict
to avoid API breakage) with keys: `gpumd_exe`, `nep_params`, `topology`, `work_dir`.

### diffsim.py

Add "gpumd" to the restart state switch:
```python
if _backend == 'gpumd':
    restart_state = {'system': md_equ.get_final_system()}
```

### State dict for users

```python
state = {
    "init_system": init_system,
    "r_cut": R_CUT,
    "quantity_dict": quantity_dict,
    "sampler_params": sampler_params,
    "sim_time_scheme": sim_time_scheme,
    "output_dir": OUTPUT_DIR,
    "sampler_backend": "gpumd",
    "lammps_config": {
        "energy_objects": nep_energy_objects,
        "nep_params": initial_nep_params,
        "topology": {},
        "gpumd_exe": "path/to/gpumd",
    },
}
```

## Testing

### test_gpumd_sampler.py
- GPUMDSampler init parameter validation
- run.in template generation correctness
- xyz.in format verification (positions, box, atom types)
- Trajectory parsing from dump_xyz.xyz files
- thermo.out parsing

### test_nep_io.py
- Parse example `nep.txt`, verify all fields match expected values
- Round-trip: read → write → read, verify numerical equality
- Parameter count matches formula: `num_types^2 * [(n_max_radial+1)*(basis_size_radial+1) + (n_max_angular+1)*(basis_size_angular+1)] + (dim+2)*neurons*num_types + 1 + dim`

### test_nep_energy.py
- Verify JAX NEP energy matches GPUMD output for same `nep.txt` + positions (tolerance: 1e-5 eV/atom)
- Verify JAX gradient exists and is non-zero
- Test with 2-atom trivial system

### test_gpumd_diffsim.py
- End-to-end DiffSim with `sampler_backend="gpumd"` and minimal 1-type NEP
- Trajectory generation → energy recomputation → gradient step → new params → write `nep.txt`

## Units Summary

| Quantity  | DiffCG    | GPUMD     | Conversion factor        |
|-----------|-----------|-----------|--------------------------|
| Length    | nm        | Å         | 1 nm = 10 Å              |
| Energy    | kJ/mol    | eV        | 1 eV = 96.485 kJ/mol    |
| Time      | fs        | fs        | 1:1                      |
| Temp      | K         | K         | 1:1                      |
