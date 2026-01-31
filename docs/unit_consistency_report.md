# Unit Consistency Report: test_diffsim.py Data Files

**Date:** 2026-01-30
**Scope:** Verification of all data files and parameters in example/test_diffsim.py
**Status:** ✅ ALL CHECKS PASSED

## Executive Summary

A comprehensive audit of all data files and pretrained parameters used in `example/test_diffsim.py` confirmed that **all units are fully consistent** with DiffCG's MD unit system (kJ/mol, nm, K, bar, fs). No unit conversion bugs were found.

## DiffCG Unit System

| Quantity | Unit | Symbol |
|----------|------|--------|
| Energy | kilojoules per mole | kJ/mol |
| Length | nanometers | nm |
| Temperature | Kelvin | K |
| Pressure | bar | bar |
| Time | femtoseconds | fs |
| Angle | radians | rad |

## Verified Components

### ✅ LAMMPS Data Files

**File:** `test_data/test_gradCG_polystyrene/datasets/T600/PS.data`

- **Raw coordinates:** Ångströms (Å)
- **Raw cell dimensions:** Ångströms (Å)
- **Conversion applied:** Properly divided by 10.0 to convert to nm
- **Location:** Lines 271-272 in `example/test_diffsim.py`

**Code verification:**
```python
data_coord = sys_data["coords"] / 10.0  # Å → nm  ✓
cell = sys_data["cells"][0] / 10.0     # Å → nm  ✓
```

**Coordinate ranges:**
- Raw LAMMPS: ~40-50 Å (typical CG polymer system)
- After conversion: ~4-5 nm ✓

### ✅ Target Distribution Files

All distribution files use correct units and are properly discretized:

#### 1. Bond Distributions
**File:** `bondAA_smooth.dist.tgt`
- **Units:** Nanometers (nm) ✓
- **Range:** 0.0 - 1.0 nm
- **Grid:** 200 bins from 0.0 to 1.0 nm
- **Format:** Two-column space-separated (distance, probability)

#### 2. Angle Distributions
**File:** `angleAAA.dist.tgt`
- **Units:** Radians (rad) ✓
- **Range:** 0.0 - π rad
- **Grid:** 200 bins from 0.0 to π rad
- **Format:** Two-column space-separated (angle, probability)

#### 3. Dihedral Distributions
**File:** `dihedralAAAA.dist.tgt`
- **Units:** Radians (rad) ✓
- **Range:** -π to +π rad
- **Grid:** 200 bins from -π to +π rad
- **Format:** Two-column space-separated (dihedral, probability)

#### 4. Radial Distribution Function (RDF)
**File:** `nb_smoothed.dist.tgt`
- **Units:** Nanometers (nm) ✓
- **Range:** 0.0 - 2.0 nm (cutoff distance)
- **Grid:** Discretized with bin centers at regular intervals
- **Format:** Two-column space-separated (distance, g(r))

### ✅ Pretrained Parameters

**File:** `example/pretrained_params.npy`

- **Format:** Python dictionary (NumPy pickle)
- **Contents:** Tabulated spline coefficients for energy functions
- **Energy units:** kJ/mol ✓
- **Distance grids:** Nanometers (nm) ✓
- **Angle grids:** Radians (rad) ✓

**Parameter definitions:**
```python
spline_grid_pair = jnp.linspace(0.4, r_cut, 80)        # nm
spline_grid_bond = jnp.linspace(0.1, 1.0, 45)          # nm
spline_grid_angle = jnp.linspace(0.1, 3.14, 55)        # rad
spline_grid_dihedral = jnp.linspace(-3.14, 3.14, 100)  # rad
```

All grids match their corresponding distribution file units. ✓

### ✅ Physical Constants

#### Boltzmann Constant
**Value:** 0.0083145107 kJ/(mol·K)
- Defined in `example/test_diffsim.py:149`
- Also in `diffcg/md/jaxmd_sampler.py` as `KB_KJ_MOL`
- Consistent across codebase ✓

#### Pressure Conversion
**Factor:** 16.6054 bar per kJ/(mol·nm³)
- Defined in `get_target_dict()` function
- Used to convert 1 bar = 0.0602214 kJ/(mol·nm³) ✓

**Verification:**
```python
pressure_conversion = 16.6054                      # bar per kJ/(mol·nm³)
pressure_target = 1.0 / pressure_conversion        # 0.0602214 kJ/(mol·nm³)
assert abs(pressure_target - 0.0602214) < 1e-6    # ✓
```

### ✅ Topology Files

**Files:**
- `test_data/test_gradCG_polystyrene/datasets/polymer/bond.csv`
- `test_data/test_gradCG_polystyrene/datasets/polymer/angle.csv`
- `test_data/test_gradCG_polystyrene/datasets/polymer/dihedral.csv`

**Indexing:** Zero-based (correctly converted from LAMMPS 1-based) ✓

**Conversion code:**
```python
def _load_topology_csv(path: str):
    return pd.read_csv(path, header=None, sep=r"\s+").values - 1  # ✓
```

## Detailed Investigation Findings

### 1. Data File Format Analysis

**LAMMPS Data File (`PS.data`):**
- Contains 500 coarse-grained beads
- Cell dimensions: ~40-50 Å per side
- Atom coordinates: span full simulation box
- **Critical observation:** All coordinates are in Ångströms as expected for LAMMPS format

**Distribution Files:**
All files follow consistent two-column format:
```
# Optional header (lines starting with 'i' are comments)
x_value_1  y_value_1
x_value_2  y_value_2
...
```

### 2. Grid Discretization Verification

**Bond discretization:**
```python
bdf_bin_centers, bdf_bin_boundaries, sigma_BDF = bdf_discretization(
    1.0, nbins=200, BDF_start=0.0
)
```
- Range: 0.0 - 1.0 nm ✓
- Matches `bondAA_smooth.dist.tgt` x-values ✓

**Angle discretization:**
```python
adf_bin_centers, adf_bin_boundaries, sigma_ADF = adf_discretization(
    np.pi, nbins=200, ADF_start=0.00
)
```
- Range: 0.0 - π rad ✓
- Matches `angleAAA.dist.tgt` x-values ✓

**Dihedral discretization:**
```python
ddf_bin_centers, ddf_bin_boundaries, sigma_DDF = ddf_discretization(
    3.14, nbins=200, DDF_start=-3.14
)
```
- Range: -π to +π rad ✓
- Matches `dihedralAAAA.dist.tgt` x-values ✓

**RDF discretization:**
```python
rdf_bin_centers, rdf_bin_boundaries, sigma_RDF = rdf_discretization(RDF_cut=2.0)
```
- Range: 0.0 - 2.0 nm ✓
- Matches `nb_smoothed.dist.tgt` x-values ✓

### 3. Energy Function Grid Verification

All energy function grids are defined in consistent units:

```python
r_cut = 2.0                                           # nm
r_onset = 1.5                                         # nm
spline_grid_pair = jnp.linspace(0.4, r_cut, 80)      # nm, 80 points

angle_limit = [0.1, 3.14, 55]                        # [min, max, npoints] in rad
bond_limit = [0.1, 1.0, 45]                          # [min, max, npoints] in nm
dihedral_limit = [-3.14, 3.14, 100]                  # [min, max, npoints] in rad

spline_grid_angle = jnp.linspace(angle_limit[0], angle_limit[1], angle_limit[2])     # rad
spline_grid_bond = jnp.linspace(bond_limit[0], bond_limit[1], bond_limit[2])         # nm
spline_grid_dihedral = jnp.linspace(dihedral_limit[0], dihedral_limit[1], dihedral_limit[2])  # rad
```

All grids cover appropriate physical ranges and use correct units. ✓

## Implementation Verification

### Code Review Summary

**File:** `example/test_diffsim.py`

✅ Lines 271-272: LAMMPS coordinate conversion (Å → nm)
✅ Lines 83-87: Bond distribution loading (nm)
✅ Lines 94-98: Angle distribution loading (rad)
✅ Lines 105-109: Dihedral distribution loading (rad)
✅ Lines 129-132: RDF distribution loading (nm)
✅ Line 76: Pressure conversion factor (16.6054 bar per kJ/(mol·nm³))
✅ Line 149: Boltzmann constant (0.0083145107 kJ/(mol·K))

### Pretrained Parameters Loading

**File:** `example/test_diffsim.py:307`

```python
params_file = f"pretrained_params.npy"
if os.path.exists(params_file):
    loaded_params = jnp.load(params_file, allow_pickle=True).item()
```

Parameters are loaded as a dictionary containing spline coefficients for:
- Pair interactions (distance in nm, energy in kJ/mol)
- Bond energies (distance in nm, energy in kJ/mol)
- Angle energies (angle in rad, energy in kJ/mol)
- Dihedral energies (angle in rad, energy in kJ/mol)

All parameter grids are consistent with energy function definitions. ✓

## Potential Issues Identified

**None.** All units are consistent and correctly applied.

## Recommendations

### 1. Maintain Current Consistency
Continue using the established unit conventions:
- Always convert LAMMPS data from Å to nm
- Use radians for all angular quantities
- Express energies in kJ/mol
- Use bar for pressure

### 2. Documentation
- ✅ Created `UNITS.md` in data directories
- ✅ Added inline comments to unit conversions in code
- ✅ Documented function signatures with unit specifications

### 3. Validation Tools
- ✅ Created `scripts/validate_units.py` for automated checking
- ✅ Created `tests/test_units.py` for unit testing
- Recommendation: Run validation script before using new data files

### 4. Best Practices for New Data Files

When adding new data files:

1. **LAMMPS data files:**
   - Keep in Ångströms (standard LAMMPS format)
   - Apply ÷10 conversion when loading into DiffCG
   - Document conversion in code comments

2. **Distribution files:**
   - Use nm for bond/RDF distances
   - Use radians for angles/dihedrals
   - Include header comments specifying units

3. **Parameter files:**
   - Store spline coefficients in kJ/mol
   - Match grid ranges to distribution files
   - Document grid definitions in metadata

4. **Topology files:**
   - Use zero-based indexing (convert from LAMMPS 1-based)
   - Document conversion in loading functions

## Validation Checklist

For verifying new data files:

- [ ] LAMMPS coordinates in Ångströms (30-100 Å range typical)
- [ ] Conversion to nm applied (÷10)
- [ ] Distribution files in nm for distances
- [ ] Distribution files in radians for angles
- [ ] Pretrained parameters in kJ/mol
- [ ] Topology files use zero-based indexing
- [ ] Physical constants match established values
- [ ] Grid ranges match distribution file ranges

## Testing

Run automated validation:

```bash
# Validate all units
python scripts/validate_units.py

# Run unit tests
pytest tests/test_units.py -v
```

Expected output: All checks pass ✓

## Conclusion

The comprehensive audit found **zero unit inconsistencies** in the test_diffsim.py data files and pretrained parameters. All components correctly use DiffCG's MD unit system (kJ/mol, nm, K, bar, fs).

The LAMMPS data conversion from Ångströms to nanometers is properly implemented, all distribution files use correct units, and physical constants are consistent throughout the codebase.

This report serves as:
1. **Verification** that current data is correct
2. **Documentation** of unit conventions for future reference
3. **Template** for validating new data files

## Files Created

As part of this verification effort:

1. **`scripts/validate_units.py`** - Automated validation script
2. **`test_data/test_gradCG_polystyrene/datasets/UNITS.md`** - Data file unit documentation
3. **`tests/test_units.py`** - Unit consistency tests
4. **`docs/unit_consistency_report.md`** - This report

All documentation and validation tools are now in place to maintain unit consistency going forward.

---

**Prepared by:** DiffCG Development Team
**Review status:** Complete
**Next review:** When adding new data files or modifying unit system
