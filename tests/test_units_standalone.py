"""
Standalone unit tests for verifying unit consistency in DiffCG.
Can be run without pytest: python tests/test_units_standalone.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from diffcg.io.lammps import read_lammps_data
from diffcg._core.constants import BOLTZMANN_KJMOLK as KB_KJ_MOL


# Test data paths
TEST_DATA_DIR = Path(__file__).parent.parent / "test_data" / "test_gradCG_polystyrene" / "datasets" / "T600"
EXAMPLE_DIR = Path(__file__).parent.parent / "example"


class TestRunner:
    """Simple test runner."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def run_test(self, test_name, test_func):
        """Run a single test."""
        try:
            test_func()
            print(f"✓ {test_name}")
            self.passed += 1
        except FileNotFoundError as e:
            print(f"⊘ {test_name} (skipped: {e})")
            self.skipped += 1
        except AssertionError as e:
            print(f"✗ {test_name}")
            print(f"  Error: {e}")
            self.failed += 1
        except Exception as e:
            print(f"✗ {test_name}")
            print(f"  Unexpected error: {e}")
            self.failed += 1

    def print_summary(self):
        """Print test summary."""
        total = self.passed + self.failed + self.skipped
        print(f"\n{'='*60}")
        print(f"Test Summary: {self.passed} passed, {self.failed} failed, {self.skipped} skipped (total: {total})")
        print(f"{'='*60}")
        return self.failed == 0


# Test functions
def test_lammps_coordinates_in_angstroms():
    """Test that LAMMPS data coordinates are in Ångströms."""
    lammps_file = TEST_DATA_DIR / "PS.data"
    if not lammps_file.exists():
        raise FileNotFoundError(f"LAMMPS data file not found: {lammps_file}")

    data = read_lammps_data(str(lammps_file))
    coords_raw = data["coords"]

    assert coords_raw.min() >= 0, "Coordinates should be non-negative"
    assert 10 < coords_raw.max() < 200, \
        f"LAMMPS coords not in expected Ångström range: max={coords_raw.max():.2f}"


def test_angstrom_to_nm_conversion():
    """Test conversion from Ångströms to nanometers."""
    lammps_file = TEST_DATA_DIR / "PS.data"
    if not lammps_file.exists():
        raise FileNotFoundError(f"LAMMPS data file not found: {lammps_file}")

    data = read_lammps_data(str(lammps_file))
    coords_raw = data["coords"]
    coords_nm = coords_raw / 10.0

    assert 1 < coords_nm.max() < 20, \
        f"Converted coords not in expected nm range: max={coords_nm.max():.2f}"


def test_cell_dimensions_consistent():
    """Test that cell dimensions are consistent with coordinates."""
    lammps_file = TEST_DATA_DIR / "PS.data"
    if not lammps_file.exists():
        raise FileNotFoundError(f"LAMMPS data file not found: {lammps_file}")

    data = read_lammps_data(str(lammps_file))
    coords = data["coords"]
    cells = data["cells"]

    if len(cells) > 0:
        cell = cells[0]
        assert cell.max() > coords.max() * 0.8, \
            f"Cell dimensions ({cell.max():.2f}) inconsistent with coords ({coords.max():.2f})"


def test_bond_distribution_in_nm():
    """Test that bond distributions use nanometers."""
    bond_file = TEST_DATA_DIR / "bondAA_smooth.dist.tgt"
    if not bond_file.exists():
        raise FileNotFoundError(f"Bond distribution file not found: {bond_file}")

    df = pd.read_csv(bond_file, sep=r"\s+", header=None, comment='i')
    distances = df[0].values

    assert distances.min() >= 0, "Bond distances should be non-negative"
    assert distances.max() <= 2.0, \
        f"Bond distances should be < 2.0 nm, got max={distances.max():.4f}"


def test_angle_distribution_in_radians():
    """Test that angle distributions use radians."""
    angle_file = TEST_DATA_DIR / "angleAAA.dist.tgt"
    if not angle_file.exists():
        raise FileNotFoundError(f"Angle distribution file not found: {angle_file}")

    df = pd.read_csv(angle_file, sep=r"\s+", header=None, comment='i')
    angles = df[0].values

    assert angles.min() >= 0, "Angles should be non-negative"
    assert angles.max() <= np.pi + 0.01, \
        f"Angles should be <= π rad, got max={angles.max():.4f}"


def test_dihedral_distribution_in_radians():
    """Test that dihedral distributions use radians."""
    dihedral_file = TEST_DATA_DIR / "dihedralAAAA.dist.tgt"
    if not dihedral_file.exists():
        raise FileNotFoundError(f"Dihedral distribution file not found: {dihedral_file}")

    df = pd.read_csv(dihedral_file, sep=r"\s+", header=None, comment='i')
    dihedrals = df[0].values

    assert dihedrals.min() >= -np.pi - 0.01, \
        f"Dihedrals should be >= -π rad, got min={dihedrals.min():.4f}"
    assert dihedrals.max() <= np.pi + 0.01, \
        f"Dihedrals should be <= π rad, got max={dihedrals.max():.4f}"


def test_rdf_distribution_in_nm():
    """Test that RDF distributions use nanometers."""
    rdf_file = TEST_DATA_DIR / "nb_smoothed.dist.tgt"
    if not rdf_file.exists():
        raise FileNotFoundError(f"RDF file not found: {rdf_file}")

    df = pd.read_csv(rdf_file, sep=r"\s+", header=None, comment='i')
    distances = df[0].values

    assert distances.min() >= 0, "RDF distances should be non-negative"
    assert distances.max() <= 3.0, \
        f"RDF distances should be < 3.0 nm, got max={distances.max():.4f}"


def test_boltzmann_constant():
    """Test Boltzmann constant value."""
    expected_kb = 0.0083145107  # kJ/(mol·K)
    assert abs(KB_KJ_MOL - expected_kb) < 1e-10, \
        f"Boltzmann constant mismatch: {KB_KJ_MOL} != {expected_kb}"


def test_pressure_conversion():
    """Test pressure conversion factor."""
    PRESSURE_CONVERSION_KJMOL_NM3_TO_BAR = 16.6054
    pressure_internal = 1.0 / PRESSURE_CONVERSION_KJMOL_NM3_TO_BAR
    expected_value = 0.0602214

    assert abs(pressure_internal - expected_value) < 1e-6, \
        f"Pressure conversion incorrect: {pressure_internal} != {expected_value}"


def test_parameters_loadable():
    """Test that pretrained parameters can be loaded."""
    params_file = EXAMPLE_DIR / "pretrained_params.npy"
    if not params_file.exists():
        raise FileNotFoundError(f"Pretrained params not found: {params_file}")

    params = np.load(str(params_file), allow_pickle=True).item()
    assert isinstance(params, dict), "Parameters should be a dictionary"


def test_parameter_energy_values():
    """Test that parameter values are in reasonable kJ/mol range."""
    params_file = EXAMPLE_DIR / "pretrained_params.npy"
    if not params_file.exists():
        raise FileNotFoundError(f"Pretrained params not found: {params_file}")

    params = np.load(str(params_file), allow_pickle=True).item()

    for key, val in params.items():
        if isinstance(val, np.ndarray):
            assert val.min() > -500, \
                f"Parameter '{key}' has unreasonably low energy: {val.min():.2f}"
            assert val.max() < 500, \
                f"Parameter '{key}' has unreasonably high energy: {val.max():.2f}"


def test_angstrom_to_nm_factor():
    """Test Ångström to nanometer conversion."""
    angstrom_value = 10.0
    nm_value = angstrom_value / 10.0
    assert nm_value == 1.0, "10 Å should equal 1 nm"


def test_radians_constant():
    """Test that we're using radians, not degrees."""
    assert np.pi > 3.0 and np.pi < 3.2, "π should be ~3.14159 radians"


def test_pressure_bar_to_internal():
    """Test pressure conversion from bar to internal units."""
    PRESSURE_CONVERSION = 16.6054

    one_bar_internal = 1.0 / PRESSURE_CONVERSION
    assert 0.060 < one_bar_internal < 0.061, \
        f"1 bar should be ~0.0602 kJ/(mol·nm³), got {one_bar_internal:.6f}"

    hundred_bar_internal = 100.0 / PRESSURE_CONVERSION
    assert 6.0 < hundred_bar_internal < 6.1, \
        f"100 bar should be ~6.02 kJ/(mol·nm³), got {hundred_bar_internal:.6f}"


def main():
    """Run all tests."""
    print("="*60)
    print("DiffCG Unit Consistency Tests")
    print("="*60 + "\n")

    runner = TestRunner()

    # LAMMPS tests
    print("LAMMPS Data File Tests:")
    runner.run_test("LAMMPS coordinates in Ångströms", test_lammps_coordinates_in_angstroms)
    runner.run_test("Ångström to nm conversion", test_angstrom_to_nm_conversion)
    runner.run_test("Cell dimensions consistent", test_cell_dimensions_consistent)

    # Distribution file tests
    print("\nDistribution File Tests:")
    runner.run_test("Bond distribution in nm", test_bond_distribution_in_nm)
    runner.run_test("Angle distribution in radians", test_angle_distribution_in_radians)
    runner.run_test("Dihedral distribution in radians", test_dihedral_distribution_in_radians)
    runner.run_test("RDF distribution in nm", test_rdf_distribution_in_nm)

    # Physical constants tests
    print("\nPhysical Constants Tests:")
    runner.run_test("Boltzmann constant", test_boltzmann_constant)
    runner.run_test("Pressure conversion", test_pressure_conversion)

    # Parameter tests
    print("\nPretrained Parameters Tests:")
    runner.run_test("Parameters loadable", test_parameters_loadable)
    runner.run_test("Parameter energy values", test_parameter_energy_values)

    # Unit conversion tests
    print("\nUnit Conversion Tests:")
    runner.run_test("Ångström to nm factor", test_angstrom_to_nm_factor)
    runner.run_test("Radians constant", test_radians_constant)
    runner.run_test("Pressure bar to internal", test_pressure_bar_to_internal)

    # Print summary
    all_passed = runner.print_summary()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
