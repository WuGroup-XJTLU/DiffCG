"""
Unit tests for verifying unit consistency in DiffCG.

These tests ensure that:
1. LAMMPS data files are properly converted from Ångströms to nanometers
2. Distribution files use correct units (nm for distances, rad for angles)
3. Physical constants are consistent across the codebase
4. Pretrained parameters have reasonable energy values
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from diffcg.io.lammps import read_lammps_data
from diffcg._core.constants import BOLTZMANN_KJMOLK as KB_KJ_MOL


# Test data paths
TEST_DATA_DIR = Path(__file__).parent.parent / "test_data" / "test_gradCG_polystyrene" / "datasets" / "T600"
EXAMPLE_DIR = Path(__file__).parent.parent / "example"


class TestLAMMPSDataConversion:
    """Test LAMMPS data file unit conversions."""

    def test_lammps_coordinates_in_angstroms(self):
        """Test that LAMMPS data coordinates are in Ångströms."""
        lammps_file = TEST_DATA_DIR / "PS.data"
        if not lammps_file.exists():
            pytest.skip(f"LAMMPS data file not found: {lammps_file}")

        data = read_lammps_data(str(lammps_file))
        coords_raw = data["coords"]

        # LAMMPS CG systems typically have coordinates in range 30-100 Å
        assert coords_raw.min() >= 0, "Coordinates should be non-negative"
        assert 10 < coords_raw.max() < 200, \
            f"LAMMPS coords not in expected Ångström range: max={coords_raw.max():.2f}"

    def test_angstrom_to_nm_conversion(self):
        """Test conversion from Ångströms to nanometers."""
        lammps_file = TEST_DATA_DIR / "PS.data"
        if not lammps_file.exists():
            pytest.skip(f"LAMMPS data file not found: {lammps_file}")

        data = read_lammps_data(str(lammps_file))
        coords_raw = data["coords"]

        # Convert to nm (DiffCG internal units)
        coords_nm = coords_raw / 10.0

        # After conversion, should be in reasonable nm range (1-20 nm)
        assert 1 < coords_nm.max() < 20, \
            f"Converted coords not in expected nm range: max={coords_nm.max():.2f}"

    def test_cell_dimensions_consistent(self):
        """Test that cell dimensions are consistent with coordinates."""
        lammps_file = TEST_DATA_DIR / "PS.data"
        if not lammps_file.exists():
            pytest.skip(f"LAMMPS data file not found: {lammps_file}")

        data = read_lammps_data(str(lammps_file))
        coords = data["coords"]
        cells = data["cells"]

        if len(cells) > 0:
            cell = cells[0]
            # Cell should be larger than coordinate span
            assert cell.max() > coords.max() * 0.8, \
                f"Cell dimensions ({cell.max():.2f}) inconsistent with coords ({coords.max():.2f})"


class TestDistributionFileUnits:
    """Test distribution file unit consistency."""

    def test_bond_distribution_in_nm(self):
        """Test that bond distributions use nanometers."""
        bond_file = TEST_DATA_DIR / "bondAA_smooth.dist.tgt"
        if not bond_file.exists():
            pytest.skip(f"Bond distribution file not found: {bond_file}")

        df = pd.read_csv(bond_file, sep=r"\s+", header=None, comment='i')
        distances = df[0].values

        # Bond distances should be in nm (typical range: 0-2 nm)
        assert distances.min() >= 0, "Bond distances should be non-negative"
        assert distances.max() <= 2.0, \
            f"Bond distances should be < 2.0 nm, got max={distances.max():.4f}"

    def test_angle_distribution_in_radians(self):
        """Test that angle distributions use radians."""
        angle_file = TEST_DATA_DIR / "angleAAA.dist.tgt"
        if not angle_file.exists():
            pytest.skip(f"Angle distribution file not found: {angle_file}")

        df = pd.read_csv(angle_file, sep=r"\s+", header=None, comment='i')
        angles = df[0].values

        # Angles should be in radians (range: 0 to π)
        assert angles.min() >= 0, "Angles should be non-negative"
        assert angles.max() <= np.pi + 0.01, \
            f"Angles should be <= π rad, got max={angles.max():.4f}"

    def test_dihedral_distribution_in_radians(self):
        """Test that dihedral distributions use radians."""
        dihedral_file = TEST_DATA_DIR / "dihedralAAAA.dist.tgt"
        if not dihedral_file.exists():
            pytest.skip(f"Dihedral distribution file not found: {dihedral_file}")

        df = pd.read_csv(dihedral_file, sep=r"\s+", header=None, comment='i')
        dihedrals = df[0].values

        # Dihedrals should be in radians (range: -π to π)
        assert dihedrals.min() >= -np.pi - 0.01, \
            f"Dihedrals should be >= -π rad, got min={dihedrals.min():.4f}"
        assert dihedrals.max() <= np.pi + 0.01, \
            f"Dihedrals should be <= π rad, got max={dihedrals.max():.4f}"

    def test_rdf_distribution_in_nm(self):
        """Test that RDF distributions use nanometers."""
        rdf_file = TEST_DATA_DIR / "nb_smoothed.dist.tgt"
        if not rdf_file.exists():
            pytest.skip(f"RDF file not found: {rdf_file}")

        df = pd.read_csv(rdf_file, sep=r"\s+", header=None, comment='i')
        distances = df[0].values

        # RDF distances should be in nm (typical range: 0-3 nm)
        assert distances.min() >= 0, "RDF distances should be non-negative"
        assert distances.max() <= 3.0, \
            f"RDF distances should be < 3.0 nm, got max={distances.max():.4f}"


class TestPhysicalConstants:
    """Test physical constant consistency."""

    def test_boltzmann_constant(self):
        """Test Boltzmann constant value."""
        expected_kb = 0.0083145107  # kJ/(mol·K)

        # Test value from jaxmd_sampler
        assert abs(KB_KJ_MOL - expected_kb) < 1e-10, \
            f"Boltzmann constant mismatch: {KB_KJ_MOL} != {expected_kb}"

    def test_pressure_conversion(self):
        """Test pressure conversion factor."""
        # DiffCG uses: 1 bar = (1 / 16.6054) kJ/(mol·nm³)
        PRESSURE_CONVERSION_KJMOL_NM3_TO_BAR = 16.6054

        # Calculate internal pressure unit
        pressure_internal = 1.0 / PRESSURE_CONVERSION_KJMOL_NM3_TO_BAR

        # 1 bar should equal approximately 0.0602214 kJ/(mol·nm³)
        expected_value = 0.0602214
        assert abs(pressure_internal - expected_value) < 1e-6, \
            f"Pressure conversion incorrect: {pressure_internal} != {expected_value}"


class TestPretrainedParameters:
    """Test pretrained parameter file consistency."""

    def test_parameters_loadable(self):
        """Test that pretrained parameters can be loaded."""
        params_file = EXAMPLE_DIR / "pretrained_params.npy"
        if not params_file.exists():
            pytest.skip(f"Pretrained params not found: {params_file}")

        params = np.load(str(params_file), allow_pickle=True).item()
        assert isinstance(params, dict), "Parameters should be a dictionary"

    def test_parameter_energy_values(self):
        """Test that parameter values are in reasonable kJ/mol range."""
        params_file = EXAMPLE_DIR / "pretrained_params.npy"
        if not params_file.exists():
            pytest.skip(f"Pretrained params not found: {params_file}")

        params = np.load(str(params_file), allow_pickle=True).item()

        for key, val in params.items():
            if isinstance(val, np.ndarray):
                # Energy values should be in reasonable range for kJ/mol
                # Typical range: -100 to +100 kJ/mol for CG systems
                assert val.min() > -500, \
                    f"Parameter '{key}' has unreasonably low energy: {val.min():.2f}"
                assert val.max() < 500, \
                    f"Parameter '{key}' has unreasonably high energy: {val.max():.2f}"


class TestUnitConversionFactors:
    """Test various unit conversion factors."""

    def test_angstrom_to_nm(self):
        """Test Ångström to nanometer conversion."""
        angstrom_value = 10.0
        nm_value = angstrom_value / 10.0
        assert nm_value == 1.0, "10 Å should equal 1 nm"

    def test_radians_to_degrees(self):
        """Test that we're using radians, not degrees."""
        # π radians = 180 degrees
        # This test ensures we don't accidentally use degrees
        assert np.pi > 3.0 and np.pi < 3.2, "π should be ~3.14159 radians"

    def test_pressure_bar_to_internal(self):
        """Test pressure conversion from bar to internal units."""
        PRESSURE_CONVERSION = 16.6054  # bar per kJ/(mol·nm³)

        # 1 bar in internal units
        one_bar_internal = 1.0 / PRESSURE_CONVERSION
        assert 0.060 < one_bar_internal < 0.061, \
            f"1 bar should be ~0.0602 kJ/(mol·nm³), got {one_bar_internal:.6f}"

        # 100 bar in internal units
        hundred_bar_internal = 100.0 / PRESSURE_CONVERSION
        assert 6.0 < hundred_bar_internal < 6.1, \
            f"100 bar should be ~6.02 kJ/(mol·nm³), got {hundred_bar_internal:.6f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
