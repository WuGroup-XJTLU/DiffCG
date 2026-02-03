# SPDX-License-Identifier: MIT
# Copyright (c) 2025 WuResearchGroup

"""Tests for LAMMPS sampler, unit conversions, and I/O writers."""

import os
import shutil
import tempfile

import numpy as np
import pytest

# --- Unit conversion tests ---


class TestUnitConversions:
    def test_positions_roundtrip(self):
        from diffcg._core.units import positions_to_lammps, positions_from_lammps

        R_nm = np.array([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]])
        R_ang = positions_to_lammps(R_nm)
        np.testing.assert_allclose(R_ang, R_nm * 10.0)
        R_back = positions_from_lammps(R_ang)
        np.testing.assert_allclose(R_back, R_nm)

    def test_cell_roundtrip(self):
        from diffcg._core.units import cell_to_lammps, cell_from_lammps

        cell_nm = np.diag([3.0, 4.0, 5.0])
        cell_ang = cell_to_lammps(cell_nm)
        np.testing.assert_allclose(cell_ang, cell_nm * 10.0)
        cell_back = cell_from_lammps(cell_ang)
        np.testing.assert_allclose(cell_back, cell_nm)

    def test_energy_roundtrip(self):
        from diffcg._core.units import energy_to_lammps, energy_from_lammps

        E_kjmol = np.array([10.0, -5.0, 0.0])
        E_kcalmol = energy_to_lammps(E_kjmol)
        np.testing.assert_allclose(E_kcalmol, E_kjmol / 4.184)
        E_back = energy_from_lammps(E_kcalmol)
        np.testing.assert_allclose(E_back, E_kjmol)

    def test_constants(self):
        from diffcg._core.units import NM_TO_ANGSTROM, KJMOL_TO_KCALMOL, RAD_TO_DEG

        assert NM_TO_ANGSTROM == 10.0
        np.testing.assert_allclose(KJMOL_TO_KCALMOL, 1.0 / 4.184)
        np.testing.assert_allclose(RAD_TO_DEG, 180.0 / np.pi)


# --- LAMMPS data writer tests ---


class TestWriteLammpsData:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def _make_system(self):
        import jax.numpy as jnp

        from diffcg.system import AtomicSystem

        R = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=jnp.float32)
        Z = jnp.array([0, 0, 1], dtype=jnp.int32)
        cell = jnp.diag(jnp.array([3.0, 3.0, 3.0], dtype=jnp.float32))
        masses = jnp.array([12.0, 12.0, 16.0], dtype=jnp.float32)
        return AtomicSystem(R=R, Z=Z, cell=cell, masses=masses, pbc=True)

    def test_write_and_read_back(self):
        from diffcg.io.lammps import LAMMPSDataReader
        from diffcg.io.lammps_writer import write_lammps_data

        system = self._make_system()
        topology = {
            "bonds": np.array([[0, 1], [1, 2]]),
            "bond_types": np.array([0, 0]),
            "angles": np.array([[0, 1, 2]]),
            "angle_types": np.array([0]),
            "dihedrals": np.empty((0, 4), dtype=int),
            "dihedral_types": np.empty(0, dtype=int),
        }
        filepath = os.path.join(self.tmpdir, "test.data")
        write_lammps_data(system, topology, filepath)

        reader = LAMMPSDataReader(filepath)
        data = reader.read()

        # Check number of atoms
        assert data["coords"].shape == (1, 3, 3)

        # Check positions are in Angstroms (nm * 10)
        expected_ang = np.array(system.R) * 10.0
        np.testing.assert_allclose(data["coords"][0], expected_ang, atol=1e-4)

        # Check bonds (0-indexed in reader output)
        assert data["bonds"].shape == (2, 2)
        np.testing.assert_array_equal(data["bonds"], np.array([[0, 1], [1, 2]]))

        # Check angles
        assert data["angles"].shape == (1, 3)
        np.testing.assert_array_equal(data["angles"], np.array([[0, 1, 2]]))

    def test_write_with_mol_ids(self):
        from diffcg.io.lammps_writer import write_lammps_data

        system = self._make_system()
        topology = {
            "bonds": np.empty((0, 2), dtype=int),
            "bond_types": np.empty(0, dtype=int),
        }
        mol_ids = np.array([0, 0, 1])
        filepath = os.path.join(self.tmpdir, "test_mol.data")
        write_lammps_data(system, topology, filepath, mol_ids=mol_ids)

        with open(filepath) as f:
            content = f.read()

        # Atom 1 should have mol_id=1, atom 3 should have mol_id=2
        assert "1 1 1" in content  # atom 1, mol 1, type 1
        assert "3 2 2" in content  # atom 3, mol 2, type 2


# --- LAMMPS table writer tests ---


class TestWriteLammpsTable:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def test_pair_table_format(self):
        from diffcg.io.lammps_writer import write_lammps_table

        x_nm = np.linspace(0.3, 1.2, 50)
        # Simple LJ-like potential in kJ/mol
        y_kjmol = 4.0 * ((0.4 / x_nm) ** 12 - (0.4 / x_nm) ** 6)

        filepath = os.path.join(self.tmpdir, "pair.txt")
        open(filepath, "w").close()
        write_lammps_table(filepath, x_nm, y_kjmol, "PAIR_0", "pair")

        with open(filepath) as f:
            lines = f.readlines()

        # Should have header, keyword, N line, blank, then data
        assert any("PAIR_0" in line for line in lines)
        assert any("N 50" in line for line in lines)

        # Parse a data line and check distance is in Angstroms
        data_lines = [l for l in lines if l.strip() and l.strip()[0].isdigit()]
        assert len(data_lines) == 50

        first = data_lines[0].split()
        r_ang = float(first[1])
        np.testing.assert_allclose(r_ang, 0.3 * 10.0, atol=1e-6)

    def test_pair_table_force_consistency(self):
        """Force should be approximately -dE/dr."""
        from diffcg.io.lammps_writer import write_lammps_table

        x_nm = np.linspace(0.3, 1.2, 200)
        y_kjmol = 4.0 * ((0.4 / x_nm) ** 12 - (0.4 / x_nm) ** 6)

        filepath = os.path.join(self.tmpdir, "pair_force.txt")
        open(filepath, "w").close()
        write_lammps_table(filepath, x_nm, y_kjmol, "TEST", "pair")

        with open(filepath) as f:
            lines = f.readlines()

        data_lines = [l for l in lines if l.strip() and l.strip()[0].isdigit()]
        rs, es, fs = [], [], []
        for line in data_lines:
            parts = line.split()
            rs.append(float(parts[1]))
            es.append(float(parts[2]))
            fs.append(float(parts[3]))

        rs = np.array(rs)
        es = np.array(es)
        fs = np.array(fs)

        # Numerical derivative of E w.r.t. r
        dEdr_num = np.gradient(es, rs)
        # Force should equal -dE/dr
        np.testing.assert_allclose(fs, -dEdr_num, atol=1e-3)

    def test_angle_table_format(self):
        from diffcg.io.lammps_writer import write_lammps_table

        x_rad = np.linspace(0.5, np.pi, 30)
        y_kjmol = 100.0 * (x_rad - 2.0) ** 2

        filepath = os.path.join(self.tmpdir, "angle.txt")
        open(filepath, "w").close()
        write_lammps_table(filepath, x_rad, y_kjmol, "ANGLE_0", "angle")

        with open(filepath) as f:
            lines = f.readlines()

        data_lines = [l for l in lines if l.strip() and l.strip()[0].isdigit()]
        assert len(data_lines) == 30

        # First angle should be in degrees
        first = data_lines[0].split()
        angle_deg = float(first[1])
        np.testing.assert_allclose(angle_deg, 0.5 * 180.0 / np.pi, atol=1e-4)


# --- Input script generation tests ---


class TestInputScriptGeneration:
    def _make_sampler(self, thermostat="langevin", **kwargs):
        import jax.numpy as jnp

        from diffcg.md.lammps_sampler import LAMMPSSampler
        from diffcg.system import AtomicSystem

        R = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=jnp.float32)
        Z = jnp.array([0, 0], dtype=jnp.int32)
        cell = jnp.diag(jnp.array([3.0, 3.0, 3.0], dtype=jnp.float32))
        masses = jnp.array([12.0, 12.0], dtype=jnp.float32)
        system = AtomicSystem(R=R, Z=Z, cell=cell, masses=masses, pbc=True)

        x_pair = np.linspace(0.3, 1.0, 20)
        y_pair = np.zeros(20)
        energy_params = {"pair": {"x": x_pair, "y": y_pair}}
        topology = {
            "bonds": np.empty((0, 2), dtype=int),
            "bond_types": np.empty(0, dtype=int),
        }

        return LAMMPSSampler(
            system,
            energy_params=energy_params,
            topology=topology,
            thermostat=thermostat,
            work_dir=None,
            **kwargs,
        )

    def test_langevin_script(self):
        sampler = self._make_sampler(thermostat="langevin")
        script = sampler.get_input_script(1000)

        assert "units real" in script
        assert "atom_style full" in script
        assert "pair_style table" in script
        assert "fix 1 all nve" in script
        assert "fix 2 all langevin" in script
        assert "run 1000" in script

    def test_nose_hoover_script(self):
        sampler = self._make_sampler(thermostat="nose-hoover")
        script = sampler.get_input_script(500)

        assert "fix 1 all nvt" in script
        assert "run 500" in script

    def test_custom_template(self):
        sampler = self._make_sampler()
        custom = "units real\nread_data {data_file}\n{pair_style_block}\nrun {steps}\n"
        sampler.set_input_template(custom)
        script = sampler.get_input_script(100)

        assert "units real" in script
        assert "run 100" in script
        # Should NOT have thermostat since custom template doesn't include it
        assert "langevin" not in script

    def test_default_special_bonds(self):
        sampler = self._make_sampler()
        script = sampler.get_input_script(100)
        assert "special_bonds lj 0.0 0.0 0.0" in script

    def test_custom_special_bonds(self):
        sampler = self._make_sampler(special_bonds="lj 0.0 0.0 1.0")
        script = sampler.get_input_script(100)
        assert "special_bonds lj 0.0 0.0 1.0" in script
        assert "special_bonds lj 0.0 0.0 0.0" not in script


# --- Dump reader sort test ---


class TestDumpReaderSort:
    def test_atoms_sorted_by_id(self):
        from diffcg.io.lammps import LAMMPSDumpReader

        # Create a dump with atoms out of order
        tmpdir = tempfile.mkdtemp()
        dump_path = os.path.join(tmpdir, "test.dump")
        with open(dump_path, "w") as f:
            f.write(
                "ITEM: TIMESTEP\n0\n"
                "ITEM: NUMBER OF ATOMS\n3\n"
                "ITEM: BOX BOUNDS pp pp pp\n"
                "0.0 30.0\n0.0 30.0\n0.0 30.0\n"
                "ITEM: ATOMS id type x y z\n"
                "3 1 9.0 9.0 9.0\n"
                "1 1 1.0 2.0 3.0\n"
                "2 2 4.0 5.0 6.0\n"
            )

        reader = LAMMPSDumpReader(dump_path)
        data = reader.read()

        # After sorting, atom 1 should come first
        np.testing.assert_allclose(data["coords"][0, 0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(data["coords"][0, 1], [4.0, 5.0, 6.0])
        np.testing.assert_allclose(data["coords"][0, 2], [9.0, 9.0, 9.0])

        shutil.rmtree(tmpdir)


# --- End-to-end LAMMPS sampler test ---


@pytest.mark.skipif(
    shutil.which("lmp") is None,
    reason="LAMMPS executable 'lmp' not found in PATH",
)
class TestLAMMPSSamplerRun:
    def test_small_system(self):
        import jax.numpy as jnp

        from diffcg.md.lammps_sampler import LAMMPSSampler
        from diffcg.system import AtomicSystem

        # Small 4-atom system in a periodic box
        R = jnp.array(
            [[0.5, 0.5, 0.5], [1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [1.0, 1.0, 0.5]],
            dtype=jnp.float32,
        )
        Z = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
        cell = jnp.diag(jnp.array([2.0, 2.0, 2.0], dtype=jnp.float32))
        masses = jnp.array([12.0, 12.0, 12.0, 12.0], dtype=jnp.float32)
        system = AtomicSystem(R=R, Z=Z, cell=cell, masses=masses, pbc=True)

        # Simple repulsive pair potential
        x_pair = np.linspace(0.2, 0.9, 50)
        y_pair = 10.0 * np.exp(-x_pair / 0.3)  # kJ/mol
        energy_params = {"pair": {"x": x_pair, "y": y_pair}}

        topology = {
            "bonds": np.empty((0, 2), dtype=int),
            "bond_types": np.empty(0, dtype=int),
        }

        tmpdir = tempfile.mkdtemp()
        sampler = LAMMPSSampler(
            system,
            energy_params=energy_params,
            topology=topology,
            temperature=300.0,
            timestep=1.0,
            cutoff=0.9,
            loginterval=10,
            work_dir=tmpdir,
            random_seed=42,
        )

        traj = sampler.run(100)

        # Check trajectory shape (LAMMPS dumps frame 0 + frames at each loginterval)
        assert traj.positions.shape[0] == 11  # 100 steps / 10 loginterval + initial frame
        assert traj.positions.shape[1] == 4  # 4 atoms
        assert traj.positions.shape[2] == 3

        # Check positions are in nm (should be roughly 0-2 nm range)
        pos = np.array(traj.positions)
        assert pos.max() < 5.0  # nm, not Angstroms
        assert pos.min() > -1.0

        # Check final system
        final = sampler.get_final_system()
        assert final.n_atoms == 4

        # Check thermo log
        thermo = sampler.get_thermo_log()
        assert thermo is not None
        assert len(thermo["step"]) > 0

        shutil.rmtree(tmpdir)
