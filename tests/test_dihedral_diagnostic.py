"""Diagnostic: compare JAX vs LAMMPS dihedral energy.

Three parts:
  A) Single-angle sweep — position 4 atoms at controlled dihedral angles
  B) Real trajectory comparison (if output_lammps_bap trajectories exist)
  C) Table round-trip — write table, read back, compare at grid points

Run with:
    conda activate diffcg
    python tests/test_dihedral_diagnostic.py
"""

import os
import shutil
import tempfile

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from diffcg._core.interpolate import MonotonicInterpolate
from diffcg._core.geometry import dihedral as compute_dihedral
from diffcg.energy import (
    TabulatedDihedralEnergy,
    harmonic_dihedral,
    _smooth_cutoff_factor,
    generic_repulsion,
    simple_spring,
    harmonic_angle,
)
from diffcg.io.lammps_writer import write_lammps_data, write_lammps_table
from diffcg.md.lammps_sampler import LAMMPSSampler
from diffcg.system import AtomicSystem
from diffcg._core.units import KCALMOL_TO_KJMOL, KJMOL_TO_KCALMOL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_dihedral_positions(theta_deg):
    """Place 4 atoms so that the dihedral angle a-b-c-d equals theta_deg.

    Atoms:
      a = (0, 0, 0)
      b = (1, 0, 0)   -- bond along x
      c = (1.5, 1, 0)  -- bend in xy-plane
      d rotated around bc axis to give desired dihedral
    """
    theta = np.radians(theta_deg)

    a = np.array([0.0, 0.0, 0.0])
    b = np.array([0.5, 0.0, 0.0])
    c = np.array([0.75, 0.5, 0.0])

    # bc vector
    bc = c - b
    bc_unit = bc / np.linalg.norm(bc)

    # A vector perpendicular to bc in the abc plane (reference direction)
    ab = b - a
    # Project ab onto plane perpendicular to bc
    perp = ab - np.dot(ab, bc_unit) * bc_unit
    perp = perp / np.linalg.norm(perp)

    # Second perpendicular direction
    perp2 = np.cross(bc_unit, perp)

    # Place d at distance 0.5 from c, at angle theta around bc
    d = c + 0.5 * (-np.cos(theta) * perp + np.sin(theta) * perp2)

    positions = np.array([a, b, c, d])
    return positions


def build_test_potential():
    """Build an asymmetric dihedral potential: spline + harmonic prior."""
    rng = np.random.RandomState(42)
    x_knots = jnp.linspace(-np.pi, np.pi, 12)
    y_knots = jnp.array(rng.randn(12) * 2.0)
    spline = MonotonicInterpolate(x_knots, y_knots)

    x_dihedral = np.linspace(-np.pi + 1e-4, np.pi - 1e-4, 500)
    dihedral_y = np.asarray(
        spline(x_dihedral) + harmonic_dihedral(x_dihedral, angle_0=1.5, epsilon=50)
    )
    return x_dihedral, dihedral_y, spline


def compute_jax_dihedral_energy(positions, x_dihedral, dihedral_y):
    """Compute dihedral energy using JAX TabulatedDihedralEnergy."""
    R = jnp.array(positions, dtype=jnp.float64)
    Z = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
    cell = jnp.diag(jnp.array([10.0, 10.0, 10.0], dtype=jnp.float64))
    masses = jnp.array([12.0, 12.0, 12.0, 12.0], dtype=jnp.float64)
    system = AtomicSystem(R=R, Z=Z, cell=cell, masses=masses, pbc=True)

    dihedral_topology = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
    tab_energy = TabulatedDihedralEnergy(
        x_vals=x_dihedral, y_vals=dihedral_y, dihedrals=dihedral_topology,
    )
    energy_fn = tab_energy.get_energy_fn()
    return float(energy_fn(system, neighbors=None)), system


def compute_lammps_dihedral_energy(system, x_dihedral, dihedral_y):
    """Run 0-step LAMMPS and extract PE (should be dihedral-only)."""
    # Zero pair/bond/angle potentials
    x_pair = np.linspace(0.05, 1.5, 100)
    y_pair = np.zeros_like(x_pair)
    x_bond = np.linspace(0.1, 2.0, 100)
    y_bond = np.zeros_like(x_bond)
    x_angle = np.linspace(0.0, np.pi, 100)
    y_angle = np.zeros_like(x_angle)

    energy_params = {
        "pair": {"x": x_pair, "y": y_pair},
        "bond": {"x": x_bond, "y": y_bond},
        "angle": {"x": x_angle, "y": y_angle},
        "dihedral": {"x": x_dihedral, "y": dihedral_y},
    }
    topology = {
        "bonds": np.array([[0, 1], [1, 2], [2, 3]], dtype=int),
        "bond_types": np.zeros(3, dtype=int),
        "angles": np.array([[0, 1, 2], [1, 2, 3]], dtype=int),
        "angle_types": np.zeros(2, dtype=int),
        "dihedrals": np.array([[0, 1, 2, 3]], dtype=int),
        "dihedral_types": np.zeros(1, dtype=int),
    }

    tmpdir = tempfile.mkdtemp(prefix="dihedral_diag_")
    try:
        sampler = LAMMPSSampler(
            system,
            energy_params=energy_params,
            topology=topology,
            temperature=300.0,
            timestep=1.0,
            cutoff=1.5,
            loginterval=1,
            work_dir=tmpdir,
            random_seed=42,
            special_bonds="lj 0.0 0.0 0.0",
        )

        sampler.run(1)
        thermo = sampler.get_thermo_log()
        lammps_pe = thermo["pe"][0]  # Already converted to kJ/mol
        return lammps_pe, tmpdir
    except Exception as e:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise e


def compute_raw_dihedral_angle(positions):
    """Compute dihedral angle in radians using the same convention as DiffCG."""
    R = jnp.array(positions, dtype=jnp.float64)
    R_ab = R[1] - R[0]
    R_bc = R[2] - R[1]
    R_cd = R[3] - R[2]
    return float(compute_dihedral(R_ab, R_bc, R_cd))


# ---------------------------------------------------------------------------
# Part A: Single-angle sweep
# ---------------------------------------------------------------------------

def part_a_angle_sweep():
    """Sweep dihedral angles and compare JAX vs LAMMPS energy."""
    print("=" * 72)
    print("PART A: Single-Angle Dihedral Sweep")
    print("=" * 72)

    x_dihedral, dihedral_y, spline = build_test_potential()

    test_angles = [-170, -135, -90, -45, 0, 45, 90, 135, 170]

    results = []
    print(f"\n{'θ_target':>10} {'θ_actual':>10} {'E_JAX':>12} {'E_LAMMPS':>12} "
          f"{'Δ(kJ/mol)':>12} {'Δ(%)':>10}")
    print("-" * 72)

    for theta_deg in test_angles:
        positions = make_dihedral_positions(theta_deg)
        theta_actual_rad = compute_raw_dihedral_angle(positions)
        theta_actual_deg = np.degrees(theta_actual_rad)

        jax_e, system = compute_jax_dihedral_energy(positions, x_dihedral, dihedral_y)

        try:
            lammps_e, tmpdir = compute_lammps_dihedral_energy(
                system, x_dihedral, dihedral_y
            )
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception as e:
            print(f"{theta_deg:>10.1f} {theta_actual_deg:>10.2f} {jax_e:>12.4f} "
                  f"{'ERROR':>12} -- {str(e)[:30]}")
            continue

        diff = lammps_e - jax_e
        pct = 100 * diff / abs(jax_e) if abs(jax_e) > 1e-8 else float('nan')

        results.append({
            'theta_target': theta_deg,
            'theta_actual': theta_actual_deg,
            'jax_e': jax_e,
            'lammps_e': lammps_e,
            'diff': diff,
            'pct': pct,
        })

        print(f"{theta_deg:>10.1f} {theta_actual_deg:>10.2f} {jax_e:>12.4f} "
              f"{lammps_e:>12.4f} {diff:>12.4f} {pct:>10.2f}")

    if results:
        diffs = [r['diff'] for r in results]
        print(f"\nSummary: max |Δ| = {max(abs(d) for d in diffs):.4f} kJ/mol, "
              f"mean |Δ| = {np.mean(np.abs(diffs)):.4f} kJ/mol")

        # Check for systematic shift: does LAMMPS E at θ match JAX E at θ+shift?
        print("\n--- Shift analysis ---")
        jax_energies = np.array([r['jax_e'] for r in results])
        lammps_energies = np.array([r['lammps_e'] for r in results])
        actual_angles = np.array([r['theta_actual'] for r in results])

        # Evaluate JAX potential at shifted angles
        for shift in [0, 90, -90, 180]:
            shifted_angles_rad = np.radians(actual_angles + shift)
            # Wrap to [-pi, pi]
            shifted_angles_rad = (shifted_angles_rad + np.pi) % (2 * np.pi) - np.pi
            # Evaluate potential directly
            shifted_e = np.interp(shifted_angles_rad, x_dihedral, dihedral_y)
            residual = np.mean(np.abs(lammps_energies - shifted_e))
            print(f"  Shift {shift:>4}°: mean |LAMMPS - JAX(θ+shift)| = {residual:.4f} kJ/mol")

    return results


# ---------------------------------------------------------------------------
# Part B: Real trajectory comparison (optional)
# ---------------------------------------------------------------------------

def part_b_trajectory_comparison():
    """Load a trajectory and compare per-frame dihedral energy JAX vs LAMMPS."""
    print("\n" + "=" * 72)
    print("PART B: Real Trajectory Comparison")
    print("=" * 72)

    # Try to find a trajectory
    base_dir = os.path.join(os.path.dirname(__file__), "..",
                            "example", "polystyrene", "output_lammps_bap")
    traj_path = os.path.join(base_dir, "iteration_1", "sample1.traj.npz")

    if not os.path.exists(traj_path):
        print(f"  Skipping: trajectory not found at {traj_path}")
        print("  (This part requires a pre-existing polystyrene trajectory)")
        return

    print(f"  Loading trajectory from {traj_path}")
    data = np.load(traj_path)
    print(f"  Keys: {list(data.keys())}")
    # Inspect what's in the trajectory
    for k in data.keys():
        print(f"    {k}: shape={data[k].shape}, dtype={data[k].dtype}")

    print("  (Full trajectory comparison would require matching topology/potential)")
    print("  See Part A for the controlled comparison.")


# ---------------------------------------------------------------------------
# Part C: Table round-trip test
# ---------------------------------------------------------------------------

def part_c_table_roundtrip():
    """Write a dihedral table, read it back, compare values."""
    print("\n" + "=" * 72)
    print("PART C: Table Round-Trip Test")
    print("=" * 72)

    x_dihedral, dihedral_y, _ = build_test_potential()

    tmpdir = tempfile.mkdtemp(prefix="table_roundtrip_")
    filepath = os.path.join(tmpdir, "dihedral_table.txt")

    try:
        # Write table
        open(filepath, "w").close()
        write_lammps_table(filepath, x_dihedral, dihedral_y, "DIHEDRAL_0", "dihedral")

        # Read it back
        angles_lammps = []
        energies_lammps = []
        forces_lammps = []
        with open(filepath, "r") as f:
            in_data = False
            for line in f:
                line = line.strip()
                if line == "DIHEDRAL_0":
                    in_data = True
                    continue
                if in_data and line.startswith("N "):
                    continue
                if in_data and line == "":
                    if angles_lammps:  # End of data section
                        break
                    continue
                if in_data and line and line[0].isdigit():
                    parts = line.split()
                    angles_lammps.append(float(parts[1]))   # degrees
                    energies_lammps.append(float(parts[2]))  # kcal/mol
                    forces_lammps.append(float(parts[3]))    # kcal/(mol*deg)

        angles_lammps = np.array(angles_lammps)
        energies_lammps = np.array(energies_lammps)
        forces_lammps = np.array(forces_lammps)

        # Convert back to DiffCG units
        angles_rad = np.radians(angles_lammps)
        energies_kjmol = energies_lammps * KCALMOL_TO_KJMOL

        # Compare with original
        print(f"\n  Table has {len(angles_lammps)} entries")
        print(f"  Angle range: [{angles_lammps[0]:.2f}°, {angles_lammps[-1]:.2f}°]")
        print(f"  Original x range: [{np.degrees(x_dihedral[0]):.2f}°, {np.degrees(x_dihedral[-1]):.2f}°]")

        # Direct comparison at original grid points
        max_e_diff = np.max(np.abs(energies_kjmol - dihedral_y))
        mean_e_diff = np.mean(np.abs(energies_kjmol - dihedral_y))
        print(f"\n  Energy round-trip error:")
        print(f"    max |Δ| = {max_e_diff:.6f} kJ/mol")
        print(f"    mean |Δ| = {mean_e_diff:.6f} kJ/mol")

        # Check angle conversion round-trip
        angle_diff = np.max(np.abs(angles_rad - x_dihedral))
        print(f"  Angle round-trip error: max |Δ| = {np.degrees(angle_diff):.6f}°")

        # Now test LAMMPS linear interpolation at midpoints
        print(f"\n  Linear interpolation test at midpoints:")
        midpoints_rad = 0.5 * (x_dihedral[:-1] + x_dihedral[1:])
        midpoints_deg = np.degrees(midpoints_rad)

        # LAMMPS uses linear interpolation between table entries
        interp_lammps = np.interp(midpoints_deg, angles_lammps, energies_lammps)
        interp_lammps_kjmol = interp_lammps * KCALMOL_TO_KJMOL

        # JAX spline evaluation at midpoints
        _, _, spline = build_test_potential()
        jax_at_midpoints = np.asarray(
            spline(midpoints_rad) +
            harmonic_dihedral(midpoints_rad, angle_0=1.5, epsilon=50)
        )

        interp_diff = np.abs(interp_lammps_kjmol - jax_at_midpoints)
        print(f"    max |Δ| = {np.max(interp_diff):.6f} kJ/mol")
        print(f"    mean |Δ| = {np.mean(interp_diff):.6f} kJ/mol")
        print(f"    (This includes both unit conversion and linear-vs-spline error)")

        # Show a few sample values for manual inspection
        print(f"\n  Sample table entries (first 5, middle 5, last 5):")
        print(f"  {'Index':>6} {'Angle(°)':>10} {'E_table(kcal)':>14} {'E_orig(kJ)':>12} {'E_table(kJ)':>12} {'Δ(kJ)':>10}")
        indices = list(range(5)) + list(range(245, 250)) + list(range(495, 500))
        for i in indices:
            if i < len(angles_lammps):
                print(f"  {i:>6} {angles_lammps[i]:>10.2f} {energies_lammps[i]:>14.6f} "
                      f"{dihedral_y[i]:>12.6f} {energies_kjmol[i]:>12.6f} "
                      f"{energies_kjmol[i] - dihedral_y[i]:>10.6f}")

        # Print the raw table file header for inspection
        print(f"\n  Raw table file (first 15 lines):")
        with open(filepath, "r") as f:
            for i, line in enumerate(f):
                if i >= 15:
                    break
                print(f"    {line.rstrip()}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Part D: Check dihedral angle computation consistency
# ---------------------------------------------------------------------------

def part_d_angle_computation():
    """Verify that make_dihedral_positions produces the expected angle."""
    print("\n" + "=" * 72)
    print("PART D: Dihedral Angle Computation Consistency")
    print("=" * 72)

    print(f"\n  {'θ_target':>10} {'θ_computed':>12} {'Error(°)':>10}")
    print("  " + "-" * 36)

    for theta_deg in [-170, -135, -90, -45, 0, 45, 90, 135, 170]:
        positions = make_dihedral_positions(theta_deg)
        theta_rad = compute_raw_dihedral_angle(positions)
        theta_computed = np.degrees(theta_rad)
        err = theta_computed - theta_deg
        # Wrap error to [-180, 180]
        err = (err + 180) % 360 - 180
        print(f"  {theta_deg:>10.1f} {theta_computed:>12.4f} {err:>10.4f}")


# ---------------------------------------------------------------------------
# Part E: Inspect actual LAMMPS input/output for one angle
# ---------------------------------------------------------------------------

def part_e_inspect_lammps():
    """Run one LAMMPS case and print the input script + thermo for debugging."""
    print("\n" + "=" * 72)
    print("PART E: LAMMPS Input/Output Inspection (θ=45°)")
    print("=" * 72)

    x_dihedral, dihedral_y, _ = build_test_potential()
    positions = make_dihedral_positions(45.0)
    theta_rad = compute_raw_dihedral_angle(positions)
    print(f"\n  Atom positions (nm):")
    for i, pos in enumerate(positions):
        print(f"    atom {i}: [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]")
    print(f"  Computed dihedral: {np.degrees(theta_rad):.4f}°  ({theta_rad:.6f} rad)")

    # JAX energy
    jax_e, system = compute_jax_dihedral_energy(positions, x_dihedral, dihedral_y)
    print(f"  JAX dihedral energy: {jax_e:.6f} kJ/mol")

    # Expected from direct interpolation
    direct_e = float(np.interp(theta_rad, x_dihedral, dihedral_y))
    print(f"  Direct np.interp at θ: {direct_e:.6f} kJ/mol")

    # Zero potentials for other terms
    x_pair = np.linspace(0.05, 1.5, 100)
    y_pair = np.zeros_like(x_pair)
    x_bond = np.linspace(0.1, 2.0, 100)
    y_bond = np.zeros_like(x_bond)
    x_angle = np.linspace(0.0, np.pi, 100)
    y_angle = np.zeros_like(x_angle)

    energy_params = {
        "pair": {"x": x_pair, "y": y_pair},
        "bond": {"x": x_bond, "y": y_bond},
        "angle": {"x": x_angle, "y": y_angle},
        "dihedral": {"x": x_dihedral, "y": dihedral_y},
    }
    topology = {
        "bonds": np.array([[0, 1], [1, 2], [2, 3]], dtype=int),
        "bond_types": np.zeros(3, dtype=int),
        "angles": np.array([[0, 1, 2], [1, 2, 3]], dtype=int),
        "angle_types": np.zeros(2, dtype=int),
        "dihedrals": np.array([[0, 1, 2, 3]], dtype=int),
        "dihedral_types": np.zeros(1, dtype=int),
    }

    tmpdir = tempfile.mkdtemp(prefix="dihedral_inspect_")
    try:
        sampler = LAMMPSSampler(
            system,
            energy_params=energy_params,
            topology=topology,
            temperature=300.0,
            timestep=1.0,
            cutoff=1.5,
            loginterval=1,
            work_dir=tmpdir,
            random_seed=42,
            special_bonds="lj 0.0 0.0 0.0",
        )

        # Print the input script
        script = sampler.get_input_script(1)
        print(f"\n  LAMMPS input script:")
        for line in script.split("\n"):
            print(f"    {line}")

        # Run
        sampler.run(1)
        thermo = sampler.get_thermo_log()
        lammps_pe = thermo["pe"][0]
        print(f"\n  LAMMPS PE (kJ/mol): {lammps_pe:.6f}")
        print(f"  JAX PE (kJ/mol):    {jax_e:.6f}")
        print(f"  Difference:         {lammps_pe - jax_e:.6f} kJ/mol")

        # Print LAMMPS log
        log_path = os.path.join(tmpdir, "lammps.log")
        if os.path.exists(log_path):
            print(f"\n  LAMMPS log (last 30 lines):")
            with open(log_path, "r") as f:
                lines = f.readlines()
            for line in lines[-30:]:
                print(f"    {line.rstrip()}")

        # Print the dihedral table
        dih_table_path = os.path.join(tmpdir, "dihedral_table.txt")
        if os.path.exists(dih_table_path):
            # Show entries around the expected angle
            theta_deg = np.degrees(theta_rad)
            print(f"\n  Dihedral table entries near θ={theta_deg:.2f}°:")
            with open(dih_table_path, "r") as f:
                in_data = False
                for line in f:
                    stripped = line.strip()
                    if stripped == "DIHEDRAL_0":
                        in_data = True
                        continue
                    if in_data and stripped.startswith("N "):
                        continue
                    if in_data and stripped and stripped[0].isdigit():
                        parts = stripped.split()
                        angle_val = float(parts[1])
                        if abs(angle_val - theta_deg) < 5.0:
                            print(f"    {stripped}")

        # Also print data file for inspection
        data_path = os.path.join(tmpdir, "system.data")
        if os.path.exists(data_path):
            print(f"\n  LAMMPS data file:")
            with open(data_path, "r") as f:
                print(f"    {f.read()}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("DiffCG Dihedral Diagnostic: JAX vs LAMMPS")
    print("=" * 72)

    # Part D first — verify angle computation is correct
    part_d_angle_computation()

    # Part C — table round-trip (no LAMMPS needed)
    part_c_table_roundtrip()

    # Part E — detailed inspection of one case
    part_e_inspect_lammps()

    # Part A — full angle sweep
    results = part_a_angle_sweep()

    # Part B — trajectory comparison (optional)
    part_b_trajectory_comparison()

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
