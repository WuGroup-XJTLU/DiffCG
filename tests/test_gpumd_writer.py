import tempfile
import os
import numpy as np
import jax.numpy as jnp
from diffcg.system import AtomicSystem
from diffcg.io.gpumd_writer import write_xyz_in


def test_write_xyz_in_basic():
    R = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=jnp.float32)
    Z = jnp.array([0, 1], dtype=jnp.int32)
    cell = jnp.eye(3) * 2.0
    system = AtomicSystem(R=R, Z=Z, cell=cell, pbc=True)

    path = tempfile.mktemp(suffix='.xyz')
    try:
        write_xyz_in(system, path)
        with open(path) as f:
            content = f.read()
    finally:
        os.unlink(path)

    lines = content.strip().split('\n')
    assert lines[0] == '2'
    assert 'Lattice=' in lines[1]
    assert 'Properties=species:S:1:pos:R:3' in lines[1]
    parts = lines[2].split()
    assert parts[0] == 'H'
    assert abs(float(parts[1]) - 1.0) < 0.01  # 0.1 nm = 1.0 A


def test_write_xyz_in_default_masses_no_mass_column():
    """When masses match standard atomic masses, mass column should not appear."""
    R = jnp.array([[0.1, 0.2, 0.3]], dtype=jnp.float32)
    Z = jnp.array([6], dtype=jnp.int32)  # N
    cell = jnp.eye(3) * 2.0
    system = AtomicSystem(
        R=R, Z=Z, cell=cell, pbc=True,
        masses=jnp.array([14.007], dtype=jnp.float32),
    )

    path = tempfile.mktemp(suffix='.xyz')
    try:
        write_xyz_in(system, path)
        with open(path) as f:
            content = f.read()
    finally:
        os.unlink(path)

    lines = content.strip().split('\n')
    assert 'Properties=species:S:1:pos:R:3' in lines[1]
    assert ':mass:R:1' not in lines[1]
    parts = lines[2].split()
    assert len(parts) == 4  # sym x y z


def test_write_xyz_in_custom_masses_includes_mass_column():
    """When masses differ from standard, mass column should appear."""
    R = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=jnp.float32)
    Z = jnp.array([0, 0], dtype=jnp.int32)  # H symbols
    cell = jnp.eye(3) * 2.0
    system = AtomicSystem(
        R=R, Z=Z, cell=cell, pbc=True,
        masses=jnp.array([18.015, 18.015], dtype=jnp.float32),
    )

    path = tempfile.mktemp(suffix='.xyz')
    try:
        write_xyz_in(system, path)
        with open(path) as f:
            content = f.read()
    finally:
        os.unlink(path)

    lines = content.strip().split('\n')
    assert 'Properties=species:S:1:pos:R:3:mass:R:1' in lines[1]
    parts = lines[2].split()
    assert len(parts) == 5  # sym x y z mass
    assert abs(float(parts[4]) - 18.015) < 1e-6


def test_write_xyz_in_custom_masses_and_velocities():
    """Custom masses + velocities should produce species:S:1:pos:R:3:mass:R:1:vel:R:3."""
    R = jnp.array([[0.1, 0.2, 0.3]], dtype=jnp.float32)
    Z = jnp.array([0], dtype=jnp.int32)
    cell = jnp.eye(3) * 2.0
    system = AtomicSystem(
        R=R, Z=Z, cell=cell, pbc=True,
        masses=jnp.array([18.015], dtype=jnp.float32),
        velocities=jnp.array([[0.01, 0.02, 0.03]], dtype=jnp.float32),
    )

    path = tempfile.mktemp(suffix='.xyz')
    try:
        write_xyz_in(system, path)
        with open(path) as f:
            content = f.read()
    finally:
        os.unlink(path)

    lines = content.strip().split('\n')
    assert 'Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3' in lines[1]
    parts = lines[2].split()
    assert len(parts) == 8  # sym x y z mass vx vy vz
    assert abs(float(parts[4]) - 18.015) < 1e-6
