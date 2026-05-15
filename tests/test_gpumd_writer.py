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
