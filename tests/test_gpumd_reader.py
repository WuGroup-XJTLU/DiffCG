import tempfile
import os
import jax.numpy as jnp
from diffcg.io.gpumd_reader import read_dump_xyz

SAMPLE_DUMP = """2
Time=10.00000000 pbc="T T T" Lattice="10.00000000 0.00000000 0.00000000 0.00000000 10.00000000 0.00000000 0.00000000 0.00000000 10.00000000" energy=5.0 virial="..." stress="..." Properties=species:S:1:pos:R:3
H 1.00000000 2.00000000 3.00000000
He 4.00000000 5.00000000 6.00000000
2
Time=20.00000000 pbc="T T T" Lattice="10.00000000 0.00000000 0.00000000 0.00000000 10.00000000 0.00000000 0.00000000 0.00000000 10.00000000" energy=4.5 virial="..." stress="..." Properties=species:S:1:pos:R:3
H 1.50000000 2.50000000 3.50000000
He 4.50000000 5.50000000 6.50000000
"""


def test_read_dump_xyz():
    path = tempfile.mktemp(suffix='.xyz')
    try:
        with open(path, 'w') as f:
            f.write(SAMPLE_DUMP)
        Z = jnp.array([0, 1], dtype=jnp.int32)
        traj = read_dump_xyz(path, Z=Z)
    finally:
        os.unlink(path)

    assert len(traj) == 2
    assert traj.positions.shape == (2, 2, 3)
    assert abs(float(traj.positions[0, 0, 0]) - 0.1) < 1e-6  # 1.0 A = 0.1 nm
    assert abs(float(traj.cell[0, 0]) - 1.0) < 1e-6  # 10 A = 1.0 nm
