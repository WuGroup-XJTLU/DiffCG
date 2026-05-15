import tempfile
import os
from diffcg.io.gpumd_thermo import read_thermo

SAMPLE_THERMO = """ 3.00e+02 1.00e+01 -5.00e+02 1.0e-04 1.0e-04 1.0e-04 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0
 3.10e+02 1.10e+01 -4.80e+02 1.0e-04 1.0e-04 1.0e-04 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0
"""


def test_read_thermo():
    path = tempfile.mktemp(suffix='.out')
    try:
        with open(path, 'w') as f:
            f.write(SAMPLE_THERMO)
        thermo = read_thermo(path)
    finally:
        os.unlink(path)

    assert len(thermo["temperature"]) == 2
    assert abs(thermo["temperature"][0] - 300.0) < 0.1
    assert thermo["pe"][0] < 0  # should be negative (bound state)
    assert thermo["box_h"].shape == (2, 3, 3)
