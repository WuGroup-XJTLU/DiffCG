from diffcg.nep.constants import C3B, C4B, C5B


def test_c3b_known_values():
    """First 3 values from GPUMD nep_utilities.cuh."""
    assert abs(float(C3B[0]) - 0.238732414637843) < 1e-5
    assert abs(float(C3B[1]) - 0.119366207318922) < 1e-5
    assert len(C3B) == 80


def test_c4b_known_values():
    assert len(C4B) == 5
    assert abs(float(C4B[0]) - (-0.007499480826664)) < 1e-5


def test_c5b_known_values():
    assert len(C5B) == 3
