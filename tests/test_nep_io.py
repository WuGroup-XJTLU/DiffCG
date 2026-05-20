import tempfile
import os
import jax.numpy as jnp
from diffcg.io.nep import read_nep, write_nep

# Minimal NEP sample: 1 type, small basis, 9 float params total
# num_types=1, n_max_r=0, n_max_a=0, basis_r=0, basis_a=0, L_max=1, neurons=1
# descriptor: 1*(1*1 + 1*1)=2, ann: (2+2)*1*1+1=5, q_scaler: dim=2 → 9 floats
SAMPLE_NEP = """nep4 1 Te
cutoff 8 4 73 8
n_max 0 0
basis_size 0 0
l_max 1 0 0
ANN 1 0
 1.0
 2.0
 3.0
 4.0
 5.0
 6.0
 7.0
 8.0
 9.0
"""


def test_read_nep_metadata():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(SAMPLE_NEP)
        f.flush()
        result = read_nep(f.name)
    os.unlink(f.name)

    assert result["version"] == 4
    assert result["num_types"] == 1
    assert result["elements"] == ["Te"]
    assert result["rc_radial"] == [8.0]
    assert result["rc_angular"] == [4.0]
    assert result["MN_radial"] == 73
    assert result["MN_angular"] == 8
    assert result["n_max_radial"] == 0
    assert result["n_max_angular"] == 0
    assert result["basis_size_radial"] == 0
    assert result["basis_size_angular"] == 0
    assert result["L_max"] == 1
    assert result["has_q_222"] == 0
    assert result["has_q_1111"] == 0
    assert result["num_neurons"] == 1
    # With GPUMD order: ANN params first (5), then descriptor (2), then q_scaler (2)
    assert jnp.isclose(result["descriptor_params"][0], jnp.float32(6.0))
    assert jnp.isclose(result["descriptor_params"][1], jnp.float32(7.0))


def test_read_write_roundtrip():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(SAMPLE_NEP)
        f.flush()
        original = read_nep(f.name)
    os.unlink(f.name)

    outpath = tempfile.mktemp(suffix='.txt')
    try:
        write_nep(outpath, original)
        reloaded = read_nep(outpath)
    finally:
        if os.path.exists(outpath):
            os.unlink(outpath)

    assert original["version"] == reloaded["version"]
    assert original["num_types"] == reloaded["num_types"]
    assert jnp.allclose(original["descriptor_params"], reloaded["descriptor_params"], atol=1e-6)
    for t in range(original["num_types"]):
        for k in ["w0", "b0", "w1"]:
            assert jnp.allclose(original["ann_params"][t][k], reloaded["ann_params"][t][k], atol=1e-6)
    assert jnp.isclose(original["b1"], reloaded["b1"], atol=1e-6)
    assert jnp.allclose(original["q_scaler"], reloaded["q_scaler"], atol=1e-6)


def test_parameter_count():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(SAMPLE_NEP)
        f.flush()
        result = read_nep(f.name)
    os.unlink(f.name)

    nt = result["num_types"]
    nmr = result["n_max_radial"]
    nma = result["n_max_angular"]
    bsr = result["basis_size_radial"]
    bsa = result["basis_size_angular"]
    dim = result["dim"]
    neurons = result["num_neurons"]

    expected_desc = nt * nt * ((nmr + 1) * (bsr + 1) + (nma + 1) * (bsa + 1))
    expected_ann = (dim + 2) * neurons * nt + 1
    expected_q = dim

    assert len(result["descriptor_params"]) == expected_desc
    total = len(result["descriptor_params"]) + expected_ann + expected_q
    assert total > 0


SAMPLE_NEP_CG = """nep_cg 1 H
soft_repulsion 3.0 0.010364 8 13.0 15.0
cutoff 8 4 73 8
n_max 0 0
basis_size 0 0
l_max 1 0 0
ANN 1 0
 1.0
 2.0
 3.0
 4.0
 5.0
 6.0
 7.0
 8.0
 9.0
"""


def test_read_nep_cg():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(SAMPLE_NEP_CG)
        f.flush()
        result = read_nep(f.name)
    os.unlink(f.name)

    assert result["model_type"] == "nep_cg"
    assert result["version"] == 4
    assert result["soft_repulsion"] is not None
    assert result["soft_repulsion"]["sigma"] == 3.0
    assert result["soft_repulsion"]["epsilon"] == 0.010364
    assert result["soft_repulsion"]["exp"] == 8.0
    assert result["soft_repulsion"]["r_onset"] == 13.0
    assert result["soft_repulsion"]["r_cutoff"] == 15.0


def test_read_write_roundtrip_nep_cg():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(SAMPLE_NEP_CG)
        f.flush()
        original = read_nep(f.name)
    os.unlink(f.name)

    outpath = tempfile.mktemp(suffix='.txt')
    try:
        write_nep(outpath, original)
        reloaded = read_nep(outpath)
    finally:
        if os.path.exists(outpath):
            os.unlink(outpath)

    assert original["model_type"] == reloaded["model_type"] == "nep_cg"
    assert original["soft_repulsion"] == reloaded["soft_repulsion"]
    assert jnp.allclose(original["descriptor_params"], reloaded["descriptor_params"], atol=1e-6)
    for t in range(original["num_types"]):
        for k in ["w0", "b0", "w1"]:
            assert jnp.allclose(original["ann_params"][t][k], reloaded["ann_params"][t][k], atol=1e-6)
    assert jnp.isclose(original["b1"], reloaded["b1"], atol=1e-6)
    assert jnp.allclose(original["q_scaler"], reloaded["q_scaler"], atol=1e-6)
