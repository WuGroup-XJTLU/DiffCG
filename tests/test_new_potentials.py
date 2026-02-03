# SPDX-License-Identifier: MIT
# Tests for LJ, Soft, Morse, and Mie pair potentials.

import jax
import jax.numpy as jnp
import pytest

from diffcg.energy import lj, soft, morse, mie

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# 1. Value tests — hand-calculated from LAMMPS formulas
# ---------------------------------------------------------------------------

class TestLJValues:
    def test_at_sigma(self):
        """At r = sigma, LJ should be zero: 4*eps*(1 - 1) = 0."""
        assert jnp.allclose(lj(jnp.array(1.0), sigma=1.0, epsilon=1.0), 0.0, atol=1e-12)

    def test_at_2_pow_1_6_sigma(self):
        """At r = 2^(1/6)*sigma (minimum), LJ = -epsilon."""
        r_min = 2.0 ** (1.0 / 6.0)
        val = lj(jnp.array(r_min), sigma=1.0, epsilon=1.0)
        assert jnp.allclose(val, -1.0, atol=1e-10)

    def test_known_value(self):
        """At r=1.5, sigma=1, eps=2: 4*2*((1/1.5)^12 - (1/1.5)^6)."""
        r = jnp.array(1.5)
        expected = 4.0 * 2.0 * ((1.0 / 1.5) ** 12 - (1.0 / 1.5) ** 6)
        assert jnp.allclose(lj(r, sigma=1.0, epsilon=2.0), expected, atol=1e-10)


class TestSoftValues:
    def test_at_zero(self):
        """At r=0, soft = A*(1 + cos(0)) = 2*A."""
        val = soft(jnp.array(0.0), A=3.0, r_cutoff=2.0)
        assert jnp.allclose(val, 6.0, atol=1e-12)

    def test_at_cutoff(self):
        """At r=r_cutoff, soft = A*(1 + cos(pi)) = 0."""
        val = soft(jnp.array(2.0), A=3.0, r_cutoff=2.0)
        assert jnp.allclose(val, 0.0, atol=1e-12)

    def test_at_half_cutoff(self):
        """At r=r_cutoff/2, soft = A*(1 + cos(pi/2)) = A."""
        val = soft(jnp.array(1.0), A=3.0, r_cutoff=2.0)
        assert jnp.allclose(val, 3.0, atol=1e-12)


class TestMorseValues:
    def test_at_r0(self):
        """At r=r0, morse = D*(1 - 2) = -D."""
        val = morse(jnp.array(1.0), D=2.0, alpha=1.5, r0=1.0)
        assert jnp.allclose(val, -2.0, atol=1e-12)

    def test_known_value(self):
        """At r=1.5, D=1, alpha=2, r0=1: dexp=exp(-2*0.5)=exp(-1)."""
        r = jnp.array(1.5)
        dexp = jnp.exp(-2.0 * 0.5)
        expected = 1.0 * (dexp * dexp - 2.0 * dexp)
        assert jnp.allclose(morse(r, D=1.0, alpha=2.0, r0=1.0), expected, atol=1e-12)

    def test_large_r(self):
        """At large r, morse -> 0 (dexp -> 0)."""
        val = morse(jnp.array(100.0), D=1.0, alpha=1.0, r0=1.0)
        assert jnp.allclose(val, 0.0, atol=1e-6)


class TestMieValues:
    def test_lj_equivalence(self):
        """Mie with gamma_r=12, gamma_a=6 should equal LJ."""
        r = jnp.linspace(0.9, 3.0, 50)
        lj_vals = lj(r, sigma=1.0, epsilon=1.0)
        mie_vals = mie(r, sigma=1.0, epsilon=1.0, gamma_r=12.0, gamma_a=6.0)
        assert jnp.allclose(lj_vals, mie_vals, atol=1e-10)

    def test_prefactor(self):
        """Check that C = (gamR/(gamR-gamA))*(gamR/gamA)^(gamA/(gamR-gamA))."""
        # For 12-6: C = (12/6)*(12/6)^(6/6) = 2*2 = 4
        r = jnp.array(1.5)
        C = 4.0
        idr = 1.0 / 1.5
        expected = C * 2.0 * (idr ** 12 - idr ** 6)
        assert jnp.allclose(mie(r, sigma=1.0, epsilon=2.0, gamma_r=12.0, gamma_a=6.0), expected, atol=1e-10)

    def test_custom_exponents(self):
        """Mie with gamma_r=9, gamma_a=6."""
        r = jnp.array(1.2)
        gr, ga = 9.0, 6.0
        C = (gr / (gr - ga)) * (gr / ga) ** (ga / (gr - ga))
        idr = 1.0 / 1.2
        expected = C * 1.5 * (idr ** gr - idr ** ga)
        assert jnp.allclose(mie(r, sigma=1.0, epsilon=1.5, gamma_r=gr, gamma_a=ga), expected, atol=1e-10)


# ---------------------------------------------------------------------------
# 2. Force tests — jax.grad vs finite difference
# ---------------------------------------------------------------------------

def _finite_diff_force(fn, r, h=1e-5, **kwargs):
    """Central finite difference of -dU/dr."""
    return -(fn(r + h, **kwargs) - fn(r - h, **kwargs)) / (2.0 * h)


class TestForces:
    @pytest.mark.parametrize("r_val", [0.9, 1.0, 1.5, 2.0])
    def test_lj_force(self, r_val):
        r = jnp.array(r_val)
        grad_force = -jax.grad(lambda r: lj(r, sigma=1.0, epsilon=1.0))(r)
        fd_force = _finite_diff_force(lj, r, sigma=1.0, epsilon=1.0)
        assert jnp.allclose(grad_force, fd_force, rtol=1e-5)

    @pytest.mark.parametrize("r_val", [0.2, 0.5, 0.9])
    def test_soft_force(self, r_val):
        r = jnp.array(r_val)
        grad_force = -jax.grad(lambda r: soft(r, A=2.0, r_cutoff=1.0))(r)
        fd_force = _finite_diff_force(soft, r, A=2.0, r_cutoff=1.0)
        assert jnp.allclose(grad_force, fd_force, rtol=1e-5)

    @pytest.mark.parametrize("r_val", [0.5, 1.0, 1.5, 2.5])
    def test_morse_force(self, r_val):
        r = jnp.array(r_val)
        grad_force = -jax.grad(lambda r: morse(r, D=1.0, alpha=2.0, r0=1.0))(r)
        fd_force = _finite_diff_force(morse, r, D=1.0, alpha=2.0, r0=1.0)
        assert jnp.allclose(grad_force, fd_force, rtol=1e-5)

    @pytest.mark.parametrize("r_val", [0.9, 1.2, 1.8, 2.5])
    def test_mie_force(self, r_val):
        r = jnp.array(r_val)
        grad_force = -jax.grad(lambda r: mie(r, sigma=1.0, epsilon=1.0, gamma_r=10.0, gamma_a=5.0))(r)
        fd_force = _finite_diff_force(mie, r, sigma=1.0, epsilon=1.0, gamma_r=10.0, gamma_a=5.0)
        assert jnp.allclose(grad_force, fd_force, rtol=1e-5)


# ---------------------------------------------------------------------------
# 3. Cutoff tests — energy is zero beyond r_cutoff via smooth cutoff
# ---------------------------------------------------------------------------

class TestCutoff:
    def test_lj_cutoff(self):
        """LJ with smooth cutoff should be zero beyond r_cutoff."""
        from diffcg.energy import _smooth_cutoff_factor
        r = jnp.array(3.0)
        raw = lj(r, sigma=1.0, epsilon=1.0)
        cutoff_val = _smooth_cutoff_factor(r, 2.0, 2.5) * raw
        assert jnp.allclose(cutoff_val, 0.0, atol=1e-12)

    def test_soft_cutoff(self):
        """Soft potential is naturally zero at r_cutoff."""
        val = soft(jnp.array(2.0), A=5.0, r_cutoff=2.0)
        assert jnp.allclose(val, 0.0, atol=1e-12)

    def test_morse_cutoff(self):
        from diffcg.energy import _smooth_cutoff_factor
        r = jnp.array(3.0)
        raw = morse(r, D=1.0, alpha=1.5, r0=1.0)
        cutoff_val = _smooth_cutoff_factor(r, 2.0, 2.5) * raw
        assert jnp.allclose(cutoff_val, 0.0, atol=1e-12)

    def test_mie_cutoff(self):
        from diffcg.energy import _smooth_cutoff_factor
        r = jnp.array(3.0)
        raw = mie(r, sigma=1.0, epsilon=1.0, gamma_r=12.0, gamma_a=6.0)
        cutoff_val = _smooth_cutoff_factor(r, 2.0, 2.5) * raw
        assert jnp.allclose(cutoff_val, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 4. Mie → LJ equivalence (vectorized)
# ---------------------------------------------------------------------------

class TestMieLJEquivalence:
    def test_array_equivalence(self):
        """Mie(12,6) == LJ over a range of distances."""
        r = jnp.linspace(0.8, 4.0, 200)
        lj_vals = lj(r, sigma=1.0, epsilon=2.5)
        mie_vals = mie(r, sigma=1.0, epsilon=2.5, gamma_r=12.0, gamma_a=6.0)
        assert jnp.allclose(lj_vals, mie_vals, atol=1e-10)

    def test_force_equivalence(self):
        """Forces from Mie(12,6) == forces from LJ."""
        r = jnp.array(1.3)
        lj_force = -jax.grad(lambda r: lj(r, sigma=1.0, epsilon=1.0))(r)
        mie_force = -jax.grad(lambda r: mie(r, sigma=1.0, epsilon=1.0, gamma_r=12.0, gamma_a=6.0))(r)
        assert jnp.allclose(lj_force, mie_force, atol=1e-10)
