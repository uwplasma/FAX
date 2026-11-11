"""Small-size checks for the GPU CI job (``pytest -m gpu``); they run on whatever backend JAX selects."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fax
from fax.continuum import MU0, generalized_eigh

pytestmark = pytest.mark.gpu


def test_uniform_field_continuum_matches_the_analytic_frequencies():
    geo = fax.circular_torus(iota=jnp.linspace(0.3, 0.6, 24))
    harmonics = fax.mode_family((1, 2), 4)
    rho = fax.mass_density(1e19, 2.0)
    f = fax.continuum(geo, harmonics, rho)
    k = (harmonics[:, 1] - harmonics[:, 0] * geo.iota[:, None]) / 5.5
    exact = jnp.sort(jnp.abs(k) * 2.7 / jnp.sqrt(MU0 * rho) / (2 * jnp.pi), axis=1)
    np.testing.assert_allclose(f, exact, rtol=1e-9)


def test_generalized_eigh_residual_is_at_roundoff():
    rng = np.random.default_rng(1)
    a, b = (rng.normal(size=(3, 9, 9)) + 1j * rng.normal(size=(3, 9, 9)) for _ in range(2))
    K, M = a + a.conj().swapaxes(-1, -2), b @ b.conj().swapaxes(-1, -2) + 9 * np.eye(9)
    lam, x = jax.jit(generalized_eigh)(jnp.asarray(K), jnp.asarray(M))
    np.testing.assert_allclose(K @ x, M @ x * lam[:, None, :], atol=1e-11)


def test_continuum_gradient_is_finite_and_float64():
    geo = fax.circular_torus(iota=jnp.linspace(0.3, 0.6, 8))
    harmonics = fax.mode_family((1, 2), 2)
    rho = fax.mass_density(1e19, 2.0)
    g = jax.grad(lambda r: jnp.sum(fax.continuum(geo, harmonics, r)))(rho)
    assert g.dtype == jnp.float64 and jnp.isfinite(g)


def test_objectives_are_differentiable():
    f = jnp.array([10e3, 125e3, 400e3])
    assert fax.window_risk(f, (50e3, 200e3), 5e3) == pytest.approx(0.5 / 3, rel=1e-3)
    assert jnp.isfinite(jax.grad(lambda x: fax.soft_max(x / 1e3))(f)).all()
