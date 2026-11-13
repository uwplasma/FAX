"""Sound-coupled continuum gates: pure sound branch, Winsor's GAM, the acoustic gap top (docs/theory.md section 2)."""

import jax.numpy as jnp
import numpy as np
import pytest

import fax

R0, B0, P, GAMMA = 5.5, 2.7, 1e3, 5.0 / 3.0
RHO = fax.mass_density(1e19, 2.0)
CS = float(jnp.sqrt(GAMMA * P / RHO))


def torus(eps, q):
    return fax.circular_torus(
        major_radius=R0, minor_radius=eps * R0 or 0.5, B0=B0, iota=1 / q, epsilon_B=eps, s=jnp.array([0.25, 0.5, 0.75])
    )


def test_sound_is_off_by_default_and_changes_nothing():
    geo, h = torus(0.05, 2.0), fax.mode_family(1, 3)
    np.testing.assert_array_equal(fax.continuum(geo, h, RHO), fax.continuum(geo, h, RHO, pressure=None))
    assert fax.continuum(geo, h, RHO, pressure=P).shape == (3, 2 * len(h))


def test_pure_sound_branch_without_curvature():
    q = 2.3  # no rational surface among the harmonics
    geo, h = torus(0.0, q), fax.mode_family((1, 2), 3, 1)
    f = fax.continuum(geo, h, RHO, pressure=P)
    sound = jnp.sort(CS * jnp.abs(h[:, 1] - h[:, 0] / q) / R0 / (2 * jnp.pi))
    exact = jnp.concatenate([jnp.broadcast_to(sound, (3, len(h))), fax.continuum(geo, h, RHO)], 1)  # the two halves decouple
    np.testing.assert_allclose(f, jnp.sort(exact, 1), rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize("q", [1.7, 3.0])
def test_geodesic_acoustic_mode_matches_winsor(q):
    geo = torus(0.02, q)
    f = fax.continuum(geo, fax.mode_family(0, 4, -4), RHO, 48, 16, pressure=P) * 2 * jnp.pi
    gam = jnp.sqrt(2 * GAMMA * P / (RHO * R0**2) * (1 + 1 / (2 * q**2)))
    nearest = jnp.take_along_axis(f, jnp.argmin(jnp.abs(f - gam), 1)[:, None], 1)[:, 0]
    np.testing.assert_allclose(nearest, gam, rtol=1e-3)  # O(beta + eps^2): 3e-4 here
    acoustic = jnp.sort(f, 1)[:, 2] / (CS / (q * R0))  # lowest nonzero branch: m = 1 acoustic, the BAAE gap top c_s/(q R)
    np.testing.assert_allclose(acoustic, 1.0, rtol=1e-3)


def test_sound_inputs_are_validated_and_geodesic_curvature_vanishes_in_a_uniform_field():
    geo, h = torus(0.05, 2.0), fax.mode_family(1, 2)
    for bad, match in ((-1.0, "pressure"), (jnp.ones(2), "pressure")):
        with pytest.raises(ValueError, match=match):
            fax.continuum(geo, h, RHO, pressure=bad)
    with pytest.raises(ValueError, match="gamma"):
        fax.continuum(geo, h, RHO, pressure=P, gamma=0.0)
    K, M = fax.continuum_matrices(torus(0.0, 2.0), h, RHO, pressure=P)
    np.testing.assert_allclose(K[:, : len(h), len(h) :], 0.0, atol=1e-12 * float(jnp.abs(K).max()))
