"""Continuum gates: analytic uniform-field limit, STELLGAP CTOK parity, gap formulas."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fax
from fax.continuum import MU0, generalized_eigh

DATA = Path(__file__).parent / "data"


def test_uniform_field_continuum_is_k_parallel_v_alfven():
    geo = fax.circular_torus(iota=jnp.linspace(0.3, 0.6, 24))
    harmonics = fax.mode_family((1, 2), 4)
    rho = fax.mass_density(1e19, 2.0)
    f = fax.continuum(geo, harmonics, rho)
    k = (harmonics[:, 1] - harmonics[:, 0] * geo.iota[:, None]) / 5.5
    exact = jnp.sort(jnp.abs(k) * 2.7 / jnp.sqrt(MU0 * rho) / (2 * jnp.pi), axis=1)
    np.testing.assert_allclose(f, exact, rtol=1e-9)


def test_matrices_are_hermitian_and_resolution_converged():
    geo = fax.read_boozmn(DATA / "boozmn_ctok.nc").surfaces(slice(10, 90, 10))
    h = fax.mode_family(2, 6)
    K, M = fax.continuum_matrices(geo, h, 3.4e-8)
    np.testing.assert_allclose(K, jnp.conj(jnp.swapaxes(K, -1, -2)), atol=1e-12 * float(jnp.abs(K).max()))
    np.testing.assert_allclose(fax.continuum(geo, h, 3.4e-8, 32, 32), fax.continuum(geo, h, 3.4e-8, 64, 64), rtol=1e-9)


def test_stellgap_ctok_parity():
    """Same boozmn_ctok.nc and STELLGAP inputs (n = 2, m = 0..12, deuterium at 1e19 m^-3)."""
    geo = fax.read_boozmn(DATA / "boozmn_ctok.nc")
    khz = np.asarray(jax.jit(lambda g: fax.continuum(g, fax.mode_family(2, 12), fax.mass_density(1e19, 2.0)))(geo)) / 1e3
    ref = np.load(DATA / "stellgap_ctok_continuum.npz")
    # STELLGAP averages half-mesh rows k, k+1 onto full surface k = 2..ns-1 but labels it
    # k/(ns-1) (stellgap rho = nsurf/nsurf_max), one cell outward of the physical (k-1)/(ns-1).
    s_ref = ref["s"] - 1.0 / 100.0  # boozmn_ctok.nc has ns_b = 101
    inside = (geo.s > s_ref[0]) & (geo.s < s_ref[-1])
    expected = np.stack([np.interp(geo.s[inside], s_ref, col) for col in ref["khz"].T], axis=1)
    errors = np.linalg.norm(khz[inside] - expected, axis=1) / np.linalg.norm(expected, axis=1)
    assert errors.size == 97
    assert np.mean(errors) < 7e-3 and np.percentile(errors, 95) < 7e-3


def test_gaps_and_gap_location():
    center, width = fax.gaps(jnp.array([[1.0, 2.0, 5.0, 5.5], [0.0, 3.0, 3.5, 4.0]]))
    np.testing.assert_allclose(center, [3.5, 1.5])
    np.testing.assert_allclose(width, [3.0, 3.0])
    assert fax.gap_location(1, 1, 1) == pytest.approx((2 / 3, 1 / 3))  # TAE: iota = 2n/(2m+1)
    assert fax.gap_location(4, 1, 0, -1, 4)[0] == pytest.approx(2.0)  # MAE: iota = (2n - nfp)/(2m)


def test_elliptic_two_mode_gap_has_full_width_two_f0_c():
    """``circular_torus(iota=1)``, ellipticity kappa on the (0, 1)-(2, 1) crossing: the edge separation is ``2 f0 |c|``,
    ``c = (kappa^2 - 1) / (2 (kappa^2 + 1))`` the complex Fourier coefficient of ``|grad s|^2 / <|grad s|^2> - 1``, and
    ``|c|^2 (iota dm - n)^2`` of the cost is the squared full width in ``w_A`` units (``w_A = B0 / (R0 sqrt(mu0 rho))``)."""
    base, kappa, harmonics, rho = fax.circular_torus(iota=1.0), 1.001, [[0, 1], [2, 1]], fax.mass_density(1e19, 2.0)
    geo = replace(base, zmns=base.zmns.at[:, 1].multiply(kappa))
    f0 = fax.continuum(base, harmonics, rho)[:, 0]  # both branches cross at f0
    c = (kappa**2 - 1) / (2 * (kappa**2 + 1))
    center, width = fax.gaps(fax.continuum(geo, harmonics, rho))
    np.testing.assert_allclose(width, 2 * f0 * c, rtol=1e-6)
    np.testing.assert_allclose(center, f0, rtol=1e-6)
    f_a = 2.7 / (5.5 * jnp.sqrt(MU0 * rho)) / (2 * jnp.pi)
    assert fax.continuum_cost(geo, (0, 0)) / 2 == pytest.approx(float(jnp.sum((width / f_a) ** 2)), rel=1e-2)  # +-(dm, n) twice


def test_radial_derivative_on_irregular_grids():
    from fax.geometry import _d_ds

    s = jnp.array([0.1, 0.2, 0.8])
    np.testing.assert_allclose(_d_ds(s**2, s), 2 * s, atol=1e-14)
    s = jnp.array([0.05, 0.12, 0.3, 0.41, 0.66, 0.7, 0.98])  # irregular: exact for quadratics, shape-agnostic
    a = jnp.stack([s**2, 1 - 3 * s], axis=1)
    np.testing.assert_allclose(_d_ds(a, s), jnp.stack([2 * s, -3 + 0 * s], axis=1), atol=1e-13)
    err = []
    for n in (16, 32, 64):  # jittered grid, second-order convergence including the ends
        x = jnp.linspace(0.0, 1.0, n) + 0.3 * jnp.sin(7.0 * jnp.arange(n)) / n
        err.append(float(jnp.abs(_d_ds(jnp.sin(3 * x), x) - 3 * jnp.cos(3 * x)).max()))
    assert err[0] / err[1] > 3.5 and err[1] / err[2] > 3.5


@pytest.mark.parametrize("shape", [(20, 3), (3, 20)])
def test_read_boozmn_orientation_follows_named_dimensions(tmp_path, shape):
    from netCDF4 import Dataset

    ns, mn = shape
    rng = np.random.default_rng(0)
    path = tmp_path / "boozmn_synth.nc"
    with Dataset(path, "w") as ds:
        for name, size in (("radius", ns + 2), ("comput_surfs", ns), ("mn_mode", mn), ("mn_modes", mn), ("pack_rad", ns)):
            ds.createDimension(name, size)
        bmnc = rng.normal(size=(ns, mn))
        for name, dims, value, kind in (
            ("bmnc_b", ("pack_rad", "mn_modes"), bmnc, "f8"),
            ("jlist", ("comput_surfs",), np.arange(2, ns + 2), "i4"),
            ("ixm_b", ("mn_mode",), np.arange(mn), "i4"),
            ("ixn_b", ("mn_mode",), np.zeros(mn), "i4"),
            *((name, ("radius",), np.ones(ns + 2), "f8") for name in ("iota_b", "bvco_b", "buco_b")),
        ):
            ds.createVariable(name, kind, dims)[:] = value
    geo = fax.read_boozmn(path)
    assert geo.bmnc.shape == shape and geo.rmnc.shape == shape
    np.testing.assert_array_equal(geo.bmnc, bmnc)


def test_circular_torus_is_traceable_and_frequency_scales_with_B0():
    sum_f = lambda b: fax.continuum(fax.circular_torus(B0=b), [[1.0, 1.0]], 3.4e-8).sum()  # noqa: E731
    assert jax.grad(sum_f)(3.0) == pytest.approx(float(sum_f(3.0) / 3.0), rel=1e-10)  # f ~ G/R = B0
    two = lambda b: fax.continuum(fax.circular_torus(B0=b), [[1.0, 1.0], [2.0, 1.0]], 3.4e-8).sum()  # noqa: E731
    assert jax.jit(two)(3.0) == pytest.approx(two(3.0))  # H = 1 under jit hits an XLA:CPU FFT layout bug in jax 0.11


def test_inputs_are_validated_on_the_host():
    geo, h, rho = fax.circular_torus(), fax.mode_family(1, 8), 3.4e-8
    with pytest.raises(ValueError, match="aliases"):
        fax.continuum(geo, h, rho, 8, 4)  # dm = 8 wraps onto dm = 0
    assert np.isfinite(fax.continuum(fax.circular_torus(epsilon_B=0.1), h, rho, 17, 4)).all()
    with pytest.raises(ValueError, match="aliases"):
        fax.continuum(geo, [[1.0, 0.0], [1.0, 2.0]], rho, 32, 4)
    with pytest.raises(ValueError, match="density"):
        fax.continuum(geo, h, 0.0, 32, 4)
    with pytest.raises(ValueError, match="increasing"):
        fax.surface_grid(replace(geo, s=geo.s[::-1]))
    with pytest.raises(ValueError, match="G \\+ iota I"):
        fax.continuum(replace(geo, G=0 * geo.G), h, rho, 32, 4)
    with pytest.raises(ValueError, match="positive"):
        fax.surface_grid(replace(geo, bmnc=geo.bmnc.at[:, 0].set(-1.0)))


def test_generalized_eigh_matches_explicit_inverse_and_residual():
    rng = np.random.default_rng(1)
    a, b = (rng.normal(size=(3, 9, 9)) + 1j * rng.normal(size=(3, 9, 9)) for _ in range(2))
    K, M = a + a.conj().swapaxes(-1, -2), b @ b.conj().swapaxes(-1, -2) + 9 * np.eye(9)
    lam, x = generalized_eigh(jnp.asarray(K), jnp.asarray(M))
    Linv = np.linalg.inv(np.linalg.cholesky(M))
    ref = np.linalg.eigvalsh(Linv @ K @ Linv.conj().swapaxes(-1, -2))
    np.testing.assert_allclose(lam, ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(K @ x, M @ x * lam[:, None, :], atol=1e-11)
    np.testing.assert_allclose(x.conj().swapaxes(-1, -2) @ M @ x, np.broadcast_to(np.eye(9), (3, 9, 9)), atol=1e-12)


@pytest.fixture
def asymmetric_torus():
    """Elongated, shifted torus with every stellarator-asymmetric harmonic switched on (small, positive Jacobian)."""
    g = fax.circular_torus(s=jnp.linspace(0.1, 0.9, 9), iota=0.4, epsilon_B=0.05)
    z = lambda c: 0.01 * jnp.outer(jnp.linspace(0.5, 1.0, 9), jnp.array(c))  # noqa: E731
    return replace(g, rmns=z([0.0, 1.0, 0.5]), zmnc=z([0.0, 1.0, 0.5]), bmns=z([0.0, 1.0, 0.5]), gmns=z([0.0, 1.0, 0.5]) * g.G[0] / 2.7**2)


def test_asymmetric_geometry_is_hermitian_finite_and_differentiable(asymmetric_torus):
    geo, h = asymmetric_torus, jnp.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
    grid = fax.surface_grid(geo, 16, 8)
    assert all(jnp.isfinite(x).all() for x in jax.tree_util.tree_leaves(grid))
    K, M = fax.continuum_matrices(geo, h, 3.4e-8, 16, 8)
    for A in (K, M):
        np.testing.assert_allclose(A, jnp.conj(jnp.swapaxes(A, -1, -2)), atol=1e-14 * float(jnp.abs(A).max()))
    assert jnp.abs(K.imag).max() > 1e-6 * jnp.abs(K.real).max()  # sine harmonics make the matrices complex
    loss = lambda g: (fax.continuum(g, h, 3.4e-8, 16, 8)[:, :3] ** 2).sum() / 1e12  # noqa: E731
    shifted = lambda name, direction, e: replace(geo, **{name: getattr(geo, name) + e * direction})  # noqa: E731
    grads = jax.grad(loss)(geo)
    for name in ("rmns", "zmnc", "bmns", "gmns"):
        direction = getattr(geo, name) / 0.01 * jnp.array([0.0, 1.0, 0.5])
        fd = (loss(shifted(name, direction, 1e-4)) - loss(shifted(name, direction, -1e-4))) / 2e-4
        assert abs(fd) > 1e-8, name
        assert float(jnp.sum(getattr(grads, name) * direction)) == pytest.approx(float(fd), rel=1e-5), name


def test_booz_xform_output_dictionary_matches_the_file_reader():
    """The traceable converter reads ``gmn_b`` when ``gmnc_b`` is absent and prefers ``gmnc_b`` when both are given."""
    from netCDF4 import Dataset

    ref = fax.read_boozmn(DATA / "boozmn_ctok.nc")
    with Dataset(DATA / "boozmn_ctok.nc") as ds:
        jlist = np.asarray(ds["jlist"][:], dtype=int) - 1
        out = {k: np.asarray(ds[k][:]) for k in ("bmnc_b", "rmnc_b", "zmns_b", "gmn_b", "ixm_b", "ixn_b")}
        out.update({k: np.asarray(ds[k][:])[jlist] for k in ("iota_b", "bvco_b", "buco_b")})
    geo = fax.geometry.from_booz_xform(out, ref.s, ref.nfp)
    for name in ("s", "iota", "G", "I", "xm", "xn", "bmnc", "rmnc", "zmns", "gmnc"):
        np.testing.assert_array_equal(getattr(geo, name), getattr(ref, name))
    assert float(jnp.abs(geo.gmnc).max()) > 0 and not jnp.any(geo.bmns) and geo.nfp == ref.nfp
    both = fax.geometry.from_booz_xform({**out, "gmnc_b": 2 * out["gmn_b"]}, ref.s, ref.nfp)
    np.testing.assert_array_equal(both.gmnc, 2 * ref.gmnc)
