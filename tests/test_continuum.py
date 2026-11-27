"""Continuum gates: analytic uniform-field limit, STELLGAP CTOK parity, gap formulas."""

import functools
import warnings
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from netCDF4 import Dataset

import fax
from fax.continuum import MU0, generalized_eigh, spectral_blocks
from fax.eigen import dense, global_matrices
from fax.geometry import symmetric

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


@functools.cache
def _ctok_khz():
    """FAX continuum of the packet ``boozmn_ctok.nc`` (n = 2, m = 0..12, deuterium at 1e19 m^-3), kHz."""
    with pytest.warns(UserWarning, match="outer half surface .* is not in the file"):  # jlist = 2..100 of ns_b = 101
        geo = fax.read_boozmn(DATA / "boozmn_ctok.nc")
    rho = fax.mass_density(1e19, 2 * 1.00727647)  # two proton masses; STELLGAP uses 2 x 1.67e-27 kg (plasma.dat)
    return np.asarray(geo.s), np.asarray(jax.jit(lambda g: fax.continuum(g, fax.mode_family(2, 12), rho))(geo)) / 1e3


def _stellgap_ctok_errors(name):
    """Relative L2 error of the 13 sorted branches per surface against a STELLGAP CTOK reference, on the 97 common surfaces."""
    s, khz = _ctok_khz()
    ref = np.load(DATA / name)
    # STELLGAP averages half-mesh rows k, k+1 onto full surface k = 2..ns-1 but labels it
    # k/(ns-1) (stellgap rho = nsurf/nsurf_max), one cell outward of the physical (k-1)/(ns-1).
    s_ref = ref["s"] - 1.0 / 100.0  # boozmn_ctok.nc has ns_b = 101
    inside = (s > s_ref[0]) & (s < s_ref[-1])
    expected = np.stack([np.interp(s[inside], s_ref, col) for col in ref["khz"].T], axis=1)
    return s[inside], np.linalg.norm(khz[inside] - expected, axis=1) / np.linalg.norm(expected, axis=1)


def test_stellgap_ctok_parity():
    """Against STELLGAP rebuilt with the outer Boozer half surface restored from the packet's WOUT (plan step 1.6:
    measured maximum 0.70%, mean 0.42%, 95th percentile 0.67%; gate maximum 1%, mean 0.5%)."""
    s, errors = _stellgap_ctok_errors("stellgap_ctok_edge_continuum.npz")
    assert errors.size == 97
    assert np.max(errors) < 1e-2 and np.mean(errors) < 5e-3


def test_archived_stellgap_ctok_fails_at_the_edge_without_the_outer_half_surface(tmp_path):
    """The packet STELLGAP output was built from a metric whose outer surface averages the last half surface with zeros
    (``metric_element_create_ver8.46.f``, lines 9-17): FAX differs by 5.20% at s = 0.965 and 16.45% at s = 0.975 (plan step 1.6),
    and the two references agree inside, where the restored surface does not enter."""
    s, errors = _stellgap_ctok_errors("stellgap_ctok_continuum.npz")
    np.testing.assert_allclose(s[-2:], [0.965, 0.975])
    np.testing.assert_allclose(errors[-2:], [0.0520, 0.1645], rtol=1e-2)
    old, new = (np.load(DATA / f) for f in ("stellgap_ctok_continuum.npz", "stellgap_ctok_edge_continuum.npz"))
    np.testing.assert_array_equal(old["s"], new["s"])
    drift = np.linalg.norm(old["khz"] - new["khz"], axis=1) / np.linalg.norm(new["khz"], axis=1)
    assert drift[old["s"] < 0.85].max() < 1e-6 and drift.max() > 0.1
    complete = tmp_path / "boozmn_ctok.nc"  # the same file with ns_b = 100: its last half surface is present
    complete.write_bytes((DATA / "boozmn_ctok.nc").read_bytes())
    with Dataset(complete, "a") as ds:
        ds["ns_b"].assignValue(100)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fax.read_boozmn(complete)


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
    assert fax.continuum_cost(geo, (0, 0)) / 2 == pytest.approx(jnp.sum((width / f_a) ** 2), rel=1e-2)  # +-(dm, n) twice


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
    assert jax.grad(sum_f)(3.0) == pytest.approx(sum_f(3.0) / 3.0, rel=1e-10)  # f ~ G/R = B0
    two = lambda b: fax.continuum(fax.circular_torus(B0=b), [[1.0, 1.0], [2.0, 1.0]], 3.4e-8).sum()  # noqa: E731
    assert jax.jit(two)(3.0) == pytest.approx(two(3.0))  # H = 1 under jit hits an XLA:CPU FFT layout bug in jax 0.11


def test_inputs_are_validated_on_the_host():
    geo, h, rho = fax.circular_torus(), fax.mode_family(1, 8), 3.4e-8
    with pytest.raises(ValueError, match="aliases"):
        fax.continuum(geo, h, rho, 8, 4)  # dm = 8 wraps onto dm = 0
    assert np.isfinite(fax.continuum(fax.circular_torus(epsilon_B=0.1), h, rho, 17, 4)).all()
    with pytest.raises(ValueError, match="aliases"):
        fax.continuum(geo, [[1.0, 0.0], [1.0, 2.0]], rho, 32, 4, parity="single")  # by default n = 0 and 2 are separate blocks
    with pytest.raises(ValueError, match="density"):
        fax.continuum(geo, h, 0.0, 32, 4)
    with pytest.raises(ValueError, match="increasing"):
        fax.surface_grid(replace(geo, s=geo.s[::-1]))
    with pytest.raises(ValueError, match="G \\+ iota I"):
        fax.continuum(replace(geo, G=0 * geo.G), h, rho, 32, 4)
    with pytest.raises(ValueError, match="positive"):
        fax.surface_grid(replace(geo, bmnc=geo.bmnc.at[:, 0].set(-1.0)))


@pytest.mark.parametrize("solver", [fax.continuum, fax.continuum_matrices, global_matrices])
@pytest.mark.parametrize(
    ("harmonics", "density", "iota", "match"),
    [
        ([[3.0, 2.0], [3.0, 2.0], [4.0, 2.0]], 3.4e-8, 0.4, "distinct"),
        ([[1.5, 1.0]], 3.4e-8, 0.4, "integers"),
        (np.zeros((0, 2)), 3.4e-8, 0.4, "non-empty"),
        ([[1.0, 1.0]], np.inf, 0.4, "finite"),
        ([[1.0, 1.0]], np.ones(3), 0.4, "one value per surface"),
        ([[1.0, 1.0]], 3.4e-8, np.nan, "iota"),
    ],
)
def test_bad_inputs_are_rejected(solver, harmonics, density, iota, match):
    geo = fax.circular_torus(s=(jnp.arange(8) + 0.5) / 8, iota=iota)
    with pytest.raises(ValueError, match=match):
        solver(geo, jnp.asarray(harmonics), density)


def test_global_matrices_validate_density_and_surfaces():
    geo, h = fax.circular_torus(s=(jnp.arange(8) + 0.5) / 8), fax.mode_family(1, 3)
    with pytest.raises(ValueError, match="density"):
        global_matrices(geo, h, 0.0, 16, 4)
    for s in (geo.s.at[0].set(0.0), geo.s.at[-1].set(1.0)):
        with pytest.raises(ValueError, match="strictly inside"):
            global_matrices(replace(geo, s=s), h, 3.4e-8, 16, 4)
    with pytest.raises(ValueError, match="distinct"):
        global_matrices(geo, jnp.concatenate([h, h[:1]]), 3.4e-8, 16, 4)
    assert jnp.all(jnp.isfinite(dense(global_matrices(geo, h, 3.4e-8, 16, 4)[0])))


def test_mass_density_is_in_atomic_mass_units():
    assert float(fax.mass_density(1e19, 2.01410178)) == pytest.approx(1e19 * 2.01410178 * 1.66053906660e-27, rel=1e-12)


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


def test_symmetric_geometry_is_solved_in_real_arithmetic(asymmetric_torus):
    """A stellarator-symmetric geometry, on the host or traced, gives real matrices equal to the complex assembly."""
    from fax.continuum import _continuum_matrices
    from fax.eigen import _global_matrices

    geo, h, rho = fax.circular_torus(s=jnp.linspace(0.1, 0.9, 9), iota=0.4, epsilon_B=0.05), fax.mode_family(1, 3), 3.4e-8
    assert symmetric(geo) and not symmetric(asymmetric_torus)
    grid = fax.surface_grid(geo, 16, 8)
    cases = (
        (lambda g: fax.continuum_matrices(g, h, rho, 16, 8), _continuum_matrices(geo, grid, h, rho, None, 5 / 3, 1)),
        (lambda g: global_matrices(g, h, rho, 16, 8), _global_matrices(geo, grid, h, rho, False, 1)),
    )
    for f, complex_path in cases:
        assert all(a.dtype == jnp.complex128 for a in jax.tree_util.tree_leaves(complex_path))
        for got in (f(geo), jax.jit(f)(geo)):
            assert all(a.dtype == jnp.float64 for a in jax.tree_util.tree_leaves(got))
            for a, b in zip(jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(complex_path), strict=True):
                np.testing.assert_allclose(a, b, atol=1e-15 * float(jnp.abs(b).max()))
    modes = lambda g, **t: fax.eigenmodes(g, h, rho, ntheta=16, nzeta=8, **t).omega2  # noqa: E731
    for target in ({}, {"target_hz": 2e5, "count": 4}):
        np.testing.assert_allclose(modes(geo, **target), jax.jit(lambda g, t=target: modes(g, **t))(geo), rtol=1e-11)


def test_parity_sectors_on_a_traced_geometry(asymmetric_torus):
    """Symmetry is a static fact of the geometry, so a traced symmetric geometry keeps its sectors and an asymmetric one is refused."""
    geo, h, rho = fax.circular_torus(s=jnp.linspace(0.1, 0.9, 9), iota=0.4, epsilon_B=0.05), fax.mode_family(1, 3), 3.4e-8
    assert fax.continuum(geo, h, rho, 16, 8, parity=1).shape == (9, 4)
    for f in (lambda g: fax.continuum(g, h, rho, 16, 8, parity=1), lambda g: global_matrices(g, h, rho, 16, 8, parity=-1)):
        for a, b in zip(jax.tree_util.tree_leaves(f(geo)), jax.tree_util.tree_leaves(jax.jit(f)(geo)), strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-15 * float(jnp.abs(a).max()))
        with pytest.raises(ValueError, match="stellarator-symmetric"):
            jax.jit(f)(asymmetric_torus)


def test_traced_geometry_keeps_the_symmetry_facts_of_its_constructor():
    """The facts that choose the real path, the coupled families and the one-period average are static fields set by each
    constructor, so eager and compiled calls agree; ``dataclasses.replace`` under ``jax.jit`` cannot inherit facts its new
    arrays may break. The ``nfp = 1`` field with an ``n = 1`` harmonic of ``R`` and ``Z`` is three-dimensional: ``n = 0``
    and ``n = 1`` couple, so the default basis is one self-conjugate family: 6 cosine and 6 sine branches per surface."""
    from fax.geometry import field_period

    s, h, rho = jnp.linspace(0.1, 0.9, 9), fax.mode_family((0, 1), 3, 1), 3.4e-8
    torus = fax.circular_torus(s=s, iota=0.4, epsilon_B=0.1)
    helical = replace(torus, xn=torus.xn.at[2].set(1.0))
    facts = lambda g: (g.axisymmetric, g.stellsym, g.nfp_periodic)  # noqa: E731
    assert facts(torus) == (True, True, True) and facts(helical) == (False, True, True)
    assert facts(replace(helical, nfp=2)) == (False, True, False) and not replace(torus, bmns=torus.bmnc).stellsym
    eager, compiled = fax.continuum(helical, h, rho, 16, 8), jax.jit(lambda g: fax.continuum(g, h, rho, 16, 8))(helical)
    assert eager.shape == compiled.shape == (9, 12)
    np.testing.assert_allclose(compiled, eager, rtol=1e-12)
    modes = lambda g: fax.eigenmodes(g, h, rho, ntheta=16, nzeta=8).omega2  # noqa: E731
    np.testing.assert_allclose(jax.jit(modes)(helical), modes(helical), rtol=1e-10)
    added = lambda g, **p: fax.continuum(replace(g, bmns=g.bmns.at[:, 1].add(5e-3)), h, rho, 16, 8, **p)  # noqa: E731
    assert added(torus).shape == (9, 9)  # eager: n = 0 on both signs (6) and n = 1 alone (3)
    with pytest.raises(ValueError, match="axisymmetry"):  # traced replace: unknown axisymmetry would decide the blocks
        jax.jit(added)(torus)
    single = lambda g: added(g, parity="single")  # noqa: E731
    np.testing.assert_allclose(jax.jit(single)(torus), single(torus), rtol=1e-12)  # no stale stellsym under jit
    zeta_points = lambda f, g: jax.jit(lambda g: fax.surface_grid(f(g), 16, 8, field_period(f(g), 8)).B)(g).shape[-1]  # noqa: E731
    two = lambda g: replace(g, nfp=2)  # noqa: E731
    assert zeta_points(lambda g: g, two(torus)) == 4 and zeta_points(two, torus) == zeta_points(two, helical) == 8  # unknown: full turn
    for f in (lambda g: field_period(two(g), 8), lambda g: zeta_points(lambda x: x, two(g))):
        with pytest.raises(ValueError, match="multiple of nfp"):
            f(helical)


def test_constructors_state_the_facts_under_jit():
    """Inside ``jax.jit`` every array of a constructor is traced; the facts come from its host inputs or are stated, so they
    equal the eager ones, and the design path (``from_booz_xform``) can solve a parity sector."""
    from fax.geometry import Geometry, field_period, from_booz_xform

    s, h, rho = jnp.linspace(0.1, 0.9, 9), fax.mode_family((0, 1), 3, 1), 3.4e-8
    facts = lambda g: (g.axisymmetric, g.stellsym, g.nfp_periodic)  # noqa: E731
    torus = lambda e, nfp=1: fax.circular_torus(s=s, iota=0.4, epsilon_B=e, nfp=nfp)  # noqa: E731
    raw = {k: np.asarray(getattr(torus(0.1), k)) for k in ("s", "xm", "xn", "bmnc", "rmnc", "zmns", "gmnc", "iota", "G", "I")}
    out = {"ixm_b": raw["xm"], "ixn_b": raw["xn"], "iota_b": raw["iota"], "bvco_b": raw["G"], "buco_b": raw["I"]}
    out.update({f"{k}_b": raw[k] for k in ("bmnc", "rmnc", "zmns", "gmnc")})
    builders = (  # host inputs inside jit, as in the design path where vmex gives NumPy mode lists
        lambda: torus(jnp.asarray(0.1)),
        lambda: Geometry.from_arrays(1, **raw),
        lambda: from_booz_xform(out, s, 1),
        lambda: from_booz_xform({**out, "bmns_b": jnp.zeros_like(out["bmnc_b"])}, s, 1, stellsym=True, axisymmetric=True),
    )
    for build in builders:
        assert facts(jax.jit(build)()) == facts(build()) == (True, True, True)
    groups = lambda e: tuple(map(tuple, fax.mode_families(torus(e), h)))  # noqa: E731
    assert groups(0.1) == ((0, 1, 2), (3, 4, 5))  # axisymmetric: one family per n
    assert jax.jit(lambda e: jnp.asarray([len(g) for g in groups(e)]))(0.1).tolist() == [3, 3]
    period = jax.jit(lambda e: fax.surface_grid(torus(e, 2), 16, 8, field_period(torus(e, 2), 8)).B)(0.1)
    assert period.shape[-1] == 4  # nfp_periodic stated by circular_torus
    sector = lambda e: fax.continuum(torus(e), h, rho, 16, 8, parity=1)[:, 0].sum()  # noqa: E731
    fd = (sector(0.1 + 1e-5) - sector(0.1 - 1e-5)) / 2e-5
    for grad in (jax.grad(sector), jax.jit(jax.grad(sector))):
        assert float(grad(0.1)) == pytest.approx(float(fd), rel=1e-6)
    np.testing.assert_allclose(jax.vmap(sector)(jnp.array([0.05, 0.1])), [sector(0.05), sector(0.1)], rtol=1e-12)
    design = lambda o: fax.continuum(from_booz_xform(o, s, 1), h, rho, 16, 8, parity=1)  # noqa: E731
    np.testing.assert_allclose(jax.jit(design)(out), design(out), rtol=1e-12)


def test_one_period_average_is_refused_when_a_toroidal_number_is_not_a_multiple_of_nfp():
    """``nfp = 2`` with an ``n = 1`` harmonic of ``R`` and ``Z``: the field has no half-turn period, so the averages over one
    period drop the ``dn = 1`` couplings. The host refuses the inconsistent geometry; a traced ``xn`` cannot be checked and
    gets the full turn."""
    from fax.continuum import _continuum_matrices
    from fax.eigen import _global_matrices
    from fax.geometry import field_period, surface_grid

    geo = fax.circular_torus(s=jnp.linspace(0.1, 0.9, 9), iota=0.4, epsilon_B=0.1, nfp=2)
    helical, h, rho = replace(geo, xn=geo.xn.at[2].set(1.0)), fax.mode_family((0, 1), 3, 1), 3.4e-8
    assert field_period(geo, 8) == 2 and field_period(geo, 7) == 1
    full, half = surface_grid(helical, 16, 8), surface_grid(helical, 16, 8, 2)
    ref = _continuum_matrices(helical, full, h, rho, None, 5 / 3, 1)
    wrong = _continuum_matrices(helical, half, h, rho, None, 5 / 3, 2)
    assert jnp.abs(wrong[0] - ref[0]).max() > 1e-3 * jnp.abs(ref[0]).max()  # the half-turn average is wrong here
    cases = (
        (lambda g: fax.continuum_matrices(g, h, rho, 16, 8), ref),
        (lambda g: global_matrices(g, h, rho, 16, 8), _global_matrices(helical, full, h, rho, False, 1)),
    )
    for f, want in cases:
        for run in (f, jax.jit(f)):  # the static nfp_periodic refuses the traced call too
            with pytest.raises(ValueError, match="multiple of nfp"):
                run(helical)
        traced_xn = lambda x, f=f: f(replace(helical, xn=x))  # noqa: E731 - a traced xn cannot be checked: full turn
        for a, b in zip(jax.tree_util.tree_leaves(jax.jit(traced_xn)(helical.xn)), jax.tree_util.tree_leaves(want), strict=True):
            np.testing.assert_allclose(a, b, atol=1e-12 * float(jnp.abs(b).max()))
    for nzeta in (8, 7):
        with pytest.raises(ValueError, match="multiple of nfp"):
            field_period(helical, nzeta)


def test_default_basis_is_both_signs_split_into_independent_blocks(asymmetric_torus):
    """n = 0 in a tokamak couples (m, 0) to (-m', 0): the default is the cos and sin sectors (symmetric geometry) or one
    complex pencil on both signs (asymmetric or traced), the spectrum of the two-sign list either way; n != 0 is unchanged."""
    sym = fax.circular_torus(s=jnp.linspace(0.1, 0.9, 9), iota=0.4, epsilon_B=0.1)
    h, both, rho = fax.mode_family(0, 4, 1), np.concatenate([fax.mode_family(0, 4, 1), -fax.mode_family(0, 4, 1)]), 3.4e-8
    assert [(b.parity, len(b.rows)) for b in spectral_blocks(sym, h)] == [(1, 4), (-1, 4)]
    assert [(b.parity, len(b.rows)) for b in spectral_blocks(asymmetric_torus, h)] == [(None, 8)]
    assert [b.parity for b in spectral_blocks(sym, fax.mode_family(1, 4))] == [None]  # n != 0 alone: the single-sign list
    pair = spectral_blocks(sym, fax.mode_family((1, -1), 3))  # n = 1 with n = -1: one pencil, (0, -1) is (0, 1) conjugated
    assert [(b.parity, len(b.rows), int((b.sign < 0).sum())) for b in pair] == [(None, 7, 3)]
    for geo in (sym, asymmetric_torus):
        two = fax.continuum(geo, both, rho, parity="single")
        np.testing.assert_allclose(fax.continuum(geo, h, rho), two, rtol=1e-10)
        np.testing.assert_allclose(jax.jit(lambda g: fax.continuum(g, h, rho))(geo), two, rtol=1e-10)  # traced: one pencil
        assert np.abs(fax.continuum(geo, h, rho, parity="single")[:, :4] - two[:, ::2]).max() > 1e-3 * two.max()
        ref = fax.eigenmodes(geo, both, rho, parity="single")
        full = fax.eigenmodes(geo, h, rho)
        np.testing.assert_allclose(full.omega2, ref.omega2, rtol=1e-10)
        mid = float(np.median(ref.frequencies_hz))
        near = fax.eigenmodes(geo, h, rho, target_hz=mid, count=6)
        np.testing.assert_allclose(near.omega2, np.sort(ref.omega2[np.argsort(np.abs(ref.omega2 - (2 * np.pi * mid) ** 2))[:6]]), rtol=1e-9)
        assert near.status.converged and full.structures.shape == (len(ref.omega2), 9, 4)
        F, G = (dense(A) for A in global_matrices(geo, both, rho))  # every mode, mapped to the listed harmonics,
        y = np.concatenate([full.structures, np.conj(full.structures)], -1).reshape(len(full.omega2), -1)  # is an eigenvector
        r = np.linalg.norm(y @ F.T - full.omega2[:, None] * (y @ G.T), axis=1) / np.linalg.norm(y @ F.T, axis=1)
        assert r.max() < 1e-8
    f = lambda r, g=sym: fax.frequency_hz(g, h, r, 2, 3)  # noqa: E731
    np.testing.assert_allclose(jax.grad(f)(rho), -f(rho) / (2 * rho), rtol=1e-8)
    loss = lambda e: fax.frequency_hz(fax.circular_torus(s=jnp.linspace(0.1, 0.9, 9), iota=0.4, epsilon_B=e), h, rho, 2, 3)  # noqa: E731
    assert jax.grad(loss)(0.1) == pytest.approx((loss(0.1 + 1e-5) - loss(0.1 - 1e-5)) / 2e-5, rel=1e-5)


def test_unknown_axisymmetry_refuses_a_toroidal_family_under_jit():
    """With nfp = 1 axisymmetry decides whether (m, n) couples to (-m', -n) through (m + m', 2 n): eagerly a tokamak n = 1 list
    is one block of 4 branches, a three-dimensional field would give 8. A geometry rebuilt inside ``jax.jit`` does not know
    which, so the default basis is refused (as for several |n|); an n = 0 list does not depend on it and is solved."""
    geo, rho = fax.circular_torus(s=jnp.linspace(0.1, 0.9, 9), iota=0.4, epsilon_B=0.1), 3.4e-8
    for h, ok in ((fax.mode_family(1, 3), False), (fax.mode_family(0, 3, 1), True)):
        f = lambda b, h=h: fax.continuum(replace(geo, bmnc=b), h, rho, 16, 8)  # noqa: E731
        if ok:
            np.testing.assert_allclose(jax.jit(f)(geo.bmnc), f(geo.bmnc), rtol=1e-12)
        else:
            assert f(geo.bmnc).shape == (9, 4)
            with pytest.raises(ValueError, match="axisymmetry"):
                jax.jit(f)(geo.bmnc)


def test_count_above_the_basis_size_raises():
    """A targeted solve asks for at most as many modes as the basis has unknowns (surfaces x harmonics, summed over the
    blocks), as for one pencil: 9 x 4 in each sector of the n = 0 family, 9 x 4 for the single n = 1 family."""
    geo, rho = fax.circular_torus(s=jnp.linspace(0.1, 0.9, 9), iota=0.4, epsilon_B=0.1), 3.4e-8
    for h, size in ((fax.mode_family(0, 4, 1), 72), (fax.mode_family(1, 3), 36)):
        with pytest.raises(ValueError, match="count"):
            fax.eigenmodes(geo, h, rho, target_hz=2e5, count=size + 1)


def stellarator(nfp, epsilon_B=0.1, sine=0.0):
    """Circular torus whose ``m = 2`` harmonic of ``|B|``, ``R`` and ``Z`` turns into ``(2, nfp)``: a stellarator-symmetric
    three-dimensional field, or an asymmetric one with a ``sin(theta)`` harmonic of ``|B|`` of amplitude ``sine`` (T)."""
    g = fax.circular_torus(s=jnp.linspace(0.1, 0.9, 9), iota=0.4, epsilon_B=epsilon_B, nfp=nfp)
    return replace(g, xn=g.xn.at[2].set(nfp), bmnc=g.bmnc.at[:, 2].set(-0.2 * g.s), bmns=g.bmns.at[:, 1].set(sine))


def test_sectors_give_the_full_basis_and_an_asymmetric_field_takes_the_complex_path():
    """N = 1 at nfp = 2 is self-conjugate: on a symmetric field the cosine and sine sectors together are the spectrum of the
    two-sign list (theory, section 2) to 1e-10. A sine harmonic of |B| (1e-2 T) breaks the symmetry: the default is then one
    complex pencil on both signs, again the two-sign spectrum, and it differs from the sector solve of the symmetric field."""
    h, rho = fax.mode_family((1, 3), 3), 3.4e-8
    both, sym, asym = np.concatenate([h, -h]), stellarator(2), stellarator(2, sine=1e-2)
    full = fax.continuum(sym, both, rho, 16, 16, parity="single")
    sectors = np.sort(np.concatenate([fax.continuum(sym, h, rho, 16, 16, parity=p) for p in (1, -1)], 1), 1)
    np.testing.assert_allclose(sectors, full, rtol=1e-10)
    np.testing.assert_allclose(fax.continuum(sym, h, rho, 16, 16), full, rtol=1e-10)
    assert [(b.parity, len(b.rows)) for b in spectral_blocks(asym, h)] == [(None, 16)]
    complex_path = fax.continuum(asym, h, rho, 16, 16)
    np.testing.assert_allclose(complex_path, fax.continuum(asym, both, rho, 16, 16, parity="single"), rtol=1e-10)
    assert np.abs(complex_path / full - 1).max() > 1e-5
    with pytest.raises(ValueError, match="stellarator-symmetric"):
        fax.continuum(asym, h, rho, 16, 16, parity=1)


def test_completion_leaves_a_family_that_is_not_self_conjugate_unchanged():
    """N = 1 at nfp = 5 has 2 N != 0 (mod 5): (m, n) never couples to (-m', -n'), so the default basis is the list as given
    and the two-sign list only doubles every branch (theory, section 2), to 1e-12. The 40-point grid is one period of
    8 points; a 32-point full turn aliases the products with the (2, 5) harmonic at 2e-7."""
    h, rho = fax.mode_family((1, 6, -4), 3), 3.4e-8
    geo = stellarator(5)
    single = fax.continuum(geo, h, rho, 16, 40, parity="single")
    np.testing.assert_allclose(fax.continuum(geo, h, rho, 16, 40), single, rtol=1e-12)
    doubled = fax.continuum(geo, np.concatenate([h, -h]), rho, 16, 40, parity="single")
    for copy in (doubled[:, ::2], doubled[:, 1::2]):
        np.testing.assert_allclose(copy, single, rtol=1e-12)


@pytest.mark.parametrize("sine, parity", [(0.0, 1), (0.0, -1), (0.0, None), (1e-3, None)])
def test_dense_and_targeted_values_and_reverse_derivatives_agree(sine, parity):
    """The real path in each sector, both sectors merged, and the complex two-sign pencil of an asymmetric field: the
    targeted solve of the most isolated interior eigenvalue equals the dense one, and so does its reverse-mode derivative in
    epsilon_B (both exact first derivatives of one simple eigenvalue)."""
    h, rho, grid = fax.mode_family((1, 3), 3), 3.4e-8, dict(ntheta=16, nzeta=16, parity=parity)
    w2 = np.asarray(fax.eigenmodes(stellarator(2, sine=sine), h, rho, **grid).omega2)
    k = 1 + int(np.argmax(np.minimum(np.diff(w2)[:-1], np.diff(w2)[1:])[: len(w2) // 2]))
    target = float(np.sqrt(w2[k] + 0.1 * (w2[k + 1] - w2[k])) / (2 * np.pi))  # off the eigenvalue: the shift stays regular
    dense_f = lambda e: fax.frequency_hz(stellarator(2, e, sine), h, rho, k, k + 1, **grid)  # noqa: E731
    near_f = lambda e: fax.frequency_hz(stellarator(2, e, sine), h, rho, 0, 1, target_hz=target, count=1, **grid)  # noqa: E731
    assert near_f(0.1) == pytest.approx(dense_f(0.1), rel=1e-10)
    assert jax.grad(near_f)(0.1) == pytest.approx(jax.grad(dense_f)(0.1), rel=1e-7)


def test_continuum_vanishes_at_the_rational_surface_and_families_match_a_full_torus_solve():
    """Spong et al. (2003): every row of the continuum stiffness carries ``m iota - n``, so at ``m iota = n`` (here (5, 2) at
    iota = 0.4) the spectrum has an exact zero on every surface, and none at iota = 0.41. The families ``n mod nfp`` do not
    couple: their blocks give the spectrum of one full-torus solve of every listed harmonic on both signs (33 toroidal
    points, not a multiple of nfp, so the whole turn), to 1e-12."""
    rho, h = 3.4e-8, fax.mode_family((0, 2), 6, 1)
    f = np.asarray(fax.continuum(stellarator(2), h, rho))
    off = fax.circular_torus(s=jnp.linspace(0.1, 0.9, 9), iota=0.41, epsilon_B=0.1, nfp=2)
    assert (f.min(1) < 1e-6 * f.max()).all() and (np.asarray(fax.continuum(off, h, rho)).min(1) > 1e-2 * f.max()).all()
    h = fax.mode_family((0, 1, 2, 3), 3, 1)
    full = fax.continuum(stellarator(2), np.unique(np.concatenate([h, -h]), axis=0), rho, 32, 33, parity="single")
    np.testing.assert_allclose(fax.continuum(stellarator(2), h, rho, 32, 32), full, rtol=1e-12)


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


# Sound-coupled continuum: pure sound branch, Winsor's GAM, the acoustic gap top (docs/theory.md section 2).
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


@pytest.mark.parametrize("p", [P, 2e5])
def test_pure_sound_branch_is_the_slow_wave_without_curvature(p):
    """Cheng & Chance (1986) eq. 13b with K_s = 0: w^2 = k_par^2 c_s^2 B^2/(B^2 + mu0 gamma p), the tube speed."""
    q = 2.3  # no rational surface among the harmonics
    geo, h = torus(0.0, q), fax.mode_family((1, 2), 3, 1)
    f = fax.continuum(geo, h, RHO, pressure=p)
    cs2 = GAMMA * p / RHO
    tube = jnp.sqrt(cs2 / (1 + 4e-7 * jnp.pi * GAMMA * p / B0**2))
    sound = jnp.sort(tube * jnp.abs(h[:, 1] - h[:, 0] / q) / R0 / (2 * jnp.pi))
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
