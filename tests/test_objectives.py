"""Mode classification, smooth objectives, rejection of unconverged selections and fixed-geometry derivatives (no vmex)."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fax
from fax.objectives import Design, _reject, classify_modes

DATA = Path(__file__).parent / "data"
HARMONICS, RHO = fax.mode_family(6, 13, 9), fax.mass_density(2e19, 1.00727647)
OPTIONS = dict(ntheta=16, nzeta=4, target_hz=6.46e4, count=4, iters=100)


def konies(N=40):
    s = (jnp.arange(N) + 0.5) / N
    return fax.circular_torus(s=s, major_radius=10.0, B0=3.0, iota=1 / (1.71 + 0.16 * s), epsilon_B=0.1)


def design(**options):
    return Design("", (), (), HARMONICS, RHO, 24, {**OPTIONS, **options})


def asymmetric(N=9):
    """Torus with every stellarator-asymmetric harmonic on, and a tangent that moves all of them and ``iota``, ``B``."""
    g = fax.circular_torus(s=jnp.linspace(0.1, 0.9, N), iota=0.4, epsilon_B=0.05, major_radius=10.0, B0=3.0)
    z = lambda c: 0.01 * jnp.outer(jnp.linspace(0.5, 1.0, N), jnp.array(c))  # noqa: E731
    geo = replace(g, rmns=z([0.0, 1.0, 0.5]), zmnc=z([0.0, 1.0, 0.5]), bmns=z([0.0, 1.0, 0.5]), gmns=z([0.0, 1.0, 0.5]) * g.G[0] / 9.0)
    tangent = jax.tree_util.tree_map(jnp.zeros_like, geo)
    return geo, replace(
        tangent, rmns=geo.rmns, zmnc=-geo.zmnc, bmns=geo.bmns / 0.01 * 1e-3, gmns=geo.gmns, bmnc=1e-3 * geo.bmnc, iota=1e-2 * jnp.ones(N)
    )


def test_classification_separates_gap_modes_from_the_continuum_on_ctok():
    """The n = 2 CTOK TAE gap (232-396 kHz at s = 0.45) holds isolated modes; the modes that peak at s > 0.6 sit on a branch."""
    geo = fax.read_boozmn(DATA / "boozmn_ctok.nc")
    r = classify_modes(geo, fax.mode_family(2, 12), fax.mass_density(1e19, 2.0), nodes=48, count=12, target_hz=275e3, iters=1000, nzeta=4)
    gap, outer = r.isolated, r.peak_s > 0.6
    assert gap.any() and outer.sum() >= 4
    assert (
        np.all(r.margin[gap] > 0.1) and np.all(np.abs(r.peak_s[gap] - 0.45) < 0.03) and np.all(np.abs(r.frequency_hz[gap] - 273e3) < 15e3)
    )
    assert not r.isolated[outer].any() and r.margin[outer].max() < 0.03
    assert np.all(r.shift[gap] < 1e-2) and np.all(r.overlap[gap] > 0.9)


def test_uniform_field_modes_are_all_continuum_like():
    """The uniform-field spectrum fills the continuum: every mode peaks where its frequency meets a branch."""
    geo = fax.circular_torus(s=(jnp.arange(40) + 0.5) / 40, iota=0.3 + 0.3 * (jnp.arange(40) + 0.5) / 40)
    r = classify_modes(geo, jnp.array([[1.0, 1.0]]), fax.mass_density(1e19, 2.0), nodes=40, count=6, ntheta=8, nzeta=4)
    assert not r.isolated.any() and r.margin.max() < 0.01
    assert r.shift.max() < 0.01 and r.overlap.min() > 0.95  # refinement alone does not tell them apart at this resolution


def test_selected_frequencies_carry_status_and_unconverged_solves_are_rejected():
    geo = konies()
    f, status = fax.geometry_frequencies(geo, design(), 3)
    modes = fax.eigenmodes(geo.interpolate(jnp.arange(1, 25) / 25), HARMONICS, RHO, **OPTIONS)
    np.testing.assert_allclose(f, modes.frequencies_hz[:3], rtol=1e-10)
    assert status.converged and status.gap > 0 and status.residual < 1e-10
    dense, dstatus = fax.geometry_frequencies(geo, design(target_hz=None), 3)  # lowest modes from the dense solve
    np.testing.assert_allclose(
        dense, fax.eigenmodes(geo.interpolate(jnp.arange(1, 25) / 25), HARMONICS, RHO, ntheta=16, nzeta=4).frequencies_hz[:3], rtol=1e-10
    )
    assert dstatus.converged and dstatus.gap > 0 and dstatus.iterations == 0
    f, status = fax.geometry_frequencies(geo, design(tol=1e-30), 3)
    assert not status.converged and jnp.isnan(f).all()
    tangent = jax.tree_util.tree_map(jnp.ones_like, geo.iota)
    assert jnp.isnan(
        jax.jvp(lambda i: fax.geometry_frequencies(replace(geo, iota=i), design(tol=1e-30), 1)[0], (geo.iota,), (tangent,))[1]
    ).all()
    ok = fax.eigen.Status(*(jnp.asarray(v) for v in (0.0, 0.0, 1, True, 1.0, False)))
    assert _reject(jnp.ones(2), ok, jnp.ones(2)).tolist() == [1.0, 1.0]
    assert jnp.isnan(_reject(jnp.ones(2), ok._replace(gap=jnp.asarray(1e-12)), jnp.ones(2))).all()  # a selection that splits a tied set
    ions = fax.FastIons(density=1e15 * (1 - jnp.arange(1, 25) / 25) ** 2, temperature_ev=150e3)
    rates = lambda tol: fax.geometry_growth_rates(geo, design(tol=tol), ions, 2, ntheta=16, nzeta=4)  # noqa: E731
    assert jnp.isfinite(rates(1e-10)).all() and jnp.isnan(rates(1e-30)).all()


def test_gap_closure_is_smooth_where_the_widest_gap_changes():
    """Two gaps of width 50 -/+ x kHz trade places at x = 0: the hard widest gap has a gradient jump, the softmax one none."""
    branch = lambda x: jnp.array([0.0, 1.0e4, 6.0e4 + x, 11.0e4])  # noqa: E731 - widths 1e4, 5e4 + x, 5e4 - x
    hard = lambda x: jnp.mean(jax.nn.softplus(-fax.gaps(branch(x))[1] / 1e4))  # noqa: E731
    soft = lambda x: fax.gap_closure(branch(x), 1e4)  # noqa: E731
    jump = lambda f: abs(jax.grad(f)(1e-3) - jax.grad(f)(-1e-3))  # noqa: E731
    assert jump(hard) > 0.1 * abs(jax.grad(hard)(2e4))
    assert jump(soft) < 1e-6 * abs(jax.grad(soft)(2e4)) + 1e-9
    for x in (-3e4, -1e3, 0.0, 1e3, 3e4):  # AD against FD on both sides of the switch
        fd = (soft(x + 1.0) - soft(x - 1.0)) / 2.0
        assert jax.grad(soft)(x) == pytest.approx(float(fd), rel=1e-5, abs=1e-14)
    w = jnp.diff(branch(0.0))
    widest = float(jnp.sum(jax.nn.softmax(w / 1e4) * w))
    assert 5e4 - 1e4 * np.log(3) < widest < 5e4  # within scale log(K) below the hard maximum


def test_continuum_cost_is_continuous_across_the_helicity_cut():
    """Helicity (0, 1) of an elliptic torus: the (2, 0) harmonic enters the cost when ``2 iota`` crosses ``|0 iota - 1| = 1``."""
    base = fax.circular_torus(s=jnp.linspace(0.2, 0.8, 7), iota=0.4)
    cost = lambda iota, softness=0.05: fax.continuum_cost(  # noqa: E731
        replace(base, iota=iota * jnp.ones(7), zmns=base.zmns.at[:, 1].multiply(1.5)), (0, 1), 16, 8, softness
    )
    hard = cost(0.5 + 0.025, 1e-9)  # the (2, 0) term fully inside the old mask
    jump = cost(0.5 + 1e-6, 1e-9) - cost(0.5 - 1e-6, 1e-9)  # the hard mask adds the whole term when 2 iota passes 1
    assert jump > 0.9 * hard and abs(cost(0.5 + 1e-6) - cost(0.5 - 1e-6)) < 1e-7 * jump  # the smooth one does not
    assert cost(0.5 + 0.025) == pytest.approx(hard, rel=1e-12) and 0.0 < cost(0.5 + 0.0125) < hard  # the smoothstep ramp
    grad = jax.grad(cost)
    assert abs(grad(0.5 + 1e-7) - grad(0.5 - 1e-7)) < 1e-4 * abs(grad(0.5 + 0.0125))  # C1 at the knot (FD there is only O(h))
    for iota in (0.49, 0.5 + 1e-3, 0.5 + 0.0125, 0.5 + 0.024, 0.526, 0.55):
        fd = (cost(iota + 1e-6) - cost(iota - 1e-6)) / 2e-6
        assert grad(iota) == pytest.approx(float(fd), rel=1e-5, abs=1e-12), iota


def _fd_sweep(f, x, tangent, steps):
    """Central FD of ``f`` along ``tangent`` at every step; ``x`` and ``tangent`` are pytrees."""
    shift = lambda h: jax.tree_util.tree_map(lambda a, d: a + h * d, x, tangent)  # noqa: E731
    return np.array([(f(shift(h)) - f(shift(-h))) / (2 * h) for h in steps])


@pytest.mark.parametrize("target", [None, 3.0e5])
def test_fixed_geometry_derivative_has_a_step_size_plateau(target):
    """AD (Hellmann-Feynman through ``geometry_frequencies``) against a decade sweep of central FD in the geometry leaves,
    dense and shift-invert, along a tangent that moves every asymmetric harmonic: ``h^2`` truncation, then a floor below 1e-6
    over at least four decades-steps (about 1e-9 at the minimum), then the ``eps / h`` growth."""
    geo, tangent = asymmetric()
    harmonics = jnp.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
    d = Design("", (), (), harmonics, 3.4e-8, 8, dict(ntheta=16, nzeta=8, target_hz=target, count=4))
    f = lambda g: fax.geometry_frequencies(g, d, 2)[0][1]  # noqa: E731
    value, ad = jax.jvp(f, (geo,), (tangent,))
    assert jnp.isfinite(value) and abs(ad) > 1e-3 * abs(value)
    err = np.abs(_fd_sweep(f, geo, tangent, 10.0 ** -np.arange(1, 10)) / ad - 1)
    assert err[0] / err[1] > 30 and err.min() < 1e-8 and (err < 1e-6).sum() >= 4 and err[-1] > 100 * err.min()


# summary and plot_summary


def test_summary_tae_gap_and_mode(tmp_path):
    """The TAE band of a circular torus is a global gap at the cylindrical crossing, and the global mode lies inside it."""
    N, R0, B0, n_i, amu = 32, 10.0, 3.0, 2e19, 1.00727647
    s = (jnp.arange(N) + 0.5) / N
    geo = fax.circular_torus(s=s, major_radius=R0, minor_radius=1.0, B0=B0, iota=0.5 + 0.35 * s, epsilon_B=0.15)
    rho = fax.mass_density(n_i, amu)
    out = fax.summary(geo, fax.mode_family(1, 2, 1), rho, ions=(n_i, amu), count=3, ntheta=16, nzeta=4)
    iota_c, w_hat = fax.gap_location(1, 1, 1)
    f_c = w_hat * B0 / np.sqrt(4e-7 * np.pi * float(rho)) / (2 * np.pi * R0) / 1e3
    gap = out["gaps"][0]
    assert gap["type"] == "TAE" and sorted(gap["harmonics"]) == [[1, 1], [2, 1]]
    assert gap["low_khz"] < gap["center_khz"] < gap["high_khz"]
    assert abs(gap["center_khz"] / f_c - 1) < 0.05 and abs(gap["iota"] - iota_c) < 0.05
    mode = out["mode"]
    assert mode["in_gap"] and mode["converged"] and gap["low_khz"] < mode["frequency_khz"] < gap["high_khz"]
    assert (out["density_m3"], out["ion_mass_amu"], out["density_kg_m3"]) == (n_i, amu, float(rho))
    png = fax.plot_summary(out, tmp_path / "summary.png")
    assert png.stat().st_size > 5000


def test_summary_without_gap_or_mode(tmp_path, monkeypatch):
    """An empty window gives no gaps and no mode, a gap without a mode in it says why, and the figure draws both."""
    s = (jnp.arange(20) + 0.5) / 20
    geo = fax.circular_torus(s=s, iota=0.5 + 0.35 * s, epsilon_B=0.3)
    rho, h = fax.mass_density(2e19, 2.0), fax.mode_family(1, 2, 1)
    out = fax.summary(geo, h, rho, window=(1e6, 2e6), ntheta=16, nzeta=4)
    assert out["gaps"] == [] and out["mode"] is None and "no global gap" in out["mode_reason"]
    assert fax.plot_summary(out, tmp_path / "none.png").exists()
    solve = fax.objectives.eigenmodes
    monkeypatch.setattr(
        fax.objectives, "eigenmodes", lambda *a, **k: (lambda m: m._replace(frequencies_hz=3 * m.frequencies_hz))(solve(*a, **k))
    )
    out = fax.summary(geo, h, rho, count=1, ntheta=16, nzeta=4)  # every mode outside the band
    assert out["gaps"] and out["mode"] is None and "lies inside" in out["mode_reason"]
    assert fax.plot_summary(out, tmp_path / "nomode.png").exists()
