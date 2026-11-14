"""Drive screen, objectives, TOML runs, CLI and plotting."""

import importlib
import sys
import types
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fax
from fax.continuum import MU0
from fax.io import main

CTOK = str(Path(__file__).parent / "data" / "boozmn_ctok.nc")


def test_objectives_are_smooth():
    f = jnp.array([10e3, 125e3, 400e3])
    assert fax.window_risk(f, (50e3, 200e3), 5e3) == pytest.approx(0.5 / 3, rel=1e-3)
    assert fax.soft_max(f / 1e3) >= 400.0
    branches = jnp.array([[1e3, 2e3, 50e3], [1e3, 1.5e3, 2e3]])
    assert jnp.isfinite(jax.grad(lambda b: fax.gap_closure(b))(branches)).all()


def test_continuum_cost_of_an_elliptic_torus():
    """``|grad s|^2`` of ellipses ``(r cos, kappa r sin)`` is ``1 + (k^2 - 1)/(k^2 + 1) cos 2 theta`` times its mean."""
    base = fax.circular_torus(s=jnp.linspace(0.2, 0.8, 31), iota=0.4)
    geo = lambda k: replace(base, zmns=base.zmns.at[:, 1].multiply(k))  # noqa: E731
    cost = lambda k: fax.continuum_cost(geo(k), (1, 0), 16, 8)  # noqa: E731
    c, dm, n = fax.compression_harmonics(geo(1.5), 16, 8)
    e = (1.5**2 - 1) / (2 * (1.5**2 + 1))
    np.testing.assert_allclose(c[:, [2, -2], 0], e, atol=1e-12)
    assert (dm[2], n[0]) == (2, 0) and jnp.abs(c).sum() == pytest.approx(2 * e * 31)
    assert cost(1.5) == pytest.approx(31 * 2 * e**2 * (2 * 0.4) ** 2) and fax.continuum_cost(geo(1.5), (2, 0), 16, 8) < 1e-20
    assert jax.grad(cost)(1.5) == pytest.approx((cost(1.5 + 1e-5) - cost(1.5 - 1e-5)) / 2e-5, rel=1e-7)


def test_boundary_parameters_accept_tuples_and_reject_ambiguous_strings():
    slot = lambda p: fax.objectives._slot(p, 3)  # noqa: E731
    assert slot("rc01") == ("rbc", 4, 0) and slot("zs1-2") == ("zbs", 1, 1) and slot(("zs", 12, -2)) == ("zbs", 1, 12)
    assert slot(("rc", 10, 11)) == ("rbc", 14, 10)
    for bad in ("rc101", "rc1-12", "rc1", "xc01"):
        with pytest.raises(ValueError, match="cannot read"):
            slot(bad)


def test_load_rejects_unknown_keys():
    with pytest.raises(ValueError):
        fax.load({"run": {"tsk": "continuum"}})
    with pytest.raises(ValueError, match="unknown table"):
        fax.load({"run": 3})
    assert fax.load({})["modes"]["n"] == [1]


@pytest.mark.parametrize("task", ["continuum", "eigenmodes"])
def test_run_cases_and_cli(tmp_path, task):
    case = tmp_path / "case.toml"
    case.write_text(
        f'[geometry]\nsource = "boozmn"\npath = "{CTOK}"\n[modes]\nn = [2]\nm_max = 4\n'
        f'[plasma]\nion_mass = 2.0\n[run]\ntask = "{task}"\noutput = "out"\nntheta = 16\nnzeta = 16\n'
    )
    assert main(["run", str(case), "--plot"]) == 0
    data = np.load(tmp_path / "out" / "results.npz")
    assert data["frequencies_hz"].shape[-1] == (5 if task == "continuum" else 8)
    assert (tmp_path / "out" / "continuum.png").exists()
    assert main(["plot", str(tmp_path / "out")]) == 0


def test_circular_geometry_run():
    out = fax.run({"run": {"ntheta": 32, "nzeta": 4}}, save=False)
    k = np.abs(1.0 - np.arange(9) * 0.4) / 5.5  # uniform field: f = |n - m iota| B0 / (R sqrt(mu0 rho)) / 2 pi
    exact = np.sort(k * 2.7 / np.sqrt(MU0 * fax.mass_density(1e19)) / (2 * np.pi))
    np.testing.assert_allclose(out["frequencies_hz"], np.broadcast_to(exact, (24, 9)), rtol=1e-9)
    with pytest.raises(ValueError, match="aliases"):
        fax.run({"run": {"ntheta": 8, "nzeta": 4}}, save=False)
    with pytest.raises(ValueError):
        fax.run({"run": {"task": "nope"}}, save=False)


def test_cli_run_without_plot_writes_results_only(tmp_path, capsys):
    case = tmp_path / "case.toml"
    case.write_text('[run]\noutput = "out"\nntheta = 32\nnzeta = 4\n')
    assert main(["run", str(case)]) == 0
    assert "(24, 9) frequencies" in capsys.readouterr().out
    assert (tmp_path / "out" / "results.npz").exists() and (tmp_path / "out" / "case.json").exists()
    assert not list((tmp_path / "out").glob("*.png"))


@pytest.fixture(autouse=True)
def cache_state(tmp_path, monkeypatch):
    """Isolate the process-wide compilation-cache config (and ``~``) from the test."""
    monkeypatch.setenv("HOME", str(tmp_path))
    from jax._src import compilation_cache

    before = jax.config.jax_compilation_cache_dir
    yield
    jax.config.update("jax_compilation_cache_dir", before)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    compilation_cache.reset_cache()  # else the worker keeps writing every later compile into the deleted temp directory


def test_cli_caches_compilations_by_default_and_on_request(tmp_path, cache_state):
    case = tmp_path / "case.toml"
    case.write_text('[run]\noutput = "out"\nntheta = 32\nnzeta = 4\n')
    jax.config.update("jax_compilation_cache_dir", None)
    assert main(["run", str(case), "--no-cache"]) == 0
    assert jax.config.jax_compilation_cache_dir is None
    assert main(["run", str(case)]) == 0
    assert jax.config.jax_compilation_cache_dir == str(tmp_path / ".cache" / "fax" / "jax")
    assert main(["run", str(case), "--cache", str(tmp_path / "jit")]) == 0
    assert jax.config.jax_compilation_cache_dir == str(tmp_path / "jit")


def test_enable_cache_respects_a_user_set_directory(tmp_path, cache_state):
    jax.config.update("jax_compilation_cache_dir", str(tmp_path / "mine"))
    assert fax.enable_cache() == str(tmp_path / "mine")
    assert fax.enable_cache(tmp_path / "other") == str(tmp_path / "other")


def test_geometry_source_is_validated():
    with pytest.raises(ValueError, match="geometry.source must be one of"):
        fax.io.geometry(fax.load({"geometry": {"source": "gs2"}}))


def test_toml_reader_falls_back_to_tomli(monkeypatch):
    fallback = types.ModuleType("tomli")
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "tomllib", None)  # ``import tomllib`` now raises ImportError
        patch.setitem(sys.modules, "tomli", fallback)
        assert importlib.reload(fax.io).tomllib is fallback
    assert importlib.reload(fax.io).tomllib is not fallback
    assert fax.load({})["modes"]["n"] == [1]


def test_documented_transforms_and_failure_behaviour():
    """The derivative contract of ``docs/api.md``: jit, vmap, NaN on an unconverged solve, NaN propagation and ``kind`` validation."""
    s = (jnp.arange(24) + 0.5) / 24
    geo = fax.circular_torus(s=s, major_radius=10.0, minor_radius=1.0, B0=3.0, iota=1 / (1.71 + 0.16 * s), epsilon_B=0.1)
    h, rho = fax.mode_family(6, 12, 7), fax.mass_density(2e19, 1.00727647)
    grid, solve = dict(ntheta=16, nzeta=4), dict(target_hz=6.46e4, count=3)
    f = lambda r, **kw: fax.frequency_hz(geo, h, r, 2, 3, **solve, **grid, **kw)  # noqa: E731
    grad = jax.vmap(jax.grad(f))(rho * jnp.array([1.0, 2.0]))  # d f / d rho = -f / 2 rho for f ~ rho^-1/2
    np.testing.assert_allclose(grad, -jax.vmap(f)(rho * jnp.array([1.0, 2.0])) / (2 * rho * jnp.array([1.0, 2.0])), rtol=1e-6)
    assert jnp.isnan(f(rho, tol=1e-30)) and jnp.isnan(jax.grad(lambda r: f(r, tol=1e-30))(rho))  # unconverged: value and tangent rejected
    omega2 = jax.jit(lambda g, r: fax.eigenmodes(g, h, r, **solve, **grid).omega2)(geo, rho)
    assert omega2.shape == (3,) and (omega2 > 0).all()
    ions, drive = fax.FastIons(1e17, 2e5), dict(helical=0, nzeta=1, ntheta=16, nodes=8)
    gamma = lambda phi: fax.growth_rate(4e5, phi, geo, ((10, 6), (11, 6)), ions, rho, **drive)  # noqa: E731
    assert jnp.isfinite(gamma(jnp.ones((24, 2)))) and jnp.isnan(gamma(jnp.full((24, 2), jnp.nan)))
    with pytest.raises(ValueError, match="unknown FastIons kind"):
        fax.FastIons(kind="beam")
    for bad in (dict(temperature_ev=0), dict(birth_ev=-1.0), dict(critical_ev=np.nan), dict(mass_amu=jnp.inf)):
        with pytest.raises(ValueError, match="positive and finite"):
            fax.FastIons(**bad)
    assert jax.tree_util.tree_map(lambda _: None, ions).mass_amu is None  # pytree rebuilds skip the check
    assert jax.grad(lambda T: fax.FastIons(temperature_ev=T).temperature_ev)(1.0) == 1.0
    with pytest.raises(ValueError, match="target_hz"):
        fax.eigenmodes(geo, h, rho, target_hz=np.nan, **grid)
    with pytest.raises(ValueError, match="target_hz"):
        fax.eigenmodes(geo, h, rho, target_hz=0.0, **grid)
    assert fax.mode_family(np.int64(2), 4).shape == (5, 2)
    with pytest.raises(ValueError, match="m_max"):
        fax.mode_family(1, 2, 3)
