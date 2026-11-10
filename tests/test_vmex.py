"""End-to-end ``vmex -> booz_xform_jax -> FAX`` gates (skipped without ``vmex``)."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fax
from fax.eigen import equilibrium_currents, force_balance_residual
from fax.geometry import jacobian_check, surface_grid

pytest.importorskip("vmex", exc_type=ImportError)
INPUT = Path(__file__).parents[1] / "examples" / "data" / "input.LandremanPaul2021_QA_lowres"
# A tilted-ellipse tokamak (R = 10 + cos t + 0.05 sin t, Z = sin t + 0.03 cos t) with up-down asymmetry.
ASYMMETRIC_TOKAMAK = """&INDATA
  LASYM = T, NFP = 1, MPOL = 5, NTOR = 0, NZETA = 1, NTHETA = 16,
  NS_ARRAY = 17, FTOL_ARRAY = 1e-10, NITER_ARRAY = 3000,
  DELT = 0.9, NSTEP = 200, PHIEDGE = 9.42477796, NCURR = 0, GAMMA = 0.0,
  PRES_SCALE = 1.0, AM = 7.17e3 -6.811e3 -3.585e2,
  AI = 5.84795322e-01 -5.47176905e-02 5.11977817e-03 -4.79011335e-04 4.47304912e-05 -4.05618024e-06 2.86012825e-07,
  RAXIS_CC = 10.0, ZAXIS_CS = 0.0, RAXIS_CS = 0.0, ZAXIS_CC = 0.0,
  RBC(0,0) = 10.0, RBC(0,1) = 1.0, ZBS(0,1) = 1.0, RBS(0,1) = 0.05, ZBC(0,1) = 0.03,
/
"""


@pytest.mark.slow
def test_vmec_input_to_continuum():
    geo = fax.from_vmec_input(INPUT, ns=16, mboz=6, nboz=6, surfaces=(0.3, 0.5, 0.7))
    assert geo.nfp == 2 and geo.s.shape == (3,)
    f = fax.continuum(geo, fax.mode_family(1, 4), fax.mass_density(1e19), 16, 16)
    assert jnp.all(jnp.isfinite(f)) and f.shape == (3, 5)


@pytest.mark.slow
def test_boundary_gradient_and_radial_refinement():
    """Lowest n = 0 global frequency of LP-QA: AD vs re-solve FD in ``zs01`` while only the FE grid is refined.

    ``ns``, ``ftol`` and ``mboz`` stay fixed; the FE nodes are uniform on ``[0, 1]``.
    At every level the FAX chain (interpolation, metric, Hellmann-Feynman) matches
    FD along the re-solved Boozer geometry to ~1e-8, and the full adjoint
    gradient matches the re-solve FD to 3e-5: the remainder is vmex's released
    m = 1 gauge, which the adjoint holds fixed and each re-solve lets drift
    (closure to 3e-9 on iota, plan.md P3). The lowest eigenvalue is the
    continuum minimum, so both it and its gradient converge at first order.
    """
    h = 1e-4
    harmonics = [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    design = fax.prepare_design(INPUT, harmonics=harmonics, density=3.4e-8, parameters=("zs01",), ntheta=8, nzeta=8)
    g0, gp, gm = (fax.design_geometry(jnp.array([x]), design) for x in (0.0, h, -h))
    dg = jax.tree_util.tree_map(lambda a, b: (a - b) / (2 * h), gp, gm)
    values, grads = [], []
    for nodes in (16, 32, 64):
        d = replace(design, nodes=nodes)
        f = lambda g: fax.geometry_frequencies(g, d, 1)[0][0]  # noqa: B023, E731
        value, ad = jax.jvp(f, (g0,), (dg,))
        np.testing.assert_allclose(ad, (f(gp) - f(gm)) / (2 * h), rtol=1e-4)
        values, grads = values + [value], grads + [ad]
    for x in (values, grads):
        ratio = (x[0] - x[1]) / (x[1] - x[2])
        assert 1.7 < ratio < 2.3  # first order
    d = replace(design, nodes=32)
    np.testing.assert_allclose(jax.grad(lambda p: fax.design_frequencies(p, d, 1)[0][0])(jnp.zeros(1))[0], grads[1], rtol=1e-4)
    targeted = replace(d, options={**d.options, "target_hz": 0.98 * float(values[1]), "count": 2})
    v, t = jax.jvp(lambda g: fax.geometry_frequencies(g, targeted, 1)[0][0], (g0,), (dg,))
    np.testing.assert_allclose([v, t], [values[1], grads[1]], rtol=1e-8)


@pytest.mark.slow
def test_continuum_cost_gradient_step_and_vmex_terms():
    """One descent step on Paul's ``f_cont`` through vmex lowers it; the vmex QS/aspect terms share the state."""
    design = fax.prepare_design(INPUT, harmonics=[[1.0, 0.0]], density=1.0, parameters=("rc11", "zs11"), nodes=8)
    cost = lambda p: fax.continuum_cost(fax.design_geometry(p, design), (1, 0))  # noqa: E731
    value, grad = jax.value_and_grad(cost)(jnp.zeros(2))
    assert cost(-1e-3 * grad / jnp.linalg.norm(grad)) < value
    from vmex.core import implicit

    assert implicit.aspect_ratio(*fax.design_state(jnp.zeros(2), design)) == pytest.approx(6.0, rel=1e-3)


@pytest.mark.slow
def test_konies_itpa_finite_beta_tae():
    """Könies (2018) ITPA case solved by vmex with p(s) = 7.17e3 - 6.811e3 s - 358.5 s^2 Pa (shifted equilibrium).

    The TAE (the most global mode near 64.6 kHz) is 4.127e5 rad/s without and 3.947e5 with the
    finite-beta terms; CAS3D (compressible) gives 4.01e5.
    """
    geo = fax.from_vmec_input(INPUT.with_name("input.ITPA_Konies2018"), mboz=12, nboz=0, surfaces=tuple(np.arange(1, 50) / 50))
    psi, spread = jacobian_check(geo)  # PHIEDGE = 3 pi, so psi' = 1.5; the first four surfaces (s < 0.1) are up to 4.5% off
    np.testing.assert_allclose(psi[4:], 1.5, rtol=5e-3)
    assert spread.max() < 1e-3 and force_balance_residual(geo).max() < 2e-4
    mu0_dp = 4e-7 * np.pi * (-6.811e3 - 717.0 * np.asarray(geo.s))  # mu0 dp/ds of AM = 7.17e3 -6.811e3 -3.585e2
    np.testing.assert_allclose(equilibrium_currents(geo, surface_grid(geo, 32, 32))[0], mu0_dp, rtol=3e-3)
    geo = geo.interpolate((jnp.arange(100) + 0.5) / 100)
    omega = []
    for finite_beta in (False, True):
        rho = fax.mass_density(2e19, 1.00727647)
        modes = fax.eigenmodes(geo, fax.mode_family(6, 16, 5), rho, target_hz=6.46e4, ntheta=32, nzeta=4, finite_beta=finite_beta)
        x = np.abs(np.asarray(modes.structures).reshape(8, -1)) ** 2
        omega.append(float(np.sqrt(modes.omega2[np.argmax(x.sum(1) ** 2 / (x**2).sum(1))])))
    assert omega[0] == pytest.approx(4.127e5, rel=2e-3) and omega[1] == pytest.approx(3.947e5, rel=2e-3)


@pytest.mark.slow
def test_stellarator_drive_gradient_and_sidebands():
    """LP-QA n = 1 modes near 104 kHz driven by peaked 150 keV deuterons: helical sidebands ``k != n`` appear, and the
    drive objective's FAX chain (FE interpolation, shift-invert eigenvectors, drive) matches FD along the re-solved geometry."""
    design = fax.prepare_design(
        INPUT,
        harmonics=fax.mode_family((1, -1, 3), 5),
        density=3.4e-8,
        parameters=("rc01",),
        nodes=16,
        ntheta=16,
        nzeta=16,
        target_hz=104e3,
        count=4,
    )
    ions = fax.FastIons(density=1e15 * (1 - jnp.arange(1, 17) / 17) ** 2, temperature_ev=150e3)
    h = 1e-5
    g0, gp, gm = (fax.design_geometry(jnp.array([x]), design) for x in (0.0, h, -h))
    dg = jax.tree_util.tree_map(lambda a, b: (a - b) / (2 * h), gp, gm)
    risk = jax.jit(lambda g: fax.drive_risk(fax.geometry_growth_rates(g, design, ions, 4, ntheta=32, nzeta=32)))
    value, ad = jax.jvp(risk, (g0,), (dg,))
    assert jnp.isfinite(value) and value > 0
    np.testing.assert_allclose(ad, (risk(gp) - risk(gm)) / (2 * h), rtol=2e-4)  # vmex 0.14: 1.4e-4 at h = 1e-5, 9e-5 at 1e-6 (flat)
    geo = g0.interpolate(jnp.arange(1, 17) / 17)
    ev = fax.eigenmodes(geo, design.harmonics, 3.4e-8, target_hz=104e3, count=4, ntheta=16, nzeta=16)
    pairs, gam = fax.growth_rate(
        jnp.sqrt(ev.omega2[-1]), ev.structures[-1], geo, design.harmonics, ions, 3.4e-8, ntheta=32, nzeta=32, resolved=True
    )
    side = jnp.sum(jnp.where(pairs[:, 1] != 1, gam, 0.0))
    pure = jnp.sum(jnp.where(jnp.abs(pairs[:, 1]) > 3, jnp.abs(gam), 0.0))  # k = -3, 5: only through the helical B harmonics
    assert jnp.all(jnp.isfinite(gam)) and 1e-6 < abs(side / jnp.sum(gam)) < 1e-2 and pure > 0


@pytest.mark.slow
def test_asymmetric_equilibrium_wout_path_and_design_tables_agree(tmp_path):
    """Up-down asymmetric tokamak: the ``wout`` file, the in-memory ``wout`` and the traceable vmex design path give one Boozer geometry.

    The asymmetric tables (``rmns``, ``zmnc``, ``bmns``) carry the boundary tilt (``RBS(0,1) = 0.05``, ``ZBC(0,1) = 0.03``,
    so ``rmns[m = 1]`` and ``zmnc[m = 1]`` are about 0.04 at the edge) and the three routes agree to the equilibrium
    tolerance, which exercises the ``lasym`` table list of :func:`fax.objectives.state_geometry`.
    """
    import vmex

    inp_path = tmp_path / "input.asym"
    inp_path.write_text(ASYMMETRIC_TOKAMAK)
    inp = vmex.VmecInput.from_file(str(inp_path))
    wout = vmex.wout_from_result(inp, vmex.solve_multigrid(inp))
    vmex.write_wout(str(tmp_path / "wout_asym.nc"), wout)
    booz = {"mboz": 4, "nboz": 0, "surfaces": (0.9,)}
    in_memory = fax.from_wout(wout, **booz)
    from_file = fax.io.geometry(fax.load({"geometry": {"source": "wout", "path": str(tmp_path / "wout_asym.nc"), **booz}}))
    for name in ("s", "iota", "G", "bmnc", "bmns", "rmnc", "rmns", "zmns", "zmnc"):
        np.testing.assert_allclose(getattr(from_file, name), getattr(in_memory, name), rtol=1e-12, atol=1e-12)
    design = fax.prepare_design(
        inp_path, harmonics=[[1.0, 0.0]], density=1.0, parameters=("rc01",), nodes=4, ns=17, ftol=1e-10, mboz=4, nboz=0
    )
    geo = fax.design_geometry(jnp.zeros(1), design)
    assert float(in_memory.s[0]) == pytest.approx(float(geo.s[-2]))
    assert abs(float(in_memory.rmns[0, 1])) == pytest.approx(0.039, abs=5e-3) and abs(float(in_memory.zmnc[0, 1])) == pytest.approx(
        0.039, abs=5e-3
    )
    for name in ("iota", "bmnc", "bmns", "rmnc", "rmns", "zmns", "zmnc"):
        np.testing.assert_allclose(np.asarray(getattr(geo, name))[-2], np.asarray(getattr(in_memory, name))[0], atol=5e-3, err_msg=name)


def _gap_design(**options):
    """LP-QA, the n = 1 family: its two gap modes (102.1 and 103.5 kHz, in the 96-142 kHz gap) are the 2 nearest 103 kHz."""
    settings = dict(harmonics=fax.mode_family((1, -1, 3), 5), density=3.4e-8, nodes=24, ntheta=16, nzeta=16, target_hz=103e3, count=2)
    return fax.prepare_design(INPUT, **{**settings, **options})


def _resolve_fd(f, p0, i, steps):
    """Central re-solve finite differences of ``f`` along unit direction ``i`` for every step."""
    e = jnp.zeros_like(p0).at[i].set(1.0)
    return np.array([(f(p0 + h * e) - f(p0 - h * e)) / (2 * h) for h in steps])


@pytest.mark.slow
def test_resolved_equilibrium_derivative_has_a_gauge_floor():
    """Gap-mode frequency of LP-QA, AD vs re-solved equilibria in ``zs01``: the two steps agree to 1% of the remainder.

    With vmex <= 0.13 the remainder (4.6e-4) is flat in the step, so it is not truncation or noise: it is the m = 1 Z_sin gauge
    that each re-solve lets drift and the adjoint holds fixed (4.4e-4 for iota, plan.md P3). vmex 0.14 holds the m = 1 family in
    the implicit solve and its derivative, which closes it (3e-7 and 1e-8, truncation falling with the step).
    """
    d = _gap_design(parameters=("zs01",))
    f = lambda p: fax.design_frequencies(p, d, 2)[0][0]  # noqa: E731
    p0 = jnp.zeros(1)
    err = np.abs(_resolve_fd(f, p0, 0, (1e-4, 1e-5)) / jax.grad(f)(p0)[0] - 1)
    assert err.max() < 1e-3 and (err[1] < 1e-6 or err[0] == pytest.approx(err[1], rel=1e-2))  # closed (vmex >= 0.14) or a flat floor


@pytest.mark.scheduled
def test_m1_gauge_floor_and_stellarator_asymmetric_direction(tmp_path):
    """Re-solved derivatives against the adjoint at two steps (1e-4, 1e-5): the remainder is flat, so it is a gauge floor and
    not truncation or noise. ``zs11`` of LP-QA (the m = 1 direction) stays at the 1e-2 level (1.1e-2); ``zc20`` of the tilted
    tokamak (``lasym``, m = 2, stellarator-asymmetric) is below 5e-2.
    vmex 0.14 holds the m = 1 family, so ``zs11`` closes to 1e-6 there; ``zc20`` (m = 2) keeps its 1e-4 remainder.

    vmex's ``frozen_path_directional_fd``, the gauge-free reference, did not work on these decks: its Newton iteration diverges
    along ``zs11`` and gives a non-finite state along ``zs01`` after 10 steps (``plan.md`` section E): the floors are reported.
    """
    path = tmp_path / "input.asym"
    path.write_text(ASYMMETRIC_TOKAMAK)
    kw = dict(harmonics=[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], density=1e-7, nodes=12, ns=17, ftol=1e-10, mboz=4, nboz=0)
    cases = ((_gap_design(parameters=("zs11",)), 2, 1e-3, 5e-2), (fax.prepare_design(path, parameters=("zc20",), **kw), 1, 0.0, 5e-2))
    for d, modes, low, high in cases:
        f = lambda p, d=d, modes=modes: fax.design_frequencies(p, d, modes)[0][0]  # noqa: E731
        ad = jax.grad(f)(jnp.zeros(1))[0]
        err = np.abs(_resolve_fd(f, jnp.zeros(1), 0, (1e-4, 1e-5)) / ad - 1)
        print(d.names, "re-solve remainder", err)
        assert np.all(err < high) and (err[1] < 1e-6 or (np.all(low < err) and err[0] == pytest.approx(err[1], rel=0.1)))


SWEEPS = {  # one resolution at a time, from the base (nodes 24, ns 16, mboz = nboz 6, 16 x 16 angles, m <= 5); the last two are compared
    "FE nodes": [dict(nodes=n) for n in (24, 48, 96)],
    "VMEC ns": [dict(ns=n) for n in (16, 24, 32)],
    "Boozer mboz = nboz": [dict(mboz=n, nboz=n) for n in (4, 6, 8)],
    "angle grid": [dict(ntheta=n, nzeta=n) for n in (16, 24, 32)],
    "harmonics m <= M": [dict(harmonics=fax.mode_family((1, -1, 3), m), ntheta=24, nzeta=24) for m in (5, 7, 9)],
}


@pytest.mark.scheduled
def test_resolution_budget_of_the_gap_mode_frequency_and_gradient():
    """The lowest gap mode of LP-QA (102 kHz) and its gradient in the four design coefficients of the QA example, while
    FE nodes, VMEC ``ns``, Boozer ``mboz = nboz``, the angle grid and the harmonic set are refined one at a time.

    ``-s`` prints the table kept in ``docs/validation.md``: for each sweep the value at the finest level and the relative change
    of ``f`` and of ``grad f`` (vector norm) from the second finest, which is the numerical error budget of that resolution.
    """
    budget = {
        "FE nodes": (3e-3, 3e-2),
        "VMEC ns": (3e-3, 3e-2),
        "Boozer mboz = nboz": (3e-3, 3e-2),
        "angle grid": (1e-6, 1e-4),
        "harmonics m <= M": (1e-3, 1e-2),
    }

    def evaluate(**level):
        d = _gap_design(parameters=("rc11", "zs11", "rc12", "zs12"), **{"mboz": 6, "nboz": 6, **level})
        f, g = jax.value_and_grad(lambda p: fax.design_frequencies(p, d, 2)[0][0])(jnp.zeros(4))
        return float(f), np.asarray(g)

    base = evaluate()
    for name, levels in SWEEPS.items():
        (f1, g1), (f2, g2) = (evaluate(**level) for level in levels[1:])
        df, dg = abs(f2 / f1 - 1), np.linalg.norm(g2 - g1) / np.linalg.norm(g2)
        print(f"{name:20s} f {f2 / 1e3:9.4f} kHz  |df/f| {df:.1e}  |dg|/|g| {dg:.1e}  (base {base[0] / 1e3:.4f})")
        assert df < budget[name][0] and dg < budget[name][1], name
