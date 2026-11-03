"""Perturbative fast-ion drive in Boozer geometry: Fu-Van Dam scaling, Koenies 2018 TAE, sign rules, gradients."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fax

S = (jnp.arange(200) + 0.5) / 200
R0 = jnp.sqrt((1.75 - 1.71) / 0.16)  # q = (m + 1/2)/n = 1.75 for m = 10, n = 6
TAE = jnp.stack([jnp.exp(-(((jnp.sqrt(S) - R0) / 0.0425) ** 2) / 2)] * 2, axis=1)
HARM = ((10, 6), (11, 6))
ITPA = fax.circular_torus(s=S, major_radius=10.0, minor_radius=1.0, B0=3.0, iota=1 / (1.71 + 0.16 * S), epsilon_B=0.1)
RHO = fax.mass_density(2e19, 1.00727647)
AXI = dict(helical=0, nzeta=1)  # axisymmetric: exact on one toroidal point


def koenies_density(scale=1.0):
    """Koenies et al., NF 58 126027 (2018), eq. 1."""
    c0, c1, c2, c3 = 0.49123, 0.298228, 0.198739, 0.521298
    return scale * 1.44131e17 * c3 * jnp.exp(-(c2 / c1) * jnp.tanh((jnp.sqrt(S) - c0) / c2))


def tae_gamma(t_kev, scale=1.0, omega=4.0e5, n=6, geo=ITPA):
    ions = fax.FastIons(density=koenies_density(scale), temperature_ev=t_kev * 1e3)
    return fax.growth_rate(omega, TAE, geo, ((10 * n // 6, n), (11 * n // 6, n)), ions, RHO, **AXI)


def test_fu_van_dam_maxwellian_scaling():
    """Weak toroidicity (|B| = B0 (1 - eps sqrt(s) cos theta), eps = 1e-4, R0 = 1e4 a), flat Maxwellian, the two
    v_par = v_A resonances (j = m + 1 of m = 10 and j = m - 1 of m = 11): gamma ~ -x (1 + 2 x^2 + 2 x^4) exp(-x^2) at fixed
    beta_h, x = v_A / v_th (Fu & Van Dam 1989; Betti & Freidberg 1992), and the absolute coefficient of the zero-orbit-width
    integral done in closed form with NumPy: kernel ``-(1 + xi^2) eps / (B0 a^2) [sqrt(s) sin(theta) phi' + i m cos(theta) phi /
    (2 sqrt(s))] e^{i m theta}``, orbit average -> ``|C|^2 = (1 + xi^2)^2 eps^2 (phi_r -+ m phi / r)^2 / (16 B0^2 a^4)``,
    resonance at ``v = v_par / xi``, ``int_0^1 2 xi dxi xi^-8 (1 + xi^2)^2 exp(-x^2 / xi^2) = 2 (1 + 2 x^2 + 2 x^4) exp(-x^2) / x^6``,
    psi' = B0 a^2 / 2, and ``W = int rho |grad phi|^2 / (2 B0^2) dV``. The prefactor ``gamma = pi^4 m_f^2 P / W`` (``P`` the radial
    integral) is that of the module docstring, fixed by the Koenies gate below, not derived here. Approximations: O(eps) toroidal
    corrections and the v_A / 3 resonances (excluded, they are ~exp(-9 x^2))."""
    R0, a, B0, eps, io, m_f, vpar, e = 1e4, 1.0, 3.0, 1e-4, 1 / 1.75, 2.01410178 * 1.66053906660e-27, 1e6, 1.602176634e-19
    rho = float(RHO)
    geo = fax.circular_torus(s=S, major_radius=R0, minor_radius=a, B0=B0, iota=io, epsilon_B=eps)
    r, r0, w = np.sqrt(np.asarray(S)), 0.5, 0.1
    f = np.exp(-(((r - r0) / w) ** 2) / 2)
    fr = -(r - r0) / w**2 * f
    trapz = lambda y: np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(r))  # noqa: E731
    W = sum(4 * np.pi**2 * R0 * rho / (2 * B0**2) * trapz(r * (fr**2 + m**2 * f**2 / r**2)) for m in (10, 11))
    kappa = abs(11 * io - 6)  # |j iota - k| of both v_A resonances
    omega = vpar * kappa / R0

    tilt = 1e-4  # n(s) = n0 (1 - tilt s); the gradient term subtracts T j tilt / (omega e |psi'|) from the profile factor (peaked drives)

    def analytic(x, m, sign, grad=False):
        v_th = vpar / x
        n = 1e22 / (0.5 * m_f * vpar**2 / x**2 / e)  # fixed beta_h
        prof = (1 - tilt * r**2) - (0.5 * m_f * v_th**2 * (m + sign) * tilt / (omega * e * 0.5 * B0 * a**2) if grad else 0.0)
        radial = trapz(prof * 0.5 * B0 * a**2 * eps**2 / (16 * B0**2 * a**4) * (fr - sign * m * f / r) ** 2 * 2 * r)
        return (
            -(np.pi**4)
            * m_f**2
            / W
            * 4
            * n
            * v_th
            * R0**2
            * (1 + 2 * x**2 + 2 * x**4)
            * np.exp(-(x**2))
            / (m_f * np.pi**1.5 * kappa * B0)
            * radial
        )

    x = np.array([1.0, 1.5, 2.0, 2.5])
    gam, grad = [], []
    for xi in x:
        t_ev = 0.5 * m_f * vpar**2 / xi**2 / e
        for ions, out in ((fax.FastIons(1e22 / t_ev, t_ev), gam), (fax.FastIons(1e22 / t_ev * (1 - tilt * S), t_ev), grad)):
            pairs, g = fax.growth_rate(omega, jnp.stack([f, f], 1), geo, HARM, ions, RHO, resolved=True, **AXI)
            by_j = dict(zip(pairs[:, 0].tolist(), np.asarray(g), strict=True))
            out.append([by_j[11], by_j[10]])
    gam, grad = np.array(gam), np.array(grad)
    assert (gam < 0).all() and (np.abs(gam) > 1e-3).all()  # flat density: pure Landau damping, nonzero
    np.testing.assert_allclose(gam[:, 0], [analytic(xi, 10, 1) for xi in x], rtol=1e-3)
    np.testing.assert_allclose(gam[:, 1], [analytic(xi, 11, -1) for xi in x], rtol=1e-3)
    shape = x * (1 + 2 * x**2 + 2 * x**4) * np.exp(-(x**2))
    np.testing.assert_allclose(gam / shape[:, None], (gam[0] / shape[0]) * np.ones((4, 1)), rtol=1e-3)
    # density tilt: the gradient term carries the orbit's poloidal harmonic j (11 and 10), not n / iota = 10.5 (a 5% change of the term)
    np.testing.assert_allclose(grad[:, 0], [analytic(xi, 10, 1, True) for xi in x], rtol=1e-3)
    np.testing.assert_allclose(grad[:, 1], [analytic(xi, 11, -1, True) for xi in x], rtol=1e-3)
    assert (np.abs(grad / gam - 1) > 0.02).all()  # the term is not a perturbation of the check


def test_koenies_2018_tae_growth_rates():
    """ZOW passing drive in the Boozer form of the ad hoc equilibrium vs Koenies et al. (2018), fig. 1a: the pooled ZLR band
    (8-10, 25-33, 38-43 x10^3/s) is not a reference here (see ``test_zow_curve_is_bracketed_by_the_passing_only_drive``); the circular
    large-aspect-ratio model gave 4.4, 16.6, 33.1, 65 x10^3/s and the Boozer form with n/iota in the
    gradient term 4.33, 16.52, 32.94, 64.85."""
    gam = np.array([tae_gamma(t) for t in (100.0, 200.0, 300.0, 400.0)])
    np.testing.assert_allclose(gam, [4.259e3, 16.27e3, 32.48e3, 64.02e3], rtol=1e-3)
    np.testing.assert_allclose(gam, [4.4e3, 16.6e3, 33.1e3, 65e3], rtol=0.04)  # O(1/n) labelling change of the gradient term, ~1.5%
    assert 0.8 * 40e3 < gam[2] < 1.2 * 40e3


def test_drive_sign_and_flat_profile():
    flat = fax.FastIons(density=1e17, temperature_ev=200e3)
    for ions in (flat, fax.FastIons(density=1e17, kind="slowing_down")):
        g = [fax.growth_rate(4e5, TAE, ITPA, ((10 * n // 6, n), (11 * n // 6, n)), ions, RHO, **AXI) for n in (6, -6)]
        assert g[0] == pytest.approx(g[1]) and g[0] < 0  # no gradient: pure dF/dE (Landau) damping
    assert tae_gamma(200.0, n=6) > 0 > tae_gamma(200.0, n=-6)  # (m, n) -> (-m, -n): counter-propagating
    hollow = fax.FastIons(density=koenies_density()[::-1], temperature_ev=200e3)
    assert fax.growth_rate(4e5, TAE, ITPA, HARM, hollow, RHO, **AXI) < 0  # dF/dP_zeta reversed
    assert fax.ion_landau_damping(4e5, TAE, ITPA, HARM, 20e3, 2e19, **AXI) > 0


@pytest.mark.parametrize("arg", [0, 1, 2])
def test_gradients_match_finite_differences(arg):
    f = lambda a: tae_gamma(*[200.0, 1.0, 4.0e5][:arg], a, *[200.0, 1.0, 4.0e5][arg + 1 :])  # noqa: E731
    x0 = [200.0, 1.0, 4.0e5][arg]
    h = 1e-4 * x0
    assert jax.grad(f)(x0) == pytest.approx(float((f(x0 + h) - f(x0 - h)) / (2 * h)), rel=1e-5)


def test_geometry_and_mode_gradients():
    """``jax.grad`` with respect to every geometry leaf and the mode, along a random direction, vs central FD."""
    f = lambda geo, phi: fax.growth_rate(4e5, phi, geo, HARM, fax.FastIons(koenies_density(), 200e3), RHO, **AXI)  # noqa: E731
    key = jax.random.PRNGKey(1)
    leaves, tree = jax.tree_util.tree_flatten((ITPA, TAE))
    du = [1e-2 * jnp.abs(x).mean() * jax.random.normal(jax.random.fold_in(key, i), x.shape) for i, x in enumerate(leaves)]
    du = jax.tree_util.tree_unflatten(tree, [d * 0 if i in (1, 2) else d for i, d in enumerate(du)])  # xm, xn are integers
    move = lambda t: jax.tree_util.tree_map(lambda x, d: x + t * d, (ITPA, TAE), du)  # noqa: E731
    grad = jax.grad(lambda p: f(*p))((ITPA, TAE))
    ad = sum(jnp.vdot(a, b) for a, b in zip(jax.tree_util.tree_leaves(grad), jax.tree_util.tree_leaves(du), strict=True))
    fd = (f(*move(1e-5)) - f(*move(-1e-5))) / 2e-5
    assert ad == pytest.approx(float(fd), rel=1e-5)


def test_slowing_down_is_normalized_with_its_smoothed_edge():
    """``int F d^3 v = 1`` for the erfc-smoothed birth edge (plain quadrature), and the edge drive ``dF/dE`` is its derivative."""
    ions = fax.FastIons(kind="slowing_down", birth_ev=300e3, critical_ev=100e3)
    m = ions.mass_amu * 1.66053906660e-27
    vb = np.sqrt(2 * 300e3 * 1.602176634e-19 / m)
    v = jnp.linspace(1e-6, 1.5 * vb, 400001)
    dfde, f = fax.drive._distribution(v, ions)
    assert float(fax.drive._integrate(4 * jnp.pi * v**2 * f, v)) == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(dfde[5:-1], np.gradient(f, v)[5:-1] / (m * v[5:-1]), rtol=1e-6, atol=1e-6 * float(jnp.abs(dfde).max()))
    with pytest.raises(ValueError, match="unknown FastIons kind"):
        fax.FastIons(kind="slowing-down")


def test_duplicated_toroidal_samples_leave_the_drive_unchanged():
    """The passing boundary uses the exact ``max |B|``: ``nzeta`` identical copies of an axisymmetric surface agree to 1e-10."""
    gam = [fax.growth_rate(4e5, TAE, ITPA, HARM, fax.FastIons(koenies_density(), 200e3), RHO, helical=0, nzeta=nz) for nz in (1, 4, 16)]
    assert gam[0] == pytest.approx(np.asarray(gam[1]), rel=1e-10) and gam[0] == pytest.approx(np.asarray(gam[2]), rel=1e-10)


def test_slowing_down_gradient_is_finite():
    f = lambda w: fax.growth_rate(
        w, TAE, ITPA, HARM, fax.FastIons(koenies_density(), birth_ev=300e3, critical_ev=100e3, kind="slowing_down"), RHO, **AXI
    )  # noqa: E731
    assert jnp.isfinite(f(4e5)) and jnp.isfinite(jax.grad(f)(4e5))


def test_ctok_global_tae_drive():
    """Full chain on CTOK: the FAX TAE (272.8 kHz, n = 2) from the shift-invert solve, then its drive by peaked deuterons."""
    geo = fax.read_boozmn(Path(__file__).parent / "data" / "boozmn_ctok.nc").full_mesh()
    harmonics, rho = fax.mode_family(2, 12), fax.mass_density(1e19, 2.0)
    ev = fax.eigenmodes(geo, harmonics, rho, target_hz=272.8e3, count=4)
    i = int(jnp.argmin(jnp.abs(ev.frequencies_hz - 272.8e3)))
    assert ev.frequencies_hz[i] == pytest.approx(272.8e3, rel=1e-3)
    omega = jnp.sqrt(ev.omega2[i])
    gam = [
        fax.growth_rate(omega, ev.structures[i], geo, harmonics, fax.FastIons(1e17 * (1 - geo.s) ** 2, t), rho, **AXI) for t in (25e3, 50e3)
    ]
    np.testing.assert_allclose(np.array(gam) / omega, [7.04e-4, 3.30e-3], rtol=1e-2)  # gamma / omega; j vs n/iota: 23% / 7% lower
    assert 0 < fax.ion_landau_damping(omega, ev.structures[i], geo, harmonics, 1e3, 1e19, mass_amu=2.0, **AXI) < 1e-4 * omega


# --- 3-D structure: the drive is built from the orbit's own harmonics (j, k) and the flux label, not from the mode's n ---

NS = 40
S40 = (jnp.arange(NS) + 0.5) / NS
MODE = jnp.stack([jnp.exp(-(((jnp.sqrt(S40) - 0.5) / 0.1) ** 2) / 2)] * 2, axis=1)
LOOSE = dict(nodes=16, ntheta=32, nzeta=32)  # not an eigenmode and not the TAE resonance: any mode and omega test the algebra


def torus(current=False):
    """Circular tokamak with q ~ 1.45 and, optionally, a toroidal current (I x30)."""
    geo = fax.circular_torus(
        s=S40, major_radius=10.0, minor_radius=1.0, B0=3.0, iota=1 / (1.45 + 0.1 * S40), epsilon_B=0.1, current=current
    )
    return replace(geo, I=30 * geo.I)


def relabel(geo, n):
    """Helical relabelling ``theta = theta' + N zeta``: a tokamak becomes a quasi-helically symmetric ``B(theta - N zeta)`` field."""
    return replace(geo, xn=geo.xn + n * geo.xm, iota=geo.iota + n, G=geo.G - n * geo.I, nfp=n)


def mirror(geo, angle):
    """The same device in ``-theta`` or ``-zeta`` coordinates (``iota -> -iota``, ``I -> -I``): negative-iota or n-reversed."""
    return replace(geo, iota=-geo.iota, I=-geo.I, **{"xm": -geo.xm} if angle == "theta" else {"xn": -geo.xn})


def drive(geo, harmonics, ions=None, phi=MODE, omega=4e5, **options):
    ions = ions or fax.FastIons(density=1e17 * jnp.exp(-8 * S40), temperature_ev=200e3)
    return fax.growth_rate(omega, phi, geo, harmonics, ions, RHO, **{**LOOSE, **options})


@pytest.mark.parametrize("current", [False, True])
@pytest.mark.parametrize("n", [2, 4])
def test_quasi_helical_relabelling_leaves_the_drive_unchanged(current, n):
    """``B(theta - N zeta)`` on ``N`` field periods and its relabelled axisymmetric twin are one device: gamma agrees to 4e-5 (the
    wrap of the small-divisor FFT on a finite grid). The gradient term carries the poloidal harmonic ``j``, which relabelling
    keeps; the mode's ``n_h / iota`` changes by up to 20% here (the previous form), and the gradient term is not small."""
    tok, harm = torus(current), ((10, 6), (11, 6))
    qh = relabel(tok, n)
    g0 = drive(tok, harm, helical=0, nzeta=1)
    g1 = drive(qh, tuple((m, k + m * n) for m, k in harm), helical=3)
    assert g1 == pytest.approx(np.asarray(g0), rel=1e-4)
    flat = fax.FastIons(density=1e17 * jnp.mean(jnp.exp(-8 * S40)), temperature_ev=200e3)
    assert abs(g0 / drive(tok, harm, flat, helical=0, nzeta=1) - 1) > 0.2


def test_axisymmetric_limit_ignores_toroidal_sidebands():
    """An axisymmetric field viewed as three field periods with helical sidebands is the ``helical = 0``, ``nzeta = 1`` result."""
    tok, harm = torus(), ((10, 6), (11, 6))
    assert drive(replace(tok, nfp=3), harm, helical=1) == pytest.approx(np.asarray(drive(tok, harm, helical=0, nzeta=1)), rel=1e-9)


@pytest.mark.parametrize("angle", ["theta", "zeta"])
def test_negative_iota_and_n_reversal_describe_the_same_device(angle):
    """``theta -> -theta`` (negative iota, ``m -> -m``) and ``zeta -> -zeta`` (``n -> -n``, ``iota -> -iota``) give the same gamma, and
    the second swaps the co- and counter-passing resonances ``(j, k) -> (j, -k)``: both passing directions carry drive."""
    qh = relabel(torus(True), 2)
    harm = ((10, 6 + 20), (11, 6 + 22))
    flipped = tuple((-m, n) for m, n in harm) if angle == "theta" else tuple((m, -n) for m, n in harm)
    (p0, g0), (p1, g1) = (drive(g, h, helical=3, resolved=True) for g, h in ((qh, harm), (mirror(qh, angle), flipped)))
    assert jnp.sum(g1) == pytest.approx(float(jnp.sum(g0)), rel=1e-12)
    d0 = dict(zip(map(tuple, p0.tolist()), g0.tolist(), strict=True))
    d1 = dict(zip(map(tuple, p1.tolist()), g1.tolist(), strict=True))
    iota = float(qh.iota[NS // 2])
    co = {jk: g for jk, g in d0.items() if jk[0] * iota - jk[1] > 0}
    counter = {jk: g for jk, g in d0.items() if jk[0] * iota - jk[1] < 0}
    assert max(map(abs, co.values())) > 1.0 and max(map(abs, counter.values())) > 1.0  # both directions carry drive
    if angle == "zeta":
        assert all(d1[(j, -k)] == pytest.approx(g, rel=1e-12, abs=1e-9) for (j, k), g in d0.items())
    else:
        assert all(d1[(-j, k)] == pytest.approx(g, rel=1e-12, abs=1e-9) for (j, k), g in d0.items())


def test_amplitude_phase_density_and_passivity():
    """gamma is independent of the mode's amplitude and global phase (but not of the harmonics' relative phase), linear in the fast-ion
    density, and a spatially uniform Maxwellian or slowing-down distribution is passive: gamma <= 0 at every frequency."""
    qh, harm = relabel(torus(), 2), ((10, 6 + 20), (11, 6 + 22))
    base = drive(qh, harm, helical=3)
    c = 3.7 * jnp.exp(0.9j)
    assert drive(qh, harm, phi=c * MODE.astype(complex), helical=3) == pytest.approx(np.asarray(base), rel=1e-12)
    twist = MODE.astype(complex) * jnp.array([1.0, jnp.exp(1.2j)])
    assert abs(drive(qh, harm, phi=twist, helical=3) / base - 1) > 1e-2
    ions = fax.FastIons(density=1e17 * jnp.exp(-8 * S40), temperature_ev=200e3)
    scaled = replace(ions, density=3.0 * ions.density)
    assert drive(qh, harm, scaled, helical=3) == pytest.approx(np.asarray(3 * base), rel=1e-12)
    for kind in ("maxwellian", "slowing_down"):
        uniform = fax.FastIons(density=1e17, temperature_ev=200e3, birth_ev=300e3, critical_ev=100e3, kind=kind)
        assert all(drive(qh, harm, uniform, omega=w, helical=3) < 0 for w in (2e5, 4e5, 8e5))


# --- regularizations are explicit parameters and gamma converges as each goes to zero ---


def helical_torus(iota, eps=0.1, delta=0.05):
    """Analytic ``B = B0 (1 - eps sqrt(s) cos(theta) - delta sqrt(s) cos(theta - zeta))``, one field period: a helical (2, 1) product
    harmonic of 1/|B| makes ``iota k_theta + k_zeta`` vanish at ``iota = 1/2``."""
    g, z = fax.circular_torus(s=S40, major_radius=10.0, minor_radius=1.0, B0=3.0, iota=iota, epsilon_B=eps), jnp.zeros_like(S40)
    pad = lambda a: jnp.concatenate([a, z[:, None]], 1)
    pads = {k: pad(getattr(g, k)) for k in ("rmnc", "rmns", "zmnc", "zmns", "pmnc", "pmns", "bmns", "gmnc", "gmns")}
    bmnc = jnp.stack([3.0 + z, -3 * eps * jnp.sqrt(S40), z, -3 * delta * jnp.sqrt(S40)], 1)
    return replace(g, xm=jnp.array([0.0, 1, 2, 1]), xn=jnp.array([0.0, 0, 0, 1]), bmnc=bmnc, **pads)


def test_passing_boundary_and_node_regularizations_converge():
    """``boundary`` removes the layer ``lambda > (1 - boundary) / B_max`` (the log-singular trapped-passing edge); gamma converges to
    the 256-node value as it goes to 0 (relative error 1.4e-1, 1.8e-2, 1.0e-3, 2e-6 at 0.1, 0.01, 0.001, 1e-4: first order up to a
    logarithm) and in the Gauss-Legendre node count (3.5e-3, 9e-4, 3.5e-4, 2e-5 at 8, 16, 32, 64)."""
    tok, harm, opt = torus(), ((10, 6), (11, 6)), dict(helical=0, nzeta=1, ntheta=64)
    ref = drive(tok, harm, nodes=256, **opt)
    err = [abs(drive(tok, harm, nodes=64, boundary=b, **opt) / ref - 1) for b in (1e-1, 1e-2, 1e-3, 1e-4)]
    assert err[0] < 0.2 and err[1] < 2.5e-2 and err[2] < 1.5e-3 and err[3] < 1e-4 and err == sorted(err, reverse=True)
    nodes = [abs(drive(tok, harm, nodes=n, **opt) / ref - 1) for n in (8, 16, 32, 64)]
    assert nodes[0] < 5e-3 and nodes[3] < 1e-4 and nodes == sorted(nodes, reverse=True)


def test_birth_edge_smoothing_converges_quadratically():
    """The erfc-smoothed birth edge of a slowing-down distribution has relative width ``birth_width``; gamma converges as it goes to 0
    with error ``O(width^2)`` (successive differences shrink by 4 - 5 per halving) once ``nodes >~ 2 / width`` resolves the edge."""
    tok, harm = torus(), ((10, 6), (11, 6))
    dens = 1e17 * jnp.exp(-8 * S40)
    g = [
        drive(
            tok,
            harm,
            fax.FastIons(dens, kind="slowing_down", birth_ev=300e3, critical_ev=100e3, birth_width=w),
            helical=0,
            nzeta=1,
            ntheta=64,
            nodes=512,
        )
        for w in (0.08, 0.04, 0.02, 0.01)
    ]
    ratio = [(g[i] - g[i + 1]) / (g[i + 1] - g[i + 2]) for i in range(2)]
    assert all(4.0 < r < 5.5 for r in ratio) and abs(g[3] / (g[3] - (g[2] - g[3]) / 3) - 1) < 1e-3


def test_small_divisor_guard_converges_and_is_sensitive_near_rational_surfaces():
    """``guard`` in ``kappa / (kappa^2 + guard^2)``, the small-divisor cutoff of the orbit-angle inversion ``(iota d_theta + d_zeta) chi
    = h / <h> - 1`` (nothing to cut in axisymmetry: ``kappa = iota k_theta`` only vanishes at ``k_theta = 0``). Relative error against
    ``guard = 1e-6`` at ``guard = 0.1, 0.01, 0.001, 0.0001``: away from low-order rationals 6e-3, 8e-5, 1e-7, 1e-8; with
    ``iota = 1/2`` crossed at ``s = 1/4`` (shear 0.3) 5e-3, 2e-4, 1e-5, 1e-7; at ``iota = 1/2`` over the whole mode (shear 0.02) the
    resonant (2, -1) component of ``h`` is in the sum for every surface and the error is 2e-2, 1e-2, 3e-2, 2e-3."""
    harm, h = ((1, 1), (2, 1)), dict(helical=1, sidebands=2, ntheta=32, nzeta=16)
    rates = {"away": 0.43 + 0.03 * S40, "crossing": 0.5 + 0.3 * (S40 - 0.25), "flat": 0.5 + 0.02 * (S40 - 0.25)}
    err = {}
    for name, iota in rates.items():
        geo = helical_torus(iota)
        ref = drive(geo, harm, guard=1e-6, **h)
        err[name] = [abs(drive(geo, harm, guard=x, **h) / ref - 1) for x in (1e-1, 1e-2, 1e-3, 1e-4)]
    assert err["away"] == sorted(err["away"], reverse=True) and err["away"][2] < 1e-6
    assert err["crossing"][0] < 1e-2 and err["crossing"][1] < 5e-4 and err["crossing"][3] < 1e-6
    assert err["flat"][2] > 1e-2 > err["flat"][3] > 1e-4  # the sensitive case: guard must go below the shear-limited |kappa| ~ 1e-3


# --- ITPA ladder, rung 2: fixed-mode zero-orbit-width, passing particles only ---

# CAS3D-K (ZOW), Koenies et al. NF 58 126027 (2018), fig. 1a dotted curve, read from the vector figure: (T_f / keV, gamma / 1e3 s^-1);
# the points hidden behind other markers (100-135 keV) are not used. Passing and trapped content of the curve is not stated in the paper.
ZOW = {82: 4.26, 139: 14.60, 163: 19.98, 191: 25.17}


def test_zow_curve_is_bracketed_by_the_passing_only_drive():
    """FAX is passing-only; the Koenies CAS3D-K (ZOW) curve is stated to have no FOW and no FLR but its trapped content is not given, so
    this is an unmatched comparison: FAX / CAS3D-K = 0.58 - 0.60 from 82 to 191 keV (a constant share, as expected for a missing
    trapped contribution of about 40%, which is not established). Frozen-mode validity: |gamma| / omega = 6e-3 - 4e-2 (omega = 4.0e5)."""
    for t, zow in ZOW.items():
        gam = float(tae_gamma(float(t)))
        assert 0.5 < gam / (zow * 1e3) < 0.7 and gam / 4e5 < 0.05
