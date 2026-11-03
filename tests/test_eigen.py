"""Global operator gates: continuum accumulation, AE3D CTOK parity, Könies ITPA TAE, gradients."""

import itertools
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fax
from fax.continuum import MU0, generalized_eigh
from fax.eigen import (
    Blocks,
    _assemble,
    _certified,
    _coefficients,
    _modes,
    dense,
    equilibrium_currents,
    force_balance_residual,
    global_matrices,
    targeted_eigenpairs,
    targeted_eigenvalues,
)
from fax.geometry import jacobian_check, surface_grid

DATA = Path(__file__).parent / "data"


def konies(N):
    """Könies et al. (2018) ITPA TAE: R0 = 10 m, a = 1 m, B0 = 3 T, q = 1.71 + 0.16 r^2, n = 6, hydrogen 2e19."""
    s = (jnp.arange(N) + 0.5) / N
    return fax.circular_torus(s=s, major_radius=10.0, minor_radius=1.0, B0=3.0, iota=1 / (1.71 + 0.16 * s), epsilon_B=0.1)


def test_uniform_field_spectrum_is_a_continuum():
    rho = fax.mass_density(1e19, 2.0)
    edges = np.array([0.4, 0.7]) * 2.7 / 5.5 / np.sqrt(MU0 * rho)  # |n - m iota| v_A / R at s = 0, 1
    spread = []
    for N in (40, 160):
        geo = fax.circular_torus(s=(jnp.arange(N) + 0.5) / N, iota=0.3 + 0.3 * (jnp.arange(N) + 0.5) / N)
        modes = fax.eigenmodes(geo, jnp.array([[1.0, 1.0]]), rho, ntheta=8, nzeta=4)
        w = np.sqrt(np.asarray(modes.omega2))
        assert edges[0] < w.min() < edges[0] * (1 + 0.5 / N) and edges[1] * (1 - 0.5 / N) < w.max() < edges[1]
        jump = np.abs(np.diff(np.asarray(modes.structures[:, :, 0]), axis=1)) ** 2  # singular: phi' localizes
        spread.append(np.median(jump.sum(1) ** 2 / (jump**2).sum(1)))
    assert spread[0] == pytest.approx(spread[1], rel=0.05)  # width in cells is fixed, so it shrinks as 1/N


def test_ae3d_ctok_parity():
    """AE3D's CTOK run (same boozmn, n = 2, m = 0..12, D at 1e19 m^-3) on its 99-point full mesh."""
    geo = fax.read_boozmn(DATA / "boozmn_ctok.nc").full_mesh()
    geo = jax.tree_util.tree_map(lambda a: jnp.concatenate([a, 2 * a[-1:] - a[-2:-1]]) if a.ndim and a.shape[0] == 98 else a, geo)
    F, G = (np.asarray(dense(A)) for A in global_matrices(geo, fax.mode_family(2, 12), fax.mass_density(1e19, 2.0), 32, 4))
    w, x = (np.asarray(a) for a in generalized_eigh(F, G))
    x = x / np.linalg.norm(x, axis=0)
    ref = np.load(DATA / "ae3d_ctok_modes.npz")
    v = ref["vectors"].reshape(len(ref["khz"]), -1).astype(float)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    khz = np.sqrt(w) / 2e3 / np.pi
    overlap = np.abs(v @ x)
    best = overlap.argmax(1)
    rel = khz[best] / ref["khz"] - 1
    np.testing.assert_array_less(np.abs(rel[1:]), 0.015)  # six global modes, overlap > 0.9
    assert overlap[np.arange(len(best)), best].min() > 0.9
    tae = v[0]  # AE3D's 337.9 kHz TAE: FAX 272.8; AE3D's iota' blocks differ from the energy (test_ae3d_v31_blocks...)
    assert np.sqrt((tae @ F @ tae).real / (tae @ G @ tae).real) / 2e3 / np.pi == pytest.approx(ref["khz"][0], rel=0.03)


def _ae3d_blocks(field, prof, h, shear=True, metric_mask=None, iota_only=False):
    """FAX coefficient blocks ``(A, B, D)`` of the bending and the inertia form from AE3D's ``ae_metric.dat`` data.

    ``field`` is ``(surfaces, theta, 11)`` (columns of ``ae_metric.dat``; the axisymmetric CTOK data do not depend on zeta) and
    ``prof`` ``(surfaces, 8)`` (``iota``, ``iota'``, ``jpol = G``, ``jpol'``, ``jtor = -I``, ``jtor'``, ...). ``metric_mask``
    keeps only the listed contravariant ``g^ij`` components; ``iota_only`` keeps ``c_s = i iota' m B / D`` alone.
    """
    S, T, nz = field.shape[0], field.shape[1], 4
    rj, B, gss, gtt, gzz, gst, gsz, gtz, Bs, Bt, Bz = (np.broadcast_to(field[:, :, i, None], (S, T, nz)).astype(float) for i in range(11))
    iota, iota_s, G, G_s, I, I_s = (prof[:, i] for i in range(6))
    D, D_s = G - iota * I, G_s - iota_s * I - iota * I_s  # G + iota I with I = -jtor
    metric = np.stack([np.stack([gss, gst, gsz], -1), np.stack([gst, gtt, gtz], -1), np.stack([gsz, gtz, gzz], -1)], -2)
    if metric_mask is not None:
        metric = metric * np.array(metric_mask, float)
    theta, zeta = np.meshgrid(2 * np.pi * np.arange(T) / T, 2 * np.pi * np.arange(nz) / nz, indexing="ij")
    dB = np.stack([0 * Bs if iota_only else Bs, Bt, Bz], -1)
    g = SimpleNamespace(theta=theta, zeta=zeta, B=B, dB=dB, sqrt_g=rj, metric=metric)
    inertia, bending = _modes(g, iota, iota_s if shear else 0 * iota_s, D, 0 * D_s if iota_only else D_s, h)
    w, rho = jnp.abs(rj), 2 * 1.67e-27 * 1e19
    return _coefficients(metric, w, *bending), _coefficients(metric, MU0 * rho * w / B**2, *inertia)


def test_ae3d_v31_blocks_and_energy_on_identical_inputs():
    """Pinned AE3D v31 comparison for the CTOK TAE (docs/validation.md): bending, shear and inertia blocks.

    ``tests/data/ae3d_v31_ctok.npz`` holds AE3D's own metric input (``ae_metric.dat``) and the blocks and assembled
    bending matrix of the pinned build (commit 383d327, gfortran 15.2.0, instrumented by ``ae3d_v31_instrumentation.patch`` to
    write its blocks, with the current and pressure terms off). Fed the same input, FAX reproduces AE3D's inertia blocks, its
    ``f'f'`` and ``ff`` bending blocks without shear, and the symmetric part of its ``iota'`` coupling. AE3D's antisymmetric
    ``iota'`` ``f'f`` coupling has the opposite sign, and its ``iota'`` ``ff`` block omits the ``a' iota'`` cross term and has the
    ``g^{s theta} iota'`` terms with the opposite sign. A real-space quadrature of ``int |grad u|^2`` on the same input agrees
    with FAX's assembled ``F`` (not with AE3D's).
    """
    ref = np.load(DATA / "ae3d_v31_ctok.npz")
    h = jnp.asarray([[m, 2.0] for m in range(13)])
    real = lambda x: np.asarray(x).real  # noqa: E731
    sym = lambda x: 0.5 * (x + x.swapaxes(1, 2))  # noqa: E731
    asym = lambda x: 0.5 * (x - x.swapaxes(1, 2))  # noqa: E731
    rel = lambda x, y: np.linalg.norm(x - y) / np.linalg.norm(y)  # noqa: E731
    (A0, B0, D0), (Ai, Bi, Di) = (
        tuple(map(real, blocks)) for blocks in _ae3d_blocks(ref["field_pair"], ref["profiles_pair"], h, shear=False)
    )
    (A1, B1, D1), _ = _ae3d_blocks(ref["field_pair"], ref["profiles_pair"], h)
    A1, B1, D1 = map(real, (A1, B1, D1))
    c = (ref["ae_a"] * Ai).sum() / (Ai * Ai).sum()  # AE3D's units: kHz^2 and its inertia scale
    cF = c / (2e3 * np.pi) ** 2
    assert rel(ref["ae_a"], c * Ai) < 1e-9 and rel(ref["ae_d"], c * Di) < 1e-9
    assert rel(ref["ae_b"], c * Bi.swapaxes(1, 2)) < 1e-9  # AE3D's b(i, j) is the coefficient of f_i f_j'
    assert rel(ref["ae_aa"], cF * A0) < 1e-5 and rel(ref["ae_dd0"], cF * D0) < 1e-5
    dbb, dB = ref["ae_bbS"] - ref["ae_bb0"], B1 - B0
    assert rel(sym(dbb), cF * sym(dB)) < 1e-4  # the symmetric iota' part agrees
    assert rel(asym(dbb), cF * asym(dB)) < 1e-3 and rel(asym(dbb), -cF * asym(dB)) > 1.9  # the antisymmetric part: opposite sign
    mask = lambda *ij: [[1.0 if (a, b) in ij or (b, a) in ij else 0.0 for b in range(3)] for a in range(3)]  # noqa: E731
    D_ss = _ae3d_blocks(ref["field_pair"], ref["profiles_pair"], h, metric_mask=mask((0, 0)), iota_only=True)[0][2]
    D_st = [real(_ae3d_blocks(ref["field_pair"], ref["profiles_pair"], h, shear, mask((0, 1)))[0][2]) for shear in (True, False)]
    dD = ref["ae_ddS"] - ref["ae_dd0"]
    assert rel(sym(dD), cF * (real(D_ss) - (D_st[0] - D_st[1]))) < 1e-4  # AE3D: iota'^2 g^ss, no cross term, -g^st iota'
    assert rel(sym(dD), cF * (D1 - D0)) > 0.1  # the energy functional has the cross term and +g^st iota'

    field, prof, s = ref["field_all"], ref["profiles_all"], ref["s"]
    (A, B, D), _ = _ae3d_blocks(field, prof, h)
    F_fax = dense(_assemble(jnp.asarray(s), jnp.arange(13) == 0, A, B, D)).real
    diag, upper = ref["ae_F_diag"].astype(float), ref["ae_F_upper"].astype(float)
    F_ae3d = np.zeros((99 * 13, 99 * 13))
    for k in range(99):
        F_ae3d[13 * k : 13 * k + 13, 13 * k : 13 * k + 13] = diag[k]
        if k < 98:
            F_ae3d[13 * k : 13 * k + 13, 13 * k + 13 : 13 * k + 26] = upper[k]
            F_ae3d[13 * k + 13 : 13 * k + 26, 13 * k : 13 * k + 13] = upper[k].T
    T = field.shape[1]
    th, ze = np.meshgrid(2 * np.pi * np.arange(T) / T, 2 * np.pi * np.arange(T) / T, indexing="ij")
    rj, Bf, *gs, Bs, Bt, Bz = (field[:, :, i, None] * np.ones(T) for i in range(11))
    metric = np.stack([np.stack([gs[0], gs[3], gs[4]], -1), np.stack([gs[3], gs[1], gs[5]], -1), np.stack([gs[4], gs[5], gs[2]], -1)], -2)
    iota, Dtot = prof[:, 0], prof[:, 2] - prof[:, 0] * prof[:, 4]
    k = np.fft.fftfreq(T, 1 / T)
    d_ang = lambda f, a: np.fft.ifft(1j * k.reshape(-1 if a == 1 else 1, -1 if a == 2 else 1) * np.fft.fft(f, axis=a), axis=a).real  # noqa: E731
    rng = np.random.default_rng(1)
    x = 2 * s - 1
    errors = {"FAX": [], "AE3D": []}
    for _ in range(5):  # smooth potentials: five Chebyshev modes per harmonic, vanishing at the axis and the edge
        y = np.zeros((99, 13))
        for j in (2, 3, 4, 5):
            y[:, j] = (s * (1 - s)) ** 4 * np.polynomial.chebyshev.chebval(x, rng.normal(size=4))
        phi = (y[:, :, None, None] * np.cos(np.arange(13)[None, :, None, None] * th - 2 * ze)).sum(1)
        u = (Bf / Dtot[:, None, None]) * (iota[:, None, None] * d_ang(phi, 1) + d_ang(phi, 2))
        grad = np.stack([np.gradient(u, s, axis=0, edge_order=2), d_ang(u, 1), d_ang(u, 2)], -1)
        density = (np.einsum("...i,...ij,...j", grad, metric, grad) * np.abs(rj)).mean((1, 2))
        energy = ((density[1:] + density[:-1]) * np.diff(s)).sum() / 2
        errors["FAX"].append(0.5 * (y.ravel() @ F_fax @ y.ravel()) / energy - 1)
        errors["AE3D"].append(0.5 * (y.ravel() @ F_ae3d @ y.ravel()) / cF / energy - 1)
    assert np.abs(errors["FAX"]).max() < 5e-3 and np.abs(errors["AE3D"]).max() > 0.05 and np.abs(errors["AE3D"]).min() > 5e-3


def test_konies_tae_frequency_and_convergence():
    h, rho = fax.mode_family(6, 16, 5), fax.mass_density(2e19, 1.0)
    omega = []
    for N in (50, 100, 200):
        modes = fax.eigenmodes(konies(N), h, rho, target_hz=6.46e4, count=3, ntheta=32, nzeta=4)
        omega.append(float(np.sqrt(modes.omega2[-1])))
    assert 4.01e5 < omega[-1] < 4.13e5  # CAS3D (shifted equilibrium) and GYGLES (ad hoc, as here)
    assert abs(omega[1] / omega[2] - 1) < 2e-4 and abs(omega[0] / omega[2] - 1) < 5e-4


def test_shift_invert_matches_dense_and_gradients_match_finite_differences():
    h, rho = fax.mode_family(6, 13, 9), fax.mass_density(2e19, 1.0)
    dense_w = fax.eigenmodes(konies(40), h, rho, ntheta=16, nzeta=4).omega2
    target = fax.eigenmodes(konies(40), h, rho, target_hz=6.46e4, count=4, ntheta=16, nzeta=4).omega2
    np.testing.assert_allclose(target, dense_w[np.sort(np.argsort(np.abs(dense_w - (2 * np.pi * 6.46e4) ** 2))[:4])], rtol=1e-10)

    def f(p, finite_beta=False, **options):
        s, pressure = konies(40).s, [7.17e3, -6.811e3, -3.585e2] if finite_beta else None
        geo = fax.circular_torus(s=s, major_radius=10.0, B0=3.0, iota=1 / (1.71 + 0.16 * s) + p[1], epsilon_B=0.1, pressure=pressure)
        return fax.frequency_hz(geo, h, rho * p[0], ntheta=16, nzeta=4, finite_beta=finite_beta, **options)

    p = jnp.array([1.0, 0.0])
    targeted = {"target_hz": 6.46e4, "count": 4, "start": 3, "stop": 4}
    for options in ({"start": 2, "stop": 5}, targeted, {**targeted, "finite_beta": True}):
        grad = jax.grad(f)(p, **options)
        fd = jnp.array([(f(p + e, **options) - f(p - e, **options)) / 2e-7 for e in 1e-7 * jnp.eye(2)])
        np.testing.assert_allclose(grad, fd, rtol=1e-5)
        np.testing.assert_allclose(grad[0], -0.5 * f(p, **options), rtol=1e-10)  # f ~ rho^(-1/2)


def _embedding(geo, n=2, T=32):
    """Grid, spectral/FD gradient, embedding basis, metric and test potential for the quadrature checks."""
    s, xm, xn = (np.asarray(a) for a in (geo.s, geo.xm, geo.xn))
    S, Th, Ze = np.meshgrid(s, *2 * [np.linspace(0, 2 * np.pi, T, endpoint=False)], indexing="ij")

    def field(c, sn):
        ang = xm * Th[..., None] - xn * Ze[..., None]
        return (np.asarray(c)[:, None, None] * np.cos(ang) + np.asarray(sn)[:, None, None] * np.sin(ang)).sum(-1)

    def grad(f):  # d/ds by second-order differences, d/dtheta and d/dzeta spectrally
        k = lambda a: np.fft.fftfreq(T, 1 / T).reshape([-1 if i == a else 1 for i in range(f.ndim)])  # noqa: E731
        spec = [np.fft.ifft(1j * k(a) * np.fft.fft(f, axis=a), axis=a).real for a in (1, 2)]
        return np.stack([np.gradient(f, s, axis=0, edge_order=2), *spec], -1)

    R, Zc, P, B = (field(geo.rmnc, geo.rmns), field(geo.zmnc, geo.zmns), field(geo.pmnc, geo.pmns), field(geo.bmnc, geo.bmns))
    X = np.stack([R * np.cos(Ze + P), R * np.sin(Ze + P), Zc], -1)
    e = np.swapaxes(grad(X), -1, -2)  # covariant basis r_i
    g, iota = e @ np.swapaxes(e, -1, -2), np.asarray(geo.iota)[:, None, None]
    Bz = B / np.sqrt(iota**2 * g[..., 1, 1] + 2 * iota * g[..., 1, 2] + g[..., 2, 2])  # B^theta = iota B^zeta
    f = {3: np.sin(np.pi * s) ** 8, 4: -2 * s * np.sin(np.pi * s) ** 8}
    phi = sum(f[m][:, None, None] * np.cos(m * Th - n * Ze) for m in f)
    u = Bz * (iota * grad(phi)[..., 1] + grad(phi)[..., 2]) / B
    y = np.zeros((len(s), 5))
    for m in f:
        y[:, m] = f[m]
    trapz = lambda a: ((a[1:] + a[:-1]) * np.diff(s)).sum() / 2  # noqa: E731
    return grad, e, g, iota, B, Bz, phi, u, y.ravel(), lambda a: trapz(a.mean((1, 2)))


def test_ctok_bending_energy_matches_real_space_quadrature():
    """F and G against finite differences of phi on the boozmn embedding; B^i from |B|, iota and g_ij only."""
    geo = fax.read_boozmn(DATA / "boozmn_ctok.nc")
    grad, e, g, iota, B, Bz, phi, u, y, integral = _embedding(geo)
    gi, J = np.linalg.inv(g), np.abs(np.linalg.det(e))
    rho = float(fax.mass_density(1e19, 2.0))
    energy = lambda a, w: integral(np.einsum("...i,...ij,...j", grad(a), gi, grad(a)) * w)  # noqa: E731
    F, G = (np.asarray(dense(A)) for A in global_matrices(geo, fax.mode_family(2, 4), rho, 32, 4))
    assert 0.5 * (y @ F @ y).real == pytest.approx(energy(u, J), rel=5e-3)
    assert 0.5 * (y @ G @ y).real == pytest.approx(energy(phi, MU0 * rho * J / B**2), rel=5e-3)


@pytest.mark.parametrize("case", ["ctok", "pressure"])
def test_finite_beta_terms_match_real_space_quadrature(case):
    """Kink and pressure terms against quadrature of J.B (curl of B_i on the grid) and the curvature.

    CTOK takes B_i and kappa = (b.grad) b from the embedding alone; the pressure case (Könies with
    its p(s)) takes B_i = (B_s, I, G) and kappa = grad_perp(mu0 p + B^2/2) / B^2 with the given p.
    """
    pressure = [7.17e3, -6.811e3, -3.585e2]
    geo = fax.read_boozmn(DATA / "boozmn_ctok.nc") if case == "ctok" else konies(128)
    if case == "pressure":
        geo = fax.circular_torus(s=geo.s, major_radius=10.0, B0=3.0, iota=geo.iota, epsilon_B=0.1, pressure=pressure)
    grad, e, g, iota, B, Bz, phi, u, y, integral = _embedding(geo)
    gi, sg = np.linalg.inv(g), np.linalg.det(e)
    ps, Bs, sig = (np.asarray(a) for a in equilibrium_currents(geo, surface_grid(geo, 32, 32)))
    up = lambda a: np.einsum("...i,...ij,...jc->...c", a, gi, e)  # noqa: E731 - covariant components -> vector
    s = np.asarray(geo.s)[:, None, None]
    if case == "ctok":
        Bv = Bz[..., None] * (iota[..., None] * e[..., 1, :] + e[..., 2, :])
        Bc, b = np.einsum("...ij,...j->...i", e, Bv), Bv / B[..., None]
        db = np.stack([grad(b[..., c]) for c in range(3)], -2)
        kappa, ps = (Bz / B)[..., None] * (iota[..., None] * db[..., 1] + db[..., 2]), ps[:, None, None]
    else:
        Bc = np.stack([Bs, *np.broadcast_arrays(*(np.asarray(a)[:, None, None] for a in (geo.I, geo.G)), Bs)[:2]], -1)
        b = up(Bc) / B[..., None]
        np.testing.assert_allclose(ps, MU0 * (pressure[1] + 2 * pressure[2] * s[:, 0, 0]), rtol=0.02)  # solvability
        ps = ps[:, None, None]
        dX = up(grad(B**2 / 2)) + ps[..., None] * up(np.stack([np.ones_like(B), 0 * B, 0 * B], -1))
        kappa = (dX - b * (b * dX).sum(-1, keepdims=True)) / B[..., None] ** 2
    d = [grad(Bc[..., k]) for k in range(3)]
    curl = np.stack([d[2][..., 1] - d[1][..., 2], d[0][..., 2] - d[2][..., 0], d[1][..., 0] - d[0][..., 1]], -1)
    sigma = (curl * Bc).sum(-1) / (sg * B**2)  # mu0 J.B / B^2
    np.testing.assert_allclose(sig[2:-2], sigma[2:-2], atol=2e-3 * np.abs(sigma).max())
    dphi, du = grad(phi), grad(u)
    kink = integral(sigma * (Bc * np.cross(dphi, du)).sum(-1) / B * np.sign(sg))
    drive = (Bc * np.cross(dphi, [1.0, 0, 0])).sum(-1) / (B * sg)  # (b x grad phi).grad s
    pres = integral(-2 * ps * drive * (kappa * np.cross(b, up(dphi))).sum(-1) / B**2 * np.abs(sg))
    rho = float(fax.mass_density(1e19, 2.0))
    F0, F1 = (np.asarray(dense(global_matrices(geo, fax.mode_family(2, 4), rho, 32, 4, finite_beta=x)[0])) for x in (False, True))
    assert 0.5 * (y @ (F1 - F0) @ y).real == pytest.approx(kink + pres, rel=5e-3)
    assert abs(pres) < 1e-4 * abs(kink) if case == "ctok" else abs(pres) > 0.1 * abs(kink)
    # The del-sigma form plus the omitted curl(b) term equal the direct form (docs/theory.md, section 4; FD-limited to 1%).
    vol, gphi = np.abs(sg), up(dphi)
    ksig = integral(-u * (Bc * np.cross(dphi, grad(sigma))).sum(-1) / B * np.sign(sg))
    omitted = integral(-u * sigma * (sigma * (b * gphi).sum(-1) + (np.cross(b, kappa) * gphi).sum(-1)) * vol)
    assert kink == pytest.approx(ksig + omitted, rel=1e-2)
    if case == "ctok":  # next order, as fractions of the bending energy: the two forms, the kappa terms of the bending energy
        quad = lambda a, c: np.einsum("...i,...ij,...j", a, gi, c) * vol  # noqa: E731
        kcov = np.einsum("...ic,...c->...i", e, kappa) * u[..., None]
        bend = integral(quad(du, du))
        assert (kink - ksig) / bend == pytest.approx(-0.0069, abs=1e-3) and omitted / bend == pytest.approx(-0.0073, abs=1e-3)
        assert integral(quad(du - kcov, du - kcov) - quad(du, du)) / bend == pytest.approx(7e-4, abs=3e-4)
        assert kink / bend == pytest.approx(-0.2265, abs=5e-3)


def test_kink_forms_differ_by_the_curl_b_term():
    """Periodic NumPy checks of the two forms of the kink term (docs/theory.md, section 4), no FAX code.

    ``K = int sigma b.(grad phi~* x grad u)`` equals ``-int u b.(grad phi~* x grad sigma)`` only with the extra term
    ``-int u sigma (curl b).grad phi~*``: for ``b = (cos z, sin z, 0)``, ``sigma = -1``, ``phi = sin x`` they are -1/4 and 0.
    With the full ``Q = curl(A b)`` (``grad u -> grad u - kappa u``, ``kappa = b.grad b``) the kappa terms cancel:
    ``K_full = K_gradsigma - int sigma^2 u u~*``. Spectral derivatives on a smooth random unit ``b`` (``sigma = b.curl b``).
    """
    n = 16
    x, z = (2 * np.pi * np.arange(n) / n).reshape(-1, 1), (2 * np.pi * np.arange(n) / n).reshape(1, -1)
    b = np.stack([np.cos(z) + 0 * x, np.sin(z) + 0 * x, 0 * x + 0 * z], -1)
    gphi, gu = (
        np.stack([np.cos(x) + 0 * z, 0 * x + 0 * z, 0 * x + 0 * z], -1),
        np.stack([-np.cos(z) * np.sin(x), 0 * x + 0 * z, -np.sin(z) * np.cos(x)], -1),
    )
    u = np.cos(z) * np.cos(x)  # b.grad(phi)
    assert (-np.sum(b * np.cross(gphi, gu), -1)).mean() == pytest.approx(-0.25, abs=1e-14)
    assert (u * np.sum(b * gphi, -1) * -1.0).mean() == pytest.approx(-0.25, abs=1e-14)  # -u sigma (curl b).grad(phi) with curl b = -b

    N, rng = 32, np.random.default_rng(3)
    X = np.meshgrid(*3 * [2 * np.pi * np.arange(N) / N], indexing="ij")
    k = np.fft.fftfreq(N, 1 / N)

    def grad(f):
        d = np.stack([np.fft.ifftn(1j * k.reshape(np.roll([-1, 1, 1], a)) * np.fft.fftn(f)) for a in range(3)], -1)
        return d if np.iscomplexobj(f) else d.real

    def wave(modes, real=False):  # random sum of plane waves with random phases
        phase = lambda c: sum(q * x for q, x in zip(c, X, strict=True)) + rng.uniform(0, 6)  # noqa: E731
        return sum(rng.normal() * (np.cos(phase(c)) if real else np.exp(1j * phase(c))) for c in modes)

    v = 1 + 0.06 * np.stack([wave(itertools.product((-1, 0, 1), repeat=3), real=True) for _ in range(3)], -1)
    b = v / np.linalg.norm(v, axis=-1, keepdims=True)
    Jb = np.stack([sum(b[..., j] * grad(b[..., i])[..., j] for j in range(3)) for i in range(3)], -1)  # kappa
    gb = [grad(b[..., i]) for i in range(3)]
    curl = np.stack([gb[2][..., 1] - gb[1][..., 2], gb[0][..., 2] - gb[2][..., 0], gb[1][..., 0] - gb[0][..., 1]], -1)
    sigma = np.sum(b * curl, -1)
    modes = list(itertools.product(range(-2, 3), repeat=3))
    phi, phit = wave(modes), wave(modes)
    u, ut = np.sum(b * grad(phi), -1), np.sum(b * grad(phit), -1)
    gpc, mean = np.conj(grad(phit)), lambda a: a.mean()  # noqa: E731
    direct = mean(sigma * np.sum(b * np.cross(gpc, grad(u)), -1))
    gradsigma = mean(-u * np.sum(b * np.cross(gpc, grad(sigma)), -1))
    omitted = mean(-u * sigma * np.sum(curl * gpc, -1))
    assert direct == pytest.approx(gradsigma + omitted, abs=2e-6 * abs(direct))
    full = mean(sigma * np.sum(b * np.cross(gpc, grad(u) - Jb * u[..., None]), -1))
    assert full == pytest.approx(gradsigma - mean(sigma**2 * u * np.conj(ut)), abs=2e-6 * abs(full))
    assert abs(omitted) > 0.05 * abs(direct)  # not a small correction for this field


def test_conventions_and_geometry_diagnostics_on_ctok():
    """Section 1 of docs/theory.md pinned on the CTOK embedding: orientation, ``G``, ``I``, ``iota`` and ``k_par`` in real space
    and in the assembled mode vectors, the flux factor ``psi' = Phi_edge / 2 pi`` and the force-balance residual.

    ``psi'`` is the input's ``PHIEDGE / 2 pi`` (AE3D ``phipf``, 0.0450886). The first surface (``s = 0.005``) is 15% low: the
    radial finite difference of ``R ~ sqrt(s)`` at the axis.
    """
    geo = fax.read_boozmn(DATA / "boozmn_ctok.nc")
    g = surface_grid(geo, 32, 32)
    D = geo.G + geo.iota * geo.I
    assert jnp.all(g.sqrt_g < 0) and jnp.all(geo.G > 0) and jnp.all(geo.iota > 0) and jnp.all(D > 0)  # (s, theta, zeta) left-handed
    psi, spread = jacobian_check(geo)
    np.testing.assert_allclose(psi[5:], 0.0450886, rtol=5e-3)
    assert spread[5:].max() < 1e-4 and spread.max() < 1e-3 and psi[0] < 0.88 * 0.0450886
    assert force_balance_residual(geo)[5:].max() < 1e-4 and force_balance_residual(geo).max() < 1e-3

    grad, e, gij, iota, B, Bz, phi, u, y, integral = _embedding(geo)  # Bz = B^zeta from B^theta = iota B^zeta
    cov = np.einsum("...ij,...j->...i", gij, np.stack([0 * Bz, iota * Bz, Bz], -1))  # (B_s, B_theta, B_zeta) of the embedding
    G, I, D = (np.asarray(a)[:, None, None] for a in (geo.G, geo.I, D))
    np.testing.assert_allclose(cov[..., 2], np.broadcast_to(G, B.shape), rtol=2e-3)
    np.testing.assert_allclose(cov[..., 1], np.broadcast_to(I, B.shape), atol=2e-3)
    np.testing.assert_allclose(Bz, B**2 / D, rtol=2e-3)  # B^zeta = B.grad(zeta) = B^2 / (G + iota I) > 0
    h = fax.mode_family(2, 4)  # exp(i(m theta - n zeta)): (B.grad e) / (i B e) = (iota m - n) B / (G + iota I)
    zeros = jnp.zeros_like(geo.s)
    bending = _modes(g, geo.iota, zeros, jnp.asarray(D[:, 0, 0]), zeros, h)[1][0]  # f' part of grad(u_j), s component: c_j e_j
    c = np.asarray(bending[..., 0] / jnp.exp(1j * (h[:, 0, None, None] * g.theta - h[:, 1, None, None] * g.zeta)))
    kpar = (
        np.asarray(geo.iota)[:, None, None, None] * np.asarray(h[:, 0])[None, :, None, None] - np.asarray(h[:, 1])[None, :, None, None]
    ) * (Bz / B)[:, None]
    np.testing.assert_allclose(c, 1j * kpar, rtol=2e-3, atol=1e-6)
    assert kpar[40, 1].mean() < 0 < kpar[40, 4].mean()  # sign of iota m - n: m = 1 against m = 4 at n = 2, iota ~ 0.8 at s ~ 0.4


def test_cylinder_newcomb_m1_and_suydam_threshold():
    """Large-aspect cylinder (epsilon_B = 0) gates of the kink and pressure terms.

    m = n = 1 rigid shift inside q = 1 (q = 0.8 + s): the kink term cancels the bending energy
    (Newcomb, m^2 - 1 = 0). m/n = 11/6 at q = 1.71 + 0.16 s with p = p0 (1 - s): Suydam
    ``r B^2 (q'/q)^2 / (8 mu0) + p' > 0`` gives p0 = 1.05e4 Pa; below it the lowest omega^2 is
    the discretization error of a marginal continuum (O(1/N^2)), above it a converged instability.
    """
    rho = fax.mass_density(2e19, 1.0)
    vacuum = [global_matrices(konies(16), fax.mode_family(6, 13, 9), rho, 16, 4, finite_beta=x) for x in (False, True)]
    jax.tree_util.tree_map(lambda a, b: np.testing.assert_array_equal(a, b), *vacuum)  # no current, no pressure

    def bending_ratio(N):
        s = (jnp.arange(N) + 0.5) / N
        geo = fax.circular_torus(s=s, major_radius=10.0, B0=3.0, iota=1 / (0.8 + s), current=True)
        F0, F1 = (np.asarray(dense(global_matrices(geo, [[1.0, 1.0]], rho, 8, 4, finite_beta=x)[0])).real for x in (False, True))
        r = np.sqrt(np.asarray(s))
        y = r * np.clip((np.sqrt(0.2) - r) / 0.02 + 0.5, 0, 1)
        return abs(y @ F1 @ y) / (y @ F0 @ y)

    # Converged 0.0201 at N = 400/800/1600; with first-order ends it drifted 0.059 -> 0.0012 -> 0.0105 (old 2e-3 pass was an artifact).
    assert bending_ratio(800) < 3e-2 and bending_ratio(1600) == pytest.approx(bending_ratio(800), rel=0.1)
    lowest = {}
    for N, p0 in itertools.product((400, 1600), (1.0e4, 2.0e4)):
        s = (jnp.arange(N) + 0.5) / N
        geo = fax.circular_torus(s=s, major_radius=10.0, B0=3.0, iota=1 / (1.71 + 0.16 * s), pressure=[p0, -p0])
        lowest[N, p0] = float(fax.eigenmodes(geo, [[11.0, 6.0]], rho, ntheta=8, nzeta=4, finite_beta=True).omega2[0])
    assert 12 < lowest[400, 1e4] / lowest[1600, 1e4] < 20 and lowest[1600, 2e4] / lowest[400, 2e4] == pytest.approx(1, abs=0.05)


def test_unstable_omega2_is_signed_with_finite_gradients():
    """Suydam-unstable cylinder (p0 = 2e4 Pa): omega^2 < 0 is kept, with the growth rate and stable frequency separate."""
    rho, s = fax.mass_density(2e19, 1.0), (jnp.arange(400) + 0.5) / 400

    def modes(r, **kw):
        geo = fax.circular_torus(s=s, major_radius=10.0, B0=3.0, iota=1 / (1.71 + 0.16 * s), pressure=[2e4, -2e4])
        return fax.eigenmodes(geo, [[11.0, 6.0]], r, ntheta=8, nzeta=4, finite_beta=True, **kw)

    ev = modes(rho)
    assert ev.omega2[0] < 0 and ev.frequencies_hz[0] == 0
    assert ev.growth_rate[0] == pytest.approx(float(jnp.sqrt(-ev.omega2[0])))

    def f(r, **kw):
        geo = fax.circular_torus(s=s, major_radius=10.0, B0=3.0, iota=1 / (1.71 + 0.16 * s), pressure=[2e4, -2e4])
        return fax.frequency_hz(geo, [[11.0, 6.0]], r, ntheta=8, nzeta=4, finite_beta=True, **kw)

    assert float(jax.grad(f)(rho)) == 0.0  # stable frequency of an unstable mode: finite, flat
    gamma = f(rho, growth=True)
    assert gamma == pytest.approx(float(ev.growth_rate[0]), rel=1e-10)
    assert jax.grad(f)(rho, growth=True) == pytest.approx(float(-0.5 * gamma / rho), rel=1e-8)  # gamma ~ rho^(-1/2)
    omega2 = lambda x: jnp.stack([fax.eigen._hz(x), fax.eigen._growth(x)])  # noqa: E731
    for x in (-4.0, 4.0, 0.0):
        assert jnp.isfinite(jax.jacfwd(omega2)(x)).all() and jnp.isfinite(jax.jacrev(omega2)(x)).all()


def test_targeted_eigenpairs_fallback_and_tied_selection():
    """The invertible [[0, 1], [1, 0]] has a zero first Schur pivot: SOLVAX's dense fallback is
    reported in the status. Selecting one of a tied pair has no derivative; selecting the pair does."""
    F, G = Blocks(jnp.zeros((2, 1, 1)), jnp.ones((1, 1, 1))), Blocks(jnp.ones((2, 1, 1)), jnp.zeros((1, 1, 1)))
    lam, y, status = targeted_eigenpairs(F, G, 0.0, 2)
    np.testing.assert_allclose(lam, [-1.0, 1.0])
    assert status.fallback and status.converged and y.shape == (2, 1, 2)
    d = jnp.array([[[0.0, 0.0], [0.0, 0.0]], [[2.0, 0.5], [0.5, 1.0]], [[3.0, 0.0], [0.0, 4.0]]])
    B = Blocks(d, jnp.array([[[1.0, 0.2], [0.3, 1.0]], [[0.4, 0.0], [0.1, 0.6]]]))
    M = Blocks(jnp.broadcast_to(jnp.eye(2), (3, 2, 2)), jnp.zeros((2, 2, 2)))
    lam, y, status = targeted_eigenpairs(B, M, 0.0, 3)
    w = np.linalg.eigvalsh(np.asarray(dense(B)))
    np.testing.assert_allclose(lam, np.sort(w[np.argsort(abs(w))[:3]]), atol=1e-12)
    assert status.converged
    tied = Blocks(jnp.stack([jnp.eye(2), jnp.diag(jnp.array([3.0, 4.0]))]), jnp.zeros((1, 2, 2)))
    I2 = Blocks(jnp.broadcast_to(jnp.eye(2), (2, 2, 2)), jnp.zeros((1, 2, 2)))
    mean = lambda t, count: targeted_eigenvalues(Blocks(tied.diag + t * jnp.eye(2), tied.upper), I2, 1.0, count).mean()  # noqa: E731
    assert targeted_eigenpairs(tied, I2, 1.0, 1)[2].gap < 1e-12  # a tie (0 or an ulp, by platform) splits the selection
    assert jnp.isnan(jax.grad(mean)(0.0, 1)) and jax.grad(mean)(0.0, 2) == pytest.approx(1.0)
    for ok, expect in ((True, 2.0), (False, np.nan)):
        fwd, rev = jax.jvp(lambda x, ok=ok: _certified(x, ok), (3.0,), (2.0,))[1], jax.grad(lambda x, ok=ok: _certified(x, ok) * 2)(3.0)
        np.testing.assert_array_equal([fwd, rev], [expect, expect])


def test_solver_status_and_rejection_of_unconverged_gradients():
    h, rho = fax.mode_family(6, 13, 9), fax.mass_density(2e19, 1.0)
    options = dict(target_hz=6.46e4, count=4, ntheta=16, nzeta=4)
    good, poor = (fax.eigenmodes(konies(40), h, rho, iters=iters, **options) for iters in (100, 3))
    assert good.status.converged and good.status.residual < 1e-10 and good.status.orthogonality < 1e-10 and good.status.gap > 0
    assert not good.status.fallback and 3 < good.status.iterations < 100
    assert not poor.status.converged and poor.status.residual > 1e-10 and poor.status.iterations == 3
    full = fax.eigenmodes(konies(40), h, rho, ntheta=16, nzeta=4)
    assert full.status.converged and full.status.iterations == 0 and full.status.residual < 1e-12

    F, G = global_matrices(konies(40), h, rho, 16, 4)
    target = float((2 * np.pi * 6.46e4) ** 2)
    assert jnp.isfinite(targeted_eigenvalues(F, G, target, 4, 100)).all()
    assert jnp.isnan(targeted_eigenvalues(F, G, target, 4, 3)).all()
    jvp = jax.jvp(lambda f: targeted_eigenvalues(Blocks(f, F.upper), G, target, 4, 3), (F.diag,), (jnp.ones_like(F.diag),))
    assert jnp.isnan(jvp[1]).all()
    run = lambda iters: fax.frequency_hz(konies(40), h, rho, 0, 1, iters=iters, **options)  # noqa: E731
    assert jnp.isfinite(run(100)) and jnp.isnan(run(3))


def test_cluster_mean_first_derivative_at_degeneracy():
    """Mean of a repeated pair: d/dt is the projected trace; only first order is supported there (the
    second derivative is NaN whenever the spectrum has a repeated eigenvalue)."""
    K, M = jnp.diag(jnp.array([1.0, 1.0, 3.0])), jnp.eye(3)
    D = jnp.array([[1.0, 0.5, 0.2], [0.5, 2.0, 0.1], [0.2, 0.1, 0.0]])
    mean = lambda t, a=0, b=2: fax.eigenvalue_mean(K + t * D, M, a, b)  # noqa: E731
    assert jax.grad(mean)(0.0) == pytest.approx(1.5, rel=1e-12)  # (D00 + D11) / 2
    assert jax.grad(mean)(0.0) == pytest.approx(float((mean(1e-6) - mean(-1e-6)) / 2e-6), rel=1e-6)
    mean = lambda t: fax.eigenvalue_mean(jnp.diag(jnp.array([1.0, 1.1, 3.0])) + t * D, M, 0, 2)  # noqa: E731
    fd = (mean(1e-3) - 2 * mean(0.0) + mean(-1e-3)) / 1e-6  # no repeated eigenvalue anywhere: second order works
    assert jax.grad(jax.grad(mean))(0.0) == pytest.approx(float(fd), rel=1e-4)
