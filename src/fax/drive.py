"""Perturbative energetic-particle drive and Landau damping of an Alfvén eigenmode in Boozer geometry.

The growth rate is the power the particles give to a fixed mode over twice its
energy (NOVA-K / CAS3D-K style), ``gamma = -P / (2 W)``, with ``W`` the inertia form
``int rho |grad phi|^2 / (2 B^2) dV`` of :func:`fax.eigen.global_matrices`.

Model: zero orbit width (ZOW), zero Larmor radius, passing guiding centres on the
surfaces of a :class:`fax.Geometry`, low-beta drift ``v_d = m v^2 (1 + xi^2) / (2 Z e B^3)
B x grad B`` with Boozer components ``(B x grad B)^i = eps^ijk B_j d_k B / sqrt(g)``,
``B_theta = I``, ``B_zeta = G`` (``B_s`` dropped). The mode is ``phi = sum_h phi_h(s)
exp[i(m_h theta - n_h zeta - omega t)]`` (the convention of :mod:`fax.eigen`). A passing
orbit of pitch ``lam = mu / E`` follows ``theta - iota zeta = const`` with ``dt/dzeta =
(G + iota I) / (v |xi| B)``; the uniform angles ``vartheta = theta + iota chi``,
``varphi = zeta + chi`` solve ``(iota d_theta + d_zeta) chi = h / <h> - 1``,
``h = 1 / (|xi| B)``. The work ``Ze v_d . E`` decomposed in them resonates at
``omega = |j iota - k| omega_zeta(v, lam)`` for ``j = m + l`` and ``k = n + k' N_fp``:
the toroidal (``l``) and helical (``k'``) sidebands of the B spectrum. The unperturbed passing orbit is a
quasi-periodic winding of the torus ``(vartheta, varphi)`` with actions ``J_vartheta = Ze psi + O(rho)`` (the
mean toroidal flux) and ``J_varphi``; ``P_zeta`` is not invariant without symmetry. For ``F(E, mu, s)`` and
the resonant harmonic ``e^{i(j vartheta - k varphi)}``, ``dJ_vartheta = j dE / omega``, so each resonance
carries ``Q_jk F = dF/dE + j dF/ds / (omega Ze psi')`` (docs/theory.md, section 8) with ``psi'`` the toroidal
flux derivative: the orbit's poloidal harmonic ``j``, not the mode's ``n``, which reduces to ``n / iota`` of
the tokamak ``P_zeta`` form up to the labelling difference ``O((j iota - n) / n)`` and is invariant under
``theta -> theta - N zeta``. The resonance is integrated exactly in speed at each pitch (Gauss-Legendre in
``lam``), so ``gamma`` is smooth::

    gamma = pi^2 mu0 m_f^2 / (2 y.H G y) int |psi'| ds int dlam sum_jk v^6 Q_jk |C|^2 / (|j iota - k| w^2),
    C = sum_h C_h,
    C_h = < (1 + xi^2) / (2 B^3 sqrt(g)) [(I B_zeta - G B_theta) phi_h' + i B_s (G m_h + I n_h) phi_h]
            e^{i(m_h theta - n_h zeta)} e^{-i(j vartheta - k varphi)} >_orbit,

with ``w = omega_zeta / v``, ``B_i = d_i |B|`` and ``v = omega / (|j iota - k| w)``. Regularizations, all explicit: the
passing-boundary layer ``boundary`` (pitches stop at ``(1 - boundary) / B_max``), the small-divisor ``guard`` of the orbit-angle
inversion ``kappa / (kappa^2 + guard^2)``, the Gauss-Legendre ``nodes`` and the birth-edge width of the slowing-down
distribution (``FastIons.birth_width``); gamma converges as each goes to zero (tests).

Trapped and precessing particles are omitted (no bounce-averaged precession resonance).
``n > 0`` with a peaked density drives (``iota psi' > 0``).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .continuum import AMU, MU0
from .eigen import _quad, global_matrices
from .geometry import Geometry, _d_ds, check, surface_grid

E = 1.602176634e-19


@dataclass(frozen=True)
class FastIons:
    """Isotropic fast ions: ``density`` (m^-3, scalar or profile on ``s``), energies in eV.

    ``kind="maxwellian"`` uses ``temperature_ev``; ``kind="slowing_down"`` uses
    ``F ~ 1/(v^3 + v_c^3)`` below ``birth_ev`` with ``critical_ev``. The birth edge is smoothed
    (``erfc`` over ``birth_width`` = 2% of ``v_b``) and ``dF/dE`` includes the smoothed edge, so ``F`` is normalized
    to ``int F d^3v = 1`` with the smoothed edge and the edge drive is differentiable. Any other
    ``kind`` raises ``ValueError``.
    """

    density: Any = 1.0e17
    temperature_ev: Any = 100.0e3
    birth_ev: Any = 3.5e6
    critical_ev: Any = 0.5e6
    mass_amu: Any = 2.01410178
    charge: Any = 1.0
    birth_width: Any = 0.02
    kind: str = "maxwellian"

    def __post_init__(self) -> None:
        if self.kind not in ("maxwellian", "slowing_down"):
            raise ValueError(f"unknown FastIons kind {self.kind!r}: use 'maxwellian' or 'slowing_down'")
        for name in ("temperature_ev", "birth_ev", "critical_ev", "mass_amu"):
            v = getattr(self, name)  # pytree rebuilds may hold None or axis specs: only numbers are checked
            ok = not isinstance(v, (int, float, np.ndarray, jax.Array)) or (jnp.isfinite(v) & (v > 0))
            check(ok, f"FastIons.{name} must be positive and finite")


jax.tree_util.register_dataclass(
    FastIons,
    data_fields=["density", "temperature_ev", "birth_ev", "critical_ev", "mass_amu", "charge", "birth_width"],
    meta_fields=["kind"],
)


def growth_rate(
    omega: Any,
    phi: Any,
    geo: Geometry,
    harmonics: Any,
    ions: FastIons,
    mass_density: Any,
    *,
    sidebands: int = 3,
    helical: int = 1,
    nodes: int = 32,
    boundary: Any = 0.0,
    guard: Any = 1e-2,
    ntheta: int = 64,
    nzeta: int = 32,
    resolved: bool = False,
    standing: bool = False,
) -> Any:
    """Fast-ion growth rate (1/s) of a mode ``phi`` (shape ``(len(geo.s), H)``, nodal values) at ``omega`` (rad/s).

    ``harmonics`` is a static ``(H, 2)`` list of ``(m, n)``; ``geo.s`` lies strictly inside
    ``(0, 1)`` as for :func:`fax.eigenmodes`. Resonances ``j = m + l``, ``|l| <= sidebands``,
    and ``k = n + k' nfp``, ``|k'| <= helical``, are summed. An axisymmetric case with
    ``helical = 0`` is exact with ``nzeta = 1``. ``resolved=True`` returns the resonances
    ``(j, k)`` and the growth rate of each instead of their sum. ``boundary`` and ``guard`` are the passing-boundary and
    small-divisor regularizations described in the module docstring (defaults 0 and 1e-2).

    ``standing=True`` reads ``phi`` as a cosine- or sine-sector mode of :func:`fax.eigenmodes`, the standing wave
    ``Re sum_h phi_h exp(i(m_h theta - n_h zeta))``: the rate is that of its travelling components ``phi / 2`` on ``(m, n)``
    and ``conj(phi) / 2`` on ``(-m, -n)`` together, with their interference; a pair ``phi = (plus, minus)`` gives those two
    coefficients explicitly. It is the rate of the mode only when the
    cosine and sine partners are further apart in frequency than the drive couples them; a nearer pair needs the 2 x 2 ledger
    of the two modes, and in an axisymmetric field, where the partners are degenerate, the travelling rates (this function
    without ``standing``, on ``(m, n)`` and on ``(-m, -n)``) are the physical ones. A mode of :func:`fax.eigenmodes` whose
    block folds ``(-m, -n)`` rows goes through :func:`fax.eigen.travelling` first.
    """
    hm = np.asarray(harmonics, dtype=float)
    pairs = np.array(
        [
            (j, k)
            for j in range(int(hm[:, 0].min()) - sidebands, int(hm[:, 0].max()) + sidebands + 1)
            for k in sorted({nn + kk * geo.nfp for nn in hm[:, 1] for kk in range(-helical, helical + 1)})
        ],
        dtype=float,
    )
    if standing:  # each travelling component keeps its own sidebands; a harmonic listed twice (as (0, 0)) sums its parts
        plus, minus = phi if isinstance(phi, tuple) else (jnp.asarray(phi) / 2, jnp.conj(jnp.asarray(phi)) / 2)
        hm, row = np.unique(np.concatenate([hm, -hm]), axis=0, return_inverse=True)
        both = jnp.concatenate([plus, minus], 1)
        phi, pairs = (
            jnp.zeros((both.shape[0], len(hm)), complex).at[:, row.reshape(-1)].add(both),
            np.unique(np.concatenate([pairs, -pairs]), axis=0),
        )
    gamma = _growth_rate(omega, jnp.asarray(phi), geo, hm, ions, mass_density, pairs, boundary, guard, nodes, ntheta, nzeta)
    return (pairs.astype(int), gamma) if resolved else jnp.sum(gamma)


@partial(jax.jit, static_argnums=(9, 10, 11))
def _growth_rate(
    omega: Any,
    phi: Any,
    geo: Geometry,
    hm: Any,
    ions: FastIons,
    mass_density: Any,
    pairs: Any,
    boundary: Any,
    guard: Any,
    nodes: int,
    ntheta: int,
    nzeta: int,
) -> jax.Array:
    """The growth rate of each resonance in ``pairs``, compiled once per shape (see :func:`growth_rate`)."""
    m, n = (hm[:, i][:, None, None] for i in (0, 1))
    g = surface_grid(geo, ntheta, nzeta)
    col = lambda a: jnp.asarray(a)[:, None, None]  # noqa: E731 - per-surface value -> (S, 1, 1)
    D = geo.G + geo.iota * geo.I
    psi = jnp.mean(g.sqrt_g * g.B**2, axis=(1, 2)) / D  # dpsi/ds, psi = toroidal flux / 2 pi
    Bs, Bt, Bz = (g.dB[..., i] for i in range(3))
    e = jnp.exp(1j * (m * g.theta - n * g.zeta)) / (g.B**3 * g.sqrt_g)[:, None]  # (S, H, T, Z)
    kernel = (col(geo.I) * Bz - col(geo.G) * Bt)[:, None] * e * _d_ds(phi, geo.s)[..., None, None]
    kernel = kernel + 1j * Bs[:, None] * (col(geo.G)[:, None] * m + col(geo.I)[:, None] * n) * e * phi[..., None, None]
    # passing pitches lam = mu/E in [0, 1/B_max), Gauss-Legendre; B_max is the exact sample maximum, so
    # duplicated toroidal samples change nothing (a log-sum-exp bound shifts with the sample count)
    x, wx = (jnp.asarray(z) for z in np.polynomial.legendre.leggauss(nodes))
    bmax = jnp.max(g.B, axis=(1, 2))[:, None]
    lam = 0.5 * (x + 1.0) * (1.0 - boundary) / bmax  # (S, L)
    xi = jnp.sqrt(1.0 - lam[..., None, None] * g.B[:, None])  # (S, L, T, Z)
    h = 1.0 / (xi * g.B[:, None])
    hbar = jnp.mean(h, axis=(2, 3))
    kt, kz = (jnp.fft.fftfreq(k, 1.0 / k)[:, None] for k in (ntheta, nzeta))
    kappa = geo.iota[:, None, None, None] * kt + kz.T  # (iota d_theta + d_zeta) -> i kappa
    chi = jnp.real(jnp.fft.ifft2(jnp.fft.fft2(h / hbar[..., None, None] - 1.0) * -1j * kappa / (kappa**2 + guard**2)))
    weight = 0.5 * (1.0 + xi**2) * h / hbar[..., None, None] * jnp.sum(kernel, 1)[:, None]  # (S, L, T, Z): C = sum_h C_h, phase is h-free
    omega_hat = 1.0 / (jnp.abs(D)[:, None] * hbar)  # omega_zeta / v
    dens = jnp.asarray(ions.density) * jnp.ones_like(geo.s)

    def resonance(jk: jax.Array) -> jax.Array:
        j, k = jk[0], jk[1]
        kap = (j * geo.iota - k)[:, None]
        C = jnp.mean(weight * jnp.exp(-1j * (kap[..., None, None] * chi + j * g.theta - k * g.zeta)), axis=(-2, -1))  # (S, L)
        rate = jnp.maximum(jnp.abs(kap) * omega_hat, 1e-12)
        v = omega / rate
        dfde, f = _distribution(v, ions)
        star = j * _d_ds(dens[:, None], geo.s) / (omega * ions.charge * E * psi[:, None])
        q = (dens[:, None] * dfde + star * f) * jnp.abs(C) ** 2
        return v**6 * q / (rate * omega_hat)

    power = jnp.abs(psi) * jnp.sum(jax.lax.map(jax.checkpoint(resonance), pairs) * wx * (1.0 - boundary) / (2 * bmax), axis=-1)  # (P, S)
    zero = jnp.zeros((len(pairs), 1))
    radial = jnp.trapezoid(jnp.concatenate([zero, power, zero], 1), jnp.concatenate([zero[0], geo.s, jnp.ones(1)]))
    inertia = global_matrices(geo, hm, mass_density, ntheta=ntheta, nzeta=nzeta)[1]
    return jnp.pi**2 * MU0 * (ions.mass_amu * AMU) ** 2 / 2 * radial / _quad(phi[..., None], inertia)[0]


class PairRates(NamedTuple):
    """Rates (1/s) of sector modes driven as pairs (see :func:`pair_rates`). Entry ``i`` of ``rates`` is a member of the set of
    the pair that holds mode ``i``, not the rate of mode ``i``."""

    rates: jax.Array  # the 2 x 2 rates of each pair as a set: larger at the lower index; an unpaired mode its own rate
    partner: jax.Array  # the other mode of the pair (the mode itself when unpaired)
    top: jax.Array  # largest eigenvalue of the projected power matrix Gamma of the pair: Lipschitz, never below the larger rate
    trace: jax.Array  # Gamma_aa + Gamma_bb of the pair (the own rate when unpaired): smooth


def pair_rates(
    omega: Any, phi: Any, sine: Any, geo: Geometry, harmonics: Any, ions: FastIons, mass_density: Any, **options: Any
) -> PairRates:
    """Growth rates (1/s) of sector modes as two-mode ledgers (plan 8.5): ``omega`` (rad/s) and ``phi`` stack the modes
    (structures of :func:`fax.eigenmodes`), ``sine`` marks the sine-sector ones; ``options`` go to :func:`growth_rate`.

    A mode and its nearest mode of the other sector, when each is the other's nearest, are one 2 x 2 problem
    ``dc/dt = (diag(-i omega) + Gamma) c`` in one wave energy. ``Gamma_aa`` is the standing-wave rate of ``a``, and
    ``|Gamma_ab|^2`` comes from the rates of the two-sign waves ``S(a) + S(b)`` and ``S(a) + i S(b)`` at the mean frequency,
    ``S(x) = (x / 2, conj(x) / 2)`` on ``(m, n)`` and ``(-m, -n)`` (the modes are orthogonal and of equal energy). The rates
    are the real parts of its eigenvalues, ``mean +- Re sqrt((Gamma_aa - Gamma_bb - i (omega_a - omega_b))^2 / 4 +
    |Gamma_ab|^2)``, reported as a set: an eigenvalue belongs to the pair, not to one of its modes (with ``Gamma_aa = Gamma_bb``,
    as in a tokamak, no rule can assign it). Far apart in frequency they are the diagonal rates; degenerate, the rates of the two
    travelling waves. A mode without such a partner keeps its own rate. At the exceptional point (zero discriminant) the rates
    are continuous and their derivative is not finite; an optimizer takes ``top``, ``lambda_max(Gamma) = mean +
    sqrt((Gamma_aa - Gamma_bb)^2 / 4 + |Gamma_ab|^2)``, an upper bound of the pair's energy growth rate, and ``trace``: the trace
    is smooth, ``top`` is smooth through the exceptional point and Lipschitz where the projected matrix is degenerate
    (``Gamma_aa = Gamma_bb``, ``Gamma_ab = 0``), with a zero subgradient there (plan 8.9: eigenvalues of a projected physical
    bilinear form)."""
    w, phi, sine = jnp.asarray(omega), jnp.asarray(phi), jnp.asarray(sine)
    rate = lambda om, plus, minus: growth_rate(om, (plus, minus), geo, harmonics, ions, mass_density, standing=True, **options)  # noqa: E731
    own = jnp.stack([rate(w[i], phi[i] / 2, jnp.conj(phi[i]) / 2) for i in range(len(w))])
    gap = jnp.where(sine[:, None] != sine[None], jnp.abs(w[:, None] - w[None]), jnp.inf)
    index = jnp.arange(len(w))
    paired = (jnp.argmin(gap, 1)[jnp.argmin(gap, 1)] == index) & jnp.isfinite(gap.min(1))
    j = jnp.where(paired, jnp.argmin(gap, 1), index)
    out = []
    for i in range(len(w)):
        a, b, mid, mean = phi[i], phi[j[i]], (w[i] + w[j[i]]) / 2, (own[i] + own[j[i]]) / 2
        real = rate(mid, (a + b) / 2, jnp.conj(a + b) / 2) - mean
        imag = rate(mid, (a + 1j * b) / 2, (jnp.conj(a) + 1j * jnp.conj(b)) / 2) - mean
        coupling = real**2 + imag**2
        root = jnp.real(jnp.sqrt(((own[i] - own[j[i]]) - 1j * (w[i] - w[j[i]])) ** 2 / 4 + coupling))
        first = i <= j[i]  # the lower index of the pair holds the larger rate
        values = (mean + jnp.where(first, 1.0, -1.0) * root, _top(own[i], own[j[i]], real, imag), 2 * mean)
        out.append([jnp.where(paired[i], v, alone) for v, alone in zip(values, (own[i], own[i], own[i]), strict=True)])
    rates, top, trace = (jnp.stack(v) for v in zip(*out, strict=True))
    return PairRates(rates, j, top, trace)


def _top(gaa: jax.Array, gbb: jax.Array, real: jax.Array, imag: jax.Array) -> jax.Array:
    """``lambda_max`` of ``[[gaa, g], [conj(g), gbb]]``, ``g = real + i imag``; the square root has a zero subgradient where it
    vanishes (double ``where``) instead of NaN."""
    u = (gaa - gbb) ** 2 / 4 + real**2 + imag**2
    return (gaa + gbb) / 2 + jnp.where(u > 0, jnp.sqrt(jnp.where(u > 0, u, 1.0)), 0.0)


def ion_landau_damping(
    omega: Any, phi: Any, geo: Geometry, harmonics: Any, temperature_ev: Any, density: Any, *, mass_amu: Any = 1.00727647, **options: Any
) -> jax.Array:
    """Bulk-ion Landau damping rate (1/s, positive damps): the same power transfer for a flat
    thermal Maxwellian (Fu & Van Dam 1989; Betti & Freidberg 1992). Electron Landau damping
    needs ``E_par`` and is not included. ``options`` go to :func:`growth_rate`."""
    bulk = FastIons(density=density, temperature_ev=temperature_ev, mass_amu=mass_amu)
    return -growth_rate(omega, phi, geo, harmonics, bulk, density * mass_amu * AMU, **options)


def _distribution(v: jax.Array, ions: FastIons) -> tuple[jax.Array, jax.Array]:
    """``(dF/dE, F)`` per unit density at speed ``v``."""
    m = ions.mass_amu * AMU
    if ions.kind == "maxwellian":
        t = ions.temperature_ev * E
        f = (m / (2 * jnp.pi * t)) ** 1.5 * jnp.exp(-0.5 * m * v**2 / t)
        return -f / t, f
    vb, vc = jnp.sqrt(2 * ions.birth_ev * E / m), jnp.sqrt(2 * ions.critical_ev * E / m)
    # int 4 pi u^2 du erfc((u - vb) / w) / 2 / (u^3 + vc^3) = 4 pi / 3 E[ln((u^3 + vc^3) / vc^3)], u ~ N(vb, w^2 / 2) (by parts)
    t, wt = (jnp.asarray(z) for z in np.polynomial.hermite.hermgauss(24))
    norm = 3.0 / (4 * jnp.pi * (jnp.sum(wt * jnp.log1p((vb * (1 + ions.birth_width * t) / vc) ** 3)) / jnp.sqrt(jnp.pi)))

    def shape(u: jax.Array) -> jax.Array:  # birth edge smoothed over 2% of v_b so the edge drive stays differentiable
        return norm * 0.5 * jax.scipy.special.erfc((u - vb) / (ions.birth_width * vb)) / (u**3 + vc**3)

    f, dfdv = jax.jvp(shape, (v,), (jnp.ones_like(v),))
    return dfdv / (m * v), f
