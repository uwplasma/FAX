"""Shear-Alfvén continuum on each flux surface (Spong, Sanchez & Weller 2003).

On a surface the low-beta continuum equation is::

    mu0 rho w^2 |grad s|^2 / B^2 E + B.grad(|grad s|^2 / B^2 B.grad E) = 0,

with ``B.grad = B^2/(G + iota I) (d_zeta + iota d_theta)`` in Boozer
coordinates. A Galerkin projection on ``exp(i(m theta - n zeta))`` gives
``K x = w^2 M x`` with::

    K_ij = < |J| |grad s|^2 k_i k_j / mu0 >,   M_ij = < rho |J| |grad s|^2 / B^2 >,
    k_j = (n_j - m_j iota) |B| / (G + iota I).

With ``pressure`` the surface also carries sound: the surface displacement couples to the parallel
displacement through the geodesic curvature (Cheng & Chance 1986; docs/theory.md section 2).
"""

from __future__ import annotations

from functools import partial
from numbers import Integral
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular

from .geometry import Geometry, check, field_period, surface_grid, symmetric

MU0 = 4.0e-7 * jnp.pi
AMU = 1.66053906660e-27


class Matrices(NamedTuple):
    """Hermitian stiffness ``K`` and mass ``M``, shape ``(surfaces, H, H)``, or ``(surfaces, 2H, 2H)`` with sound."""

    K: jax.Array
    M: jax.Array


def mass_density(n_ion: Any, ion_mass_amu: float = 1.0) -> jax.Array:
    """Ion mass density ``rho = n_i m_i`` in kg/m^3 (ion mass in atomic mass units)."""
    return jnp.asarray(n_ion) * ion_mass_amu * AMU


def mode_family(n: int | tuple[int, ...], m_max: int, m_min: int = 0) -> np.ndarray:
    """Harmonics ``(m, n)`` for ``m_min <= m <= m_max`` and the given toroidal mode numbers."""
    ns = (n,) if isinstance(n, Integral) else tuple(n)
    check(m_max >= m_min, f"m_max={m_max} must be >= m_min={m_min}")
    return np.asarray([[m, nn] for nn in ns for m in range(m_min, m_max + 1)], dtype=float)  # host data: harmonics are static


def check_modes(geo: Geometry, h: jax.Array, density: Any) -> None:
    """Host checks shared by :func:`continuum_matrices` and :func:`fax.eigen.global_matrices`."""
    d = jnp.asarray(density)
    check(h.ndim == 2 and h.shape[0] > 0 and h.shape[1] == 2, "harmonics must be a non-empty list of (m, n) pairs")
    check(h == jnp.round(h), "harmonics must be integers")
    check(jnp.sum(jnp.all(h[:, None] == h[None], -1)) == len(h), "harmonics must be distinct (duplicate (m, n) make the mass singular)")
    check(jnp.isfinite(d) & (d > 0), "density must be positive and finite")
    check(d.ndim == 0 or d.shape == geo.s.shape, "density must be a scalar or have one value per surface")
    check(jnp.isfinite(geo.iota), "iota must be finite")


def herm(a: jax.Array) -> jax.Array:
    """Conjugate transpose over the last two axes."""
    return jnp.conj(jnp.swapaxes(a, -1, -2))


@jax.jit
def generalized_eigh(K: jax.Array, M: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Solve ``K x = lam M x`` for Hermitian ``K`` and positive-definite ``M`` (batched).

    Eigenvectors are ``M``-orthonormal: ``x.H M x = 1``.
    """
    L = jnp.linalg.cholesky(M)
    tri = lambda b, trans: solve_triangular(L, b, lower=True, trans=trans)  # noqa: E731
    A = herm(tri(herm(tri(K, 0)), 0))  # L^-1 K L^-H by two triangular solves
    lam, y = jnp.linalg.eigh(0.5 * (A + herm(A)))
    return lam, tri(y, 2)  # L^-H y


def angle_mean(f: jax.Array, h: jax.Array, period: int = 1) -> jax.Array:
    """``<f exp(i(dm theta - dn zeta))>`` for a real ``f`` of shape ``(..., theta, zeta)`` and all harmonic pairs, shape ``(..., H, H)``
    (row ``i`` conjugated, ``dm = m_j - m_i``): the DFT of ``f`` at ``(-dm, dn)``, one real FFT instead of an ``(H, H, T, Z)`` tensor.
    With ``period``, ``f`` is sampled on one sector ``2 pi / period`` of a ``2 pi / period``-periodic field, so only ``dn`` that are
    multiples of ``period`` survive the average."""
    T, Z = f.shape[-2:]
    dm, dn = (h[None, :, 0] - h[:, None, 0]).astype(int), (h[None, :, 1] - h[:, None, 1]).astype(int)
    a, b = -dm % T, (dn // period) % Z
    hi = b > Z // 2  # beyond the half spectrum of a real f: F[a, b] = conj(F[-a, -b])
    F = jnp.fft.rfft2(f)[..., jnp.where(hi, -a % T, a), jnp.where(hi, Z - b, b)]
    return jnp.where(dn % period == 0, jnp.where(hi, jnp.conj(F), F) / (T * Z), 0)


def sound_coupling(
    geo: Geometry, h: jax.Array, rho: jax.Array, pressure: Any, gamma: float, grid: Any, period: int = 1
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Sound blocks added to the shear-Alfven pencil, same ``<.>`` weighting and a common factor ``psi'`` dropped.

    The unknown is ``(E_j, c_j)``: the surface displacement potential and ``c = xi_par / |B|``. With the geodesic
    curvature ``kappa_s / 2 = (I d_zeta|B| - G d_theta|B|) / (psi' (G + iota I) |B|)`` the compression is
    ``D = kappa_s E + B.grad c`` and ``B.grad c_j = -i a_j |B|^2 c_j``, ``a_j = (n_j - m_j iota) / (G + iota I)``.
    The energy ``<|sqrt g_B| gamma p B^2/(B^2 + mu0 gamma p) |D|^2>`` (the factor is 1 at low beta) and the kinetic
    energy ``rho <|sqrt g_B| |B|^2 |c|^2>`` give the blocks
    ``(pp, pc, cc)`` of the stiffness (the ``cp`` block is ``pc^H``) and the mass ``mc`` of ``c``.
    """
    GI = geo.G + geo.iota * geo.I
    psi = (jnp.abs(grid.sqrt_g) * grid.B**2).mean((1, 2)) / GI  # psi' from the embedding, per surface
    cur = (geo.I[:, None, None] * grid.dB[..., 2] - geo.G[:, None, None] * grid.dB[..., 1]) / (psi * GI)[:, None, None] / grid.B
    a = (h[:, 1] - h[:, 0] * geo.iota[:, None]) / GI[:, None]
    J, B = jnp.abs(grid.jacobian), grid.B
    gp = (gamma * jnp.broadcast_to(jnp.asarray(pressure, dtype=float), geo.s.shape))[:, None, None]
    gp = gp / (1.0 + MU0 * gp / B**2)  # the stiff compressional Alfven response: gamma p B^2 / (B^2 + mu0 gamma p) (Cheng & Chance eq. 15)
    mean = lambda f: angle_mean(f, h, period)  # noqa: E731
    pp = mean(4.0 * gp * J * cur**2)
    pc = -2j * a[:, None, :] * mean(gp * J * cur * B**2)
    cc = a[:, :, None] * a[:, None, :] * mean(gp * J * B**4)
    mc = rho[:, None, None] * mean(J * B**2)
    return pp, pc, cc, mc


def continuum_matrices(
    geo: Geometry,
    harmonics: Any,
    density: Any,
    ntheta: int = 32,
    nzeta: int = 32,
    pressure: Any = None,
    gamma: float = 5.0 / 3.0,
    parity: int | str | None = None,
) -> Matrices:
    """Assemble the per-surface continuum matrices with FFT-exact angle quadrature.

    Raises ``ValueError`` (host inputs only) unless ``rho > 0``, ``G + iota I != 0`` and the grid
    resolves the requested couplings: ``ntheta > 2 max|dm|`` and ``nzeta > 2 max|dn|``, since
    ``<f exp(i(dm theta - dn zeta))>`` of an ``N``-point grid aliases ``dm`` with ``dm - N``.

    ``pressure`` (Pa, a scalar or one value per surface; default ``None`` = off) adds the acoustic coupling of
    :func:`sound_coupling` with adiabatic index ``gamma``: the matrices become ``2H x 2H``, the first ``H`` unknowns the
    surface displacement and the last ``H`` the parallel displacement over ``|B|``. Include ``-m`` and ``m`` (``mode_family``
    with ``m_min < 0``) when ``n = 0``, since the geodesic curvature couples ``m`` to ``-m`` through the conjugate sideband.

    These are the matrices of the list as given (``parity=None`` or ``"single"``); :func:`continuum` combines the blocks of
    :func:`spectral_blocks`. ``parity`` (stellarator-symmetric geometry only) gives one real sector of the potential: ``1``
    the ``cos(m theta - n zeta)`` basis (STELLGAP's), ``-1`` the ``sin`` basis (AE3D's); see :func:`parity_basis`.
    """
    h = jnp.asarray(harmonics, dtype=float)
    check_modes(geo, h, density)
    h, project = parity_basis(geo, h, parity)
    check(jnp.all((geo.G + geo.iota * geo.I) != 0), "G + iota I must be nonzero on every surface")
    for name, n, col in (("ntheta", ntheta, 0), ("nzeta", nzeta, 1)):
        check(2 * (h[:, col].max() - h[:, col].min()) < n, f"{name}={n} aliases the harmonics: need {name} > 2 max|d{'mn'[col]}|")
    period = field_period(geo, nzeta)
    grid = surface_grid(geo, ntheta, nzeta, period)
    if pressure is not None:
        check(jnp.all(jnp.isfinite(jnp.asarray(pressure)) & (jnp.asarray(pressure) >= 0)), "pressure must be finite and non-negative")
        check(
            jnp.asarray(pressure).ndim == 0 or jnp.asarray(pressure).shape == geo.s.shape,
            "pressure must be a scalar or have one value per surface",
        )
        check(gamma > 0, "gamma must be positive")
    KM = _continuum_matrices(geo, grid, h, density, pressure, gamma, period)
    return jax.tree_util.tree_map(project, KM) if symmetric(geo) else KM


def parity_basis(geo: Geometry, h: jax.Array, parity: int | str | None) -> tuple[jax.Array, Any]:
    """Harmonics and the map applied to each assembled matrix: ``(h, real part)`` without ``parity``.

    The single-sign basis ``exp(i(m theta - n zeta))`` of ``h`` drops the coupling of ``(m, n)`` to ``(-m', -n')`` through
    the field harmonic ``(m + m', n + n')``, which exists whenever ``2 n`` is a multiple of ``nfp`` (every family at
    ``nfp = 2``, ``n = 0`` in a tokamak). A real potential needs both signs. With stellarator symmetry the problem on
    ``h`` and ``-h`` splits exactly into the ``cos`` (``parity = 1``) and ``sin`` (``-1``) sectors, each real and of size
    ``len(h)``; this returns ``h`` and ``-h`` and the projection onto the requested sector.
    """
    if parity is None or parity == "single":
        return h, jnp.real
    check(parity in (1, -1), "parity must be 1 (cos sector) or -1 (sin sector)")
    check(symmetric(geo), "parity sectors need a stellarator-symmetric geometry")
    check(~jnp.any(jnp.all(h[:, None] == -h[None], -1) & ~jnp.eye(len(h), dtype=bool)), "parity sectors take each (m, n) once up to sign")
    check(parity == 1 or ~jnp.any(jnp.all(h == 0, -1)), "the sin sector has no (0, 0) harmonic")
    q = jnp.concatenate([jnp.eye(len(h)), parity * jnp.eye(len(h))]) / jnp.sqrt(2.0)

    def project(A: jax.Array) -> jax.Array:  # with sound the unknowns are (E, c), each on h and -h
        Q = q if A.shape[-1] == q.shape[0] else jax.scipy.linalg.block_diag(q, q)
        return jnp.einsum("ia,...ij,jb->...ab", Q, A, Q).real

    return jnp.concatenate([h, -h]), project


class Block(NamedTuple):
    """One independent pencil of :func:`spectral_blocks`: assemble ``harmonics`` with ``parity``; its unknown ``k`` is
    ``sign[k]`` times the harmonic of row ``rows[k]`` of the user's list."""

    harmonics: np.ndarray
    parity: int | None
    rows: np.ndarray
    sign: np.ndarray


def spectral_blocks(geo: Geometry, harmonics: Any, parity: int | str | None = None) -> list[Block]:
    """The independent pencils of the real potential spanned by ``harmonics`` (default) or the requested basis.

    A real potential holds ``(m, n)`` and ``(-m, -n)`` together. The operator couples ``(m, n)`` to ``(-m', -n')`` through the
    field harmonic ``(m + m', n + n')``, present when ``n + n'`` is a multiple of ``nfp`` (``n + n' = 0`` in an axisymmetric
    field), which the single-sign list drops. The default therefore solves the list on both signs, split into the blocks
    that do not couple: a family ``N = n mod nfp`` with its conjugate ``-N`` is one complex pencil on ``(m, n)`` of family
    ``N`` and ``(-m, -n)`` of family ``-N`` (the conjugate pencil has the same eigenvalues and is skipped); a self-conjugate
    family (``N = -N``: ``N = 0`` or ``nfp / 2``, or ``n = 0`` in a tokamak) is its cosine and sine sectors, real, when the
    geometry is stellarator symmetric, and one complex pencil on both signs otherwise. A list holding one family ``N != -N``
    is its own block, so the result equals the single-sign one (any tokamak ``n != 0``). Rows whose negative is listed
    earlier are dropped. A traced geometry counts as asymmetric, and as axisymmetric when ``nfp = 1``.
    ``parity="single"`` is the list as given; ``1`` or ``-1`` one sector of it.
    """
    h = np.asarray(harmonics, dtype=float)
    every = np.arange(len(h))
    if parity is not None:
        return [Block(h, None if parity == "single" else parity, every, np.ones(len(h)))]
    keep = np.array([i for i in every if not (h[:i] == -h[i]).all(1).any()])
    axisymmetric = geo.nfp == 1 if isinstance(geo.xn, jax.core.Tracer) else not np.any(np.asarray(geo.xn))
    key = (lambda n: n) if axisymmetric else (lambda n: n % geo.nfp)
    blocks = []
    for F in dict.fromkeys(key(h[keep, 1])):
        if any(key(-b.harmonics[0, 1]) == F for b in blocks):
            continue  # the conjugate of a block already taken
        own, other = keep[key(h[keep, 1]) == F], keep[key(-h[keep, 1]) == F]
        if key(-F) != F:  # family F and its conjugate -F: one complex pencil
            rows, sign = np.concatenate([own, other]), np.concatenate([np.ones(len(own)), -np.ones(len(other))])
            blocks.append(Block(h[rows] * sign[:, None], None, rows, sign))
        elif symmetric(geo):
            for p in (1, -1):
                rows = own if p == 1 else own[(h[own] != 0).any(1)]
                blocks.append(Block(h[rows], p, rows, np.ones(len(rows))))
        else:
            minus = own[(h[own] != 0).any(1)]
            rows, sign = np.concatenate([own, minus]), np.concatenate([np.ones(len(own)), -np.ones(len(minus))])
            blocks.append(Block(h[rows] * sign[:, None], None, rows, sign))
    return blocks


def on_rows(y: jax.Array, block: Block, size: int) -> jax.Array:
    """Vectors ``y`` (last axis the block's unknowns) as complex coefficients ``x`` of the user's harmonics, the potential
    being ``Re sum_j x_j exp(i(m_j theta - n_j zeta))``: a cosine-sector vector is ``x``, a sine-sector one ``-i x``, and a
    ``(-m, -n)`` unknown adds its conjugate to row ``(m, n)``."""
    y = y * (-1j if block.parity == -1 else 1)
    x = jnp.zeros((*y.shape[:-1], size), complex)
    for s in (1, -1):
        cols = np.flatnonzero(block.sign == s)
        x = x.at[..., block.rows[cols]].add(y[..., cols] if s == 1 else jnp.conj(y[..., cols]))
    return x


@partial(jax.jit, static_argnums=6)
def _continuum_matrices(geo: Geometry, grid: Any, h: jax.Array, density: Any, pressure: Any, gamma: float, period: int) -> Matrices:
    weight = jnp.abs(grid.jacobian) * grid.grad_s2 / grid.B**2
    a = (h[:, 1] - h[:, 0] * geo.iota[:, None]) / (geo.G + geo.iota * geo.I)[:, None]  # k_j = a_j |B|
    rho = jnp.broadcast_to(jnp.asarray(density, dtype=float), geo.s.shape)
    K = a[:, :, None] * a[:, None, :] * angle_mean(weight * grid.B**4 / MU0, h, period)
    M = rho[:, None, None] * angle_mean(weight, h, period)
    if pressure is not None:
        pp, pc, cc, mc = sound_coupling(geo, h, rho, pressure, gamma, grid, period)
        Z = jnp.zeros_like(mc)
        K = jnp.concatenate([jnp.concatenate([K + pp, pc], -1), jnp.concatenate([herm(pc), cc], -1)], -2)
        M = jnp.concatenate([jnp.concatenate([M, Z], -1), jnp.concatenate([Z, mc], -1)], -2)
    return Matrices(0.5 * (K + herm(K)), 0.5 * (M + herm(M)))


def continuum(
    geo: Geometry,
    harmonics: Any,
    density: Any,
    ntheta: int = 32,
    nzeta: int = 32,
    pressure: Any = None,
    gamma: float = 5.0 / 3.0,
    parity: int | str | None = None,
) -> jax.Array:
    """Continuum frequencies in Hz, sorted on each surface: every branch of the real potential on ``harmonics``
    (:func:`spectral_blocks`; twice as many with ``pressure``, see :func:`continuum_matrices`). ``parity="single"`` is the
    single-sign list as given (one branch per harmonic), ``1`` or ``-1`` one stellarator-symmetric sector."""
    w2 = [
        generalized_eigh(*continuum_matrices(geo, b.harmonics, density, ntheta, nzeta, pressure, gamma, b.parity))[0]
        for b in spectral_blocks(geo, harmonics, parity)
    ]
    return jnp.sqrt(jnp.maximum(jnp.sort(jnp.concatenate(w2, -1), -1), 0.0)) / (2.0 * jnp.pi)


def gaps(frequencies_hz: Any) -> tuple[jax.Array, jax.Array]:
    """Largest gap between adjacent sorted branches on each surface: ``(center, width)`` in Hz, the width edge to edge
    (a Fourier coefficient ``c`` of the coupling opens ``2 f0 |c|``)."""
    f = jnp.sort(jnp.asarray(frequencies_hz), axis=-1)
    width = jnp.diff(f, axis=-1)
    i = jnp.argmax(width, axis=-1)
    lower = jnp.take_along_axis(f, i[..., None], axis=-1)[..., 0]
    w = jnp.take_along_axis(width, i[..., None], axis=-1)[..., 0]
    return lower + 0.5 * w, w


def gap_location(n: int, m: int, dm: int, dn: int = 0, nfp: int = 1) -> tuple[float, float]:
    """Cylindrical crossing of ``(m, n)`` and ``(m + dm, n + dn nfp)``: ``(iota, w R / v_A)``.

    TAE ``dm = 1``, EAE ``dm = 2``, MAE ``dm = 0, dn != 0``, HAE ``dm, dn != 0``.
    """
    iota = (2 * n + dn * nfp) / (2 * m + dm)
    return iota, abs(dm * n - dn * m * nfp) / (2 * m + dm)
