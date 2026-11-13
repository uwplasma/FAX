"""Shear-Alfvén continuum on each flux surface (Spong, Sanchez & Weller 2003).

On a surface the low-beta continuum equation is::

    mu0 rho w^2 |grad s|^2 / B^2 E + B.grad(|grad s|^2 / B^2 B.grad E) = 0,

with ``B.grad = B^2/(G + iota I) (d_zeta + iota d_theta)`` in Boozer
coordinates. A Galerkin projection on ``exp(i(m theta - n zeta))`` gives
``K x = w^2 M x`` with::

    K_ij = < |J| |grad s|^2 k_i k_j / mu0 >,   M_ij = < rho |J| |grad s|^2 / B^2 >,
    k_j = (n_j - m_j iota) |B| / (G + iota I).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from .geometry import Geometry, check, surface_grid

MU0 = 4.0e-7 * jnp.pi
AMU = 1.66053906660e-27


class Matrices(NamedTuple):
    """Hermitian stiffness ``K`` and mass ``M``, shape ``(surfaces, H, H)``."""

    K: jax.Array
    M: jax.Array


def mass_density(n_ion: Any, ion_mass_amu: float = 1.0) -> jax.Array:
    """Ion mass density ``rho = n_i m_i`` in kg/m^3 (ion mass in atomic mass units)."""
    return jnp.asarray(n_ion) * ion_mass_amu * AMU


def mode_family(n: int | tuple[int, ...], m_max: int, m_min: int = 0) -> jax.Array:
    """Harmonics ``(m, n)`` for ``m_min <= m <= m_max`` and the given toroidal mode numbers."""
    ns = (n,) if isinstance(n, int) else tuple(n)
    return jnp.asarray([[m, nn] for nn in ns for m in range(m_min, m_max + 1)], dtype=float)


def herm(a: jax.Array) -> jax.Array:
    """Conjugate transpose over the last two axes."""
    return jnp.conj(jnp.swapaxes(a, -1, -2))


def generalized_eigh(K: jax.Array, M: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Solve ``K x = lam M x`` for Hermitian ``K`` and positive-definite ``M`` (batched).

    Eigenvectors are ``M``-orthonormal: ``x.H M x = 1``.
    """
    L = jnp.linalg.cholesky(M)
    tri = lambda b, trans: solve_triangular(L, b, lower=True, trans=trans)  # noqa: E731
    A = herm(tri(herm(tri(K, 0)), 0))  # L^-1 K L^-H by two triangular solves
    lam, y = jnp.linalg.eigh(0.5 * (A + herm(A)))
    return lam, tri(y, 2)  # L^-H y


def angle_mean(f: jax.Array, h: jax.Array) -> jax.Array:
    """``<f exp(i(dm theta - dn zeta))>`` for a real ``f`` of shape ``(..., theta, zeta)`` and all harmonic pairs, shape ``(..., H, H)``
    (row ``i`` conjugated, ``dm = m_j - m_i``): the DFT of ``f`` at ``(-dm, dn)``, one real FFT instead of an ``(H, H, T, Z)`` tensor."""
    T, Z = f.shape[-2:]
    dm, dn = (h[None, :, 0] - h[:, None, 0]).astype(int), (h[None, :, 1] - h[:, None, 1]).astype(int)
    a, b = -dm % T, dn % Z
    hi = b > Z // 2  # beyond the half spectrum of a real f: F[a, b] = conj(F[-a, -b])
    F = jnp.fft.rfft2(f)[..., jnp.where(hi, -a % T, a), jnp.where(hi, Z - b, b)]
    return jnp.where(hi, jnp.conj(F), F) / (T * Z)


def continuum_matrices(geo: Geometry, harmonics: Any, density: Any, ntheta: int = 32, nzeta: int = 32) -> Matrices:
    """Assemble the per-surface continuum matrices with FFT-exact angle quadrature.

    Raises ``ValueError`` (host inputs only) unless ``rho > 0``, ``G + iota I != 0`` and the grid
    resolves the requested couplings: ``ntheta > 2 max|dm|`` and ``nzeta > 2 max|dn|``, since
    ``<f exp(i(dm theta - dn zeta))>`` of an ``N``-point grid aliases ``dm`` with ``dm - N``.
    """
    h = jnp.asarray(harmonics, dtype=float)
    check(jnp.all(jnp.asarray(density) > 0), "density must be positive")
    check(jnp.all((geo.G + geo.iota * geo.I) != 0), "G + iota I must be nonzero on every surface")
    for name, n, col in (("ntheta", ntheta, 0), ("nzeta", nzeta, 1)):
        check(2 * (h[:, col].max() - h[:, col].min()) < n, f"{name}={n} aliases the harmonics: need {name} > 2 max|d{'mn'[col]}|")
    grid = surface_grid(geo, ntheta, nzeta)
    weight = jnp.abs(grid.jacobian) * grid.grad_s2 / grid.B**2
    a = (h[:, 1] - h[:, 0] * geo.iota[:, None]) / (geo.G + geo.iota * geo.I)[:, None]  # k_j = a_j |B|
    rho = jnp.broadcast_to(jnp.asarray(density, dtype=float), geo.s.shape)
    K = a[:, :, None] * a[:, None, :] * angle_mean(weight * grid.B**4 / MU0, h)
    M = rho[:, None, None] * angle_mean(weight, h)
    return Matrices(0.5 * (K + herm(K)), 0.5 * (M + herm(M)))


def continuum(geo: Geometry, harmonics: Any, density: Any, ntheta: int = 32, nzeta: int = 32) -> jax.Array:
    """Continuum frequencies in Hz, sorted on each surface, shape ``(surfaces, H)``."""
    K, M = continuum_matrices(geo, harmonics, density, ntheta, nzeta)
    w2 = generalized_eigh(K, M)[0]
    return jnp.sqrt(jnp.maximum(w2, 0.0)) / (2.0 * jnp.pi)


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
