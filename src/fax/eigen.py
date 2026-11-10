"""Global shear-Alfvén eigenmodes (AE3D weak form) and differentiable eigenvalues.

The reduced ideal-MHD weak form of Spong, D'Azevedo & Todo (2010) for the
electrostatic potential ``phi`` is::

    w^2 int mu0 rho / B^2 grad(phi~)* . grad(phi) dV = int grad(u~)* . grad(u) dV,   u = B.grad(phi) / B,

without the pressure-gradient and parallel-current (kink) terms (``finite_beta``, restricted and experimental:
``docs/theory.md``, section 4, derives them and the order at which they are retained). With ``phi = sum_j f_k(s) e_j``, linear
radial hat functions ``f_k`` on the geometry surfaces and
``e_j = exp(i(m_j theta - n_j zeta))``, the Galerkin matrices ``F y = w^2 G y``
are Hermitian block tridiagonal with ``H x H`` blocks. ``phi = 0`` at ``s = 1``
and, for ``m != 0``, at the axis; ``m = 0`` harmonics are flat between the axis
and the first surface (AE3D's regularity condition). Coefficients are averaged
over each radial element, as in AE3D.

Eigenvalue derivatives use the Hermitian identity ``d lam = y.H (dF - lam dG) y``
for ``G``-normalized ``y``. Averaging over a cluster of eigenvalues gives the
trace of the projected perturbation, which stays smooth through crossings
inside the cluster. That is a first-order statement: ``eigenvalue_mean`` has a NaN second derivative when the
spectrum has a repeated eigenvalue (``eigh`` eigenvectors are undefined there); without one, higher orders work.

``omega^2`` stays signed (``omega^2 < 0`` is an ideal instability): ``frequencies_hz`` is the stable frequency
(0 for ``omega^2 <= 0``) and ``growth_rate`` is ``sqrt(-omega^2)``, both with finite gradients.

Targeted solves are :func:`solvax.generalized_eigenpairs` (block-Thomas shift-invert with a dense pivoted fallback and
implicit first-order derivatives of values and vectors); FAX adds the :class:`Status` report and rejects, with a NaN
tangent, any derivative of an unconverged solve or of a selection that splits a tied set (``gap <= 0``).
"""

from __future__ import annotations

from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import solvax

from .continuum import MU0, generalized_eigh
from .geometry import Geometry, _d_ds, check, surface_grid


class Blocks(NamedTuple):
    """Hermitian block-tridiagonal matrix; ``upper[k]`` couples surfaces ``k`` and ``k + 1``."""

    diag: jax.Array
    upper: jax.Array


class Status(NamedTuple):
    """Solver report: worst relative residual ``||F y - lam G y|| / ((||F|| + |lam| ||G||) ||y||)``, ``||Y^H G Y - I||``,
    iterations used, ``converged`` (residual within ``tol``; False for NaN), the selection ``gap`` (distance of the nearest
    excluded eigenvalue to the target minus that of the farthest selected one; 0 splits a tied set, so no gradient) and
    whether the dense ``fallback`` ran (block Thomas broke down)."""

    residual: jax.Array
    orthogonality: jax.Array
    iterations: jax.Array
    converged: jax.Array
    gap: jax.Array
    fallback: jax.Array


class Eigenmodes(NamedTuple):
    """Global eigenmodes: ``structures`` has shape ``(modes, surfaces, harmonics)``, ``G``-normalized.

    ``omega2`` is signed; ``frequencies_hz`` is 0 and ``growth_rate`` (1/s) positive where ``omega2 < 0``.
    """

    s: jax.Array
    omega2: jax.Array
    frequencies_hz: jax.Array
    structures: jax.Array
    growth_rate: jax.Array
    status: Status


def global_matrices(
    geo: Geometry, harmonics: Any, density: Any, ntheta: int = 32, nzeta: int = 32, finite_beta: bool = False
) -> tuple[Blocks, Blocks]:
    """Bending ``F`` and inertia ``G`` (``w^2`` in rad^2/s^2); surfaces must lie strictly inside ``(0, 1)``.

    ``finite_beta`` (restricted, experimental: no sound or full-MHD claim) adds the Hermitian parts of the ``J_par`` (kink)
    and pressure-curvature terms of ``delta W`` (see :func:`equilibrium_currents` and ``docs/theory.md``, section 4)::

        int sigma b.(grad phi~* x grad u) dV - 2 mu0 int (b x grad phi.grad p)(b x grad phi~*.kappa) / B^2 dV.
    """
    h = jnp.asarray(harmonics, dtype=float)
    check(jnp.all(jnp.asarray(density) > 0), "density must be positive")
    check(jnp.all((geo.s > 0) & (geo.s < 1)), "surfaces must lie strictly inside (0, 1)")
    g = surface_grid(geo, ntheta, nzeta)
    D, (iota_s, D_s) = geo.G + geo.iota * geo.I, _d_ds(jnp.stack([geo.iota, geo.G + geo.iota * geo.I], 1), geo.s).T
    inertia, bending = _modes(g, geo.iota, iota_s, D, D_s, h)
    r = lambda a: a[:, None, None, None]  # noqa: E731 - per-surface value -> (S, H, T, Z)
    rho = jnp.broadcast_to(jnp.asarray(density, dtype=float), geo.s.shape)
    volume = jnp.abs(g.sqrt_g)
    pairs = ((volume, bending), (r(MU0 * rho)[:, 0] * volume / g.B**2, inertia))
    F, G = (_assemble(geo.s, h[:, 0] == 0, *_coefficients(g.metric, w, *x)) for w, x in pairs)
    if not finite_beta:
        return F, G
    p_s, B_s, sigma = equilibrium_currents(geo, g)
    Bc = jnp.stack([B_s, *(jnp.broadcast_to(r(a)[:, 0], B_s.shape) for a in (geo.I, geo.G))], -1)  # (B_s, I, G)
    eps = jnp.array([[[0, 0, 0], [0, 0, 1], [0, -1, 0]], [[0, 0, -1], [0, 0, 0], [1, 0, 0]], [[0, 1, 0], [-1, 0, 0], [0, 0, 0]]])
    cross = jnp.einsum("ijk,...k->...ij", eps, Bc)  # a.cross.c = Bc.(a x c)
    kappa = (
        jnp.concatenate([r(p_s)[:, 0, ..., None] + g.B[..., None] * g.dB[..., :1], g.B[..., None] * g.dB[..., 1:]], -1)
        / g.B[..., None] ** 2
    )
    outer = jnp.cross(kappa, Bc)[..., :, None] * jnp.cross(jnp.array([1.0, 0, 0]), Bc)[..., None, :]
    terms = (
        (cross, sigma * jnp.sign(g.sqrt_g) / g.B, (*inertia, *bending)),
        (outer, -2 * r(p_s)[:, 0] * volume / (g.B**4 * g.sqrt_g**2), (*inertia, *inertia)),
    )
    for tensor, w, x in terms:
        F = jax.tree_util.tree_map(jnp.add, F, _assemble(geo.s, h[:, 0] == 0, *_coefficients(tensor, w, *x)))
    return F, G


def _modes(g: Any, iota: jax.Array, iota_s: jax.Array, D: jax.Array, D_s: jax.Array, h: jax.Array) -> tuple[tuple, tuple]:
    """Covariant ``(s, theta, zeta)`` components, as ``(f' part, f part)``, of ``grad(f e_j)`` (inertia) and ``grad(u_j)``
    (bending) for every harmonic. ``g`` is a :class:`~fax.geometry.SurfaceGrid`; ``D = G + iota I`` and ``iota``, ``D`` are
    per surface with radial derivatives ``iota_s``, ``D_s``; ``u_j = c_j e_j f`` with ``c_j = i (iota m - n) B / D``."""
    m, n = h[:, 0, None, None], h[:, 1, None, None]
    e = jnp.exp(1j * (m * g.theta - n * g.zeta))
    zero = jnp.zeros_like(e)
    r = lambda a: a[:, None, None, None]  # noqa: E731 - per-surface value -> (S, H, T, Z)
    k = r(iota) * m - n
    c = 1j * k * g.B[:, None] / r(D)
    c_s = 1j * (r(iota_s) * m * g.B[:, None] + k * (g.dB[:, None, ..., 0] - g.B[:, None] * r(D_s / D))) / r(D)
    c_t, c_z = 1j * k * g.dB[:, None, ..., 1] / r(D), 1j * k * g.dB[:, None, ..., 2] / r(D)
    inertia = jnp.stack([e, zero, zero], -1), 1j * jnp.stack([zero, m * e, -n * e], -1)
    bending = jnp.stack([c * e, 0 * c, 0 * c], -1), jnp.stack([c_s, c_t + 1j * m * c, c_z - 1j * n * c], -1) * e[..., None]
    return inertia, bending


def equilibrium_currents(geo: Geometry, g: Any) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``mu0 dp/ds``, covariant ``B_s`` and ``sigma = mu0 J.B / B^2`` from Boozer force balance.

    The radial force balance ``(i d_theta + d_zeta) B_s = G' + iota I' + mu0 p' (G + iota I) / B^2``
    fixes ``mu0 p'`` by solvability (its surface average) and ``B_s`` (Pfirsch-Schlueter) by
    inverting ``i (iota m + n)`` in Fourier space, skipping resonant harmonics. Then
    ``sigma = [I (d_zeta B_s - G') - G (d_theta B_s - I')] / (sqrt(g) B^2)``. ``g`` is a
    :class:`~fax.geometry.SurfaceGrid` of ``geo``.
    """
    r = lambda a: a[:, None, None]  # noqa: E731
    I_s, G_s = _d_ds(jnp.stack([geo.I, geo.G], 1), geo.s).T
    w, rhs = 1 / g.B**2, G_s + geo.iota * I_s
    p_s = -rhs / ((geo.G + geo.iota * geo.I) * w.mean((1, 2)))
    T, Z = w.shape[1:]
    k, l = jnp.fft.fftfreq(T, 1 / T)[:, None], jnp.fft.fftfreq(Z, 1 / Z)[None]
    res = r(geo.iota) * k + l
    hat = jnp.fft.fft2(r(rhs) * (1 - w / w.mean((1, 2), keepdims=True))) / jnp.where(jnp.abs(res) > 1e-8, 1j * res, jnp.inf)
    B_s, Bt, Bz = (jnp.fft.ifft2(a * hat).real for a in (1, 1j * k, 1j * l))
    sigma = (r(geo.I) * (Bz - r(G_s)) - r(geo.G) * (Bt - r(I_s))) * w / g.sqrt_g
    return p_s, B_s, sigma


def force_balance_residual(geo: Geometry, ntheta: int = 32, nzeta: int = 32) -> jax.Array:
    """Per surface, the rms difference between ``B_s`` from :func:`equilibrium_currents` (Boozer force balance, before the
    Hermitian symmetrization of the assembled terms) and the covariant radial field of the embedding,
    ``B.r_s = B^zeta (iota g_{s theta} + g_{s zeta})`` with ``B^theta = iota B^zeta``, over the rms of ``|B|``."""
    g = surface_grid(geo, ntheta, nzeta)
    B_s = equilibrium_currents(geo, g)[1]
    cov = jnp.linalg.inv(g.metric)
    iota = geo.iota[:, None, None]
    B_zeta = g.B / jnp.sqrt(iota**2 * cov[..., 1, 1] + 2 * iota * cov[..., 1, 2] + cov[..., 2, 2])
    embedded = B_zeta * (iota * cov[..., 0, 1] + cov[..., 0, 2])
    return jnp.sqrt(((B_s - embedded) ** 2).mean((1, 2)) / (g.B**2).mean((1, 2)))


def _coefficients(metric: jax.Array, w: jax.Array, X0: Any, X1: Any, Y0: Any = None, Y1: Any = None) -> tuple[jax.Array, ...]:
    """Hermitian parts of the surface averages ``<w X_a* . metric . Y_b>`` (``Y = X`` by default)
    for the ``f' f'``, ``f' f`` and ``f f`` products, shape ``(S, H, H)``."""
    Y0, Y1 = (X0, X1) if Y0 is None else (Y0, Y1)
    shape = jnp.broadcast_shapes(*map(jnp.shape, (X0, X1, Y0, Y1)), (w.shape[0], 1, *w.shape[1:], 3))
    X0, X1, Y0, Y1 = (jnp.broadcast_to(a, shape) for a in (X0, X1, Y0, Y1))
    gw = metric * (w / (w.shape[-1] * w.shape[-2]))[..., None, None]
    dot = lambda P, Q: jnp.einsum("shtzi,stzij,sqtzj->shq", jnp.conj(P), gw, Q)  # noqa: E731
    H = lambda a: jnp.conj(jnp.swapaxes(a, -1, -2))  # noqa: E731
    A, D = dot(X0, Y0), dot(X1, Y1)
    return 0.5 * (A + H(A)), 0.5 * (dot(X0, Y1) + H(dot(X1, Y0))), 0.5 * (D + H(D))


def _assemble(s: jax.Array, flat: jax.Array, A: jax.Array, Bm: jax.Array, D: jax.Array) -> Blocks:
    """Linear-element assembly with element-averaged coefficients on nodes ``[0, s..., 1]``."""
    ext = lambda a: jnp.concatenate([a[:1], a, a[-1:]])  # noqa: E731 - axis and edge elements reuse the end surfaces
    A, Bm, D = (0.5 * (ext(a)[1:] + ext(a)[:-1]) for a in (A, Bm, D))
    h = jnp.diff(jnp.concatenate([jnp.zeros(1), s, jnp.ones(1)]))[:, None, None]
    BH = jnp.conj(jnp.swapaxes(Bm, -1, -2))
    LL, RR = A / h - 0.5 * (Bm + BH) + D * h / 3, A / h + 0.5 * (Bm + BH) + D * h / 3
    LR, RL = -A / h - 0.5 * (Bm - BH) + D * h / 6, -A / h + 0.5 * (Bm - BH) + D * h / 6
    P = jnp.diag(flat.astype(float))
    diag = (RR[:-1] + LL[1:]).at[0].add(RL[0] @ P + P @ LR[0] + P @ LL[0] @ P)
    return Blocks(diag, LR[1:-1])


def dense(A: Blocks) -> jax.Array:
    """Dense ``(N H, N H)`` matrix of a block-tridiagonal one."""
    N, H = A.diag.shape[:2]
    i = jnp.arange(N)
    full = jnp.zeros((N, N, H, H), A.diag.dtype).at[i, i].set(A.diag).at[i[:-1], i[1:]].set(A.upper)
    full = full.at[i[1:], i[:-1]].set(jnp.conj(jnp.swapaxes(A.upper, -1, -2)))
    return jnp.swapaxes(full, 1, 2).reshape(N * H, N * H)


def band(A: Blocks) -> tuple[jax.Array, jax.Array, jax.Array]:
    """SOLVAX band triple ``(lower, diag, upper)``: ``upper`` padded with a zero block, ``lower`` its adjoint."""
    upper = jnp.concatenate([A.upper, jnp.zeros_like(A.diag[:1])])
    return jnp.roll(jnp.conj(jnp.swapaxes(upper, -1, -2)), 1, axis=0), A.diag, upper


def _quad(y: jax.Array, A: Blocks) -> jax.Array:
    """``y_k.H A y_k`` for each column ``k`` of ``y`` (shape ``(N, H, p)``)."""
    return jnp.real(jnp.einsum("kip,kip->p", jnp.conj(y), solvax.block_tridiag_matvec(*band(A), y)))


@jax.custom_jvp
def _certified(x: jax.Array, ok: jax.Array) -> jax.Array:
    """``x``, whose tangent is NaN unless ``ok``: an unconverged or tied-selection derivative is rejected, not used."""
    return x


@_certified.defjvp
def _certified_jvp(primals: tuple, tangents: tuple) -> tuple:
    return primals[0], tangents[0] * jnp.where(primals[1], 1.0, jnp.nan)  # a product, so reverse mode also gives NaN


def targeted_eigenpairs(
    F: Blocks, G: Blocks, target: Any, count: int = 8, iters: int = 100, tol: float = 1e-10
) -> tuple[jax.Array, jax.Array, Status]:
    """``count`` eigenpairs of ``F y = lam G y`` nearest ``target`` (ascending) by :func:`solvax.generalized_eigenpairs`.

    Eigenvectors have shape ``(N, H, count)`` and are ``G``-normalized; ``iters`` bounds the shift-invert
    applications and ``tol`` is the relative residual (see :class:`Status`). First derivatives of values and
    vectors are implicit; their tangents are NaN unless the solve ``converged`` and ``gap > 1e-8 max|lam|``
    (smaller gaps are ties).
    """
    r = solvax.generalized_eigenpairs(band(F), band(G), target, count, tol, iters)
    ok = r.converged & (r.gap > 1e-8 * jnp.max(jnp.abs(r.values)))  # a gap of a few ulps is a tie
    status = Status(r.residuals.max(), r.orthogonality, r.iterations, r.converged, r.gap, r.fallback)
    return _certified(r.values, ok), _certified(r.vectors, ok), status


def targeted_eigenvalues(F: Blocks, G: Blocks, target: Any, count: int = 8, iters: int = 100, tol: float = 1e-10) -> jax.Array:
    """Differentiable eigenvalues of :func:`targeted_eigenpairs`, all NaN unless the solve ``converged``."""
    lam, _, status = targeted_eigenpairs(F, G, target, count, iters, tol)
    return lam + jnp.where(status.converged, 0.0, jnp.nan)  # a sum, so the (NaN) tangent is kept


def _dense_status(K: jax.Array, M: jax.Array, lam: jax.Array, X: jax.Array, tol: float) -> Status:
    """:class:`Status` of the dense pairs ``K X = lam M X`` (columns of ``X``)."""
    scale = jnp.linalg.norm(K) + jnp.abs(lam) * jnp.linalg.norm(M)
    r = jnp.max(jnp.linalg.norm(K @ X - lam * (M @ X), axis=0) / (scale * jnp.linalg.norm(X, axis=0)))
    ortho = jnp.linalg.norm(jnp.conj(X.T) @ M @ X - jnp.eye(X.shape[1]))
    return Status(r, ortho, jnp.asarray(0), (r <= tol) & (ortho <= tol), jnp.asarray(jnp.inf), jnp.asarray(False))


@partial(jax.custom_jvp, nondiff_argnums=(2, 3))
def eigenvalue_mean(K: jax.Array, M: jax.Array, start: int = 0, stop: int = 1) -> jax.Array:
    """Mean of dense sorted generalized eigenvalues ``start <= i < stop`` (``stop = start + 1``: one eigenvalue).

    The first derivative is exact through degeneracies inside the cluster. Second derivatives are
    unsupported at a repeated eigenvalue anywhere in the spectrum: they are NaN there.
    """
    return jnp.mean(generalized_eigh(K, M)[0][start:stop])


@eigenvalue_mean.defjvp
def _eigenvalue_mean_jvp(start: int, stop: int, primals: tuple, tangents: tuple) -> tuple[jax.Array, jax.Array]:
    (K, M), (dK, dM) = primals, tangents
    lam, x = generalized_eigh(K, M)
    lam, x = lam[start:stop], x[:, start:stop]
    d = jnp.einsum("ik,ij,jk->k", jnp.conj(x), dK, x) - lam * jnp.einsum("ik,ij,jk->k", jnp.conj(x), dM, x)
    return jnp.mean(lam), jnp.mean(jnp.real(d))


def _root(x: jax.Array) -> jax.Array:
    """``sqrt(max(x, 0))`` with a finite (zero) gradient at ``x <= 0``; NaN propagates."""
    return jnp.where(x <= 0, 0.0, jnp.sqrt(jnp.where(x <= 0, 1.0, x)))


def _hz(omega2: jax.Array) -> jax.Array:
    """Stable frequency (Hz): 0 for ``omega^2 <= 0``; the instability is :func:`_growth`."""
    return _root(omega2) / (2 * jnp.pi)


def _growth(omega2: jax.Array) -> jax.Array:
    """Ideal growth rate ``sqrt(-omega^2)`` (1/s), 0 for ``omega^2 >= 0``."""
    return _root(-omega2)


def eigenmodes(
    geo: Geometry,
    harmonics: Any,
    density: Any,
    *,
    target_hz: float | None = None,
    count: int = 8,
    iters: int = 100,
    tol: float = 1e-10,
    **grid: Any,
) -> Eigenmodes:
    """Global eigenmodes: all of them densely, or ``count`` nearest ``target_hz`` by :func:`targeted_eigenpairs`.

    ``grid`` (``ntheta``, ``nzeta``) is passed to :func:`global_matrices`. ``status`` reports the solve
    (``iterations`` is 0 for the dense solve); unlike :func:`frequency_hz`, the modes are returned even if not ``converged``.
    """
    F, G = global_matrices(geo, harmonics, density, **grid)
    N, H = F.diag.shape[:2]
    if target_hz is None:
        K, M = dense(F), dense(G)
        omega2, x = generalized_eigh(K, M)
        status, x = _dense_status(K, M, omega2, x, tol), jnp.swapaxes(x, 0, 1).reshape(-1, N, H)
    else:
        omega2, y, status = targeted_eigenpairs(F, G, (2 * jnp.pi * target_hz) ** 2, count, iters, tol)
        x = jnp.moveaxis(y, -1, 0)
    return Eigenmodes(geo.s, omega2, _hz(omega2), x, _growth(omega2), status)


def frequency_hz(
    geo: Geometry,
    harmonics: Any,
    density: Any,
    start: int = 0,
    stop: int = 1,
    *,
    target_hz: float | None = None,
    growth: bool = False,
    **options: Any,
) -> jax.Array:
    """Differentiable ``sqrt(mean omega^2) / 2 pi`` of eigenvalues ``start:stop`` (all modes, or those nearest ``target_hz``).

    The stable frequency is 0 if the mean ``omega^2 <= 0``; ``growth=True`` returns ``sqrt(-mean omega^2)`` (1/s)
    instead. Both have finite gradients. With a target the result is NaN unless the solve converged to ``tol``.
    ``options`` are ``count``, ``iters`` and ``tol`` for the targeted solve and the grid sizes.
    """
    solver = {key: options.pop(key) for key in ("count", "iters", "tol") if key in options}
    F, G = global_matrices(geo, harmonics, density, **options)
    if target_hz is None:
        mean = eigenvalue_mean(dense(F), dense(G), start, stop)
    else:
        mean = jnp.mean(targeted_eigenvalues(F, G, (2 * jnp.pi * target_hz) ** 2, **solver)[start:stop])
    return _growth(mean) if growth else _hz(mean)
