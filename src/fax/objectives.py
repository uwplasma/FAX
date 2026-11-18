"""Smooth Alfvén objectives and the differentiable VMEC-boundary pipeline.

``boundary offsets -> vmex implicit equilibrium -> booz_xform_jax -> FAX``.
The equilibrium is differentiated with the implicit-function theorem
(``vmex.core.implicit.solve_implicit``), so gradients are those of the
converged equilibrium rather than of a truncated solver trajectory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .continuum import continuum, continuum_matrices, generalized_eigh, on_rows, spectral_blocks
from .drive import FastIons, growth_rate
from .eigen import Status, _hz, eigenmodes, selected, solve, split_solver
from .geometry import Geometry, from_booz_xform, surface_grid


def window_risk(frequencies_hz: Any, window_hz: tuple[float, float], softness_hz: float) -> jax.Array:
    """``0.5 mean(sigma^2)`` where ``sigma`` is a smooth indicator of ``f`` inside ``window_hz``."""
    f = jnp.asarray(frequencies_hz)
    inside = jax.nn.sigmoid((f - window_hz[0]) / softness_hz) * jax.nn.sigmoid((window_hz[1] - f) / softness_hz)
    return 0.5 * jnp.mean(inside**2)


def soft_max(x: Any, sharpness: float = 10.0) -> jax.Array:
    """Log-sum-exp smooth maximum; it avoids the kinks of ``max`` at eigenvalue crossings."""
    return jax.nn.logsumexp(sharpness * jnp.asarray(x)) / sharpness


def drive_risk(gammas: Any, threshold: float = 0.0, scale: float = 1.0e3) -> jax.Array:
    """``softplus(soft_max(gamma_k) - gamma_th)`` in units of ``scale`` (1/s): a smooth penalty on the most unstable mode."""
    return jax.nn.softplus(soft_max(jnp.asarray(gammas) / scale) - threshold / scale)


def gap_closure(frequencies_hz: Any, scale_hz: float = 1.0e4) -> jax.Array:
    """Smooth penalty that grows as the widest continuum gap on each surface closes.

    The widest gap is the softmax-weighted mean ``sum_k p_k w_k``, ``p = softmax(w / scale_hz)``, of the widths ``w_k`` of the
    adjacent sorted branches, not their ``argmax``: it is continuous where two gaps trade places and lies within
    ``scale_hz log(K)`` below the true maximum of ``K`` gaps. The penalty is ``mean_s softplus(-w_s / scale_hz)``.
    """
    w = jnp.diff(jnp.sort(jnp.asarray(frequencies_hz), axis=-1), axis=-1)
    return jnp.mean(jax.nn.softplus(-jnp.sum(jax.nn.softmax(w / scale_hz, axis=-1) * w, axis=-1) / scale_hz))


def compression_harmonics(geo: Geometry, ntheta: int = 32, nzeta: int = 64) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Complex harmonics of ``eps = |grad s|^2 / <|grad s|^2> - 1`` (Paul et al. 2025, eq. 4.28).

    ``eps = sum c[:, i, j] exp(i (dm[i] theta - n[j] zeta))`` in Boozer angles, with
    ``<.>`` the angle average and ``n`` the full toroidal mode number (``N_P dn``). The harmonic
    ``(dm, n)`` opens a gap at ``w0 = |iota dm - n| w_A / 2`` of **full width** ``2 w0 |c|`` (edge to edge;
    ``c`` and its conjugate ``(-dm, -n)`` together make the real amplitude ``2 |c|``).
    """
    g = surface_grid(geo, ntheta, nzeta).grad_s2
    c = jnp.fft.fft2(g / jnp.mean(g, axis=(1, 2), keepdims=True) - 1.0) / (ntheta * nzeta)
    return c, jnp.fft.fftfreq(ntheta, 1.0 / ntheta), -jnp.fft.fftfreq(nzeta, 1.0 / nzeta)


def _nodes(geo: Geometry, n: int) -> Geometry:
    """``geo`` on ``n`` uniform interior finite-element nodes ``k / (n + 1)``."""
    return geo.interpolate(jnp.arange(1, n + 1) / (n + 1))


def continuum_cost(
    geo: Geometry, helicity: tuple[int, int] = (1, 1), ntheta: int = 32, nzeta: int = 64, softness: float = 0.05
) -> jax.Array:
    """Paul et al. (2025) eq. 7.4: ``sum_s sum m_k |eps_{dm,n}|^2 (iota dm - n)^2`` over the gaps above the helicity gap.

    Each term is the squared full width ``(2 w0 |c| / w_A)^2 = |c|^2 (iota dm - n)^2`` of the gap that ``(dm, n)``
    opens at ``w0 / w_A = |iota dm - n| / 2``, so the cost narrows every gap above the frequency of the
    quasisymmetry helicity ``(M, N)``. The paper's hard cut ``|k| > |iota M - N N_P|``, ``k = iota dm - n``, jumps when
    ``iota`` moves a harmonic across it; here the membership is the C1 smoothstep ``m = 3t^2 - 2t^3`` of
    ``t = clip((|k| - cut) / softness, 0, 1)``: 0 up to the cut (so the helicity harmonic itself stays out), 1 from
    ``cut + softness`` on and continuous with its derivative between (``softness`` is in units of ``|k|``, i.e. ``2 w0 / w_A``).
    """
    c, dm, n = compression_harmonics(geo, ntheta, nzeta)
    k = geo.iota[:, None, None] * dm[:, None] - n
    cut = jnp.abs(geo.iota * helicity[0] - helicity[1] * geo.nfp)[:, None, None]
    t = jnp.clip((jnp.abs(k) - cut) / softness, 0.0, 1.0)
    return jnp.sum((3 * t**2 - 2 * t**3) * jnp.abs(c) ** 2 * k**2)


class ModeClass(NamedTuple):
    """Per-mode result of :func:`classify_modes` (host arrays, one entry per mode of the coarse solve, ascending in its frequency).

    ``frequency_hz`` and ``peak_s`` (radius of the largest ``|phi|^2``) are those of the refined solve; ``shift`` is the
    relative frequency change and ``overlap`` the nodal ``L2`` overlap with the best match of the refined solve; ``margin``
    is ``min(f - f_lo, f_hi - f) / f`` to the adjacent local continuum branches at ``peak_s``; ``localization`` is the growth
    of the derivative-jump participation count under the refinement (about 1 for a mode pinned to a fixed number of cells,
    about 2 for one of fixed radial width); ``isolated`` combines the tests of :func:`classify_modes`.
    """

    frequency_hz: np.ndarray
    peak_s: np.ndarray
    shift: np.ndarray
    overlap: np.ndarray
    margin: np.ndarray
    localization: np.ndarray
    isolated: np.ndarray


def classify_modes(
    geo: Geometry,
    harmonics: Any,
    density: Any,
    *,
    nodes: int = 96,
    count: int = 8,
    shift_tol: float = 1.0e-2,
    overlap_tol: float = 0.9,
    margin_tol: float = 0.05,
    **options: Any,
) -> ModeClass:
    """Classify the ``count`` lowest (with ``target_hz``: nearest) global modes as isolated gap modes or continuum-like.

    ``geo`` is interpolated onto ``nodes`` uniform FE nodes ``k / (nodes + 1)`` and onto the nested ``2 nodes + 1`` of the
    refined grid, and both are solved (``options`` go to :func:`fax.eigenmodes`: ``target_hz``, ``iters``, ``ntheta``,
    ``nzeta``). A mode is **isolated** if and only if

    * both solves converged;
    * refinement: its frequency moves by less than ``shift_tol`` (relative) to its best match among the refined modes and
      the nodal ``L2`` overlap of the prolonged mode with that match exceeds ``overlap_tol``. The overlap is not the ``G``
      one, since a resonant tail makes ``phi'`` singular and dominates the ``G`` norm; a discrete continuum mode is a
      different mode at every resolution;
    * gap membership: at ``peak_s`` the frequency is further than ``margin_tol`` (relative) from every branch of
      :func:`fax.continuum`. A continuum-like mode peaks where its frequency meets a branch (a margin of one cell of
      ``df/ds``, percent level), a gap mode inside a gap.

    A gap mode may carry a weak resonant tail (the CTOK TAE does), so ``localization`` is reported but not tested. The
    result serves selection and reporting and is not differentiable.
    """
    solve = lambda n, k: eigenmodes(_nodes(geo, n), harmonics, density, count=k, **options)  # noqa: E731
    coarse, fine = solve(nodes, count), solve(2 * nodes + 1, 2 * count + 4)  # twice the nodes: twice the continuum modes per window
    n = 2 * nodes + 1
    f_c, f_f = (np.asarray(m.frequencies_hz) for m in (coarse, fine))
    y_c, y_f, f_c = np.asarray(coarse.structures)[:count], np.asarray(fine.structures), f_c[:count]
    m = np.asarray(harmonics)[:, 0]
    mid = np.concatenate([np.where(m == 0, y_c[:, :1], 0.0), y_c, np.zeros_like(y_c[:, :1])], 1)  # axis and edge values
    up = np.zeros((count, n, m.size), complex)
    up[:, 1::2], up[:, 0::2] = y_c, 0.5 * (mid[:, :-1] + mid[:, 1:])  # the P1 prolongation onto the nested grid
    norm = lambda y: np.sqrt((np.abs(y) ** 2).sum((1, 2)))  # noqa: E731
    overlap = np.abs(np.einsum("qkh,pkh->pq", y_f.conj(), up)) / norm(up)[:, None] / norm(y_f)[None]
    best = overlap.argmax(1)
    f_b, y_b = f_f[best], y_f[best]
    peak = np.argmax((np.abs(y_b) ** 2).sum(-1), axis=1)
    fine_geo = _nodes(geo, n)
    branches = np.asarray(continuum(fine_geo, harmonics, density, options.get("ntheta", 32), options.get("nzeta", 32)))[peak]
    lower = np.where(branches <= f_b[:, None], branches, 0.0).max(1)
    upper = np.where(branches >= f_b[:, None], branches, np.inf).min(1)
    margin = np.minimum(f_b - lower, upper - f_b) / np.maximum(f_b, 1e-30)
    cells = lambda y: ((j := (np.abs(np.diff(y, axis=1)) ** 2).sum(-1)).sum(1)) ** 2 / (j**2).sum(1)  # noqa: E731
    shift, overlap = np.abs(f_c - f_b) / np.maximum(f_b, 1e-30), overlap[np.arange(count), best]
    ok = bool(coarse.status.converged & fine.status.converged) & (shift < shift_tol) & (overlap > overlap_tol) & (margin > margin_tol)
    return ModeClass(f_b, np.asarray(fine_geo.s)[peak], shift, overlap, margin, cells(y_b) / cells(y_c), ok)


_BOUNDARY = {"rc": "rbc", "rs": "rbs", "zc": "zbc", "zs": "zbs"}


def _slot(parameter: str | tuple[str, int, int], ntor: int) -> tuple[str, int, int]:
    """``(field, row, m)`` of a boundary coefficient given as ``("rc", m, n)`` or ``"rc" + m + n`` (single digits, ``n`` signed)."""
    if isinstance(parameter, str):
        found = re.fullmatch(r"([rz][cs])(\d)(-?\d)", parameter)
        if found is None:
            raise ValueError(f"cannot read {parameter!r}: use 'rc01' (one digit each) or the tuple ('rc', m, n) for multi-digit modes")
        parameter = (found[1], int(found[2]), int(found[3]))
    name, m, n = parameter
    return _BOUNDARY[name], n + ntor, m


@dataclass(frozen=True)
class Design:
    """VMEC-boundary design problem; build it with :func:`prepare_design`."""

    input_path: str
    names: tuple[str, ...]
    slots: tuple[tuple[str, int, int], ...]
    harmonics: Any
    density: float
    nodes: int
    options: dict[str, Any] = field(default_factory=dict)
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)


def prepare_design(
    input_path: str | Path,
    *,
    harmonics: Any,
    density: float,
    parameters: tuple[str | tuple[str, int, int], ...] = ("rc01",),
    nodes: int = 24,
    mboz: int = 3,
    nboz: int = 3,
    ns: int = 16,
    ftol: float = 1.0e-12,
    max_iter: int = 5000,
    **options: Any,
) -> Design:
    """Prepare a design: parameters are boundary names ``rc|rs|zc|zs`` + ``m`` + ``n`` (e.g. ``"zs11"``, one digit each,
    ``n`` signed) or tuples ``("zs", m, n)``, which are required for multi-digit modes.

    The equilibrium (``ns``, ``ftol``) is transformed on the half-mesh rows ``j >= 1`` and
    interpolated linearly in ``s`` onto ``nodes`` uniform interior finite-element
    nodes ``k / (nodes + 1)``, so the radial domain is always ``[0, 1]`` and the
    FE grid refines independently of ``ns`` and ``mboz``. ``options`` go to
    :func:`fax.eigen.global_matrices` (``ntheta``, ``nzeta``) and the
    shift-invert solve (``target_hz``, ``count``, ``iters``).
    """
    from vmex import VmecInput
    from vmex.core import implicit

    path = str(Path(input_path).expanduser().resolve())
    inp = VmecInput.from_file(path)
    config = implicit.make_config(inp, ns=ns, ftol=ftol, max_iterations=max_iter)
    params = implicit.params_from_input(inp)
    slots = tuple(_slot(name, int(inp.ntor)) for name in parameters)
    s_full = np.asarray(implicit.runtime_from_params(params, config).setup.s_full)
    s = jnp.asarray(0.5 * (s_full[:-1] + s_full[1:]))  # every half-mesh row
    cache = dict(config=config, params=params, s=s, mboz=int(mboz), nboz=int(nboz))
    return Design(
        path, tuple(map(str, parameters)), slots, jnp.asarray(harmonics, dtype=float), float(density), int(nodes), dict(options), cache
    )


def _compiled(design: Design, key: Any, function: Any) -> Any:
    """``jax.jit(function)`` kept on the design: one compiled program per stage, reused by every call and gradient."""
    cache = design._cache.setdefault("jit", {})
    if key not in cache:
        cache[key] = jax.jit(function)
    return cache[key]


def design_params(offsets: Any, design: Design) -> Any:
    """vmex ``ImplicitParams`` of the boundary ``input + offsets``."""
    params = design._cache["params"]
    for value, (name, row, m) in zip(jnp.asarray(offsets), design.slots, strict=True):
        params = replace(params, **{name: getattr(params, name).at[row, m].add(value)})
    return params


def state_geometry(state: Any, runtime: Any, design: Design) -> Geometry:
    """Boozer geometry on the half-mesh rows ``j >= 1`` from a vmex state and runtime (traceable), one call to vmex."""
    from vmex.core.omnigenity import boozer_spectrum_state

    c = design._cache
    out = boozer_spectrum_state(state, runtime, surfaces=c["s"], mboz=c["mboz"], nboz=c["nboz"], oversample=1, geometry=True)
    return from_booz_xform(out, out["s_b"], out["nfp"])


def design_state(offsets: Any, design: Design) -> tuple[Any, Any]:
    """vmex ``(state, runtime)`` of the boundary ``input + offsets``, for vmex objectives (QS, aspect, iota)."""
    from vmex.core import implicit

    params, config = design_params(offsets, design), design._cache["config"]
    return implicit.solve_implicit(params, config), implicit.runtime_from_params(params, config)


def design_geometry(offsets: Any, design: Design) -> Geometry:
    """Half-mesh Boozer geometry of the boundary ``input + offsets`` (traceable in ``offsets``)."""
    return _compiled(design, "geometry", lambda s, r: state_geometry(s, r, design))(*design_state(offsets, design))


def _reject(x: jax.Array, status: Status, omega2: jax.Array) -> jax.Array:
    """``x``, NaN (value and tangent) unless the solve ``converged`` and its selection ``gap`` exceeds ``1e-8 max|omega^2|``
    (a smaller gap is a tie, as in :func:`fax.eigen.targeted_eigenpairs`; plan rule 3)."""
    return x + jnp.where(selected(status.converged, status.gap, omega2), 0.0, jnp.nan)


def geometry_frequencies(geo: Geometry, design: Design, mode_count: int = 3) -> tuple[jax.Array, Status]:
    """Global frequencies (Hz) of ``geo`` on the design's FE nodes and the solver :class:`~fax.eigen.Status`.

    Without ``target_hz`` the lowest ``mode_count`` from the dense solve (``status.gap`` is the distance of the next
    eigenvalue); with it, the first ``mode_count`` of the ``count`` modes nearest ``target_hz`` (shift-invert). The frequencies
    are NaN, with NaN derivatives, unless the solve converged and the selection does not split a tied set (``gap > 1e-8 max|omega^2|``).
    They carry first-order Hellmann-Feynman derivatives only; use :func:`classify_modes` to decide whether a selected
    mode is a gap mode rather than continuum-like before treating it as one.
    """
    solver, opts = split_solver(design.options)
    target = opts.pop("target_hz", None)
    target = None if target is None else (2 * jnp.pi * target) ** 2
    lam, _, status = solve(_nodes(geo, design.nodes), design.harmonics, design.density, target, **solver, vectors=False, **opts)
    if target is None:
        status = status._replace(gap=jax.lax.stop_gradient(lam[mode_count] - lam[mode_count - 1]))
    return _reject(_hz(lam[:mode_count]), status, lam), status


def design_frequencies(offsets: Any, design: Design, mode_count: int = 3) -> tuple[jax.Array, Status]:
    """``boundary offsets -> vmex -> booz_xform_jax -> FE nodes -> (frequencies in Hz, Status)``, differentiable (first order)."""
    run = _compiled(design, ("frequencies", mode_count), lambda g: geometry_frequencies(g, design, mode_count))
    return run(design_geometry(offsets, design))


def geometry_growth_rates(geo: Geometry, design: Design, ions: FastIons, mode_count: int = 3, **drive: Any) -> jax.Array:
    """Fast-ion growth rates (1/s) of the first ``mode_count`` of the ``count`` modes nearest ``target_hz``.

    The modes come from the shift-invert solve on the design's FE nodes (differentiated
    through the iteration); ``ions.density`` is a scalar or a profile on those nodes and
    ``drive`` goes to :func:`fax.growth_rate`. The rates are NaN unless the solve converged and its selection
    does not split a tied set (``gap > 1e-8 max|omega^2|``).
    """
    geo = _nodes(geo, design.nodes)
    ev = eigenmodes(geo, design.harmonics, design.density, **design.options)
    rate = lambda w2, y: growth_rate(jnp.sqrt(w2), y, geo, design.harmonics, ions, design.density, **drive)  # noqa: E731
    return _reject(jnp.stack([rate(ev.omega2[i], ev.structures[i]) for i in range(mode_count)]), ev.status, ev.omega2)


def _gap_type(dm: int, dn: int, sound: bool = False) -> str:
    """Name of a gap from the poloidal and toroidal (in field periods) mode-number differences of the coupled harmonics."""
    if sound:
        return "BAE"
    if dn == 0:
        return {0: "other", 1: "TAE", 2: "EAE"}.get(dm, "NAE")  # NAE: |dm| >= 3 (Spong et al. 2003, table I)
    return "MAE" if dm == 0 else "HAE"


def _axisymmetric(geo: Geometry) -> bool:
    """True when no harmonic of ``|B|``, ``R`` or ``Z`` depends on the toroidal angle."""
    spectra = np.abs(np.concatenate([np.asarray(getattr(geo, k)) for k in ("bmnc", "bmns", "rmnc", "rmns", "zmns", "zmnc")], -1))
    return bool(spectra[:, np.tile(np.asarray(geo.xn), 6) != 0].max(initial=0.0) <= 1e-9 * spectra.max())


def mode_families(geo: Geometry, harmonics: Any) -> list[np.ndarray]:
    """Row indices of ``harmonics`` that the equilibrium couples in a real potential: one group per ``|n|`` when the geometry is
    axisymmetric, one per pair ``{N, -N}``, ``N = n mod nfp``, otherwise (``(m, n)`` couples to ``(-m', -n')`` when
    ``n + n'`` is a multiple of ``nfp``, see :func:`fax.continuum.spectral_blocks`). Different groups never couple, so their
    continua are separate spectra."""
    h = np.asarray(harmonics, dtype=int)
    key = np.abs(h[:, 1]) if _axisymmetric(geo) else np.minimum(np.mod(h[:, 1], geo.nfp), np.mod(-h[:, 1], geo.nfp))
    return [np.flatnonzero(key == k) for k in np.unique(key)]


def alfven_harmonics(geo: Geometry, m_max: int = 12, width: float | None = None) -> np.ndarray:
    """A harmonic set ``(m, n)``, ``0 <= m <= m_max``, holding the resonant branches of each distinct mode family.

    Axisymmetric: ``n = 1, 2`` with the sign of the rotational transform (``k_par ~ n - m iota`` vanishes only there).
    Otherwise every family ``N = 0 .. nfp - 1`` with each ``n = N mod nfp`` in ``[iota_min m - W, iota_max m + W]``,
    ``W = width`` (default ``2 max(2, nfp)``, in units of ``v_A / R``), as in Paul et al. (2025, section 5.1); both signs
    of ``n`` thereby enter, and a negative ``iota`` selects negative ``n``. With ``m >= 0`` only, family ``nfp - N`` holds
    the resonant branches that family ``N`` has at ``m < 0`` (its complex conjugate), so all ``N`` are kept. ``(0, 0)`` is
    left out.
    """
    iota = np.asarray(geo.iota, dtype=float)
    lo, hi = iota.min(), iota.max()
    if _axisymmetric(geo):
        sign = 1 if np.mean(iota) >= 0 else -1
        pairs = [(m, sign * n) for n in (1, 2) for m in range(m_max + 1)]
    else:
        nfp, w = geo.nfp, 2 * max(2, geo.nfp) if width is None else width
        pairs = [
            (m, n)
            for N in range(nfp)
            for m in range(m_max + 1)
            for n in range(int(np.floor(min(lo * m, hi * m) - w)), int(np.ceil(max(lo * m, hi * m) + w)) + 1)
            if (n - N) % nfp == 0 and (m, n) != (0, 0)
        ]
    return np.array(sorted(set(pairs)), dtype=float)


def _local_gaps(f: np.ndarray, min_width: float, edge: int) -> list[tuple[int, int, int, int, float, float]]:
    """Avoided crossings of adjacent sorted branches ``k, k + 1``: a local minimum of their separation at surface ``i``
    (at least ``edge`` surfaces from either end) wider than ``min_width`` of its centre. The band is ``[max f_k, min f_k+1]``
    over the run ``a..b`` of surfaces around ``i``, short of the neighbouring maxima of the separation (the next crossing
    of the same pair), where it stays at least half as wide as at ``i``.
    Returns ``(k, i, a, b, low, high)``."""
    S, found = len(f), []
    for k in range(f.shape[1] - 1):
        d = f[:, k + 1] - f[:, k]
        for i in range(edge, S - edge):
            if not (d[i] < d[i - 1] and d[i] <= d[i + 1] and d[i] >= min_width * 0.5 * (f[i, k] + f[i, k + 1])):
                continue  # one minimum per valley of the separation (the first of a flat bottom)
            peaks = [j for j in range(1, S - 1) if d[j] > d[j - 1] and d[j] >= d[j + 1]]  # the next crossings' territory
            first, last = max([j for j in peaks if j < i], default=0), min([j for j in peaks if j > i], default=S - 1)
            a = b = i
            while a > first and min(f[a - 1 : b + 1, k + 1]) - max(f[a - 1 : b + 1, k]) >= 0.5 * d[i]:
                a -= 1
            while b < last and min(f[a : b + 2, k + 1]) - max(f[a : b + 2, k]) >= 0.5 * d[i]:
                b += 1
            found.append((k, i, a, b, float(f[a : b + 1, k].max()), float(f[a : b + 1, k + 1].min())))
    return found


def _spectrum(geo: Geometry, h: np.ndarray, rho: Any, pressure: Any, grid: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sorted continuum (kHz), branch power per unknown (the listed harmonics, then their sound unknowns) and the dominant
    ``m`` of each branch (NaN for a sound branch), from the blocks of :func:`fax.continuum.spectral_blocks`."""
    w2, power = [], []
    for b in spectral_blocks(geo, h):
        lam, y = generalized_eigh(*continuum_matrices(geo, b.harmonics, rho, pressure=pressure, parity=b.parity, **grid))
        y, n = np.swapaxes(np.asarray(y), 1, 2), len(b.rows)  # (surfaces, branches, unknowns)
        x = [on_rows(y[..., i : i + n], b, len(h)) for i in range(0, y.shape[-1], n)]  # Alfven, then sound unknowns
        w2.append(np.asarray(lam))
        power.append(np.swapaxes(np.abs(np.concatenate(x, -1)) ** 2, 1, 2))
    order = np.argsort(np.concatenate(w2, -1), -1)
    w2 = np.take_along_axis(np.concatenate(w2, -1), order, -1)
    power = np.take_along_axis(np.concatenate(power, -1), order[:, None], -1)  # (surfaces, unknowns, branches)
    dom = np.argmax(power, 1)
    return np.sqrt(np.maximum(np.asarray(w2), 0.0)) / (2 * np.pi) / 1e3, power, np.where(dom < len(h), h[dom % len(h), 0], np.nan)


def summary(
    geometry: Geometry,
    harmonics: Any,
    mass_density: Any,
    *,
    pressure: Any = None,
    window: tuple[float, float] | None = None,
    count: int = 4,
    ions: tuple[float, float] | None = None,
    max_gaps: int = 6,
    min_width: float = 0.05,
    edge: float = 0.05,
    ntheta: int = 32,
    nzeta: int = 32,
    iters: int = 100,
    tol: float = 1e-10,
) -> dict[str, Any]:
    """Continuum gaps and a gap eigenmode, as one plain dictionary (host arrays, not differentiable).

    ``mass_density`` is the ion mass density in kg/m^3, a scalar or one value per surface (:func:`fax.mass_density`);
    ``pressure`` (Pa, scalar or per surface) adds the sound coupling of :func:`fax.continuum` to the drawn continuum and
    opens BAE gaps; the shear-Alfven gaps, the eigenmode and its continuum check stay without it, since the sound branches
    cut through those gaps with little coupling. ``ions = (n_i, amu)`` only records what the density was built from.

    The harmonics are split into the families the equilibrium couples (:func:`mode_families`) and each family's continuum
    is searched for **local gaps**: avoided crossings of adjacent branches, each with a frequency band and the radial range
    ``s_range`` where that band is open. Each branch is labelled by its own dominant harmonic at the crossing, and a gap is
    kept only if the two harmonics cross there: ``k_par R = n - m iota`` of opposite signs and magnitudes within a factor
    two (Spong et al. 2003), one gap per pair. Gaps are typed from the two harmonics:
    TAE (``dm = 1``), EAE (``dm = 2``), NAE (``|dm| >= 3``), MAE (``dm = 0``), HAE (``dn != 0``), and BAE when either branch
    holds a tenth of its energy in the sound displacement (the sound continuum splits one Alfven harmonic).
    Gaps whose harmonics include the highest ``|m|`` of their family are truncation artefacts and dropped (the resonant
    branches of every ``m``, ``m_max`` included, reach zero at their rational surfaces, so a frequency cut cannot mark the
    truncation), as are gaps centred below ``0.05 f_A`` or narrower than ``0.01 f_A``.
    ``window = (low, high)`` in kHz (default ``(0, max(2, nfp) f_A)``, which holds the TAE, EAE and HAE gaps) keeps gaps
    whose centre lies inside. The ``max_gaps`` relatively widest are returned, widest first, BAE gaps last.

    For each gap other than BAE in turn, the ``count`` global modes of that family nearest the gap centre are solved
    (:func:`fax.eigenmodes`). A mode is kept if its frequency is inside the band, it spans at least three surfaces where
    its power exceeds a tenth of the peak, it is smooth on the radial grid (second differences carry less than half its
    energy), both harmonics of the gap reach a fifth of its peak amplitude (a gap mode is their
    coupling; one harmonic alone is a continuum mode), it peaks within 0.1 in ``s`` of the gap's radial range, it crosses no
    shear-Alfven branch where its power exceeds a hundredth of the
    peak, and its peak lies at least ``edge`` from the axis and the boundary.
    Otherwise ``mode`` is ``None`` and ``mode_reason`` says why.

    Returns ``harmonics``, density fields, ``alfven_khz`` (``v_A / 2 pi R_0`` on the innermost surface), ``window_khz``,
    ``gaps`` (``center_khz``, ``width_khz``, ``low_khz``, ``high_khz``, ``s``, ``s_range``, ``iota``, ``type``,
    ``harmonics``, ``family``), ``mode`` (``frequency_khz``, ``peak_s``, ``support``, ``family``, ``gap``, ``converged``,
    ``residual``, ``structure`` (phase-aligned, real, shape ``(surfaces, family harmonics)``), ``harmonics``,
    ``cross_section`` (``R``, ``Z``, ``phi`` at ``zeta = 0``)), ``mode_reason``, ``family`` (the index drawn by
    :func:`fax.plot.plot_summary`) and ``profiles`` (``s``, ``iota``, ``families``: ``harmonics``, ``continuum_khz``,
    ``dominant_m``, and ``shear_alfven_khz``, the continuum without sound).
    """
    h = np.asarray(harmonics, dtype=float)
    rho = np.asarray(mass_density, dtype=float)
    grid = {"ntheta": ntheta, "nzeta": nzeta}
    s, iota = np.asarray(geometry.s), np.asarray(geometry.iota)
    xm, xn = np.asarray(geometry.xm), np.asarray(geometry.xn)
    m0 = np.flatnonzero((xm == 0) & (xn == 0))[0]
    B0, R0 = abs(float(geometry.bmnc[0, m0])), float(geometry.rmnc[0, m0])
    f_alfven = B0 / np.sqrt(4e-7 * np.pi * float(np.ravel(rho)[0])) / (2 * np.pi * R0) / 1e3
    edge_i = max(1, int(np.ceil(edge * len(s))))
    window = (0.0, max(2, geometry.nfp) * f_alfven) if window is None else window
    families, gaps = [], []
    for fi, rows in enumerate(mode_families(geometry, h)):
        hf, H = h[rows], len(rows)
        spectra = [_spectrum(geometry, hf, rho, None, grid)]  # gaps and crossings of the shear-Alfven modes
        if pressure is not None:  # BAE gaps only: sound branches cut through the Alfven gaps without closing them
            spectra.append(_spectrum(geometry, hf, rho, pressure, grid))
        mmax = np.abs(hf[:, 0]).max()
        families.append(
            dict(
                harmonics=hf.astype(int).tolist(),
                continuum_khz=spectra[-1][0],
                dominant_m=spectra[-1][2],
                shear_alfven_khz=spectra[0][0],
            )
        )
        for with_sound, (f, power, _) in enumerate(spectra):
            for k, i, a, b, low, high in _local_gaps(f, min_width, edge_i):
                center = 0.5 * (low + high)
                pk = power[i][:, k : k + 2]
                sound = bool((pk[H:].sum(0) > 0.1 * pk.sum(0)).any())  # a branch with a tenth of its energy in sound
                ua, ub = pk[:H].argmax(0)  # each branch's own shear-Alfven harmonic
                (ma, na), (mb, nb) = hf[ua], hf[ub]
                if sound != bool(with_sound) or (not sound and (ua == ub or mmax in (abs(ma), abs(mb)))):
                    continue
                ka, kb = na - ma * iota[i], nb - mb * iota[i]  # k_par R of the two branches
                if not sound and (ka * kb >= 0 or not 0.5 <= abs(ka / kb) <= 2):
                    continue  # not two counter-propagating branches meeting: sorted neighbours, no crossing
                if center < 0.05 * f_alfven or high - low < 0.01 * f_alfven or not window[0] <= center <= window[1]:
                    continue
                gaps.append(
                    dict(
                        center_khz=float(center),
                        width_khz=float(high - low),
                        low_khz=float(low),
                        high_khz=float(high),
                        s=float(s[i]),
                        s_range=[float(s[a]), float(s[b])],
                        iota=float(iota[i]),
                        type=_gap_type(int(abs(ma - mb)), round(abs(na - nb) / geometry.nfp), sound),
                        harmonics=[[int(ma), int(na)], [int(mb), int(nb)]],
                        family=fi,
                    )
                )
    widest: dict = {}  # one gap per crossing pair: the relatively widest of its avoided crossings
    for g in sorted(gaps, key=lambda g: (g["type"] == "BAE", -g["width_khz"] / g["center_khz"])):
        widest.setdefault((g["family"], *sorted(map(tuple, g["harmonics"]))), g)
    gaps = list(widest.values())[:max_gaps]
    mode, reasons = None, []
    for gi, g in [(i, g) for i, g in enumerate(gaps) if g["type"] != "BAE"]:  # the eigenmode has no sound coupling
        fam = families[g["family"]]
        hf, f = np.asarray(fam["harmonics"], dtype=float), fam["shear_alfven_khz"]
        i0 = int(np.argmin(abs(s - g["s"])))
        em = eigenmodes(geometry, hf, jnp.asarray(rho), target_hz=1e3 * g["center_khz"], count=count, iters=iters, tol=tol, **grid)
        freq = np.asarray(em.frequencies_hz) / 1e3
        why = "outside the band"
        for j in np.argsort(abs(freq - g["center_khz"])):
            if not g["low_khz"] < freq[j] < g["high_khz"]:
                continue
            xj = np.asarray(em.structures)[j]
            p = (np.abs(xj) ** 2).sum(-1)
            support = p > 0.1 * p.max()
            peak = s[p.argmax()]
            below = (f < freq[j]).sum(1)  # shear-Alfven branches below the mode: constant unless it crosses the continuum
            if support.sum() < 3:
                why = "resolved on fewer than three surfaces"
                continue
            if (np.abs(np.diff(xj, 2, axis=0)) ** 2).sum() > 0.5 * p.sum():
                why = "oscillating on the radial grid"
                continue
            pair = [int(np.flatnonzero((hf == q).all(1))[0]) for q in np.asarray(g["harmonics"], dtype=float)]
            if np.abs(xj[:, pair]).max(0).min() < 0.2 * np.abs(xj).max():
                why = "not carried by both harmonics of the gap"
                continue
            if not g["s_range"][0] - 0.1 <= peak <= g["s_range"][1] + 0.1:
                why = "peaked away from the gap"
                continue
            if not edge < peak < 1 - edge:
                why = "pinned to the axis or the boundary"
                continue
            if (below[p > 1e-2 * p.max()] != below[i0]).any():  # tails too: a mode cut off at a crossing is continuum-damped
                why = "crossing the continuum"
                continue
            big = xj.ravel()[np.abs(xj).argmax()]
            c = xj * np.conj(big) / abs(big)
            theta = np.linspace(0, 2 * np.pi, 129)
            ang = theta[:, None] * xm[None]
            R = np.asarray(geometry.rmnc) @ np.cos(ang).T + np.asarray(geometry.rmns) @ np.sin(ang).T
            Z = np.asarray(geometry.zmns) @ np.sin(ang).T + np.asarray(geometry.zmnc) @ np.cos(ang).T
            phi = np.real(c @ np.exp(1j * hf[:, 0][:, None] * theta[None]))
            mode = dict(
                frequency_khz=float(freq[j]),
                peak_s=float(peak),
                support=[float(s[support].min()), float(s[support].max())],
                family=g["family"],
                gap=gi,
                in_gap=True,
                converged=bool(em.status.converged),
                residual=float(em.status.residual),
                structure=np.real(c),
                harmonics=hf.astype(int).tolist(),
                cross_section=dict(R=R, Z=Z, phi=phi / np.abs(phi).max()),
            )
            break
        if mode:
            break
        reasons.append(why)
    tally = ", ".join(f"{reasons.count(r)} {r}" for r in dict.fromkeys(reasons))
    reason = "" if mode else (f"{len(reasons)} gaps searched, the nearest of {count} modes: {tally}" if reasons else "no gap in the window")
    n_i, amu = ions if ions else (None, None)
    pick = mode["family"] if mode else (gaps[0]["family"] if gaps else 0)
    return dict(
        harmonics=h.astype(int).tolist(),
        density_kg_m3=rho.tolist(),
        density_m3=n_i,
        ion_mass_amu=amu,
        sound=pressure is not None,
        alfven_khz=float(f_alfven),
        window_khz=window,
        gaps=gaps,
        mode=mode,
        mode_reason=reason,
        family=pick,
        profiles=dict(s=s, iota=iota, families=families),
    )
