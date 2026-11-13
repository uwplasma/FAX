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

from .continuum import continuum, continuum_matrices, generalized_eigh
from .drive import FastIons, growth_rate
from .eigen import (
    Status,
    _dense_status,
    _hz,
    dense,
    eigenmodes,
    eigenvalue_mean,
    global_matrices,
    selected,
    split_solver,
    targeted_eigenpairs,
)
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
    return state_geometry(*design_state(offsets, design), design)


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
    F, G = global_matrices(_nodes(geo, design.nodes), design.harmonics, design.density, **opts)
    if target is None:
        K, M = dense(F), dense(G)
        lam, x = jax.lax.stop_gradient(generalized_eigh(K, M))
        status = _dense_status(K, M, lam, x, solver.get("tol", 1e-10))._replace(gap=lam[mode_count] - lam[mode_count - 1])
        omega2 = jnp.stack([eigenvalue_mean(K, M, i, i + 1) for i in range(mode_count)])
    else:
        lam, _, status = targeted_eigenpairs(F, G, (2 * jnp.pi * target) ** 2, **solver)
        omega2 = lam[:mode_count]
    return _reject(_hz(omega2), status, lam), status


def design_frequencies(offsets: Any, design: Design, mode_count: int = 3) -> tuple[jax.Array, Status]:
    """``boundary offsets -> vmex -> booz_xform_jax -> FE nodes -> (frequencies in Hz, Status)``, differentiable (first order)."""
    return geometry_frequencies(design_geometry(offsets, design), design, mode_count)


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


def _gap_type(dm: int, dn: int) -> str:
    """Name of a gap from the poloidal and toroidal (in field periods) mode-number differences of the coupled harmonics."""
    if dn == 0:
        return {1: "TAE", 2: "EAE"}.get(dm, "other")
    return "MAE" if dm == 0 else "HAE"


def summary(
    geometry: Geometry,
    harmonics: Any,
    mass_density: Any,
    *,
    window: tuple[float, float] | None = None,
    count: int = 4,
    ions: tuple[float, float] | None = None,
    max_gaps: int = 6,
    min_width: float = 0.02,
    ntheta: int = 32,
    nzeta: int = 32,
    iters: int = 100,
    tol: float = 1e-10,
) -> dict[str, Any]:
    """Global continuum gaps and the global mode inside the widest one, as one plain dictionary (host arrays, not differentiable).

    ``mass_density`` is the ion mass density in kg/m^3, as for :func:`fax.continuum` (see :func:`fax.mass_density`); frequencies
    scale as its inverse square root. ``ions = (n_i, amu)`` only records the number density in m^-3 and the ion mass behind it.
    ``harmonics`` are ``(m, n)`` pairs, ``n`` a multiple of ``nfp`` (see :func:`fax.mode_family`). ``window = (low, high)`` in kHz
    keeps the gaps whose centre lies inside.

    A **gap** is a global one: for adjacent sorted branches ``k`` and ``k + 1`` it is the band ``[max_s f_k, min_s f_{k+1}]``,
    kept if it is non-empty and wider than ``min_width`` of its centre, so that no branch crosses it at any surface. The
    widest ``max_gaps`` are returned, widest first. Each is typed from the two harmonics that dominate the branch eigenvectors
    on the surface where the branches are closest: TAE (``dm = 1``), EAE (``dm = 2``), MAE (``dm = 0``, ``dn != 0``), HAE
    (both nonzero), ``other`` otherwise. The ``count`` global modes nearest the centre of the widest gap are solved
    (:func:`fax.eigenmodes`; ``iters``, ``tol``, ``ntheta``, ``nzeta``) and the one inside the band nearest its centre is the
    ``mode``. Without one, ``mode`` is ``None`` and ``mode_reason`` says why.

    Returns ``harmonics``, ``density_kg_m3``, ``density_m3`` and ``ion_mass_amu`` (the last two ``None`` without ``ions``),
    ``window_khz``, ``gaps`` (``center_khz``, ``width_khz``, ``low_khz``, ``high_khz``, ``s`` and ``iota`` of the closest
    approach, ``type``, ``harmonics``), ``mode`` (``frequency_khz``, ``peak_s``, ``in_gap``, ``converged``, ``residual``,
    ``structure`` of shape ``(surfaces, harmonics)``), ``mode_reason`` and ``profiles`` (``s``, ``iota``, ``continuum_khz``),
    which :func:`fax.plot.plot_summary` draws.
    """
    h = np.asarray(harmonics, dtype=float)
    rho = jnp.asarray(mass_density)
    grid = {"ntheta": ntheta, "nzeta": nzeta}
    f = np.asarray(continuum(geometry, h, rho, **grid)) / 1e3
    s, iota = np.asarray(geometry.s), np.asarray(geometry.iota)
    low, high = f[:, :-1].max(0), f[:, 1:].min(0)
    found = []
    for k in np.flatnonzero(high - low > 0):
        center = 0.5 * (low[k] + high[k])
        if high[k] - low[k] >= min_width * center and (window is None or window[0] <= center <= window[1]):
            found.append((high[k] - low[k], k, center))
    found = sorted(found, reverse=True)[:max_gaps]
    gaps = []
    for w, k, center in found:
        i = int(np.argmin(f[:, k + 1] - f[:, k]))  # the surface where the two branches approach
        near = np.arange(3) + min(max(i - 1, 0), len(s) - 3)  # surface_grid needs three surfaces
        K, M = continuum_matrices(geometry.surfaces(near), h, rho, **grid)
        x = np.asarray(generalized_eigh(K, M)[1])[i - near[0], :, k : k + 2]
        a, b = np.argsort(-(np.abs(x) ** 2).sum(1))[:2]
        dm, dn = abs(h[a, 0] - h[b, 0]), round(abs(h[a, 1] - h[b, 1]) / geometry.nfp)
        gaps.append(
            dict(
                center_khz=float(center),
                width_khz=float(w),
                low_khz=float(low[k]),
                high_khz=float(high[k]),
                s=float(s[i]),
                iota=float(iota[i]),
                type=_gap_type(int(dm), dn),
                harmonics=h[[a, b]].astype(int).tolist(),
            )
        )
    mode, reason = None, "no global gap in the window"
    if gaps:
        g = gaps[0]
        em = eigenmodes(geometry, h, rho, target_hz=1e3 * g["center_khz"], count=count, iters=iters, tol=tol, **grid)
        freq = np.asarray(em.frequencies_hz) / 1e3
        inside = np.flatnonzero((g["low_khz"] < freq) & (freq < g["high_khz"]))
        if inside.size:
            j = inside[np.argmin(abs(freq[inside] - g["center_khz"]))]
            power = (np.abs(np.asarray(em.structures)[j]) ** 2).sum(-1)
            mode = dict(
                frequency_khz=float(freq[j]),
                peak_s=float(s[power.argmax()]),
                in_gap=True,
                converged=bool(em.status.converged),
                residual=float(em.status.residual),
                structure=np.abs(np.asarray(em.structures)[j]),
            )
            reason = ""
        else:
            reason = f"none of the {count} modes nearest {g['center_khz']:.1f} kHz lies inside the widest gap"
    n_i, amu = ions if ions else (None, None)
    return dict(
        harmonics=h.astype(int).tolist(),
        density_kg_m3=float(rho),
        density_m3=n_i,
        ion_mass_amu=amu,
        window_khz=window,
        gaps=gaps,
        mode=mode,
        mode_reason=reason,
        profiles=dict(s=s, iota=iota, continuum_khz=f),
    )
