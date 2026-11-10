"""Optimize a VMEC boundary for its Alfvén spectrum while keeping quasisymmetry.

Two cases, both ``scipy.optimize.least_squares`` on a few boundary coefficients,
with an exact Jacobian (one vmex + FAX forward, one adjoint per residual row):

* ``qh``: Paul et al. (2025) eq. 7.1 on the Landreman-Paul QH deck,
  ``(A - A*)^2 + f_QS + f_iota + f_cont`` with ``f_cont`` = :func:`fax.continuum_cost`.
* ``qa``: Landreman-Paul QA with two global modes pushed out of a frequency window
  (:func:`fax.window_risk`) at fixed QS and aspect ratio. ``--modes gap`` (default) takes the two n = 1 modes nearest
  103 kHz, which :func:`fax.classify_modes` finds to be isolated gap modes; ``--modes continuum`` takes the lowest
  n = 0 modes, which it finds to be continuum-like (each sits on the continuum minimum at its own radius).
  The modes are classified before and after, and the first-order prediction ``J dp`` is compared with the re-solved
  change. With ``--drive`` instead the fast-ion growth rates of the four n = 1 modes nearest 104 kHz are lowered
  (:func:`fax.drive_risk` of :func:`fax.geometry_growth_rates`, 150 keV Maxwellian deuterons).

Run: python examples/optimize_alfven.py, after editing the settings below (needs ``pip install "alfvex[vmex,plot]"``;
a few minutes per case on a laptop CPU).
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares
from vmex.core import implicit
from vmex.core.optimize import QuasisymmetryRatioResidual

import fax

# settings
CASE = "both"  # "qh", "qa" or "both"
MODES_KIND = "gap"  # QA window: "gap" (the n = 1 gap modes) or "continuum" (the n = 0 continuum minima)
DRIVE = False  # QA: lower the fast-ion growth rates instead of moving modes out of the window
NS, NODES, NFEV = 16, 24, 16  # VMEC surfaces, FE nodes, least-squares evaluations
OUT = Path(".")  # figure folder

DATA = Path(__file__).parent / "data"
CASES = {
    "qh": dict(
        deck="input.LandremanPaul2021_QH_reactorScale_lowres",
        helicity=(1, -1),
        iota_floor=1.03,
        parameters=tuple(f"{x}{m}{n}" for m in range(4) for n in range(-2, 3) if (m, n) > (0, 0) for x in ("rc", "zs")),
        qs_weight=0.1,
        harmonics=fax.mode_family((1, -3, 5, -7), 6, -6),
    ),
    "qa": dict(
        deck="input.LandremanPaul2021_QA_lowres",
        helicity=(1, 0),
        iota_floor=0.38,
        parameters=("rc11", "zs11", "rc12", "zs12"),
        qs_weight=1.0,
    ),
}
MODES = {  # the global modes of QA: the n = 1 gap modes (102-104 kHz in the 96-142 kHz gap) or the n = 0 continuum minima
    "gap": dict(harmonics=fax.mode_family((1, -1, 3), 5), ntheta=16, nzeta=16, target_hz=103e3, count=2),
    "continuum": dict(harmonics=fax.mode_family(0, 3, 1), ntheta=8, nzeta=8),
}
DENSITY = {"qh": fax.mass_density(1e20), "qa": 3.4e-8}  # QA: the P3 test density (a few hundred kHz on the 1 m device)
DRIVE_MODES = dict(harmonics=fax.mode_family((1, -1, 3), 5), density=DENSITY["qa"], ntheta=16, nzeta=16, target_hz=104e3, count=4)


def setup(case, ns, nodes, drive=False, modes="gap"):
    c = CASES[case]
    eig = dict(density=DENSITY["qa"], **MODES[modes]) if case == "qa" else {}
    eig = DRIVE_MODES if drive else eig
    ions = fax.FastIons(density=1e15 * (1.0 - jnp.arange(1, nodes + 1) / (nodes + 1)) ** 2, temperature_ev=150e3)
    design = fax.prepare_design(
        DATA / c["deck"],
        parameters=c["parameters"],
        ns=ns,
        nodes=nodes,
        mboz=6,
        nboz=6,
        **{"harmonics": [[1.0, 0.0]], "density": 1.0, **eig},
    )
    qs = QuasisymmetryRatioResidual(np.linspace(0.1, 1.0, 10), *c["helicity"])

    def terms(p):  # every scalar the objective and the report need
        state, rt = fax.design_state(p, design)
        geo = fax.state_geometry(state, rt, design)
        out = dict(
            aspect=implicit.aspect_ratio(state, rt),
            qs=qs.total_state(state, rt),
            iota=jnp.mean(geo.iota),
            cont=fax.continuum_cost(geo, c["helicity"]),
        )
        out["iota_pen"] = jnp.sum(jnp.minimum(jnp.abs(geo.iota) - c["iota_floor"], 0.0) ** 2)
        if case == "qa":
            out["freq"] = fax.geometry_frequencies(geo, design, 4 if drive else 2 if modes == "gap" else 3)[0]
        if drive:
            out["gamma"] = fax.geometry_growth_rates(geo, design, ions, 4, ntheta=32, nzeta=32)
        return out

    return design, terms


def softness(t0):
    """0.34% of the mode frequency: 1 kHz at the 294 kHz continuum minima, 0.35 kHz at the 103 kHz gap modes."""
    return 3.4e-3 * float(np.mean(t0["freq"]))


def residual_fn(case, t0, window):
    """Scale every row to 1 at the start: ``(A - A*)/A*``, QS, iota, and the Alfvén term."""

    def rows(t):
        if "gamma" in t:
            alfven = 10.0 * fax.drive_risk(t["gamma"]) / fax.drive_risk(t0["gamma"])
        elif case == "qh":
            alfven = jnp.sqrt(t["cont"] / t0["cont"])
        else:
            alfven = 30.0 * jnp.sqrt(fax.window_risk(t["freq"], window, softness(t0)))
        return jnp.stack(
            [
                10.0 * (t["aspect"] / t0["aspect"] - 1.0),
                CASES[case]["qs_weight"] * jnp.sqrt(t["qs"] / t0["qs"]),
                10.0 * jnp.sqrt(t["iota_pen"] + 1e-30),
                alfven,
            ]
        )

    return rows


def run(case, ns, nodes, nfev, drive=False, modes="gap"):
    design, terms = setup(case, ns, nodes, drive, modes)
    p0 = jnp.zeros(len(design.names))
    t0 = jax.tree_util.tree_map(np.asarray, terms(p0))
    window = None
    if case == "qa" and not drive:
        f0 = np.asarray(t0["freq"])
        window = (float(f0.min()) - softness(t0), float(f0.max()) + 20 * softness(t0))  # modes sit just inside the lower edge
    rows = residual_fn(case, t0, window)
    history, cache = [], {}

    def evaluate(p):  # least_squares calls fun then jac at the same x: one forward, its vjp reused for the rows
        key = np.asarray(p).tobytes()
        if key not in cache:
            (r, t), vjp = jax.vjp(lambda q: (lambda t: (rows(t), t))(terms(q)), jnp.asarray(p))
            cache.clear()
            cache[key] = r, vjp, t
            history.append(dict(cost=0.5 * float(jnp.sum(r**2)), **{k: np.asarray(v).tolist() for k, v in t.items()}))
            print(
                f"  eval {len(history):2d}: cost {history[-1]['cost']:.4e}  QS {history[-1]['qs']:.3e}  f_cont {history[-1]['cont']:.4f}"
                + (f"  gamma_max {max(history[-1]['gamma']):.0f}" if "gamma" in t else ""),
                flush=True,
            )
        return cache[key]

    def jac(p):
        r, vjp, t = evaluate(p)
        zero = jax.tree_util.tree_map(jnp.zeros_like, t)
        return np.stack([np.asarray(vjp((e, zero))[0]) for e in jnp.eye(r.size)])

    step = 2e-3 * float(design._cache["params"].rbc[int(design._cache["config"].inp.ntor), 1])  # 0.2% of the minor radius
    result = least_squares(
        lambda p: np.asarray(evaluate(p)[0]),
        np.zeros(p0.size),
        jac=jac,
        x_scale=step,
        max_nfev=nfev,
        ftol=1e-4,
        xtol=1e-8,
        bounds=(-10 * step, 10 * step),
    )
    t1 = jax.tree_util.tree_map(np.asarray, terms(jnp.asarray(result.x)))
    return design, p0, jnp.asarray(result.x), t0, t1, history, window


def gap_table(geo, helicity, count=6):
    """Paul's perturbative gaps of the largest f_cont terms: ``(dm, n, w0 / w_A, full width / w_A)``, surface-averaged.

    ``w0 = |iota dm - n| / 2`` is the gap centre and ``2 w0 |c_{dm,n}|`` its full (edge to edge) width (Paul et al. 2025, eqs. 4.5, 4.16).
    """
    c, dm, n = (np.asarray(x) for x in fax.compression_harmonics(geo))
    w0 = 0.5 * np.abs(np.asarray(geo.iota)[:, None, None] * dm[:, None] - n)
    cut = 0.5 * np.abs(np.asarray(geo.iota) * helicity[0] - helicity[1] * geo.nfp)[:, None, None]
    width = np.where(w0 > cut, 2 * w0 * np.abs(c), 0.0).mean(0)
    keep = [(i, j) for i, j in zip(*np.unravel_index(np.argsort(-width.ravel()), width.shape), strict=True) if (dm[i], n[j]) > (0, 0)]
    return [(int(dm[i]), int(n[j]), w0[:, i, j].mean(), width[i, j]) for i, j in keep[:count]]


def report(case, design, p0, p1, t0, t1, history, window, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = CASES[case]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), constrained_layout=True)
    s = np.linspace(0.05, 0.95, 40)
    labels = []
    rho = DENSITY[case] * (1.0 - 0.5 * s) if case == "qh" else DENSITY[case]  # QH: a peaked profile, so branches sweep
    for ax, p, t, label in ((axes[0], p0, t0, "initial"), (axes[1], p1, t1, "optimized")):
        geo = fax.design_geometry(p, design).interpolate(s)
        f = np.asarray(fax.continuum(geo, np.asarray(design.harmonics) if case == "qa" else c["harmonics"], rho, 32, 64))
        ax.plot(s, f / 1e3, ".", ms=1.5, color="0.3")
        if case == "qa" and window is None:
            top = 2.0 * max(t["freq"])
            for fk in t["freq"]:
                ax.axhline(fk / 1e3, color="C3", lw=0.8)
        elif case == "qa":
            top = 2.0 * window[1]
            ax.axhspan(window[0] / 1e3, window[1] / 1e3, color="C3", alpha=0.2)
            for fk in t["freq"]:
                ax.axhline(fk / 1e3, color="C3", lw=0.8)
        else:  # Paul's cut w / w_A0 = |iota M - N N_P| / 2, with w_A0 = v_A0 / R0 on axis
            vA0 = float(geo.bmnc[0, 0]) / np.sqrt(4e-7 * np.pi * float(DENSITY[case]))
            cut = abs(float(t["iota"]) * c["helicity"][0] - c["helicity"][1] * geo.nfp) * vA0 / (4 * np.pi * float(geo.rmnc[0, 0]))
            top = 2.0 * cut
            ax.axhline(cut / 1e3, color="C3", ls="--", lw=1)
            table = gap_table(geo, c["helicity"], 40)
            rows = "; ".join(f"({a},{b}) {w:.3f} {d:.4f}" for a, b, w, d in table[:6])
            labels = labels or [f"({a},{b})" for a, b, _, _ in table[:6]]  # the initial top six
            widths = {f"({a},{b})": d for a, b, _, d in table}
            axes[3].bar(np.arange(6) + (0.2 if p is p1 else -0.2), [widths.get(k, 0.0) for k in labels], 0.4, label=label)
            print(f"  {label} gaps above the cut, (dm,n) w0/w_A width/w_A: {rows}")
        ax.set(title=f"{case.upper()} {label}", xlabel="s", ylabel="frequency (kHz)", ylim=(0, top / 1e3))
    cost = [h["cost"] for h in history]
    axes[2].semilogy(cost, "o-", label="total")
    axes[2].semilogy([h["qs"] / history[0]["qs"] for h in history], "s-", label="QS / QS$_0$")
    axes[2].semilogy([h["cont"] / history[0]["cont"] for h in history], "^-", label="$f_{cont}$ / $f_{cont,0}$")
    axes[2].set(xlabel="evaluation", title="objective history")
    axes[2].legend(fontsize=8)
    if case == "qh":
        axes[3].set(xticks=range(6), xticklabels=labels, title="gap width / $\\omega_A$ above the cut", xlabel="($\\delta m$, n)")
        axes[3].legend(fontsize=8)
    elif window is None:
        axes[3].bar(np.arange(4) - 0.2, np.asarray(t0["gamma"]) / 1e3, 0.4, label="initial")
        axes[3].bar(np.arange(4) + 0.2, np.asarray(t1["gamma"]) / 1e3, 0.4, label="optimized")
        axes[3].set(xlabel="mode", ylabel="$\\gamma$ (10$^3$ s$^{-1}$)", title="fast-ion growth rates")
        axes[3].legend(fontsize=8)
    else:
        axes[3].bar(["initial", "optimized"], [min(t0["freq"]) / 1e3, min(t1["freq"]) / 1e3], color="0.5")
        axes[3].axhspan(window[0] / 1e3, window[1] / 1e3, color="C3", alpha=0.2)
        axes[3].set(ylim=(window[0] / 1e3 - 6, window[1] / 1e3), ylabel="lowest global mode (kHz)", title="mode vs window")
    path = out / f"optimize_alfven_{case}{'_drive' if window is None and case == 'qa' else ''}.png"
    fig.savefig(path, dpi=90)
    plt.close(fig)
    for k in ("aspect", "iota", "qs", "cont", *(("freq",) if case == "qa" else ()), *(("gamma",) if "gamma" in t0 else ())):
        print(f"  {k:7s} {np.round(np.asarray(t0[k]), 6)} -> {np.round(np.asarray(t1[k]), 6)}")
    print(f"  offsets {np.round(np.asarray(p1), 5)} ({', '.join(design.names)}); wrote {path}")
    if case == "qa":
        classify(design, p0, p1, t0["freq"], t1["freq"])


def classify(design, p0, p1, f0, f1):
    """Are the selected modes gap modes? Class of each mode at both ends, and the first-order contract ``J dp`` against the re-solve."""
    for label, p in (("initial", p0), ("optimized", p1)):
        geo = fax.design_geometry(p, design)
        r = fax.classify_modes(geo, design.harmonics, design.density, nodes=design.nodes, **design.options)
        print(f"  {label} modes: f (kHz), peak s, shift, overlap, margin, localization, isolated")
        for row in zip(*r, strict=True):
            print(f"    {row[0] / 1e3:8.3f} {row[1]:6.3f} {row[2]:9.1e} {row[3]:6.3f} {row[4]:6.3f} {row[5]:6.2f}  {bool(row[6])}")
    jac = jax.jacrev(lambda q: fax.design_frequencies(q, design, len(f0))[0])(p0)
    predicted, actual = np.asarray(jac @ (p1 - p0)), np.asarray(f1) - np.asarray(f0)
    k = lambda a: np.round(a / 1e3, 4)  # noqa: E731
    print(f"  first order: J dp = {k(predicted)} kHz, re-solved df = {k(actual)} kHz, remainder {k(actual - predicted)} kHz")


for case in ("qh", "qa") if CASE == "both" else (CASE,):
    start = time.perf_counter()
    print(f"{case.upper()}:")
    report(case, *run(case, NS, NODES, NFEV, DRIVE and case == "qa", MODES_KIND), OUT)
    print(f"  {time.perf_counter() - start:.0f} s")
