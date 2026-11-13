"""Optimize a VMEC boundary for its Alfven spectrum at fixed quasisymmetry, with an exact Jacobian.

``scipy.optimize.least_squares`` on a few boundary coefficients; one VMEX + FAX forward and one adjoint per residual row. Runs:

* qh (RESEARCH): Paul et al., J. Plasma Phys. 91, E101 (2025), eq. 7.1 on the Landreman-Paul QH deck,
  ``(A - A*)^2 + f_QS + f_iota + f_cont`` with ``f_cont`` = continuum_cost (compression_harmonics gives the gap table);
* qa, gap: Landreman-Paul QA (PRL 128, 035001 (2022)) with the two n = 1 modes nearest 103 kHz pushed out of a window
  (window_risk) at fixed QS and aspect ratio; classify_modes shows they are isolated gap modes and the first-order prediction
  ``J dp`` is compared with the re-solved change (RESEARCH adds the n = 0 continuum minima, which classify_modes finds continuum-like,
  and the fast-ion growth-rate objective: geometry_growth_rates and drive_risk, 150 keV Maxwellian deuterons).

Needs the vmex and plot extras (Python 3.11 or newer). Writes optimize_alfven_<run>.png. The fast path is the QA gap-mode run with
3 evaluations on a 12 x 12 angle grid (about 2 min, not converged); RESEARCH adds the other three cases and runs every case with 16
evaluations on a 16 x 16 grid (about 8 min each).
"""

# ruff: noqa: B023  (the closures below run within the loop pass that defines them)
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from vmex.core import implicit
from vmex.core.optimize import QuasisymmetryRatioResidual

import fax

RESEARCH = False  # True: 16 evaluations on a finer angle grid, the QH case and two more QA objectives (continuum minima, fast-ion drive)
RUNS = [("qa", "gap", False)] + [("qh", "gap", False), ("qa", "continuum", False), ("qa", "gap", True)] * RESEARCH  # case, modes, drive
NS, NODES, GRID, NFEV = (16, 24, 16, 16) if RESEARCH else (16, 24, 12, 3)  # VMEC surfaces, FE nodes, angle grid, least-squares evaluations
OUT = Path(".")  # figure folder

DATA = Path(__file__).resolve().parent / "data"
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
MODES = {  # QA global modes: the n = 1 gap modes (102-104 kHz in the 96-142 kHz gap) or the n = 0 continuum minima
    "gap": dict(harmonics=fax.mode_family((1, -1, 3), 5), ntheta=GRID, nzeta=GRID, target_hz=103e3, count=2),
    "continuum": dict(harmonics=fax.mode_family(0, 3, 1), ntheta=8, nzeta=8),
}
DRIVE_MODES = dict(harmonics=fax.mode_family((1, -1, 3), 5), ntheta=GRID, nzeta=GRID, target_hz=104e3, count=4)  # four n = 1 modes
DENSITY = {"qh": fax.mass_density(1e20), "qa": 3.4e-8}  # QA: a few hundred kHz on the 1 m device
FAST_ION = dict(temperature_ev=150e3, density_m3=1e15)  # drive run: Maxwellian deuterons, n_f = 1e15 (1 - s)^2 m^-3


def classify_report(design, p0, p1, f0, f1):
    """Class of each selected mode at both ends and the first-order contract ``J dp`` against the re-solve."""
    for label, p in (("initial", p0), ("optimized", p1)):
        r = fax.classify_modes(fax.design_geometry(p, design), design.harmonics, design.density, nodes=design.nodes, **design.options)
        print(f"  {label} modes: f (kHz), peak s, shift, overlap, margin, localization, isolated")
        for row in zip(*r, strict=True):
            print(f"    {row[0] / 1e3:8.3f} {row[1]:6.3f} {row[2]:9.1e} {row[3]:6.3f} {row[4]:6.3f} {row[5]:6.2f}  {bool(row[6])}")
    _, vjp = jax.vjp(lambda q: fax.design_frequencies(q, design, len(f0))[0], p0)
    predicted, actual = np.stack([vjp(e)[0] for e in jnp.eye(len(f0))]) @ np.asarray(p1 - p0), np.asarray(f1) - np.asarray(f0)
    print(f"  first order: J dp = {np.round(predicted / 1e3, 4)} kHz, re-solved df = {np.round(actual / 1e3, 4)} kHz")


for case, modes_kind, drive in RUNS:
    start, c = time.perf_counter(), CASES[case]
    name = f"{case}{'_drive' if drive else '_continuum' if modes_kind == 'continuum' else ''}"
    print(f"{name.upper()}:")
    eig = DRIVE_MODES if drive else MODES[modes_kind] if case == "qa" else {}
    n_modes = 4 if drive else 2 if modes_kind == "gap" else 3
    ions = fax.FastIons(
        density=FAST_ION["density_m3"] * (1.0 - jnp.arange(1, NODES + 1) / (NODES + 1)) ** 2, temperature_ev=FAST_ION["temperature_ev"]
    )
    design = fax.prepare_design(
        DATA / c["deck"],
        parameters=c["parameters"],
        ns=NS,
        nodes=NODES,
        mboz=6,
        nboz=6,
        **{"harmonics": [[1.0, 0.0]], "density": 1.0, **({"density": DENSITY["qa"], **eig} if eig else {})},
    )
    qs = QuasisymmetryRatioResidual(np.linspace(0.1, 1.0, 10), *c["helicity"])

    def terms(p):
        """Every scalar the objective and the report need."""
        state, rt = fax.design_state(p, design)
        geo = fax.state_geometry(state, rt, design)
        out = dict(
            aspect=implicit.aspect_ratio(state, rt),
            qs=qs.total_state(state, rt),
            iota=jnp.mean(geo.iota),
            cont=fax.continuum_cost(geo, c["helicity"]),
            iota_pen=jnp.sum(jnp.minimum(jnp.abs(geo.iota) - c["iota_floor"], 0.0) ** 2),
        )
        if case == "qa":
            out["freq"] = fax.geometry_frequencies(geo, design, n_modes)[0]
        if drive:
            out["gamma"] = fax.geometry_growth_rates(geo, design, ions, 4, ntheta=32, nzeta=32)
        return out

    p0 = jnp.zeros(len(design.names))
    t0 = jax.tree_util.tree_map(np.asarray, terms(p0))
    soft = 3.4e-3 * float(np.mean(t0["freq"])) if case == "qa" else None  # 0.34% of the mode frequency
    window = None
    if case == "qa" and not drive:
        f0 = np.asarray(t0["freq"])
        window = (float(f0.min()) - soft, float(f0.max()) + 20 * soft)  # modes sit just inside the lower edge

    def rows(t):
        """Each row is 1 at the start: ``(A - A*)/A*``, QS, iota, and the Alfven term."""
        if drive:
            alfven = 10.0 * fax.drive_risk(t["gamma"]) / fax.drive_risk(t0["gamma"])
        elif case == "qh":
            alfven = jnp.sqrt(t["cont"] / t0["cont"])
        else:
            alfven = 30.0 * jnp.sqrt(fax.window_risk(t["freq"], window, soft))
        return jnp.stack(
            [
                10.0 * (t["aspect"] / t0["aspect"] - 1.0),
                c["qs_weight"] * jnp.sqrt(t["qs"] / t0["qs"]),
                10.0 * jnp.sqrt(t["iota_pen"] + 1e-30),
                alfven,
            ]
        )

    history, cache = [], {}

    def evaluate(p):
        """One forward, its vjp reused for the Jacobian rows (``least_squares`` calls fun then jac at the same x)."""
        key = np.asarray(p).tobytes()
        if key not in cache:
            (r, t), vjp = jax.vjp(lambda q: (lambda t: (rows(t), t))(terms(q)), jnp.asarray(p))
            cache.clear()
            cache[key] = r, vjp, t
            history.append(dict(cost=0.5 * float(jnp.sum(r**2)), **{k: np.asarray(v).tolist() for k, v in t.items()}))
            print(
                f"  eval {len(history):2d}: cost {history[-1]['cost']:.4e}  QS {history[-1]['qs']:.3e}  f_cont {history[-1]['cont']:.4f}",
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
        max_nfev=NFEV,
        ftol=1e-4,
        xtol=1e-8,
        bounds=(-10 * step, 10 * step),
    )
    p1 = jnp.asarray(result.x)
    t1 = jax.tree_util.tree_map(np.asarray, terms(p1))
    for k in ("aspect", "iota", "qs", "cont", *(("freq",) if case == "qa" else ()), *(("gamma",) if drive else ())):
        print(f"  {k:7s} {np.round(np.asarray(t0[k]), 6)} -> {np.round(np.asarray(t1[k]), 6)}")
    print(f"  offsets {np.round(np.asarray(p1), 5)} ({', '.join(design.names)})")
    if case == "qa":
        classify_report(design, p0, p1, t0["freq"], t1["freq"])

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), constrained_layout=True)
    s = np.linspace(0.05, 0.95, 40)
    labels = []
    rho = DENSITY[case] * (1.0 - 0.5 * s) if case == "qh" else DENSITY[case]  # QH: a peaked profile, so branches sweep
    for ax, p, t, label in ((axes[0], p0, t0, "initial"), (axes[1], p1, t1, "optimized")):
        geo = fax.design_geometry(p, design).interpolate(s)
        f = np.asarray(fax.continuum(geo, np.asarray(design.harmonics) if case == "qa" else c["harmonics"], rho, 32, 64))
        ax.plot(s, f / 1e3, ".", ms=1.5, color="0.3")
        if case == "qa":
            top = 2.0 * (window[1] if window else max(t["freq"]))
            if window:
                ax.axhspan(window[0] / 1e3, window[1] / 1e3, color="C3", alpha=0.2)
            for fk in t["freq"]:
                ax.axhline(fk / 1e3, color="C3", lw=0.8)
        else:  # Paul's cut w / w_A0 = |iota M - N N_P| / 2, with w_A0 = v_A0 / R0 on axis
            vA0 = float(geo.bmnc[0, 0]) / np.sqrt(4e-7 * np.pi * float(DENSITY[case]))
            cut = abs(float(t["iota"]) * c["helicity"][0] - c["helicity"][1] * geo.nfp) * vA0 / (4 * np.pi * float(geo.rmnc[0, 0]))
            top = 2.0 * cut
            ax.axhline(cut / 1e3, color="C3", ls="--", lw=1)
            cc, dm, n = (np.asarray(x) for x in fax.compression_harmonics(geo))  # Paul's perturbative gaps, eqs. 4.5, 4.16
            w0 = 0.5 * np.abs(np.asarray(geo.iota)[:, None, None] * dm[:, None] - n)
            gate = 0.5 * np.abs(np.asarray(geo.iota) * c["helicity"][0] - c["helicity"][1] * geo.nfp)[:, None, None]
            width = np.where(w0 > gate, 2 * w0 * np.abs(cc), 0.0).mean(0)  # full width / w_A, surface-averaged
            order = [
                (i, j) for i, j in zip(*np.unravel_index(np.argsort(-width.ravel()), width.shape), strict=True) if (dm[i], n[j]) > (0, 0)
            ]
            table = {f"({int(dm[i])},{int(n[j])})": width[i, j] for i, j in order[:40]}
            labels = labels or list(table)[:6]  # the initial top six
            axes[3].bar(np.arange(6) + (0.2 if p is p1 else -0.2), [table.get(k, 0.0) for k in labels], 0.4, label=label)
        ax.set(title=f"{name.upper()} {label}", xlabel="s", ylabel="frequency (kHz)", ylim=(0, top / 1e3))
    axes[2].semilogy([h["cost"] for h in history], "o-", label="total")
    axes[2].semilogy([h["qs"] / history[0]["qs"] for h in history], "s-", label="QS / QS$_0$")
    axes[2].semilogy([h["cont"] / history[0]["cont"] for h in history], "^-", label="$f_{cont}$ / $f_{cont,0}$")
    axes[2].set(xlabel="evaluation", title="objective history")
    axes[2].legend(fontsize=7)
    if case == "qh":
        axes[3].set(xticks=range(6), xticklabels=labels, title="gap width / $\\omega_A$ above the cut", xlabel="($\\delta m$, n)")
        axes[3].legend(fontsize=7)
    elif drive:
        axes[3].bar(np.arange(4) - 0.2, np.asarray(t0["gamma"]) / 1e3, 0.4, label="initial")
        axes[3].bar(np.arange(4) + 0.2, np.asarray(t1["gamma"]) / 1e3, 0.4, label="optimized")
        axes[3].set(xlabel="mode", ylabel="$\\gamma$ (10$^3$ s$^{-1}$)", title="fast-ion growth rates")
        axes[3].legend(fontsize=7)
    else:
        axes[3].bar(["initial", "optimized"], [min(t0["freq"]) / 1e3, min(t1["freq"]) / 1e3], color="0.5")
        axes[3].axhspan(window[0] / 1e3, window[1] / 1e3, color="C3", alpha=0.2)
        axes[3].set(ylim=(window[0] / 1e3 - 6, window[1] / 1e3), ylabel="lowest global mode (kHz)", title="mode vs window")
    fig.savefig(OUT / f"optimize_alfven_{name}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote optimize_alfven_{name}.png in {time.perf_counter() - start:.0f} s")
