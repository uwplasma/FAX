"""Figures for a results folder written by :func:`fax.io.run`."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_results(folder: str | Path) -> list[Path]:
    """Write ``continuum.png`` and, for eigenmode runs, ``modes.png``; return the paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = Path(folder)
    data = np.load(folder / "results.npz")
    branches = data["continuum_hz"] if "continuum_hz" in data else data["frequencies_hz"]
    fig, ax = plt.subplots(figsize=(5.0, 3.8), constrained_layout=True)
    ax.plot(data["s"], branches / 1e3, ".", ms=2, color="0.3")
    if "structures" in data:
        for f in data["frequencies_hz"]:
            ax.axhline(f / 1e3, lw=0.8, color="C3", alpha=0.6)
    ax.set(xlabel="s (normalized toroidal flux)", ylabel="frequency (kHz)", ylim=(0, np.percentile(branches, 98) / 1e3))
    paths = [folder / "continuum.png"]
    fig.savefig(paths[0], dpi=160)
    plt.close(fig)
    if "structures" in data:
        count = min(4, len(data["frequencies_hz"]))
        fig, axes = plt.subplots(1, count, figsize=(3.2 * count, 2.8), constrained_layout=True, squeeze=False)
        for k, ax in enumerate(axes[0]):
            for j, (m, n) in enumerate(data["harmonics"]):
                profile = np.abs(data["structures"][k, :, j])
                if profile.max() > 0.1 * np.abs(data["structures"][k]).max():
                    ax.plot(data["mode_s"], profile, label=f"({m:.0f},{n:.0f})")
            ax.set(title=f"{data['frequencies_hz'][k] / 1e3:.1f} kHz", xlabel="s")
            ax.legend(fontsize=7)
        paths.append(folder / "modes.png")
        fig.savefig(paths[1], dpi=160)
        plt.close(fig)
    return paths


def plot_summary(summary: dict, path: str | Path, *, colors: int = 64, dpi: int = 130) -> Path:
    """Draw the dictionary of :func:`fax.summary` as one small palette PNG at ``path`` and return the path.

    Three panels, as Alfven-eigenmode codes draw them: the continuum of one mode family against ``r/a = sqrt(s)``, each
    branch point coloured by its dominant ``m`` (sound branches grey) and the axis cut at 1.6 times the highest gap, with the local
    gaps shaded over their radial range and the mode drawn as a bar over its radial support (right axis: ``f / f_A``,
    ``f_A = v_A / 2 pi R_0``); the signed, phase-aligned radial harmonics of the mode; and the mode in the poloidal plane
    ``zeta = 0``. Without a mode the last two panels say why. ``colors`` is the palette size; ``dpi`` the resolution.
    """
    import io
    import textwrap

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    prof, mode, fa = summary["profiles"], summary["mode"], summary["alfven_khz"]
    fam = prof["families"][summary["family"]]
    r, f, dm = np.sqrt(prof["s"]), np.asarray(fam["continuum_khz"]), np.asarray(fam["dominant_m"])
    gaps = [g for g in summary["gaps"] if g["family"] == summary["family"]]
    top = max([g["high_khz"] for g in gaps] + ([mode["frequency_khz"]] if mode else []), default=0.0)
    ceiling = 1.6 * top if top > 0 else min(2.0 * fa, np.percentile(f, 90))
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(11.0, 3.6), constrained_layout=True)
    ns = sorted({n for _, n in fam["harmonics"]})
    shown = f < ceiling
    rr = np.broadcast_to(r[:, None], f.shape)
    alfven = shown & np.isfinite(dm)
    pts = a0.scatter(rr[alfven], f[alfven], c=dm[alfven], s=5, cmap="viridis", lw=0)
    a0.scatter(rr[shown & ~np.isfinite(dm)], f[shown & ~np.isfinite(dm)], s=2, color="0.75", lw=0)
    fig.colorbar(pts, ax=a0, label="dominant m", shrink=0.8)
    for g in gaps:
        x0, x1 = np.sqrt(g["s_range"])
        a0.fill_between([x0, x1], g["low_khz"], g["high_khz"], color="C1", alpha=0.35, lw=0)
        a0.annotate(g["type"], (x0, g["high_khz"]), xytext=(2, -1), textcoords="offset points", va="top", fontsize=6)
    if mode is not None and mode["family"] == summary["family"]:
        x0, x1 = np.sqrt(mode["support"])
        a0.plot([x0, x1], [mode["frequency_khz"]] * 2, color="C3", lw=2)
        a0.plot(np.sqrt(mode["peak_s"]), mode["frequency_khz"], "o", color="C3", ms=4)
    a0.set(xlabel="r/a", ylabel="frequency (kHz)", xlim=(0, 1), ylim=(0, ceiling), title=f"continuum, n = {', '.join(map(str, ns))}")
    a0.secondary_yaxis("right", functions=(lambda x: x / fa, lambda y: y * fa)).set_ylabel("f / f_A")
    if mode is None:
        for ax, text in ((a1, summary["mode_reason"]), (a2, "")):
            ax.text(0.5, 0.5, textwrap.fill(text, 34), ha="center", va="center", fontsize=8, transform=ax.transAxes)
            ax.set_axis_off()
        a1.set_title("no gap mode")
    else:
        c = mode["structure"]
        big = np.abs(c).max()
        for j in [j for j in range(c.shape[1]) if np.abs(c[:, j]).max() > 0.1 * big]:
            a1.plot(r, c[:, j] / big, label="({},{})".format(*mode["harmonics"][j]))
        a1.axhline(0, color="0.7", lw=0.6)
        g = summary["gaps"][mode["gap"]]
        flag = "" if mode["converged"] else ", unconverged"
        a1.set(xlabel="r/a", ylabel="\u03c6_mn (normalized)", xlim=(0, 1), title=f"{g['type']} {mode['frequency_khz']:.1f} kHz{flag}")
        a1.legend(fontsize=7, ncol=2)
        cs = mode["cross_section"]
        a2.pcolormesh(cs["R"], cs["Z"], cs["phi"], cmap="RdBu_r", vmin=-1, vmax=1, shading="gouraud")
        a2.plot(cs["R"][-1], cs["Z"][-1], color="k", lw=0.8)
        a2.set(xlabel="R (m)", ylabel="Z (m)", aspect="equal", title="\u03c6 at \u03b6 = 0")
    n_i, amu = summary["density_m3"], summary["ion_mass_amu"]
    rho = np.ravel(summary["density_kg_m3"])[0]
    plasma = f"n_i(0) = {np.ravel(n_i)[0]:.2g} m\u207b\u00b3, {amu:g} amu" if n_i is not None else f"rho(0) = {rho:.3g} kg/m\u00b3"
    fig.suptitle(f"{plasma}, f_A = {fa:.0f} kHz" + (", with sound coupling" if summary["sound"] else ""), fontsize=9)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    path = Path(path)
    Image.open(buf).convert("RGB").quantize(colors=colors, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)
    return path
