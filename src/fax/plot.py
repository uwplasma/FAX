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

    Three panels: the continuum against ``s`` with each global gap shaded as a frequency band, labelled with its type and width, and
    the mode marked at its peak radius; the rotational transform with a mark where each gap's branches are closest; the radial
    harmonics of the global mode (or the reason there is none). The title carries the density, and the ion mass when given.
    ``colors`` is the palette size (the PNG is about 20 KB); ``dpi`` sets the resolution.
    """
    import io
    import textwrap

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    prof, mode, gaps = summary["profiles"], summary["mode"], summary["gaps"]
    s, f = prof["s"], prof["continuum_khz"]
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(10.0, 3.0), constrained_layout=True)
    a0.plot(s, f, color="0.35", lw=0.8)
    for g in gaps:
        a0.axhspan(g["low_khz"], g["high_khz"], color="C1", alpha=0.35, lw=0)
        a0.annotate(f"{g['type']} {g['width_khz']:.1f} kHz", (s[-1], g["center_khz"]), ha="right", va="center", fontsize=7)
    if mode is not None:
        a0.plot(mode["peak_s"], mode["frequency_khz"], "o", color="C3", ms=5, label="global mode")
        a0.legend(fontsize=7, loc="upper left")
    a0.set(
        xlabel="s",
        ylabel="frequency (kHz)",
        xlim=(s[0], s[-1]),
        ylim=(0, 1.1 * np.percentile(f, 98)),
        title="continuum and gaps",
        box_aspect=1,
    )
    a1.plot(s, prof["iota"], color="C0")
    a1.plot([g["s"] for g in gaps], [g["iota"] for g in gaps], "s", color="C1", ms=5)
    for j, g in enumerate(gaps):
        a1.annotate(g["type"], (g["s"], g["iota"]), textcoords="offset points", xytext=(-26, 4 - 11 * j), fontsize=7)
    a1.set(xlabel="s", ylabel="\u03b9", title="rotational transform", box_aspect=1)
    if mode is None:
        a2.text(0.5, 0.5, textwrap.fill(summary["mode_reason"], 28), ha="center", va="center", fontsize=8, transform=a2.transAxes)
        a2.set(xlabel="s", title="no global mode", box_aspect=1)
    else:
        top = mode["structure"].max()
        keep = [j for j in range(len(summary["harmonics"])) if mode["structure"][:, j].max() > 0.1 * top]  # the dominant harmonics
        for j in keep:
            a2.plot(s, mode["structure"][:, j] / top, label="({},{})".format(*summary["harmonics"][j]))
        a2.set(
            xlabel="s",
            ylabel="|\u03c6_mn| (normalized)",
            box_aspect=1,
            title=f"{mode['frequency_khz']:.1f} kHz, s = {mode['peak_s']:.2f}" + ("" if mode["converged"] else ", unconverged"),
        )
        a2.legend(fontsize=7)
    n_i, amu = summary["density_m3"], summary["ion_mass_amu"]
    plasma = f"n_i = {n_i:.2g} m\u207b\u00b3, ion mass {amu:g} amu" if n_i else f"rho = {summary['density_kg_m3']:.3g} kg/m\u00b3"
    fig.suptitle(plasma, fontsize=9)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    path = Path(path)
    Image.open(buf).convert("RGB").quantize(colors=colors, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)
    return path
