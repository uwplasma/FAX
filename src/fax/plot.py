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
