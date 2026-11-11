"""Figures of the FAX documentation. Run ``python docs/make_figures.py`` from the repository root.

It writes the ``physics_*``, ``theory_*`` and ``compare_*`` PNGs in ``docs/_static``, quantized to a small palette. The
physics figures are computed with FAX; the comparison figures plot numbers quoted in ``docs/validation.md``.
"""

import io
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

import fax

OUT = Path(__file__).parent / "_static"
BLUE, ORANGE, GREEN, PURPLE, GREY, RED, DARK = "#0f5c8c", "#d98b1f", "#2a9d6f", "#a0508f", "#8a8a8a", "#d55e00", "#555555"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8, "figure.dpi": 120})


def save(fig, name, colors=64):
    """Write ``fig`` as a palette PNG in ``docs/_static``."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    Image.open(buf).convert("RGB").quantize(colors=colors, method=Image.Quantize.MEDIANCUT).save(OUT / name, optimize=True)
    print(name, (OUT / name).stat().st_size // 1024, "KB")


# settings
R0, A, B0, DENSITY, MASS_AMU = 10.0, 1.0, 3.0, 2e19, 1.00727647  # the Koenies et al. (2018) torus
N_SURF = 400


def fig_tae_gap():
    """Two coupled harmonics (m, n) = (1, 1), (2, 1) on a circular torus against a uniform-field (cylinder) reference."""
    s = (jnp.arange(N_SURF) + 0.5) / N_SURF
    iota = 0.5 + 0.35 * s
    rho = fax.mass_density(DENSITY, MASS_AMU)
    vA = B0 / np.sqrt(4e-7 * np.pi * float(rho))
    unit = vA / (2 * np.pi * R0)  # f at omega R0 / v_A = 1
    h = fax.mode_family(1, 2, 1)
    out = {}
    for label, eps in (("cyl", 0.0), ("torus", 0.3)):
        geo = fax.circular_torus(s=s, major_radius=R0, minor_radius=A, B0=B0, iota=iota, epsilon_B=eps)
        out[label] = np.asarray(fax.continuum(geo, h, rho)) / unit
    x = np.asarray(iota)
    cross, w_cross = fax.gap_location(1, 1, 1)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for k in range(2):
        ax.plot(x, out["cyl"][:, k], color=GREY, lw=1.2, ls="--", label="uniform field" if k == 0 else None)
    for k in range(2):
        ax.plot(x, out["torus"][:, k], color=BLUE, lw=2.0, label="torus, $\\epsilon_B=0.3$" if k == 0 else None)
    lo, hi = out["torus"][:, 0], out["torus"][:, 1]
    j = np.argmin(np.abs(x - cross))
    ax.annotate("", xy=(cross, lo[j]), xytext=(cross, hi[j]), arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.4))
    ax.text(cross + 0.015, 0.5 * (lo[j] + hi[j]) + 0.012, "TAE gap", color=ORANGE, va="center")
    ax.axvline(cross, color=ORANGE, lw=0.8, ls=":")
    ax.set_xlabel("rotational transform $\\iota$ (radial coordinate: $s$ increases left to right)")
    ax.set_ylabel("$\\omega R_0 / v_A$")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc="upper left")
    save(fig, "physics_tae_gap.png")
    return cross, w_cross, float(lo[j]), float(hi[j])


def fig_gap_families():
    """Cylindrical branches |n - m iota| of the family n = 1 + k N_fp (N_fp = 2) and the crossings each helicity opens."""
    nfp = 2
    iota = np.linspace(0, 1, 400)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for n in (-1, 1, 3):
        for m in range(1, 5):
            y = np.abs(n - m * iota)
            ax.plot(iota, y, color="#cfcfcf", lw=0.7, zorder=1)
    styles = {
        "TAE": (BLUE, "o", "TAE: $\\Delta m=1$, $\\Delta n=0$"),
        "EAE": (ORANGE, "s", "EAE: $\\Delta m=2$, $\\Delta n=0$"),
        "MAE": (GREEN, "^", "MAE: $\\Delta m=0$, $\\Delta n\\neq0$"),
        "HAE": (PURPLE, "D", "HAE: $\\Delta m,\\Delta n\\neq0$"),
    }
    seen = set()
    for n in (-1, 1, 3):
        for m in range(1, 5):
            for dm in (0, 1, 2):
                for dn in (-1, 0, 1):
                    if (dm == 0 and dn <= 0) or (dm == 0 and dn == 0):
                        continue
                    kind = "TAE" if (dm == 1 and dn == 0) else "EAE" if (dm == 2 and dn == 0) else "MAE" if dm == 0 else "HAE"
                    if (m + dm) < 0 or (2 * m + dm) == 0:
                        continue
                    io, wr = fax.gap_location(n, m, dm, dn, nfp)
                    if not (0.05 <= io <= 0.95 and 0.02 <= wr <= 1.45):
                        continue
                    col, mk, lab = styles[kind]
                    ax.scatter([io], [wr], color=col, marker=mk, s=26, zorder=3, label=None if kind in seen else lab)
                    seen.add(kind)
    ax.set_xlabel("rotational transform $\\iota$")
    ax.set_ylabel("$\\omega R / v_A$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.5)
    ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, 1.2))
    save(fig, "physics_gap_families.png")


def fig_schematic():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
    # (a) a field line carrying a shear-Alfven wave: the line is displaced and the tension pulls it back
    z = np.linspace(0, 4 * np.pi, 300)
    for y0 in (-0.9, 0, 0.9):
        a.plot(z, y0 + 0.28 * np.sin(z), color=BLUE, lw=1.6)
        a.plot(z, np.full_like(z, y0), color="#cfcfcf", lw=0.8, ls="--")
        a.annotate("", xy=(4 * np.pi, -1.55), xytext=(0, -1.55), arrowprops=dict(arrowstyle="->", color="black", lw=0.8))
    a.text(2 * np.pi, -1.9, "along $\\mathbf{B}$ ($k_\\parallel$, wavelength $2\\pi/k_\\parallel$)", ha="center", fontsize=8.5)
    a.text(0.0, 1.45, "field lines displaced by the wave;\ntension straightens them", fontsize=8.5, va="center")
    a.set_ylim(-2.15, 1.75)
    a.axis("off")
    # (b) resonance with passing ions: where the parallel speed matches the wave phase speed
    v = np.linspace(0, 1.6, 400)
    F = np.exp(-(((v - 0.55) / 0.35) ** 2)) + 0.0
    b.plot(v, np.gradient(F, v) * 0 + F, color=GREY, lw=1.5)
    for x, lab in ((1.0, "$v_\\parallel = v_A$"), (1 / 3, "$v_A/3$")):
        b.axvline(x, color=ORANGE, lw=1.2, ls=":")
        b.text(x + 0.02, 1.0, lab, rotation=90, va="top", ha="left", fontsize=8.5, color=ORANGE)
    b.text(1.45, 0.55, "ions near these\nspeeds exchange\nenergy with the wave", ha="right", fontsize=8.5)
    b.set_xlabel("parallel ion speed $v_\\parallel$")
    b.set_ylabel("fast-ion distribution")
    b.set_yticks([])
    b.set_xticks([])
    b.set_ylim(0, 1.1)
    a.set_title("(a)", loc="left", fontsize=9)
    b.set_title("(b)", loc="left", fontsize=9)
    save(fig, "physics_schematic.png")


def fig_basis():
    """Hat functions on the nodes 0 < s_1 < ... < s_N < 1 and the block-tridiagonal pattern of F and G."""
    N = 6
    nodes = np.array([0.0, 0.12, 0.3, 0.46, 0.64, 0.8, 0.92, 1.0])  # axis, s_1..s_N, edge
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 2.8), gridspec_kw={"width_ratios": [1.25, 1]})
    for k in range(1, N + 1):
        a.plot(nodes[k - 1 : k + 2], [0, 1, 0], color=BLUE, lw=1.6)
    a.plot(nodes[:2], [1, 0], color=ORANGE, lw=1.6, ls="--")
    a.plot(nodes[-2:], [0, 0], color=GREY, lw=1.6, ls=":")
    a.plot(nodes[1:-1], np.zeros(N), "o", color=BLUE, ms=4)
    a.plot([0, 1], [0, 0], "s", color=GREY, ms=5)
    a.annotate(
        "axis node: dropped for $m \\neq 0$,\nflat to $s_1$ for $m = 0$",
        (0.0, 0.98),
        (0.1, 1.28),
        fontsize=8,
        color=ORANGE,
        arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 0.8},
    )
    arrow = {"arrowstyle": "-", "color": GREY, "lw": 0.8}
    a.annotate("edge node:\n$\\phi = 0$", (1.0, 0.03), (0.74, 1.25), fontsize=8, color=GREY, arrowprops=arrow)
    a.set(xlabel="$s = \\Phi_t/\\Phi_{\\rm edge}$", ylabel="$f_k(s)$", ylim=(-0.12, 1.65), yticks=[0, 1], title="radial hat functions")
    for k in range(1, N + 1):
        a.text(nodes[k], -0.07, f"$s_{k}$", ha="center", va="top", fontsize=7, color=BLUE)
    b.set_aspect("equal")
    for i in range(N):
        for j in range(N):
            if abs(i - j) <= 1:
                b.add_patch(Rectangle((j, N - 1 - i), 1, 1, fc=BLUE if i == j else "#7fb2d4", ec="white", lw=1))
    b.add_patch(Rectangle((0, N - 1), 1, 1, fc="none", ec=ORANGE, lw=2))
    b.set(xlim=(0, N), ylim=(0, N), xticks=[], yticks=[], title=f"$F$ and $G$: $N = {N}$ blocks of $H \\times H$")
    note = "each block is $H \\times H$ ($H = 3$ here); orange: axis block\nwith the $m = 0$ flat condition"
    b.text(N / 2, -0.35, note, ha="center", va="top", fontsize=8)
    b.spines[:].set_visible(False)
    save(fig, "theory_basis.png")


def fig_axis():
    """Linear extrapolation of a harmonic below the first surface against the regular behaviour (s/s0)^(m/2)."""
    s0 = 0.05
    s = np.linspace(0, 0.3, 300)
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.8), sharey=True)
    for a, m, colour in zip(ax, (1, 4), (BLUE, PURPLE), strict=True):
        truth = lambda x, m=m: (x / 0.2) ** (m / 2)  # a harmonic that is regular at the axis
        first, slope = truth(s0), (truth(0.2) - truth(s0)) / (0.2 - s0)  # value at s_0 and the slope of the line through two surfaces
        a.plot(s[s >= s0], truth(s[s >= s0]), color=colour, lw=1.8, label="resolved surfaces")
        a.plot(s[s <= s0], first + slope * (s[s <= s0] - s0), color=ORANGE, ls="--", lw=1.6, label="linear extrapolation")
        a.plot(s[s <= s0], first * (s[s <= s0] / s0) ** (m / 2), color=GREEN, lw=1.8, label="$(s/s_0)^{m/2}$")
        a.axvline(s0, color=GREY, lw=0.8, ls=":")
        a.set(title=f"$m = {m}$", xlabel="$s$", xlim=(0, 0.3), ylim=(-0.2, 1.3))
        a.text(s0 + 0.005, 1.2, "$s_0$", color=GREY, fontsize=8)
    ax[0].set_ylabel("harmonic amplitude (arb.)")
    ax[0].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 0.92))
    save(fig, "theory_axis.png")


def fig_ritz():
    """Why the Ritz pairs of the shift-invert operator are ranked by |mu|, with a ghost between two eigenvalues."""
    lam = np.array([-1.4, -0.9, -0.35, 0.3, 0.75, 1.3, 1.9])
    sigma, count = 0.0, 4
    mu = 1 / (lam - sigma)
    fig, (a, b) = plt.subplots(2, 1, figsize=(7.4, 3.3), gridspec_kw={"height_ratios": [1, 1.3]})
    near = np.argsort(np.abs(lam - sigma))[:count]
    a.axhline(0, color=GREY, lw=0.8)
    a.plot(lam, 0 * lam, "o", color=GREY, ms=7)
    a.plot(lam[near], 0 * lam[near], "o", color=BLUE, ms=9)
    a.axvline(sigma, color=ORANGE, lw=1.4)
    a.text(sigma + 0.04, 0.25, "$\\sigma$ (target)", color=ORANGE, fontsize=9)
    a.plot([0.0], [0.0], marker="x", color=PURPLE, ms=9, mew=2)
    ghost = "ghost: the Ritz value of a mix of two\neigenvectors on both sides of $\\sigma$"
    a.annotate(ghost, (0.0, 0.0), (0.6, 0.45), color=PURPLE, fontsize=8, arrowprops={"arrowstyle": "-", "color": PURPLE, "lw": 0.8})
    a.set(xlim=(-1.7, 2.2), ylim=(-0.5, 0.9), yticks=[], xlabel="$\\lambda = \\omega^2$ (arb.)")
    a.set_title(f"the {count} eigenvalues nearest $\\sigma$ (blue) are wanted")
    a.spines["left"].set_visible(False)
    b.bar(np.arange(len(lam)), np.abs(mu), color=[BLUE if i in near else GREY for i in range(len(lam))], width=0.6)
    b.set(
        xticks=np.arange(len(lam)),
        xticklabels=[f"{x:.2f}" for x in lam],
        xlabel="$\\lambda$ of the Ritz value",
        ylabel="$|\\mu| = 1/|\\lambda - \\sigma|$",
        title="Ritz values of the shift-invert operator, ranked by $|\\mu|$",
    )
    fig.tight_layout()
    save(fig, "theory_ritz.png")


# (label, relative difference, gate or None, status). Sources: validation.md Summary table and the
# sections Continuum, Global operator, Independent 3-D reference; the last two are open items.
AGREEMENT = [
    ("STELLGAP, CTOK continuum (mean)", 6.3e-3, 7e-3, "matched"),
    ("STELLGAP, precise-QA continuum (mean)", 3.0e-3, 4e-3, "matched"),
    ("AE3D, six CTOK global modes (worst)", 1.4e-2, 1.5e-2, "matched"),
    ("CAS3D, Könies TAE, no pressure (4.061 vs 4.01)", 4.061 / 4.01 - 1, None, "matched"),
    ("GYGLES, Könies TAE (4.061 vs 4.13)", 1 - 4.061 / 4.13, None, "matched"),
    ("CAS3D, Könies TAE, vmex + finite-$\\beta$ (3.948)", 1 - 3.9482 / 4.01, None, "matched"),
    ("AE3D, CTOK TAE Rayleigh quotient (324.2 vs 338.2 kHz)", 1 - 324.2 / 338.2, None, "open"),
    ("AE3D, QA global mode, energy at 162 harmonics", 5.1e-2, None, "open"),
    ("CAS3D-K, drive at 82-191 keV (ratio 0.58-0.60)", 0.41, None, "open"),
]


def agreement():
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    n = len(AGREEMENT)
    for i, (_, v, gate, st) in enumerate(AGREEMENT):
        y = n - 1 - i
        c = BLUE if st == "matched" else RED
        ax.plot([v], [y], "o", ms=8, mfc=c if st == "matched" else "white", mec=c, mew=2, zorder=3)
        if gate:
            ax.plot([gate, gate], [y - 0.28, y + 0.28], color=DARK, lw=2, zorder=2)
    ax.set_yticks(range(n))
    ax.set_yticklabels([a[0] for a in AGREEMENT][::-1], fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlim(1.5e-3, 1.0)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("relative difference (dimensionless)")
    ax.grid(axis="x", color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
    ax.plot([], [], "o", mfc=BLUE, mec=BLUE, label="matched")
    ax.plot([], [], "o", mfc="white", mec=RED, mew=2, label="open")
    ax.plot([], [], "|", color=DARK, ms=10, mew=2, label="test gate")
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    save(fig, "compare_agreement.png")


# validation.md, Performance at equal accuracy, Scaling table (0.32 solve, iterations) and the paragraph below it (0.36).
N = [200, 400, 800, 1600, 3200]
IT32 = [17, 33, 112, 310, 1000]  # 3200: stopped at the 1000-iteration cap, eta 2.8e-10
IT36 = [4, 5, 11, 34, 121]
T32 = [0.060, 0.167, 1.08, 5.97, 40.1]
T36 = [0.05, 0.11, 0.39, 2.3, 15]


def scaling():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.3))
    for ax, y32, y36, lab in ((a, IT32, IT36, "shift-invert iterations"), (b, T32, T36, "warm solve time (s)")):
        ax.loglog(N[:4], y32[:4], "o-", color=RED, label="SOLVAX 0.32")
        ax.loglog(N[3:], y32[3:], "--", color=RED)
        ax.loglog(N[4:], y32[4:], "o", color=RED, mfc="white", mew=1.5)
        ax.loglog(N, y36, "s-", color=BLUE, label="SOLVAX 0.36")
        ax.set_xlabel("radial nodes $N$")
        ax.set_ylabel(lab)
        ax.set_xticks(N)
        ax.set_xticklabels([str(x) for x in N], fontsize=8.5)
        ax.minorticks_off()
        ax.grid(color="#dddddd", lw=0.6)
        ax.set_axisbelow(True)
    a.annotate(
        "stopped at the cap\n(eta 2.8e-10)",
        (3200, 1000),
        (200, 220),
        fontsize=7.5,
        color=RED,
        arrowprops={"arrowstyle": "-", "color": RED, "lw": 0.8},
    )
    b.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    save(fig, "compare_scaling.png")


if __name__ == "__main__":
    for make in (fig_tae_gap, fig_gap_families, fig_schematic, fig_basis, fig_axis, fig_ritz, agreement, scaling):
        make()
