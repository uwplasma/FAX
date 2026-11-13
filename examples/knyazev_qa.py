"""Precise-QA comparison with Knyazev et al., arXiv:2603.03118 (2026): FAX against the STELLGAP continuum and the AE3D gap band.

Reference: the finite-beta precise-QA equilibrium (nfp = 2, aspect 6), STELLGAP continuum and AE3D with its kink term removed, a flat
mass density of 1.88e-6 kg/m^3 (docs/validation.md). The offline fixture tests/data/knyazev_qa.npz comes from the Zenodo archive
(DOI 10.5281/zenodo.21299907, Knyazev_JPP2026.zip, 1.13 GB, CC-BY-4.0); the helpers are in tests/knyazev_ref.py. Compares the
continuum branches, the archived global mode (30.31 kHz) and the AE3D band with FAX's sine-basis modes at growing resolution.
Needs matplotlib. Fast path about 2 minutes from the fixture (24 and 40 harmonics) and writes knyazev_qa.png; RESEARCH runs the
eight resolutions of docs/validation.md (about 15 minutes and 8 GB) and, if ARCHIVE is set, first rebuilds the fixture.
"""

import io
import sys
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import knyazev_ref as kr  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESEARCH = False  # True: eight resolutions, the sine-sector continuum error and the subspace-containment table
ARCHIVE = None  # path of Knyazev_JPP2026.zip: rebuild the fixture and print the SHA-256 values for tests/data/PROVENANCE.md (RESEARCH only)
OUTPUT = "knyazev_qa.png"  # figure file; docs/_static/knyazev_qa.png is the research one
CONFIGS = [  # (radial nodes, harmonics, theta grid, zeta grid)
    (49, 24, 48, 32),
    (49, 40, 48, 32),
    *((49, 64, 48, 32), (49, 64, 64, 48), (49, 96, 48, 32), (49, 162, 48, 32), (24, 40, 48, 32), (99, 40, 48, 32)) * RESEARCH,
]
PR9_KHZ = {"FAX, physical indices": 27.57, "public AE3D formulas": 30.30}  # PR #9 diagnostic reconstruction (not rerun here)

if RESEARCH and ARCHIVE:
    for name, value in kr.build(ARCHIVE).items():
        print(value, name)
fx = kr.load()
print("fixture", kr.FIXTURE.stat().st_size, "bytes")

sub = fx if RESEARCH else {**fx, "sg_s": fx["sg_s"][::8], "sg_khz": fx["sg_khz"][::8]}  # every 8th surface in the fast path
err, edge, s = kr.continuum_error(sub, 64, 48)
print(f"continuum, {len(s)} surfaces: complex basis mean {err.mean():.1e} p95 {np.percentile(err, 95):.1e} max {err.max():.1e}, ", end="")
print(f"gap edges (s > 0.25) max {edge[s > 0.25].max():.1e}")
if RESEARCH:
    err2 = kr.continuum_error(fx, 112, 72, sine=True)[0]
    print(f"  sine sector mean {err2.mean():.1e} max {err2.max():.1e}")

runs, rows = {}, {}
for cfg in CONFIGS:
    r = kr.run(fx, *cfg)
    c = rows[cfg] = kr.compare(fx, r)
    shift = c["rq"] / (fx["band_khz"] if len(c["rq"]) > 1 else kr.GLOBAL_KHZ) - 1
    print(
        f"nodes {cfg[0]} harmonics {cfg[1]} grid {cfg[2]}x{cfg[3]}: {c['band_all']} modes in {kr.BAND}, {len(c['band'])} band modes", end=""
    )
    print(
        f" {c['band'].min():.1f}-{c['band'].max():.1f} kHz; global: shift {shift[c['global']]:+.4f}, best {c['best_khz']:.2f} kHz", end=""
    )
    print(f" ({c['best_ov']:.2f}), IPR {c['ipr']:.1f} vs {c['ipr_ae3d']:.1f}, s {c['centroid']:.2f} vs {c['centroid_ae3d']:.2f}")
    print(f"  band shift {shift.mean():+.4f}")
    runs[cfg] = r._replace(F=None, y=r.y if cfg[1] < 162 else None, G=r.G if cfg[1] < 162 else None)

if RESEARCH:
    base = runs[(49, 64, 48, 32)]
    for name, small, ref in (
        ("grid 48x32 vs 64x48", base, runs[(49, 64, 64, 48)]),
        ("harmonics 40 vs 64", runs[(49, 40, 48, 32)], base),
        ("harmonics 64 vs 96", base, runs[(49, 96, 48, 32)]),
        ("nodes 24 vs 49", runs[(24, 40, 48, 32)], runs[(49, 40, 48, 32)]),
        ("nodes 49 vs 99", runs[(49, 40, 48, 32)], runs[(99, 40, 48, 32)]),
    ):
        cos = kr.contained(small, ref, fx)
        print(f"FAX band subspace in itself, {name}: {(cos**2).sum() / small.band.sum():.2f} (smallest cosine {cos.min():.2f})")

main = [k for k in CONFIGS if k[0] == 49 and k[2] == 48]  # the 48 x 32 grid at 49 nodes, ordered by harmonics
best = max((k for k in main if k[1] <= 96), key=lambda k: k[1])
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
sg_s, sg = fx["sg_s"] - 0.02, fx["sg_khz"].astype(float)
ax[0].plot(np.repeat(sg_s, sg.shape[1]), sg.ravel(), ".", ms=1, color="0.6", label="STELLGAP")
f = kr.continuum_khz(fx, kr.geometry_half(fx).interpolate(sg_s))
ax[0].plot(np.repeat(sg_s, f.shape[1]), f.ravel(), ".", ms=1, color="C1", alpha=0.5, label="FAX")
r = runs[best]
ax[0].plot(
    kr.radial_weight(r.y[r.band]) @ r.s, r.khz[r.band], "o", ms=3, color="C3", label=f"FAX gap band ({r.band.sum()}, {best[1]} harm.)"
)
ref = fx["band_vec"].astype(float)
ax[0].plot(kr.radial_weight(ref) @ fx["ae_s"], fx["band_khz"], "kx", ms=4, label=f"AE3D gap band ({len(ref)})")
ax[0].set(xlabel="$s$", ylabel="kHz", ylim=(0, 60), title="(a) continuum and gap band")
ax[0].legend(fontsize=7, loc="upper right")
for k, color in zip(main[-2:], ("C0", "C3"), strict=False):
    ax[1].vlines(runs[k].khz, 0, rows[k]["spread"], color=color, lw=1.2, label=f"{k[1]} harmonics")
ax[1].axvline(kr.GLOBAL_KHZ, color="k", ls="--", lw=1, label="archived AE3D 30.31")
for (label, v), ls in zip(PR9_KHZ.items(), (":", "-."), strict=True):
    ax[1].axvline(v, color="0.4", ls=ls, lw=1, label=f"{label} {v}")
ax[1].set(
    xlim=(26, 33), xlabel="FAX eigenfrequency (kHz)", ylabel="overlap with the archived global mode", title="(b) one mode, many FAX modes"
)
ax[1].legend(fontsize=6)
hls = [k[1] for k in main]
ax[2].plot(hls, [100 * (rows[k]["rq"][rows[k]["global"]] / kr.GLOBAL_KHZ - 1) for k in main], "o-", label="global mode")
small = [k for k in main if k[1] <= 96]
ax[2].plot([k[1] for k in small], [100 * (rows[k]["rq"] / fx["band_khz"] - 1).mean() for k in small], "s-", label="band mean")
ax[2].set(xscale="log", xlabel="harmonics kept", ylabel="Rayleigh-quotient shift (%)", title="(c) energy of the AE3D modes in FAX")
ax[2].axhline(0, color="k", lw=0.5)
ax[2].legend(fontsize=7)
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
buf.seek(0)
Image.open(buf).convert("RGB").quantize(colors=64, method=Image.Quantize.MEDIANCUT).save(OUTPUT, optimize=True)
print("wrote", OUTPUT)
