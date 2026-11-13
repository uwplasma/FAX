"""STELLGAP CTOK test case from a boozmn file: the TOML cases, the same calculation through the Python API, and the CLI.

CTOK is a circular tokamak (n = 2, m = 0..12, deuterium at 1e19 m^-3). The continuum follows STELLGAP (Spong et al., Phys. Plasmas
10, 3217 (2003); docs/validation.md); the eigenmodes are the AE3D-form modes
nearest 384 kHz (Spong et al., Phys. Plasmas 17, 022106 (2010)).
Covers fax.load, fax.run (continuum and eigenmodes tasks, TOML file and dict), read_boozmn, Geometry.interpolate and full_mesh,
continuum, gaps, eigenmodes and the fax command line (RESEARCH only). Needs tests/data from a checkout and matplotlib.
Writes ctok_boozmn.png; fast path about 10 s, RESEARCH adds the dense all-mode solve, the full mesh and the command line.
"""

import io
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

import fax

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESEARCH = False  # True: dense solve of every mode, the file's full mesh, and `fax run --plot --cache` on both TOML cases
EXAMPLES = Path(__file__).resolve().parent  # folder of the TOML cases
TARGET_KHZ, COUNT = 384.0, 6  # shift-invert target and number of modes (as in ctok_eigenmodes.toml)
OUTPUT = "ctok_boozmn.png"  # figure file

cont_case, mode_case = EXAMPLES / "ctok_continuum.toml", EXAMPLES / "ctok_eigenmodes.toml"
config = fax.load(cont_case)  # TOML merged over the defaults, paths resolved against the file
print("case", config["modes"], config["plasma"], config["run"]["task"])
cont, modes = fax.run(cont_case, save=False), fax.run(mode_case, save=False)  # dicts of NumPy arrays

geo = fax.read_boozmn(config["geometry"]["path"])  # the same through the Python API
harmonics, rho = fax.mode_family(2, 12), fax.mass_density(1e19, 2.0)
f_hz = fax.continuum(geo, harmonics, rho)
center, width = fax.gaps(f_hz[:, :6])  # widest gap among the six lowest branches
k = 50  # a mid-radius surface
print(f"continuum {f_hz.shape}; widest gap of the six lowest branches at s = {float(geo.s[k]):.2f}:")
print(f"  centre {center[k] / 1e3:.1f} kHz, width {width[k] / 1e3:.1f} kHz")
print(f"API against fax.run: {np.abs(np.asarray(f_hz) - cont['frequencies_hz']).max():.1e} Hz")
direct = fax.eigenmodes(geo, harmonics, rho, target_hz=TARGET_KHZ * 1e3, count=COUNT, ntheta=32, nzeta=4)
print(f"modes near {TARGET_KHZ:g} kHz (kHz): {np.round(modes['frequencies_hz'] / 1e3, 1)}; converged {bool(direct.status.converged)}")
print(f"API against fax.run: {np.abs(np.asarray(direct.frequencies_hz) - modes['frequencies_hz']).max():.1e} Hz")
sub = fax.run({"geometry": config["geometry"], "modes": {"n": [2], "m_max": 6}, "run": {"task": "continuum"}}, save=False)  # dict case
print(f"dict case, m <= 6: {sub['frequencies_hz'].shape} frequencies on {sub['s'].size} surfaces")
coarse = geo.interpolate(np.linspace(0.05, 0.95, 19))
print(f"interpolated to {coarse.s.size} surfaces: iota {float(coarse.iota[0]):.3f} to {float(coarse.iota[-1]):.3f}")

if RESEARCH:
    dense = fax.eigenmodes(geo, harmonics, rho, ntheta=32, nzeta=4)  # every mode: O((N H)^3)
    near = np.sort(np.asarray(dense.frequencies_hz)[np.argsort(np.abs(np.asarray(dense.frequencies_hz) - TARGET_KHZ * 1e3))[:COUNT]])
    print(f"dense solve, {dense.frequencies_hz.size} modes: nearest {np.round(near / 1e3, 1)} kHz, status {dense.status}")
    full = geo.full_mesh()
    on_full = fax.eigenmodes(full, harmonics, rho, target_hz=TARGET_KHZ * 1e3, count=COUNT, ntheta=32, nzeta=4)
    print(f"full mesh {full.s.size} surfaces (AE3D's own grid): {np.round(on_full.frequencies_hz / 1e3, 1)} kHz")
    for case in (cont_case, mode_case):
        fax.io.main(["run", str(case), "--plot", "--cache"])  # same as the shell command; writes ../outputs/<case>/ with PNGs

fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.2), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.2, 1]))
ax[0].plot(np.repeat(cont["s"], cont["frequencies_hz"].shape[1]), cont["frequencies_hz"].ravel() / 1e3, ".", ms=1.5, color="0.3")
for f in modes["frequencies_hz"]:
    ax[0].axhline(f / 1e3, lw=0.8, color="C3", alpha=0.6)
ax[0].set(xlabel="s", ylabel="frequency (kHz)", ylim=(0, 800), title="CTOK n = 2 continuum, modes near 384 kHz")
kmode = int(np.argmin(np.abs(modes["frequencies_hz"] - TARGET_KHZ * 1e3)))
profile = np.abs(modes["structures"][kmode]) / np.abs(modes["structures"][kmode]).max()
for j in np.argsort(-profile.max(0))[:3]:
    ax[1].plot(modes["mode_s"], profile[:, j], label=f"m = {int(modes['harmonics'][j, 0])}")
ax[1].set(xlabel="s", ylabel="|phi| (arb.)", title=f"mode at {modes['frequencies_hz'][kmode] / 1e3:.1f} kHz")
ax[1].legend(fontsize=7)
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
buf.seek(0)
Image.open(buf).convert("RGB").quantize(colors=64, method=Image.Quantize.MEDIANCUT).save(OUTPUT, optimize=True)
print("wrote", OUTPUT)
