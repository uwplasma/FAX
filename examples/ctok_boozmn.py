"""STELLGAP CTOK test case from a boozmn file: the TOML cases, the same calculation through the Python API, and the CLI.

CTOK is a circular tokamak (n = 2, m = 0..12, deuterium at 1e19 m^-3). The continuum follows STELLGAP (Spong et al., Phys. Plasmas
10, 3217 (2003); docs/validation.md); the eigenmodes are the AE3D-form modes nearest 384 kHz (Spong et al., Phys. Plasmas 17, 022106
(2010)). Needs tests/data from a checkout; prints only. Fast path about 10 s; RESEARCH adds the dense all-mode solve, the full mesh
and `fax run --plot --cache` on both cases.
"""

from pathlib import Path

import numpy as np

import fax

RESEARCH = False  # True: dense solve of every mode, the file's full mesh, and `fax run --plot --cache` on both TOML cases
EXAMPLES = Path(__file__).resolve().parent  # folder of the TOML cases
TARGET_KHZ, COUNT = 384.0, 6  # shift-invert target and number of modes (as in ctok_eigenmodes.toml)
GRID = dict(ntheta=32, nzeta=4)  # angle grid of the global operator

cases = EXAMPLES / "ctok_continuum.toml", EXAMPLES / "ctok_eigenmodes.toml"
config = fax.load(cases[0])  # TOML merged over the defaults, paths resolved against the file
print("case", config["modes"], config["plasma"], config["run"]["task"])
cont, modes = (fax.run(c, save=False) for c in cases)  # dicts of NumPy arrays

geo = fax.read_boozmn(config["geometry"]["path"])  # the same through the Python API
harmonics, rho = fax.mode_family(2, 12), fax.mass_density(1e19, 2.0)
f_hz = fax.continuum(geo, harmonics, rho)
center, width = fax.gaps(f_hz[:, :6])  # widest gap among the six lowest branches
k = 50  # a mid-radius surface
print(f"continuum {f_hz.shape}; widest gap of the six lowest branches at s = {float(geo.s[k]):.2f}:")
print(f"  centre {center[k] / 1e3:.1f} kHz, width {width[k] / 1e3:.1f} kHz")
print(f"API against fax.run: {np.abs(np.asarray(f_hz) - cont['frequencies_hz']).max():.1e} Hz")
direct = fax.eigenmodes(geo, harmonics, rho, target_hz=TARGET_KHZ * 1e3, count=COUNT, **GRID)
print(f"modes near {TARGET_KHZ:g} kHz (kHz): {np.round(modes['frequencies_hz'] / 1e3, 1)}; converged {bool(direct.status.converged)}")
print(f"API against fax.run: {np.abs(np.asarray(direct.frequencies_hz) - modes['frequencies_hz']).max():.1e} Hz")
sub = fax.run({"geometry": config["geometry"], "modes": {"n": [2], "m_max": 6}, "run": {"task": "continuum"}}, save=False)  # dict case
print(f"dict case, m <= 6: {sub['frequencies_hz'].shape} frequencies on {sub['s'].size} surfaces")
coarse = geo.interpolate(np.linspace(0.05, 0.95, 19))
print(f"interpolated to {coarse.s.size} surfaces: iota {float(coarse.iota[0]):.3f} to {float(coarse.iota[-1]):.3f}")

if RESEARCH:
    dense = fax.eigenmodes(geo, harmonics, rho, **GRID)  # every mode: O((N H)^3)
    f = np.asarray(dense.frequencies_hz)
    print(f"dense solve, {f.size} modes: nearest {np.round(np.sort(f[np.argsort(np.abs(f - TARGET_KHZ * 1e3))[:COUNT]]) / 1e3, 1)} kHz")
    full = geo.full_mesh()
    on_full = fax.eigenmodes(full, harmonics, rho, target_hz=TARGET_KHZ * 1e3, count=COUNT, **GRID)
    print(f"full mesh {full.s.size} surfaces (AE3D's own grid): {np.round(on_full.frequencies_hz / 1e3, 1)} kHz")
    for case in cases:
        fax.io.main(["run", str(case), "--plot", "--cache"])  # same as the shell command; writes ../outputs/<case>/ with PNGs
