"""Koenies et al. (2018) ITPA TAE: zero-orbit-width, passing-only growth rate vs fast-ion temperature.

Needs the plot extra: pip install "alfvex[plot]".  Run: python examples/koenies_tae_drive.py  (about 10 s).
The CAS3D-K (ZOW) points of fig. 1a are plotted for orientation only: whether that curve holds trapped particles is not stated, so the
comparison is unmatched (docs/validation.md, ITPA ladder rung 2). The pooled ZLR band is not a reference.
"""

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt

import fax

# settings
OMEGA = 4.0e5  # rad/s, the model TAE
TEMPERATURES_KEV = jnp.linspace(50.0, 500.0, 19)
OUTPUT = Path("koenies_tae_drive.png")
CAS3DK_ZOW = {82: 4.26, 139: 14.60, 163: 19.98, 191: 25.17}  # keV: 10^3 s^-1, read off fig. 1a

s = (jnp.arange(200) + 0.5) / 200
r = jnp.sqrt(s)
c0, c1, c2, c3 = 0.49123, 0.298228, 0.198739, 0.521298
density = 1.44131e17 * c3 * jnp.exp(-(c2 / c1) * jnp.tanh((r - c0) / c2))  # eq. 1 of the paper
mode = jnp.exp(-(((r - 0.5) / 0.0425) ** 2) / 2)
phi = jnp.stack([mode, mode], axis=1)  # model TAE: m = 10, 11 centred on q = 1.75
geo = fax.circular_torus(s=s, major_radius=10.0, minor_radius=1.0, B0=3.0, iota=1 / (1.71 + 0.16 * s), epsilon_B=0.1)
rho = fax.mass_density(2e19, 1.00727647)
drive = lambda t: fax.growth_rate(OMEGA, phi, geo, ((10, 6), (11, 6)), fax.FastIons(density, t * 1e3), rho, helical=0, nzeta=1)  # noqa: E731
gamma = jnp.array([drive(t) for t in TEMPERATURES_KEV])
for t in (100.0, 200.0, 300.0, 400.0):
    print(f"T_f = {t:.0f} keV: gamma = {drive(t) / 1e3:.2f} x10^3 1/s, gamma/omega = {drive(t) / OMEGA:.3f}")

plt.plot(TEMPERATURES_KEV, gamma / 1e3, "k-", label="FAX (ZOW, passing only)")
plt.plot(list(CAS3DK_ZOW), list(CAS3DK_ZOW.values()), "o", label="CAS3D-K (ZOW), unmatched content")
plt.axvspan(200, 500, color="0.92", label="perturbative validity doubtful above ~200 keV")
plt.xlabel("T_f (keV)")
plt.ylabel("gamma (10^3 / s)")
plt.legend()
plt.savefig(OUTPUT, dpi=120)
