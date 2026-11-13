"""Fast-ion drive and bulk-ion Landau damping of the ITPA TAE of Koenies et al., Nucl. Fusion 58, 126027 (2018).

A frozen-mode, zero-orbit-width, passing-only growth rate against fast-ion temperature (theory section 8). The CAS3D-K (ZOW) points of
fig. 1a are drawn for orientation only: whether that curve holds trapped particles is not stated, so the comparison is unmatched
(docs/validation.md, ITPA ladder rung 2). The pooled ZLR band is not a reference. Covers FastIons (Maxwellian and slowing-down),
growth_rate (also resolved=True) and ion_landau_damping on circular_torus geometry. Needs the plot extra; writes
koenies_tae_drive.png. Fast path about 10 s; RESEARCH adds the slowing-down distribution, the resonance table and the damping.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import fax

RESEARCH = False  # True: 19 temperatures, slowing-down ions, the resonance table, bulk-ion Landau damping
OMEGA = 4.0e5  # rad/s, the model TAE
TEMPERATURES_KEV = jnp.linspace(50.0, 500.0, 19 if RESEARCH else 7)  # fast-ion temperatures
SLOWING_DOWN = dict(birth_ev=3.5e6, critical_ev=0.5e6)  # RESEARCH: fusion-alpha-like birth and critical energies
BULK_TEMPERATURE_EV = 10e3  # RESEARCH: bulk-ion temperature of the Landau-damping estimate
OUTPUT = "koenies_tae_drive.png"  # figure file
CAS3DK_ZOW = {82: 4.26, 139: 14.60, 163: 19.98, 191: 25.17}  # keV: 10^3 s^-1, read off fig. 1a

s = (jnp.arange(200) + 0.5) / 200
r = jnp.sqrt(s)
c0, c1, c2, c3 = 0.49123, 0.298228, 0.198739, 0.521298
density = 1.44131e17 * c3 * jnp.exp(-(c2 / c1) * jnp.tanh((r - c0) / c2))  # eq. 1 of the paper
mode = jnp.exp(-(((r - 0.5) / 0.0425) ** 2) / 2)
phi = jnp.stack([mode, mode], axis=1)  # model TAE: m = 10, 11 centred on q = 1.75
harmonics = ((10, 6), (11, 6))
geo = fax.circular_torus(s=s, major_radius=10.0, minor_radius=1.0, B0=3.0, iota=1 / (1.71 + 0.16 * s), epsilon_B=0.1)
rho = fax.mass_density(2e19, 1.00727647)
drive = lambda ions: fax.growth_rate(OMEGA, phi, geo, harmonics, ions, rho, helical=0, nzeta=1)  # noqa: E731
gamma = jnp.array([drive(fax.FastIons(density, t * 1e3)) for t in TEMPERATURES_KEV])
for t in (100.0, 200.0, 300.0, 400.0):
    g = drive(fax.FastIons(density, t * 1e3))
    print(f"T_f = {t:.0f} keV: gamma = {g / 1e3:.2f} x10^3 1/s, gamma/omega = {g / OMEGA:.3f}")

if RESEARCH:
    sd = drive(fax.FastIons(density, kind="slowing_down", **SLOWING_DOWN))
    print(f"slowing-down ions ({SLOWING_DOWN['birth_ev'] / 1e6:g} MeV birth): gamma = {sd / 1e3:.2f} x10^3 1/s")
    pairs, parts = fax.growth_rate(OMEGA, phi, geo, harmonics, fax.FastIons(density, 200e3), rho, helical=0, nzeta=1, resolved=True)
    for (j, k), part in sorted(zip(pairs.tolist(), np.asarray(parts), strict=True), key=lambda x: -abs(x[1]))[:4]:
        print(f"  resonance j = {j}, k = {k}: {part / 1e3:+.3f} x10^3 1/s")
    damping = fax.ion_landau_damping(OMEGA, phi, geo, harmonics, BULK_TEMPERATURE_EV, 2e19, helical=0, nzeta=1)
    print(f"bulk-ion Landau damping at {BULK_TEMPERATURE_EV / 1e3:g} keV: {damping / 1e3:.3g} x10^3 1/s")

plt.figure(figsize=(4.6, 3.2))
plt.plot(TEMPERATURES_KEV, gamma / 1e3, "k-", label="FAX (ZOW, passing only)")
plt.plot(list(CAS3DK_ZOW), list(CAS3DK_ZOW.values()), "o", label="CAS3D-K (ZOW), unmatched content")
plt.axvspan(200, 500, color="0.92", label="perturbative validity doubtful above ~200 keV")
plt.xlabel("T_f (keV)")
plt.ylabel("gamma (10^3 / s)")
plt.legend(fontsize=7)
plt.savefig(OUTPUT, dpi=110, bbox_inches="tight")
print("wrote", OUTPUT)
