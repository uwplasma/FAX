"""Standalone quickstart on analytic geometry (no data files, no plotting): a continuum, a global TAE and one gradient.

The Koenies et al. (2018) ITPA torus: R0 = 10 m, a = 1 m, B0 = 3 T, q = 1.71 + 0.16 s, hydrogen at 2e19 m^-3, n = 6, m = 5..16.
Run: python examples/quickstart.py
"""

import jax
import jax.numpy as jnp

import fax

# settings
N_SURFACES, N_TOROIDAL, M_MIN, M_MAX = 200, 6, 5, 16
TARGET_HZ, N_MODES = 6.46e4, 3  # shift-invert target and the number of modes kept; the TAE is the highest of the three
DENSITY_M3, ION_MASS_AMU = 2e19, 1.00727647  # a proton
GRID = dict(ntheta=32, nzeta=4)

s = (jnp.arange(N_SURFACES) + 0.5) / N_SURFACES
geo = fax.circular_torus(s=s, major_radius=10.0, minor_radius=1.0, B0=3.0, iota=1 / (1.71 + 0.16 * s), epsilon_B=0.1)
harmonics = fax.mode_family(N_TOROIDAL, M_MAX, M_MIN)
rho = fax.mass_density(DENSITY_M3, ION_MASS_AMU)

f_hz = fax.continuum(geo, harmonics, rho)  # (surfaces, harmonics), Hz, sorted on each surface
f_q175 = f_hz[N_SURFACES // 4]  # q = 1.75 at s = 0.25: the surface of the m = 10, 11 crossing
below, above = f_q175[f_q175 < TARGET_HZ].max(), f_q175[f_q175 > TARGET_HZ].min()
print(f"continuum {f_hz.shape}: TAE gap at s = 0.25 spans {below / 1e3:.1f}-{above / 1e3:.1f} kHz")

modes = fax.eigenmodes(geo, harmonics, rho, target_hz=TARGET_HZ, count=N_MODES, **GRID)
tae = jnp.sqrt(modes.omega2[-1])
print(f"global modes (kHz) {modes.frequencies_hz / 1e3}; TAE omega = {tae:.4e} rad/s (CAS3D 4.01e5, GYGLES 4.13e5)")
print(f"status: converged {bool(modes.status.converged)}, residual {float(modes.status.residual):.1e}, gap {float(modes.status.gap):.1e}")

frequency = lambda r: fax.frequency_hz(geo, harmonics, r, N_MODES - 1, N_MODES, target_hz=TARGET_HZ, count=N_MODES, **GRID)  # noqa: E731
f, dfdrho = frequency(rho), jax.grad(frequency)(rho)
print(f"TAE f = {f / 1e3:.3f} kHz; df/d rho = {dfdrho:.4e} (analytic -f/2rho = {-f / (2 * rho):.4e})")
