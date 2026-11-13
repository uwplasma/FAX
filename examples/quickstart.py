"""Analytic-geometry tour: continuum, gaps, targeted and dense global modes, smooth objectives and a gradient.

Reproduces the TAE of the ITPA benchmark torus of Koenies et al., Nucl. Fusion 58, 126027 (2018): R0 = 10 m, a = 1 m, B0 = 3 T,
q = 1.71 + 0.16 s, hydrogen at 2e19 m^-3, n = 6, m = 5..16. The global TAE is 4.06e5 rad/s here (CAS3D 4.01e5, GYGLES 4.13e5).
Covers circular_torus, mode_family, mass_density, continuum, continuum_matrices, eigenvalue_mean, gaps, gap_location, surface_grid,
global_matrices, eigenmodes (targeted and dense), frequency_hz, gap_closure, soft_max, window_risk and jax.grad.
Needs no data. Writes quickstart.png when matplotlib is installed. Fast path about 15 s; RESEARCH adds a dense cross-check and more nodes.
"""

import io

import jax
import jax.numpy as jnp
import numpy as np

import fax

RESEARCH = False  # True: 400 nodes, the dense all-mode solve as a cross-check, a convergence scan in the angle grid
N_SURFACES = 400 if RESEARCH else 200  # radial surfaces (finite-element nodes)
N_TOROIDAL, M_MIN, M_MAX = 6, 5, 16  # toroidal mode number and the poloidal range
TARGET_HZ, N_MODES = 6.46e4, 3  # shift-invert target and number of modes kept; the TAE is the highest of the three
DENSITY_M3, ION_MASS_AMU = 2e19, 1.00727647  # a proton plasma
GRID = dict(ntheta=32, nzeta=4)  # angle grid of the global operator (nzeta = 4 is exact for the axisymmetric torus)
OUTPUT = "quickstart.png"  # figure file; skipped without matplotlib

s = (jnp.arange(N_SURFACES) + 0.5) / N_SURFACES
geo = fax.circular_torus(s=s, major_radius=10.0, minor_radius=1.0, B0=3.0, iota=1 / (1.71 + 0.16 * s), epsilon_B=0.1)
harmonics = fax.mode_family(N_TOROIDAL, M_MAX, M_MIN)
rho = fax.mass_density(DENSITY_M3, ION_MASS_AMU)

f_hz = fax.continuum(geo, harmonics, rho)  # (surfaces, harmonics), Hz, sorted on each surface
center, width = fax.gaps(f_hz)  # widest gap on each surface
i = N_SURFACES // 4  # q = 1.75 at s = 0.25, where m = 10 and 11 cross
below, above = f_hz[i][f_hz[i] < TARGET_HZ].max(), f_hz[i][f_hz[i] > TARGET_HZ].min()
print(f"continuum {f_hz.shape}: TAE gap at s = 0.25 spans {below / 1e3:.1f}-{above / 1e3:.1f} kHz")
iota_x, w_hat = fax.gap_location(N_TOROIDAL, 10, 1)  # cylindrical (10, 6)-(11, 6) crossing
print(f"cylinder crossing at iota = {iota_x:.4f} (q = {1 / iota_x:.3f}), omega R / v_A = {w_hat:.3f}")
grid = fax.surface_grid(geo, **GRID)
print(f"surface grid {grid.B.shape}: B {float(grid.B.min()):.2f}-{float(grid.B.max()):.2f} T")

modes = fax.eigenmodes(geo, harmonics, rho, target_hz=TARGET_HZ, count=N_MODES, **GRID)
tae = jnp.sqrt(modes.omega2[-1])
print(f"global modes (kHz) {np.round(modes.frequencies_hz / 1e3, 3)}; TAE omega = {tae:.4e} rad/s")
print(f"status: converged {bool(modes.status.converged)}, residual {float(modes.status.residual):.1e}, gap {float(modes.status.gap):.1e}")

F, G = fax.global_matrices(geo, harmonics, rho, **GRID)  # block-tridiagonal pencil F y = omega^2 G y
print(f"global_matrices: {F.diag.shape[0]} blocks of {F.diag.shape[1]} harmonics")
K, M = fax.continuum_matrices(geo.surfaces(jnp.arange(i - 1, i + 2)), harmonics, rho, **GRID)  # three neighbouring surfaces
w2 = fax.eigenvalue_mean(K[1], M[1], 0, 1)  # lowest eigenvalue of the middle one
print(f"lowest branch at s = 0.25: {float(jnp.sqrt(w2) / (2 * jnp.pi)) / 1e3:.2f} kHz, from continuum {float(f_hz[i, 0]) / 1e3:.2f} kHz")

frequency = lambda r: fax.frequency_hz(geo, harmonics, r, N_MODES - 1, N_MODES, target_hz=TARGET_HZ, count=N_MODES, **GRID)  # noqa: E731
f, dfdrho = frequency(rho), jax.grad(frequency)(rho)
print(f"TAE f = {f / 1e3:.3f} kHz; df/d rho = {dfdrho:.4e} (analytic -f/2rho = {-f / (2 * rho):.4e})")

print(f"gap_closure {float(fax.gap_closure(f_hz)):.4f}, soft_max of the TAE-gap width {float(fax.soft_max(width / 1e3)):.2f} kHz")
print(f"window_risk of the TAE in a +-1 kHz window {float(fax.window_risk(f, (f - 1e3, f + 1e3), 0.2e3)):.4f}")

if RESEARCH:
    dense = fax.eigenmodes(geo, harmonics, rho, **GRID)  # every mode, dense generalized solve
    near = jnp.sort(dense.frequencies_hz[jnp.argsort(jnp.abs(dense.frequencies_hz - TARGET_HZ))[:N_MODES]])
    print(
        f"dense solve agrees with the targeted one to {float(jnp.max(jnp.abs(near / modes.frequencies_hz - 1))):.1e}; status {dense.status}"
    )
    for ntheta in (16, 32, 64):
        g = fax.eigenmodes(geo, harmonics, rho, target_hz=TARGET_HZ, count=N_MODES, ntheta=ntheta, nzeta=4)
        print(f"ntheta {ntheta}: TAE {float(g.frequencies_hz[-1]) / 1e3:.4f} kHz")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
except ImportError:
    print("matplotlib not installed: no figure")
else:
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.2), constrained_layout=True)
    ax[0].plot(np.repeat(np.asarray(s), f_hz.shape[1]), np.asarray(f_hz).ravel() / 1e3, ".", ms=1, color="0.3")
    ax[0].axhspan(below / 1e3, above / 1e3, xmin=0.2, xmax=0.3, color="C1", alpha=0.4)
    for fk in modes.frequencies_hz:
        ax[0].axhline(fk / 1e3, color="C3", lw=0.8)
    ax[0].set(xlabel="s", ylabel="frequency (kHz)", ylim=(0, 150), title="continuum and global modes")
    tae_structure = np.abs(np.asarray(modes.structures[-1]))  # (nodes, harmonics)
    for k in np.argsort(-tae_structure.max(0))[:3]:
        ax[1].plot(np.asarray(s), tae_structure[:, k] / tae_structure.max(), lw=1.2, label=f"m = {int(harmonics[k, 0])}")
    ax[1].set(xlabel="s", title=f"TAE structure, {modes.frequencies_hz[-1] / 1e3:.1f} kHz", ylabel="|phi| (arb.)")
    ax[1].legend(fontsize=7)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    Image.open(buf).convert("RGB").quantize(colors=64, method=Image.Quantize.MEDIANCUT).save(OUTPUT, optimize=True)
    print("wrote", OUTPUT)
