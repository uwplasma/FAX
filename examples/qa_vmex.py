"""Precise QA from a VMEC input: global frequencies and their gradient with respect to the plasma boundary.

Reproduces the low-resolution precise-QA configuration of Landreman and Paul, Phys. Rev. Lett. 128, 035001 (2022): VMEX solves the
input, booz_xform_jax transforms it, FAX returns the global modes, and the implicit VMEX adjoint differentiates a
smooth frequency-window objective (docs/theory.md, "gradients through the equilibrium"). The same case as examples/qa_continuum.toml.
Needs the vmex extra (Python 3.11 or newer); prints only. Fast path a few minutes (it compiles the equilibrium adjoint); RESEARCH
adds the n = 1 gap modes and the checks of from_vmec_input, from_wout and the TOML run.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import vmex

import fax

RESEARCH = False  # True: the n = 1 gap modes, a 5-point line scan and the cross-check against from_vmec_input, from_wout and the TOML case
DECK = Path(__file__).resolve().parent / "data" / "input.LandremanPaul2021_QA_lowres"  # VMEC input
PARAMETERS = ("rc01", "zs01")  # boundary coefficients RBC(n=1, m=0) and ZBS(n=1, m=0) that are differentiated
NS, NODES, MBOZ, NBOZ = 16, 24, 8, 8  # VMEC surfaces, finite-element nodes, Boozer resolution
DENSITY = 3.4e-8  # kg/m^3, a few hundred kHz on this 1 m device
WINDOW_HZ, SOFTNESS_HZ = (250e3, 290e3), 10e3  # the frequency window the n = 0 modes are scored against
# n = 0 harmonics m = 1..3 (continuum-like modes) in the fast path; RESEARCH uses the n = 1 gap family nearest 103 kHz
if RESEARCH:
    HARMONICS, MODES, TARGET = fax.mode_family((1, -1, 3), 5), 2, dict(target_hz=103e3, count=2)
    WINDOW_HZ, SOFTNESS_HZ = (102e3, 140e3), 1e3
else:
    HARMONICS, MODES, TARGET = [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], 3, {}
STEPS = np.array([-2, -1, 0, 1, 2] if RESEARCH else [-1, 0, 1]) * 1e-3  # offsets along -grad of the line scan

design = fax.prepare_design(
    DECK, harmonics=HARMONICS, density=DENSITY, parameters=PARAMETERS, ns=NS, nodes=NODES, mboz=MBOZ, nboz=NBOZ, ntheta=8, nzeta=8, **TARGET
)
objective = lambda p: fax.window_risk(fax.design_frequencies(p, design, MODES)[0], WINDOW_HZ, SOFTNESS_HZ)  # noqa: E731
p0 = jnp.zeros(len(PARAMETERS))
geo = fax.design_geometry(p0, design)  # VMEX + Boozer on the finite-element nodes
f, status = fax.design_frequencies(p0, design, MODES)
print(f"{design.names}: frequencies (kHz) {np.round(np.asarray(f) / 1e3, 3)}, converged {bool(status.converged)}")
print(f"iota {float(geo.iota[0]):.4f} to {float(geo.iota[-1]):.4f} across the {NODES} nodes")
value, grad = jax.value_and_grad(objective)(p0)
print(f"risk = {value:.4e}, d risk / d {design.names} = {np.asarray(grad)}")
direction = -grad / jnp.linalg.norm(grad)
scan = np.array([float(objective(t * direction)) for t in STEPS])
print("risk along -grad (offsets", STEPS, "):", np.array2string(scan, precision=5))

state, runtime = fax.design_state(p0, design)  # the pieces design_frequencies chains, for custom objectives
geo_state = fax.state_geometry(state, runtime, design)
f_state, _ = fax.geometry_frequencies(geo_state, design, MODES)
print(f"design_params -> {type(fax.design_params(p0, design)).__name__}")
print(f"geometry route differs from design_frequencies by {float(jnp.max(jnp.abs(f_state - f))):.1e} Hz")

if RESEARCH:
    booz = dict(mboz=MBOZ, nboz=NBOZ, surfaces=(0.1, 0.3, 0.5, 0.7, 0.9))
    direct = fax.from_vmec_input(DECK, ns=NS, **booz)
    inp = vmex.VmecInput.from_file(str(DECK))
    wout = vmex.wout_from_result(inp, vmex.solve_multigrid(inp, [NS], [1e-12], [5000]))
    in_memory = fax.from_wout(wout, **booz)
    toml = fax.run(Path(__file__).resolve().parent / "qa_continuum.toml", save=False)
    print(f"from_vmec_input vs from_wout: iota differs by {float(jnp.max(jnp.abs(direct.iota - in_memory.iota))):.1e}")
    print(f"TOML run: {toml['frequencies_hz'].shape} frequencies, lowest {toml['frequencies_hz'].min() / 1e3:.1f} kHz")
