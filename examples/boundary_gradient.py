"""Gradient of global Alfvén frequencies with respect to VMEC boundary coefficients.

Run: python examples/boundary_gradient.py   (needs ``pip install "alfvex[vmex]"``)
"""

from pathlib import Path

import jax
import jax.numpy as jnp

import fax

design = fax.prepare_design(
    Path(__file__).parent / "data" / "input.LandremanPaul2021_QA_lowres",
    harmonics=[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
    density=3.4e-8,
    parameters=("rc01", "zs01"),
    ntheta=8,
    nzeta=8,
)
objective = lambda p: fax.window_risk(fax.design_frequencies(p, design)[0], (250e3, 290e3), 10e3)  # noqa: E731
p = jnp.zeros(2)
value, grad = jax.value_and_grad(objective)(p)
print(f"frequencies (kHz): {fax.design_frequencies(p, design)[0] / 1e3}")
print(f"risk = {value:.4e}, d risk / d {design.names} = {grad}")
step = -1e-3 * grad / jnp.linalg.norm(grad)
print(f"risk after a 1e-3 step along -grad: {objective(p + step):.4e}")
