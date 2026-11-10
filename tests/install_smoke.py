"""Installed-wheel smoke test: run from a directory without the checkout (``python install_smoke.py``).

Uses only analytic geometry shipped in the package: the uniform-field continuum, a global mode and an eigenvalue gradient.
"""

from importlib.metadata import version

import jax
import jax.numpy as jnp
import numpy as np

import fax
from fax.continuum import MU0

assert "site-packages" in fax.__file__, fax.__file__  # the wheel, not a checkout
assert version("alfvex") == fax.__version__
geo, h, rho = fax.circular_torus(iota=jnp.linspace(0.3, 0.6, 24)), fax.mode_family((1, 2), 4), fax.mass_density(1e19, 2.0)
k = (h[:, 1] - h[:, 0] * geo.iota[:, None]) / 5.5  # k_par = (n - iota m) / R
exact = jnp.sort(jnp.abs(k) * 2.7 / jnp.sqrt(MU0 * rho) / (2 * jnp.pi), axis=1)  # k_par v_A / 2 pi
np.testing.assert_allclose(fax.continuum(geo, h, rho), exact, rtol=1e-9)
modes = fax.eigenmodes(geo, h, rho, target_hz=float(exact.mean()), count=3)
assert bool(modes.status.converged) and bool(jnp.all(modes.frequencies_hz > 0))
f = lambda r: fax.frequency_hz(geo, h, r, 0, 1)  # noqa: E731
np.testing.assert_allclose(jax.grad(f)(rho), -f(rho) / (2 * rho), rtol=1e-6)  # f ~ rho^-1/2
print(f"fax {fax.__version__}: continuum, eigenmodes and gradient OK")
