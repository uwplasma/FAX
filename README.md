# FAX

[![Tests](https://github.com/uwplasma/fax/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/uwplasma/fax/actions/workflows/tests.yml)
[![Docs](https://readthedocs.org/projects/fax/badge/?version=latest)](https://fax.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

FAX is a lean JAX code for shear-Alfvén physics in stellarators: the continuum, AE3D-form global modes and a perturbative
fast-ion drive, with first-order derivatives with respect to profiles and, through VMEX, the VMEC boundary.

<p align="center">
  <img src="docs/_static/ctok_eigenmodes_continuum.png" width="44%" alt="CTOK continuum and global modes">
  <img src="docs/_static/koenies_tae_drive.png" width="48%" alt="Könies TAE drive">
</p>

*Left: the n = 2 CTOK continuum with the six global modes nearest 384 kHz (red). Right: the passing-only drive of the Könies et al. (2018) TAE.*

## Scope

| Supported | Not supported |
|---|---|
| Low-β continuum on each Boozer surface (STELLGAP form) | Sound coupling, so no BAE or BAAE |
| AE3D-form reduced ideal-MHD global modes, signed $\omega^2$ | Continuum damping and electron damping |
| `finite_beta=True`: restricted $J_\parallel$ and pressure terms (experimental) | A full-MHD or compressional model |
| Drive of a fixed mode: zero orbit width, passing ions only | Trapped particles, finite orbit width, finite Larmor radius |

The drive is perturbative and valid up to about 200 keV in the Könies case ($\gamma/\omega \le 4\%$). It is about 0.6 of the CAS3D-K
zero-orbit-width curve; the ratio is unexplained (see the validation page). Derivatives are first order and
are NaN unless the eigensolve converged and its selection is not a tie.

## Install

Python 3.10 or newer, JAX 0.4.38 or newer, `solvax>=0.36.0`. FAX is on PyPI as **`alfvex`**; the code is imported as `fax`.
FAX will soon be renamed ALFVEX everywhere (repository, import and command line). Do not `pip install fax`: that is an unrelated package.

```bash
pip install alfvex            # core
pip install "alfvex[vmex]"    # VMEC solves and boundary gradients (Python >= 3.11)
pip install "alfvex[plot]"    # matplotlib, for the figure examples
```

From a clone, `pip install .` (with the same extras) installs the development version.

## Quickstart

This runs from an installed wheel with no data files: a continuum, a global TAE and one gradient on the analytic Könies et al. (2018) torus.

```python
import jax, jax.numpy as jnp, fax

s = (jnp.arange(200) + 0.5) / 200
geo = fax.circular_torus(s=s, major_radius=10.0, minor_radius=1.0, B0=3.0, iota=1 / (1.71 + 0.16 * s), epsilon_B=0.1)
harmonics, rho = fax.mode_family(6, 16, 5), fax.mass_density(2e19, 1.00727647)
f_hz = fax.continuum(geo, harmonics, rho)  # (surfaces, harmonics)
modes = fax.eigenmodes(geo, harmonics, rho, target_hz=6.46e4, count=3, ntheta=32, nzeta=4)
print(modes.frequencies_hz[-1], modes.status.converged)  # the TAE, 64.63 kHz
f = lambda r: fax.frequency_hz(geo, harmonics, r, 2, 3, target_hz=6.46e4, count=3, ntheta=32, nzeta=4)
print(jax.grad(f)(rho))  # equals -f / (2 rho)
```

The command line runs the continuum and global-mode tasks from a TOML case; the drive, objectives and boundary gradients are Python only.

## Examples (checkout only)

These read files in the repository or need an extra; each is a plain script with its settings at the top and no command-line parsing.

| File | Needs | Shows |
|---|---|---|
| `examples/quickstart.py` | nothing | the quickstart above, plus gaps, a dense solve and the smooth objectives |
| `examples/ctok_boozmn.py`, `ctok_continuum.toml`, `ctok_eigenmodes.toml` | `tests/data`, `plot` | CTOK from a `boozmn` file: the TOML cases, the Python API and `fax run <case> --plot` |
| `examples/qa_vmex.py`, `qa_continuum.toml` | `vmex`, `plot` | VMEC solve, Boozer transform, continuum and the gradient with respect to boundary coefficients |
| `examples/koenies_tae_drive.py` | `plot` | fast-ion drive against temperature, slowing-down ions and Landau damping |
| `examples/optimize_alfven.py` | `vmex`, `plot` | QS plus Paul's continuum cost (QH) or a classified gap-mode window (QA) |
| `examples/knyazev_qa.py` | `tests/data`, `plot` | the STELLGAP and AE3D comparison; `RESEARCH` and an archive rebuild the fixture |
| `examples/benchmark.py` | `tests/data` | equal-accuracy timings of the eigensolvers |

Every script has a `RESEARCH = False` switch at the top: the default is a fast run, `True` selects the finer settings and extra analysis.

## Evidence

| Check | Result |
|---|---|
| STELLGAP CTOK continuum (97 surfaces) | 6.3e-3 mean, 6.6e-3 p95 relative |
| AE3D v31 CTOK, six global modes | within 1.4% |
| AE3D CTOK TAE, 337.9 kHz | not reproduced. On identical input three $\iota'$ terms of AE3D's bending blocks differ from the energy (pinned block comparison; real-space quadrature FAX 0.1–0.3%, AE3D 2–37%). The $J_\parallel$ blocks still disagree, cause open. Not reported upstream |
| Knyazev et al. (2026) precise QA | continuum 3.0e-3 mean from STELLGAP. The AE3D global mode (30.31 kHz) is not a FAX eigenmode (overlap 0.18, 5.1% below); cause unresolved |
| Könies et al. (2018) TAE (n = 6) | 4.061e5 rad/s (CAS3D 4.01e5, GYGLES 4.13e5), converged to 1e-4 |
| Drive, 100/200/300/400 keV | 4.29/16.39/32.72/64.48 ×10³ s⁻¹; 0.58–0.60 of CAS3D-K ZOW at 82–191 keV (content unmatched, not a validation) |
| Gradients | fixed-geometry AD vs finite differences to about 1e-9 at best; LP-QA gap-mode shift of 4 kHz predicted to 0.05 kHz, resolution budget 0.13 kHz; the $m = 1$ gauge leaves a 1e-2 floor on `zs11`, not closed |

The [validation page](https://fax.readthedocs.io/en/latest/validation.html) has every gate, its test and what is not done.

## Ecosystem

[SOLVAX](https://github.com/uwplasma/SOLVAX) provides the generalized eigensolver and its implicit derivatives;
[VMEX](https://github.com/uwplasma/vmex) the equilibrium and its implicit derivative;
[booz_xform_jax](https://github.com/uwplasma/booz_xform_jax) the Boozer transform. ESSOS orbit transport of FAX modes is planned, not implemented.

## Documentation and testing

[fax.readthedocs.io](https://fax.readthedocs.io): getting started and TOML reference, tutorials, theory, validation, API with its derivative contract.

```bash
pip install -e ".[dev,vmex]"
python -m pytest -q -m 'not scheduled'
ruff check . && ruff format --check . && sphinx-build -W -b html docs docs/_build/html
```

## Citing and license

Cite the repository through [`CITATION.cff`](CITATION.cff) and the papers whose models FAX implements ([references](https://fax.readthedocs.io/en/latest/references.html)). MIT license.
