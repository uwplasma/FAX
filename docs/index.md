# FAX

FAX is a lean JAX code for shear-Alfvén physics in stellarators. From a Boozer-coordinate equilibrium it computes

- the **shear-Alfvén continuum** on each flux surface and its TAE, EAE, MAE and HAE gaps;
- **global Alfvén eigenmodes** from the reduced ideal-MHD weak form of AE3D, with radial finite elements and a
  block-tridiagonal shift-invert solver ([SOLVAX](https://github.com/uwplasma/SOLVAX));
- a **perturbative fast-ion drive** (power transfer over mode energy) and bulk-ion Landau damping;
- **first-order derivatives** of these with respect to profiles and, through
  [VMEX](https://github.com/uwplasma/vmex) and [booz_xform_jax](https://github.com/uwplasma/booz_xform_jax), the VMEC plasma boundary.

## Scope

| Supported | Not supported |
|---|---|
| Low-β continuum (STELLGAP form) | Sound coupling, so no BAE or BAAE |
| AE3D-form reduced global modes, signed $\omega^2$ | Continuum damping and electron damping |
| `finite_beta=True`: restricted $J_\parallel$ and pressure terms, experimental | A full-MHD or compressional model |
| Drive of a fixed mode: zero orbit width, passing ions | Trapped particles, finite orbit width, finite Larmor radius |

The drive is about 0.6 of the CAS3D-K zero-orbit-width curve of Könies et al. (2018), plausibly the missing trapped share (not
established), and is perturbative: $\gamma/\omega \le 4\%$ up to 200 keV in that case. Derivatives are first order and are NaN
unless the eigensolve converged and its selection is not a tie ({doc}`api`). {doc}`validation` lists every gate and what is open.

```console
pip install alfvex                  # PyPI name; imported as fax (FAX will soon be renamed ALFVEX)
python examples/quickstart.py       # continuum, global TAE and a gradient, no data files
fax run examples/ctok_continuum.toml --plot
```

```{image} _static/ctok_eigenmodes_continuum.png
:alt: CTOK n = 2 continuum with the six global modes nearest 384 kHz
:width: 60%
```

| Page | Contents |
|---|---|
| {doc}`getting-started` | install, the standalone quickstart, command line, every TOML key |
| {doc}`tutorials` | the examples, step by step, with their source and figures |
| {doc}`theory` | derivations: Boozer metric, continuum, gaps, AE3D weak form, solver, gradients, drive, objectives |
| {doc}`validation` | every gate with its numbers, and what is not done |
| {doc}`api` | derivative contract of every function, equation to function to test map, API reference |
| {doc}`development` | tests, contributing, citing |
| {doc}`references` | papers FAX implements or compares with |

```{toctree}
:hidden:
:maxdepth: 2

getting-started
tutorials
theory
validation
api
development
references
```
