# Getting started

## Five minutes

1. Install: `pip install alfvex` (version 1.0.1 on PyPI; the import name is `fax`).
2. Run the quickstart below. It needs no data file and prints the TAE gap, the modes inside it and a gradient.
3. Run a case from a checkout: `fax run examples/ctok_continuum.toml --plot --cache`. It writes `results.npz`, `case.json` and
   `continuum.png` to the case's output folder (relative to the case file); `--cache` keeps JAX compilations on disk, so a second run starts faster.
4. Go on to {doc}`tutorials` for the worked examples, {doc}`theory` for the equations, {doc}`validation` for what is tested and
   {doc}`api` for the derivative contract.

## Install

FAX needs Python 3.10 or newer, JAX 0.4.38 or newer and `solvax>=0.36.0`; the `vmex` extra needs Python 3.11 or newer and `vmex>=0.13.0`. FAX is on PyPI as
`alfvex` (`pip install alfvex`, extras as below) and is imported as `fax`; it will soon be renamed ALFVEX everywhere. Do not
`pip install fax`, an unrelated package. From a clone, the commands below install the development version.

```console
pip install .              # core: JAX, SOLVAX, NumPy, netCDF4
pip install ".[vmex]"      # VMEC solves and boundary gradients (Python >= 3.11)
pip install ".[plot]"      # matplotlib, for the figure examples
```

From PyPI the same extras are `pip install "alfvex[vmex]"` and `pip install "alfvex[plot]"`.

`pip install -e ".[dev,vmex]"` installs everything used by the tests and the docs. FAX switches JAX to float64 on import: Alfvén
operators are stiff, and their eigenvalues are not reliable in float32.

## Quickstart

The quickstart needs no data file and no checkout (it draws a figure only if matplotlib is installed): it runs from an installed wheel on the analytic Könies et al.
(2018) ITPA torus ({func}`fax.circular_torus`). Settings sit at the top; edit and rerun.

```{literalinclude} ../examples/quickstart.py
:language: python
```

Output: the TAE gap at $s = 0.25$ spans 63.0–69.7 kHz; the three modes nearest the target are 62.96, 63.01 and 64.63 kHz, and
the TAE ($\omega$ = 4.0610e5 rad/s, CAS3D 4.01e5, GYGLES 4.13e5) is the highest; the solve converged (residual 1e-11); and the
gradient equals the analytic $-f/2\rho$ (a frequency of the Alfvén operator scales as $\rho^{-1/2}$).

`frequency_hz(geo, harmonics, density, start, stop, ...)` is $\sqrt{\overline{\omega^2}}/2\pi$ of the eigenvalues `start:stop`,
counted in ascending $\omega^2$ among the `count` modes nearest `target_hz` (or among all modes without a target). It is the
frequency of the **mean** $\omega^2$, not the mean of the frequencies. It is 0 where that mean is not positive (`growth=True`
returns the growth rate instead) and **NaN, with a NaN gradient, unless the solve converged**.

## Command line

The command line runs two tasks from a TOML case: the continuum and the global modes.

```console
fax run case.toml            # writes <run.output>/results.npz and case.json
fax run case.toml --plot     # also continuum.png (and modes.png for eigenmodes); needs the plot extra
fax run case.toml --cache    # persists JAX compilations in ~/.cache/fax/jax (--cache DIR for another); later runs skip them
fax plot outputs/case        # figures for an existing results folder
```

**Python only:** the fast-ion drive and Landau damping, density profiles, `finite_beta`, solver options (`iters`, `tol`, the
returned `Status`), mode classification, every objective (`window_risk`, `continuum_cost`, `gap_closure`, `drive_risk`),
the VMEC-boundary design path and every derivative. The TOML `[plasma]` table is a uniform density.

`results.npz` always holds `s`, `iota` and `harmonics`. A `continuum` run adds
`frequencies_hz` (surfaces × harmonics), `gap_center_hz` and `gap_width_hz`
(the widest gap on each surface). An `eigenmodes` run adds `frequencies_hz`,
`structures` (modes × surfaces × harmonics), `mode_s` and `continuum_hz`.

## Python

The same steps from Python ({func}`fax.run` takes a TOML path or a dict):

```python
import fax

out = fax.run("examples/ctok_continuum.toml", save=False)  # dict of NumPy arrays
config = fax.load({"modes": {"n": [2], "m_max": 12}})  # defaults merged, unknown keys rejected
```

Every function below `run` is a pure JAX function with the derivative contract of {doc}`api` (first order, static shape
arguments, NaN for rejected solves); see {doc}`tutorials`.

## TOML reference

A case has four tables. Every key is optional and takes the default below
(`fax.io.DEFAULTS`); unknown tables or keys raise `ValueError`. The solver rejects a non-positive density, surfaces outside
the open interval (0, 1) and a repeated `(m, n)` harmonic. Relative `geometry.path` and `run.output` are resolved against the case file.

### `[geometry]`

| Key | Default | Meaning |
|---|---|---|
| `source` | `"circular"` | `"boozmn"` (BOOZ_XFORM file), `"wout"` (VMEC file through booz_xform_jax), `"vmec"` (VMEC input solved by vmex, then booz_xform_jax), `"circular"` ({func}`fax.circular_torus` with its defaults) |
| `path` | none | file for `boozmn`, `wout` and `vmec` |
| `mboz`, `nboz` | `8`, `8` | Boozer resolution for `wout` and `vmec` |
| `surfaces` | `[0.1, ..., 0.9]` | surfaces `s` for `wout` and `vmec` |
| `ns` | none | `vmec` only: one stage at `ns` surfaces; none runs the input's `NS_ARRAY` ladder |
| `ftol` | `1e-12` | `vmec` only: force tolerance of that stage |
| `max_iter` | `5000` | `vmec` only: iteration limit of that stage |

### `[modes]`

| Key | Default | Meaning |
|---|---|---|
| `n` | `[1]` | toroidal mode numbers (a Boozer family `n0 + k Nfp` is a list) |
| `m_max` | `8` | largest poloidal mode number |
| `m_min` | `0` | smallest poloidal mode number |

The harmonics are every `(m, n)` with `m_min <= m <= m_max` and `n` in the list
({func}`fax.mode_family`).

### `[plasma]`

| Key | Default | Meaning |
|---|---|---|
| `density` | `1e19` | ion density in m⁻³ (uniform) |
| `ion_mass` | `1.0` | ion mass in atomic mass units (a proton is 1.00727647) |

### `[run]`

| Key | Default | Meaning |
|---|---|---|
| `task` | `"continuum"` | `"continuum"` or `"eigenmodes"` |
| `output` | `"outputs"` | results folder |
| `ntheta`, `nzeta` | `32`, `32` | angle grid for the surface averages |
| `target_khz` | `0.0` | `eigenmodes`: shift-invert target; `0` solves densely for all modes |
| `count` | `8` | `eigenmodes`: number of modes kept |
