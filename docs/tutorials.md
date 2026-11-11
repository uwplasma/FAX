# Tutorials

The scripts in `examples/` walk from an analytic continuum to boundary gradients and a fast-ion drive, in the order the physics builds up
in {doc}`theory`. Each has editable settings at the top and no command-line parsing. Run them from the repository root. Every section
below says what it teaches, the command, the output to expect and the run time; the times are one laptop CPU core and include JAX
compilation. Examples that draw figures need the plot extra (`pip install ".[plot]"`); the VMEC ones need the vmex extra
(`pip install ".[vmex]"`, Python 3.11 or newer). The two commands `fax run` and `fax plot` are the package command line and run the
continuum and global-mode tasks only ({doc}`getting-started`).

| Example | Needs | Section |
|---|---|---|
| `examples/quickstart.py` | nothing | 1 |
| `examples/ctok_continuum.toml`, `examples/ctok_eigenmodes.toml` | checkout (`tests/data`) | 2, 3 |
| `examples/qa_continuum.toml` | `vmex` | 4 |
| `examples/boundary_gradient.py` | `vmex` | 5 |
| `examples/koenies_tae_drive.py` | `plot` | 6 |
| `examples/optimize_alfven.py` | `vmex`, `plot` | 7 |
| `examples/knyazev_qa.py` | Zenodo archive (1.1 GB) | 8 |
| `examples/benchmark.py` | checkout; timings, see {doc}`validation` | |

## 1. A continuum, a global mode and a gradient on analytic geometry

You will learn the three basic calls: continuum, targeted eigenmode, derivative. Command: `python examples/quickstart.py` (about 15 s;
it needs no data and is also listed in {doc}`getting-started`). It prints the continuum summary, the TAE frequency and the solver `Status`. The script builds the Könies et al. (2018) ITPA torus,
computes its continuum, finds the TAE (4.0610e5 rad/s) by shift-invert with its solver `Status`, and differentiates the TAE
frequency with respect to the mass density.

## 2. A continuum from a `boozmn` file

You will learn how to read a continuum from a `boozmn` file and what the gaps in it mean (sections 2 and 3 of {doc}`theory`). The input
`examples/ctok_continuum.toml` is the STELLGAP CTOK test problem: a circular tokamak, `n = 2`, `m = 0..12`, deuterium at
10¹⁹ m⁻³.

```{literalinclude} ../examples/ctok_continuum.toml
```

```console
fax run examples/ctok_continuum.toml --plot      # about 4 s
```

```{image} _static/ctok_continuum.png
:alt: CTOK n = 2 shear-Alfvén continuum
:width: 60%
```

Each dot is one eigenvalue of the surface problem {eq}`continuum-galerkin` on one of the 99 half-mesh surfaces. Branches with
neighbouring `m` repel where they would cross, which opens the TAE gaps (`m, m + 1`) and, higher up, the EAE gaps. The same
continuum in Python:

```python
import fax

geo = fax.read_boozmn("tests/data/boozmn_ctok.nc")
f_hz = fax.continuum(geo, fax.mode_family(2, 12), fax.mass_density(1e19, 2.0))
center, width = fax.gaps(f_hz)  # widest gap on each surface, full width edge to edge
iota, w_hat = fax.gap_location(2, 3, 1)  # cylindrical (3,2)-(4,2) TAE crossing: iota, omega R / v_A
```

## 3. Global eigenmodes near a target frequency

You will learn how to target a frequency and how to tell a continuum-like mode from a gap mode (theory sections 4 and 5). The input
`examples/ctok_eigenmodes.toml` sets `task = "eigenmodes"`, `target_khz = 384` and `count = 6`, so SOLVAX's block-tridiagonal
shift-invert returns the six eigenpairs of `F y = ω² G y` nearest 384 kHz.

```console
fax run examples/ctok_eigenmodes.toml --plot     # about 6 s
```

```{image} _static/ctok_modes.png
:alt: radial structure of the four lowest CTOK modes near 384 kHz
```

The modes found here (377.6, 378.6, 386.7, 388.1, 389.8, 390.9 kHz) sit inside the continuum, so most of them are
continuum-like: their harmonics peak where `ω` crosses a continuum branch (red lines in the figure on the {doc}`index`).
This case runs on the 99 half-mesh surfaces of the file; the AE3D comparison in {doc}`validation` uses AE3D's own full mesh, and
there the AE3D TAE at 337.9 kHz is not reproduced (a pinned block comparison traces it to three $\iota'$ terms of AE3D v31).

```python
modes = fax.eigenmodes(geo, fax.mode_family(2, 12), fax.mass_density(1e19, 2.0), target_hz=384e3, count=6, ntheta=32, nzeta=4)
modes.frequencies_hz, modes.structures.shape, modes.status.converged  # (6,), (6, 99, 13), True
```

The solver `Status` reports `converged`, the worst residual and the selection `gap`. Omit `target_hz` (or set `target_khz = 0`) for the dense
solve of every mode; it is `O((N H)^3)` and only sensible for small problems.

## 4. A stellarator from a VMEC input

You will learn how a VMEC input becomes Boozer geometry inside FAX. The input `examples/qa_continuum.toml` solves the low-resolution precise-QA input (Landreman & Paul 2022) with VMEX at `ns = 16`, transforms
nine surfaces to Boozer coordinates with booz_xform_jax (`mboz = nboz = 8`) and computes the continuum of the `n = 0, ±2` family
(`Nfp = 2`, so these couple).

```console
fax run examples/qa_continuum.toml --plot        # about 1 min
```

```{image} _static/qa_continuum.png
:alt: precise-QA continuum for the n = 0, 2, -2 family
:width: 60%
```

`iota` falls from 0.422 to 0.417 across these surfaces. In Python the geometry step is
`fax.from_vmec_input(path, ns=16, mboz=8, nboz=8, surfaces=...)`; `fax.from_wout` does the same from a `wout` file or in-memory wout.

## 5. A gradient with respect to the plasma boundary

You will learn how to differentiate a spectral objective with respect to the plasma boundary ({ref}`vmex-gradients`). Command: `python examples/boundary_gradient.py`, a few minutes. The script differentiates a smooth frequency-window objective, {func}`fax.window_risk`, of the three lowest
global frequencies of precise QA with respect to the boundary coefficients `rc01` (`RBC(n=1, m=0)`) and `zs01` (`ZBS(n=1, m=0)`).

```{literalinclude} ../examples/boundary_gradient.py
:language: python
```

```text
frequencies (kHz): [294.37821076 294.71453201 295.2077278 ]
risk = 7.1758e-02, d risk / d ('rc01', 'zs01') = [-0.62269764  0.54587794]
risk after a 1e-3 step along -grad: 7.0927e-02
```

{func}`fax.prepare_design` builds the static part (VMEX configuration, Boozer tables, FE nodes) once;
{func}`fax.design_frequencies` is then a traceable function of the boundary offsets, differentiated through the VMEX implicit
equilibrium ({ref}`vmex-gradients`) and first order only. It returns the solver `Status`, and the frequencies and their tangents are
NaN unless the solve converged. The run takes a few minutes on a CPU (it compiles the equilibrium adjoint). This is a tiny model
(three harmonics, 8 × 8 angles, 24 radial nodes), and its $n = 0$ modes are the continuum-like ones of section 7, so it demonstrates the
gradient chain, not a gap-mode objective; {doc}`validation` lists what the gradient is checked against and the open $m = 1$ gauge floor.

## 6. Fast-ion drive of the Könies ITPA TAE

You will learn how to compute a perturbative growth rate and how it scales with fast-ion temperature (theory section 8). The script `examples/koenies_tae_drive.py` (needs the plot extra) evaluates the perturbative growth rate {func}`fax.growth_rate` of a model
TAE (harmonics `m = 10, 11`, `n = 6`, Gaussian radial envelope centred on `q = 1.75`, `ω = 4.0e5` rad/s) on the Boozer form of the
ad hoc equilibrium ({func}`fax.circular_torus`), driven by the Maxwellian fast-ion profile of Könies et al. (2018), for
temperatures from 50 to 500 keV.

```{literalinclude} ../examples/koenies_tae_drive.py
:language: python
:start-at: from pathlib
```

```console
python examples/koenies_tae_drive.py             # about 10 s, writes koenies_tae_drive.png
```

```{image} _static/koenies_tae_drive.png
:alt: FAX passing-only growth rate vs fast-ion temperature, with the CAS3D-K ZOW points of Könies 2018
:width: 60%
```

The printed values at 100, 200, 300 and 400 keV are 4.29, 16.39, 32.72 and 64.48 ×10³ s⁻¹. The dots are the CAS3D-K (ZOW) curve of
Könies et al. (2018, fig. 1a): FAX is a constant 0.58–0.60 of it from 82 to 191 keV, unexplained (the paper
does not say what the curve contains, so the content is unmatched and this is not a validation). The pooled ZLR band mixes codes
with different physics and is not used. FAX is zero-orbit-width, passing-only and has no damping; $\gamma/\omega$ is 4% at 200 keV
and 8% and 16% at 300 and 400 keV, where the frozen-mode picture fails ({doc}`validation`). Every argument of `growth_rate`
(density profile, temperature, `ω`, the mode, the geometry) is differentiable. The same call takes a stellarator
{class}`fax.Geometry` and a mode from {func}`fax.eigenmodes`; pass `helical` for the helical sidebands and a full `nzeta`.

## 7. Optimizing the Alfvén spectrum with quasisymmetry (exploratory)

You will learn how the pieces combine into an optimization with an exact Jacobian. The script `examples/optimize_alfven.py` (needs the vmex and plot extras) runs `scipy.optimize.least_squares` on VMEC boundary coefficients
with residual rows (aspect ratio, $\sqrt{f_{QS}}$, $\iota$ floor, Alfvén term), each scaled to about one at the start, and an exact
Jacobian: one VMEX + FAX forward pass and one adjoint per row ({ref}`continuum-objective`). Edit the settings and run it:

```{literalinclude} ../examples/optimize_alfven.py
:language: python
:start-at: CASE =
:end-at: OUT =
```

- **QH** (`CASE = "qh"`; Paul et al. 2025, eq. 7.1): 34 coefficients with $m \le 3$, $|n| \le 2$; rows $10(A/A_0 - 1)$,
  $0.1\sqrt{f_{QS}/f_{QS,0}}$, $10\sqrt{f_{\iota}}$ with $|\iota| \ge 1.03$, and $\sqrt{f_\mathrm{cont}/f_{\mathrm{cont},0}}$. It lowers
  Paul's geometric proxy $f_\mathrm{cont}$ by 26% (and narrows the high-frequency gaps by up to 29%) at QS ×4.3; no independent
  spectrum or transport check has been made, so this is a proxy reduction.
- **QA, gap modes** (`CASE = "qa"`, `MODES_KIND = "gap"`, about 8 min): two global modes pushed out of a window at fixed aspect ratio
  and QS (24 FE nodes). It takes the two $n = 1$ modes nearest 103 kHz ($m = 0$–5, $n \in \{-1, 1, 3\}$), which
  {func}`fax.classify_modes` finds to be isolated gap modes in the 96–142 kHz gap (102.1 and 103.9 kHz); the window is
  $[f_\mathrm{min} - s, f_\mathrm{max} + 20 s]$ with $s$ = 0.34% of the mean frequency and the Alfvén row is
  $30\sqrt{\text{window\_risk}}$. The script prints the classification of the selected modes before and after, and the first-order
  prediction $J\,\delta p$ against the re-solved change: a 4 kHz shift predicted to 0.05 kHz, inside a resolution budget of 0.13 kHz.
- **QA, continuum minima** (`MODES_KIND = "continuum"`): the three lowest $n = 0$ modes ($m = 1$–3), which the classifier finds to
  be continuum-like: each sits on the continuum minimum at its own radius, so moving them moves the branch minimum
  ($\propto |\iota m - n|$), not a mode of the geometry. Kept only as a test of the gradient chain; the earlier reading of this run
  as "modes leave the window" was a statement about that branch.
- **QA, drive** (`DRIVE = True`): the Alfvén row is $\text{drive\_risk}/\text{drive\_risk}_0$ of the growth rates of the four $n = 1$
  modes ($m = 0$–5, $n \in \{-1, 1, 3\}$) nearest 104 kHz, driven by 150 keV Maxwellian deuterons with
  $n_f = 10^{15}(1 - s)^2$ m⁻³ ({func}`fax.geometry_growth_rates`, {func}`fax.drive_risk`). The 4-coefficient run lowers
  $\gamma_\mathrm{max}$ by 4.4% (8500 to 8123 s⁻¹) at QS 1.1e-6 to 1.6e-6; that change is not bounded by an error budget, so it is
  not claimed as a reduction.

None of these runs shows that the optimized device has fewer Alfvén eigenmodes or lower fast-ion transport: the QA runs move
selected modes, the QH run a proxy, and the drive is passing-only. The script prints the objective history, aspect, $\iota$, QS,
$f_\mathrm{cont}$ and the largest gaps above the cut, and writes the figures below; {doc}`validation` lists the numbers.

```{image} _static/optimize_alfven_qh.png
:alt: LP-QH continuum before and after continuum optimization, and the objective history
```

```{image} _static/optimize_alfven_qa.png
:alt: LP-QA continuum branches, the modes before and after moving them out of a window, and the objective history
```

## 8. The independent 3-D comparison (checkout and archive only)

You will learn how FAX was compared with an independent 3-D code. The script `examples/knyazev_qa.py` rebuilds the offline fixture `tests/data/knyazev_qa.npz` from the Zenodo archive of Knyazev et al. (2026)
(DOI 10.5281/zenodo.21299907, 1.1 GB, CC-BY-4.0; the archive is data and nothing in it is executed) and reproduces the precise-QA
comparison of {doc}`validation` (about 15 minutes and 8 GB). Set the switches and the archive path at the top:

```{literalinclude} ../examples/knyazev_qa.py
:language: python
:start-at: ARCHIVE =
:end-at: REBUILD, STUDY
```

The continuum agrees with STELLGAP (3.0e-3 mean). The AE3D global mode (30.31 kHz) is not a FAX eigenmode (best overlap 0.18; its energy
in FAX's operator is 5.1% below its AE3D eigenvalue); the cause is unresolved. `benchmark.py` is described in {doc}`validation`.
