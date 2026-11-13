# FAX plan

Revised 2026-11-10 against main `d0afdf4` (items checked against code and tests; literature and other codes re-surveyed). Previously revised 2026-10-23 after an independent review of snapshot `2f7228f` (main before #16/#17); the review and its evidence bundle are kept outside the repository. This replaces the phase plan P0–P7 and its logbook. Both are still in the git history of this file (last at `2f7228f`).

## Goal

A small, validated, end-to-end differentiable JAX code for **stable shear-Alfvén spectra and their energetic-particle response in 3-D stellarators**, usable inside VMEX boundary optimization and installable with `pip install alfvex` (PyPI name; FAX will be renamed ALFVEX). The scientific targets:

1. **Methods.** Accurate boundary-to-spectrum sensitivities for 3-D Alfvén modes and spectral observables, including internal crossings, at a cost comparable to one forward solve.
2. **Physics.** Test whether a cheap differentiated spectral or kinetic objective reduces independently computed wave-induced fast-ion transport without degrading QS or confinement.

Novelty boundary from the literature: continuum-gap optimization (Paul et al. 2025, JPP 91 E101; five reactor-scale continua in Hyder, Paul and Knyazev, arXiv:2611.07182), differentiable finite-n ideal MHD (AGNI, Gaur et al. 2026) and computed-mode orbit transport (Knyazev et al. 2026; FIRM3D) already exist. FAX's contribution is qualified, differentiated *stable* spectra plus kinetic response in a single optimization loop.

## Ownership

| Code | Owns | FAX consumes |
|---|---|---|
| **FAX** | Boozer-geometry Alfvén physics: continuum, global weak form, boundary conditions, kinetic power, mode classification, physical objectives | — |
| **SOLVAX** | Block factor/solve, generalized targeted eigensolver with status, implicit eigenpair/projector derivatives | solver results plus residuals and selection diagnostics |
| **VMEX** | Equilibrium and its implicit derivative | public state → Boozer-input tables |
| **booz_xform_jax** | Boozer transform, mode conventions, `boozmn` reader | public array schema (replaces FAX's orientation heuristic) |
| **ESSOS** | Guiding-centre orbits, prescribed-wave transport (#177) | FAX exports mode potentials with physical amplitude |
| FIRM3D, AE3D, STELLGAP | External comparators | pinned fixtures only |
| DKX / NEOPAX | Profiles, bootstrap current, E_r (optional) | arrays only; never a mandatory install |

## State

**Implemented** (PRs #4–#15):
- STELLGAP-weighted continuum (CTOK p95 6.6e-3).
- AE3D-form global operator: linear FE, block-tridiagonal shift-invert, Hellmann–Feynman gradients.
- Optional current and pressure terms.
- Passing zero-orbit-width drive in Boozer geometry.
- VMEX/booz_xform_jax design path.
- Paul continuum objective and optimization example.
- MyST docs.
- 1,547 source lines in 8 modules, 78 test functions; statement and branch coverage 100/100 on the PR tier.
- Since then: FFT assembly (#38), SOLVAX 0.36 and input checks (#39), #9/#17 physical-potential vorticity with r^m axis interpolation, trims (#40), Boozer geometry from vmex `boozer_spectrum_state` (#41). Published on PyPI as `alfvex` 1.0.1.


**Corrections the review established, all reproduced on current main.** Earlier claims overstated these points:

| Claim | Status |
|---|---|
| Fu–Van Dam drive test passes | **Vacuous**: the four growth rates are exactly 0 (no toroidicity in that geometry). |
| AE3D CTOK TAE is an "AE3D sign error" | Supported by real-space quadrature, #9's index-swap manufactured test and the Knyazev QA archive (30.31 kHz AE3D vs 27.57 physical), but **not established** until a pinned AE3D build and its blocks are compared term by term. Say "unresolved discrepancy, evidence points to AE3D shear cross terms". |
| Docs: kink term ≡ AE3D's ∇(J/B) form | Integration by parts omits `−∫uσ(∇×b)·∇φ*`. The code assembles the direct form, so this is a derivation gap, not a known matrix error. |
| 99% coverage | Statement coverage 93.5% on the fast suite and 99% with the slow suite; **branch coverage 85% (51/60)**. |
| Optimization "suppresses AEs" | Exploratory. The QA example moves continuum-like lowest eigenvalues; the QH result is a proxy reduction; the 4.4% drive cut is not bounded by error budgets. |
| Robust numerics | Block Thomas gives NaN on the invertible matrix `[[0,1],[1,0]]`; negative ω² is clipped to 0 Hz with a NaN gradient; shift-invert has a fixed 40 iterations and no status; `_d_ds` is wrong on irregular grids (0.9 vs 0.4); the reader transposes a 20×3 array; `circular_torus(B0)` is not traceable; the cluster mean has a NaN second derivative at degeneracy. |
| Gap width | Full width is `2 f0 |c|` for the complex coefficient; one convention must govern theory, tables and plots. |

## Rules

1. One physical or numerical claim per PR. Keep model corrections separate from solver migration.
2. Every physical claim needs an independent analytic or published comparison with matched assumptions. Self-regression and AD-vs-FD do not count.
3. Every eigensolve returns residual, `G`-orthogonality, iteration count and selection status. Gradients from unconverged or non-isolated modes are rejected.
4. Signed ω² is preserved. Instability is reported, never clipped.
5. Derivative support is stated per route (inputs, static choices, transforms, order, isolation). First order is promised; higher orders only where tested.
6. Statement **and** branch coverage ≥ 95% each, without excluding physics, adapters or error paths.
7. Fewer lines only by removing duplication. Never remove an equation, a failure check or an independent test to save lines. Module boundaries stay: geometry, continuum, eigen, drive, objectives, io, plot.
8. Python ≥ 3.10 stays supported. The core JAX floor moves to the SOLVAX floor (0.4.38, which still supports 3.10). The vmex extra needs ≥ 3.11 and its own floor. Both stacks are tested in CI.
9. Speed is measured at equal accuracy: cold compile, synchronized warm run, gradient and peak memory, with device and `(N, H)` recorded.
10. Scientific examples are top-level Python with editable settings. Command parsing lives in the CLI only.

## Work packages

A and B start now. C runs in parallel with A/B. D needs C. E needs B and D. F can start on frozen geometry once B is done.

### A. Repair the contract (FAX)
- [x] Replace the vacuous Fu–Van Dam test with a weak-toroidicity limit: nonzero growth rate, absolute coefficient, sign and asymptotic scaling.
- [x] Reject aliased angle grids (`ntheta` below what the harmonic products need), invalid density/B/geometry and singular mass. Tests assert finiteness and residuals, not shapes. Singular mass: `global_matrices` rejects duplicate `(m, n)` harmonics host-side (no factorization; `test_global_matrices_validate_density_and_surfaces`).
- [x] Keep signed `omega2`. Define a stable frequency and an ideal growth rate separately, with a status flag.
- [x] Shift-invert reports convergence, residuals and orthogonality. Detect Schur-pivot failure and fall back to a pivoted or dense reference (moves to SOLVAX in D).
- [x] Second-order nonuniform `_d_ds` with a convergence test on irregular grids.
- [~] Dimension-aware `boozmn` reading through booz_xform_jax's reader, behind a lazy optional import. FAX's own reader follows the named dimensions (`test_read_boozmn_orientation_follows_named_dimensions`); the vmex path no longer needs a file.
- [x] Traceable geometry constructors: host validation separated from the jnp construction.
- [x] Gap width uses one full-width convention (two-mode test `2 f0 |c|`). Fix docs and example tables.
- [x] Slowing-down distribution: integrated-density check, correct the docstring, reject unknown `kind`.
- [x] Parameter names for multi-digit modes (e.g. `("rc", m, n)` tuples). Asymmetric-geometry fixtures with nonzero `rmns`/`zmnc`.
- [x] Reach statement and branch coverage ≥ 95% each.

**Gate:** every reproduced defect has a test that fails at `2f7228f` and passes after its fix; CTOK, Könies and drive references stay within their stated budgets.

### B. Qualify the reduced model (FAX)
- [x] Convention table: `docs/theory.md` section 1. Pinned by `test_conventions_and_geometry_diagnostics_on_ctok`: orientation ($\sqrt g < 0$), $G$, $I$, $\iota$, $B^\zeta$, $k_\parallel = (\iota m - n)B/(G + \iota I)$ in real space and in the assembled mode vectors, and $\psi' = \Phi_\mathrm{edge}/2\pi$.
- [x] Re-derived from $\delta W$ (`docs/theory.md` section 4, `test_kink_forms_differ_by_the_curl_b_term`): the direct kink form is the energy term at the retained order $O(\delta^2)$; it differs from the $\nabla\sigma$ form by $-\int u\sigma(\nabla\times\mathbf b)\cdot\nabla\phi^*$ (exact, $O(\delta^3)$), the complete $\mathbf Q_\perp$ gives $K_{\nabla\sigma} - \int\sigma^2|u|^2$ together with a bending correction, 1% of the bending energy on CTOK. No code change; Könies with $p$ stays 3.948e5 rad/s, CTOK `finite_beta` keeps its near-axis instability. Docs equivalence statement corrected, boundary and axis terms stated.
- [x] Jacobian and force-balance residuals: `jacobian_check`, `force_balance_residual`. CTOK $\psi'$ 0.4% and spread 5e-5 for $s \ge 0.05$ (first surface 15% low: radial finite difference of $R \sim \sqrt s$), $B_s$ vs embedding 1e-5; ITPA $\psi'$ 0.3%, spread 3e-4, $B_s$ 8e-5, $\mu_0p'$ vs `AM` 0.15%.
- [x] CTOK discrepancy: AE3D v31 pinned (commit 383d327, gfortran 15.2.0, unmodified build reproduces the shipped answers; instrumented patch and fixture in `tests/data`). Blocks compared separately: inertia and base bending agree (1e-9 to 2e-6); three $\iota'$ terms of AE3D's bending blocks differ from the energy (antisymmetric $f'f$ sign, missing $a'\iota'$ cross term, $g^{s\theta}\iota'$ sign); correcting them in AE3D's own Fortran moves its localized gap modes from 240.9, 244.9, 338.2, 355.6 to 258.9, 264.9, 266.4 kHz (FAX 268.5, 270.1, 279.1). Quadrature on AE3D's input: FAX 0.1-0.3%, AE3D 2-37%. The $J_\parallel$ blocks still disagree (cause open; the TAE does not depend on it). Docs and README state this; not reported upstream.
- [x] `finite_beta=True` labelled restricted/experimental in the docstring, theory, validation and README. No sound or full-MHD claim.

**Gate:** equations, code, quadrature and one matched external calculation describe the same model.

### C. SOLVAX contracts (SOLVAX)
- [x] `generalized_eigenpairs(F, G, target, count, tol, max_iter)` on block-tridiagonal Hermitian pencils with moving positive mass. It returns values, `G`-normalized vectors, residuals, orthogonality, iterations, status and selection diagnostics.
- [x] Factor/solve of `F − σG` with explicit transpose/adjoint semantics (the current scalar-identity `shift` is not enough), multiple right-hand sides, bad-pivot fallback, and dense references for small problems.
- [x] Implicit first-order derivatives: eigenvalue `y†(dF−λdG)y`; eigenvector via a bordered solve with normalization and phase conditions; isolated-cluster projector `P = Y Y† G` with separation checks.
- [x] Support table: eager/JIT/JVP/VJP, first vs higher order, CPU/GPU, precision.

**Gate:** qualified against dense references, including singular and indefinite cases and clusters.

### D. Migrate numerics and geometry (FAX + SOLVAX, VMEX, booz_xform_jax)
- [x] Replace FAX's `block_factor`/`block_solve`/`shift_invert`/`targeted_eigenvalues` with the SOLVAX contract. FAX keeps a thin wrapper only. Results stay within error budgets. Done; numbers and tests in `docs/validation.md`.
- [x] Use the public VMEX state → Boozer bridge. Drop the half-row-1 workaround once a VMEX release contains #631. Done in #41 (vmex ≥ 0.13.0).
- [x] Continuum: triangular solves instead of the explicit Cholesky inverse. Triangular solves done. Sharing surface grids: not worth it. Under `jit` XLA already merges the identical grids (continuum + eigenmodes, 4.5 ms total); only the eager path repeats the 36 ms grid (78 ms for both, 1 grid shared would be ~40 ms), and `growth_rate` defaults to a different 64x32 grid.

**Gate:** no duplicated solver code; same physical outputs; pinned compatible releases tested.

### E. Spectral objectives and boundary gradients (FAX + SOLVAX)
- [x] Classify modes as isolated/gap vs continuum-like by radial refinement and localization. Objectives act on classified modes or physical windows only. Done; numbers and tests in `docs/validation.md`.
- [x] Replace the hard `argmax` in `gap_closure` and the hard ι mask in `continuum_cost` with smooth or certified-isolated forms. Done; numbers and tests in `docs/validation.md`.
- [~] Gradient checks with step-size plateaus. Done except the frozen-path reference: fixed-geometry (asymmetric tangent, decade sweep, floor 1e-9) and re-solved (flat gauge floors 1e-4 to 1e-2) are tested separately; ns, mboz/nboz, angles, harmonics and nodes are swept one at a time (budget in `docs/validation.md`); vmex's `frozen_path_directional_fd` diverged on `zs11` and gave a non-finite state on `zs01` of LP-QA lowres, so the m = 1 gauge is bounded, not closed. Original item: Fixed-geometry and re-solved-equilibrium derivatives are tested separately, including asymmetric and `m = 1` directions (frozen-path FD in the scheduled tier). Refine VMEX, Boozer, angle, harmonic and radial resolutions, not radial only.

**Gate:** one real 3-D boundary objective with a first-order contract and a stated numerical error budget. Met for the LP-QA n = 1 gap modes: 4 kHz shift predicted to 0.05 kHz, budget 0.13 kHz.

### F. Independent 3-D and kinetic validation (FAX)
- [x] Reduce one QA or QH case of the Knyazev et al. 2026 dataset (DOI 10.5281/zenodo.21299907; AE3D kink term off) to an offline fixture with a provenance manifest. Compare the continuum, isolated frequencies, eigenfunctions/subspaces and convergence. Done for precise QA (`tests/data/knyazev_qa.npz`, `PROVENANCE.md`, `examples/knyazev_qa.py`): continuum within 3e-3 of STELLGAP; AE3D has no isolated eigenvalue (dense band), so the band and the archived global mode are compared: FAX is 5.1% below its eigenvalue at matched settings with no single-mode match (overlap 0.18, 27.6 kHz). Unresolved discrepancy; see `docs/validation.md`.
- [x] Derive the 3-D passing source from a guiding-centre Lagrangian or linear drift-kinetic equation; the toroidal-momentum argument alone is insufficient. Test axisymmetric, QS-relabelled, ±passing, negative-ι and n-reversal limits. Done (theory §8): each resonance `(j, k)` carries `Q F = ∂_E F + j ∂_s F / (ω Z e ψ')` (orbit's poloidal harmonic, mean-flux label), not the mode's `n_h/ι`; the old form failed helical relabelling by 17–20%. Könies 4.33/16.52/32.94/64.85 → 4.26/16.27/32.48/64.02 ×10³ s⁻¹. Open: the `O(ω/nΩ)` labelling (F of mean flux vs of the P_ζ surface, 1.5% at n = 6, 23% at CTOK n = 2), I/G and ε² terms of J_ϑ.
- [x] Make the passing-boundary, birth-edge and small-divisor regularizations explicit parameters with convergence demonstrated. Done; numbers and tests in `docs/validation.md`.
- [ ] Matched ITPA ladder: analytic limit → fixed-mode ZOW with matched passing/trapped content → FOW/ZLR → FLR → damping. No fitting to the pooled band. Report `|γ|/ω` and frozen-mode validity. Rung 1 done (Fu–Van Dam closed form, flat and with a density gradient). Rung 2 is **unmatched**: FAX (passing only) is 0.58–0.60 of the Könies CAS3D-K (ZOW) curve at 82–191 keV, whose trapped content the paper does not state; `|γ|/ω` ≤ 4% to 200 keV, 16% at 400 keV. Open: a passing-only or trapped-resolved ZOW reference, then FOW/ZLR, FLR, damping. Könies et al. (2018) uses a full Maxwellian with f0(s), not f0(P_φ) (which halves γ in NOVA-K), and gives growth rates only in figures, so the 0.58–0.60 ratio needs a trapped contribution or a digitised curve with that caveat; arXiv:2608.06764 (TEK) re-examines the same case but gives linear growth rates only in a figure and no ZOW or trapped/passing split; no source found separates trapped from passing ZOW drive for this case. Because the FAX/CAS3D-K ratio is flat over 82–191 keV while a missing trapped share (estimated small there, since trapped resonance needs v ≳ 3 v_A) would vary with T, check FAX's drive normalisation and ∂f/∂E vs ω* weighting before attributing the gap to trapped ions. A term-by-term audit of `drive.py` (energy and power halves, both v∥ signs, velocity Jacobian, drift weight, ω*, s = r² factor, deuteron mass) found no missing factor. The likely locus is the test's mode: `tests/test_drive.py` uses a Gaussian proxy in (10, 6) and (11, 6) only, so the next step is the drive of the actual `eigenmodes` n = 6 TAE (m = 5–16), then the n_f normalisation (axis vs peak) of Könies eq. 1.

**Gate:** one independent 3-D spectrum and one matched nonzero kinetic response.

### G. Release 1.0 (FAX)
- [x] Coverage ≥ 95/95. Three test tiers: PR (analytic and contracts), scheduled (VMEX gradients, 3-D references, min/max dependency stacks), release (external comparisons, CPU/GPU). Done: markers `slow`/`scheduled` only; PR matrix and coverage job (`-m 'not scheduled'`, 100/100); `scheduled.yml` adds the minimum (3.10, JAX 0.4.38, SOLVAX 0.36.0), minimum-vmex (3.11, vmex 0.13.0, JAX 0.9.2) and latest (3.13) stacks; the release checklist is in `docs/development.md`.
- [x] One fixture provenance manifest (identifier, hash, model switches, units, license, command, tolerance). Add `CITATION.cff`. Done; numbers and tests in `docs/validation.md`.
- [x] README of 600–900 words with scope and unsupported physics stated next to supported physics. Fix the AE3D and drive-band wording. The installed-wheel quickstart runs outside a checkout. Done; numbers and tests in `docs/validation.md`.
- [x] Docs and examples pass (rule 10). Done; numbers and tests in `docs/validation.md`.
- [x] Correct the generalized dense benchmark (`eigh(F, G)`, not `eigvalsh(F)`) and publish a performance table at equal accuracy. Done (`examples/benchmark.py`, `docs/validation.md` performance): all methods solve the same $F y=\lambda G y$ with $\eta \le 10^{-10}$; warm forward of the targeted solve is 17-280x faster than `scipy.linalg.eigh(F, G)` on CTOK, Könies $N = 200, 400$ and Knyazev QA (CPU, one BLAS thread, load < 4 except one run); factor storage scales exactly as $N H^2$, but the solve is iteration-bound (iterations grow with $N$) and the factor time is $H^{2.0}$, not $H^3$, at these sizes; no GPU (office busy).
- [~] Make the repo public and do the trusted PyPI release (needs owner approval). Prepared: `publish.yml` runs on a `v*` tag with trusted publishing and a tag-equals-version check, metadata is release-ready, nothing is bumped or tagged.

**Gate:** a new user can install, run, interpret, reproduce and cite the qualified model.

**Open performance item (from the G benchmark).** SOLVAX 0.36 (Rayleigh–Ritz on a 3-block Krylov window) cut the Könies shift-invert iterations at η ≤ 1e-10 from 17, 33, 112, 310, >1000 to 4, 5, 11, 34, 121 at N = 200 … 3200 (N = 3200 now converges); warm solve 0.05, 0.11, 0.39, 2.3, 15 s on the Mac with 8 threads, so it still scales about as N^2. Iterations still grow as the continuum crowds the target; a polynomial filter or deflation in SOLVAX is the next lever, measured at equal residual.

### H. Extensions after 1.0

| Extension | Entry requirement | Acceptance |
|---|---|---|
| FAX → ESSOS mode potentials | Qualified modes; released ESSOS wave API | Units, phase (`α=(mι−n)Φ/[ω(G+ιI)]` for `−ωt`), induction, `E∥`, axis, energy; one matched FIRM3D case |
| Causal continuum damping | Non-Hermitian SOLVAX contract | Bowden–Hole–Könies (arXiv:1507.02072) tokamak TAE, ε = 0.225, tanh density: Ω = 0.39515 − 0.00174465i, γ/ω = −4.415e-3 (CKA contour; their large-aspect-ratio model −4.697e-3); arXiv:1402.6389 n = 6 case (γ/ω = −7.81e-7) as a resolution check; contour invariance and causal sign; their Nfp = 20 torsatron TAE (γ/ω = −1.15e-2) last. An unmerged prototype on the reduced model gives Ω = 0.37541 − 0.00218i (γ/ω = −5.8e-3), confirmed by dense solves as the only contour-invariant eigenvalue in the gap for contour depths 4 and 8. An independent 1-D two-mode model (Berk et al. 1992 form as printed in arXiv:1402.6389, 5/2 coupling; its placement inferred) gives 0.36962 − 0.00160i, 0.34% from Könies–Kleiber in Re Ω and 8% low in Im Ω; CKA and Könies–Kleiber agree on Im Ω to 0.2%. FAX's Im Ω is 1.37x the 1-D value at ε = 0.225 and grows as ε falls (1.77, 2.03, 2.22 at ε = 0.15, 0.1, 0.05) while Re Ω converges, so a first-order ingredient differs. Ruled out: harmonic truncation (m = 1–4: 2%), the |B| variation (5–10%), the Jacobian array (unused), and the Shafranov shift (a consistent zero-β shift raises Im Ω to 1.55x and 2.7x). Root tracking in contour depth α (dense, N = 200–600, Gaussian-bump contours): the real-axis TAE (0.37474) moves continuously into A = 0.37984 − 0.00591i by α ≈ 0.1, and A is N- and shape-invariant for every contour tried at α ≤ 1 (the outer m = 3 resonance sits near r ≈ 0.927, almost on the real axis). A is lost at moderate α for wide or early bumps, plausibly as the contour nears the tanh-density pole at r = 0.8 − 0.157i (not proven: an entire-profile control also loses its root). The earlier 0.37541 − 0.00218i (B) is a separate root that appears only after A is lost, in a narrow band of contour shapes, so it is not the continuation of the TAE. The prototype's answer is therefore A, with Im Ω 3.4x CKA and 3.7x the 1-D model. Open: find the first-order ingredient behind that excess (compare the resonant-layer coupling of the prototype with the 1-D model term by term); the ε scan of A (damping turns on at ε ≈ 0.1–0.15) did not resolve an ε² downshift. A VMEX zero-β equilibrium was built (Boozer ι matches) but needs a contour-independent analytic continuation. No code lands until FAX reproduces Im Ω of CKA within the 1-D/CKA spread |
| Sound coupling (BAE/BAAE) | Explicit coupled ordering; CONTI form (Nührenberg 1999; arXiv:1202.3204 Eqs. 9–10) checked against Cheng–Chance 1986; geodesic curvature from the Boozer spectrum; a Hermitian (ξ∥) form of the pencil | Pure sound branch ω = \|k∥\| c_s; tokamak BAAE gap top c_s/(qR) and Winsor GAM ω² = (2γp/ρR²)(1 + 1/2q²) on a circular section; Könies–Eremin 2010 coupled continuum |
| Trapped / FOW / FLR | Qualified passing response; ESSOS orbits | Each approximation separately vs the matched ITPA tier |
| Controlled design study | E, F, H-ESSOS | Arms A–D below with common constraints and a held-out geometry |

Design study arms, all under common equilibrium, profile and compute budgets:
- **A.** QS and constraints only.
- **B.** Paul geometric proxy.
- **C.** Qualified spectrum/window objective.
- **D.** Qualified drive or net-growth objective.

Each arm is followed by independent high-resolution spectra and ESSOS/FIRM3D transport with paired seeds. Report loss and redistribution, not only objective values.

Gap objectives: Paul et al. (2025) penalise squared Boozer harmonics, not eigenvalue gaps; no smooth widest-gap objective was found in the literature, so `gap_closure`'s softmax width is new and is compared against `continuum_cost`, not a published target.

Out of scope: a particle pusher inside FAX, nonlinear saturation from prescribed waves, broad GPU, Pallas or sharding rewrites before profiling shows the need.

### I. `vmex --alfven`

One flag that runs FAX on the equilibrium VMEX just solved, or on an existing `wout`, and writes one informative figure and a JSON summary. It mirrors `--turbulence` and `--neoclassical`: a lazy import in `vmex/core/cli.py`, a new `vmex/core/alfven_run.py`, an extra `vmex[alfven] = ["alfvex>=1.0.1", "matplotlib"]`, a floor in `_compat.OPTIONAL_MINIMUMS`, and the usual `MISSING OR OUTDATED OPTIONAL DEPENDENCY` error. FAX must be imported after the solve, because it enables x64 on import.

- FAX side first: `fax.summary(geometry, harmonics, density, window=None, count=4)` returns the gaps, the widest-gap mode and its convergence status. `fax.plot_summary(summary, path)` draws three panels: the continuum with labelled gaps, ι(s) with the gap crossings, and the mode's radial harmonics. vmex then keeps only the defaults and file naming, about 100 lines.
- Defaults from the equilibrium:
  - n families `j·nfp` for j = 0, 1, 2, plus n = ±1 when ι < 1;
  - m up to about `3·max(1, ι_max·n) + 4`, capped at 12;
  - 12 surfaces, `mboz = nboz = 6`, a 32 × 32 angle grid;
  - the eigenmode target at the centre of the widest gap.
  The drive runs only when a `[fast_ions]` table is given.
- Optional `--alfven-config alfven.toml` uses the `fax run` case schema, and unknown keys are rejected. `--alfven-density` (default 1e19 m⁻³) and `--alfven-ion-mass` (default 2 amu) override it. Both values are printed in the JSON and the figure title, because kHz scales as ρ^(-1/2).
- Geometry comes in memory, through `boozer_spectrum_state(..., geometry=True)` and `from_booz_xform`, with no file round trip.
- Outputs: `<case>_alfven.png`, and `<case>_alfven.json` with the gaps (centre and width in kHz, s, type), the mode frequency, its peak radius, its status, and the growth rate if computed.
- Tests: the missing-extra error, which copies the turbulence test; default selection from nfp and ι; and one end-to-end smoke test under 60 s on a small tokamak (`ns = 8`, `mboz = nboz = 4`).
- Risks: harmonic selection for HAE gaps, since the cost grows as about H³ per surface; first-call compile time, so the persistent JAX cache should be enabled; and a returned mode that sits in the continuum, which is filtered on `status.gap` and reported.

## History cleanup (conditional, measured)

- A fresh clone is **29 MiB** for a 1.1 MiB tree. The largest blobs are deleted pre-slim `docs/_static` SVGs (1.5–1.75 MB each, several versions).
- The gain is material, so do it as the DKX rewrite was done: strip deleted paths only.
- Steps: a verified offline mirror; open PRs (#9/#17) merged or recreated first; a reviewed path list; `git filter-repo` in a disposable mirror; tree hashes and all gates verified; collaborators re-clone; publish after owner approval.

## Start

1. A (repair), with B's convention table and derivation in parallel.
2. Open C in SOLVAX.
3. Prepare the Knyazev fixture for F.
4. Merge #17 → #9 → main first, so A starts from one operator.
