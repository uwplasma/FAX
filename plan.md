# FAX plan

Revised 2026-10-23 after an independent review of snapshot `2f7228f` (main before #16/#17); the review and its evidence bundle are kept outside the repository. This replaces the phase plan P0–P7 and its logbook. Both are still in the git history of this file (last at `2f7228f`).

## Goal

A small, validated, end-to-end differentiable JAX code for **stable shear-Alfvén spectra and their energetic-particle response in 3-D stellarators**, usable inside VMEX boundary optimization and installable with `pip install alfvex` (PyPI name; FAX will be renamed ALFVEX). The scientific targets:

1. **Methods.** Accurate boundary-to-spectrum sensitivities for 3-D Alfvén modes and spectral observables, including internal crossings, at a cost comparable to one forward solve.
2. **Physics.** Test whether a cheap differentiated spectral or kinetic objective reduces independently computed wave-induced fast-ion transport without degrading QS or confinement.

Novelty boundary from the literature: continuum-gap optimization (Paul et al. 2025), differentiable finite-n ideal MHD (AGNI, Gaur et al. 2026) and computed-mode orbit transport (Knyazev et al. 2026; FIRM3D) already exist. FAX's contribution is qualified, differentiated *stable* spectra plus kinetic response in a single optimization loop.

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
- 1,293 source lines in 8 modules, 33 tests.

Pending: #17, which folds #9's operator into `eigen.py` on the #9 branch, ports its manufactured solution and r^m axis interpolation, and preserves the original commits.

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
- [ ] Replace the vacuous Fu–Van Dam test with a weak-toroidicity limit: nonzero growth rate, absolute coefficient, sign and asymptotic scaling.
- [ ] Reject aliased angle grids (`ntheta` below what the harmonic products need), invalid density/B/geometry and singular mass. Tests assert finiteness and residuals, not shapes.
- [ ] Keep signed `omega2`. Define a stable frequency and an ideal growth rate separately, with a status flag.
- [ ] Shift-invert reports convergence, residuals and orthogonality. Detect Schur-pivot failure and fall back to a pivoted or dense reference (moves to SOLVAX in D).
- [ ] Second-order nonuniform `_d_ds` with a convergence test on irregular grids.
- [ ] Dimension-aware `boozmn` reading through booz_xform_jax's reader, behind a lazy optional import.
- [ ] Traceable geometry constructors: host validation separated from the jnp construction.
- [ ] Gap width uses one full-width convention (two-mode test `2 f0 |c|`). Fix docs and example tables.
- [ ] Slowing-down distribution: integrated-density check, correct the docstring, reject unknown `kind`.
- [ ] Parameter names for multi-digit modes (e.g. `("rc", m, n)` tuples). Asymmetric-geometry fixtures with nonzero `rmns`/`zmnc`.
- [ ] Reach statement and branch coverage ≥ 95% each.

**Gate:** every reproduced defect has a test that fails at `2f7228f` and passes after its fix; CTOK, Könies and drive references stay within their stated budgets.

### B. Qualify the reduced model (FAX)
- [x] Convention table: `docs/theory.md` section 1. Pinned by `test_conventions_and_geometry_diagnostics_on_ctok`: orientation ($\sqrt g < 0$), $G$, $I$, $\iota$, $B^\zeta$, $k_\parallel = (\iota m - n)B/(G + \iota I)$ in real space and in the assembled mode vectors, and $\psi' = \Phi_\mathrm{edge}/2\pi$.
- [x] Re-derived from $\delta W$ (`docs/theory.md` section 4, `test_kink_forms_differ_by_the_curl_b_term`): the direct kink form is the energy term at the retained order $O(\delta^2)$; it differs from the $\nabla\sigma$ form by $-\int u\sigma(\nabla\times\mathbf b)\cdot\nabla\phi^*$ (exact, $O(\delta^3)$), the complete $\mathbf Q_\perp$ gives $K_{\nabla\sigma} - \int\sigma^2|u|^2$ together with a bending correction, 1% of the bending energy on CTOK. No code change; Könies with $p$ stays 3.948e5 rad/s, CTOK `finite_beta` keeps its near-axis instability. Docs equivalence statement corrected, boundary and axis terms stated.
- [x] Jacobian and force-balance residuals: `jacobian_check`, `force_balance_residual`. CTOK $\psi'$ 0.4% and spread 5e-5 for $s \ge 0.05$ (first surface 15% low: radial finite difference of $R \sim \sqrt s$), $B_s$ vs embedding 1e-5; ITPA $\psi'$ 0.3%, spread 3e-4, $B_s$ 8e-5, $\mu_0p'$ vs `AM` 0.15%.
- [x] CTOK discrepancy: AE3D v31 pinned (commit 383d327, gfortran 15.2.0, unmodified build reproduces the shipped answers; instrumented patch and fixture in `tests/data`). Blocks compared separately: inertia and base bending agree (1e-9 to 2e-6); three $\iota'$ terms of AE3D's bending blocks differ from the energy (antisymmetric $f'f$ sign, missing $a'\iota'$ cross term, $g^{s\theta}\iota'$ sign); correcting them in AE3D's own Fortran moves its localized gap modes from 240.9, 244.9, 338.2, 355.6 to 258.9, 264.9, 266.4 kHz (FAX 268.5, 270.1, 279.1). Quadrature on AE3D's input: FAX 0.1-0.3%, AE3D 2-37%. The $J_\parallel$ blocks still disagree (cause open; the TAE does not depend on it). Docs and README state this; not reported upstream.
- [x] `finite_beta=True` labelled restricted/experimental in the docstring, theory, validation and README. No sound or full-MHD claim.

**Gate:** equations, code, quadrature and one matched external calculation describe the same model.

### C. SOLVAX contracts (SOLVAX)
- [ ] `generalized_eigenpairs(F, G, target, count, tol, max_iter)` on block-tridiagonal Hermitian pencils with moving positive mass. It returns values, `G`-normalized vectors, residuals, orthogonality, iterations, status and selection diagnostics.
- [ ] Factor/solve of `F − σG` with explicit transpose/adjoint semantics (the current scalar-identity `shift` is not enough), multiple right-hand sides, bad-pivot fallback, and dense references for small problems.
- [ ] Implicit first-order derivatives: eigenvalue `y†(dF−λdG)y`; eigenvector via a bordered solve with normalization and phase conditions; isolated-cluster projector `P = Y Y† G` with separation checks.
- [ ] Support table: eager/JIT/JVP/VJP, first vs higher order, CPU/GPU, precision.

**Gate:** qualified against dense references, including singular and indefinite cases and clusters.

### D. Migrate numerics and geometry (FAX + SOLVAX, VMEX, booz_xform_jax)
- [x] Replace FAX's `block_factor`/`block_solve`/`shift_invert`/`targeted_eigenvalues` with the SOLVAX contract. FAX keeps a thin wrapper only. Results stay within error budgets. Done: `eigen.py` calls `solvax.generalized_eigenpairs` (SOLVAX #160/#161, released in 0.32.0), adds `Status` (with `gap`, `fallback`) and NaN tangents for unconverged or tie-splitting solves; CTOK, Könies, drive and vmex references unchanged; timings in `docs/validation.md`. The dense all-modes solve and `eigenvalue_mean` stay in FAX: `dense_limit` only covers SOLVAX's pivot-breakdown fallback.
- [ ] Use the public VMEX state → Boozer bridge. Drop the half-row-1 workaround once a VMEX release contains #631.
- [ ] Continuum: triangular solves instead of the explicit Cholesky inverse. Share surface grids across continuum, global and drive.

**Gate:** no duplicated solver code; same physical outputs; pinned compatible releases tested.

### E. Spectral objectives and boundary gradients (FAX + SOLVAX)
- [x] Classify modes as isolated/gap vs continuum-like by radial refinement and localization. Objectives act on classified modes or physical windows only. Done: `classify_modes` (shift, nodal overlap, peak-radius margin to the local continuum, jump localization); CTOK TAE gap modes isolated (margin > 0.1, s = 0.45), modes at s > 0.6 and the uniform field all continuum-like (margin < 0.03); `geometry_frequencies`/`design_frequencies` return `Status` and are NaN if unconverged or tied; the earlier QA n = 0 modes are continuum-like.
- [x] Replace the hard `argmax` in `gap_closure` and the hard ι mask in `continuum_cost` with smooth or certified-isolated forms. Done: softmax-weighted widest gap (temperature `scale_hz`) and a C¹ smoothstep membership of width `softness`; gradient continuity across the old switches tested.
- [~] Gradient checks with step-size plateaus. Done except the frozen-path reference: fixed-geometry (asymmetric tangent, decade sweep, floor 1e-9) and re-solved (flat gauge floors 1e-4 to 1e-2) are tested separately; ns, mboz/nboz, angles, harmonics and nodes are swept one at a time (budget in `docs/validation.md`); vmex's `frozen_path_directional_fd` diverged on `zs11` and gave a non-finite state on `zs01` of LP-QA lowres, so the m = 1 gauge is bounded, not closed. Original item: Fixed-geometry and re-solved-equilibrium derivatives are tested separately, including asymmetric and `m = 1` directions (frozen-path FD in the scheduled tier). Refine VMEX, Boozer, angle, harmonic and radial resolutions, not radial only.

**Gate:** one real 3-D boundary objective with a first-order contract and a stated numerical error budget. Met for the LP-QA n = 1 gap modes: 4 kHz shift predicted to 0.05 kHz, budget 0.13 kHz.

### F. Independent 3-D and kinetic validation (FAX)
- [x] Reduce one QA or QH case of the Knyazev et al. 2026 dataset (DOI 10.5281/zenodo.21299907; AE3D kink term off) to an offline fixture with a provenance manifest. Compare the continuum, isolated frequencies, eigenfunctions/subspaces and convergence. Done for precise QA (`tests/data/knyazev_qa.npz`, `PROVENANCE.md`, `examples/knyazev_qa.py`): continuum within 3e-3 of STELLGAP; AE3D has no isolated eigenvalue (dense band), so the band and the archived global mode are compared: FAX is 5.1% below its eigenvalue at matched settings with no single-mode match (overlap 0.18, 27.6 kHz). Unresolved discrepancy; see `docs/validation.md`.
- [x] Derive the 3-D passing source from a guiding-centre Lagrangian or linear drift-kinetic equation; the toroidal-momentum argument alone is insufficient. Test axisymmetric, QS-relabelled, ±passing, negative-ι and n-reversal limits. Done (theory §8): each resonance `(j, k)` carries `Q F = ∂_E F + j ∂_s F / (ω Z e ψ')` (orbit's poloidal harmonic, mean-flux label), not the mode's `n_h/ι`; the old form failed helical relabelling by 17–20%. Könies 4.33/16.52/32.94/64.85 → 4.26/16.27/32.48/64.02 ×10³ s⁻¹. Open: the `O(ω/nΩ)` labelling (F of mean flux vs of the P_ζ surface, 1.5% at n = 6, 23% at CTOK n = 2), I/G and ε² terms of J_ϑ.
- [x] Make the passing-boundary, birth-edge and small-divisor regularizations explicit parameters with convergence demonstrated. Done: `boundary`, `FastIons.birth_width`, `guard` (and `nodes`); sensitivity near rational surfaces reported in validation.
- [ ] Matched ITPA ladder: analytic limit → fixed-mode ZOW with matched passing/trapped content → FOW/ZLR → FLR → damping. No fitting to the pooled band. Report `|γ|/ω` and frozen-mode validity. Rung 1 done (Fu–Van Dam closed form, flat and with a density gradient). Rung 2 is **unmatched**: FAX (passing only) is 0.58–0.60 of the Könies CAS3D-K (ZOW) curve at 82–191 keV, whose trapped content the paper does not state; `|γ|/ω` ≤ 4% to 200 keV, 16% at 400 keV. Open: a passing-only or trapped-resolved ZOW reference (Könies and Kleiber 2012), then FOW/ZLR, FLR, damping.

**Gate:** one independent 3-D spectrum and one matched nonzero kinetic response.

### G. Release 1.0 (FAX)
- [x] Coverage ≥ 95/95. Three test tiers: PR (analytic and contracts), scheduled (VMEX gradients, 3-D references, min/max dependency stacks), release (external comparisons, CPU/GPU). Done: markers `slow`/`scheduled` only; PR matrix and coverage job (`-m 'not scheduled'`, 100/100); `scheduled.yml` adds the minimum (3.10, JAX 0.4.38, SOLVAX 0.36.0), minimum-vmex (3.11, vmex 0.11.10, JAX 0.9.2) and latest (3.13) stacks; the release checklist is in `docs/development.md`.
- [x] One fixture provenance manifest (identifier, hash, model switches, units, license, command, tolerance). Add `CITATION.cff`. Done: `tests/data/PROVENANCE.md` covers all 9 fixtures in `tests/data` and `examples/data`, `tests/test_provenance.py` checks sizes and SHA-256, `make_ctok_fixtures.py` rebuilds the CTOK `.npz` bit for bit; `CITATION.cff` validates with cffconvert.
- [x] README of 600–900 words with scope and unsupported physics stated next to supported physics. Fix the AE3D and drive-band wording. The installed-wheel quickstart runs outside a checkout. Done: 824 words; the quickstart (`examples/quickstart.py`, Könies torus, TAE 4.0610e5 rad/s) was run from the built wheel in an isolated interpreter; the drive is compared with the unmatched CAS3D-K ZOW curve, not the pooled ZLR band.
- [x] Docs and examples pass (rule 10). Done: `docs/api.md` derivative contract per function (differentiable inputs, static arguments, transforms, first order, `Status`/NaN behaviour, pinned by `test_documented_transforms_and_failure_behaviour`) and an equation → function → test map; scientific examples are top-level scripts with editable settings and no `argparse`; tutorials `literalinclude` the sources; the CLI section lists what is Python-only.
- [x] Correct the generalized dense benchmark (`eigh(F, G)`, not `eigvalsh(F)`) and publish a performance table at equal accuracy. Done (`examples/benchmark.py`, `docs/validation.md` performance): all methods solve the same $F y=\lambda G y$ with $\eta \le 10^{-10}$; warm forward of the targeted solve is 17-280x faster than `scipy.linalg.eigh(F, G)` on CTOK, Könies $N = 200, 400$ and Knyazev QA (CPU, one BLAS thread, load < 4 except one run); factor storage scales exactly as $N H^2$, but the solve is iteration-bound (iterations grow with $N$) and the factor time is $H^{2.0}$, not $H^3$, at these sizes; no GPU (office busy).
- [~] Make the repo public and do the trusted PyPI release (needs owner approval). Prepared: `publish.yml` runs on a `v*` tag with trusted publishing and a tag-equals-version check, metadata is release-ready, nothing is bumped or tagged.

**Gate:** a new user can install, run, interpret, reproduce and cite the qualified model.

**Open performance item (from the G benchmark).** At fixed accuracy the shift-invert iteration count grows with radial resolution (Könies: 17, 33, 112, 310 iterations at N = 200, 400, 800, 1600; above 1000 at 3200), so the targeted solve scales about as N^2.3 instead of N. The continuum crowds the target as N grows. Try a block Krylov/Lanczos method or a deflated/polynomial-filtered subspace in SOLVAX, and measure at equal residual.

### H. Extensions after 1.0

| Extension | Entry requirement | Acceptance |
|---|---|---|
| FAX → ESSOS mode potentials | Qualified modes; released ESSOS wave API | Units, phase (`α=(mι−n)Φ/[ω(G+ιI)]` for `−ωt`), induction, `E∥`, axis, energy; one matched FIRM3D case |
| Causal continuum damping | Non-Hermitian SOLVAX contract | Analytic tokamak, contour invariance, causal sign, Bowden–Hole–Könies stellarator case |
| Sound coupling (BAE/BAAE) | Explicit coupled ordering | Könies–Eremin 2010 coupled continuum, acoustic limits |
| Trapped / FOW / FLR | Qualified passing response; ESSOS orbits | Each approximation separately vs the matched ITPA tier |
| Controlled design study | E, F, H-ESSOS | Arms A–D below with common constraints and a held-out geometry |

Design study arms, all under common equilibrium, profile and compute budgets:
- **A.** QS and constraints only.
- **B.** Paul geometric proxy.
- **C.** Qualified spectrum/window objective.
- **D.** Qualified drive or net-growth objective.

Each arm is followed by independent high-resolution spectra and ESSOS/FIRM3D transport with paired seeds. Report loss and redistribution, not only objective values.

Out of scope: a particle pusher inside FAX, nonlinear saturation from prescribed waves, broad GPU, Pallas or sharding rewrites before profiling shows the need.

## History cleanup (conditional, measured)

- A fresh clone is **29 MiB** for a 1.1 MiB tree. The largest blobs are deleted pre-slim `docs/_static` SVGs (1.5–1.75 MB each, several versions).
- The gain is material, so do it as the DKX rewrite was done: strip deleted paths only.
- Steps: a verified offline mirror; open PRs (#9/#17) merged or recreated first; a reviewed path list; `git filter-repo` in a disposable mirror; tree hashes and all gates verified; collaborators re-clone; publish after owner approval.

## Start

1. A (repair), with B's convention table and derivation in parallel.
2. Open C in SOLVAX.
3. Prepare the Knyazev fixture for F.
4. Merge #17 → #9 → main first, so A starts from one operator.
