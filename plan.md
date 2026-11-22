# ALFVEX plan

Final plan of 2026-11-19. It was written against main `0c84e00` (release 1.2.0 is `081bea9`), vmex 0.15.0, solvax 0.36.2, booz_xform_jax 0.5.3 and essos 0.22.0. It replaces every earlier plan in this file; those stay in the git history (last at `0c84e00`). The code is called FAX up to release 1.2.0 and ALFVEX from step 0.2 on. This file uses the new name except where it cites the code as it is today.

The plan rests on four reviews made on 2026-11-18 and 2026-11-19, one by the maintainers and three reports by independent collaborators with their reproduction archives, and on the 53 PDF files of the reference folder, read with their page images. The archives are kept outside the repository. Every number quoted here was either measured in one of those reviews or is printed in the cited paper; section 6 says which.

Contents: 1 goal, 2 what success means, 3 principles, 4 the name, 5 ownership, 6 state, 7 conventions, 8 models and equations, 9 repository structure, 10 tests, 11 examples and documentation, 12 performance, 13 steps, 14 order and releases, 15 decisions that need data, 16 data to request, 17 out of scope, 18 references.

## 1. Goal

A small differentiable code that answers three questions about a stellarator equilibrium and supplies the answers as terms of a VMEX optimization.

1. Which Alfvén eigenmodes exist: the continuum, gap modes and sound-coupled modes, at finite $\beta$.
2. Which are unstable: the drive from every fast-ion orbit class, less continuum, radiative, electron and ion damping.
3. What they do: the saturated amplitude, and the fast-ion flux and loss it causes, computed with ESSOS.

The research result to aim for first: a boundary optimized against a resolved Alfvén spectrum, or against a stability margin, lowers the collisional, energy-weighted fast-ion loss at stated wave amplitudes on plasmas that were not used in the optimization, by more than a geometric gap penalty does and at equal quasisymmetry, aspect ratio and wave-off confinement.

What the code would add to the field. Each item is a claim only after its gate in section 13 is met.

- A Hermitian global operator with sound coupling and $\delta B_\parallel$ in a potential formulation (section 8.4). The published global formulations with sound (Cheng and Chance 1986; Cheng 1992; CAS3D) are either nonlinear in $\omega$ or solve the three-component displacement.
- A complex spectrum with continuum and radiative damping, differentiated with respect to the plasma boundary. No published code differentiates one.
- A kinetic response with trapped ions, the orbit-averaged drift in the resonance and $\delta B_\parallel$ in the coupling, checked by an orbit-integrated evaluation in ESSOS that includes orbit width and Larmor radius.
- Electron damping from a finite-frequency drift-kinetic solve with the linearized Fokker-Planck operator.
- An optimization whose objective is a stability margin or a wave-induced loss, checked by orbit following at higher resolution. Published optimizations act on the continuum only (Paul et al. 2025); published wave-to-orbit studies evaluate given designs (Knyazev et al. 2026).
- A public kinetic benchmark for a stellarator, with inputs, modes, signed powers and outputs. None exists.

Already published elsewhere, and therefore reused or compared against, not claimed: continuum-gap optimization (Paul et al. 2025; Hyder, Paul and Knyazev 2026), shape adjoints (Antonsen et al. 2019), differentiable bounce averaging (Unalmis et al. 2026, in DESC), three-dimensional wave-to-orbit pipelines (STELLGAP, AE3D and FIRM3D in Knyazev et al. 2026), reduced Alfvénic transport models (Slaby et al. 2025), and global kinetic response in CAS3D-K, CKA-EUTERPE, LIGKA, MEGA and FAR3d.

## 2. What success means

### Per release

| Release | Contains | Success is |
|---|---|---|
| 2.0 | The name ALFVEX, the full signed basis, the repairs of package 1, one evaluation path with status, VMEX terms, mode export | Every number in the README and the validation page is produced by a test or an executed example at pinned versions. Python, the `alfvex` command and a TOML case give the same arrays and the same status. `vmex --alfven` is released and finds the 96 to 142 kHz gap of the precise quasi-axisymmetric case. The published design example is rerun on the full basis. |
| 2.1 | Current and pressure terms in the form CKA solves, the sound-coupled global operator, continuum damping, the complex spectrum and its derivative, radiative and electron damping | The gates of packages 3 and 4: one sound-coupled global mode with a published counterpart; the ITPA frequency within 1.5% of $4.01\times10^5$ rad/s with the CAS3D pressure profile; the damping of the Könies and Kleiber (2012) tokamak case within 1% between two methods; the derivative of a damped pole equal to finite differences to $10^{-4}$. |
| 2.2 | Trapped ions, precession, the full kinetic coupling, the orbit-integrated evaluator in ESSOS, collisions, saturation | Three evaluators of the passing drive agree within 0.5% and with reflected ions within 5%. The mode passes from ALFVEX to ESSOS with one call and conserves the wave invariant to $10^{-12}$. The energy ledger of package 6 closes within the stated sampling error. |
| 2.3 | Objectives from geometric to direct, and the optimization study | The rank correlation of each proxy with ESSOS losses is stated over at least 20 configurations. In the four-arm study one arm lowers the paired, energy-weighted loss by more than the predeclared minimum, with the lower bound of its interval above zero, at equal constraints, on held-out profiles and at higher resolution. |
| 3.0 | The kinetic response inside the eigenproblem, the reduced transport model, free-boundary and single-stage runs | A complex kinetic root and its boundary derivative on the ITPA case, with the frequency shift 4.012, 3.968, 3.880 and $3.732\times10^5$ rad/s at 0, 100, 200 and 300 keV (CAS3D-K) reproduced within the stated tolerance. |

### At every release

- Source size at or under 4000 lines in 12 modules (2037 lines in 8 modules at 1.2.0), counted by `wc -l src/alfvex/*.py`.
- Statement coverage and branch coverage each at or above 95% on the pull-request tier, with no excluded production line.
- The pull-request tier takes under 6 minutes of test time and under 10 minutes in all on the CI machines.
- The documentation builds with warnings as errors and executes the examples.
- A profile of the reference cases is stored with the release, and the performance test passes.
- `tests/test_hygiene.py` passes on every tracked file.

### What does not count as success

A frequency match on a mode whose eigenfunction was not compared. A solve that did not converge, used as a low objective. A growth rate matched by a fitted factor. A loss reduction at an amplitude that was chosen after the result. A gradient checked only against the same code at the same resolution. A coverage figure presented as validation of the physics.

## 3. Principles

1. **The code stays small.** New physics enters as terms of the existing weak forms and kernels. Every pull request states its net line count. Lines are removed by removing duplication, never by removing an equation, a failure check or a test.
2. **Algorithms live in SOLVAX**, orbits in ESSOS, drift-kinetic solves in DKX, equilibria and optimization drivers in VMEX. ALFVEX holds weak forms, kinetic kernels, objectives, the case schema and the mode record.
3. **Three entry points, one path.** Python, the `alfvex` command and a TOML case reach the same features under the same names through one pure function (section 9.3). A feature is merged with all three and with its documentation page.
4. **Differentiable end to end.** Every numerical result is differentiable to first order in the declared continuous inputs: boundary coefficients, profiles, distribution parameters and the mode amplitude. The contract of each function states its inputs, its static choices and what is host-only. File reading, harmonic selection and hard loss flags are outside the differentiable map.
5. **Performance is measured.** Each release carries a profile down to the compiled kernels. One test bounds ratios between kernels and planned memory for three reference cases.
6. **Tests are physics.** Each test is an equation, a limit, a conservation law or a published number with matched assumptions, and names its source. A test that only repeats the code's own output is a regression test and is labelled so.
7. **Examples are plain scripts** in the style of the other uwplasma repositories: settings at the top, no argument parsing, no `main()` function, no `if __name__` block.
8. **Text is plain and specific** in code, commits, pull requests and documentation, and passes `tests/test_hygiene.py`. Commits and pull requests carry no tool attribution and no co-author trailers. The author of every commit is the repository owner.
9. **A model is not relabelled.** Results are tagged observed, published or proposed. Names say what is computed: a prescribed-amplitude loss is not a predicted loss, test-particle collisions are not a conserving operator, a local gyrokinetic growth rate is not an Alfvén damping rate.

Rules for pull requests:

1. One physical or numerical claim per pull request.
2. A physical claim needs an independent analytic or published comparison with matched assumptions.
3. Every eigen-solve returns the signed eigenvalue, residuals, orthogonality, iteration count and a selection certificate that includes the residual of the nearest excluded pair. Derivatives of unconverged or non-isolated modes are rejected.
4. Signed $\omega^2$ and complex $\omega$ are kept in results and exports. Instability and damping are reported, never clipped.
5. A kinetic or dissipative number needs two independent evaluators that agree, or a published value whose content is known.
6. Complex eigenvalues report left and right residuals, biorthogonality and independence of the contour or regularization.
7. Powers are signed and share one wave energy (section 8.5). A mechanism is counted once.
8. Monte Carlo results state markers, seed and statistical error. Their derivatives use fixed random numbers and are checked against finite differences on the same numbers. No derivative is taken through the end point of a chaotic orbit or through a hard loss flag.
9. An interface to another code is a versioned record with a conventions table and a round-trip test.
10. Speed is compared at equal accuracy, with the machine, its load and the problem size recorded.
11. An optimization result is a claim only after a check at higher resolution and by an independent orbit-following calculation.
12. If a reference disagrees, the operator, the source term, the measure, the wave energy and the model switches are compared separately before any coefficient changes.

## 4. The name

FAX is distributed on PyPI as `alfvex` because the name `fax` belongs to an unrelated package. The repository, the import name, the command and the prose name now follow the distribution.

| Item | Was | Becomes |
|---|---|---|
| Repository | `uwplasma/FAX` | `uwplasma/alfvex` |
| Import and source folder | `fax`, `src/fax` | `alfvex`, `src/alfvex` |
| Command | `fax run`, `fax plot` | `alfvex run`, `alfvex plot` |
| Name in prose | FAX | ALFVEX |
| PyPI distribution | `alfvex` | unchanged |
| Documentation | `fax.readthedocs.io` | `alfvex.readthedocs.io`, with a redirect from the old address |
| Compilation cache | `~/.cache/fax/jax` | `~/.cache/alfvex/jax` |
| vmex | `import fax` in `vmex/core/alfven_run.py`, `fax run` in help text, the nightly import check | `import alfvex`; the extra `vmex[alfven]` and its requirement `alfvex>=` keep their names |
| ESSOS pull request 235 | `examples/particle_tracing/fax_alfven_mode.py`, `tests/test_fax_eigenmode.py` | `alfvex_mode.py`, `test_alfvex_mode.py` |
| Other uwplasma repositories | "FAX" in changelogs, ecosystem pages and plans | ALFVEX, in the next change to each file |

Rules of the rename:

- No compatibility package. A wheel that installs a top-level `fax` would collide with the unrelated distribution. Release 2.0.0 states the change in its notes and in the README: versions up to 1.2.0 are imported as `fax`, later ones as `alfvex`.
- The rename is one mechanical pull request with no change of behaviour (step 0.2). It lands after the four draft pull requests that touch `src/fax` (step 0.1), so that nothing has to be rebased across it, and before any other work.
- Historical records keep the old name: the git history, fixture provenance that cites `uwplasma/fax#9`, and release notes of past versions.
- GitHub redirects the old repository address. The PyPI trusted publisher and the Read the Docs project do not follow a rename and are changed by the owner (step 0.3).

## 5. Ownership

| Code | Owns | What ALFVEX needs from it |
|---|---|---|
| ALFVEX | Weak forms, kinetic kernels, mode classification, objectives, the case schema, the mode record | |
| SOLVAX | Block solves, eigen-solvers, root finding, quadrature, interpolation, derivative rules | (a) A selection certificate that includes the residual of the nearest excluded pair. (b) Pencil solves $(F-\sigma G)^{-1}b$ with complex batched $\sigma$. (c) Generalized eigenpairs of complex-symmetric block-tridiagonal pencils with left and right vectors and implicit derivatives in forward and reverse mode under `jit`; `eigenpair_reverse` works in eager reverse mode only today. (d) A root and contour solver for $D(\omega)y=0$ with the bordered tangent of section 8.9. (e) An explicit `dense_limit` with no hidden dense fallback under `vmap`. (f) The routines now in FAX: `generalized_eigh`, `eigenvalue_mean`, `lowest_eigenvalues`. (g) Factors applied as stored block products, a run-time iteration limit, warm starts. (h) A banded radial spline for ESSOS. |
| VMEX | Equilibrium, implicit derivative, optimization drivers, bounce kernels | Boozer geometry and its radial derivatives at arbitrary $s$ from the spline state. The fix of the first-surface $m=1$ harmonics. Density and temperature profile objects. A release that contains `--alfven`. Batched adjoint right-hand sides and reuse of the last linearization. `wout_tables_from_state` (pull request 819) qualified for orbits. |
| booz_xform_jax | Boozer transform | Nothing new: `boozer_spectrum_high_order` lives in VMEX and calls it; $R$, $Z$, the angle offset and the Jacobian with radial derivatives come from there (V10). |
| ESSOS | Guiding-centre and full orbits, test-particle collisions, loss events | One array constructor for a wave in the convention of section 7, replacing pull requests 177, 196 and 235: traceable, no branch on traced values, zero outside the radial grid, regular at the axis. `psi0_from_vmec` without the `float()` cast. Live Boozer arrays (118) and the covariant radial term (205). Marker weights, samplers, an accumulated work integral, orbit classification, mutually exclusive end states (249), a smooth loss indicator (61). The term $\mu\,\delta B_\parallel$ in the guiding-centre Hamiltonian. A stated Coulomb logarithm. |
| DKX | Drift-kinetic solves with Fokker-Planck collisions | A finite-frequency term and a complex state. A wave-source builder. A phase factor for toroidal numbers that are not multiples of the field period. |
| NEOPAX, NTX | Profiles, $E_r$, D-T source, photon losses | Arrays in SI with shell volumes. Photon radiation stays there; ALFVEX computes wave radiation only. |
| GKX | Local gyrokinetics | The local kinetic-Alfvén dispersion relation as a cross-check. Its growth rates are not damping rates of global modes. |
| STELLGAP, AE3D, FIRM3D and its GPU tracer CATAPULT, FAR3d, MEGA, cuGMEC, ASCOT5, AGNI, DESC | External comparators | Pinned fixtures and exported modes only. STELLGAP, AE3D and FIRM3D are MIT; MEGA is MIT-style with a required citation; cuGMEC is GPLv3, so only its published numbers are used. Published speeds to compare with: CATAPULT 5 to 10 times (vacuum) and 40 to 60 times (waves) over 128 CPU threads; cuGMEC up to 20 times over a Xeon Gold 6348. |

A dependency rule follows from the table: the core install needs JAX, SOLVAX, NumPy and netCDF4. VMEX, ESSOS, DKX and plotting are extras, imported inside the functions that use them.

## 6. State on 2026-11-19

### 6.1 What release 1.2.0 holds

The continuum in STELLGAP's form with optional sound coupling. Global modes in AE3D's weak form with linear radial elements, solved densely or by shift-invert through SOLVAX with implicit derivatives. Restricted current and pressure terms (`finite_beta=True`). A passing-ion drive at zero orbit width for Maxwellian and slowing-down ions, and the same kernel for bulk-ion Landau damping. Smooth objectives and boundary gradients through VMEX. A TOML runner and a command for the continuum and global modes. 2037 source lines in 8 modules, 96 test functions. Main `0c84e00` adds compiled kernels (pull request 66). Draft pull requests 67, 68, 69 and 72 are stacked on it.

Not in it: sound coupling in the global problem, any damping model, trapped ions, orbit width, Larmor radius, the electron response, collisions, saturation, and a differentiable entry point above the numerical functions.

### 6.2 Measurements

Observed on 2026-11-18 and 2026-11-19 with alfvex 1.2.0 unless stated. Rows marked R were obtained by one review and reproduced by another.

| Area | Result | Step |
|---|---|---|
| Tests | The pull-request tier passes: 117 tests in 437 s and in 699 s on two loaded machines, statements 95.34%, branches 95.14%. With the slow tests on main: 99.90% and 99.32% in 21 min 45 s. | 9.1 |
| Basis (R) | Completing the basis with the conjugate harmonics changes the modes of the quasi-axisymmetric example in 102 to 105 kHz from two to three (8.3). On the precise quasi-axisymmetric continuum the error against STELLGAP is 0.300% with one sign and 0.160% in the cosine sector; 4.65% of branch samples move by more than 1%. | 0.1 |
| Draft stack | Pull requests 67, 68, 69 and 72 never ran in CI. The head of the stack passes its suite on Linux, but gives different results eagerly and under `jit` on a three-dimensional field with one field period (12 against 9 continuum branches per surface), clips `count` silently, and moves documented numbers that were not regenerated. | 0.1 |
| Radial accuracy | Second order on the analytic torus. Through VMEX the 102 kHz mode is $1.7\times10^{-3}$ off at 24 nodes, $2.3\times10^{-4}$ with nodes on midpoints. | 1.1 |
| Equilibrium path | The first-surface $m=1$ harmonics of $R$ and $Z$ on the design path are 0.47 of the `Booz_xform` values. `vmex --alfven` is not in the released wheel, merges surfaces and finds no gap on the precise quasi-axisymmetric case. | 1.2, 1.3 |
| CTOK continuum | Against a rebuilt STELLGAP: mean 0.63%, maximum 16.45% at the edge, caused by a missing outer surface in the reference. With the surface restored: maximum 0.70%. | 1.6 |
| CTOK global modes | Six AE3D modes agree in frequency within 0.19 to 1.35%; two of them have inertia-norm overlaps of 0.37 and 0.08. The 337.92 kHz mode is not matched. The three $\iota'$ bending terms of FAX agree with independent quadrature; the parallel-current block is unresolved. | 1.7, 3.2 |
| Precise quasi-axisymmetric global mode | The archived 30.309 kHz AE3D mode has a best squared overlap of 0.18 near 27.61 kHz; the Rayleigh quotient is 5.1% low. Not a match. The fixture has no independent mass density. | 3.2, 16 |
| Entry points (R) | The command, a file path and a mapping give bitwise equal arrays. `fax.run` cannot be differentiated. Exports drop the sign of $\omega^2$ and the solver status. | 2.1 |
| Solver certificate | The convergence flag of SOLVAX ignores the excluded pair that sets the selection gap (8.9). `eigenpair_reverse` returns a finite derivative for a supplied pair that is not an eigenpair; its guard is on conditioning, not on the residual. | 1.8, 4.3 |
| Design path | A density profile cannot be given: the design casts it to a float and the classification fails on arrays. Several coarse modes can claim one refined mode. A 10% density change moves the 103 kHz mode by 4.79 kHz. | 2.2, 2.7 |
| Drive: numerics (R) | An independent implementation reproduces the 32 temperature and sideband values of the ITPA case to $2\times10^{-13}$ and $W$ to $10^{-15}$. The damping formula of Kolesnichenko and Tykhyy is met to $2\times10^{-5}$. An orbit-based evaluation outside the code reproduces the passing drive to 0.3 to 0.45%. | 5.1 |
| Drive: physics | The passing drive is 59.5 to 61.0% of the CAS3D-K curve of Könies et al. (2018, fig. 1a), 66 to 69% on a solved finite-pressure mode, and 86 to 106% of the STAE-K curve of Slaby et al. (2016). The CAS3D-K curve is the net rate of Fehér (2013, fig. 5.8), the sum of electron, bulk-ion and fast-ion parts, shifted by $2.44\times10^3$ s$^{-1}$ in one panel and $1.22\times10^3$ in the other; compared by increments, which no shift can change, the ratio is still 0.60. Reflected ions add 0.001 to 0.07% at zero precession, 7 to 29% once the orbit-averaged precession is in the resonance. With full drift orbits the drive is 2.5 times the zero-width value at 82 keV and 0.47 times at 400 keV. | 5.2, 5.3, 5.6 |
| Drive: defects | A temperature profile is mis-broadcast or its gradient dropped; `guard = 0` gives NaN; a zero charge divides by zero; two mass conventions sit in one module; the sign example of the docstring is reversed; a sampled $B_{\max}$ has a kink that differentiation does not see. | 1.5 |
| Drive: derivative | Through the eigenvector the derivative of $\gamma$ with respect to a field parameter agrees with re-solved differences to $7\times10^{-10}$; holding $\Phi$ and $\omega$ fixed underestimates it by 4.1%. | 5.1 |
| Continuum damping | A rational fit to the resolvent of the Hermitian matrices finds the damped TAE. With the current term it gives $\mathrm{Im}\,\Omega^2=-8.84\times10^{-4}$ on the tokamak case of Könies and Kleiber (2012), whose figure shows $-8.9\times10^{-4}$; without it the damping is 3.4 times larger. On the case of Bowden et al. (2015) as printed the result is 1.19 times the published damping, and the printed $\iota$ polynomial does not match their figure 1. | 4.1, 4.2 |
| ESSOS interface | Six manual steps were needed to pass a mode through pull request 235; the sign of time flips and the amplitude is discarded. With the mapping of 8.10 the fields agree to $4\times10^{-15}$. The wave invariant is conserved to $5\times10^{-15}$. One constructor is differentiable, the other is not. `psi0_from_vmec` casts the edge flux to a float. | 6.1, 6.2 |
| Collisions | ESSOS and ASCOT5 coefficients differ by up to 4.61% with their own Coulomb logarithms and by $2.8\times10^{-6}$ with a common one; the Itô drift of the speed differs by up to 1.34%. One pitch-scattering step at $\nu_D\Delta t=0.01$ has an error of $-3.3\times10^{-3}$ in the second Legendre moment at $\xi_0=0.99$. | 6.4 |
| Performance | Section 12. | 2.9, 2.10 |
| Installed wheel | On main `0c84e00`, not on 1.2.0: Python 3.10.20 with JAX 0.6.2 crashes while compiling the gradient of the assembly that pull request 66 put under `jit`, on macOS arm64. Removing that one `jit` restores the gradient to $10^{-15}$. | 0.5 |

### 6.3 Open questions that only data can close

- Which orbit classes the CAS3D-K run of the ITPA benchmark kept, and its bulk part. The thesis that holds the curve does not say. Two observations in it point at the two candidates: in a NOVA-K comparison passing ions alone give 0.36 to 0.65 of the total, and on W7-AS dropping the pressure term of the mode leaves the frequency unchanged but cuts the drive to 0.63 to 0.73. The wave energy is defined as here, so no factor of two hides there (8.6, section 16).
- Which $\iota$ profile figure 1 of Bowden et al. (2015) used.
- An independent mass density and mode vectors for the precise quasi-axisymmetric AE3D case.

## 7. Conventions

One set of conventions governs the code, the tests, the mode record and every fixture. A fixture taken from a paper states the conversion from the paper's conventions.

| Item | Convention |
|---|---|
| Radial label | $s=\Phi_t/\Phi_{\rm edge}\in(0,1)$, normalized toroidal flux. $\rho=\sqrt s$ where a paper uses $r/a$. Papers that use poloidal flux are converted through $\iota$ and the conversion is stored with the fixture. |
| Flux factor | $\psi=\Phi_t/2\pi$, $\psi'=d\psi/ds=\Phi_{\rm edge}/2\pi$ with its sign, in Wb/rad. |
| Field | $\mathbf B=\psi'(\nabla s\times\nabla\theta+\iota\,\nabla\zeta\times\nabla s)=G\,\nabla\zeta+I\,\nabla\theta+B_s\nabla s$, with $G$ and $I$ in T m. $\lvert\sqrt g\rvert=\lvert\psi'\rvert(G+\iota I)/B^2$. For VMEC and `booz_xform` input $(s,\theta,\zeta)$ is left-handed against $(R,\phi,Z)$; `circular_torus` follows the same handedness. AE3D's `jpol` is $G$ and its `jtor` is $-I$. |
| Parallel derivative | $\mathbf B\cdot\nabla=\dfrac{B^2}{G+\iota I}(\partial_\zeta+\iota\,\partial_\theta)$. |
| Harmonics | Physical fields are $\mathrm{Re}\sum_{mn}\Phi_{mn}(s)\,e^{i(m\theta-n\zeta-\omega t)}$ with $\omega>0$. $n$ is the full-torus number, never $n/N_{fp}$. A family is $n=n_0+kN_{fp}$. $(m,n)\to(-m,-n)$ is the complex conjugate. |
| Parallel wavenumber | $k_\parallel=(m\iota-n)B/(G+\iota I)$. |
| Complex frequency | $\omega=\omega_r+i\gamma$. A damped mode has $\gamma<0$. A positive damping rate is written $\Gamma=-\gamma$ and never called $\gamma$. |
| Eigenvalue | $\lambda=\omega^2$, signed. $Fy=\lambda Gy$ with $y^\dagger Gy=1$; $y_{kj}$ is the amplitude of harmonic $j$ at node $k$. |
| Fields of a mode | $\Phi$ in volts. $\boldsymbol\xi_\perp=(i/\omega)\,\mathbf b\times\nabla\Phi/B$. $\delta\mathbf A=\alpha\mathbf B$ with $\alpha$ in metres, $\delta\mathbf B=\nabla\times(\alpha\mathbf B)$, $\delta\mathbf E=-\nabla\Phi-\partial_t(\alpha\mathbf B)$. |
| Amplitude | The eigenvector norm is algebraic. A physical amplitude is a separate real number with its rule: the maximum of $\lvert\delta B\rvert/B$ over a stated domain, or the wave energy $W$ in joules. |
| Power and energy | $P_{a\to w}$ in watts, positive when species or channel $a$ gives energy to the wave. $W>0$ in joules. $\gamma=\sum_aP_{a\to w}/(2W)$. |
| Distribution | $F_0$ per unit of $d^3x\,d^3v$. $E=m v^2/2$ in joules, pitch variable $\lambda=(1-\xi^2)/B$ with $\xi=v_\parallel/v$. Radial derivatives of $F_0$ are taken at fixed $E$ and $\mu$. |
| Magnetic moment | $\mu=mv_\perp^2/2B$ in J/T. ESSOS has three conventions today: per unit mass in the Boozer tracer, J/T in `GuidingCenterMu`, and $\mu/(mc^2)$ with $v_\parallel/c$ in the collisional states. The trace record carries J/T for all three (step E10 of the ESSOS list). |
| Species | Nuclear masses (deuteron 2.013553 amu), charge numbers explicit, never implied by a default. |
| Units | SI inside the kernels. Case files give temperatures and energies in eV, densities in m$^{-3}$, frequencies in kHz under keys ending in `_khz`. Outputs ending in `_hz` are $\omega/2\pi$. |

## 8. Models and equations

Each subsection states what the code solves today, what is added, and the equations in the form to be implemented. Gates and tests are in sections 10 and 13.

### 8.1 Geometry

Present. A `Geometry` holds the Boozer harmonics of $|B|$, $R$, $Z$, the angle offset $p=\phi-\zeta$ and the Jacobian on a set of surfaces, with $\iota$, $G$, $I$. `surface_grid` synthesizes them on a uniform grid in $(\theta,\zeta)$ over the full torus, builds the covariant basis $\mathbf r_s,\mathbf r_\theta,\mathbf r_\zeta$ from the cylindrical embedding $(R,\zeta+p,Z)$, and inverts $g_{ij}=\mathbf r_i\cdot\mathbf r_j$ for $g^{ij}$. Radial derivatives of the harmonics use second-order finite differences in $s$.

Added.

- The inverse metric in closed form, $\nabla s=\mathbf r_\theta\times\mathbf r_\zeta/\sqrt g$ and cyclic, in place of a LAPACK inverse (step 2.9).
- Geometry and its radial derivatives at the quadrature points from the VMEX spline state, in place of linear interpolation of half-mesh harmonics, which gives jumps of 4 to 40% in $\partial_s|B|$ (step 2.4).
- Angle integrals over one field period where the integrand has the period (pull request 69).
- The curvature and the parallel current from the field, not from reconstructions: $\boldsymbol\kappa=\mathbf b\cdot\nabla\mathbf b$, $\sigma=\mu_0J_\parallel/B=\mathbf b\cdot\nabla\times\mathbf b$, and the geodesic factor $\kappa_s=2(G\,\partial_\theta B-I\,\partial_\zeta B)/[\psi'(G+\iota I)B]$.
  Corrected 2026-11-21: the formula read $2(I\,\partial_\zeta B-G\,\partial_\theta B)/[\psi'(G+\iota I)B]$, the opposite sign. Source: $K_s=2\boldsymbol\kappa\cdot(\mathbf B\times\nabla\psi)/B^2$ of Cheng and Chance (1986, p. 3696, below eq. 9, checked on the page), with $\nabla s$ in place of $\nabla\psi$. With $\mathbf B=G\nabla\zeta+I\nabla\theta+\beta\nabla s$, $(\mathbf B\times\nabla s)\cdot\nabla\theta=G/\sqrt g$, $(\mathbf B\times\nabla s)\cdot\nabla\zeta=-I/\sqrt g$ and $\sqrt g=\psi'(G+\iota I)/B^2$, all signed, this gives the expression above. The convention is that $\sqrt g$ and $\psi'$ keep their signs; the sign then does not change with the handedness of the coordinates (it would only if $|\sqrt g|$ or $|\psi'|$ were used), and the polarization test of 8.4 checks it. The code of 1.2.0 (`continuum.sound_coupling`) still uses the old sign with $|\sqrt g|$, which equals the expression above times $-\operatorname{sign}[\psi'(G+\iota I)]$; step 3.3 changes it to the signed form and the polarization test of 8.4 checks it.

### 8.2 The continuum

Present. On each surface the trial functions $e_j=e^{i(m_j\theta-n_j\zeta)}$ give the Hermitian pencil $Kx=\omega^2Mx$ of

$$
\mathbf B\cdot\nabla\Big(\frac{|\nabla s|^2}{B^2}\,\mathbf B\cdot\nabla\Phi\Big)+\mu_0\rho\,\omega^2\frac{|\nabla s|^2}{B^2}\,\Phi=0,
$$

$$
K_{jk}=\frac{a_ja_k}{\mu_0}\big\langle w\,B^4e_j^*e_k\big\rangle,\qquad M_{jk}=\rho\,\big\langle w\,e_j^*e_k\big\rangle,\qquad w=\lvert\sqrt g\rvert\,\frac{|\nabla s|^2}{B^2},\qquad a_j=\frac{n_j-m_j\iota}{G+\iota I},
$$

with $\langle\cdot\rangle$ the angle average, computed by FFT. With a pressure the unknowns double to $(E_j,c_j)$, the surface displacement and $\xi_\parallel/B$, and the pencil is that of Cheng and Chance (1986) in the Hermitian form

$$
K=\begin{pmatrix}K_A+\langle4\gamma_Bp\,J\hat\kappa^2e_i^*e_j\rangle&-2i\,a_j\langle\gamma_Bp\,J\hat\kappa B^2e_i^*e_j\rangle\\ \text{h.c.}&a_ia_j\langle\gamma_Bp\,JB^4e_i^*e_j\rangle\end{pmatrix},\qquad
M=\begin{pmatrix}M_A&0\\0&\rho\langle JB^2e_i^*e_j\rangle\end{pmatrix},
$$

with $\gamma_Bp=\gamma p\,B^2/(B^2+\mu_0\gamma p)$, $J=\lvert\sqrt g\rvert$ and $\hat\kappa=\kappa_s/2$. Its limits are tested: the sound branch $\omega^2=k_\parallel^2\gamma_Bp/\rho$, which is $|k_\parallel|c_s$ as $\beta\to0$, the geodesic acoustic frequency $\omega^2=(2\gamma p/\rho R^2)(1+1/2q^2)$ of Winsor, and the acoustic gap top $c_s/(qR)$.

Added.

- The full signed basis and parity sectors (8.3) apply to the continuum as to the global problem.
- Qualification of the sound pencil against CONTI on the HSX case of Könies and Eremin (2010) and on the H-1 case of Bertram et al. (2012), step 3.5.
- A kinetic inertia for low frequencies: the fluid term $\gamma p$ is replaced by the kinetic thermal-ion response of Zonca, Chen and Santoro (1996), which gives the kinetic frequency of the beta-induced gap and its Landau damping (8.8, step 4.9).

### 8.3 Global modes: the shear operator, the basis, the current and pressure terms

Present. The reduced ideal-MHD weak form of AE3D for the potential, with $u=\mathbf b\cdot\nabla\Phi$:

$$
\omega^2\int\frac{\mu_0\rho}{B^2}\,\nabla\tilde\Phi^*\cdot\nabla\Phi\,dV=\int\nabla\tilde u^*\cdot\nabla u\,dV .
$$

Linear finite elements in $s$ times the harmonics $e_j$ give $Fy=\omega^2Gy$ with block-tridiagonal $F$ and $G$ of block size $H$. Harmonics with $m\ne0$ vanish on the axis, $m=0$ is flat in the first interval, $\Phi=0$ at $s=1$. An independent quadrature of the three bending terms proportional to $\iota'$ agrees with this operator to 0.1 to 0.3% and with AE3D v31 to 2 to 37%; the patch for AE3D goes upstream (step 1.7).

Added: the basis.

- Both signs of every harmonic. The set $\{(m,n)\}$ is completed with $\{(-m,-n)\}$ whenever the family couples to its conjugate, which happens when $2n_0\equiv0\pmod{N_{fp}}$. On the quasi-axisymmetric example at $N_{fp}=2$, $n_0=1$ the one-sign basis finds 102.355 and 104.579 kHz at 24 nodes and the full basis 103.128, 103.946 and 104.840 kHz (pull request 72, step 0.1).
  Corrected 2026-11-21: these numbers are regressions at $m\le5$ and are not converged in $m$: the first mode moves by $-0.15\%$, from 103.128 to 102.971 kHz (overlap 0.997), when the harmonic set of Paul et al. at $m_{\max}=12$ replaces $m\le5$, and the gap around 103 kHz narrows from 91.24–127.58 kHz at $m_{\max}=12$ to 102.35–103.72 kHz at $m_{\max}=30$. They are to be regenerated at the $m_{\max}$ that passes the gate of step 2.6; until then they are labelled regressions and are not quoted as physics.
- Parity sectors. For stellarator-symmetric geometry the completed space splits into the cosine and sine sectors $\Phi^\pm_{mn}=(\Phi_{m,n}\pm\Phi_{-m,-n})/\sqrt2$, each real and of half the size (pull requests 67 and 68). Asymmetric geometry uses the full complex basis. The archived STELLGAP and AE3D comparisons use different sectors; each fixture states its sector.
- Harmonic sets after Paul et al. (2025): for each $m\le m_{\max}$ the $n$ of one family in $[\iota_{\min}m-W,\ \iota_{\max}m+W]$ with $W=2N_{fp}$; $m_{\max}$ is chosen by a convergence test whose result is reported with the spectrum (step 2.6).

Added: radial discretization. Elements of order $p$ grouped per element, so that the pencil stays block tridiagonal with blocks of size $pH$; Gauss quadrature in each element; nodes placed on the half-mesh of the equilibrium; the coordinate $\rho=\sqrt s$ as an option, with the axis behaviour $\rho^{|m|}$ in the first element (steps 1.1 and 2.5).

Added: the current and pressure terms, `wave = "current"`. The model is the one CKA solves (Bowden, Hole and Könies 2015, eq. 1, without its fourth-order term, which is in 8.8):

$$
-\omega^2\,\nabla\cdot\Big(\frac{1}{v_A^2}\nabla_\perp\Phi\Big)
=\nabla\cdot\Big\{\mathbf b\,\nabla\cdot\Big[\Big(1-\frac{\mu_0p}{B^2}\Big)\nabla_\perp(\mathbf b\cdot\nabla\Phi)\Big]\Big\}
+\nabla\cdot\Big\{\mathbf b\times\boldsymbol\kappa\,\frac{2\mu_0\,\mathbf b\times\nabla p}{B^2}\cdot\nabla\Phi\Big\}
-\nabla\cdot\Big\{\frac{\mu_0j_\parallel}{B}\,\nabla\times\big[\mathbf b\,(\mathbf b\cdot\nabla\Phi)\big]\Big\}.
$$

The option `finite_beta=True` of 1.2.0 already holds the kink and pressure-curvature terms of the ideal-MHD energy for the $E\times B$ displacement,

$$
\omega^2\!\int\!\frac{\mu_0\rho}{B^2}|\nabla_\perp\Phi|^2dV=\int\Big[\,|\nabla_\perp u-\boldsymbol\kappa u|^2+\sigma\,\mathbf b\cdot\big(\nabla\Phi^*\times(\nabla u-\boldsymbol\kappa u)\big)-2\mu_0\frac{(\mathbf b\times\nabla\Phi\cdot\nabla p)(\mathbf b\times\nabla\Phi^*\cdot\boldsymbol\kappa)}{B^2}+\Big|\sigma u-\frac{\mu_0(\mathbf b\times\nabla p)\cdot\nabla\Phi}{B^2}\Big|^2\Big]dV,
$$

with $\sigma$, $p'$ and $B_s$ reconstructed from $G$, $I$ and $|B|$. Step 3.1 compares the two forms term by term, takes $\sigma$ and $p'$ from the equilibrium, and removes the label "experimental" when the gates are met. The current term is not optional for damping: without it the continuum damping of the Könies and Kleiber case is 3.4 times too large (8.8).

### 8.4 The sound-coupled global operator and $\delta B_\parallel$

This is the first new model. It extends the surface pencil of 8.2 to a global Hermitian pencil that is linear in $\omega^2$.

Unknowns per harmonic: the stream function $\chi$ of the perpendicular displacement and the parallel displacement $c$,

$$
\boldsymbol\xi_\perp=\frac{\mathbf b\times\nabla\chi}{B},\qquad \xi_\parallel=B\,c,\qquad \chi=\frac{i}{\omega}\Phi .
$$

The compressional Alfvén wave is eliminated by perpendicular pressure balance, $B\,\delta B_\parallel/\mu_0+\delta p=0$ with $\delta p=-\boldsymbol\xi_\perp\cdot\nabla p-\gamma p\,\nabla\cdot\boldsymbol\xi$. Writing $d=\nabla\cdot\boldsymbol\xi_\perp+2\boldsymbol\kappa\cdot\boldsymbol\xi_\perp$ and $D_0=\mathbf B\cdot\nabla c-2\boldsymbol\kappa\cdot\boldsymbol\xi_\perp$, the energy $(B^2/\mu_0)d^2+\gamma p\,(d+D_0)^2$ is stationary at $d=-(\mu_0\gamma_Bp/B^2)D_0$ and equals $\gamma_Bp\,D_0^2$ there. The functionals are

$$
\delta W=\frac12\int\Big[\frac{1}{\mu_0}\big|\nabla(\mathbf b\cdot\nabla\chi)\big|^2+\gamma_Bp\,\big|\mathbf B\cdot\nabla c-2\boldsymbol\kappa\cdot\boldsymbol\xi_\perp\big|^2+\delta W_{J,p}\Big]dV,\qquad
T=\frac12\int\rho\Big[\frac{|\nabla_\perp\chi|^2}{B^2}+B^2|c|^2\Big]dV,
$$

with $\delta W_{J,p}$ the kink and the pressure-curvature term of 8.3 (its second and third terms), and $\delta W=\omega^2T$. The fourth term of 8.3 is not part of it: for $\boldsymbol\xi_\perp=\mathbf b\times\nabla\Phi/B$ one has $d=[\sigma u-\mu_0(\mathbf b\times\nabla p)\cdot\nabla\Phi/B^2]/B$, so that term is $B^2d^2$, the compression energy that the elimination above replaces by $\gamma_Bp\,D_0^2$; kept, it would count the compression twice. The price is that the $\chi\chi$ block then differs from the operator of 8.3 by that term, so step 3.3 derives both functionals from one displacement and tests the two limits ($\gamma p\to0$ and large $\gamma p$) before the term is dropped or kept, and carries it as a named switch until then. The discrete pencil has blocks of size $2H$; its $\chi\chi$ block is the operator of 8.3 up to the term just named, its $cc$ block holds no radial derivative, and the coupling is the geodesic term. The spectrum does not depend on the sign of $\kappa_s$ (it changes under $c\to-c$ only), but $\delta B_\parallel$ below does, and the explicit formula for $\kappa_s$ of 8.1 changes sign with the handedness of the coordinates (corrected 2026-11-21: only if $|\sqrt g|$ or $|\psi'|$ is used; with both signed, the convention of 8.1, it does not): the sign is fixed by a polarization test, the phase of $\xi_\parallel$ relative to $\xi_s$ against Cheng and Chance (1986, eq. 13b). The magnetic compression follows without a new unknown,

$$
\frac{\delta B_\parallel}{B}=\frac{\mu_0\gamma_Bp}{B^2}\big(\mathbf B\cdot\nabla c-2\boldsymbol\kappa\cdot\boldsymbol\xi_\perp\big)+\frac{\mu_0\,\boldsymbol\xi_\perp\cdot\nabla p}{B^2},
$$

and is stored with the mode, because the kinetic coupling of 8.6 needs it. Step 3.3 derives the operator from the fluid energy of Cheng (1992, eq. 3.70), states every dropped term in `theory.md`, and fixes the sign of the last term above.

What must hold, in this order (steps 3.3 to 3.5): the limit of large radial wavenumber reproduces the surface pencil of 8.2 to rounding; the geodesic acoustic frequency and the acoustic gap top appear as global modes; the system of Cheng and Chance (1986, eqs. 7 to 9), which is nonlinear in $\omega$, gives the same modes on the same equilibrium, with the $n=1$ TAE at $(\omega/\omega_A)^2\approx0.50$; the frequencies fall with $\beta$ to those of 8.3 as $\beta\to0$; the sound-coupled modes of a stellarator case with a published counterpart are found (HSX in Könies and Eremin 2010, H-1 in Bertram et al. 2012).

What is not in it: the compressional Alfvén branch, anisotropic pressure, flow, and the kinetic closure of the sound wave, which is strongly Landau damped at $T_i\approx T_e$. The fluid sound continuum is therefore reported with that caveat and replaced by the kinetic inertia of 8.8 where damping is asked for.

### 8.5 Wave energy and the power ledger

For a mode of the Hermitian pencil with the stored amplitude,

$$
W=\frac{2\pi^2}{\mu_0}\,y^\dagger Gy\quad\text{(joules)},\qquad 2\gamma W=\sum_aP_{a\to w},
$$

where $G$ is the inertia matrix with its factor $\mu_0\rho$ and the $4\pi^2$ comes from the two angle integrals. On the two 1 V Gaussian harmonics of the ITPA case an independent finite-element integral gives $W=4.279791098\times10^{-5}$ J, which is a test. For a problem that depends on $\omega$ through a kinetic response, $D(\omega)y=[F-\omega^2G-\Pi(\omega)]y=0$, the energy is $W=-(\pi^2/\mu_0\omega)\,y^\dagger\partial_\omega D\,y$, which reduces to the form above when $\Pi=0$.

`stability` returns one ledger per mode: $W$; the power of each channel, split by species, orbit class (passing co- and counter-going, trapped) and source term (the energy derivative and the radial derivative of $F_0$); the outgoing flux of the kinetic Alfvén wave; the continuum absorption; and $\gamma$ as their sum over $2W$. Rules: the sum of the bins reproduces each total to rounding; a rigid rescaling of the mode changes $P$ and $W$ by the same factor and leaves $\gamma$ unchanged; a mechanism enters once, so that continuum absorption computed by a complex contour and the flux of a converted kinetic wave are not both added for the same resonance; several species share one $W$. For two modes closer than their line widths the ledger is the $2\times2$ projected power and energy matrices, whose eigenvalues do not depend on the basis of the pair.

### 8.6 Kinetic drive: resonance sums

Present. For a fixed mode, passing ions at zero orbit width and zero Larmor radius. With $\xi=v_\parallel/v$, $\lambda=(1-\xi^2)/B$, $D=G+\iota I$, $h=1/(B|\xi|)$ and $w=1/(|D|\langle h\rangle)$, the orbit frequencies are $\Omega_\zeta=vw$ and $\Omega_\theta=\iota\,\Omega_\zeta$, and a harmonic $(j,k)$ of the orbit resonates at $v_*=\omega/(|j\iota-k|\,w)$. With the drift work written as $Ze\,\mathbf v_d\cdot\delta\mathbf E=-m_fv^2C$ and $C_{jk}$ its harmonic in the orbit angles, the mode harmonics summed before squaring,

$$
\gamma=\frac{\pi^2\mu_0m_f^2}{2\,y^\dagger Gy}\int|\psi'|\,ds\int d\lambda\sum_{j,k}\frac{v_*^6\,(QF_0)\,|C_{jk}|^2}{|j\iota-k|\,w^2},\qquad QF_0=\partial_EF_0+\frac{j}{\omega Ze\,\psi'}\,\partial_sF_0 .
$$

The sidebands $(j,k)=(m+\mu,\ n+\nu N_{fp})$ are the resonances that the harmonic $(\mu,\nu)$ of $|B|$ opens in a stellarator (Kolesnichenko et al. 2002). One sign of $v_\parallel$ resonates for each signed $(j,k)$; both signs contribute across the sidebands. On the ITPA case ($\omega=4\times10^5$ rad/s, two Gaussian harmonics $m=10,11$, $n=6$, 200 surfaces, 32 pitches) the code gives:

| At 82, 139, 163, 191 keV | 82 | 139 | 163 | 191 | Unit |
|---|---|---|---|---|---|
| Power to the wave | 0.21324 | 0.75218 | 1.00299 | 1.30369 | W |
| $\partial_E$ part of $\gamma$ | $-687.06$ | $-1285.97$ | $-1431.75$ | $-1560.41$ | s$^{-1}$ |
| $\partial_s$ part of $\gamma$ | $+3178.28$ | $+10073.55$ | $+13149.48$ | $+16791.16$ | s$^{-1}$ |
| $\gamma$ | 2491.22 | 8787.58 | 11717.72 | 15230.76 | s$^{-1}$ |

The sidebands $j=9$ and 12, at $|v_\parallel|\approx v_A/3$, carry 98.8 to 99.8% of it. Against published curves, in $10^3$ s$^{-1}$:

| Series | 82 | 139 | 163 | 191 |
|---|---|---|---|---|
| This code, prescribed mode | 2.49 | 8.79 | 11.72 | 15.23 |
| This code, solved finite-pressure mode | 2.84 | 9.77 | 12.97 | 16.78 |
| CAS3D-K, Könies et al. (2018) fig. 1a | 4.10 | 14.65 | 19.70 | 24.95 |
| CAS3D-K, same paper fig. 1b | 2.89 | 13.45 | 18.50 | 23.74 |
| CAS3D-K net rate, Fehér (2013) fig. 5.8 | 1.68 | 12.25 | 17.30 | 22.56 |
| STAE-K, Slaby et al. (2016) fig. 5 | 2.68 | 9.76 | 13.16 | 17.65 |

The three CAS3D-K series differ by constants and share one slope, 0.191 per keV, against 0.117 for the prescribed mode here: an additive damping correction cannot close the gap. The thesis curve is a net rate at zero orbit width, piecewise linear between computed points, with $\gamma(175)-\gamma(100)=15.8$, $\gamma(200)-\gamma(100)=20.0$ and $\gamma(300)-\gamma(200)=22.3\times10^3$ s$^{-1}$; gates compare such increments. Its fast-ion density profile equals the formula of the 2018 paper only if its radial variable is the square root of the normalized poloidal flux. The ratio of 0.60 is a diagnostic and is never turned into a factor.

Added, in this order.

1. Resonance weights (step 5.1). The geometry and orbit part is computed once per mode and equilibrium and stored as three arrays over resonance, surface and pitch: a resonant energy $e_i$ and two weights. For a Maxwellian of uniform temperature $\gamma(T)=T^{-3/2}\sum_ie^{-e_i/T}\,[A_i-B_i/T]$, and at a frequency $x\omega$ on the same mode $e_i\to x^2e_i$, $A_i\to x^5A_i$, $B_i\to x^6B_i$. Every distribution, temperature and species of the ledger is a sum over the same weights. Changing the equilibrium, the mode or a profile rebuilds them inside the derivative graph.
2. The full radial derivative (step 1.5), $\partial_sF_0$ at fixed $E$ and $\mu$ for every distribution, including temperature, critical energy, birth energy and anisotropy profiles.
3. The kinetic coupling of Könies (2000, eqs. 31 and 82 to 85), which acts on the perturbed Lagrangian of the displacement,

$$
L=-(m_fv_\parallel^2-\mu B)\,\boldsymbol\xi_\perp\cdot\boldsymbol\kappa+\mu B\,\nabla\cdot\boldsymbol\xi_\perp=-(m_fv_\parallel^2+\mu B)\,\boldsymbol\xi_\perp\cdot\boldsymbol\kappa-\mu\,\delta B_\parallel ,
$$

the second form at low $\beta$. The first term is the drift work of 1.2.0; the second needs $\delta B_\parallel$ of 8.4. Step 5.2 evaluates both contractions on the same orbit before squaring, with a slow reference written from the paper's functional for one passing orbit and two harmonics: the source operator applied to $L$ before the orbit projection must equal the multiplier $QF_0$ applied after it. The candidate mapping of the radial term to mode numbers, $(mG+nI)/(G+\iota I)$ in place of $j$, is tested there and adopted only if the reference requires it.
4. The orbit-averaged drift in the resonance (step 5.3). The denominator of Könies (2000, eq. 82) is $\omega+\sigma p\,\omega_c-m\langle\omega_d^\vartheta\rangle-n\langle\omega_d^\varphi\rangle$ in that paper's convention, with $\omega_c$ the transit or bounce frequency and $\langle\omega_d\rangle$ the orbit-averaged drifts; it is mapped to the convention of section 7 in the step. The drift term shifts the passing resonances at high energy and gives trapped ions their precession resonance.
5. Trapped ions (step 5.3). Bounce harmonics $\ell$ of the drift work over the closed bounce orbit, with resonance $\omega=\ell\,\omega_b+n\,\bar\omega_{\rm pr}$. In one magnetic well with $B=B_0(1-\varepsilon\cos\theta)$ the substitution $\sin(\theta/2)=k\sin u$, $k^2=(1-\lambda B_{\min})/(2\lambda B_0\varepsilon)$, removes the turning-point singularity; in a stellarator the bounce integrals come from the bounce kernels of VMEX over every well, with the well topology held fixed along a design step (8.9). At zero precession on the ITPA case reflected ions give 0.023, 1.36, 4.00 and 11.07 s$^{-1}$, under 0.1% of the passing rate, although they are 20% of phase space at $s=0.25$; with the precession in the resonance they add 7 to 29%. Both numbers are tests.
6. Distributions (step 5.4): the isotropic slowing-down form $F\propto(v^3+v_c^3)^{-1}$ below the birth speed, with alphas from profiles as $F_\alpha=S_\alpha\tau_s/[4\pi(v^3+v_c^3)]$; an anisotropic slowing-down form for beams; a bi-Maxwellian. Each is normalized to its density and stationary under the retained unperturbed motion; a toroidal canonical momentum is used as a label only in axisymmetry.

Absolute checks that already hold and stay as tests: the damping of a flat Maxwellian in the weak-toroidicity limit, eq. 34 of Kolesnichenko and Tykhyy (2018),

$$
\frac{\gamma}{\omega}=-\frac{\sqrt\pi}{8}\,\frac{m_fn_f}{\rho}\,\frac{Q(x)}{\kappa^2},\qquad Q(x)=\frac{(2x^4+2x^2+1)\,e^{-x^2}}{x},\quad x=\frac{|v_{\parallel\rm res}|}{v_{\rm th}},\quad\kappa=(m+\mu)\iota-n,
$$

times a radial factor that is one for $\mu=\pm1$, met to $2.2\times10^{-5}$ at the $v_A$ resonance and at both $v_A/3$ resonances (for example $\gamma/\omega=-0.00998227$ at $x=1$); the coefficient of Fu and Van Dam (1989); invariance of $\gamma$ under a complex rescaling of the mode; exact zero of the radial term and damping by every resonance for a flat Maxwellian; reversal of the radial term under $(m,n)\to(-m,-n)$.

### 8.7 The orbit-integrated evaluator

The same power, computed in ESSOS without a resonance expansion, is the independent evaluator that every kinetic number needs. Markers sit on a deterministic quadrature of phase space. Each follows its unperturbed guiding-centre or full orbit in Boozer coordinates while the work rate of the wave, $q\,\mathbf v\cdot\delta\mathbf E$ and $\mu\,\partial_t\delta B_\parallel$, is accumulated with a window in time. The power is

$$
P_{f\to w}=\int d\Gamma\,\big[D_{EE}\,\partial_EF_0+D_{E\bar s}\,\partial_{\bar s}F_0\big],
$$

with $D_{EE}$ and $D_{E\bar s}$ the diffusion coefficients of the accumulated work and of the orbit-averaged surface $\bar s$. The sign is that of section 7: the power to the particles is $-P_{f\to w}$, and a Maxwellian, with $\partial_EF_0<0$, damps the wave. A complex frequency $\omega+i\nu$ gives convergence in time and stands for collisional decorrelation; a gyro-average of the fields gives the Larmor radius. In the limit of zero width and zero Larmor radius it must reproduce 8.6: a prototype outside ESSOS meets the passing drive to 0.3 to 0.45% and gives 9.9, 25.5 and $30.5\times10^3$ s$^{-1}$ at 100, 200 and 300 keV with full drift orbits: the first two are inside the band of the published codes (8 to 10 and 25 to 33) and the third is below it (38 to 43). The distribution label matters once orbits have width: the flux surface, the orbit-averaged surface and, in axisymmetry, $P_\zeta$ give different answers (a factor near two in one published NOVA-K comparison), and the choice is reported with every finite-width result. The evaluator runs at the validation tier; the resonance sums are what an optimization calls.

### 8.8 Damping and the thermal kinetic response

Release 1.2.0 computes no damping except the Landau damping of passing bulk ions through the drive kernel, and that is normalized by the inertia of its own species; in the ledger every species shares the total $W$.

**Continuum damping.** A Hermitian pencil on the real axis has no damped mode; the damped mode is a pole of the resolvent. Three methods, in the order of use.

1. Resolvent fit, the reference, which needs only the Hermitian matrices. Sample $R(z)=b^\dagger(F-z^2G)^{-1}b$ on a line $z=\omega+i\eta$ above the real axis, fit a rational function, read the pole in the lower half plane. The height must exceed the local spacing $\Delta\omega$ of the discrete continuum, $\eta\ge\Delta\omega\,\ln(2/\mathrm{rtol})/2\pi$, and the pole must not move when $\eta$, the radial mesh and the right-hand side change.
2. Small dissipation: the fourth-order term below with a small imaginary coefficient, $-i\delta\,\nabla_\perp^2(\omega^2v_A^{-2}\nabla_\perp^2\Phi)$, in the double limit $\delta\to0$, $N\to\infty$ on a mesh packed at the resonance. It is the only method that also holds radiative damping, and it works on a numerical equilibrium.
3. A contour in complex $s$, $s=x+i\alpha x(x-1)$ (Bowden, Hole and Könies 2015, eqs. 5 to 7), where the equilibrium has an analytic continuation: analytic profiles, or harmonics fitted by polynomials in $s$. The operator on the contour is complex symmetric and is never symmetrized.

The current term of 8.3 is part of the model: Berk et al. (1992) show that it cancels part of the bending term at low $\beta$, and without it the damping of the Könies and Kleiber case is 3.4 times too large. Reference values: the tokamak case of Könies and Kleiber (2012; aspect ratio 10, $q=1.05+0.55r^2$, density $\tfrac12[1-\tanh((r-0.8)/0.1)]$, harmonics $m=2,3$ of $n=2$ in this file's convention) has $\mathrm{Im}\,\Omega^2\approx-8.9\times10^{-4}$ read from its figure 2, and the resolvent fit with the current term and harmonics $m=1$ to 4 gives $-8.84\times10^{-4}$; the gate keeps that harmonic set, since the truncation to two harmonics moves the imaginary part by a few per cent. The two-harmonic model of Bowden et al., $\Omega=0.370887-0.00174204i$, is reproduced to six digits by a one-dimensional contour solve of its equations and is the exact referee. Their full tokamak value $0.39515-0.00174465i$ (resistive check $0.395151-0.00174521i$) is a gate only after the $\iota$ profile behind their figure 1 is known; their W7-X value is not used, because its text and figure differ by a factor 7.5; their H-1 value is deferred, because the printed ratio and the printed frequency disagree. Their harmonics $(2,-2)$, $(3,-2)$ are $(2,2)$, $(3,2)$ here.

**The non-ideal term, kinetic Alfvén waves and radiative damping.** Finite ion Larmor radius and the parallel electric field add a fourth-order term. In weak form, added to the bending side,

$$
\delta F(\tilde\Phi,\Phi)=\int\varrho_K^2\,\big[\mathbf b\cdot\nabla(\nabla_\perp^2\tilde\Phi)\big]^*\big[\mathbf b\cdot\nabla(\nabla_\perp^2\Phi)\big]\,dV,\qquad
\varrho_K^2=\tfrac34\rho_i^2+\Lambda_e\,\rho_s^2,\quad\rho_i^2=\frac{m_iT_i}{q_i^2B^2},\quad\rho_s^2=\frac{m_iT_e}{q_i^2B^2},
$$

which is eq. 4 of Candy and Rosenbluth (1994) to first order in the inverse aspect ratio and the fourth-order term of CKA with the ion part written on the inertia side. $\Lambda_e$ is the electron response: 1 in the ideal limit of massless electrons, $1-i\delta_e$ with the closed form of Candy and Rosenbluth for collisionless and collisional electrons, and the forms of Mett and Mahajan (1992) and Berk, Mett and Lindberg (1993) in their limits. Linear elements cannot hold a fourth-order term, so the discrete form is mixed, with a vorticity unknown $w=\nabla_\perp^2\Phi$ per harmonic and blocks of size $2H$. The local dispersion relation is $\omega^2=k_\parallel^2v_A^2[1+k_\perp^2(\tfrac34\rho_i^2+\rho_s^2)]$. With a real $\varrho_K^2$ the pencil is Hermitian and yields the kinetic TAEs above the gap but no damping. Radiative damping is the flux of the converted wave away from the gap; it needs a small $\mathrm{Im}\,\Lambda_e<0$, or an absorbing layer, to remove that wave, and the result must be shown not to depend on it. The layer parameter that orders the regimes is $\lambda_K^2=16m^2\hat s^2\varrho_K^2/(\hat\varepsilon^3r^2)$ with $\hat s$ the shear. In the tunnelling regime $-\gamma/\omega_0=(\pi^2\hat s^2\hat\varepsilon/12)\,e^{-\pi a/2}$ with $a=\pi^2\hat s^2/(4\sqrt2\,\lambda)$ (Breizman and Sharapov 2025, eq. 42; the same exponent with $1/8$ in Berk, Mett and Lindberg 1993). No published formula covers $0.5<a<3$, so that range is checked against the layer equation of Candy and Rosenbluth (their eq. 28) solved as an ordinary differential equation in the tests. The recursion and Fourier-layer methods of those papers are specific to tokamaks and serve only as oracles.

**Thermal species on a computed mode.** One perturbative form covers electron Landau damping, ion Landau damping and their trapped parts (Fu, Cheng and Wong 1993, eqs. 18, 19, 22, 35, 39):

$$
\frac{\gamma}{\omega}=\frac{\mathrm{Im}\,\delta W_k}{2\omega^2\,\delta K},\qquad
\delta W_k\propto\sum_{\rm orbits}\int d\Gamma\,(\omega-\omega_*)\,\frac{\partial F}{\partial E}\,\frac{|G_p|^2}{\omega-\langle\omega_d\rangle-p\,\omega_{t,b}},\qquad
G=\frac{q}{2E}\Big(\frac{i}{\omega}\,\mathbf v_d\cdot\delta\mathbf E_\perp+\Psi+\frac{\mu}{q}\,\delta B_\parallel\Big),
$$

with $p$ the transit harmonic for passing and the bounce harmonic for trapped particles. This is the kernel of 8.6 with two more couplings: the magnetic compression of 8.4 and the potential $\Psi$ of the parallel electric field, $\delta E_\parallel=-\mathbf b\cdot\nabla\Psi$, which in the long-wavelength limit is $\Psi\approx\rho_s^2\,\nabla\cdot\delta\mathbf E_\perp-(i/\omega)\langle\mathbf v_{de}\rangle\cdot\delta\mathbf E_\perp$ and otherwise comes from the vorticity unknown above. In those papers $\omega_*$ acts on the field harmonics, so the gradient term carries the mode number $m$ of the field; for a distribution $F(E,\mu,s)$ the orbit harmonic $j$ of 8.6 is the consistent choice. The two agree for isotropic distributions summed over resonances and differ for beams (weights 2 and 1 against 1.5 at $n=1$, $q=1.5$); the reference of step 5.2 decides. The resonance weights of 8.6 are computed once for each species mass and charge and reused.

Collisional damping on trapped electrons uses the bounce-averaged equation with pitch-angle scattering, $(-i\omega-\langle C\rangle)g=\langle H\rangle$, expanded in the eigenfunctions $g_l$ of the bounce-averaged Lorentz operator (same paper, eqs. 40 to 48): each term carries the factor $1/(1+iC_E\lambda_l)$ with $C_E\propto\nu_e/\omega$, so the damping grows as about the square root of the collision frequency below $\nu_{\rm eff}/\omega\approx0.1$, $\nu_{\rm eff}=2Z_{\rm eff}\nu_e/\delta\Lambda$, and saturates above. The analytic low-collisionality limit scales as $\beta_e(\nu_e/\omega)^{1/2}$ with logarithmic factors (Gorelenkov and Sharapov 1992), from the curvature source alone, which the numerical solution finds about ten times smaller than the parallel-field source. On the TFTR case of Fu, Cheng and Wong the numerical rate is $\gamma_e/\omega=-3.0$, $-5.3$, $-7.0$, $-9.0$ and $-11.1\times10^{-4}$ at $\nu_{\rm eff}/\omega=0.035$, 0.10, 0.18, 0.37 and 1.29, half or less of the analytic estimate. In a stellarator the eigenfunctions are those of each trapping class; where that expansion is not practical the finite-frequency drift-kinetic solve of DKX with the linearized Fokker-Planck operator replaces it and is its check (step 5.7).

**The kinetic inertia of the low-frequency continuum.** Below the TAE gap the fluid sound branch of 8.2 is not a valid closure. Two steps. First, in the surface pencil, the kinetic adiabatic indices $\gamma p\to\tfrac74p_i+p_e$ and the diamagnetic inertia $\omega^2\to\omega(\omega-\omega_{*pi})$: the matrices stay Hermitian but the problem becomes quadratic in $\omega$, no longer a pencil in $\omega^2$, and it gives the leading-order kinetic frequency of the beta-induced gap, $\omega^2=(\tfrac74+\tau)\,2T_i/(m_iR_0^2)$ with $\tau=T_e/T_i$. This step is an inference from the fluid limits of the two sources below, not a formula of either, and is gated against them. Second, the local kinetic continuum (Zonca, Chen and Santoro 1996, eqs. 19 and 23; Lauber 2013, eq. 20), a scalar complex root on each surface,

$$
\omega^2\Big(1-\frac{\omega_{*pi}}{\omega}\Big)+\frac{2T_i}{m_iR_0^2}\,\frac{\omega}{\omega_{ti}}\,\mathcal C\Big(\frac{\omega}{\omega_{ti}}\Big)=k_\parallel^2v_A^2,\qquad
\mathcal C(x)=\Big(1-\frac{\omega_{*n}}{\omega}\Big)F(x)-\frac{\omega_{*T}}{\omega}G(x)-\frac{N^2(x)}{D(x)},
$$

with $F$, $G$, $N$, $D$ combinations of the plasma dispersion function and $\omega_{ti}=\sqrt{2T_i/m_i}/(qR_0)$. It gives the kinetic gap frequency with its ion Landau damping. The published form assumes one geodesic sideband with $k_\parallel=\pm1/(qR_0)$. In a stellarator every harmonic $(\mu,\nu)$ of $|B|$ adds a sideband with $k_\parallel'=[(m+\mu)\iota-(n+\nu)]B/(G+\iota I)$ and its own argument; that form is in none of the sources and is derived in step 4.9 from the kernel above. The coefficient functions are rederived, not copied: the Larmor-radius coefficient is misprinted in both Lauber papers, and the imaginary part of Zonca's eq. C.29 appears to lack an exponential factor.

**What is counted once.** Continuum absorption found with a contour and the flux of a converted kinetic wave are the same energy path seen in two models; a result quotes one. A species response solved with collisions already holds the collisional part; the separate collisional power is a partition of it, not an addition.

### 8.9 Complex and nonlinear eigenproblems, and their derivatives

Three kinds of eigenproblem occur, each with its derivative rule. The solvers and the rules live in SOLVAX; ALFVEX supplies the operators and checks the certificates.

Hermitian pencil, simple mode, $y^\dagger Gy=1$:

$$
d\lambda=y^\dagger\,(dF-\lambda\,dG)\,y .
$$

The derivative of the eigenvector needs a projected solve with a gauge condition. For a cluster of modes that are not separated compared with the solver tolerance and the scale of the response, the differentiable quantities are those of the subspace: the trace of $\lambda$ over the cluster, the spectral projector, and the eigenvalues of a physical bilinear form projected into the cluster. An objective that takes the largest drive over basis vectors of a degenerate pair depends on the rotation inside the pair and is not allowed.

Decided 2026-11-21 (pull request 80): the drive of a near-degenerate cosine/sine pair is computed from the $2\times2$ problem for reporting, and the pair is reported as an unordered set; an optimizer receives the cluster quantities, the largest eigenvalue of the power matrix projected into the pair and its trace. Reasons, measured on the quasi-axisymmetric pair: the $2\times2$ rates are $+14357$ and $-14692$ s$^{-1}$ where the diagonal rates and the complex two-sign basis both give $-160$ and $-175$ s$^{-1}$, so neither replaces the $2\times2$ problem; at the exceptional point (0.140 of the fast-ion density) the derivative of the per-mode $2\times2$ rates is off by 0.6 and 20 times steeper, and per-mode labels flip where the two diagonal rates are equal; a switch to the $2\times2$ problem below a threshold jumps by about 22 s$^{-1}$; the cluster quantities stay smooth through the exceptional point, at the finite-difference floor of $4\times10^{-3}$, measured at one step size.

Complex-symmetric pencil (a contour in complex $s$, or the kinetic-Alfvén term with a complex coefficient), with right and left vectors $y_R$, $y_L$:

$$
d\lambda=\frac{y_L^\dagger\,(dF-\lambda\,dG)\,y_R}{y_L^\dagger\,G\,y_R},\qquad d\omega=\frac{d\lambda}{2\omega}.
$$

In the real parity basis of 8.3 the pencil on a contour is complex symmetric, $F^T=F$ and $G^T=G$, so the left vector is the conjugate of the right one and no second solve is needed: $d\lambda=y^T(dF-\lambda\,dG)\,y/(y^TGy)$. In the complex Fourier basis the left vector comes from the problem on the conjugate contour. The operator is never symmetrized into a Hermitian one. The certificate holds both residuals, the biorthogonality $y_L^\dagger Gy_R$, and the change of the root under a change of contour or of the dissipation parameter.

Nonlinear problem $D(\omega,p)\,y=0$ with left vector $\ell$, $\ell^\dagger D=0$:

$$
\frac{d\omega}{dp}=-\frac{\ell^\dagger D_p\,y}{\ell^\dagger D_\omega\,y},\qquad
\begin{pmatrix}D&D_\omega y\\ c^\dagger&0\end{pmatrix}\begin{pmatrix}dy\\ d\omega\end{pmatrix}=-\begin{pmatrix}D_p\,y\,dp\\ 0\end{pmatrix},
$$

with $c$ a fixed gauge vector. $D_\omega$ includes the frequency dependence of the kinetic response. Where $D$ depends on $\omega^*$ or on absolute values, the real and imaginary parts are differentiated as a real system. A vanishing denominator means the root is not simple, and the evaluation is rejected.

Certificates returned with every solve: the signed eigenvalue; the residuals of the selected pairs; the residual of the nearest excluded pair; orthogonality; the selection gap; iterations; and, for derivatives, the residual of the tangent or adjoint solve. In SOLVAX 0.36.2 the convergence flag is computed after the excluded pair is sliced away: on a diagonal test pencil at tolerance $10^{-10}$ the selected residual is $1.16\times10^{-20}$, the excluded residual $5.85\times10^{-7}$, and the flag is true. Step 1.8 corrects this. A positive gap of the Ritz values is not a proof that the window is complete: small cases are checked against the dense spectrum in the window, large ones with overlapping shifts and independent starting subspaces.

Mode identity across a design step: modes are matched one to one by overlap in the inertia norm together with frequency; both the nodal and the inertia-norm overlap are reported; a guard band of harmonics and of frequency around the window detects modes that enter; an evaluation whose matching is ambiguous is rejected and the step is shortened (step 2.7). In 1.2.0 `classify_modes` lets several coarse modes claim one refined mode.

Non-smooth points. A sampled maximum of $|B|$ over the angle grid has a kink where the maximizing grid point changes: on a 64-point cosine the one-sided derivatives of $1/B_{\max}$ are $\pm7.42\times10^{-4}$ T$^{-1}$/rad at a tie while automatic differentiation returns zero. Extrema that enter a kernel are refined by a local quadratic fit (step 1.5). Corrected 2026-11-21 (pull request 77): $B_{\max}$ in the drive kernel is refined by a Newton solve on the Fourier series of $|B|$, started from the best grid sample, as a follow-up to step 1.5. On a quasi-helical field at the kernel grid the quadratic fit has an error of $5.1\times10^{-5}$ and its derivative jumps by $6.3\times10^{-4}$ T$^{-1}$/rad under a translation of the field, where the true derivative is zero; the Newton solve (four steps) gives $4.4\times10^{-16}$ and $2.7\times10^{-10}$ in 0.24 to 0.35 ms, against 1 to 2 s per growth rate. Since $d\ln\gamma/d\ln B_{\max}\approx-1.1$ at 150 keV, the rate inherits the relative error of $B_{\max}$. A smooth maximum is biased, and a finer grid costs more and keeps the jump. Trapped-passing boundaries, bounce points and resonant surfaces move with the design: their motion is differentiated with the topology held fixed, and a change of topology (a well that appears or merges) is detected and rejected, never smoothed silently.

### 8.10 The mode record and the interface to ESSOS

Under the ideal ordering the perturbation is $\delta\mathbf A=\alpha\mathbf B$ with $E_\parallel=0$, which gives for each harmonic

$$
\alpha_{mn}(s)=\frac{(m\iota-n)\,\Phi_{mn}(s)}{\omega\,(G+\iota I)}\quad\text{(metres)},\qquad \delta\mathbf B=\nabla\times(\alpha\mathbf B),\qquad \delta\mathbf E=-\nabla\Phi-\partial_t(\alpha\mathbf B).
$$

The guiding-centre Hamiltonian in ESSOS is $H=\tfrac12mv_\parallel^2+\mu B+q\Phi$ with $v_\parallel$ taken from the canonical momentum that contains $\alpha$; it holds no $\mu\,\delta B_\parallel$ term. A mode with finite $E_\parallel$ (8.8) supplies $\Phi$ and $\alpha$ as independent profiles; a mode with $\delta B_\parallel$ (8.4) needs the added term $\mu\,\delta B_\parallel$, which is step 6.7.

`export_mode` writes one mode: the fields of the `Modes` record on a radial spline with the axis behaviour $s^{|m|/2}$ and the zero edge value, signed $(m,n)$, $\omega$, $N_{fp}$, the sign of $\psi'$, the amplitude rule and value, and $W$. ESSOS reads this record through one array constructor in the same convention. The analytic wave class of ESSOS pull request 177 uses a sine with $+\omega t$ and non-negative $m$; for a coefficient $Ae^{i\varphi}$ the equivalent call is $|m|$, $en$, $-e\omega$, $A$ and phase $e\varphi+\pi/2$ with $e=-1$ for $m<0$ and $+1$ otherwise. That mapping reproduces $\Phi$, $\alpha$ and their first derivatives to $4.4\times10^{-15}$ on 504 points and is kept as a test of the constructor, not as the interface.

Tests of the interface (step 6.2): Faraday's law and $E_\parallel=0$ at shared points; regularity at the axis and periodicity; a sign error in time is detected (it gives a relative error of 1.28 in $\Phi$); energy and $P_\zeta$ conserved without the wave; $E+(\omega/n)P_\zeta$ conserved with a single-$n$ wave in an axisymmetric field, with the sign of $P_\zeta$ fixed by the handedness of section 7; its helical counterpart in a quasisymmetric field; the energy gained by the markers equal to the accumulated work $q\int\mathbf v\cdot\delta\mathbf E\,dt$; the derivative of the radial drift with respect to a field parameter equal to central differences (reached to $1.1\times10^{-11}$ with the live-array constructor of pull request 118, and failing with a tracer error in the constructor of pull request 177).

### 8.11 Saturation and transport

Amplitudes. With the mode structures fixed, the complex amplitudes $\Phi_j$ of the potential and $\mathcal A_j$ of the parallel vector potential obey (Slaby et al. 2024, eqs. 28 to 50, in laboratory-frame amplitudes)

$$
\sum_jn_{jk}\Big(\frac{d\mathcal A_j}{dt}+i\omega_j\Phi_j\Big)=\upsilon_k,\qquad
\sum_jm_{jk}\Big(\frac{d\Phi_j}{dt}+i\omega_j\mathcal A_j\Big)=-2\tau_k-2\gamma_{d,k}\sum_jm_{jk}\Phi_j,
$$

with $m_{jk}=\int(\rho/B^2)\,\nabla_\perp\Phi_{0j}\cdot\nabla_\perp\Phi_{0k}^*\,dV$ and $n_{jk}$ the same overlap of the vector potentials, $\tau_k$ the power of the fast-ion drift current on mode $k$ and $\upsilon_k$ the parallel-field term. For one mode this is $\gamma=P/(2W)-\Gamma_d$ of 8.5 with $W=\tfrac12m_{kk}\lvert\Phi_k\rvert^2$ and $P=-\mathrm{Re}(\tau_k\Phi_k^*)$. The power comes from the markers of ESSOS through the accumulated work of 8.10; modes couple through the overlaps and through the shared particles. The published damping $\gamma_d=1.05\times10^4$ s$^{-1}$ used with these equations is ten times the value of the ITPA paper it is attributed to; gates use the published rates only together with the value actually used.

Saturation levels. With $\omega_b^2$ proportional to the amplitude through a coefficient computed from the mode and the resonant orbits, the levels of Berk and Breizman are a small closed set, each with its regime:

| Regime | Level | Source |
|---|---|---|
| No collisions, no sources | $\omega_b=3.2\,(\gamma_L-\gamma_d)$ | collisionless bound |
| Steady state with renewal at rate $\nu$, far from threshold | $\omega_b\approx1.9\,\nu\,\gamma_L/\gamma_d$ (1.96 from the paper's own integrals) | Berk and Breizman (1990, I) |
| Pitch-angle scattering at rate $\nu_d$ | $\nu_{\rm eff}=\nu_d\,\omega^2/\omega_b^2$, $\omega_b\approx(\nu_d\,\omega^2\gamma_L/\gamma_d)^{1/3}$ | Berk and Breizman (1990, II) |
| Near threshold, renewal | $\omega_b=8^{1/4}\,\nu\,(1-\gamma_d/\gamma_L)^{1/4}$, stable for $\hat\nu=\nu/(\gamma_L-\gamma_d)>4.38$ | Berk, Breizman and Pekker (1996) |
| Near threshold, drag alone | no steady state; explosive growth | Lilley, Breizman and Sharapov (2009) |

`saturation` in `transport.py` returns the level and names the regime. The near-threshold equation

$$
\frac{dA}{d\tau}=A-\frac12\int_0^{\tau/2}dz\,z^2A(\tau-z)\int_0^{\tau-2z}dx\,e^{-\hat\nu(2z+x)}A(\tau-z-x)\,A^*(\tau-2z-x)
$$

is solved in the tests as an oracle: $\hat\nu=5$ from $A(0)=1$ settles at $A=2\sqrt2\,\hat\nu^2=70.71$; oscillations start at $\hat\nu=4.38$; $\hat\nu=2.4$ diverges before $\tau=30$. The effective rate at the resonance follows from the collision operator of ESSOS through $D_{\Omega\Omega}\simeq\nu_d(1-\xi^2)(\partial_\xi\Omega)^2+D_r(\partial_r\Omega)^2$ (Chen and Zonca 2016, eqs. 4.155 and 4.156).

Transport. Two models, the second a comparison only.

1. The quasilinear flux from the diffusion coefficients of 8.7 at the saturated amplitude, and the amplitude at which neighbouring resonances overlap. Both use quantities the code already computes and are checked against ESSOS losses (step 7.6).
2. The reduced model of Slaby et al. (2025, eqs. 10 to 17): $D(s)=\bar D\pi^2\sum_iH(\gamma_i)\,\gamma_i^3/[k_{\perp i}^2(\omega_i^2+\gamma_i^2)]\,e^{-(s-s_i^*)^2/\Delta_i^2}E(s)+D_{\rm turb}$ with $E(s)=\tanh[10(1-s)]$, summed over continuum crossings, with the density relaxing toward its slowing-down value (source $+S_0(1-n_f/n_f^{SD})$; their eq. 15 prints the opposite sign). It has three free parameters, no loss channel, and its result for a reactor case changes forty-fold with the mode-number cut-off. It is implemented in a test as a published comparison, not as an objective.

Reference cases: the ITPA single mode of Slaby et al. (2024), with public inputs (CKA frequency 63.11 kHz; $\gamma_L=4.723\times10^4$ s$^{-1}$ at doubled fast-ion density; saturation at 2.955 kV, $\delta B/B=2.1\times10^{-3}$); the tokamak case of Slaby et al. (2025) ($R_0=3$ m, $a=1$ m, 3 T; gap modes at 205.0, 178.4 and 154.9 kHz); and the W7-X discharge 20181009.024 at 3.9 s of Slaby et al. (2020), with modes at 56.7, 39.9 and 191.2 kHz, a fast-ion drive of a few hundred s$^{-1}$ against ion damping of about $-4\times10^3$ s$^{-1}$, and all 26 modelled modes stable although modes were observed. That last result is the reason the plan does not treat linear stability as proof of absence of mode activity.

### 8.12 Objectives

From cheap to costly; each level is validated against the next (step 7.6).

1. Geometric. The continuum cost of Paul et al. (2025, eq. 7.4), $f_{\rm cont}=\sum_s\sum_{|\iota\delta m-N_P\delta n|>|\iota-4|}|\varepsilon_1^{\delta m,\delta n}|^2(\iota\delta m-N_P\delta n)^2$, on the harmonics of the metric factor $\varepsilon_1=|\nabla\psi|^2/\langle|\nabla\psi|^2\rangle-1$ (their eq. 4.28), restricted by the frequency cut $|\iota\delta m-N_P\delta n|>|\iota-4|$. Its leading two-mode gap is proportional to $|\varepsilon_1+\varepsilon_2|$; the root-sum-square of the paper's eq. 4.30 does not reduce to its eq. 4.9 for equal coupling parameters, so tests use the $2\times2$ calculation.
   Corrected 2026-11-21: this item described the cost of Paul et al. as one on the compression harmonics of $|B|$. Eq. 7.4 (Paul et al. 2025, p. 24, checked on the page) sums $|\varepsilon_1|^2$ only, with $\varepsilon_1$ of eq. 4.28 (p. 16), the harmonics of $|\nabla\psi|^2$, and with the cut $|\iota\delta m-N_P\delta n|>|\iota-4|$, which the item did not state; $|B|$ enters only through $\varepsilon_2=(|\nabla\psi|^2/B^4)/\langle|\nabla\psi|^2/B^4\rangle-1$ of the same equation, which eq. 7.4 does not use. A cost on $|\varepsilon_1+\varepsilon_2|$ is therefore a choice of this plan, not the cost of the paper. Page 24 also says that each summand quantifies the squared gap width, not the width.
   Decided 2026-11-21: both geometric costs are kept. Eq. 7.4 exactly as published ($\varepsilon_1$ only, the hard cut, the helicity of the device) is the control of step 8.3, the only cost that can be called a reproduction of Paul et al. (2025); the same sum on $|\varepsilon_1+\varepsilon_2|$ is the objective of this level. Reason, from two-harmonic surface solves: the gap width equals $|\varepsilon_1+\varepsilon_2|$ within 1.5% on 28 of 28 gaps (quasi-axisymmetric and quasi-helical), while $\varepsilon_1$ alone is off by factors 0.02 to 2.8 and eq. 4.30 by up to $\sqrt2$. Both costs rank the quasi-axisymmetric case above the quasi-helical one, but they rank a family of five quasi-axisymmetric boundaries in opposite order (Spearman $-1$); that opposite ranking is reported as a result. Not yet measured: the correlation of either cost with the widest low-frequency gap.
2. Spectral. A smooth penalty on resolved modes inside a window, weighted by the resonance density of the fast-ion distribution, $w(\omega)\propto\sum_{jk}\int d\Gamma\,F_0\,\delta(\omega-j\Omega_\theta+k\Omega_\zeta)$, in place of a window built from the starting eigenvalues. The alternative that needs no mode selection is the forced response $\int_{\omega_1}^{\omega_2}\mathrm{Im}\,b^\dagger[F-(\omega+i\eta)^2G]^{-1}b\,d\omega$ of 8.8. Both are normalized by the Alfvén frequency and evaluated over a set of density profiles, because a 10% change of density moves the 103 kHz mode by 4.79 kHz, as much as the shape change of the published example.
3. Linear kinetic. For a fixed, complete candidate set of $M$ modes, the smooth upper bound $R_\tau=\tau\log\sum_ae^{\gamma_a/\tau}$ with $\max_a\gamma_a\le R_\tau\le\max_a\gamma_a+\tau\log M$, and the critical fast-ion pressure $\beta_{f,\rm crit}=\beta_f\,\Gamma_{\rm damp}/\gamma_{\rm drive}$ of each mode, valid while the mode and equilibrium are frozen. Drive and damping stay separate in the record.
4. Transport. The amplitude at which resonances overlap. The flux of the reduced model of 8.11 is a published comparison in a test, not an objective.
5. Direct. The energy-weighted loss from ESSOS at stated amplitudes, wave on minus wave off, with paired markers, a smooth loss indicator for the gradient and the hard flags for the reported number.

Rules for all levels: a rejected or unconverged evaluation is returned as rejected, never as a small value; windows and candidate sets are fixed before the run; the trial basis is fixed along a differentiable step and re-chosen between steps.

## 9. Repository structure

### 9.1 Layout at release 3.0

```text
alfvex/
  pyproject.toml  README.md  LICENSE  CITATION.cff  plan.md
  src/alfvex/
    __init__.py     public names and the version
    geometry.py     Geometry, the readers (boozmn, wout, VMEX input), circular_torus, the surface grid and metric
    plasma.py       Plasma and FastIons: species and profiles on s, distributions and their derivatives   (new)
    continuum.py    surface pencils (shear, sound, kinetic inertia), gaps, mode families
    eigen.py        global weak forms (8.3, 8.4, 8.8), block assembly, solves through SOLVAX, Modes and Status
    damping.py      resolvent response and continuum damping; radiative, electron, ion and collisional power   (new)
    drive.py        resonance weights and sums for passing and trapped ions; the power ledger
    transport.py    saturation amplitude, resonance overlap, quasilinear flux                                 (new)
    objectives.py   smooth objectives, harmonic sets, mode classification and tracking, summary
    vmex.py         VMEX terms as (state, runtime) callables; geometry from the spline state                  (new)
    io.py           case schema, load, evaluate, save, run, export_mode, the command line
    plot.py         figures
  tests/            one file per module, test_examples.py, test_hygiene.py, conftest.py, data/
  examples/         plain scripts and TOML cases (section 11)
  docs/             index, getting-started, tutorials, theory, validation, comparisons, api, development, references
  .github/workflows tests.yml (pull request and main), scheduled.yml (nightly and weekly), publish.yml
```

Twelve modules. A module is added only when its responsibility is not held by an existing one; the four new modules take over code that today sits in `drive.py` (the distribution), in `objectives.py` (the design path, 250 of its 599 lines) and in prototypes outside the repository.

Budget in lines, counted with `wc -l`: geometry 380, plasma 180, continuum 260, eigen 620, damping 420, drive 520, transport 200, objectives 520, vmex 300, io 330, plot 160, `__init__` 90. Total 3980. A pull request that exceeds the budget of its module says which lines it removes elsewhere.

### 9.2 Public names

| Area | Names |
|---|---|
| Geometry | `Geometry`, `circular_torus`, `read_boozmn`, `from_wout`, `from_vmec_input`, `surface_grid` |
| Plasma | `Plasma`, `FastIons`, `mass_density` |
| Continuum | `continuum`, `continuum_matrices`, `gaps`, `gap_location`, `mode_family`, `alfven_harmonics`, `mode_families` |
| Global modes | `global_matrices`, `eigenmodes`, `frequency_hz`, `Modes`, `Status` |
| Stability | `stability` (the ledger of 8.5 for a set of modes), `growth_rate`, `ion_landau_damping`, `resolvent`, `continuum_damping` |
| Transport | `saturation`, `quasilinear_flux` |
| Objectives | `ModeClass`, `window_risk`, `soft_max`, `drive_risk`, `gap_closure`, `continuum_cost`, `compression_harmonics`, `critical_beta`, `classify_modes`, `summary` |
| VMEX | `alfvex.vmex.geometry`, `alfvex.vmex.frequencies`, `alfvex.vmex.growth_rates`, `alfvex.vmex.term` |
| Cases | `load`, `evaluate`, `save`, `run`, `export_mode`, `enable_cache` |
| Figures | `plot_summary`, `plot_results` |

`prepare_design`, `Design`, `design_state`, `design_params`, `design_geometry`, `design_frequencies`, `state_geometry`, `geometry_frequencies` and `geometry_growth_rates` are replaced by the four functions of `alfvex.vmex` in release 2.0, which is already a breaking release. `eigenvalue_mean` moves to SOLVAX.

### 9.3 One evaluation path

```python
import alfvex

case = alfvex.load("case.toml")  # host: read, validate, resolve paths, reject unknown and inactive keys
out = alfvex.evaluate(case)  # pure: a pytree of JAX arrays with a Status; no file access
alfvex.save(out, case)  # host: results.npz, case.json, and the mode record when asked
out = alfvex.run("case.toml")  # load, evaluate and save in one call; what `alfvex run` does
```

`evaluate(case, **inputs)` is the only function that computes. `inputs` overrides the continuous quantities of the case with arrays: `geometry` (a `Geometry`), `boundary` (VMEX boundary offsets), `density`, `te`, `ti`, `pressure`, `fast` and `amplitude`. It is differentiable in those inputs in forward and reverse mode, and it compiles once per case shape. The case itself is static: mesh sizes, the harmonic set, the model switches and the task. `run` converts to NumPy and writes files only after `evaluate` returns. In 1.2.0 `fax.run` raises a tracer conversion error under differentiation, and its export drops the sign of $\omega^2$ and the solver status; `evaluate` and `save` remove both defects.

The result of every task carries: `s`, `iota`, `harmonics` (signed $m$, $n$), `omega2` (signed, or complex), `frequencies_hz`, `structures`, `status` (residual, orthogonality, selection gap, residual of the nearest excluded pair, iterations, converged), and for a stability task the ledger of section 8.5. A solve that did not converge sets `status.converged` to false, the command exits with a nonzero code, and objectives reject the evaluation.

### 9.4 Case schema

A case has the tables of 1.2.0 and two more. A case without `schema` is read as version 1, which stays valid.

```toml
schema = 2

[geometry]
source = "vmec"              # boozmn | wout | vmec | circular
path = "input.LandremanPaul2021_QA"
mboz = 16
nboz = 16
ns = 49

[modes]
n = [1]                      # one family; without it, the resonant branches of every family
m_max = 30
signs = "both"               # both | one: whether the conjugate harmonics are included
parity = "auto"              # auto | cos | sin | full

[plasma]
ion_mass = 2.5               # amu
density = [1.0e20, -0.5e20]  # ions per cubic metre, polynomial in s
te = [8.0e3, -6.0e3]         # eV
ti = [8.0e3, -6.0e3]         # eV
gamma = 1.6667

[fast]
species = "alpha"            # or mass and charge
distribution = "slowing_down"  # maxwellian | slowing_down
density = [1.0e18, -1.0e18]
energy = 3.52e6              # eV: birth energy, or the temperature of a Maxwellian

[model]
wave = "shear"               # shear | current | sound
drive = ["passing", "trapped"]
damping = ["continuum", "ion", "electron", "collisional", "radiative"]

[run]
task = "stability"           # continuum | eigenmodes | summary | stability | response
output = "outputs/qa"
ntheta = 48
nzeta = 48
nodes = 48
order = 2
target_khz = 103.0
count = 6
tol = 1.0e-10
dense_max = 2000
export = "modes"             # also write the mode record of section 8.10
```

Rules of the loader: unknown keys are rejected; keys that the chosen source or model does not use are rejected (`surfaces` with `source = "boozmn"`); units are fixed per key and stated in the reference page, which is generated from the defaults; densities and temperatures must be positive; `count` must not exceed the basis size. The command line is `alfvex run case.toml [--plot] [--cache DIR | --no-cache]` and `alfvex plot folder`. `vmex --alfven` accepts the same file.

### 9.5 Records passed between codes

| Record | Content |
|---|---|
| `Geometry` | $s$, $N_{fp}$, signed $\psi'$, $\iota$, $G$, $I$, the Boozer harmonics of $B$, $R$, $Z$, the angle offset and the Jacobian with their $(m,n)$, the pressure and the radial covariant field where the model needs them (step 3.1), and a flag that the radial covariant component is dropped otherwise |
| `Plasma` | species masses and charges; density, $T_e$, $T_i$, pressure, $Z_{\rm eff}$ and $E_r$ on $s$ |
| `FastIons` | mass, charge, density and energy or temperature on $s$, the distribution kind and its anisotropy parameters, the variables held fixed in its radial derivative |
| `Modes` | $s$ nodes, signed harmonics, complex $\Phi_{mn}(s)$ in volts, $\omega$ in rad/s, $\omega^2$, the wave energy $W$ in joules for the stored amplitude, the harmonics of $c$, $\delta B_\parallel$, $\Psi$ and $\alpha$ where the model has them, `status` |
| Mode file (`export_mode`) | the `Modes` fields of one mode on a radial spline, `Geometry` orientation, the phase convention, the amplitude rule and its value, schema version, versions and input hashes |
| Particle result (ESSOS) | marker identifier, weight, birth energy, pitch, phase and random key; one of the end states retained, thermalized, lost at the boundary, struck the wall, stopped at the inner cut-off, failed; energy at the event; accumulated work of the wave; deposited energy per species |

## 10. Tests

### 10.1 Tiers

| Tier | Content | Runs | Budget |
|---|---|---|---|
| Pull request | Unmarked tests: analytic limits, contracts of each function, one derivative direction per kernel, the parity of the three entry points, one real fixture per code path, one small VMEX solve with its boundary derivative, the installed-wheel job, lint, the documentation build. Statement and branch coverage at or above 95% each, over every production module. | Every pull request and push | 6 min of tests, 10 min in all |
| Integration | `slow`: the other VMEX solves and boundary derivatives, the 64-harmonic AE3D band, the mode record read by ESSOS. | Pull requests that touch `vmex.py`, `geometry.py` or the mode record; nightly | 30 min |
| Scheduled | `scheduled`: re-solved finite differences over step ladders, resolution budgets, cross-code fixtures, the minimum and the latest dependency stacks, rebuilds of STELLGAP and AE3D from source. | Weekly and on demand | 4 h |
| Release | External comparisons at high resolution, the damping and kinetic benchmarks at full size, the profile, one GPU run. | Before a tag, recorded in the release notes | by hand |

### 10.2 What a test is

- An equation, a limit, a conservation law, or a published number whose assumptions match. The docstring names the source with its equation, table or figure and page, and the tolerance with its reason: discretization error, the reading error of a figure, or the spread of published codes.
- A number read from a figure carries its reading error. Signs, exponents and minus signs of printed equations are taken from the page image; the text layers of older papers drop them.
- Both regimes of a traced branch are asserted with inputs in each regime; tracing a branch does not test it.
- For every sign and normalization that decides a result, the docstring states which wrong value the test would catch, and the test is shown to fail on it once.
- Failure paths are tests: aliasing of the angle grid, a harmonic set that is not a family, an unconverged solve, a tie in selection, a rotation inside a degenerate pair, a negative $\lambda$, a moved trapped-passing boundary, a zero divisor.
- A comparison of two solvers on the same matrices is a numerical test and is labelled so. It does not count as a physical comparison.
- A test that fixes the output of the code itself is labelled a regression and cites the pull request that produced the number.

### 10.3 Tests anchored in the literature

Kept from 1.2.0: the uniform-field continuum $k_\parallel v_A$; the geodesic acoustic frequency and the acoustic gap top; the exact sheared three-dimensional solution; the density scaling $f\propto\rho^{-1/2}$; second-order radial convergence; the ITPA frequency; the coefficient of Fu and Van Dam. Added, with the step that brings each:

| Quantity | Value and tolerance | Source | Step |
|---|---|---|---|
| Rational zero of the continuum at $m\iota=n$; allowed families against a full-torus solve | exact | Spong et al. (2003) | 0.1 |
| Parity sectors and completed basis | 8.3; cosine sector 0.160% against STELLGAP | this plan, measured | 0.1 |
| CTOK continuum against regenerated STELLGAP | maximum 1%, mean 0.5% | Spong et al. (2003), rebuilt | 1.6 |
| CTOK global modes against AE3D | six frequencies within 1.4%, both overlaps stated | Spong et al. (2010) | 1.7 |
| Two-harmonic gap | width $\propto\lvert\varepsilon_1+\varepsilon_2\rvert$; half-width $\lvert\varepsilon_B-\varepsilon_g/2\rvert$ | Paul et al. (2025); Kolesnichenko et al. (2001) | 7.1 |
| Helical gap pair $(3,1)$, $(5,6)$ of the Helias model | $\omega R_0/\bar v_A=1.4406$ and 1.9464 for a homogeneous plasma, to the accuracy of $\iota(r)$ digitized from their fig. 2 (a few per cent); the continuum results that need no profile (fig. 4, eqs. 38, 46, 51) to $10^{-3}$ | Kolesnichenko et al. (2001) | 2.6 |
| Wistell-A continuum | HAE (3,2) narrows; (4,2) and (2,2) stay above $\omega/\omega_A=1.5$ | Paul et al. (2025) | 8.3 |
| ITPA frequency with pressure | within 1% of $4.01\times10^5$ rad/s | Könies et al. (2018) | 3.1 |
| ITPA mode shape | $m=10$ peak at $\sqrt s=0.485$, width 0.090; $m=11$ peak at 0.513, width 0.094, amplitude ratio 0.86; reading error 0.01 | Fehér (2013) | 3.1 |
| Fixed-wall $n=1$ TAE | $(\omega/\omega_A)^2\approx0.50$, gap 0.40 to 0.90 | Cheng and Chance (1986) | 3.4 |
| Sound-coupled continuum, analytic | $\omega^2(k_\parallel=0)=\omega_{G0}^2(1+\mathfrak S)$; W7-X $(9,8)$: $\omega_s=18.5$ kHz, $\omega_{G0}=10.5$ kHz, read from a figure, $\pm1$ kHz, with $R=5.5$ m, $\gamma=5/3$ and $T_e=T_i$ assumed | Kolesnichenko et al. (2011) | 3.5 |
| HSX and H-1 sound-coupled spectra | HSX gap near 47 to 88 kHz; H-1 gaps at 9, 12 and 30 kHz; published geodesic roots 9.4, 13.2 and 31.1 kHz, which follow their eq. 13 as printed, without a factor 2 (with it a hand estimate gives 9.9, 15.2 and 32.3). Gap positions only, no tolerance: density, species and field of the HSX case, and $\gamma$, $R$ and the $\iota$ profile of the H-1 case, are not printed | Könies and Eremin (2010); Bertram et al. (2012) | 3.5 |
| Continuum damping, two-harmonic | $0.370887-0.00174204i$, $10^{-6}$, for a one-dimensional solve of the large-aspect-ratio model (Berk et al. 1992, eq. 35): a referee for the contour method, not for the global operator, which gives $0.3773-0.0057i$ on this case | Bowden et al. (2015) | 4.1 |
| Continuum damping, tokamak | $\mathrm{Im}\,\Omega^2=-8.9\times10^{-4}$, 5%; two methods within 1% | Könies and Kleiber (2012) | 4.1 |
| Kinetic Alfvén wave | $\omega^2=k_\parallel^2v_A^2[1+k_\perp^2(\tfrac34\rho_i^2+\rho_s^2)]$ | Lauber (2013) | 4.6 |
| Radiative damping | $g_M=-1.014-0.0707i$; $-\mathrm{Im}\,h=6.5\times10^{-5}$; $\gamma/\omega=-5.3\times10^{-2}$; tunnelling law | Candy and Rosenbluth (1994); Berk, Mett and Lindberg (1993); Mett and Mahajan (1992); Breizman and Sharapov (2025) | 4.7 |
| Trapped-electron collisional damping | $\gamma_e/\omega=-3.0$ to $-11.1\times10^{-4}$ over $\nu_{\rm eff}/\omega=0.035$ to 1.29, read from a figure: logarithmic slope 0.5 to 0.6 below 0.1, saturation above 0.5, magnitude within a factor 1.5 (boundary shape and $\beta$ are not printed) | Fu, Cheng and Wong (1993) | 4.8 |
| ITPA bulk damping | $-567$ to $-1705$ s$^{-1}$; LIGKA $-1052$ and $-1517$ s$^{-1}$ | Könies et al. (2018) | 4.8 |
| Kinetic geodesic and beta-induced frequency | 2.487, 2.797, 2.853 $\omega_{ti}$; damping from the numerical root; $-0.040\,\omega_{ti}$ of the large-$q$ formula at $\tau=1$, $q=1.5$ as an order check | Lauber (2013); Zonca et al. (1996) | 4.9 |
| Wave energy of the ITPA envelopes | $4.279791098\times10^{-5}$ J, $10^{-12}$ | independent quadrature | 5.1 |
| Passing drive, power table | 8.6, $10^{-10}$ | independent implementation | 5.1 |
| Local alpha threshold | critical $\beta_\alpha=2.67$, 8.12, $12.8\times10^{-4}$ with the corrected coefficient (2.61, 7.76, 11.9 as printed) at $m=2$, $q=1.5$, 2% on CIT and ITER | Cheng (1991) | 5.4 |
| Landau damping of a flat Maxwellian | $\gamma/\omega=-0.00998227$ at $x=1$; $2\times10^{-5}$ at three resonances | Kolesnichenko and Tykhyy (2018) | 5.1 |
| Reflected ions, one well | 0.0232, 1.359, 4.002, 11.07 s$^{-1}$, 1% | independent calculation | 5.3 |
| Helias resonances and relative growth | 5.5 | Kolesnichenko et al. (2002) | 5.5 |
| ITPA drive at zero orbit width | increments 15.8, 20.0, 22.3 $\times10^3$ s$^{-1}$ of the CAS3D-K curve, each $\pm0.2$; reported with the ratio, not a gate while the ratio of 0.60 stands | Fehér (2013) | 5.1, 5.6 |
| ITPA drive with orbit width | the band of the benchmark codes, 8.5, 28.2, 38.1 and 42.8 $\times10^3$ s$^{-1}$ at 100 to 400 keV, $\pm15\%$ | Könies et al. (2018) | 5.6 |
| ITPA frequency shift | 4.012, 3.968, 3.880, 3.732 $\times10^5$ rad/s; the shifts $-4.4$, $-13.2$, $-28.0\times10^3$ rad/s within $0.5\times10^3$ | Könies et al. (2018) | 5.8 |
| W7-AS 39042 | 5.10 | Fehér (2013); Könies et al. (2008) | 5.10 |
| Field transfer to ESSOS | $4\times10^{-15}$; a time-sign error gives 1.28 | this plan, measured | 6.2 |
| Wave invariant | $H+(\omega/n)P_\zeta$ to $10^{-12}$; island width $\Delta\psi=4q[(q/q')(g+nI/m)\alpha_{nm}]^{1/2}$ | White and Chance (1984) | 6.2 |
| Pitch scattering | $\langle P_\ell\rangle=P_\ell(\xi_0)e^{-\ell(\ell+1)\nu_Dt/2}$ | Lorentz operator | 6.4 |
| Collision coefficients | $2.8\times10^{-6}$ with a common Coulomb logarithm | ASCOT5 | 6.4 |
| Saturation levels and the near-threshold equation | 8.11: 3.2, 1.9, $8^{1/4}$, onset at 4.38, $A=70.71$ | Berk and Breizman (1990); Berk et al. (1996); Lilley et al. (2009) | 6.6 |
| ITPA single-mode saturation | $\gamma_L=4.723\times10^4$ s$^{-1}$; 2.955 kV, $\delta B/B=2.1\times10^{-3}$ | Slaby et al. (2024) | 6.6 |
| Wave-off losses | zero exits of 5000 in three of the four configurations; bound 0.060% | Landreman and Paul (2022) | 8.3 |
| Wave-on losses | passing stochasticity near $\sqrt{\delta U/U_0}=10^{-4}$, prompt loss near $10^{-3}$ | Knyazev et al. (2026) | 6.3 |

Not gates, and why: the full tokamak, W7-X and H-1 values of Bowden et al. (8.8); the absolute CAS3D-K curve (8.6); the transport model of Slaby et al. (8.11); the JET cases of LIGKA, whose profiles exist only as figures; the TFTR threshold tables, whose beam and plasma parameters are not all printed.

### 10.4 Hygiene and text

`tests/test_hygiene.py` holds the rules as patterns and runs on every tracked text file, on docstrings and on comments:

- no machine paths, no personal handles, no tool attribution strings;
- no promotional or filler vocabulary; the list is in the test, with the plain word to use instead;
- every sentence that claims speed, accuracy or agreement holds a number or cites a test;
- test names and paths cited in the documentation exist;
- the old name does not occur outside the allow-list of step 0.2;
- examples hold no argument parser, no `main` and no `if __name__` block.

Commit messages and pull-request texts follow the same rules: what changed, why, the test and its result. The check on them runs in CI on the pull-request title and body.

## 11. Examples and documentation

Examples are plain scripts. Each starts with a docstring that says what it computes and which published number it reproduces, then the settings as module-level constants, then the calculation. No argument parsing, no `main()`, no `if __name__` block. Each runs in under two minutes on a laptop CPU unless its docstring says otherwise, and each is executed by `tests/test_examples.py` or by the documentation build.

| File | Shows | Release |
|---|---|---|
| `quickstart.py` | Continuum, global modes and an eigenvalue gradient on the analytic torus | 2.0 |
| `ctok_continuum.toml`, `ctok_eigenmodes.toml`, `qa_continuum.toml` | The three entry points on the same cases | 2.0 |
| `ctok_boozmn.py`, `knyazev_qa.py` | The STELLGAP and AE3D comparisons on the regenerated fixtures | 2.0 |
| `qa_vmex.py` | A VMEX input to a spectrum summary with both signs of each harmonic | 2.0 |
| `optimize_alfven.py` | A VMEX least-squares run with ALFVEX terms and `jac="implicit"` | 2.0 |
| `itpa_tae.py` (now `koenies_tae_drive.py`) | The ITPA frequency and the signed power ledger per orbit class | 2.0, extended in 2.2 |
| `benchmark.py` | Timings at equal accuracy; the job goes to its worker on standard input, not on `sys.argv` | 2.0 |
| `sound_modes.py` | The geodesic acoustic frequency, the acoustic gap and one sound-coupled global mode | 2.1 |
| `continuum_damping.py` | The Könies and Kleiber tokamak case by resolvent fit and by complex eigenpair | 2.1 |
| `essos_orbits.py` | A computed mode passed to ESSOS; wave on and off with paired markers | 2.2 |
| `stability_margin.py` | Critical fast-ion pressure of a mode set and its boundary gradient | 2.3 |

The step that brings each new example: `sound_modes.py` step 3.4, `continuum_damping.py` step 4.1, `essos_orbits.py` step 6.2, `stability_margin.py` step 7.3; `itpa_tae.py` is the rename of step 2.14.

Documentation rules:

- Every model change updates, in the same pull request, its equations and ordering in `theory.md`, its keys in `getting-started.md`, its entry in `api.md`, its row in `validation.md` and its limits in `physics.md`.
- `theory.md` ends with one table that maps each equation of section 8 to its function, its test and its source.
- `validation.md` lists each gate with the test that asserts it, the measured value, the tolerance, the versions and the date. `tests/test_hygiene.py` checks that every test named in the documentation exists. The performance table is written by `examples/benchmark.py` with its machine, versions and date; the table of 1.2.0 was produced with FAX 0.1.0 and SOLVAX 0.32.1 and is regenerated in step 1.9.
- The first page a new user reads is short: install, one TOML case, the equivalent Python call, the outputs, the next accuracy check.
- The build treats warnings as errors and runs from the installed package.

## 12. Performance

### 12.1 Method

Each kernel and size runs in a fresh process. The first call is reported as cold (tracing, lowering, compilation, one run) and the minimum of at least three later calls as warm, synchronized with `block_until_ready`. Stages are compiled separately on the outputs of the previous stage and their sum is checked against the public function. Tools: `jax.profiler` traces read in Perfetto, with named scopes for geometry, Fourier projection, assembly, factorization, inverse applications, Ritz extraction, response and pull-back; the compile events of `jax.monitoring`; `compiled.memory_analysis()` for sizes that cannot be run; optimized HLO for kernel attribution. TensorFlow is not needed on CPU; XProf is used when a GPU kernel has to be attributed. Timings are quoted with the machine, its load, the thread settings and the residual reached. `VECLIB_MAXIMUM_THREADS` limits Accelerate on macOS; on Linux `taskset` limits the thread pools of the XLA runtime, which sizes them from the affinity mask, and `OMP_NUM_THREADS` does not; on macOS no setting was found, so single-thread timings are claimed on Linux only. On Linux the LAPACK calls of jaxlib go through OpenBLAS, whose threads are a separate setting: with `OPENBLAS_NUM_THREADS` unset, each small per-surface factorization starts a thread team, and the continuum at 52 harmonics takes 5.1 to 7.5 s against 0.74 s with one BLAS thread; at 162 harmonics four calls did not finish in 570 s against 2.25 s. Every CPU timing on Linux states the BLAS threads, and the documentation tells users to set one.

### 12.2 Present bottlenecks

Measured at 1.2.0 on an Apple M2 with 8 GB that also ran other jobs; times are upper bounds. Pull request 66, merged since, compiles the surface grid, the assemblies and the drive kernel; rows marked (66) are measured again in step 2.9.

| Bottleneck | Measured | Remedy | Step |
|---|---|---|---|
| The all-mode dense solve is the default of `eigenmodes`, `frequency_hz` and the design path | 18 s at 2400 unknowns against 0.075 s targeted; $(NH)^{3.1}$; 8 GB at about 7900 unknowns | The targeted solve is the default wherever a window is known; `summary` scans windows with targeted solves; the dense path raises above `dense_max`. | 2.9 |
| Sweeps of the targeted solve | 84 to 95% of the call; about $10^4$ block solves of 4 $\mu$s each inside `while` loops; the block factorization is 1.2% of the solve at $N=64$; sweeps grow from 2 to 9 by $N=1600$ | SOLVAX: apply the factors as stored block products; test fewer applications per sweep; start from the vectors of the previous optimization step. | 2.10 |
| The $3\times3$ metric inverse in `surface_grid` | 63 to 73% of `surface_grid`, which is 50 to 97% of the continuum for $H\le52$ | A closed-form inverse from cross products of the basis vectors. | 2.9 |
| Complex arithmetic where the pencil is real | A real LU is 5 to 8 times faster than the complex one; a real `eigh` 1.9 to 3.9 times | The real-arithmetic and parity paths of pull requests 67 and 68. | 0.1 |
| The resonance loop of the drive | 77 to 97% of `growth_rate`, on one core; 5.0 s at the keyword defaults on 200 surfaces. Four temperatures: 243 ms as separate calls, 51 ms batched, 2.3 ms on prepared weights | Resonance weights computed once per mode and geometry and reused for every distribution, temperature and species (8.6); `vmap` over resonances in chunks. | 2.9, 5.1 |
| Public functions outside `jit` (66) | At 1.2.0 `eigenmodes` 1.9 times, `growth_rate` 5 times and the Boozer stage of `optimize_alfven.py` 70 times slower than compiled | `evaluate` compiles once per case shape; the example calls it. | 2.1 |
| Equilibrium and adjoint in the optimization loop | 98% of an evaluation (1.27 of 1.29 s) and of a gradient (3.15 s); one adjoint solve per residual row, 6.6 of 13.4 s for four rows, because the design path of 1.2.0 calls `jax.vjp` row by row | The VMEX optimizer already builds a forward-mode block Jacobian with one factorization for all columns and reuses a nearby factorization: the terms of step 2.3 go through it. In VMEX: an early hand-off from the fixed-point iteration to Newton, and the refinement's factorization passed to the scalar adjoint. | 2.3, 8.1 |
| Compilation | 2 s per shape for the targeted solve; 34 s for the gradient through VMEX; 10.8 s for the reverse-mode drive gradient through the eigenvector | The persistent cache in the runner and the examples; a run-time iteration limit in SOLVAX; one shape per case; a test that counts compilations. | 2.9 |

### 12.3 Bottlenecks expected with the work packages

| Bottleneck | Measured or projected | Remedy | Step |
|---|---|---|---|
| Memory of the assembly at large $H$ | 32 to 34 arrays of $N\times H\times H$ forward, 102 to 149 in reverse mode; planned 9.3 and 30.7 GB at $N=200$, $H=300$. The full signed basis doubles $H$ where parity does not apply | Parity sectors for symmetric fields; assembly in radial chunks; the eigenvalue derivative from $y^\dagger(dF-\lambda\,dG)y$ evaluated on the angle grid without storing $dF$. | 0.1, 2.10 |
| A second unknown per harmonic (sound, vorticity) | Blocks of $2H$: 2.4 to 2.8 times the time, 3.9 times the temporaries | Budgeted in each gate. | 3.3, 4.6 |
| Radial elements of order $p$ | 1.2, 1.6 and 2.4 times at equal unknowns for $p=2,4,8$ | $p$ chosen by cost at equal accuracy. | 2.5 |
| The non-Hermitian eigen-solve | `eig` of the projected problem is 35 times `eigh` at order 84; a dense `eig` is 3 to 5 times `eigh` on one core | SOLVAX: a small projected problem per mode, or a contour solve with batched shifts. No all-mode dense `eig`. | 4.3 |
| The continuum at $H$ of several hundred | $H^{2.45}$ per surface; 9.6 GB planned for 200 surfaces at $H=1000$ | Surfaces in chunks; eigenvalues in a window by shift-invert; families and parities solved separately. | 2.9 |
| Orbit integrals of the kinetic response | 16 ns per orbit point and resonance; $2.6\times10^5$ markers and 25 s for 1% in the orbit-integrated check | Shared orbit data between modes; quadrature nodes on the resonances; the orbit-integrated evaluator at the validation tier only. | 5.6 |
| Field and mode evaluation in ESSOS | A dense radial spline and direct harmonic sums in pull request 118. In the wave runs of experiment x5 the cost was the linear interpolation of the mode, which makes the adaptive stepper reject steps | A twice-differentiable spline of the mode in the constructor of 6.1; a banded spline in SOLVAX and angular tables, at equal field and derivative accuracy. | 6.1 |
| ALFVEX inside the optimization loop | 0.04 s of 13.4 s today; seconds at $N=800$, at $H=96$, or with the drive at its defaults | The rows above, and a resolution ladder: coarse settings first, fine settings for the last steps. | 8.2 |
| Timing tests on shared machines | Times vary by 2.6 times with load | The performance test bounds ratios between kernels in one process, planned memory and the number of compilations, not seconds. | 9.4 |
| GPU | The primitives are measured on one RTX A4000 (measurements of 2026-11-19, kept with the programme's assessment outside this repository): a solve inside a `scan` step costs 25 to 56 microseconds against 5 for a product with a stored inverse, and block cyclic reduction is 6 to 68 times faster than the scanned block Thomas solve. Main on the same card, against one CPU thread: the continuum is 6 to 12 times faster, the dense solve 15 times, the drive 35 times; the targeted solve is 1.5 times slower on the ITPA case at (200, 12) (0.37 s against 0.25 s; 2.77 s at (800, 12)) and 1.8 times faster on the precise quasi-axisymmetric case. No kernel calls back to the host; the cost of the targeted solve is in its sequential loops. The largest sizes for 16 GB are planned figures, not runs: about 18,000 unknowns for the dense solve and $NH^2\approx4\times10^7$ for the targeted one | Stored block products and cyclic reduction in SOLVAX (S7 of its list; G2 and G3 of the GPU list), then the measurement of 9.6. | 9.6 |

Gates of the performance steps, at results equal to $10^{-12}$: the continuum at $H=13$ twice as fast; the drive at its defaults three times as fast; four temperatures on prepared weights under 5 ms; `evaluate` within 10% of the compiled kernels; planned memory of the gradient at $N=200$, $H=96$ no more than 1.5 times that of the forward solve (3.7 times at 1.2.0).

## 13. Steps

Each step is one pull request unless it says otherwise. A step lists the change, the files, the tests to add and the gate that closes it. A step is ticked in this file by the pull request that closes it, with the measured gate value. Directions that hold for every step:

- Branch from main, one claim per branch. Write the test first; it fails for the reason the step names.
- Numbers in tests come from section 8, section 10 or the cited paper, never from a first run of the new code, unless the test is labelled a regression.
- State in the pull request: what changed, the physical reason, the test case, the measured result with units, the net line count and the versions.
- Update the documentation pages of section 11 in the same pull request.
- Run before pushing: `ruff check . && ruff format --check .`, the pull-request tier, `tests/test_hygiene.py`, and the documentation build.
- Work that belongs to SOLVAX, VMEX, ESSOS or DKX is a pull request there, with its own tests, followed by a small pull request here that uses the release.
- Commits and pull requests carry no tool attribution and no co-author trailers; the author is the repository owner.
- A step is finished when its pull request is ready for review. The owner reads it and merges it; nobody merges for the owner, enables auto-merge, forces a push or goes around a required check.
- No step rewrites a module, a test file or a page wholesale. A change is the smallest diff that passes its gate; a file that continues an old one is moved with `git mv` and then edited.
- Steps named with a letter (E10, V2b, S7, G2, R8b, D3) belong to the work lists of ESSOS, VMEX, SOLVAX, the GPU work, the repository rules and the documentation. Those lists are kept with the programme, outside this repository, and each such step is a pull request in its own repository.

### Package 0. The name and the baseline

- [ ] **0.1 Land the basis stack.** Pull requests 67, 68, 69 and 72 are drafts, each based on the previous, and CI never ran on them because the workflow skips drafts. Retarget 67 to main and take them in that order: real arithmetic for stellarator-symmetric problems, parity sectors, angle averages over one field period, both signs of each harmonic by default. Source lines grow from 2073 to 2312. Files: `continuum.py`, `eigen.py`, `geometry.py`, `objectives.py`, `io.py`. Six pull requests in this order, each merged by the owner once CI passes: 67 as it is; 68 with the refusal of a traced geometry stated in the docstring of `parity` and a test of it; 69 with the check that every toroidal number is a multiple of the field period, on the host and under trace; a new one that makes the symmetry flags and the period condition static fields of `Geometry`, set by every constructor; then 72 in three (the block structure and the default of `continuum` and `eigenmodes`; the default angle grid; the design path, `summary`, `mode_families` and the runner). Each test listed below names the pull request that adds it. What 72 must change first:
  - On a traced geometry the stack decides symmetry from the field-period count alone, so a three-dimensional field with $N_{fp}=1$ gives 12 continuum branches per surface eagerly and 9 under `jit`, where main gives 6 both ways. The symmetry facts (axisymmetric, stellarator-symmetric, every toroidal number a multiple of $N_{fp}$) become static fields of `Geometry`, set by its constructors. The same fields let the design path, where every array is traced, use a parity sector instead of the complex pencil on both signs.
    Decided 2026-11-21 (pull requests 75, 78 and 80): one tolerance, $10^{-9}$ of the largest symmetric harmonic, decides a symmetry fact on the host and checks a stated fact; under trace, where the harmonics cannot be inspected, a guard returns NaN when the traced harmonics contradict a stated fact. Reasons, measured: the VMEX design path and `booz_xform_jax` return exactly 0.0 for the asymmetric tables of a symmetric state, but inputs that pass through a file are not exact, so exact zero is too strict; a wrong `stellsym=True` gives an error of about $1.3\times10^3\varepsilon^2$ in the asymmetry amplitude $\varepsilon$, under $10^{-10}$ at $\varepsilon\approx3\times10^{-7}$ and 4% at $\varepsilon=10^{-2}$, so no check allows a silent 4% error; a fact wrongly left unset costs 4.3 times on the continuum and 1.65 times on the global solve; the guard costs nothing measurable and works with `grad` and `vmap`. Not yet implemented: the heads of pull requests 78 and 80 decide `stellsym` by exact zero on the host and have no traced guard; both change to this rule before merge.
  - A `count` above the basis size raises, as on main; the stack clips it silently.
  - The numbers that the stack moves are regenerated in the same pull requests: the quasi-axisymmetric example goes from 294.11, 294.44 and 294.92 kHz to 245.55, 245.76 and 246.06 kHz, and the lowest $n=0$ eigenvalue no longer converges at first order on uniform nodes (step 1.1).
  - The drive of a cosine- or sine-sector mode: `growth_rate` must sum both signs of each harmonic of the standing wave. This is read from the code, not measured: first compare the rate of a cosine-sector mode with that of the same mode written on the full basis, and change `drive.py` only if they differ. Test: the rate of a sector mode equals the sum over its two travelling components with their interference.
  - The release notes list what changes for users: twice the branches for a self-conjugate family, finer default angle grids, `mode_family` returning NumPy, the grouping of `mode_families`.

  Tests to add beyond those in the stack: on a symmetric stellarator field the cosine and sine sectors together give the full-basis spectrum to $10^{-10}$; on a field with an added sine harmonic of $|B|$ the complex path differs from a sector solve; for a family with $2n_0\not\equiv0\pmod{N_{fp}}$ at $N_{fp}=4$ or 5 the completion leaves the spectrum unchanged to $10^{-12}$; dense and targeted values and their reverse-mode derivatives agree on the real path, in each sector and on the complex two-sign path; the one-period average is refused, on the host and traced, when a toroidal number is not a multiple of $N_{fp}$; eager and compiled results agree on the $N_{fp}=1$ three-dimensional field; the regression numbers of 8.3; the precise quasi-axisymmetric continuum against STELLGAP at 0.160% in the cosine sector (0.300% with one sign). The suite of the stack head passes on Linux with Python 3.12 (128 tests in 23.5 minutes, the VMEX file alone 11.5). Gate: CI green on each pull request on every supported stack; timings of the real and complex paths at equal error in the pull-request text; coverage holds.
  - The release notes name what stops working: a call with an explicit angle grid that is valid on main can be refused after the stack (`continuum(geo, mode_family(0, 6, 1), rho, 16, 4)` raises an aliasing error, since both signs double the span of the harmonics). The gate reports the cold compilation time of the real and of the complex path: the real path compiles the eigen-solver a second time, for 64-bit reals beside complex numbers. Dependabot pull request 65 is ready; the owner merges it before step 0.2.
  - The stack brings the test of 10.3 on the rational zero of the continuum at $m\iota=n$ and on the allowed families against a full-torus solve (Spong et al. 2003).
  - The Python 3.10 job on macOS arm64 fails on main since pull request 66 (step 0.5). The change of the test matrix in step 0.5 goes first, as a pull request of its own, so that the stack can show a green run.
- [ ] **0.2 Rename.** `git mv src/fax src/alfvex`; the token `fax` becomes `alfvex` in Python, TOML cases, workflows and documentation, and FAX becomes ALFVEX in prose; `pyproject.toml` (the script entry, the URLs, the lint path), `CITATION.cff`, the README badges, `docs/conf.py`, the cache folder, the wheel-content check of `tests.yml`, test and example names; version `2.0.0.dev0`. No other change to code. Tests: the suite passes with no numerical change; `tests/test_hygiene.py` fails on the old name in tracked text outside an allow-list (this file, `tests/data/PROVENANCE.md`, the README and `docs/index.md`, where the old name is said once; the installation page and the changelog join the list when steps D2 and D7 of the documentation list add them); the installed-wheel job runs `alfvex --help` and checks that the wheel holds `alfvex/*.py` only. Gate: `git grep -iw fax` returns allow-listed lines only; the net line count of `src/` and `tests/` is zero apart from the hygiene test. The rename touches 42 files: the pull request states that exception to the file limit in its text, and raises `pr_max_files` for itself if the rule checker is already in the repository. This file is replaced in a pull request of its own.
- [ ] **0.3 Owner actions.** Rename the repository to `alfvex` in its settings. Point the PyPI trusted publisher of `alfvex` at the new repository name. Create the Read the Docs project `alfvex` and redirect the old one. Update the organization profile and the links in other repositories. Update the remote of each clone and runner checkout. These are done by the owner, on the day 0.2 merges.
- [ ] **0.4 Dependents.** vmex: `vmex/core/alfven_run.py`, `tests/test_alfven_cli.py`, the help text of `cli.py`, the nightly import check, the documentation. The requirement stays `alfvex>=1.2.0` and the module is imported as `alfvex` with a fallback to `fax`, because the wheel of 1.2.0 imports as `fax` and vmex is released (step 1.3) before ALFVEX 2.0; the floor becomes `>=2.0` and the fallback goes after that release. ESSOS: nothing, since pull request 235 is superseded by E13 of its list. SOLVAX: the wording changes with the next edit of its changelog. The vmex part is step V3 of the VMEX list, not a pull request of its own. Gate: `vmex --alfven` passes its test against `alfvex` 2.0.0.dev0 and against the wheel of 1.2.0. The nightly of VMEX runs the tests of ALFVEX that call VMEX against VMEX main, and VMEX exports the names they use (V15 of its list).
- [ ] **0.5 Supported stacks.** `requires-python = ">=3.11"`, the floor of DKX and NTX (vmex, ESSOS and SOLVAX declare 3.10); remove the `tomli` fallback. Test Python 3.11 to 3.14. Set the JAX floor to the oldest release that passes the scheduled tier and the installed-wheel job on Python 3.11. Record in the validation page the stack that fails today: Python 3.10.20, JAX 0.6.2, SOLVAX 0.36.2 on macOS arm64 exits with signal 11 while compiling the density gradient through the assembly compiled in pull request 66. If a supported stack still crashes, the fallback is a narrower `jit` boundary around `_global_matrices` (the collaborators tested the removal of the outer one); after any change of a `jit` boundary the wheel job compares `jax.grad` of the analytic case with its closed form to $10^{-12}$. Gate: the installed-wheel job passes on the floor stack and on the latest stack. Candidates for the floor: JAX 0.4.38 (in CI today), 0.6.2 and the latest. The installed-wheel job and the pull-request tier run on each; the scheduled tier runs on the chosen floor only.

### Package 1. Repairs

- [ ] **1.1 Radial nodes.** Put the finite-element nodes (the nodes, not the element centres) at the cell midpoints $(k+\tfrac12)/n$ in `_nodes`; the VMEX runner changes in step V3 of the VMEX list; use `ns` of at least 49 wherever a frequency is quoted. Evidence: through VMEX the 102 kHz gap mode of the precise quasi-axisymmetric case is $1.7\times10^{-3}$ from its 192-node value at 24 nodes on the layout $k/(N+1)$ and $2.3\times10^{-4}$ on midpoints; `ns = 16` adds a bias of $1.6\times10^{-3}$. Gate: within $3\times10^{-4}$ at 24 nodes; second-order convergence on the analytic torus kept (difference ratios 3.15, 4.05, 4.01). `classify_modes` refines to $2n+1$ nodes today, which do not contain the midpoints: it refines to $3n$ cells, which do. The regression numbers of step 0.1 move with the nodes and are updated in the same pull request. The gate is taken on the full basis, at the mode near 103.1 kHz.
- [ ] **1.2 First-surface harmonics (VMEX).** A test that compares `boozer_spectrum_state(geometry=True)` with `Booz_xform` on every surface, then the fix. Evidence: `rmnc(1,0)` is 0.01316 m against 0.02825 m at $s=1/30$, a factor 0.47; the gap mode moves by 0.22 kHz. Gate: every harmonic agrees to $10^{-12}$ relative on every surface (measured $3.4\times10^{-15}$ with the patch), and the design path and the file path give the same gap mode to $10^{-5}$ when both start from the same solved state (the file path reads the `wout` written from the state of the design path). An implicit solve of VMEX holds the $m=1$ constrained family at the initial guess and a plain solve releases it, so the two are different roots; the earlier difference of $6\times10^{-5}$ T in `bmnc` between the paths is of that size. That difference is measured and stated in `validation.md`, and is not part of the gate (step V2b of the VMEX list). A follow-up pull request here raises the vmex floor to the release of V4; the test of the design path against the file path is in `tests/test_vmex.py`.
- [ ] **1.3 `vmex --alfven`.** Release it in a vmex wheel, with the import fallback of step 0.4; request Boozer data on the half-mesh radii instead of merging surfaces (42 of 48 are used today); use both signs. Gate, checked against ALFVEX main after step 1.4, which repairs the gap search: the default run on the precise quasi-axisymmetric case reports the 96 to 142 kHz gap and the mode near 103.1 kHz (the full basis; 102 kHz on one sign) within 0.5 kHz at ns = 49. The notes of the vmex release say that with the 1.2.0 wheel of ALFVEX the gap search still misses that gap.
- [ ] **1.4 Gap search of `summary`.** For the $n=1$ family it lists gaps at 554 and 1219 kHz and misses 96 to 142 kHz. Gate: a test with the three gaps and their edges within 1 kHz. The reference edges come from the STELLGAP continuum of the archived quasi-axisymmetric fixture (cosine sector), not from the code's own output, and the pull request names the cause of the miss.
- [ ] **1.5 Drive kernel.** (a) The radial derivative of a Maxwellian with a temperature profile, $\partial_sF_0=f_M\{n'+n[E/T-\tfrac32]T'/T\}$ at fixed energy; a temperature given per surface is accepted on any pitch grid. Test: $T=139[1+0.8(s-0.25)]$ keV, $n=10^{17}(1-0.65s)$ m$^{-3}$ gives $-3549.33$ s$^{-1}$ on $32\times32$ and $-3598.46$ s$^{-1}$ on $40\times16$ surfaces and pitches; 1.2.0 returns $+2297.92$ or $+1318.16$, or a shape error. (b) With `guard = 0` the constant Fourier component of the orbit angle is 0/0 and the rate is NaN; set that component to zero. Test: 8220.74853 s$^{-1}$ at 24 surfaces and 8 pitches, equal to the result at `guard = 1e-6` within $10^{-13}$; a zero divisor of a non-constant component on a rational surface raises on the host (`growth_rate` returns a scalar and has no place for a status). (c) Species mass and charge are explicit arguments with nuclear masses (`FastIons` defaults to the atomic mass of deuterium, `ion_landau_damping` to the nuclear proton mass), and a zero charge is refused; it divides by zero today. The arguments keep defaults, with the nuclear deuteron mass 2.013553 u, so that existing calls run; test values that move with the mass are relabelled regressions. (d) The sign example of the docstring and of `theory.md` is reversed: the radial term drives for $n\,\iota\,\psi'<0$, and $\psi'<0$ in `circular_torus`. The comment of the $v_A/3$ test should read $\exp(-x^2/9)$ with $x=v_A/v_{\rm th}$. (e) The sampled $B_{\max}$ is refined by a quadratic fit, with a test of both one-sided derivatives at a tie (corrected 2026-11-21: then by a Newton solve on the Fourier series of $|B|$, see 8.9). Both (a) and (b) fit in no added lines: a term `(kappa == 0)` in the denominator of the Fourier inverse, and a `jax.jvp` of the distribution along the temperature column for $\partial_TF_0\,T'$. The oracle of test (a) is a finite difference of the distribution at fixed resonant energy, written in the test. Gate: the drive tests of main and the new ones pass, those whose value moves with the mass relabelled as regressions; statement and branch coverage of `drive.py` stay at 100%.
- [ ] **1.6 CTOK continuum fixture.** The archived STELLGAP reference was built without the outer Boozer half surface, and its metric there is extrapolated: FAX differs from it by 16.45% at $s=0.975$ and 5.20% at $s=0.965$. Regenerate the reference with the surface restored from the shipped WOUT, keep the old archive beside it, record both in `PROVENANCE.md`. Measured against the regenerated reference: maximum 0.70%, mean 0.42%, 95th percentile 0.67% over 97 surfaces. The test of 1.2.0 bounds the mean and the 95th percentile only, which hides the edge. Regeneration needs gfortran, a C++ compiler and NetCDF, and is scripted in the scheduled tier; the arrays are committed. `read_boozmn` reports a file whose last half surface is absent. Both mass conventions are stated: STELLGAP uses $1.67\times10^{-27}$ kg per proton, the test $2\times1.00727647$ u. Gate: a test with maximum 1% and mean 0.5%, and a test that the old archive fails at the edge for the stated reason.
- [ ] **1.7 AE3D fixtures.** The reader of the archive keeps the sign of the eigenvalue (the file stores $\sqrt\lambda$ for $\lambda\ge0$ and $\lambda$ for the five negative roots of 1287) and symmetrizes $A$ correctly, stating the difference from AE3D v31, whose symmetrization uses the same index twice. The comparison table lists, for each mode, the frequency difference, the nodal overlap, the inertia-norm overlap and the Rayleigh quotient of the AE3D vector on ALFVEX matrices. Six modes agree in frequency to 0.185 to 1.346% with nodal overlap above 0.90, but two of them (738.23 and 741.21 kHz) have inertia-norm overlaps of 0.371 and 0.077 and are listed as frequency matches only. The 337.92 kHz mode stays unmatched: nodal 0.907 at 272.77 kHz, inertia-norm 0.442 at 350.36 kHz, Rayleigh quotient 329.33 kHz. Prepared for the owner to send upstream to AE3D, with the patch and the fixture: the three $\iota'$ terms, and the symmetrization of the inertia matrix in `ae_solve_version31.f`, which uses the same index twice. Gate: frequencies and both overlaps are asserted for every mode; the unmatched mode is asserted to be unmatched, so that a later change of the current block shows.
- [ ] **1.8 Selection certificate (SOLVAX).** `generalized_eigenpairs` returns the residual of the nearest excluded pair and sets `converged` only if it passes; the docstring states six applications per sweep, as the code does. Test: the diagonal pencil of 8.9. ALFVEX releases its eigenvalue tangent on that flag (`eigen.py`, `selected`), so it then requires the SOLVAX release, stores the new field in `Status` and passes `dense_limit` through, which it does not today. These are two pull requests in SOLVAX: S2 (the certificate) and S3 (the three test modules that no workflow runs, and separate statement and branch gates). `dense_limit` is SOLVAX's fallback limit and is passed through after S4; the limit of ALFVEX's own all-mode dense solve is a different setting, `dense_max` in the case schema. The same wording, three applications per sweep where the code does six, is corrected here in `docs/theory.md` (line 501).
- [ ] **1.9 Documented numbers.** Regenerate every number of the README and `validation.md` at the pinned versions: the quasi-axisymmetric gap modes (documented 101.896 and 104.441 kHz, measured 102.355 and 104.576 kHz on one sign with a 12 by 12 angle grid, 104.579 with 16 by 16), the passing-ion table of the ITPA case (the page shows 2.47, 8.72, 11.63 and 15.12, from before a change of mass unit; the code gives 2.491, 8.788, 11.718 and 15.231 $\times10^3$ s$^{-1}$), the reading of the CAS3D-K curve (4.26, 14.60, 19.98 and 25.17 in the test; 4.10, 14.65, 19.70 and 24.95 from the vector paths of the figure), the performance table, the coverage figures. Correct: the README note on the $zs_{11}$ gauge floor (about 1% there, $1.2\times10^{-6}$ in the validation page); the growth-rate formula that cites the wrong equation of Chen and Zonca (2016); the first-order gap width, which is eq. 4.9 of Paul et al., not 4.16; the test cited in `docs/api.md` line 51, which does not exist; the name of the CPU test file that says GPU; the description of the pull-request matrix in `development.md`; the installed-wheel job, which stops at its first failing command; the sentence on CAS3D-K, replaced by the table of 8.6. Two more: the comment at `tests/test_drive.py` lines 344 to 345 speaks of a missing trapped contribution of about 40%, where the measured reflected share is 0.001 to 0.07%; and the trial field of the drive test is 4.95% off shell (magnetic over kinetic energy 1.0495), which the validation page states with its effect on the rate. Gate: the hygiene check that cited tests exist passes; each number has its test or example. The check that tests cited in the documentation exist is added to `tests/test_hygiene.py` in this step (the file has one test today). The numbers are regenerated on the defaults that steps 0.1 and 1.1 set, and once more as the last pull request of package 2, since steps 2.4, 2.5, 2.6 and 2.9 move them again. The change of the installed-wheel job is a pull request of its own.
- [ ] **1.10 The design example on the full basis.** Rerun `optimize_alfven.py` and the drive example with both signs and with the window of release 1.2.0 (the window of 8.12 item 2 arrives with step 7.2), after steps 2.3 and 2.4, which replace the design functions and the geometry that the example uses. The results of 1.2.0 (a predicted 4 kHz shift met to 0.05 kHz; the largest of four passing-ion rates lowered from 8500 to 8123 s$^{-1}$, two of the four modes being continuum-like) were obtained on one sign. Gate: the validation page reports the rerun with its mode count, its matching and its resolution estimate.

- [ ] **1.11 Inputs that fail silently.** Found by probing the public interface of main. (a) Float32 inputs are computed in float32 inside `surface_grid` and then promoted, with a relative error of $5.5\times10^{-5}$ and no warning: 32-bit input is refused with a message. (b) `continuum` accepts surfaces at $s=0$ and $s=1$ and returns NaN there; `global_matrices` refuses them: `continuum` refuses $s\le0$ and $s\ge1$ too. (c) A NaN in a geometry harmonic other than $\iota$ gives NaN eigenvalues without an error, or the misleading "|B| must be positive". (d) `frequency_hz` does not check `start` and `stop`: a reversed range gives NaN and a range beyond the solved modes gives the mean of fewer values than asked. (e) A negative fast-ion density flips the sign of the rate without an error; a zero mode or zero frequency gives 0/0. (f) Unclear errors in the drive for an empty harmonic list, a mode array of the wrong shape and `nodes=0`. Each gets a host check with a message that names the cause; where the input is traced the check runs at construction, since the drive returns a scalar. Tests: the reproductions of the probe script of the 2026-11-19 assessment (`bughunt.py`, kept outside this repository), one per defect, moved into `tests/`. Gate: every probe of that script either returns a correct number or raises an error that names the argument.
- [ ] **1.12 Small defects found while the documentation was redrafted.** Each is fixed in code with a test, or the page states the behaviour: the docstring of the drive gives the sign rule reversed (the code and its test have the drive for $n\iota\psi'<0$); the module docstring of `io.py` gives defaults that `DEFAULTS` does not hold; `global_matrices` lacks the aliasing check that `continuum` has, and that check covers the span of the harmonic list only, not the spectrum of the geometry; the dense `frequency_hz` has no convergence check; `gaps` and `gap_center_hz` return the widest spacing between sorted branches, which the quickstart prints as a TAE-gap width (159 kHz against a gap of 6.7 kHz); `classify_modes` with its default margin does not call the ITPA TAE isolated; the default ion mass differs between `mass_density` (1.0 amu), `ion_landau_damping` (1.00727647) and `FastIons` (2.01410178); `continuum_cost` defaults to a helicity that is neither the quasi-axisymmetric nor the quasi-helical one; `ctok_continuum.toml` uses 2.0 amu where the STELLGAP test uses two proton masses (0.36% in frequency); the target of `ctok_eigenmodes.toml` sits where all six modes are continuum-like, while three isolated modes lie near 273 kHz; the command caches by default although the page presents the cache as an option; `from_wout` needs `booz_xform_jax`, which no extra declares; `circular_torus` and the readers do not raise the errors the API page promised. Gate: every item of the list kept with the programme is closed in a pull request that names it, and `tests/test_contract.py`, the probes of `jit`, `grad`, `jvp`, `vjp` and `vmap` behind the derivative-contract table, is in the suite.

Gate of package 1: every number in the README and the validation page is produced by a test or an executed example at the pinned versions.

### Package 2. Numerics and interfaces

- [ ] **2.1 One evaluation path.** `evaluate`, `save` and `run` of 9.3; schema 2 of 9.4; signed $\omega^2$, the ideal growth rate and `Status` in the results and the files (`Eigenmodes` is renamed `Modes`, `Status` is exported, `ModeClass` stays public); a nonzero exit code for a failed solve (the command always returns 0 today); the keys of the circular torus, which the runner ignores today; `surfaces` sets the radial grid of the analytic sources and is rejected for `boozmn`, whose file defines the surfaces. The key `run.parity` of the draft stack ("auto", "single", 1, -1) maps to `modes.signs` ("single" becomes "one") and `modes.parity` (1 becomes "cos", -1 becomes "sin"). Files: `io.py`, `eigen.py`. Tests: the arrays of a continuum case and of a mode case are bitwise equal through the command, a file path and a mapping; `jax.grad` of the mean frequency through `evaluate` equals $-\bar f/2$ under a uniform density scaling to $10^{-12}$; unknown and inactive keys are rejected; a version 1 case gives the same arrays as before. Gate: the three entry points agree in arrays and status. Two pull requests. (a) `evaluate`, `save` and `run`, signed $\omega^2$, `Status`, the exit code and `Modes`, with the inputs `geometry` and `density` only; the other inputs of 9.3 arrive with steps 2.2, 2.3 and 2.11. (b) The loader of schema 2 with the tasks continuum, eigenmodes and summary and the shear model only; keys of later steps are rejected as inactive until their step. A vmex pull request passes `source = "wout"`, which the stricter loader needs.
- [ ] **2.2 Plasma and fast ions.** `plasma.py` with `Plasma` and `FastIons`: profiles are functions of $s$ (polynomial coefficients or arrays on stated surfaces), resampled on any mesh; `mass_density` from the species; the Doppler shift from $E_r$ is step 3.6. This removes the `float()` cast of the density in the design path and the shape error of `classify_modes` with a density array. Tests: a constant profile gives the scalar result; $f\propto\rho^{-1/2}$; a profile given on the source surfaces works on a coarse and a fine mesh. The cast sits in `prepare_design`, which step 2.3 removes, so this step follows 2.3. Resampling is by a cubic spline in $s$, differentiable in the data.
- [ ] **2.3 VMEX terms.** `vmex.py` with `geometry`, `frequencies`, `growth_rates` and `term`, each a `(state, runtime)` callable usable in `vmex.core.optimize` with `jac="implicit"`. The design functions of 1.2.0 are removed. Tests: a least-squares run with one ALFVEX term; re-solved directional finite differences over a decade of step sizes in a symmetric, an asymmetric and a mixed boundary direction ($zs_{01}$ and $zs_{11}$ reach $3\times10^{-7}$ and $1.2\times10^{-6}$ today); the force-balance residual and the Boozer Jacobian are checked at each accepted point. Gate: the least-squares test is in the suite, and the re-solved finite differences agree to $10^{-5}$ relative at the best step. Three pull requests: `alfvex.vmex` added; the examples and tests moved to it; the old names removed, with the pages that cite them. The whole design value and gradient is compiled as one program (the first eager gradient of a fresh process takes about a minute and a thousand compilations in VMEX; V6b). A term whose rule is reverse-only is handled by V17b of the VMEX list until S10 gives the eigenpair rule a forward mode. An implicit solve that missed its tolerance gives NaN tangents after V2c, and the term reports that status instead of a number.
- [ ] **2.4 Geometry from the spline state (VMEX step V10, then here).** Boozer harmonics of $|B|$, $R$, $Z$, the angle offset and the Jacobian, with radial derivatives, at the quadrature points. `Geometry.interpolate` leaves the design path. Gate: the jump of $\partial_s|B|$ across element boundaries is below $10^{-8}$ relative; the gap mode within $10^{-4}$ of its converged value at 24 elements.
- [ ] **2.5 Radial elements of order $p$.** As in 8.3, with $\rho=\sqrt s$ as an option. Gate: the eigenvalue error falls as $N^{-2p}$ on the analytic torus for $p=1,2,3$; the cost at equal accuracy is reported for the ITPA case and the precise quasi-axisymmetric case, and the default $p$ is chosen from it.
- [ ] **2.6 Harmonic sets.** The rule of 8.3 with the convergence test in $m_{\max}$; `alfven_harmonics` stops at $m=12$ today, Paul et al. use 30 to 60. Gate: the isolated global eigenvalues, matched across resolutions by an eigenvector overlap above 0.99, and the edges of named gaps each change by less than $10^{-3}$ (relative) when $m_{\max}$ is raised by 25%; the continuum branches are not gated. Corrected 2026-11-21: the gate read "the reported spectrum changes by less than 0.1%", which cannot be decided: Paul et al. (2025, sec. 5.1, p. 18) state that true convergence of the continuum in $m_{\max}$ cannot be expected, and on the quasi-axisymmetric example the lowest ten branches move by 12 to 63% per step while their count grows with $m_{\max}$. What converges, measured for $m_{\max}=12\to15\to19\to24\to30$: the three gap modes near 102 to 103 kHz move by $1.6\times10^{-4}$, $5.9\times10^{-5}$, $7.8\times10^{-6}$, $3.3\times10^{-7}$ with overlap 1.0000, and the edges of the (2,1)/(3,1) gap by $3.8\times10^{-3}$, $7.7\times10^{-4}$, $2.4\times10^{-4}$, $5\times10^{-5}$, so the gate is first met on raising $m_{\max}$ from 15 to 19, which chooses $m_{\max}=15$. With it the test of 10.3 on the helical gap pair $(3,1)$, $(5,6)$ of the Helias model of Kolesnichenko et al. (2001): $\omega R_0/\bar v_A=1.4406$ and 1.9464 for a homogeneous plasma, to 3%, the accuracy of $\iota(r)$ digitized from their fig. 2; the continuum results that need no profile to $10^{-3}$.
- [ ] **2.7 Mode identity.** One-to-one matching, both overlaps, the cluster trace, the guard band and the rejection rule of 8.9. Tests: invariance of every objective to the phase of a mode and to a rotation inside a degenerate pair; a constructed crossing where the trace is smooth and the individual eigenvalues are not; a mode that enters the window is reported.
- [ ] **2.8 Routines to SOLVAX.** `generalized_eigh`, `eigenvalue_mean` and `lowest_eigenvalues` (S11 of the SOLVAX list), each with its tests, then removed here; `_d_ds` and `angle_mean` are geometry helpers and stay. Gate: the public results are unchanged to $10^{-12}$; the line count falls.
- [ ] **2.9 Low-cost speed.** The closed-form metric inverse; the targeted solve as the default with `dense_max`; `summary` on targeted windows; the continuum in surface chunks with `eigvalsh`; the persistent cache in the runner and the examples; one shape per case. Gates of 12.3, and a test that counts compilations of a repeated call. One pull request per remedy. The gate of 12.3 on prepared weights belongs to step 5.1. Without a target frequency the solve is dense up to `dense_max` and raises above it.
- [ ] **2.10 Memory and sweeps (SOLVAX, then here).** Assembly in radial chunks; the eigenvalue derivative from the quadratic form on the angle grid; in SOLVAX the factors applied as stored block products, a run-time iteration limit, warm starts from the previous step. Gates of 12.3.
- [ ] **2.11 Mode export.** `export_mode` and the record of 8.10 and 9.5, with the conventions table in `theory.md`; a writer for the IMAS `mhd_linear` structure behind an optional import. Test: write, read, and compare the fields at 504 points to $10^{-14}$. Test also: the ratio $\alpha$ of the record against its formula of 8.10 by an independent quadrature, which catches a sign of $\alpha$ or of time that a round trip cannot. The IMAS writer is a pull request of its own, with a test that writes a mode and reads it back under the names of the data dictionary.
- [ ] **2.12 Import without side effects.** The package no longer sets `jax_enable_x64` at import; the entry points raise a clear error when 64-bit floats are off, as VMEX does, and the command and the examples turn them on. Tests: `import alfvex` in a fresh process leaves `jax_enable_x64` as it was; each entry point raises an error that names the setting when 64-bit floats are off. Gate: the installed-wheel job imports the package with 32-bit floats set, finds the flag unchanged, and then runs the quickstart with 64-bit floats. The change is announced in the notes of release 2.0; `alfven_run.py` of vmex relies on the import to enable 64-bit floats and is adapted in the vmex pull request that follows 2.0 (V3b).

- [ ] **2.13 Any equilibrium through arrays.** `Geometry` built from arrays is the contract: the readers (VMEX state, `wout`, `boozmn`) and the VMEX terms all go through it, and a `Geometry` computed by a user's own function of design parameters is passed to `evaluate` as `geometry=` and differentiated through, whatever code produced it. An example builds one from a DESC equilibrium behind an optional import. Tests: the same Boozer arrays through the file reader and through the array constructor give equal results to rounding (this holds today); `jax.grad` through a user's map to a `Geometry` and through `evaluate` equals finite differences. The DESC example runs in the nightly tier only.
- [ ] **2.14 Examples to the rule.** Measured on main, every example exits 0, and these break the rule of section 11: `knyazev_qa.py` imports `tests/knyazev_ref.py` and reads `tests/data`, so it runs only in a checkout, and it is silent for 87 s; `benchmark.py` is silent for 316 s and reads `sys.argv`; `qa_vmex.py` is silent for 35 s during the adjoint gradient; `ctok_boozmn.py` and `qa_vmex.py` neither save nor plot (`quickstart.py` prints only, which is its declared exception); three docstrings give a run time that is off ("about 2 minutes" for 174 s and 248 s). The changes: reference values and fixtures that an example needs are fetched with a hash from a release asset (the wheel holds `alfvex/*.py` only); every stage prints a line before it starts and no interval without output exceeds 10 s on the laptop of 12.1; every example except the quickstart prints, saves and plots at its end; run times in docstrings are the measured ones. `tests/test_examples.py` runs every example at a reduced size set from the test side (today it runs two of seven) and asserts the printed reference number; the three slow ones (174, 248 and 534 s on the workstation) run in the integration tier. Four pull requests: `benchmark.py`; `knyazev_qa.py`; outputs and docstrings; `test_examples.py`. The TOML cases write to an `outputs` folder beside the working directory, not into the checkout. The version string on main is a development version, so that an example does not print the number of the last release. Gate: rules E1 to E9 of the repository rules pass, with the quickstart's exception as the only one. `koenies_tae_drive.py` is renamed `itpa_tae.py`.
- [ ] **2.15 Sector blocks only.** After the stack of step 0.1, a parity sector is assembled on the full conjugate basis and then projected, at four times the cost of the single-sign assembly. Assemble only the blocks that the sector keeps. Test: equal to the projected assembly to $10^{-13}$. Gate: time and planned memory of a sector at or below half of the projected form.
- [ ] **2.16 Release 2.0.** The notes of the release and the first entry of the changelog list what changes for a user: the rename; the basis and the defaults of step 0.1; the design names replaced by `alfvex.vmex`; the import that no longer enables 64-bit floats; the mass defaults of the drive; the Python floor. Before the owner tags: the documented numbers regenerated once more (step 1.9), the cited-test check, the performance test with its stored profile (9.4) and the test of the code blocks of the documentation (D8) are in the repository and pass. Gate: every change marked as breaking in the pull requests of packages 0 to 2 has its line in the notes.

Gate of package 2: a VMEX least-squares run with an ALFVEX term and `jac="implicit"` is in the test suite, and release 2.0 is tagged.

### Package 3. The fluid model: current, pressure, sound

- [ ] **3.1 Current and pressure terms.** `wave = "current"`: the model of 8.3 in weak form, with $\sigma=\mu_0J_\parallel/B$ and $p'$ taken from the equilibrium. Compare each term with `finite_beta=True` of 1.2.0 on the analytic torus and on a VMEX equilibrium, then remove the old switch. Files: `eigen.py`, `geometry.py`. Tests: the operator stays Hermitian to rounding; at zero current and pressure it equals the shear operator; for a manufactured potential the quadratic form $y^\dagger Fy$ equals an independent quadrature of the energy of 8.3 to $10^{-10}$, term by term; the force-balance residual of the geometry is below a stated tolerance where the terms are used. Gate: the ITPA frequency with the CAS3D pressure profile $p(s)=(7170-6811s-358.5s^2)$ Pa within 1.5% of $4.01\times10^5$ rad/s (CAS3D, a compressible calculation; LIGKA 4.00, NOVA 4.055: the codes spread over 1.4%). A solved mode on a regenerated VMEC equilibrium of that case sits at $3.949\times10^5$ rad/s with the terms of 1.2.0, 1.5% low. The step opens with the derivation in `theory.md`, read by the owner before any code. It has to settle the operator: the equation of 8.3 as CKA solves it carries the factor $(1-\mu_0p/B^2)$ on the bending term and neither the $\boldsymbol\kappa u$ term nor the fourth term of the form of 1.2.0, and the current term alone is not Hermitian where $\mathbf b\cdot\nabla\sigma\ne0$ (1.2.0 takes its Hermitian part): the Hermitian part is kept and the norm of the rest is reported. The pressure comes from the equilibrium, and a pressure given in `Plasma` is refused with this model; `Geometry` gains the pressure and the radial covariant field, which it does not hold today (9.5). The VMEC input of the ITPA case with $p(s)$ is committed as a fixture. The removal of `finite_beta` is announced in the changelog. Four pull requests: the equilibrium fields; the terms with their quadrature tests; the fixture and the gate; the removal of the switch.
- [ ] **3.2 The open global comparisons.** With the current term from the equilibrium: the 337.92 kHz mode of the CTOK case and the parallel-current block of AE3D v31, compared block by block on a common test function; an instrumented AE3D run with current and pressure off moves that mode to 338.18 kHz, so the difference is not in the switch. The precise quasi-axisymmetric 30.309 kHz mode, on both parity sectors, once a deck with an independent density is available (section 16). Gate: each mode is either matched in frequency, overlap and projected Rayleigh quotient, or recorded as unmatched with the block that differs.
- [ ] **3.3 The sound-coupled operator.** The derivation of 8.4 in `theory.md` with every dropped term, then `wave = "sound"` in `global_matrices`: blocks of size $2H$, the same radial elements, the same solver. Files: `eigen.py`, `continuum.py`. Tests: Hermitian to rounding with a positive definite mass; at $p=0$ the $\chi$ block alone; the surface limit against the pencil of 8.2 by extrapolation in the radial wavenumber, to $10^{-8}$ (at one finite wavenumber the corrections are of order $1/k$); the polarization test of 8.4 (the sign of the geodesic coupling); $\delta B_\parallel$ of the stored mode satisfies perpendicular pressure balance to rounding. The unknown $c$ has no radial boundary condition, and $c=0$ on the axis for $m\ne0$. The fourth term of 8.3 is the case key `model.compression_term`. `Modes` gains $c$ and the harmonics of $\delta B_\parallel$ (9.5). The wave energy in the variables $(\chi,c)$ is written in the derivation, since 8.5 gives it for $\Phi$ only, and is checked against 8.5 at zero pressure. Three pull requests: the derivation and the switch; the assembly and its tests; $\delta B_\parallel$ and the polarization. Gate: the tests.
- [ ] **3.4 Global sound gates on a tokamak.** The geodesic acoustic frequency of Winsor and the acoustic gap top $c_s/(qR)$ are edges of the continuum, not isolated modes: the extreme global eigenvalues converge to them as $N^{-2}$, within $10^{-3}$ at flat temperature and density and $a/R=0.02$ (Winsor's formula is a large-aspect-ratio one); the system of Cheng and Chance (1986, eqs. 7 to 9) solved by root search on the same equilibrium ($R/a=4$, $\beta=0.04\%$, $q(0)=1.0408$, $q(a)=2.3$, $n=1$, fixed wall) as an independent reference kept in the tests, with the TAE at $(\omega/\omega_A)^2=0.50\pm0.03$, $\omega_A=v_A(0)/[q(a)R]$, on the $q$ profile inferred in the reading notes (the paper prints neither $q(r)$ nor $p(r)$); the root solver of their equations is a pull request of its own; the frequencies fall to those of 3.1 as $\beta\to0$.
- [ ] **3.5 Sound in three dimensions.** The continuum against CONTI on HSX (Könies and Eremin 2010: shear $(1,-1)$ coupled to acoustic $(0,3)$ and $(2,-5)$, peaked and flat pressure, the gap near 47 to 88 kHz) and on H-1 (Bertram et al. 2012), then one sound-coupled global mode with a published counterpart. A gap that the calculation keeps where the experiment loses the mode (the high-mirror HSX case) is reported, not tuned away. Gate: the analytic case of Kolesnichenko et al. (2011) on the same code path, $\omega_s=18.5\pm1$ and $\omega_{G0}=10.5\pm1$ kHz for the W7-X $(9,8)$ harmonic with the inputs of 10.3. The HSX and H-1 cases are reported with the inputs that had to be assumed and become gates when their decks are obtained (section 16).
- [ ] **3.6 Doppler shift.** The $E\times B$ rotation from $E_r$ of the `Plasma` shifts $\omega$ to the laboratory frame per harmonic. Test: a rigid rotation shifts every mode of toroidal number $n$ by $n\Omega_E$. In a stellarator a mode family mixes several toroidal numbers, so the shift enters the inertia as $(\omega-n_j\Omega_E)^2$ per harmonic and the problem becomes quadratic in $\omega$; it is solved by linearization. $\Omega_E$ is defined from $E_r$, with its sign, in section 7, in the derivation that opens the step. Test also: a family with two toroidal numbers against the dense quadratic problem. Gate: the tests.
- [ ] **3.7 Published spectra.** Further targets from the collaborators' evidence, each matched or recorded as blocked by its missing input: the continuum against CONTI on W7-X ($\beta=0.583\times10^{-2}$, $n_e=2\times10^{20}$ m$^{-3}$), W7-AS 56358 and LHD; the modes of LHD 47645 at 37 and 73.7 kHz and its reversed-shear modes at 49.7, 64.9, 88.1 and 101.6 kHz (Spong et al. 2010); TJ-II at 148 and 330 kHz; QPS at 149 kHz; W7-AS 40173 and 42873 (Spong et al. 2003). Reported per case: residuals, units, the energy identity, angular and radial convergence, and the agreement of the eigenfunction or of the subspace. Gate: one row per target in the validation page: matched, with the difference, or blocked, with the missing input named. One pull request per device.

Gate of package 3: one sound-coupled global mode with a published counterpart whose inputs are complete (the tokamak case of 3.4 if no stellarator deck is obtained), and the frequency gate of 3.1.

### Package 4. Dissipation and the complex spectrum

- [ ] **4.1 Resolvent reference.** `resolvent` and `continuum_damping` in `damping.py`: samples of $b^\dagger(F-z^2G)^{-1}b$ on a line, a rational fit, the pole, with the sampling rule of 8.8. Host-side reference first, then on the batched complex solves of step 4.3. Tests: a two-level analytic model with a known pole; the one-dimensional two-harmonic model (Berk et al. 1992, eq. 35, in $\Omega=\omega R_0/v_A(0)$ and $r=\sqrt s$) discretized and passed through `resolvent`, against $0.370887-0.00174204i$ to $10^{-4}$; the Könies and Kleiber tokamak case with `wave = "current"` and harmonics $m=1$ to 4. Gate: the two-level model and the one-dimensional fixture. The tokamak case is reported in this step, the fit against the reading of the figure, and becomes a gate after 4.5, when a second method exists: the two agree to 1% and both are within 5% of $\mathrm{Im}\,\Omega^2=-8.9\times10^{-4}$. The rational fit is step S13 of the SOLVAX list. Three pull requests: the two-level model; the one-dimensional fixture; the tokamak case.
- [ ] **4.2 The case of Bowden et al.** Close the 19% between the resolvent result with the current term ($0.374936-0.002074i$) and the published $0.39515-0.00174465i$: run both the printed $\iota$ polynomial and $q=1.05+0.55r^2$, a first-order torus, and a VMEX equilibrium; ask the authors which profile figure 1 used. Then the scan of Berk et al. (1992) in $q(a)$ through zero, one and several resonances. Gate: a stated cause for the difference, or the case recorded as open with both variants.
- [ ] **4.3 Complex pencils (SOLVAX, then here).** Batched complex shifts and the generalized eigenpairs of a complex-symmetric block-tridiagonal pencil with left and right vectors, usable under `jit` in forward and reverse mode. `Status` gains the left residual, the biorthogonality and the contour or dissipation used. Tests in SOLVAX: analytic two- and three-mode pencils, a supplied pair that is not an eigenpair is refused, a coalescing pair is reported. Step S14 of the SOLVAX list covers the complex-symmetric case, which holds in a parity sector. On the complex Fourier basis the left vector is not the conjugate of the right one: contours are restricted to parity sectors, unless a SOLVAX step for general non-Hermitian block-tridiagonal pairs with left vectors is done first (S14b). Gate: the SOLVAX tests, and one ALFVEX pencil of a parity sector solved with both residuals below the tolerance.
- [ ] **4.4 Derivative of a damped mode.** The rule of 8.9 on the eigenpair started from the resolvent pole. Gate: the derivative of $\mathrm{Im}\,\omega$ with respect to a profile parameter and to a boundary coefficient equals the finite difference of re-solved poles on the same contour to $10^{-4}$. The resolvent pole is not an eigenvalue of the Hermitian pencil: the eigenpair is that of the dissipative pencil of 4.6 or of the contour of 4.5, so this step follows both. The tolerance is relative, on $\mathrm{Im}\,\omega$. A boundary coefficient enters only through the polynomial fit of 4.5.
- [ ] **4.5 Contour method.** The contour of 8.8 for analytic and polynomial-fitted equilibria, as the third method. The root of the two-harmonic model is also found by an independent Riccati shooting solve, so that contour and shooting agree before either is trusted. The torsatron values of Bowden et al. (TAE $0.29031-0.00333875i$, EAE $0.48197-0.00248407i$) are recorded as targets and become gates once their equilibrium is reproduced. Gate: the root does not move beyond the discretization error under two contour shapes and three depths; non-physical poles of the fitted equilibrium are counted and stay outside. `Geometry.from_arrays` casts to real numbers and the assembly takes Hermitian parts; on a contour the geometry is complex and the assembly keeps transposes, so the step adds a geometry on complex $s$ and a symmetric assembly. The contour is drawn in $\rho=\sqrt s$.
- [ ] **4.6 The non-ideal term.** The mixed form of 8.8 with the vorticity unknown; real $\varrho_K^2$ first. The real term is not a damping: its case key is `wave = "kinetic"`, and `damping = ["radiative"]` adds the complex part of step 4.7. Tests: the local dispersion relation $\omega^2=k_\parallel^2v_A^2[1+k_\perp^2(\tfrac34\rho_i^2+\rho_s^2)]$ to $10^{-8}$ in a uniform field; Hermitian to rounding; the kinetic TAEs of Candy and Rosenbluth at small shear, $g_K=1+\lambda/\sqrt2$. The mixed form leaves the rows of the vorticity unknown without inertia, so $G$ is singular, and the eigen-solver needs a positive definite $G$: the unknown is eliminated with a lumped mass, or an indefinite solve is stated and tested in SOLVAX first. The unknown vanishes at $s=1$. The dispersion relation holds to $10^{-8}$ only with both parts of $\varrho_K^2$ on the bending side; with the ion part on the inertia side, as 8.8 has it, it holds to first order in $k_\perp^2\rho_i^2$, and the tolerance of the test is the size of the next order. Gate: the tests.
- [ ] **4.7 Radiative damping.** A complex $\Lambda_e$ and the independence check. Gates, each on the dimensionless eigenvalue of the source: Candy and Rosenbluth (1994), $\lambda=0.1(1-0.2i)$: $g_M=-1.014-0.0707i$ at small shear, unchanged when $\mathrm{Im}\,\lambda$ doubles, and least damping $\mathrm{Im}\,g_M\approx-0.018$ near shear 0.32; Berk, Mett and Lindberg (1993), $n=1$, $q=1+2(r/a)^2$: $-\mathrm{Im}\,h=6.5\times10^{-5}$ at $|\eta|=10^{-3}$ and $3.8\times10^{-3}$ at 0.03; Mett and Mahajan (1992, table I), $n=2$, $m=2,3$: $\gamma/\omega=-5.3\times10^{-2}$; the tunnelling law of 8.8 at shear 0.3, $\hat\varepsilon=0.1$, $a=3$, where $-\gamma/\omega_0\approx6.7\times10^{-5}$ and the eigenvalue must be accurate to $10^{-5}$; the small-dissipation limit equal to the resolvent pole of 4.1. The interpolation formula 122 of Berk, Mett and Lindberg is not used; its integral is computed by quadrature. Breizman and Sharapov (2025) write $\rho_i^2$ with $2T_i$ under their eq. 1 and with $T_i$ below eq. 7: the convention of section 7 is stated next to each formula taken from them. Which of these can gate. The global case of Berk, Mett and Lindberg is their third, with TAEs at $\Omega\approx0.300$ and 0.328; $6.5\times10^{-5}$ belongs to their layer problem with a prescribed coupling and is compared with a solve of the layer equation, not with a global solve. $g_M=-1.014-0.0707i$ is the small-shear asymptote (the figure reads $-0.066$); the worked point at shear 0.3, $-0.8614-0.0165i$, is compared with the layer equation. The tunnelling law is checked against the layer equation as well, since two published prefactors differ by a factor 1.5 at $a=3$. The table of Mett and Mahajan is reported only: its profiles are given as parabolic, which does not determine them. The maps from $\omega$ to $g$ and to $h$ are written in the derivation that opens the step. One pull request per source.
- [ ] **4.8 Thermal species on a mode.** The form of 8.8 for electrons and ions with the couplings $\Psi$ and $\delta B_\parallel$, on the resonance weights of step 5.1; then the collisional trapped-electron term. Tests: the electron and ion parts vanish for a mode without $\Psi$, curvature and compression; the Landau integral equals $(n/T)\zeta Z(\zeta)$ at $\gamma/\omega=-0.1\%$ and $-10\%$; the collisional curve of Fu, Cheng and Wong (1993, fig. 5) within a factor 1.5, with the slope and the saturation of 10.3, once the TFTR equilibrium is rebuilt from the printed profiles. Gates on the ITPA case: bulk damping without fast ions inside the published spread, $-567$ to $-1705$ s$^{-1}$, with the co- and counter-propagating modes separated ($-1052$ and $-1517$ s$^{-1}$ in LIGKA); the collisional part of the order of the NOVA-K value, $-237$ s$^{-1}$. Before the tokamak case: a slab kinetic Alfvén wave with collisions against the Legendre reference of Slaby, Kleiber and Könies (2017), frequency and damping against collisionality; and the frozen-mode scans of Mishchenko, Könies and Hatzky (2011), in temperature and in charge at fixed temperature, both at fixed $T_i/Z_i^2$ ($B_0=3.45$ T, $R_0=3$ m, $a/R=0.333$, $n=-6$), where the frequency follows the pressure and the stabilisation follows the gyroradius. This step needs the weights of step 5.1 and follows it. The bulk damping of the ITPA case is reported, not gated, until step 5.7 gives its second evaluator: the published spread includes effects beyond Landau damping. Six pull requests: the Landau term with $\Psi$ and $\delta B_\parallel$; the checks against the plasma dispersion function; the slab wave; the collisional term; the ITPA case; the two scans.
- [ ] **4.9 Kinetic inertia of the continuum.** The two steps of 8.8. Tests with complete inputs: at $\tau=1$, $q=1.5$ the leading-order frequency $\omega/\omega_{ti}=2.487$, the root of the full relation 2.797, the large-$q$ formula 2.853, and the damping from the numerical root of the full relation (Lauber 2013, eqs. 25 and 26, evaluated); the large-$q$ formula gives $\gamma/\omega_{ti}=-0.040$, an order check only, since $q=1.5$ is outside its range; the accumulation points of Zonca, Chen and Santoro (1996) at $\beta_i=0.01$: $2.72-0.07i$ at $\Omega_{*n}=1$ and $2.78-0.05i$ at 0.5, each $\pm0.03$, and $\Omega=\Omega_{*n}$ for the kinetic ballooning branch at $\eta_i=0$; the diamagnetic gap $\omega(\omega-\omega_{*pi})=k_\parallel^2v_A^2$. Then the form with several sidebands, with the analytic sound-coupled continuum of Kolesnichenko et al. (2011, eqs. 7 to 14) as its fluid limit. The first of the two steps, the fluid pencil with $\gamma_i=7/4$ and $\gamma_e=1$, keeps the factor $1+1/(2q^2)$ and gives $2.750\,\omega_{ti}$ at $q=1.5$; 2.487 is the leading term of the local relation and is a test of the second step. The quadratic surface problem is solved with `quadratic_eigenpairs` of SOLVAX (S11) and the complex root per surface with its certified root solve (S6d).
- [ ] **4.10 The parallel field in the mode record.** $\Phi$ and $\alpha$ as independent profiles when the non-ideal term is on. Test: $E_\parallel$ of the exported mode equals $-\mathbf b\cdot\nabla\Psi$ of the solve. The relation between $\alpha$, $\Phi$ and $\Psi$ at finite $E_\parallel$ is written in 8.10, in the convention of section 7, before the code: 8.10 gives it for $E_\parallel=0$ only. Gate: that test.

Gate of package 4: one continuum-damping number and one radiative-damping number matched to a source whose content is known, each obtained by two methods, and the derivative gate of 4.4.

### Package 5. Kinetic response

- [ ] **5.1 Weights and the ledger.** Split `growth_rate` into the resonance weights of 8.6 and the sum over a distribution; `stability` returns the ledger of 8.5; `vmap` over resonances in chunks. Tests: the four ITPA rates and the table of 8.6 to $10^{-10}$; 24 values in temperature and frequency from one set of weights to $10^{-14}$; the bins of power sum to each total to $10^{-12}$; $W=4.279791098\times10^{-5}$ J on the 1 V envelopes; the derivative of $\gamma$ with respect to a field parameter through the eigenvector against re-solved differences to $10^{-8}$ (measured $7\times10^{-10}$; $+16569.88$ s$^{-1}$ per unit parameter on the 32-surface case, of which $+758.58$ from the eigenvector and $-79.37$ from the frequency). Gates of 12.3. With it the test of 10.3 on the Landau damping of a flat Maxwellian (Kolesnichenko and Tykhyy 2018): $\gamma/\omega=-0.00998227$ at $x=1$, to $2\times10^{-5}$ at three resonances. The four rates and the table of 8.6 are the code's own output at the atomic deuterium mass: they are a regression, regenerated after step 1.5 with the nuclear mass. The value of $W$ is pinned on linear elements, which step 2.5 generalizes. The derivative test names its field parameter and its case in the test. Three pull requests: the weights; the ledger; the derivative.
- [ ] **5.2 The source-first reference and the full coupling.** A slow reference in the tests, written from the functional of Könies (2000) for one passing orbit and two harmonics, in which the source operator acts on the Lagrangian before the orbit projection; then the coupling $L$ of 8.6 with $\delta B_\parallel$. The reference decides the form of the radial term. Tests: the cylindrical single-harmonic limit of Könies (2000, eq. 90); invariance under a common phase; the drift-work limit at $\delta B_\parallel=0$ equal to step 5.1. The change of the four ITPA rates and of their increments is reported with the result. Gate: the source-first form and the multiplier form agree to $10^{-8}$ on the two-harmonic orbit; if they do not, their difference is the reported result, and the form of the radial term follows the source-first one.
- [ ] **5.3 Precession and trapped ions.** The orbit-averaged drift in the passing resonance; bounce harmonics with $\omega=\ell\omega_b+n\bar\omega_{\rm pr}$ for trapped ions in one well, then over all wells with the bounce kernels of VMEX; the resonance $\omega+s\omega_b-(m\pm1)\omega_D=0$ of locally trapped ions (Kolesnichenko et al. 2011). No published growth-rate formula for trapped ions in a stellarator was found; the expression is derived here and checked by step 5.6. Tests: the pitch measure of passing and trapped ions together recovers $4\langle B^{-2}\rangle$ to $10^{-5}$; at zero precession on the ITPA case reflected ions give 0.0232, 1.359, 4.002 and 11.07 s$^{-1}$ (within 1%); with precession they add 7 to 29%, to be met by the orbit-integrated evaluator within 5%; the drive changes sign exactly where $\omega_*=\omega$. The settings behind these numbers are part of the tests: the reflected-ion rates were obtained with the radial source $n/\iota$ on 400 surfaces, 64 pitches and 768 orbit points with 48 harmonics, and the 7 to 29% with a constant toroidal precession, with which the passing rates also change by $-4$ to $+48\%$. The comparison with the orbit-integrated evaluator is part of step 5.6. The bounce kernels over all wells are step V28 of the VMEX list, which precedes the stellarator part. The bounce quadrature and the weighted Gauss rules come from SOLVAX (S21), and bounce points from its certified root solve (S6d).
- [ ] **5.4 Distributions.** Those of 8.6 item 6, each with its normalization test and its derivatives at fixed invariants. Tests: the alpha drive of Fu and Van Dam, $\gamma/\omega=\tfrac94\beta_f(\omega_*/\omega-\tfrac12)F(v_A/v_\alpha)$ with $F(x)=x(1+2x^2+2x^4)e^{-x^2}$, in its limit; the local threshold of Cheng (1991, eq. 37) for slowing-down alphas at $m=2$, $q=1.5$, 10 keV, for its three machines (TFTR at $v_\alpha/v_A=1.88$, CIT and ITER at 1.5). The printed coefficient $3-4\hat v^2$ should read $3-4\hat v^4$, and the figure and the quoted $2.5\times10^{-4}$, $7.5\times10^{-4}$ and $1.2\times10^{-3}$ follow the print. The test evaluates both forms: 2.61, 7.76 and $11.9\times10^{-4}$ as printed, 2.67, 8.12 and $12.8\times10^{-4}$ corrected. It tells them apart on CIT and ITER, to 2%, where they differ by 4.6% and 7.6%; on TFTR they differ by 2.3% and the value is reported. A sum of three slowing-down populations, which step 5.10 needs, is one of the distributions. The code is in `plasma.py`, one pull request per distribution.
- [ ] **5.5 Stellarator resonances.** The Helias case of Kolesnichenko et al. (2002): $N=4$, $\iota_*\approx0.886$, $|\varepsilon_h/\varepsilon_t|\approx1.9$, $v_\alpha=3.2v_A$, $\omega_d/\omega=10$. Tests: the resonant speeds $v_\parallel/v_A=[1+2j(\mu\iota_*-\nu N)/(\mu_0\iota_*-\nu_0N)]^{-1}$, for example 1.795 and 0.693 for the mirror gap mode; the growth rates of the TAE, EAE, HAE(2,1), HAE(1,1), MAE and HAE(2,2) relative to the TAE, 1.000, 1.988, 4.727, 2.072, 4.371 and 1.068, within 5%, with `helical = 2`. Kinetic formulas of those papers map to this file with $(m,n)\to(-m,-n)$; the prefactor of ion Landau damping is $\sqrt\pi/6$, as in the 2002 paper, not the $\pi/6$ printed in the 2011 review.
- [ ] **5.6 The orbit-integrated evaluator (ESSOS).** As in 8.7, after step 6.1. Gates on the ITPA case: zero width, passing ions, within 0.5% of step 5.1; with reflected ions within 5% of step 5.3; at zero width, the increments $\gamma(175)-\gamma(100)$, $\gamma(200)-\gamma(100)$ and $\gamma(300)-\gamma(200)$ against 15.8, 20.0 and 22.3 $\times10^3$ s$^{-1}$ of the CAS3D-K curve, which is a zero-orbit-width result, reported with the ratio; with orbit width and without Larmor radius, the absolute rates against the band of the eight codes of the benchmark (mean 8.5, 28.2, 38.1 and 42.8 at 100 to 400 keV, $\pm15\%$); with Larmor radius, inside 3.5 to 8.6, 14.0 to 19.9, 19.2 to 26.2 and 18.7 to 27.9. The published drive of the TFTR beam case falls by a factor 7 to 13 for trapped ions and rises by 3.9 below the primary resonance when orbit width is kept (Fu, Cheng and Wong 1993, figs. 2 and 3): the same direction is required here. ESSOS needs a zero-orbit-width mode of the tracer and the estimator of $D_{EE}$ and $D_{E\bar s}$ (E14d of its list). The codes of the benchmark label the distribution by the local surface; the orbit-averaged label of 8.7 gives 8 to 11% less: the benchmark runs use the local label, and both are reported. This is the step most at risk: the prototype gives 30.5 and $30.6\times10^3$ s$^{-1}$ at 300 and 400 keV, below the lower edge of the band (32.4 and 36.4), where the orbit shift exceeds $1/k_\theta$. A result outside the band is reported with that reason; nothing is tuned.
- [ ] **5.7 Finite-frequency drift-kinetic solve (DKX).** The term $-i\omega$, a complex state and a wave source on a surface, for electrons and thermal ions with the linearized Fokker-Planck operator. Tests: density, momentum and energy conserved; no response of a Maxwellian to a uniform potential; the collisionless limit equal to step 4.8; the trapped-electron curve of step 4.8. DKX has no work list in the programme: its two changes, the term $-i\omega$ with a complex state and the wave source, are written as steps and put to the owner before this step starts. Gate: the tests.
- [ ] **5.8 The real part.** The perturbative frequency shift from $\mathrm{Re}\,\delta W_k$. Gate, on the shifts from the value without fast ions on the ITPA case (CAS3D-K: 4.012, 3.968, 3.880 and $3.732\times10^5$ rad/s at 0, 100, 200 and 300 keV): $-4.4$, $-13.2$ and $-28.0\times10^3$ rad/s, each within $0.5\times10^3$; a tolerance of 1% on the frequency would exceed the first shift. The resonance weights of 5.1 hold the resonant part only: the principal values and the adiabatic terms of Könies (2000, eqs. 80 and 81) are a pull request of their own.
- [ ] **5.9 The response in the eigenproblem.** $D(\omega)y=[F-\omega^2G-\sum_a\Pi_a(\omega)]y=0$ with the root solver of SOLVAX and the derivative of 8.9. The fluid terms that the kinetic response replaces are removed, so that nothing is counted twice; the fluid model is recovered in its limit. Gate: release 3.0 of section 2. $\Pi_a(\omega)$ is defined in 8.8 before the code: its continuation below the real axis and the fluid terms it replaces. The comparison of release 3.0 is with the non-perturbative codes of the benchmark (Könies et al. 2018, fig. 2c), not with the perturbative curve. The step follows S15 of the SOLVAX list. Three pull requests: the response; the root and its derivative; the removal of the fluid terms.
- [ ] **5.10 First stellarator kinetic case.** The $n=2$ TAE of W7-AS discharge 39042 once its equilibrium is obtained: frequency $3.33\times10^5$ rad/s (CKA; CAS3D $3.23\times10^5$), the $m=6$ harmonic peaking at $\sqrt s=0.87$; the CAS3D-K drive 2.0, 6.56, 10.75 and $16.5\times10^3$ s$^{-1}$ at $v/v_A=0.431$, 0.557, 0.642 and 0.719; ion Landau damping at 400 eV between $-0.9$ and $-2.7\times10^3$ s$^{-1}$; reflected ions 15 to 25% of the passing drive (Könies et al. 2008). Gate: with the equilibrium in hand, the frequency lies between the two published values or within 3% of one of them, and the increments of the drive between the four speeds are reported with their ratio to CAS3D-K; until then the step is recorded as blocked by the missing equilibrium (section 16) and no number is claimed. The field, the size, the pressure and the VMEC input of the case are not printed, and its distribution is the sum of three slowing-down populations of step 5.4.

Gate of package 5: three evaluators of the passing drive agree to 0.5%; the stated increments and bands of 5.6 are reported; no rate is matched by a factor.

### Package 6. ESSOS: the interface, collisions, saturation

- [ ] **6.1 One wave constructor (ESSOS).** Steps E8, E12 and E13 of the ESSOS list, in that order. An array constructor that reads the record of 8.10 in the conventions of section 7, replacing those of pull requests 177, 196 and 235: no Python branch on traced values, no positive-frequency guard, zero outside the radial grid (pull request 235 clamps), the axis behaviour $s^{|m|/2}$, spline interpolation without derivative jumps (ESSOS's own twice-differentiable spline until SOLVAX step S16 is released and ESSOS adopts it), built from live arrays as in pull request 118. `psi0_from_vmec` loses its `float()` cast. Gate: the field and its derivatives agree to $10^{-13}$ with a mode file written by `export_mode` (step 2.11) and committed to the tests of ESSOS with its schema version and hash; `jax.jvp` through the constructor equals central differences to $10^{-9}$.
- [ ] **6.2 Interface tests.** Those listed in 8.10, in ESSOS (E12, E13, E14, E13b and E16 of its list); on this side the round trip of the record is the test of step 2.11. Main of ESSOS has no perturbation in either coordinate path today: the Boozer path comes first (E12 and E13 of the ESSOS list) and the VMEC path after it has passed these tests (E13b). Both are then covered: the Boozer tracer, which drops the radial covariant component, and the VMEC tracer; the two paths are compared with the radial covariant term included (E16), and without that term the difference in resonant work is reported, not gated. Gate: every test of 8.10 passes in ESSOS on the committed mode file.
- [ ] **6.3 Cross-check against FIRM3D.** In ESSOS, in the scheduled tier, with 1000 markers per case and seed; fixtures under 1 MB are fetched with a hash and the build of FIRM3D is recorded in the provenance table. Of the archive only the quasi-axisymmetric mode is in the repository today; the other two are fetched first. On the quasi-axisymmetric, quasi-helical and quasi-isodynamic modes of the Knyazev et al. (2026) archive at matched $\delta B/B$: invariants and loss fractions first, trajectories at early times only. The frequency conversion of FIRM3D's AE3D reader is checked on a known case before. Reference points: passing-orbit stochasticity in the quasi-helical case rises near $\sqrt{\delta U/U_0}=10^{-4}$ and prompt passing losses start near $10^{-3}$. Gate: the wave invariant is conserved to the integration tolerance in both codes; loss fractions agree within the Wilson 95% intervals of both codes at $\delta B/B=10^{-4}$ and $10^{-3}$; the amplitudes at which stochasticity and prompt losses set in lie within half a decade of the reference points.
- [ ] **6.4 Collisions (ESSOS).** This is step E11 of the ESSOS list, and its gates are checked there. ESSOS has no fixed logarithm and no deposition per species today; both are added first (E11 of the ESSOS list). A stated Coulomb logarithm shared with the reference: ESSOS and ASCOT5 drag, parallel diffusion and deflection agree to $2.8\times10^{-6}$ with a common logarithm and differ by up to 4.61% with their own. The Itô drift of the speed, which differs by up to 1.34% because ESSOS differentiates its speed-dependent logarithm, is tested in finite-time slowing down against ASCOT5. The pitch step near $|\xi|=1$: the Legendre moments obey $\langle P_\ell(\xi)\rangle=P_\ell(\xi_0)e^{-\ell(\ell+1)\nu_Dt/2}$, and the one-step error of the second moment at $\xi_0=0.99$ is $-3.3\times10^{-3}$ at $\nu_D\Delta t=0.01$ and $-6.3\times10^{-5}$ at $0.001$. Gates: the moment law within the sampling error over 100 collision times; slowing-down time and deposited power per species within 1% of ASCOT5 for a 3.52 MeV alpha in a 5 keV plasma of deuterium, tritium and electrons at $10^{20}$ m$^{-3}$, with $10^5$ markers and three standard errors of the mean. The ASCOT5 numbers are a fixture taken from the collaborators' evidence; the Spitzer slowing-down time is the analytic check beside it.
- [ ] **6.5 The energy ledger and the factorial.** Per-marker records of 9.5. Four runs on the same markers: waves on and off, collisions on and off. The ledger: initial plus injected energy plus net field work equals retained energy plus escapes plus collisional deposition, within the integration error: the ledger is per marker and on fixed keys, so no sampling error enters it. Boundary crossing and wall interception are separate events. Deposition profiles go to NEOPAX in SI with shell volumes, in place of its local alpha-heating model for that population; profiles are iterated only if the ranking of designs changes. Reference: on 100 markers the archived benchmark gives 12 to 13 losses at the boundary and 0 to 8 inner stops with the wave on; the published 5000-marker end points are 16.2% and 19.4%. Steps E6, E9, E10 and E14 of the ESSOS list come first. Three pull requests: the ledger per marker; the four runs on 5000 markers with Wilson intervals; the export to NEOPAX, once a step for it is written in that repository (none exists). Gate: the ledger closes to the integration tolerance for every marker in each of the four runs; the 5000-marker loss fractions agree with 16.2% and 19.4% within their Wilson intervals (about one percentage point). The 100-marker counts are a smoke test only: 13 of 100 has an interval of 7.8 to 21% and cannot tell the two apart.
- [ ] **6.6 Amplitude evolution and saturation.** The amplitude equations of 8.11 with fixed mode structure, the power taken from the markers through the work integral of 6.2, and pitch-angle scattering on the markers. It needs 6.5 and is four pieces of work, each at least one pull request: (a) `saturation` and the near-threshold equation in `transport.py`, in ALFVEX alone; (b) a one-dimensional bump-on-tail model with marker weights, as the referee of the feedback loop; (c) a time-dependent amplitude fed back from the markers in ESSOS (E14b of its list; E12 holds the amplitude fixed); (d) the ITPA single mode, with the damping rate actually used stated. For (c) and (d) the number of markers needed to resolve the saturated amplitude is estimated and written down before the run. Gates: the oracle numbers of the near-threshold equation of 8.11; a collisionless single-mode run saturates at $\omega_b/(\gamma_L-\gamma_d)=3.2$ within 10%; with scattering the level follows $(\nu_d\,\omega^2\gamma_L/\gamma_d)^{1/3}$ over a decade of $\nu_d$; on the ITPA single mode $\gamma_L$ is within 5% of $4.723\times10^4$ s$^{-1}$ at doubled fast-ion density and the saturated $\delta B/B$ is reported against $2.1\times10^{-3}$.
- [ ] **6.7 Magnetic compression in the orbits (ESSOS).** The term $\mu\,\delta B_\parallel$ in the guiding-centre Hamiltonian, with $\delta B_\parallel$ of 8.4. Test: the work on a deeply trapped particle in a uniform $\delta B_\parallel$ oscillation equals $\mu\,\partial_t\delta B_\parallel$ integrated along the orbit. Step E12b of the ESSOS list. The mode record gains the harmonics of $\delta B_\parallel$ and of the ratio $\alpha$ (9.5). Gate: that test in ESSOS, and zero $\delta B_\parallel$ bitwise equal to the run without the term.

Gate of package 6: a computed mode goes from ALFVEX to ESSOS in one call, the invariants and the ledger close, and the loss fraction of FIRM3D's quasi-axisymmetric wave example (0.62%) is reproduced within its Wilson interval (E12 of the ESSOS list).

### Package 7. Transport and objectives

- [ ] **7.1 Geometric.** The continuum cost of Paul et al. (2025) as implemented, with normalized radial quadrature weights whose meaning is stated (plasma volume, birth profile or drive) in place of a plain sum over samples, with the $2\times2$ gap test of 8.12 item 1. Gate: that test and a test of the weights (they sum to one; a uniform weight reproduces the plain mean). The gap widths that STELLGAP gives at the starting and at the final boundary of the quasi-helical example are reported, not gated. With it the two-harmonic gap test of 10.3: width proportional to $\lvert\varepsilon_1+\varepsilon_2\rvert$, half-width $\lvert\varepsilon_B-\varepsilon_g/2\rvert$.
  Decided 2026-11-21: the step provides both costs of 8.12 item 1, eq. 7.4 exactly as published (the control of step 8.3) and the sum on $|\varepsilon_1+\varepsilon_2|$ (the objective), with a test that the two agree when the $\varepsilon_2$ input is set to zero (an identity of the code, not a physical state).
- [ ] **7.2 Spectral.** The resonance-weighted window and the forced response of 8.12 item 2, over a set of density profiles. Tests: the weight integrates to one; the penalty is unchanged by a relabelling of modes; the forced response equals the sum over modes of residues on a small dense case.
- [ ] **7.3 Linear kinetic.** `critical_beta` and the log-sum-exp bound on the ledger of 8.5. Test: the bound of 8.12 item 3 on random sets; $\gamma_{\rm drive}\propto n_f$ at frozen mode. The range 0.5 to 1.0% published for the critical fast-ion pressure of a reactor-scale stellarator is a plausibility check, not a gate.
- [ ] **7.4 Transport proxies.** The amplitude at which two resonances overlap, in `transport.py`, after 6.6. Tests: the island width of White and Chance (10.3) for each resonance, and the overlap amplitude of two resonances against the sum of their half-widths; the derivative against finite differences. The flux of the reduced model of 8.11 is computed in a test as a published comparison and is not an objective. Release 2.3 holds the overlap amplitude; a quasilinear flux is for release 3.0. Gate: the tests are in the suite and the proxy enters the table of 7.6.
- [ ] **7.5 Direct loss.** The energy-weighted loss from ESSOS in ALFVEX modes with paired markers, a smooth indicator and fixed random numbers. Tests: the smooth loss tends to the hard one as the smoothing scale falls; the derivative with respect to a boundary coefficient equals a central difference after rebuilding equilibrium, transform, modes and orbits on both sides with the same markers. Steps E6, E8 and E15 of the ESSOS list and V11 of the VMEX list come first, and the chain from the VMEX state to the Boozer field of ESSOS goes through them. The horizon is at most $10^{-4}$ s, since no gradient is offered through long chaotic orbits; 256 markers on fixed keys; the amplitude is held fixed in $\delta B/B$ across the finite difference; the smoothing width is stated. Gate: relative agreement to $10^{-4}$ at three step sizes. The code is in `transport.py` with ESSOS imported inside the function; the public name `direct_loss` joins 9.2, and the line budget of `transport.py` in 9.1 is revised in the same pull request.
- [ ] **7.6 Ranking.** The set of configurations is built, stored as boundaries and hashed by a pull request of its own first: three come from the archive, the others are generated. Over at least 20 configurations, the rank correlation of each proxy with the direct loss, with its interval. The three equilibria of the Knyazev et al. (2026) archive come with published loss tables. The usual proxies (quasisymmetry error, $\Gamma_c$, $\Gamma_\alpha$, $\epsilon_{\rm eff}$) are ranked in the same table. Gate: the table gives, for each proxy, the Spearman coefficient with its 95% interval from a bootstrap over the configurations, and the direct losses with their Wilson intervals. A proxy is called useful only if its interval excludes zero, and no proxy becomes an objective in package 8 without its row. Each run has 2000 markers on paired keys. With 20 configurations the 95% interval of a rank correlation is about $\pm0.4$, and the tracing costs about three GPU-days on one RTX A4000: the full table is a release-tier campaign, and the scheduled tier holds a smaller version, the three archive configurations with 500 markers.
- [ ] **7.7 Gap bands against resonances.** After 7.2: the overlap of the gap bands with the frequencies at which the fast ions resonate, as an objective of its own. Tests: zero when no resonance falls in a band; equal to an independent count on the analytic torus; smooth in the boundary coefficients. Gate: the tests are in the suite.

Gate of package 7: a proxy is used in optimization only if its rank correlation with the direct loss is stated.

### Package 8. Optimization with VMEX

- [ ] **8.1 Cost of a step (VMEX).** Steps V1, V6 and V8 of the VMEX list, then here; the step does not depend on packages 6 and 7 and belongs to the speed phase of the programme, before 7.5. The optimizer of VMEX already has what the 13.4 s of the row-by-row design path lacked: a forward-mode block Jacobian with one factorization for all columns, reuse of the factorization at the same parameters (reuse at a nearby point is draft pull request 810 and step V9 of the VMEX list), hot restarts and a first-order warm start. What remains, in VMEX: the scalar adjoint builds the block linearization a second time (pass it the bands of the refinement); the fixed-point iteration runs to the full tolerance before a Newton refinement that costs about as much (hand off earlier); the two tests that fail on main. Measured on the workstation with four rows of VMEX's own (aspect ratio, volume, $\iota$ on axis and at the edge) and eight boundary coefficients at ns = 16: 1.94 s warm on the quasi-axisymmetric deck and 2.49 s on the quasi-helical one through `make_problem`, against 7.4 s through `solve_implicit`, which does not restart hot. Gate: one factorization per Jacobian, read from the counters, and the Jacobian of four ALFVEX residual rows equal to re-solved finite differences to $10^{-8}$. The warm time on the idle workstation is reported against 4 s; wall time warns and does not gate. The Jacobian is taken with `jacobian_adjoint_tol=1e-10`, since the defaults of `make_problem` are looser than this gate, and the certifier residual is reported.
- [ ] **8.2 Scenario and resolution ladder.** A scenario fixes species, the energy and pitch distribution, the profiles, the radial source weight, the field and size scales and the mode families before a window is chosen. Objectives are evaluated over a range of density. Coarse settings are used first and fine ones for the last steps, with the same objective definition. Gate: the scenario is a file under version control, written before any objective is evaluated; the difference of the objective and of its gradient between the coarse and the fine setting is reported at the starting point, and the coarse setting is used only while the decrease per step exceeds that difference. Step V25 of the VMEX list is its counterpart.
- [ ] **8.3 Controls.** Replay three published results before the study, each with its own launch measure, boundary and exposure: the wave-off losses of Landreman and Paul (2022, fig. 6: 5000 alphas of 3.5 MeV born at $s=0.25$, minor radius 1.7 m, 5.7 T, traced to $s=1$ or 0.2 s; zero exits in three of four configurations, an upper bound of 0.060% at 95%); the four cases of Bader et al. (2019) with first exit through the penultimate surface; the Wistell-A continuum optimization of Paul et al. (2025) at aspect ratio 6.7, with the reduction of the HAE (3,2) gap and the survival of (4,2) and (2,2). Gate: each control reproduces its published number within the stated statistical bound, or the difference is traced to a named input; the study of 8.4 does not start before. Three pull requests, one per control, and the replay of the wave-off losses comes before 7.5 and 7.6, since it validates the direct loss. In full that replay is about four GPU-hours; the scheduled tier holds a smaller version, 1000 markers over 0.02 s. Its decision rule: at most 3 exits of 5000 in each of the three configurations with none published, the bound that zero observed events allow. Inputs of Bader et al. that are not public are requested (section 16).
  Decided 2026-11-21: the Wistell-A control uses eq. 7.4 of Paul et al. (2025) exactly as published (8.12 item 1), not the $|\varepsilon_1+\varepsilon_2|$ objective. The Wistell-A case is not yet in the repository.
- [ ] **8.4 The four-arm study.** Common constraints, starting boundaries, boundary modes, optimizer budget and step rules. Arm A: equilibrium and wave-off confinement only. Arm B: A plus the geometric cost. Arm C: A plus the spectral objective. Arm D: A plus the stability margin or the direct loss; which of the two is fixed in the protocol. Start from the quasi-axisymmetric case, repeat on a quasi-helical one, hold out a quasi-isodynamic one for spectral validation. Fixed in advance: aspect ratio, field and size, transform range, the quasisymmetry tolerance, $\beta$, $E_r$, the exposure time, the range of $\delta B/B$, the minimum effect that counts. At accepted designs the full band and the losses are recomputed at higher resolution, with paired launches, phases and random increments, on held-out seeds, profiles and amplitudes. Reported for every arm: achieved $\beta$, field, aspect ratio, transform, quasisymmetry error, wave-off loss, the objective, the paired loss change with its interval, the amplitude threshold curve, failed evaluations and cost. Gate: the report is complete for every arm, with the quantities fixed in advance unchanged. Each difference between arms is stated with its statistical error and against the minimum effect fixed in advance; a difference below either is reported as no difference. The study measures; it does not set out to show a reduction. The protocol (the arms, the minimum effect, markers, seeds, the held-out set, the budget of direct-loss evaluations inside the loop) is a file committed and read by the owner before any run. This is a campaign of many pull requests, not one.
- [ ] **8.5 Free boundary and single stage.** After the fixed-boundary study holds: one arm of 8.4 repeated with a free boundary from the coils of the starting configuration, and with the coils as unknowns if the single-stage lane of VMEX is ready. Gate: the same report as 8.4, with the difference from the fixed-boundary result and its error. This is a placeholder for release 3.0: it is written out as steps after 8.4, and it needs coils and `mgrid` input (E2b of the ESSOS list).
- [ ] **8.6 Further controls.** Three benchmark cases specified by the collaborators in their reproduction kit, each run with its acceptance rule or recorded as blocked by the inputs it lacks: the shape gradient of the NCSX equilibrium at finite pressure, as a control of the VMEX geometry derivative before any wave objective; the ideal-ballooning optimization of Gaur et al. (2023), whose inputs are public but whose final `wout` is not; the FAR3d cases for CFQS and QPS as frequency checks, whose equilibria and profiles are not public. Gate: one row per case in the validation page: met, with the numbers, or blocked, with the missing input. One pull request per case, in the repository the case belongs to (the ballooning case belongs to VMEX); a case without inputs becomes an issue for the owner, not a step.

Gate of package 8: the tables of the study are published for every arm, whatever the sign of the result. A statement that an arm lowers the independently computed, energy-weighted loss needs more than the predeclared minimum at equal constraints, on a configuration and on profiles that were not used to tune the objective.

### Package 9. Validation, performance and releases

Runs throughout.

- [ ] **9.1 Tiers.** Section 10.1. Move the coverage gates to the pull-request tier. Without VMEX the fast suite sits at 942 of 988 statements and 137 of 144 branches, a margin of 3 statements and no branch, and every miss is on the VMEX path: the tier therefore keeps one small VMEX solve with its boundary derivative. The 30 slowest tests take 75.5% of the 699 s; they are the first to be shortened by reusing compiled fixtures and reducing the number of distinct shapes. Shards are balanced by measured duration. A check compares the collected test inventory with the union of the job selections, so that no test and no marker is left out of every job. A test marked for GPU that runs on CPU is named so. Measured over the last 50 runs (2026-11-13 to 2026-11-19): a push to main runs eight jobs and takes 46 runner-minutes, a night about two runner-hours; the coverage job runs the whole suite with the slow tests in one process (median 15.6 min, twice cancelled at its 90-minute limit); 82 of 327 jobs waited more than five minutes for a runner. The changes: a push to main runs the pull-request tier on one Python version and the full matrix runs at night; the coverage job runs under `pytest-xdist` without the slow tests; each job on the shared workstation is pinned to its share of the cores (four runners share 36 threads, and every test worker otherwise starts a thread pool over all of them); `cancel-in-progress` applies to pull-request refs only, so that a merge does not cancel the run of the merge before it; every job has a timeout near twice its median; the workflows set `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to 1 (section 12.1), which is the likely cause of the 1280 s median of the coverage job on the workstation against 809 s on the Macs. Gate: the pull-request run has a median under 6 min and a 95th percentile under 10 min of wall time over a week, and no run on main is cancelled by a later one. Counterparts in the rules list: R8, R8b and R9. One change per pull request, with the run time of the changed job measured on the branch as its gate; the weekly median and percentile are a measurement after the merges. The pull-request tier installs the ESSOS extra once packages 6 and 7 add code that needs it, so that coverage counts that code.
- [ ] **9.2 Fixture manifest.** One table in `tests/data/PROVENANCE.md` with, for every external number or file: identifier, hash, source revision, licence, model switches, units, radial label, sector, command and tolerance. `tests/data/PROVENANCE.md` and `tests/test_provenance.py` exist: the table and the pattern of the test are extended together. Test: every row has the new columns filled. This step comes before 6.3, 7.6 and 8.3, which add fixtures.
- [ ] **9.3 Hygiene.** `tests/test_hygiene.py` keeps what is specific to this repository: the old-name check of step 0.2, the check that cited tests and paths exist, and the prose patterns of section 10.4. The example rule and the general text rules are checked by the repository rule checker (rules E and T) and leave this file once that check is required.
- [ ] **9.4 Performance test.** Ratios between kernels in one process, planned memory from `memory_analysis()`, and the number of compilations, for three reference cases: the ITPA torus, the precise quasi-axisymmetric case, the optimization step. The profile of each release is stored with it. Gates: one compilation per repeated call; the planned memory of a gradient within the bound of 12.3. The profiles are stored on the `bench-data` branch (R6 and R7 of the rules list).
- [ ] **9.5 Public stellarator benchmark.** One quasi-axisymmetric and one quasi-isodynamic case with equilibrium, Boozer arrays, density, continuum, global modes with vectors, damping, and the signed power ledger by species and orbit class, published with inputs, outputs and hashes. Gate: from a clean checkout and the published inputs, the published command reproduces every published array to the stated tolerance, and the hashes match. The maintainers prepare the archive; the owner publishes it (corrected 2026-11-21: names who prepares the archive by role). Data above 1 MB is fetched with a hash. It needs packages 4 and 5 and step E10 of the ESSOS list for the orbit classes.
- [ ] **9.6 GPU.** The continuum at $H=162$ and the targeted solve at $N=3200$ on one named GPU, after step 2.10, with equal values and gradients in 64-bit floats. Gate: the targeted solve at (800, 12) at least 5 times faster on the GPU than on one CPU thread at equal residual (today it is slower: 2.77 s on an RTX A4000). If the gate is not met, the ratio is recorded and the GPU path stays opt-in.
- [ ] **9.7 Experiment.** One W7-X discharge with neutral beams and an identified mode (8.11 lists the candidates), with equilibrium, profiles and rotation: frequency, mode numbers, radial location and a synthetic signal predicted with uncertainties before the comparison. Gate: the prediction, with its uncertainties and its inputs, is committed before the measured frequency and mode numbers are looked at; the comparison then reports each quantity as inside or outside its stated uncertainty. Blocked until the data arrive (section 16). It also needs a synthetic magnetic-probe signal, which becomes a step when the data are in hand.

## 14. Order and releases

1. Package 0, then packages 1 and 2: release 2.0. The change of the test matrix in step 0.5 comes first, then steps 0.1 and 0.2 in that order. After 0.2, the steps of package 1 touch separate files and can run side by side: 1.1 (`objectives.py`), 1.2 and 1.3 (VMEX), 1.4 (`objectives.py`, after 1.1), 1.5 (`drive.py`), 1.6 and 1.7 (`tests/data`), 1.8 (SOLVAX), 1.11 (input checks, after 1.5 where it touches `drive.py`), 1.9 (documentation, last, and repeated at the end of package 2), 1.10 (after 2.3 and 2.4). Step 1.4 precedes the gate of 1.3. In package 2, 2.1 comes first, then 2.3, then 2.2; 2.5, 2.6, 2.8, 2.9 and 2.11 are independent of each other.
2. Packages 3 and 4: release 2.1. Step 3.1 precedes 4.1, because damping needs the current term. Steps 4.3 and 4.4 wait for SOLVAX; 4.5 and 4.6 precede 4.4 and the tokamak gate of 4.1. Step 4.8 needs the weights of step 5.1 and is released with package 5.
3. Packages 5 and 6: release 2.2. Steps 5.1 and 6.4 can start as soon as 2.0 is out; 6.1 waits for E8 and E12 of the ESSOS list. Step 5.6 needs 6.1; 6.6 needs 6.5; 7.4 needs 6.6. Steps 6.x and 7.5 need a release of ESSOS with the wave interface, pinned by an optional extra here.
4. Packages 7 and 8: release 2.3.
5. Steps 5.9, 7.4 in full and 8.5: release 3.0.

Package 9 runs throughout. The SOLVAX, VMEX, ESSOS and DKX items of section 5 are opened as issues in those repositories at the start, since packages 4 to 6 wait on them.

## 15. Decisions that need data

| Decision | Measurement that settles it |
|---|---|
| A three-component ideal-MHD operator | The model of 8.4 against AGNI or CAS3D on one equilibrium at 3 to 5% $\beta$. Go ahead only if frequencies differ by more than 5% or an instability is missed. |
| Order of the radial elements | Cost at equal accuracy for $p=1,2,3$ (step 2.5). Known: at equal unknowns $p=2$ costs 1.2 times and $p=4$ 1.6 times. |
| Matrix-free solves | Known: at $N=200$, $H=300$ the limit is memory, 9.3 GB planned and almost all in the assembly, not time (8 to 17 s). Decide after step 2.10: go matrix-free only if the stored pencil, four arrays of $N\times H\times H$, still does not fit. |
| Trapped ions in the resonance sums | Agreement with the orbit-integrated evaluator within 5% on one quasisymmetric and one quasi-isodynamic field. Otherwise the orbit evaluator is used alone for those fields. |
| Continuum damping by contour or by resolvent | Both on the Könies and Kleiber case and on one VMEX equilibrium. The contour needs an analytic continuation of the equilibrium; a spline equilibrium continued by Taylor expansion converged poorly in Bowden et al. (2015). |
| Complex contour modes in ESSOS | Whether a damped mode passed to ESSOS as a real field changes the losses compared with the undamped mode of the same gap. |
| GPU | Step 9.6. |

## 16. Data and literature to request

| Request | Content | Decides |
|---|---|---|
| The ITPA drive packet (authors of Könies et al. 2018) | The numerical curves of figure 1; the complex mode and $\omega$; the field variable and the wave energy; power by species and orbit class; raw, damping and corrected rates at each temperature; the zero-fast-ion baseline; the intercept of the 400 keV density scan; the switches of the CAS3D-K run (reflected ions, orbit width, Larmor radius); the definition of $F_0$ and its radial coordinate | The tolerance of the drive gates of package 5 |
| Stellarator beam cases (Könies et al. 2008; Fehér 2013) | The W7-AS discharge 39042 equilibrium and profiles, and the W7-X 55 keV deck, with their thresholds | Step 5.10 and the reflected-ion gate |
| Continuum damping decks (Könies and Kleiber 2012; Bowden et al. 2015) | Inputs, the $\iota$ profile behind figure 1 of Bowden et al., contour maps, complex eigenfunctions, the normalized frequency tables, the H-1 damping ratio | Steps 4.1 and 4.2 |
| An independent three-dimensional wave case | The precise quasi-axisymmetric case and a held-out quasi-isodynamic or HSX case: WOUT and Boozer files, an independent mass density, signed modes, parity, radial mesh, axis and edge conditions, model switches, vectors | Step 3.2 |
| Sound-coupled references | CONTI spectra for HSX and H-1 with the H-1 equilibrium and the adiabatic index used, neither of which is published; CAS3D global modes with compressibility on one of them | Steps 3.4 and 3.5 |
| An experiment | One W7-X or LHD discharge and time window with equilibrium, profiles and their uncertainties, rotation, the beam phase space, diagnostic geometry and calibration, and the terms of use of the raw signals | Step 9.7 |
| The design case | The starting boundary and its bounds, field and size, current and pressure profiles, species and impurities, the loss surface, the accepted trade in confinement, plausible mode amplitudes | Step 8.4 |
| Saturation reference | A multimode time series from CKA-EUTERPE with damping, sources and wave normalization | Step 6.6 |
| Papers not yet in the reference folder | Slaby et al. 2016 (STAE-K); Slaby, Kleiber and Könies 2017 (collisions); Slaby et al. 2018 (nonlinear saturation); Mishchenko, Könies and Hatzky 2009; Kolesnichenko and Tykhyy (the damping formula of 8.6); Zhang, Sheng and Fu 2026; Todo 2025; Riemann et al. 2025; Unalmis et al. 2026; Hyder, Paul and Knyazev 2026; Knyazev et al. 2026; Fu and Van Dam 1989; Winsor et al. 1968 | Tests that cite them |

Every packet states units, the radial label, the frame, commands, versions, hashes and terms of use. Switches are not inferred from a plotted curve.

## 17. Out of scope

A particle pusher in ALFVEX. Field solves for nonlinear mode structure. The compressional Alfvén branch and ion-cyclotron physics. Photon radiation, impurities and charge-state balance, which stay in NEOPAX. Turbulence-driven excitation of stable modes, which linear stability does not exclude (Riemann et al. 2025) and which is recorded as a limit of every stability objective. Rewrites for hardware before a profile shows the need. A compatibility package under the old name.

## 18. References

Read in full from the reference folder, with page images:

- Antonsen, Paul and Landreman (2019), J. Plasma Phys. 85, 905850207. Shape gradients by adjoints.
- Bader et al. (2019), J. Plasma Phys. 85, 905850508. Energetic-particle losses in optimized equilibria.
- Berk and Breizman (1990), Phys. Fluids B 2, 2226, 2235 and 2246. Saturation of a single mode, parts I to III.
- Berk, Breizman and Pekker (1996), Phys. Rev. Lett. 76, 1256. A driven mode near marginal stability.
- Berk, Mett and Lindberg (1993), Phys. Fluids B 5, 3969. Boundary-layer theory of nonideal TAEs.
- Berk, Van Dam, Guo and Lindberg (1992), Phys. Fluids B 4, 1806. Continuum damping of low-$n$ TAEs.
- Bertram, Blackwell and Hole (2012), arXiv:1202.3204. Low-frequency Alfvén waves in the H-1 heliac.
- Bowden, Hole and Könies (2015), Phys. Plasmas 22, 092114. Continuum damping in tokamak and stellarator equilibria.
- Breizman and Sharapov (2025), Fundam. Plasma Phys. Radiative damping of the TAE at low shear.
- Candy and Rosenbluth (1994), Phys. Plasmas 1, 356. Nonideal theory of TAEs.
- Chen and Zonca (2016), Rev. Mod. Phys. 88, 015008. Alfvén waves and energetic particles.
- Cheng (1991), Phys. Fluids B 3, 2463. Alpha-particle destabilization of the TAE.
- Cheng (1992), Phys. Rep. 211, 1. Kinetic extensions of magnetohydrodynamics.
- Cheng and Chance (1986), Phys. Fluids 29, 3695. Low-$n$ shear Alfvén spectra.
- Fehér (2013), thesis, University of Greifswald. Alfvén waves and fast particles: CKA-EUTERPE.
- Fu, Cheng and Wong (1993), Phys. Fluids B 5, 4040. Stability of the TAE in axisymmetric equilibria.
- Fu et al. (1996), Phys. Plasmas 3, 4036. Alpha-driven TAEs in TFTR.
- Gaur et al. (2023), J. Plasma Phys. 89, 905890518. Adjoint optimization against ballooning modes.
- Gorelenkov and Sharapov (1992), Phys. Scr. 45, 163. Collisional damping of TAEs on trapped electrons.
- Heidbrink (2008), Phys. Plasmas 15, 055501. Basic physics of Alfvén instabilities.
- Kolesnichenko et al. (2001), Phys. Plasmas 8, 491. Continuum and high-frequency eigenmodes in optimized stellarators.
- Kolesnichenko et al. (2002), Phys. Plasmas 9, 517, and Nucl. Fusion 42, 949. Destabilization by circulating ions in Wendelstein-line stellarators.
- Kolesnichenko et al. (2011), Plasma Phys. Control. Fusion 53, 024007. Energetic-ion instabilities in two and three dimensions.
- Könies (2000), Phys. Plasmas 7, 1139. A kinetic MHD energy integral in three dimensions.
- Könies and Eremin (2010), Phys. Plasmas 17, 012107. Coupling of Alfvén and sound waves in stellarators.
- Könies and Kleiber (2012), Phys. Plasmas 19, 122111. Continuum damping in two and three dimensions.
- Könies et al. (2008), AIP Conf. Proc. 1069, 133. From kinetic MHD in stellarators to a fully kinetic description.
- Könies et al. (2018), Nucl. Fusion 58, 126027. The ITPA benchmark of the fast-particle-driven TAE.
- Landreman and Paul (2022), Phys. Rev. Lett. 128, 035001, with its supplement. Precise quasisymmetry.
- Lauber (2013), Phys. Rep. 533, 33. Super-thermal particles in hot plasmas.
- Lauber et al. (2007), J. Comput. Phys. 226, 447. LIGKA.
- Lilley, Breizman and Sharapov (2009), Phys. Rev. Lett. 102, 195003. Dynamical friction near threshold.
- Mett and Mahajan (1992), Phys. Fluids B 4, 2885. Kinetic theory of TAEs.
- Mishchenko, Könies and Hatzky (2011), Phys. Plasmas 18, 012504. Pressure effects on Alfvénic modes.
- Nemov et al. (2008), Phys. Plasmas 15, 052501. Poloidal motion of trapped particles.
- Paul et al. (2025), J. Plasma Phys. 91, E101. The shear Alfvén continuum of quasisymmetric stellarators.
- Slaby et al. (2020), Nucl. Fusion 60, 112004; (2024), 64, 126016; (2025), 65, 036016. Mode activity in W7-X, the multi-mode model, the transport model.
- Spong (2015), Phys. Plasmas 22, 055602. Three-dimensional toroidal physics.
- Spong, D'Azevedo and Todo (2010), Phys. Plasmas 17, 022106. AE3D.
- Spong, Sanchez and Weller (2003), Phys. Plasmas 10, 3217. STELLGAP.
- Varela et al. (2024), Front. Phys. 12, 1422411. FAR3d.
- White and Chance (1984), Phys. Fluids 27, 2455. Hamiltonian guiding-centre orbits.
- Zonca, Chen and Santoro (1996), Plasma Phys. Control. Fusion 38, 2011. Kinetic theory of low-frequency Alfvén modes.

Cited through the reviews, not yet in the folder (section 16): Fu and Van Dam (1989), Phys. Fluids B 1, 1949; Hyder, Paul and Knyazev (2026), arXiv:2611.07182; Knyazev et al. (2026), J. Plasma Phys., doi:10.1017/S0022377826101974; Kolesnichenko and Tykhyy (2018), Plasma Phys. Control. Fusion 60, 125004; Mishchenko, Könies and Hatzky (2009); Riemann et al. (2025), Phys. Rev. Lett. 134, 025103; Slaby et al. (2016); Todo (2025), Nucl. Fusion; Unalmis et al. (2026), J. Plasma Phys., doi:10.1017/S0022377826101652; Winsor, Johnson and Dawson (1968), Phys. Fluids 11, 2448; Zhang, Sheng and Fu (2026), J. Plasma Phys., doi:10.1017/S0022377826102219.

Codes: STELLGAP and AE3D (ORNL-Fusion, MIT), FIRM3D (MIT), MEGA v1.1.0 (MIT-style, citation required), FAR3d, ASCOT5, AGNI, DESC, CAS3D and CAS3D-K, CONTI, CKA-EUTERPE, LIGKA, NOVA-K (the last six through publications only).
