# Validation

Every physics claim in FAX is a test in `tests/`, tied to an analytic result
or a published number. CI runs them on Python 3.10–3.13 in float64. The numbers
below are the test thresholds and the measured values of the tests named here.

**Scope.** FAX is a reduced model: restricted finite-$\beta$ terms (experimental), no sound coupling or BAE, no continuum or
electron damping, no trapped particles, finite orbit width or finite Larmor radius in the drive, which is zero-orbit-width,
passing-only and perturbative (valid to about 200 keV in the Könies case). Every gate below is a statement inside that scope;
"Not done" at the end lists the rest. Derivatives are first order ({doc}`api`).

## Summary

| Gate | Test | Result |
|---|---|---|
| Uniform-field continuum, $\omega = \lvert n - m\iota\rvert v_A/R$ | `test_uniform_field_continuum_is_k_parallel_v_alfven` | exact to 1e-9 |
| Angle quadrature converged, $K$, $M$ Hermitian | `test_matrices_are_hermitian_and_resolution_converged` | 32² vs 64² to 1e-9 |
| STELLGAP CTOK continuum | `test_stellgap_ctok_parity` | mean 6.3e-3, p95 6.6e-3 (gate 7e-3) |
| Gap formulas | `test_gaps_and_gap_location` | TAE $\iota = 2n/(2m+1)$, MAE $\iota = (2n - N_{fp})/2m$ |
| Uniform-field global spectrum is a continuum | `test_uniform_field_spectrum_is_a_continuum` | edges within $0.5/N$; mode width ∝ $1/N$ (N = 40, 160) |
| AE3D CTOK global modes | `test_ae3d_ctok_parity` | 6 modes within 1.4%, overlap > 0.9 |
| AE3D CTOK TAE (337.9 kHz) | `test_ae3d_v31_blocks_and_energy_on_identical_inputs` | not reproduced: three $\iota'$ terms of AE3D v31's bending blocks differ from the energy; pinned block comparison below |
| Independent 3-D spectrum, precise QA of Knyazev et al. 2026 (STELLGAP continuum, AE3D gap band) | `test_stellgap_continuum_parity`, `test_ae3d_gap_band_in_the_sine_basis` | continuum 3.0e-3 mean (gate 4e-3); archived global mode within 2% in energy at 24-64 harmonics, 5.1% at AE3D's 162; no single-mode match (overlap 0.18) |
| Exact sheared 3-D manufactured solution (benchmark of uwplasma/fax#9) | `test_manufactured_solution_and_regular_axis_interpolation` | nodal error 9e-3, 3.1e-3, 8.9e-4 at N = 16, 32, 64 (→ second order) |
| Bending and inertia vs real-space quadrature | `test_ctok_bending_energy_matches_real_space_quadrature` | 0.3% (gate 0.5%) |
| Conventions ({ref}`conventions`), embedding Jacobian vs the Boozer identity, force balance | `test_conventions_and_geometry_diagnostics_on_ctok`, `test_konies_itpa_finite_beta_tae` | CTOK $\psi'$ = 0.04509 to 0.4% and angular spread $\le$ 5e-5 for $s \ge 0.05$ (first surface 15% low); force-balance $B_s$ vs embedding $\le$ 1e-5 of $\lvert B\rvert$; ITPA ($\psi' = 3\pi/2\pi$): 0.3% for $s \ge 0.1$ (the first four surfaces up to 4.5% off), spread $\le$ 3e-4, force balance $\le$ 8e-5, $\mu_0p'$ vs the input's `AM` to 0.15% |
| Forms of the kink term ({ref}`finite-beta`) | `test_kink_forms_differ_by_the_curl_b_term` | $K_\mathrm{direct} = K_{\nabla\sigma} - \int u\sigma(\nabla\times\mathbf b)\cdot\nabla\tilde\phi^*$ and $K_\mathrm{full} = K_{\nabla\sigma} - \int\sigma^2u\tilde u^*$ to 2e-6 (random periodic field); counterexample $-1/4$, 0, $-1/4$ |
| Finite-$\beta$ terms vs real-space quadrature ($\sigma$ from curl $\mathbf B$; CTOK and Könies with $p$) | `test_finite_beta_terms_match_real_space_quadrature` | 0.3% CTOK, 0.2% pressure (gate 0.5%) |
| Cylinder: Newcomb $m = 1$ cancellation; Suydam threshold | `test_cylinder_newcomb_m1_and_suydam_threshold` | bending + kink = 1e-3 of bending; stable at 1.0e4 Pa, unstable at 2.0e4 (Suydam 1.05e4) |
| Könies ITPA with $p(s)$, vmex equilibrium | `test_konies_itpa_finite_beta_tae` | 4.127e5 → 3.947e5 rad/s with finite-$\beta$ terms (CAS3D 4.01e5) |
| Könies 2018 ITPA TAE frequency | `test_konies_tae_frequency_and_convergence` | 4.0610e5 rad/s, $N_s$-converged to 1e-4 |
| Shift-invert (SOLVAX) vs dense | `test_shift_invert_matches_dense_and_gradients_match_finite_differences` | 1e-10 |
| Eigenvalue gradients vs finite differences | same | 1e-5 (cluster mean and shift-invert) |
| Fu–Van Dam Maxwellian scaling and closed-form ZOW coefficient (weak toroidicity), flat and with a density gradient (the gradient term carries $j$ = 11, 10) | `test_fu_van_dam_maxwellian_scaling` | 1e-3 |
| Slowing-down $\int F\,d^3v = 1$ with the smoothed edge; duplicated toroidal samples | `test_slowing_down_is_normalized_with_its_smoothed_edge`, `test_duplicated_toroidal_samples_leave_the_drive_unchanged` | 1e-6; 1e-10 |
| Elliptic two-mode gap, full width $2f_0\lvert c\rvert$ | `test_elliptic_two_mode_gap_has_full_width_two_f0_c` | 1e-6 |
| Drive vs Könies 2018 (Boozer form of the circular model) | `test_koenies_2018_tae_growth_rates` | 4.29, 16.39, 32.72, 64.48 ×10³ s⁻¹ (pinned 1e-3); old circular model within 4% |
| 3-D structure: helical relabelling of an axisymmetric torus (with and without $I$), axisymmetric limit of a 3-period description | `test_quasi_helical_relabelling_leaves_the_drive_unchanged`, `test_axisymmetric_limit_ignores_toroidal_sidebands` | 1e-4 (4e-5 measured, FFT wrap); 1e-9; the previous $n_h/\iota$ form is 17–20% off |
| Negative $\iota$ ($\theta$-mirror), $n$-reversal ($\zeta$-mirror, co ↔ counter), amplitude and global phase, density scaling, passivity of a uniform distribution | `test_negative_iota_and_n_reversal_describe_the_same_device`, `test_amplitude_phase_density_and_passivity` | 1e-12; $\gamma < 0$ at every $\omega$ |
| Regularizations: `boundary`, `nodes`, `FastIons.birth_width`, `guard` | `test_passing_boundary_and_node_regularizations_converge`, `test_birth_edge_smoothing_converges_quadratically`, `test_small_divisor_guard_converges_and_is_sensitive_near_rational_surfaces` | see the table below |
| ITPA rung 2: passing-only drive vs the CAS3D-K (ZOW) curve | `test_zow_curve_is_bracketed_by_the_passing_only_drive` | ratio 0.58–0.60, 82–191 keV; unmatched content, not a validation |
| Drive sign rules | `test_drive_sign_and_flat_profile` | flat, peaked, hollow, $\pm n$ |
| Drive gradients vs finite differences | `test_gradients_match_finite_differences`, `test_geometry_and_mode_gradients` | 1e-5 (density, $T_f$, $\omega$, every geometry leaf and the mode) |
| CTOK global TAE driven | `test_ctok_global_tae_drive` | 272.8 kHz mode, $\gamma/\omega$ = 7.0e-4, 3.3e-3 at 25, 50 keV (9.2e-4, 3.6e-3 with the $n/\iota$ form) |
| LP-QA drive: helical sidebands, gradient through vmex geometry | `test_stellarator_drive_gradient_and_sidebands` | $k \ne n$ resonances present; FAX chain AD vs FD ≤ 1e-4 |
| vmex → booz_xform_jax → FAX continuum | `test_vmec_input_to_continuum` | runs, finite |
| Compression harmonics of an elliptic torus, $\epsilon_{\pm2,0} = (\kappa^2-1)/2(\kappa^2+1)$ | `test_continuum_cost_of_an_elliptic_torus` | exact to 1e-12; $f_\mathrm{cont}$ analytic; AD vs FD 1e-7 |
| $f_\mathrm{cont}$ descent step through vmex | `test_continuum_cost_gradient_step_and_vmex_terms` | decreases; vmex aspect on the same state |
| Boundary gradient and FE radial refinement | `test_boundary_gradient_and_radial_refinement` | AD vs FD 1e-4 at 16, 32, 64 nodes; first-order convergence |
| Mode classification: CTOK TAE gap modes isolated, modes at $s > 0.6$ continuum-like; uniform field all continuum-like | `test_classification_separates_gap_modes_from_the_continuum_on_ctok`, `test_uniform_field_modes_are_all_continuum_like` | gap modes margin > 0.1 at $s = 0.45$; the rest < 0.03; uniform field < 0.01 |
| Smooth `gap_closure` and `continuum_cost` | `test_gap_closure_is_smooth_where_the_widest_gap_changes`, `test_continuum_cost_is_continuous_across_the_helicity_cut` | gradient jump of the hard forms removed; AD vs FD on both sides 1e-5 |
| Status-gated frequencies | `test_selected_frequencies_carry_status_and_unconverged_solves_are_rejected` | NaN value and tangent if unconverged or tied |
| Fixed-geometry derivative, asymmetric tangent, step sweep | `test_fixed_geometry_derivative_has_a_step_size_plateau` | $h^2$ then floor below 1e-6 over four decades-steps (about 1e-9 at best), dense and shift-invert |
| Re-solved equilibrium derivative, gauge floor | `test_resolved_equilibrium_derivative_has_a_gauge_floor`, `test_m1_gauge_floor_and_stellarator_asymmetric_direction` | flat remainder 4.6e-4 (`zs01`); see Gradients |
| Resolution budget of the gap-mode objective | `test_resolution_budget_of_the_gap_mode_frequency_and_gradient` (scheduled) | table in Gradients |

## Continuum

**Uniform field.** {func}`fax.circular_torus` with $\epsilon_B = 0$ and
$\iota$ from 0.3 to 0.6, harmonics $n = 1, 2$, $m = 0..4$: every branch equals
$|n - m\iota|\,v_A/R_0$ to 1e-9.

**STELLGAP CTOK.** The public STELLGAP test problem (circular tokamak,
`boozmn_ctok.nc` with $n_s = 101$, $n = 2$, $m = 0..12$, deuterium at
10¹⁹ m⁻³), the STELLGAP output vendored in `tests/data`. Relative error of the
13 branches per surface on 97 surfaces: mean 6.3e-3, 95th percentile 6.6e-3.
STELLGAP labels its full-mesh surface $k$ as $k/(n_s - 1)$ instead of
$(k - 1)/(n_s - 1)$ (`rho = nsurf/nsurf_max` in `stellgap_serial_ver5.f`), so
the test shifts its labels by one cell; a scan of the offset has its minimum
there. Timings of the eigensolvers, at equal accuracy, are in the {ref}`performance <performance>` section.

(performance)=
### Performance at equal accuracy

`python examples/benchmark.py` solves one assembled pencil $F y = \lambda G y$ three ways and checks every answer
before comparing speed: `targeted` is `fax.eigenmodes(target_hz, count)` (SOLVAX shift-invert, `tol = 1e-10`),
`dense-jax` is `fax.eigenmodes()` (FAX's jitted dense generalized solve of all $NH$ modes) and `dense-scipy` is
`scipy.linalg.eigh(F, G)`, the reference. Each (case, method) runs in a fresh process: cold is compile plus first call,
warm is the median of five `block_until_ready` calls (two for the slow dense rows), forward includes assembly, and the
gradient is `jax.grad` of `frequency_hz` (mean of the requested modes) with respect to every geometry array. $\eta$ is
the largest $\|Fy-\lambda Gy\|/((\|F\|+|\lambda|\|G\|)\|y\|)$ of the requested modes (infinity norm of the matrices),
recomputed from the returned vectors; err is the largest relative eigenvalue difference from `dense-scipy`. Speed-ups
are quoted only where both methods reach $\eta \le 10^{-10}$.

**Conditions.** Apple M4 (10 cores, 24 GB), macOS 26.6, CPU only, float64, Python 3.12.13, FAX 0.1.0 (`a6364cc`),
SOLVAX 0.32.1, JAX and jaxlib 0.12.0.1, NumPy 2.5.4, SciPy 1.18.1. BLAS/LAPACK pinned to one thread; XLA's own
dots are not (the assembly used about two cores, the Knyazev assembly about five; `eigh` and the shift-invert are single-thread).
The 1-minute load before each of the 27 runs was 2.8-5.4 (median 3.8, 26 below 4; the Könies $N=400$ `targeted` run
had 5.4). Peak RSS is above a 267 MB post-import baseline and includes compilation. The office GPUs were busy
(load 41, both at 100%), so there is no GPU column.

| case ($N$, $H$, modes) | method | $\eta$ | err | fwd cold (s) | fwd warm (s) | grad cold (s) | grad warm (s) | peak RSS fwd / grad (MB) |
|---|---|---|---|---|---|---|---|---|
| CTOK $n=2$ (98, 13, 6) | `dense-scipy` | 1.6e-16 | 0 | 1.64 | 1.14 | | | 436 / |
| | `dense-jax` | 1.8e-16 | 1.2e-14 | 1.90 | 1.35 | 2.58 | 1.16 | 493 / 717 |
| | `targeted` | 4.2e-11 | 1.9e-14 | 1.01 | 0.048 | 2.11 | 0.081 | 329 / 627 |
| Könies (200, 12, 3) | `dense-scipy` | 4.1e-16 | 0 | 8.76 | 8.29 | | | 1012 / |
| | `dense-jax` | 4.2e-16 | 1.7e-14 | 10.2 | 9.75 | 9.63 | 8.46 | 1230 / 1444 |
| | `targeted` | 1.2e-11 | 8.8e-14 | 1.44 | 0.102 | 2.69 | 0.155 | 385 / 725 |
| Könies (400, 12, 3) | `dense-scipy` | 4.9e-16 | 0 | 78.2 | 73.9 | | | 2908 / |
| | `dense-jax` | 5.3e-16 | 3.0e-14 | 87.0 | 89.4 | 83.7 | 92.0 | 4190 / 4190 |
| | `targeted` | 7.0e-11 | 3.3e-13 | 1.56 | 0.264 | 2.83 | 0.367 | 518 / 1090 |
| Knyazev QA (49, 64, 6) | `dense-scipy` | 3.1e-16 | 0 | 24.2 | 24.1 | | | 6095 / |
| | `dense-jax` | 3.1e-16 | 2.7e-14 | 23.6 | 23.9 | 26.1 | 24.6 | 6877 / 11393 |
| | `targeted` | 4.3e-12 | 2.2e-14 | 2.48 | 1.38 | 7.86 | 4.09 | 5039 / 11112 |

Angle grids: $32 \times 4$ (CTOK, Könies) and $64 \times 48$ (Knyazev: the 64 AE3D harmonics of largest band energy, kept
complex, $\iota$ and $B$ of the precise-QA fixture, flat $\rho = 1.88\times10^{-6}$ kg/m³, target 30.3 kHz). Warm
forward of `targeted` at equal accuracy against `dense-scipy` / `dense-jax`: 24x / 28x (CTOK), 81x / 96x (Könies 200),
280x / 339x (Könies 400), 17x / 17x (Knyazev); warm gradient against `dense-jax`: 14x, 55x, 250x, 6.0x. The gradients of
`targeted` and `dense-jax` agree to a relative L2 difference of 8.6e-9, 3.9e-8, 2.2e-7 and 1.1e-8 over all geometry
arrays. The Knyazev dense spectrum is a continuum band, so six modes is only a timing case, not a mode identification
(see the Knyazev paragraph above). Dense cost grows as $(NH)^3$ (9.2x for $N$ doubled, $\eta$ and err unchanged);
the dense warm time is the cost of every call, including the first derivative.

**Scaling** (`targeted`, from Könies $N=200$, $H=12$, 3 modes; one of $N$, $H$, modes changed at a time; the 1000-iteration
cap is raised from SOLVAX's default of 100 so that every point can reach $\eta$; factor is one block-Thomas factorization of
$F-\sigma G$, solve is the whole `generalized_eigenpairs` call on assembled blocks, and the two MB columns are the factor's
output size and XLA's static temporary buffers for the solve):

| varied | $N$ | $H$ | modes | $\eta$ | iterations | factor (ms) | solve (s) | fwd warm (s) | grad warm (s) | factor MB | solve MB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| $N$ | 200 | 12 | 3 | 1.2e-11 | 17 | 1.9 | 0.060 | 0.107 | 0.161 | 1.39 | 10.3 |
| | 400 | 12 | 3 | 7.0e-11 | 33 | 1.7 | 0.167 | 0.258 | 0.367 | 2.78 | 20.6 |
| | 800 | 12 | 3 | 9.0e-11 | 112 | 3.1 | 1.08 | 1.19 | 1.36 | 5.57 | 41.2 |
| | 1600 | 12 | 3 | 9.9e-11 | 310 | 6.1 | 5.97 | 5.82 | 5.97 | 11.1 | 82.4 |
| | 3200 | 12 | 3 | **2.8e-10** | 1000 (cap) | 22.5 | 40.1 | 45.8 | 46.0 | 22.3 | 165 |
| $H$ | 200 | 12 | 3 | 1.2e-11 | 17 | 0.63 | 0.043 | 0.087 | 0.140 | 1.39 | 10.3 |
| | 200 | 16 | 3 | 3.0e-11 | 17 | 1.0 | 0.058 | 0.187 | 0.369 | 2.47 | 16.4 |
| | 200 | 24 | 3 | 1.6e-11 | 15 | 3.4 | 0.106 | 0.221 | 0.464 | 5.55 | 32.5 |
| | 200 | 32 | 3 | 5.6e-11 | 13 | 6.3 | 0.135 | 0.348 | 0.785 | 9.86 | 54.0 |
| | 200 | 48 | 3 | 2.7e-11 | 13 | 13.9 | 0.274 | 0.690 | 1.66 | 22.2 | 111 |
| modes | 200 | 12 | 1 | 1.2e-17 | 22 | 0.63 | 0.038 | 0.085 | 0.148 | 1.39 | 9.1 |
| | 200 | 12 | 3 | 1.2e-11 | 17 | 0.62 | 0.043 | 0.088 | 0.135 | 1.39 | 10.3 |
| | 200 | 12 | 6 | 1.4e-16 | 55 | 0.62 | 0.170 | 0.218 | 0.288 | 1.39 | 12.4 |
| | 200 | 12 | 12 | 9.7e-12 | 28 | 0.63 | 0.164 | 0.206 | 0.268 | 1.39 | 16.1 |
| | 200 | 12 | 24 | 1.8e-13 | 44 | 0.63 | 0.557 | 0.603 | 0.664 | 1.39 | 24.4 |

Observed log-log exponents (points with $\eta \le 10^{-10}$ only; the $N=3200$ row is excluded). Expected: factorization
work $O(N H^3)$, factor storage $O(N H^2)$.

| varied | factor time | factor MB | solve MB | solve time | forward | gradient |
|---|---|---|---|---|---|---|
| $N$ (expected 1) | 0.59 (0.9-1.2 above $N = 400$) | 1.00 | 1.00 | 2.26 | 1.95 | 1.75 |
| $H$ (expected 3 and 2) | 2.32 (2.0 from 24 to 48) | 2.00 | 1.72 | 1.32 | 1.36 | 1.63 |
| modes (expected 0) | 0.00 | 0.00 | 0.30 | 0.85 | 0.60 | 0.46 |

Storage follows the model exactly: the factors are $N H^2$ complex numbers (1.00 in $N$, 2.00 in $H$). The factorization
time is not yet in its asymptotic regime: at 1-14 ms it is dominated by the sequential scan over the $N$ blocks and small-block
overhead, and from $H = 24$ to $48$ it grows as $H^{2.0}$, not $H^3$. Two things the model does not capture dominate the
solve: it is iteration-bound, not factorization-bound (the factor is at most 3% of the solve time at $H = 12$), and the iteration
count at fixed $\eta$ is not constant. For the Könies TAE it rises with $N$ (17, 33, 112, 310, more than 1000 iterations at
$N$ = 200 ... 3200) as the continuum accumulates around the gap, so the solve time grows faster than linearly in $N$ (about
$N^{2.3}$) while the dense cost $(NH)^3$ stays 100x-1000x larger at these sizes; it also varies non-monotonically with the
number of modes (22, 17, 55, 28, 44). The $N = 3200$ solve stopped at the iteration cap with $\eta = 2.8\times10^{-10}$ and
is not a valid equal-accuracy point. SOLVAX's dense pivoted fallback is compiled in only for $NH \le 2048$ (`dense_limit`),
which is why the CTOK static buffers (84 MB) are larger than the factor alone would need and why the sweeps start above that size.

**Replaced claims.** The previous table compared a fixed 40 iterations at `tol = 1e-6` with `generalized_eigenpairs` at
`tol = 1e-10`, so its two columns were not at equal accuracy, and the dense timing in `benchmark.py` was `eigvalsh(F)`,
a standard eigenproblem with no eigenvectors rather than $F y = \lambda G y$. Both are removed, as are the unmeasured
"STELLGAP 0.8 s" and "dense 0.9 s" statements; STELLGAP was not re-timed, and a continuum scan is a different computation from a
targeted global solve. The machine is shared: the identical configuration ($N=200$, $H=12$, 3 modes) gave solve times of 0.060, 0.043 and
0.043 s in three runs of one benchmark, and a complete earlier run changed the fast warm times by up to 15%, so
differences below about 40% on the 40 ms rows are not significant and the exponents carry that uncertainty.

## Global operator

**Continuum accumulation.** For a uniform field with sheared $\iota$ and a
single harmonic, the global spectrum fills
$[\omega_A^2(0), \omega_A^2(1)]$ with edges within $0.5/N$, and each mode's
$\phi'$ jump is localized to a fixed number of cells, so its width shrinks as
$1/N$ ($N = 40$ and $160$). No discrete mode appears outside the continuum.

**AE3D CTOK.** The same `boozmn` file averaged to AE3D's 99-point full mesh
with AE3D's edge extrapolation, $n = 2$, $m = 0..12$. The matrix leading
blocks and the operator continuum agree within 1%. Six global modes, matched
to AE3D's eigenvectors by overlap > 0.9, agree within 1.4%:

AE3D's 343.6, 384.1, 685.6, 703.3, 738.2 and 741.2 kHz modes are all
reproduced within 1.4% (gate 1.5%).

**Independent operator and the QA archive (uwplasma/fax#9).** E. Lascas Neto
built a separate physical-potential assembly of the same weak form (Gauss
quadrature at sampled geometry, Knyazev et al. 2026). On CTOK it agrees with
{func}`fax.global_matrices` to within 5.6e-4 on every AE3D-matched mode, with
identical mode counts, while FAX assembles 2.4× faster (0.15 s vs 0.35 s at
$N = 99$, $H = 13$). On the precise-QA case of Knyazev et al. (archive DOI
10.5281/zenodo.21299908) the archived AE3D mode is at 30.31 kHz: public AE3D
formulas reproduce it (30.30 kHz, profile error 4%), while the physical index
assignment gives 27.57 kHz. This is the same AE3D shear-index discrepancy as
the pinned CTOK block comparison below, found independently. Its manufactured-solution benchmark is
ported to `test_manufactured_solution_and_regular_axis_interpolation`, and its
regular axis extension $(s/s_0)^{m/2}$ to {meth}`fax.geometry.Geometry.interpolate`.

**The CTOK TAE: pinned block comparison.** AE3D's TAE at 337.9 kHz is not
reproduced by FAX. Both operators were fed the same metric input and compared
block by block.

- *Build.* AE3D v31 (`ae_solve_version31.f`, ORNL-Fusion/AE3D commit
  `383d327c53bf722026d7e432797ea37c50347b68`), gfortran 15.2.0 (Homebrew GCC 15.2.0_1) on arm64 macOS,
  `-O2 -ffixed-form -ffixed-line-length-none -fallow-argument-mismatch`, LAPACK from Accelerate (`dggev`). Input: the
  CTOK `ae_metric.dat`, `fourier.dat`, `plasma.dat` of `ae3d_test_problem.tar.gz` ($n_s = 99$, 80 × 80 grid, $n = 2$,
  $m = 0..12$, deuterium at 10¹⁹ m⁻³, $\rho = k/(n_s + 1)$). The unmodified build reproduces the shipped `a_matrix.dat`,
  `b_matrix.dat` (identical text) and every eigenvalue of `egn_values.dat` of
  `ae3d_test_problem_answers.tar.gz`. The instrumented build
  (`tests/data/ae3d_v31_instrumentation.patch`) zero-initialises `bb1`, `bb2`, `cc`, `cc2` (v31 allocates them without
  initialising; the result is unchanged), writes the coefficient blocks and matrices in double precision and switches the
  shear, current and pressure terms from the environment. `tests/data/ae3d_v31_ctok.npz` keeps the blocks of surfaces 45
  and 46 ($s = 0.45$, $0.46$) and the assembled matrix.
- *Inertia and base bending agree.* With FAX fed AE3D's own input, the inertia blocks $a$, $b$, $d$ agree to 1e-9 (AE3D's
  $b(i, j)$ is the coefficient of $f_if_j'$, FAX's $B^T$), and the $f'f'$ and $ff$ bending blocks without shear to 2e-6 and
  5e-7. The symmetric part of the $\iota'$ $f'f$ block agrees to 3e-6.
- *Three $\iota'$ terms differ* (AE3D `ae_solve_version31.f` line numbers):

  | Term | AE3D v31 | Energy functional (FAX, quadrature) |
  |---|---|---|
  | antisymmetric $\iota'$ $f'f$ block, `stell_rfp_specific_term2` (912–913, `bb2`, 935) | $-g^{ss}a^2\iota'\,m_j\kappa_i$: the transposed index pattern of the $f_if_j'$ coefficient that `element` (1430–1436) assumes, so the antisymmetric part has the opposite sign (equal magnitude, 3e-5) | $-g^{ss}a^2\iota'\,m_i\kappa_j$ |
  | $\iota'$ $ff$ block, `term3` (914–917, `dd1`, 947) | $a'^2\kappa_i\kappa_j + a^2\iota'^2m_im_j$, and $-g^{ss}aa'\iota'(\kappa_im_j - \kappa_jm_i)$ is antisymmetric, so `dd = dd1 + dd2 + dd2^T` and the final symmetrization drop it | adds $g^{ss}aa'\iota'(\kappa_im_j + \kappa_jm_i)$ |
  | $\iota'g^{s\theta}$ $ff$ terms, `term4`, `term5` (918–921, `dd2`) | opposite sign (least squares over the 99 surfaces: the AE3D increment is $\iota'^2g^{ss} - \iota' g^{s\theta}$ to 5e-6) | $+\iota'g^{s\theta}$ |

  The $\iota'g^{s\zeta}$ terms (`term6`, `term7`, lines 922–925) are computed but commented out of `dd2` (967–968); they are
  0.1% of the $\iota'$ $ff$ block for CTOK and not examined further. Here $a = B/(G + \iota I)$ and $\kappa = n - \iota m$.
- *Quadrature.* Using AE3D's own metric input, a real-space quadrature of $\int|\nabla u|^2dV$ ($u$ from $B/(G + \iota I)$,
  $\iota$, spectral angle derivatives and second-order radial differences, no FAX code) for five smooth potentials
  ($m = 2..5$, $n = 2$, radial profiles $(s(1-s))^4$ times Chebyshev polynomials) differs from FAX's assembled $F$ by
  0.13–0.30% and from AE3D's assembled matrix by 1.8–37%.
- *The TAE.* The Rayleigh quotient of AE3D's 338.2 kHz eigenvector (no current term) is 338.2 kHz in AE3D's blocks and
  324.2 kHz in FAX's blocks under AE3D's assembly (326.9 kHz under FAX's). Correcting the three terms in AE3D's own Fortran
  (switches `1`, `2`, `3` of the instrumented build, assembled by AE3D's own routine) removes the localized
  ($s \approx 0.46$) gap modes at 240.9, 244.9, 338.2 and 355.6 kHz and gives them at 258.9, 264.9 and 266.4 kHz; FAX's blocks
  assembled by AE3D's `element` give 258.8, 264.7 and 266.3 kHz. Alone, the index swap (`1`) gives localized modes at 298.8, 309.4 and 324.0 kHz, the $g^{s\theta}$ sign (`3`) at 298.2 and 303.3 kHz, and the restored $a'\iota'$ cross term (`2`) leaves them at 241.4, 245.2, 338.1 and 355.4 kHz (a negligible term here). With FAX's own assembly the localized gap modes
  are 268.5, 270.1 and 279.1 kHz. The 3–4% left between the two assemblies is AE3D's
  `element`: it uses the node value on the diagonal and `rho_int_1drv_a = rho_int_1drv_b = 0`, so the symmetric part of the
  $f'f$ coupling contributes nothing, although $\int b\,(f_if_j)'\,ds = -\int b'f_if_j\,ds$ does not vanish; with the
  corrected blocks it still errs by 4–17% on the smooth potentials above, and by 0.13–0.7% under FAX's element-averaged
  assembly.

The cause of the CTOK TAE discrepancy is therefore established for the bending blocks: the three $\iota'$ terms above (and,
at the 3–4% level, AE3D's treatment of the symmetric $f'f$ coupling). It does not follow that AE3D's published results
change: the Könies case is insensitive to $\iota'$ (4.061e5 vs 4.067e5 rad/s without it). This has not been reported upstream.

**Könies 2018 ITPA TAE.** {func}`fax.circular_torus` with $R_0 = 10$ m,
$a = 1$ m, $B_0 = 3$ T, $q = 1.71 + 0.16s$, $\epsilon_B = a/R_0$, hydrogen at
2·10¹⁹ m⁻³, $n = 6$, $m = 5..16$:

| $N_s$ | 50 | 100 | 200 | 400 |
|---|---|---|---|---|
| $\omega$ (10⁵ rad/s) | 4.0598 | 4.0607 | 4.0610 | 4.0610 |

against CAS3D 4.01e5 (+1.3%, shifted equilibrium) and GYGLES 4.13e5 (−1.7%,
ad hoc equilibrium as here). The test requires $4.01 < \omega/10^5 < 4.13$.

**Finite-$\beta$ terms (restricted, experimental; no sound or full-MHD claim).** The $J_\parallel$ term is tested in a cylinder
($\epsilon_B = 0$, $q = 0.8 + s$): for an $m = n = 1$ rigid shift inside
$q = 1$ it cancels the bending energy to 1e-3, Newcomb's $(m^2 - 1)$
marginality. This also fixes its sign; the opposite sign doubles the energy.
The pressure term reproduces the Suydam threshold for $m/n = 11/6$ with
$p = p_0(1 - s)$: $p_0 = 1.05\cdot10^4$ Pa from
$rB^2(q'/q)^2/8\mu_0 + p' = 0$. At $1.0\cdot10^4$ Pa the lowest
$\omega^2$ falls as $1/N^2$ (a marginal continuum), and at $2\cdot10^4$ Pa it is
a converged instability. Suydam growth is exponentially small near
threshold, so the test brackets it rather than locating it.

Könies ITPA, $p(s) = 7.17\cdot10^3 - 6.811\cdot10^3 s - 358.5 s^2$ Pa
(Könies et al. 2018), TAE in rad/s:

| equilibrium | no finite-$\beta$ terms | $J_\parallel$ and pressure |
|---|---|---|
| `circular_torus`, `current=True`, $N_s = 400$ | 4.0610e5 | 4.0637e5 ($J_\parallel$ only, +0.07%) |
| `circular_torus` with $p(s)$ | 4.0601e5 | Mercier-unstable (see below) |
| vmex, $p = 0$ (`PRES_SCALE = 0`), $N_s = 400$ | 4.0958e5 | 4.0978e5 (+0.05%) |
| vmex with $p(s)$, $N_s = 100$–400 | 4.1271e5 | 3.9482e5 (−4.3%, overlap 0.98) |

CAS3D, which is compressible and uses the shifted finite-$\beta$
equilibrium, gives 4.01e5. FAX with both terms is 1.5% below it, and 2.9% above
it without them. The remaining gap is plausibly the sound coupling
($\gamma p|\nabla\cdot\boldsymbol\xi|^2$), which the reduced model drops
and which raises the frequency. In the ad hoc equilibrium, which has no
Shafranov shift, the pressure term gives a converged localized instability
($\omega^2 = -3.1\cdot10^9$ s⁻²) at $q = 11/6$, below the cylindrical
Suydam threshold. That equilibrium lacks the $O(\epsilon^2)$ terms behind
the $(1 - q^2)$ Mercier stabilization. On the vmex equilibrium the
negative eigenvalues fall as $1/N^2$. Use a real equilibrium with
`finite_beta=True`.

On CTOK ($q_0 \approx 0.996$, $\sigma \approx -0.7$ m⁻¹), the $J_\parallel$ term
is large. It gives a converged near-axis $m = n = 2$ instability
($\omega^2 \approx -2.4\cdot10^{10}$ s⁻²), and it moves the AE3D-matched
modes by 0.1–10% (388 → 428 kHz; overlaps with AE3D fall to 0.72–0.87).
`test_ae3d_ctok_parity` keeps the default `finite_beta=False`.

AE3D's shipped CTOK run already includes its $J_\parallel$ term (`jprl_coef` is nonzero in `ae_metric.dat`). It has five
unstable near-axis modes (lowest $\omega^2 = -1.9\cdot10^{10}$ s⁻², $-480$ kHz²; one, $-15$ kHz², with the term off), like FAX's, but it moves the
modes between 100 and 800 kHz by at most 0.39% (median 0.02%; those near 388 kHz by at most 0.06%), whereas FAX's moves them by up to
10%. The $J_\parallel$ blocks do not agree: AE3D's block (`F(SJPD) - F(SD)` of the instrumented runs, norm 299) against FAX's
(norm 370 in the same units) differ by a relative 1.2–1.4 for either sign, and by 0.86 even when FAX's $K$ is fed AE3D's own
$J_\parallel/B$ and $B_\rho$. AE3D's input is itself inconsistent with its embedding: its `brho` has rms 1.3e-2–3.6e-2, while
$\mathbf B\cdot\mathbf r_s$ of the file's own Boozer embedding is below 1e-5, and its
$J_\parallel/B$ has a standard deviation of 0.2–0.8 over a surface where the curl of the embedded field is constant to 2e-3
of $\max|\sigma|$ (`test_finite_beta_terms_match_real_space_quadrature`). The cause of the $J_\parallel$ disagreement is not established, and the
CTOK TAE does not depend on it (338.18 kHz without, 337.92 kHz with the term).

**Form of the kink term.** FAX assembles the direct form
$\int\sigma\,\mathbf b\cdot(\nabla\tilde\phi^*\times\nabla u)\,dV$, not AE3D's paper form with $\nabla\sigma$; the two differ by
$-\int u\sigma(\nabla\times\mathbf b)\cdot\nabla\tilde\phi^*\,dV$ ({doc}`theory`, section 4). On the CTOK test potential of
`test_finite_beta_terms_match_real_space_quadrature` ($n = 2$, $m = 3$, $4$), as fractions of the bending energy
(8.34): $K = -0.226$, $K - K_{\nabla\sigma} = -0.0069$, the omitted term $-0.0073$, the $\boldsymbol\kappa$ correction of the
bending energy $+0.0007$; the identity holds to 0.2% of $K$. The eigenvalues in the table above are unchanged by this
analysis (no implementation change).

## Independent 3-D reference: precise QA of Knyazev et al.

The archive of Knyazev et al. (2026; Zenodo DOI 10.5281/zenodo.21299907, CC-BY-4.0) holds a precise-QA equilibrium
(`nfp = 2`, aspect 6, finite-$\beta$ VMEC, `boozmn_betaQA.nc`), its STELLGAP continuum and its AE3D spectrum with the kink
term removed. `tests/data/knyazev_qa.npz` (0.67 MB) reduces it; `tests/data/PROVENANCE.md` has hashes and switches, and
`examples/knyazev_qa.py` rebuilds the fixture and reproduces every number below (15 min, 8 GB). FAX runs with
`finite_beta=False`, a flat density of 1.88e-6 kg/m$^3$ (the decks are not in the archive; the value reproduces the
continuum), AE3D's 49 nodes $s = 0.02..0.98$ and its 162 harmonics. AE3D's sine basis is FAX's operator on
$(m, n), (-m, -n)$ restricted to the odd subspace.

```{image} _static/knyazev_qa.png
:alt: Precise-QA continuum and gap band, the archived global mode in FAX's eigenbasis, convergence with the harmonics
:width: 100%
```

**Continuum.** STELLGAP labels surface $k$ one cell (0.02) outward, as for CTOK. On its 98 stored surfaces (162 sorted
branches each) the relative L2 error is mean 3.0e-3, p95 3.4e-3, max 4.3e-3 in the complex basis of the harmonic list
(3.5e-3 mean, 4.4e-3 max in the odd sector), unchanged between grids of 48 x 32 and 112 x 72 where unaliased. The branches
that bracket the archived global mode's gap at $s > 0.25$ agree to 1.0% (the lower, median 0.4%) and 3.0% (the upper, near 35 kHz, median 1%). The
spectrum of AE3D is dense: only 3 of its 7938 eigenvalues above 5 kHz are 1% from both neighbours, so **no isolated
frequency exists** in this case, and 56 eigenvalues are negative (AE3D's own, down to $-84$ kHz$^2$); FAX's operator is
positive by construction.

**The gap band.** The paper follows one radially global mode at 30.309 kHz. It sits in a continuum gap (26-35 kHz at
mid-radius) that holds 82 AE3D eigenvalues, of which 20 are radially extended (inverse participation ratio of at least 0.3
of the nodes) and mid-gap (mean position at least 0.25 of the way between the STELLGAP branches that bracket them, radius-weighted).
FAX is judged on the same rule. $\hat f$ is the Rayleigh quotient of the archived eigenvector in FAX's $F$, $G$;
"best" is the FAX mode with the largest $G$-overlap with the archived global mode; "contained" is the mean fraction of each
archived band mode inside FAX's band subspace.

| nodes | harmonics | grid | modes in gap (AE3D 82) | band modes (AE3D 20) | global $\hat f$ shift | best FAX mode kHz (overlap) | band mean $\hat f$ shift | contained |
|---|---|---|---|---|---|---|---|---|
| 49 | 24 | 48 x 32 | 73 | 14 | +0.9% | 29.58 (0.14) | +8.6% | 0.22 |
| 49 | 40 | 48 x 32 | 74 | 22 | -0.0% | 29.79 (0.19) | +3.5% | 0.49 |
| 49 | 64 | 48 x 32, 64 x 48 | 91 | 21 | -1.1% | 28.58 (0.14) | +1.3% | 0.39 |
| 49 | 96 | 48 x 32 | 73 | 20 | -4.2% | 27.77 (0.19) | -0.9% | 0.40 |
| 49 | 162 (AE3D's) | 48 x 32 | 78 | 18 | -5.1% | 27.61 (0.18) | n/a | 0.17 (global mode) |
| 24 | 40 | 48 x 32 | 39 | 18 | -0.2% | 30.11 (0.33) | +5.1% | 0.44 |
| 99 | 40 | 48 x 32 | 139 | 29 | -0.5% | 30.35 (0.14) | +2.9% | 0.40 |

Harmonics are the largest by the energy of the archived band. The last 162-harmonic, 49-node row is AE3D's own setting;
its global-mode quotient is 28.76 kHz against the archived 30.31. The localization of the best FAX mode matches the archived
one (IPR 32 against 35 nodes, centroid $s$ 0.41 against 0.49).

**Convergence.** The angular grid is converged: 48 x 32 and 64 x 48 give identical numbers to the digits printed, and the band
subspaces coincide (cosines 1.00). Harmonics and nodes are not: FAX's own band subspace is 0.55 inside itself between 40 and
64 harmonics, 0.55 between 64 and 96, 0.64 between 24 and 49 nodes and 0.58 between 49 and 99. The number of modes in the gap
scales with the nodes (39, 74, 139 at 24, 49, 99), so counts compare only at AE3D's 49 nodes; the quotient of the archived
mode converges only when all of its harmonics are kept (+0.9% to -5.1%).

**Verdict.** What agrees: the continuum (0.3%, gap edges 1-3%), the count (18-22 against 20), frequency range
(28.4-34.9 against 28.6-33.2 kHz) and localization of the gap band at AE3D's resolution. What does not: the archived global
mode is not a FAX eigenmode. At matched settings its energy in FAX's operator is 5.1% below its AE3D eigenvalue, and it is
spread over many FAX modes (27.6-30.7 kHz, largest overlap 0.18 at 27.61 kHz). PR #9 recorded the same from a separate
assembly: FAX with physical shear indices 27.57 kHz, and a reconstruction of AE3D's public formulas 30.30 kHz (diagnostic
numbers, not rerun here; its operator is not in this repository). The 27.6 kHz mode is therefore reproducible, but it is
not unique: in this near-degenerate band the mode-by-mode eigenvectors are ill-conditioned, FAX's own band subspace moves
by 36-45% under refinement, and the overlap with AE3D (0.4) is of that size. This is consistent with, but does not
establish, a difference through AE3D's shear cross terms (the CTOK discussion above): the pinned AE3D v31 build exists, but the
QA deck has not been run through its blocks here, so the cause is unresolved. `finite_beta` terms are off, as in the archive.

## Gradients

- **Eigenvalues.** For the Könies operator at $N = 40$, the Hellmann–Feynman
  derivative of a dense cluster mean and of a (SOLVAX) shift-invert eigenvalue with
  respect to density and an $\iota$ offset matches central differences to
  1e-5, and $\partial f/\partial\ln\rho = -f/2$ to 1e-10. The same holds with
  `finite_beta=True` and the Könies pressure (central step 1e-7).
- **Drive.** `jax.grad` of $\gamma_h$ with respect to the fast-ion density
  scale, $T_f$ and $\omega$ matches central differences to 1e-5; the
  slowing-down gradient is finite through the smoothed birth edge.
- **Boundary.** On low-resolution precise QA ($n_s = 16$, ftol 1e-12,
  `mboz = 3`), with the lowest $n = 0$ global frequency ($m = 1$–3), only the
  FE radial grid is refined:

  | nodes | f (kHz) | df/d`zs01` (kHz) | FAX AD vs FD |
  |---|---|---|---|
  | 16 | 294.648 | -62.528 | 3.4e-8 |
  | 32 | 294.219 | -61.230 | 3.5e-8 |
  | 64 | 293.989 | -60.583 | 3.6e-8 |
  | 128 | 293.873 | -60.261 | 3.6e-8 |

  Both converge at first order (differences halve), as expected for the
  continuum minimum. The full adjoint gradient matches re-solve finite
  differences to 3e-5 for `zs01`, `rc01` and `rc10`; the remainder is the
  $m = 1$ gauge drift of the re-solves ({ref}`vmex-gradients`), which is
  2.7e-3 for `zs11`.

**Derivative contract (first order).** Derivatives are promised to first order in the boundary offsets (and the
geometry leaves), through the Hellmann–Feynman eigenvalue rule and the vmex implicit adjoint; static choices (FE nodes,
harmonics, angle grid, `ns`, `mboz`) are not differentiated; the frequencies and their tangents are NaN unless the
solve converged and the selection does not split a tied set. Each route is tested on its own:

- **Fixed geometry** (`test_fixed_geometry_derivative_has_a_step_size_plateau`). A tilted torus with every
  stellarator-asymmetric harmonic (`rmns`, `zmnc`, `bmns`, `gmns`) and $\iota$, $B$ moved together; dense and
  shift-invert. Relative error of AD against central FD at $h = 10^{-1}, \dots, 10^{-9}$:

  | route | $10^{-1}$ | $10^{-2}$ | $10^{-3}$ | $10^{-4}$ | $10^{-5}$ | $10^{-6}$ | $10^{-7}$ | $10^{-8}$ |
  |---|---|---|---|---|---|---|---|---|
  | dense (`eigenvalue_mean`) | 1.2e-3 | 1.2e-5 | 1.2e-7 | 1.2e-9 | 7e-11 | 7e-9 | 5e-8 | 2e-6 |
  | shift-invert (SOLVAX) | 2e-7 | 2e-9 | 1e-10 | 1e-9 | 6e-9 | 1e-7 | 2e-6 | 5e-6 |

  Truncation falls as $h^2$, then a floor of 1e-10 to 1e-9 spans $10^{-3}$–$10^{-5}$, then roundoff grows as $1/h$:
  the test requires at least four steps below 1e-6.
- **Re-solved equilibrium** (LP-QA, the gap mode of the next section; tilted tokamak for `lasym`). Remainder of the
  re-solve FD against the adjoint at $h = 10^{-4}$ and $10^{-5}$:

  | direction | remainder (both steps) | reading |
  |---|---|---|
  | `zs01` (LP-QA) | 4.6e-4 | flat, so not truncation or noise |
  | `zs11` (LP-QA, $m = 1$) | 1.14e-2 | flat: the $m = 1$ $Z_\mathrm{sin}$ gauge |
  | `zc20` (tilted tokamak, asymmetric, $m = 2$) | 1.2e-4 | flat |
  | `rs20` (same, asymmetric) | 2.8e-4 | flat |

  With vmex 0.14, which holds the $m = 1$ family in the implicit solve and its derivative, `zs01` and `zs11` close to 3e-7 and 1.2e-6 (falling with the step); `zc20` keeps its 1.2e-4.

  The remainder does not move with the step, so it is a systematic difference between the re-solve and the adjoint,
  which the plan attributes to the released $m = 1$ gauge (the adjoint holds it fixed, each re-solve lets it
  drift; 4.4e-4 for $\iota$ on `zs01`). **Not closed:** vmex's gauge-free reference `frozen_path_directional_fd`
  did not run on these decks. Along `zs11` its Newton iteration diverges (the residual grows from 2e-3 to 67 in two
  steps), and along `zs01` it reaches a non-finite state after 10 steps. The `zs11` and `zs01` sensitivities
  are therefore verified to the floors above and not to 1e-5, and `plan.md` keeps that item open.

**Resolution budget of the real objective.** The lowest of the two $n = 1$ gap modes of LP-QA (102 kHz, next section) and
its gradient in the four coefficients of the QA example (`rc11, zs11, rc12, zs12`). Base: 24 FE nodes, $n_s = 16$,
`mboz = nboz = 6`, $16 \times 16$ angles, $m \le 5$ ($f$ = 101.8964 kHz). One resolution is refined at a time
(`test_resolution_budget_of_the_gap_mode_frequency_and_gradient`, scheduled tier); the change is between the two finest
levels, which for a first-order discretization is also the size of the remaining error:

| refined | levels | $f$ at finest (kHz) | $\lvert\Delta f/f\rvert$ | $\lVert\Delta\nabla f\rVert/\lVert\nabla f\rVert$ |
|---|---|---|---|---|
| FE nodes | 24, 48, 96 | 102.1833 | 1.1e-3 | 2.4e-3 |
| VMEC `ns` | 16, 24, 32 | 102.0752 | 5.6e-4 | 1.4e-3 |
| Boozer `mboz = nboz` | 4, 6, 8 | 101.8934 | 3.0e-5 | 9.4e-4 |
| angle grid | 16², 24², 32² | 101.8965 | 2e-10 | 7.6e-9 |
| harmonics $m \le M$ | 5, 7, 9 | 101.7096 | 1.3e-4 | 4.2e-4 |

Added in quadrature the budget is 1.2e-3 for $f$ (0.13 kHz) and 3.0e-3 for the gradient, plus the 1.1e-2 gauge floor of
the `zs11` component. The FE-node level matters most: 24 nodes sit 0.28% below 96 nodes (the same bias in the
before and after values of an optimization). The angle grid is exact once it resolves the harmonic products.

## Optimization

`examples/optimize_alfven.py` ({doc}`tutorials`, section 7), $n_s = 16$,
`mboz = nboz = 6`, office CPU (`JAX_PLATFORMS=cpu`). These runs are **exploratory**: the QH run lowers a geometric proxy
($f_\mathrm{cont}$), the QA runs move selected modes, and the drive run changes a passing-only growth rate by an amount not bounded by
an error budget. None is an independent demonstration that the optimized device has fewer Alfvén eigenmodes or lower fast-ion
transport.

**QH: Paul et al. (2025) eq. 7.1 on LP-QH** (reactor scale, $N_P = 4$, helicity
$(1, -1)$, cut $|\iota + 4|$), 34 coefficients, 60 evaluations, 31 min:

| | initial | optimized |
|---|---|---|
| $f_\mathrm{cont}$ | 7.038 | 5.226 (−26%) |
| QS total ($f_{QS}$) | 1.32e-4 | 5.72e-4 |
| aspect ratio | 8.0000 | 8.0037 |
| mean $\iota$ | −1.2431 | −1.2393 |

Largest gaps above the cut, centre $\omega^{(0)}/\omega_A$ and width
$2\omega^{(0)}|\epsilon|/\omega_A$ (full width, surface-averaged):

| $(\delta m, n)$ | centre | width before | width after |
|---|---|---|---|
| (3, −8) HAE | 2.14 | 0.280 | 0.278 |
| (0, 4) MAE | 2.00 | 0.232 | 0.196 |
| (2, −8) HAE | 2.76 | 0.174 | 0.136 |
| (0, 8) MAE | 4.00 | 0.138 | 0.098 |
| (3, −12) HAE | 4.14 | 0.132 | 0.116 |

The high-frequency gaps
narrow (up to 29%, the direction of Paul's figures 8–9) while QS degrades moderately (×4.3, still 5.7e-4) and
aspect ratio and $\iota$ hold. The reduction is smaller than Paul's
(Wistell-A start, many more evaluations): the cost was still falling at
60 evaluations, and the dominant (3, −8) gap barely moved, since closing it
costs QS on this precise-QS start (QS row weight 0.1).

**QA: which modes did the earlier example move?** `fax.classify_modes` on the initial geometry (LP-QA, $n_s = 16$,
`mboz = 6`, 24 nodes, `count = 6`) shows that the three lowest $n = 0$ global modes of the example below are
**continuum-like**, not gap modes. Their frequencies 293.96–296.38 kHz sit on the $n = 0$ continuum minimum (the
branch at 294–304 kHz over $s$), each peaks at its own radius ($s$ = 0.96, 0.92, 0.88, 0.84, 0.80, 0.76), the margin
to the local branch is 0.000 for all, and none is isolated. Their frequency is $\min_s f_{m=1}(s) \propto |\iota m - n|$:
the earlier run moved the continuum minimum (mainly through $\iota$, −0.8%), and its "global modes leave the window"
is a statement about that branch, not about a mode that a fast ion could resonate with. The result stays valid as a
test of the gradient chain (it is what the gradient tests above differentiate for the continuum minimum), and is kept in
the example as `--modes continuum`.

**QA: global modes out of a window.** The three lowest $n = 0$ global modes
($m = 1$–3, 24 nodes) start at 294.10–294.92 kHz inside
[293.10, 314.92] kHz; 4 coefficients (`rc11, zs11, rc12, zs12`),
30 evaluations, 18 min:

| | initial | optimized |
|---|---|---|
| modes (kHz) | 294.10, 294.44, 294.92 | 290.65, 290.99, 291.47 |
| window risk row (cost) | 143.2 | 5.17 |
| QS total | 1.09e-6 | 3.96e-6 |
| aspect ratio | 6.0000 | 5.9881 |
| mean $\iota$ | 0.4191 | 0.4159 |

All three modes leave the window (2.4 kHz below its edge, 2.4 softness
widths) with QS still at the 4e-6 level; the shift comes mainly from
$\iota$ (−0.8%), as $f \propto |\iota m - n|$ near the continuum minimum.

**QA with classified gap modes (the 3-D boundary objective of work package E).** The two $n = 1$ modes
($m = 0$–5, $n \in \{-1, 1, 3\}$, $16 \times 16$ angles) nearest 103 kHz, `python examples/optimize_alfven.py --case qa
--modes gap --nfev 8` (8 evaluations, 8 min on a CPU; 24 FE nodes), window $[f_\mathrm{min} - s, f_\mathrm{max} + 20 s]$,
$s$ = 0.35 kHz. In the 96–142 kHz gap of the $n = 1$ continuum, `classify_modes` (24 and 49 nodes) finds both isolated:

| mode | $f$ refined (kHz) | peak $s$ | shift | overlap | margin | isolated |
|---|---|---|---|---|---|---|
| 1, initial | 102.074 | 0.580 | 1.7e-3 | 0.995 | 0.079 | yes |
| 2, initial | 103.859 | 0.020 | 5.6e-3 | 0.980 | 0.119 | yes |
| 1, optimized | 98.080 | 0.580 | 1.7e-3 | 0.994 | 0.080 | yes |
| 2, optimized | 99.515 | 0.020 | 5.7e-3 | 0.982 | 0.120 | yes |

(The other modes in the window, 94.5–96.1 kHz, are continuum-like: margin 0.000, one per radius from $s$ = 0.68 to 0.96.
The `--drive` selection of the four modes nearest 104 kHz includes two of them.) The optimizer moves both modes out of the
window through its lower edge and they remain isolated gap modes at the same radii:

| | initial | optimized |
|---|---|---|
| modes at 24 nodes (kHz) | 101.896, 104.441 | 97.909, 100.086 |
| $J\,\delta p$ (kHz), first order | | −3.941, −4.364 |
| re-solved $\delta f$ (kHz) | | −3.988, −4.355 |
| remainder (second order) | | −0.046, +0.008 kHz (1.2%, 0.2%) |
| QS total | 1.09e-6 | 7.4e-6 |
| aspect ratio | 6.0000 | 6.0134 |
| mean $\iota$ | 0.4191 | 0.4227 |

The contract: the shift of 4 kHz is predicted to 0.05 kHz by the adjoint Jacobian (first order), and the numerical error
budget of each value is 0.13 kHz (resolution table in Gradients, 1.2e-3), plus a 1.1e-2 gauge floor in the `zs11`
component of the gradient; the 4 kHz change exceeds the budget by a factor 30. The coarse 24-node value of mode 2 is 0.6 kHz
above the 49-node one (it is axis-localized, `shift` 5.6e-3): the before and after values carry the same bias.
The $f_\mathrm{cont}$ row of the example (11.1 before, 11.09 after) is printed for reference and is not optimized here.
The $f_\mathrm{cont}$ values of the QH run above used Paul's hard mask; the smoothed mask of
{func}`fax.continuum_cost` (softness 0.05) differs from it only for harmonics within 0.05 of the cut.

## Fast-ion drive

The drive is evaluated in Boozer geometry ({doc}`theory`, section 8); the
tokamak gates use {func}`fax.circular_torus`, axisymmetric (`helical = 0`,
`nzeta = 1`).

- **Fu–Van Dam.** Weak toroidicity ($\varepsilon_B = 10^{-4}$, $R_0 = 10^4a$,
  nonzero drift coupling) and a flat Maxwellian with the two
  $v_\parallel = v_A$ resonances: $\gamma$ at fixed $\beta_h$ scales as
  $x(1 + 2x^2 + 2x^4)e^{-x^2}$, $x = v_A/v_{th}$, to 1e-3, and its absolute
  value equals the closed-form ZOW integral (NumPy, derived in the test
  docstring: orbit-averaged kernel, $\int 2\xi\,d\xi\,\xi^{-8}(1 + \xi^2)^2e^{-x^2/\xi^2}
  = 2(1 + 2x^2 + 2x^4)e^{-x^2}/x^6$, independent $W$) to 1e-3. The prefactor
  $\pi^2\mu_0m_f^2/2W$ is that of {eq}`gamma-zow`, pinned by the Könies gate,
  not derived there. The earlier test used $\varepsilon_B = 0$, where $\gamma = 0$
  exactly (no $|B|$ variation, no drift coupling), so it checked nothing.
- **Könies et al. (2018).** Model TAE ($m = 10, 11$, $n = 6$, Gaussian
  envelope of 10 cm FWHM at $q = 1.75$), $\omega = 4.0$e5 rad/s, the paper's
  fast-ion density profile (its eq. 1), `circular_torus(epsilon_B = a/R0)`:

  | $T_f$ (keV) | 100 | 200 | 300 | 400 |
  |---|---|---|---|---|
  | FAX, Boozer geometry, $j$-form (10³ s⁻¹) | 4.29 | 16.39 | 32.72 | 64.48 |
  | FAX, Boozer geometry, previous $n_h/\iota$ form | 4.33 | 16.52 | 32.94 | 64.85 |
  | FAX, previous circular large-aspect-ratio model | 4.4 | 16.6 | 33.1 | 65 |
  | Könies pooled ZLR band (not a reference) (10³ s⁻¹) | 8–10 | 25–33 | 38–43 | |

  The geometric path and the circular model differ in $O(\varepsilon)$ terms
  ($dl/d\zeta = G/B$ instead of $R_0$, the Boozer angle, the exact $\nabla B$) and in the
  label of the gradient term ({doc}`theory`, section 8: $-1.3$ to $-1.7\%$); they agree within 4%. The pooled ZLR band mixes codes with
  trapped particles, FOW-free but FLR-free kinetics and different damping, and is **not** the comparison used
  here (rule 2: no fitting to the pooled band).
- **ITPA ladder, rung 2 (fixed-mode ZOW).** Könies et al. (2018) fig. 1a has one zero-orbit-width curve, CAS3D-K (ZOW), valid at small
  $T_f$ (table 1: no FOW, no FLR). The paper does not say whether it contains trapped particles, and FAX is passing-only,
  so the content is **not matched**. The dotted curve read off the vector figure (the points between 100 and 135 keV are hidden
  behind other markers) and FAX (same mode, $\omega = 4.0\cdot10^5$ rad/s, eq. 1 density, Maxwellian):

  | $T_f$ (keV) | 82 | 139 | 163 | 191 |
  |---|---|---|---|---|
  | CAS3D-K ZOW (10³ s⁻¹) | 4.26 | 14.60 | 19.98 | 25.17 |
  | FAX, passing only (10³ s⁻¹) | 2.47 | 8.72 | 11.63 | 15.12 |
  | ratio | 0.58 | 0.60 | 0.58 | 0.60 |
  | $\gamma/\omega$ | 0.006 | 0.022 | 0.029 | 0.038 |

  FAX is a constant 0.58–0.60 of the ZOW curve over a factor 2.3 in $T_f$, which is what a missing trapped contribution of
  about 40% would look like; that attribution is not established and needs either a passing-only CAS3D-K ZOW run (Könies and
  Kleiber 2012, not in the reference folder) or trapped particles in FAX. Missing damping is unknown, not zero: the paper
  subtracts each code's damping so that $\gamma(T_f = 0) = 0$, FAX has no continuum, electron or ion Landau damping of the
  mode ($\gamma_d$ would be subtracted). Frozen-mode validity: $\gamma/\omega$ stays below 5% up to 200 keV and reaches 8% and 16%
  at 300 and 400 keV, where the perturbative picture also loses the frequency shift of fig. 2 and finite orbit width. Rungs 3–5
  (FOW/ZLR, FLR, damping) are not started.
- **Signs.** A flat profile gives equal damping for $(m, n)$ and $(-m, -n)$;
  a peaked profile drives $n > 0$ and damps $n < 0$ ($\iota\psi' > 0$); a
  hollow profile damps; bulk-ion Landau damping is positive.
- **Gradients.** `jax.grad` with respect to every {class}`fax.Geometry` leaf and
  the mode, along a random direction, matches central differences to 1e-5.
- **3-D structure** ({doc}`theory`, section 8). The gradient term is $j\,\partial_sF/(\omega Ze\psi')$ with the orbit's poloidal harmonic $j$,
  not the mode's $n_h/\iota$. (i) A tokamak (with and without $I$) and its helical relabelling $\theta' = \theta - N\zeta$
  ($N$ = 2, 4; $\iota \to \iota + N$, $n \to n + mN$) have the same $\gamma$ to 4e-5 (the old form: 17–20%).
  (ii) Weak-toroidicity closed form with a density tilt: the gradient term, 27% of the profile factor at $x = 1$, is
  reproduced with $j = 11, 10$ to 1e-3 (the $n/\iota$ form is 1–2% off). (iii) $\theta$- and $\zeta$-mirrors
  ($\iota \to -\iota$ with $m \to -m$, or $n \to -n$, $I \to -I$) agree to 1e-12 and the second maps each $(j, k)$ to $(j, -k)$ with
  the same $\gamma_{jk}$, so co- and counter-passing resonances both carry drive. (iv) Amplitude and global phase of $\phi$,
  and density scaling, exact to 1e-12; a uniform Maxwellian or slowing-down distribution has $\gamma < 0$ at every $\omega$.
  The model uncertainty of the label ($F$ of the mean flux versus of the $P_\zeta$ surface) is $O(\omega/n\Omega_\varphi)$: 1.5% for $n = 6$, 23% and 7% at 25 and 50 keV for the CTOK
  $n = 2$ mode.
- **Regularizations** (explicit arguments, defaults in brackets; relative error against the converged value):

  | Parameter | 0.1 | 0.01 | 0.001 | 1e-4 | note |
  |---|---|---|---|---|---|
  | `boundary` [0] | 1.5e-1 | 1.8e-2 | 1.0e-3 | 2e-6 | first order up to a log; Könies-like torus |
  | `guard` [0.01], away from rationals | 6e-3 | 8e-5 | 1e-7 | 1e-8 | helical test torus, $\iota \in [0.43, 0.46]$ |
  | `guard`, $\iota = 1/2$ crossed (shear 0.3) | 5e-3 | 2e-4 | 1e-5 | 1e-7 | |
  | `guard`, $\iota \approx 1/2$ on the whole mode (shear 0.02) | 1.7e-2 | 1.1e-2 | 3.1e-2 | 2e-3 | not converged above 1e-4 |

  `FastIons.birth_width` [0.02] converges as $O(w^2)$ (1865.6, 1848.6, 1845.0 s⁻¹ at 0.04, 0.02, 0.01;
  difference ratios 5.0 and 4.7 per halving) provided the pitch nodes resolve the edge ($\gtrsim 2/w$); at 32 nodes and $w \le 10^{-3}$ the
  edge falls between nodes and its share of $\gamma$ (about 9% in that case) is lost without any warning. `nodes` [32]: 3.5e-3, 9e-4, 3.5e-4, 2e-5 at 8, 16, 32, 64.
  Near a rational surface of a 3-D equilibrium $\gamma$ depends on `guard` at the percent level (a true singularity, not
  noise): refine it there.
- **CTOK.** The FAX TAE of the CTOK `boozmn` (272.8 kHz, $n = 2$, $m = 0..12$,
  D at 10¹⁹ m⁻³) from the shift-invert solve, driven by deuterons with
  $n_f = 10^{17}(1 - s)^2$ m⁻³: $\gamma/\omega$ = 7.0e-4 (25 keV), 3.3e-3
  (50 keV), 1.57e-2 (100 keV); 9.2e-4, 3.6e-3, 1.6e-2 with the previous $n_h/\iota$ form; bulk-ion Landau damping at 1 keV is 2.6 s⁻¹.
- **Stellarators.** LP-QA (ns 16, 16 FE nodes, $n = 1$ family $m = 0..5$,
  $n \in \{-1, 1, 3\}$, the four modes nearest 104 kHz, 150 keV Maxwellian
  deuterons, $n_f = 10^{15}(1 - s)^2$): resonances with $k \ne 1$ carry a
  fraction $10^{-6}$–$10^{-2}$ of $\gamma$, as expected for quasi-axisymmetry,
  and the pure helical sidebands $k = -3, 5$ (outside the mode's own $n$) are
  nonzero. `jax.jvp` of {func}`fax.drive_risk` of {func}`fax.geometry_growth_rates`
  (FE interpolation, shift-invert eigenvectors differentiated implicitly,
  drive) along the re-solved geometry tangent of `rc01` matches FD
  to ≤ 1e-4 (5e-8 measured at 24 nodes). The full `jax.grad` through the vmex
  adjoint differs from re-solve FD by 1.6e-3: vmex's released $m = 1$ gauge
  ({ref}`vmex-gradients`), amplified by the drive's sensitivity. In LP-QH
  (helical $B \sim \cos(\theta + 4\zeta)$, $N_{fp} = 4$) the helical sidebands
  dominate some modes: for the 43.9 kHz gap mode of the $n = 1$ family
  driven by slowing-down alphas, 99% of $\gamma$ comes from $k = -7$, a
  toroidal number absent from the mode.

## Not done

- **Sound and fast-wave coupling** (no BAE gap; $\gamma p|\nabla\cdot\boldsymbol\xi|^2$
  dropped) and the cause of the disagreement between AE3D's and FAX's $J_\parallel$ blocks for CTOK.
- **Trapped and precessing fast ions** in the drive (bounce-averaged
  precession resonances), and the covariant $B_s$ and $\delta B_\parallel$ in the drift.
- **Finite orbit width** and finite Larmor radius in the drive.
- **Continuum and electron Landau damping** are not computed (subtract an
  external $\gamma_d$).
- **AE3D LHD (37, 73.7 kHz) and QPS (149 kHz)** published frequencies need
  their equilibria; Mishchenko et al. (2011) is not yet a gate.
- **Optimization**: HSX optimization (the HSX fixed-boundary decks are
  mpol 18, ntor 24, and were not reduced or run); a larger $\gamma_\mathrm{max}$ reduction (the 4-coefficient LP-QA
  `--drive` run lowers it by only 4.4%, 8500 → 8123 s⁻¹, at QS 1.1e-6 → 1.6e-6); a
  fast-ion resonance-band term; higher resolution ($n_s \ge 25$) and longer
  QH runs.
- {func}`fax.circular_torus` has no Shafranov shift: it reproduces Könies'
  ad hoc equilibrium, not the CAS3D one.
