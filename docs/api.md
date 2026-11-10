# API reference

`import fax` exposes the functions below at the top level (for example
`fax.continuum` is {func}`fax.continuum.continuum`). Importing FAX enables
JAX float64.

## Derivative contract

Derivatives are promised to **first order**. Second derivatives are not part of the contract (a dense `eigenvalue_mean` Hessian
runs when no eigenvalue is repeated; nothing else is tested beyond first order). Every function below is a pure JAX function.
Arguments that fix a shape or an index (`harmonics`, `ntheta`, `nzeta`, `nodes`, `count`, `iters`, `start`, `stop`, `helical`,
`sidebands`, `kind`, `finite_beta`, `nfp`) are **static**: they are concrete Python or NumPy values, are not differentiated
and are validated only when concrete. `Geometry` is a pytree whose float leaves are differentiable; its integer leaves `xm`, `xn`
need `jax.grad(..., allow_int=True)`.

| Function | Differentiable inputs | Transforms | Status and failure behaviour |
|---|---|---|---|
| `circular_torus` | `s`, radii, `B0`, `iota`, `epsilon_B`, `pressure` | jit, grad, jvp | host `ValueError` for invalid concrete values; traced values are not checked |
| `read_boozmn`, `from_wout`, `from_vmec_input` | none (host I/O) | none | `ValueError` on malformed files; `from_vmec_input` needs the `vmex` extra |
| `surface_grid`, `jacobian_check` | every `Geometry` float leaf | jit, grad | diagnostics only; no status |
| `continuum`, `continuum_matrices` | `Geometry` leaves, `density` | jit, grad, jvp, vjp, vmap | host `ValueError` for `rho <= 0`, `G + iota I = 0`, or an angle grid that aliases the harmonics (`ntheta > 2 max|dm|`) |
| `gaps`, `gap_location` | `frequencies_hz` (`gaps`) | jit, grad | the widest gap switches branch where two gaps tie: a kink (use `gap_closure`) |
| `global_matrices` | `Geometry` leaves, `density` | jit, grad, jvp | `finite_beta` is restricted and experimental |
| `eigenmodes`, targeted | `Geometry` leaves, `density`: `omega2`, `structures` | jit, jvp, vjp | **first order only.** `status` (`Status`: `residual`, `orthogonality`, `iterations`, `converged`, `gap`, `fallback`) is returned, not enforced: modes come back even if unconverged. Tangents are NaN unless `converged` and `gap > 1e-8 max|omega2|` |
| `eigenmodes`, dense (`target_hz=None`) | same | jit, jvp, vjp | `status` from the dense residual; derivatives of single modes are undefined at a degeneracy |
| `frequency_hz` | `Geometry` leaves, `density` | jit, grad, jvp, vjp, vmap | value **and** tangent are NaN unless the targeted solve `converged` (and, for the tangent, `gap` is not a tie). Returns 0 where the mean `omega2 <= 0`; `growth=True` gives the ideal growth rate instead |
| `eigenvalue_mean` | `K`, `M` | jit, jvp, vjp, vmap | exact first derivative through degeneracies inside `start:stop`; the cluster must be separated from the rest of the spectrum |
| `growth_rate`, `ion_landau_damping` | `omega`, `phi`, `Geometry` leaves, `mass_density`, every `FastIons` field except `kind` (static) | jit, grad, jvp, vmap | NaN inputs propagate to a NaN result (no raise); `kind` outside `maxwellian`, `slowing_down` raises `ValueError`; `boundary`, `guard`, `nodes`, `birth_width` are regularizations that must be converged by the caller |
| `window_risk`, `soft_max`, `drive_risk`, `gap_closure` | their array arguments | jit, grad, jvp, vmap | smooth by construction; scale parameters (`softness_hz`, `sharpness`, `scale_hz`) are static |
| `compression_harmonics`, `continuum_cost` | `Geometry` leaves | jit, grad, jvp | `helicity`, `softness` static |
| `prepare_design` | none (host: runs vmex once) | none | builds the static part; needs the `vmex` extra |
| `design_geometry`, `state_geometry`, `design_state`, `design_params` | boundary `offsets` | jit, grad, jvp, vjp | implicit vmex adjoint, one extra equilibrium solve per scalar; the equilibrium must be converged (`ftol`) |
| `geometry_frequencies`, `design_frequencies` | boundary `offsets` or `Geometry` leaves | jit, grad, jvp, vjp | returns `(frequencies_hz, Status)`; frequencies **and** tangents are NaN unless `converged` and `gap > 1e-8 max|omega2|` |
| `geometry_growth_rates` | `Geometry` leaves, `ions.density` | jit, grad, jvp | NaN unless the underlying solve converged without a tie |
| `classify_modes` | none | none (host, not traceable) | reports `ModeClass`; it decides which modes an objective may act on and is never differentiated |
| `load`, `run`, `main` | none | none | `ValueError` for unknown TOML tables or keys; the command line runs only the `continuum` and `eigenmodes` tasks |

`Status.gap` is the distance of the nearest excluded eigenvalue to the target minus that of the farthest selected one; `gap <= 0`
means the selection splits a tied set and has no derivative. The transforms and failure behaviour above are pinned by `test_documented_transforms_and_failure_behaviour`; vmap of the vmex design path is not tested.

## Equation, function and test

| Equation (theory) | Function | Test |
|---|---|---|
| Conventions, $\sqrt g$, $B^\zeta$, $k_\parallel$ (section 1) | `surface_grid`, `jacobian_check`, `equilibrium_currents` | `test_conventions_and_geometry_diagnostics_on_ctok` |
| $K x = \omega^2 M x$, {eq}`continuum-galerkin` (section 2) | `continuum_matrices`, `continuum` | `test_uniform_field_continuum_is_k_parallel_v_alfven`, `test_stellgap_ctok_parity` |
| Gap location and full width $2 f_0 \lvert c\rvert$, {eq}`gap-location` (section 3) | `gaps`, `gap_location` | `test_gaps_and_gap_location`, `test_elliptic_two_mode_gap_has_full_width_two_f0_c` |
| $F y = \omega^2 G y$, {eq}`global-problem` (section 4) | `global_matrices`, `eigenmodes` | `test_ae3d_ctok_parity`, `test_ctok_bending_energy_matches_real_space_quadrature`, `test_konies_tae_frequency_and_convergence` |
| AE3D block comparison, shear term {eq}`shear-term` | `global_matrices` | `test_ae3d_v31_blocks_and_energy_on_identical_inputs` |
| Kink and pressure terms, {eq}`kink-forms` (restricted) | `global_matrices(finite_beta=True)` | `test_kink_forms_differ_by_the_curl_b_term`, `test_finite_beta_terms_match_real_space_quadrature`, `test_cylinder_newcomb_m1_and_suydam_threshold` |
| Shift-invert, `Status` (section 5) | `eigenmodes`, `targeted_eigenpairs` | `test_shift_invert_matches_dense_and_gradients_match_finite_differences`, `test_solver_status_and_rejection_of_unconverged_gradients` |
| $\delta\lambda = y^\dagger(\delta F - \lambda\delta G)y$, {eq}`hellmann-feynman` (section 6) | `frequency_hz`, `eigenvalue_mean` | `test_cluster_mean_first_derivative_at_degeneracy`, `test_fixed_geometry_derivative_has_a_step_size_plateau` |
| vmex boundary adjoint (section 7) | `design_frequencies`, `geometry_frequencies` | `test_boundary_gradient_and_radial_refinement`, `test_resolution_budget_of_the_gap_mode_frequency_and_gradient` |
| $\gamma = -P/2W$, {eq}`gamma-power`, {eq}`resonance`, {eq}`gamma-zow` (section 8) | `growth_rate`, `ion_landau_damping` | `test_fu_van_dam_maxwellian_scaling`, `test_koenies_2018_tae_growth_rates`, `test_quasi_helical_relabelling_leaves_the_drive_unchanged` |
| Smooth objectives (section 9) | `window_risk`, `soft_max`, `drive_risk`, `gap_closure` | `test_objectives_are_smooth`, `test_gap_closure_is_smooth_where_the_widest_gap_changes` |
| Compression harmonics and $f_\mathrm{cont}$ (section 10) | `compression_harmonics`, `continuum_cost` | `test_continuum_cost_of_an_elliptic_torus`, `test_continuum_cost_is_continuous_across_the_helicity_cut` |
| Mode classification (section 10) | `classify_modes` | `test_classification_separates_gap_modes_from_the_continuum_on_ctok` |

## Geometry

```{eval-rst}
.. automodule:: fax.geometry
   :members:
```

## Continuum

```{eval-rst}
.. automodule:: fax.continuum
   :members:
```

## Global eigenmodes

```{eval-rst}
.. automodule:: fax.eigen
   :members:
```

## Fast-ion drive

```{eval-rst}
.. automodule:: fax.drive
   :members:
```

## Objectives and the vmex pipeline

```{eval-rst}
.. automodule:: fax.objectives
   :members:
```

## Cases and command line

```{eval-rst}
.. automodule:: fax.io
   :members:
```

## Plots

```{eval-rst}
.. automodule:: fax.plot
   :members:
```
