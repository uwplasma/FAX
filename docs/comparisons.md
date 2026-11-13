# FAX and other codes

Shear-Alfvén physics in stellarators has been computed for three decades, mostly by Fortran codes that each cover part of the
problem. This page says where FAX overlaps with them, which comparisons FAX has actually made and how they came out, and what
the other codes do that FAX does not. Every FAX number is taken from the {doc}`validation` page, the {doc}`theory` page or the
tests they cite. Statements about other codes are limited to what their publications say; where FAX has not compared against a
code, the table says only what the code is for.

## Capability table

The columns describe what each code computes by design, not what a particular run has used. "Public" means a repository or
archive we know of; a blank means we do not know and have left it out.

| Code | Geometry | Model | Fast-ion drive | Damping | Differentiable | Language, hardware | Public | Status |
|---|---|---|---|---|---|---|---|---|
| FAX | 3-D, Boozer coordinates from VMEC | Continuum (STELLGAP form); reduced ideal-MHD global modes (AE3D weak form), restricted finite-$\beta$ terms | Perturbative, zero orbit width, passing ions | None computed; an external $\gamma_d$ can be subtracted | Yes: eigenvalues, drive and the VMEC boundary through vmex | Python, JAX; CPU, GPU not benchmarked | MIT | Active; 1.0.1 on PyPI as `alfvex` |
| STELLGAP | 3-D, Boozer | Low-$\beta$ continuum on each surface (Spong et al. 2003) | None | Continuum structure only | No | Fortran | ORNL-Fusion on GitHub | Last updated 2025 |
| AE3D | 3-D, Boozer | Reduced ideal-MHD global modes, clustered frequency analysis (Spong et al. 2010) | None | None | No | Fortran | ORNL-Fusion; last updated 2019 | Not under active development |
| FAR3D | 3-D | Reduced MHD with a gyrofluid energetic-particle model (Varela et al. 2024) | Gyrofluid | Via the gyrofluid closure | No | | No public repository | |
| FIRM3D | 3-D, VMEC and Boozer fields | Guiding-centre orbits in prescribed Alfvén modes from AE3D or FAR3D | Transport and losses from computed orbits | | No | | MIT; no tagged releases | Active |
| CAS3D | 3-D, full ideal MHD | Compressible ideal-MHD stability code | Not in the base code | Not in the base code | No | | | |
| CAS3D-K | 3-D | CAS3D with a perturbative kinetic energetic-particle response | Zero-orbit-width curve used for the ITPA comparison (Könies et al. 2018) | | No | | | |
| CKA | 3-D | Continuum damping by complex contour deformation | | Continuum damping | No | | | |
| NOVA, NOVA-K | Axisymmetric | MHD with a kinetic perturbative response in NOVA-K | In NOVA-K | In NOVA-K | No | | | |
| MEGA and other gyrokinetic or hybrid codes | 2-D and 3-D | Hybrid kinetic-MHD or gyrokinetic particle models | Self-consistent, nonlinear | Included by the model | No | | | Context only |
| AGNI | 3-D | Differentiable finite-$n$ ideal MHD (Gaur et al. 2026) | | | Yes | | | Recent |

The row for FAX is the only one verified line by line. The others summarise the publications cited in {doc}`references`,
and the blank cells are blanks on purpose. FAX is on PyPI as `alfvex`; see {doc}`getting-started`.

## Where FAX sits

FAX combines the physics of STELLGAP and AE3D, which Spong and coworkers published as separate Fortran programs, in one
differentiable package. The continuum and the global operator share one geometry object, so a continuum gap and a global mode in
that gap come from the same equilibrium. The only kinetic input is a perturbative drive evaluated on a fixed mode, which is the
level at which NOVA-K and CAS3D-K work as well. FAX does not try to reach the self-consistent level of MEGA or FAR3D.

The distinguishing feature is the derivative. Eigenvalues, the drive and the objective functions of {doc}`theory` are
differentiated with respect to profiles and, through vmex, to the VMEC boundary coefficients, to first order. The
near-axis theory of Paul et al. (2025) gives continuum structure and gap widths of quasisymmetric fields in closed form
and uses them as optimization targets; Hyder, Paul and Knyazev (arXiv:2611.07182) extend that to optimization. FAX evaluates the
same quantities from a finite equilibrium and can be used to check the near-axis predictions on a full VMEC solution.
Knyazev et al. (2026) work with the same weak form and supply the independent QA data that FAX is compared against below. AGNI
(Gaur et al. 2026) is also differentiable but treats finite-$n$ ideal MHD; the two codes address different mode families and
neither replaces the other.

## What has been compared

Each entry says what was compared, the number, and whether we consider it matched. The first figure collects them.

```{image} _static/compare_agreement.png
:alt: Relative difference of FAX from STELLGAP, AE3D, CAS3D, GYGLES and CAS3D-K for each comparison, with test gates
:width: 100%
```

*Relative difference of FAX from the other code, from the tables of {doc}`validation`. Filled markers are matched, open markers
are open discrepancies, bars are the gates in the tests. The Rayleigh-quotient point is 324.2 against 338.2 kHz, and the
CAS3D-K point is one minus the ratio 0.59.*

### Continuum against STELLGAP

For the STELLGAP test problem (circular tokamak, $n_s = 101$, $n = 2$, $m = 0..12$) the 13 branches on 97 surfaces differ from
the vendored STELLGAP output by a mean of 6.3e-3 and a 95th percentile of 6.6e-3, with a test gate of 7e-3. STELLGAP labels
its full-mesh surface $k$ as $k/(n_s-1)$, and the test shifts its labels by one cell; a scan of the offset has its minimum
there. On the precise-QA equilibrium of Knyazev et al. (2026) the mean is 3.0e-3 over 98 surfaces and 162 branches, with a
gate of 4e-3. The branches bracketing the gap of the archived global mode agree to 1.0% and 3.0%. This comparison is matched.

### Global modes against AE3D on CTOK

With the same `boozmn` file averaged to AE3D's 99-point mesh, six global modes (343.6 to 741.2 kHz) agree with AE3D within
1.4%, matched by eigenvector overlap above 0.9, against a gate of 1.5%. The leading matrix blocks and the operator continuum
agree within 1%.

The TAE at 337.9 kHz is not reproduced, and this one is open. A pinned build of AE3D v31, which reproduces its shipped
matrices and eigenvalues, was fed the same input as FAX and compared block by block. The inertia blocks agree to 1e-9 and the
bending blocks without shear to 2e-6 and 5e-7. Three terms proportional to $\iota'$ differ from the energy functional: an
index transposition that flips the sign of the antisymmetric part, a dropped antisymmetric cross term, and an opposite sign on
the $\iota' g^{s\theta}$ terms. A real-space quadrature of the bending energy differs from FAX's matrix by 0.13 to 0.30% and
from AE3D's matrix by 1.8 to 37%. The Rayleigh quotient of AE3D's 338.2 kHz eigenvector is 338.2 kHz in AE3D's blocks and 324.2 kHz
in FAX's. After the three terms are corrected inside AE3D's own Fortran, its localized gap modes move to 258.9, 264.9 and
266.4 kHz, against 258.8, 264.7 and 266.3 kHz for FAX's blocks assembled by AE3D's element routine. The remaining 3 to 4%
between the two assemblies comes from how AE3D's element routine treats the symmetric part of the $f'f$ coupling.

We read this as an explained disagreement in the bending blocks, and we have not reported it upstream. It does not
imply that published AE3D results change. The Könies case is insensitive to $\iota'$ (4.061e5 against 4.067e5 rad/s without
it). The $J_\parallel$ blocks of the two codes also disagree, by a relative 1.2 to 1.4, and the cause of that is open.
AE3D's own input for CTOK is not consistent with its embedding: its radial field component has an rms of 1.3e-2 to 3.6e-2
where the Boozer embedding gives less than 1e-5. The CTOK TAE does not depend on the $J_\parallel$ term in either code
(338.18 kHz without it, 337.92 kHz with it).

### The precise-QA global mode of Knyazev et al.

The paper follows one radially global mode at 30.309 kHz in a gap of the continuum. At AE3D's own settings (49 nodes, 162
harmonics) the Rayleigh quotient of that eigenvector in FAX's operator is 5.1% below its AE3D eigenvalue, and it is spread over
many FAX modes with a largest overlap of 0.18 at 27.61 kHz. At 24 to 64 harmonics the energy is within 2%. The gap band as a
whole matches: 18 to 22 radially extended modes against AE3D's 20, in 28.4 to 34.9 kHz against 28.6 to 33.2 kHz. The mode-by-mode
eigenvectors are ill-conditioned in this near-degenerate band, since FAX's own band subspace moves by 36 to 45% under refinement.
A separate assembly of the same weak form, built by E. Lascas Neto, gave 27.57 kHz with physical shear indices and 30.30 kHz with a
reconstruction of AE3D's public formulas, which points to the same shear-index issue as on CTOK. The QA deck has not been run
through the instrumented AE3D build, so the cause is unresolved. The continuum is matched and the single mode is open.

### TAE frequency against CAS3D and GYGLES

The ITPA test case of Könies et al. (2018) ($n = 6$, $m = 5..16$) gives $\omega = 4.0610\times10^5$ rad/s, converged to
1e-4 in the number of radial nodes. CAS3D reports 4.01e5 (FAX is 1.3% above, with a shifted equilibrium on the CAS3D side)
and GYGLES 4.13e5 (FAX is 1.7% below, with the same ad hoc equilibrium). The test requires a value between the two. With the
vmex equilibrium including pressure and both finite-$\beta$ terms, FAX gives 3.948e5, 1.5% below CAS3D, against 4.127e5 without the
terms. The remaining gap is plausibly the sound coupling the reduced model drops, which we have not tested. Matched, with that
caveat.

### Drive against CAS3D-K

The drive of the ITPA test case, evaluated with the circular-model Boozer form, is 4.29, 16.39, 32.72 and 64.48 $\times10^3$
s$^{-1}$ at 100, 200, 300 and 400 keV, and the test pins them to 1e-3. Against the CAS3D-K zero-orbit-width curve of Könies et
al. (2018) the passing-only drive is a factor of 0.58 to 0.60 from 82 to 191 keV. The ratio is unexplained: the content of
the published curve is not fully stated, and the candidates are listed on the validation page. We do not count it as a
validation. Independent of this comparison, FAX reproduces the Fu and Van Dam Maxwellian scaling and the closed-form
zero-orbit-width coefficient to 1e-3.

### Solver speed

Dense solves of the full pencil are the baseline that a targeted shift-invert solve replaces. At equal accuracy
($\eta \le 10^{-10}$) on an Apple M4 CPU, the warm forward time of the targeted solve against `scipy.linalg.eigh` is faster by
24 times for CTOK, 81 times for the Könies case at $N = 200$, 280 times at $N = 400$, and 17 times for the 64-harmonic Knyazev
case. Against FAX's own dense JAX solve the factors are 28, 96, 339 and 17. The warm gradient is 14, 55, 250 and 6.0 times faster
than the dense gradient. These are comparisons between solvers inside FAX, not against STELLGAP or AE3D, which were not
re-timed. A comparison with them is not available, and a continuum scan is a different computation from a targeted global solve.

The iteration count of the shift-invert solve grows with $N$ because the continuum accumulates around the gap. For the Könies TAE
it was 17, 33, 112, 310 and more than 1000 at $N$ = 200, 400, 800, 1600 and 3200 with SOLVAX 0.32, and is 4, 5, 11, 34 and 121
with SOLVAX 0.36. The warm solve with 0.36 takes 0.05, 0.11, 0.39, 2.3 and 15 s. The 0.32 run at $N = 3200$ stopped at its
iteration cap with $\eta = 2.8\times10^{-10}$. The 0.32 times used one BLAS thread and the 0.36 times eight, in different
sessions, so the right panel shows how each scales with $N$ and should not be read as a speed-up. The iteration counts on the
left are the like-for-like comparison.

```{image} _static/compare_scaling.png
:alt: Shift-invert iterations and warm solve time against the number of radial nodes for SOLVAX 0.32 and 0.36
:width: 100%
```

*Könies TAE, $H = 12$, three modes. Open marker: the 0.32 solve stopped at the iteration cap. Source: {doc}`validation`, performance
section.*

## What FAX does not do

The other codes above cover several things that FAX leaves out.

Fast-ion physics is the largest gap. FAX has no trapped or precessing particles, no finite orbit width (FOW) and no finite Larmor
radius (FLR) in the drive, and its drive is a perturbation of a fixed mode, valid to about 200 keV in the Könies case where
$\gamma/\omega \le 4\%$. Gyrokinetic and hybrid codes such as MEGA, and gyrofluid codes such as FAR3D, include these effects and
let the mode structure respond to the particles. The global model also leaves out sound and compressional coupling, so FAX has no
BAE or BAAE eigenmode (the continuum has opt-in sound coupling and shows the gaps), and its finite-$\beta$ terms are restricted and experimental.

FAX does not compute damping. It has no continuum damping, which CKA addresses with a complex contour, and no electron Landau
damping. Ion Landau damping of the bulk plasma is available through {func}`fax.ion_landau_damping`, and an external damping
rate can be subtracted from the drive. Because the global operator is Hermitian, FAX gives real frequencies and cannot say how a
mode radiates into the continuum.

FAX is linear. Nonlinear saturation, mode coupling and fast-ion transport belong to MEGA and to gyrokinetic codes; coupling FAX
modes to ESSOS orbit transport is planned and not implemented. FAX's derivatives are first order and are NaN when an eigensolve has not converged.

## References

Chen and Zonca (2016); Gaur et al. (2026, arXiv:2608.01750); Hyder, Paul and Knyazev (arXiv:2611.07182); Knyazev et al. (2026,
arXiv:2603.03118; data set 10.5281/zenodo.21299907); Könies et al. (2018); Paul et al. (2025); Spong, Sanchez and Weller (2003);
Spong, D'Azevedo and Todo (2010); Varela et al. (2024). The full list is on the {doc}`references` page.
