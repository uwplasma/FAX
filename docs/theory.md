# Theory and numerics

This page derives every equation FAX solves, in the order the code applies
them, and names the function that implements each one. Units are SI;
frequencies `ω` are in rad/s and the public functions report `f = ω/2π` in Hz.

**Scope.** The model is the low-β shear-Alfvén ordering of Spong et al.: restricted finite-β terms (section 4, experimental), no
sound coupling (so no BAE or BAAE), no continuum or electron damping, and a drive for zero orbit width, zero Larmor radius and passing
ions only (section 8). {doc}`api` maps each equation to its function and test and states the derivative contract of each function.

## 1. Boozer coordinates and the metric

FAX works on flux surfaces labelled by the normalized toroidal flux
$s = \Phi_t/\Phi_\mathrm{edge} \in (0, 1)$, with Boozer angles $(\theta, \zeta)$.
With $\psi = \Phi_t/2\pi$ and $\psi' = d\psi/ds = \Phi_\mathrm{edge}/2\pi$ (VMEC's
`PHIEDGE`$/2\pi$) the field is

$$
\mathbf B = \psi'\left(\nabla s\times\nabla\theta + \iota\,\nabla\zeta\times\nabla s\right)
          = G(s)\,\nabla\zeta + I(s)\,\nabla\theta + B_s\,\nabla s ,
$$

so the embedding Jacobian is $|\sqrt g| = \psi'(G + \iota I)/B^2$ (the Boozer Jacobian
stored in a `boozmn` file, `gmnc`, is the same without $\psi'$) and the parallel
derivative is

$$
\mathbf B\cdot\nabla = \frac{B^2}{G + \iota I}\left(\partial_\zeta + \iota\,\partial_\theta\right).
$$ (bdotgrad)

A {class}`fax.Geometry` stores, on each surface, the Boozer harmonics of
$|B|$ (`bmnc`, `bmns`), of $R$, $Z$ and of $p = \phi - \zeta$ (the cylindrical
angle offset), of the Jacobian (`gmnc`, `gmns`), and the profiles $\iota$,
$G$ (`bvco`) and $I$ (`buco`). Every harmonic is
$c\cos(m\theta - n\zeta) + s\sin(m\theta - n\zeta)$ with $n$ in units of the
full torus (booz_xform convention, $n$ a multiple of $N_{fp}$).

(conventions)=
### Conventions

One set of conventions governs the code, the tests, the exports and this page.

| Item | Convention |
|---|---|
| Radial label | $s = \Phi_t/\Phi_\mathrm{edge}$, toroidal flux with its sign removed; finite-element nodes lie strictly in $(0, 1)$, with the axis and edge added as end nodes |
| Flux factor | $\psi = \Phi_t/2\pi$, $\psi' = \Phi_\mathrm{edge}/2\pi$ (Wb/rad). {func}`fax.geometry.jacobian_check` returns $\psi'$ from the embedding |
| Field components | $B_\zeta = G$ (poloidal current, `bvco_b`), $B_\theta = I$ (toroidal current, `buco_b`), both in T m; $G + \iota I > 0$ for the standard orientation, so $B^\zeta = B\cdot\nabla\zeta = B^2/(G + \iota I) > 0$ and $B^\theta = \iota B^\zeta$. AE3D's `jpol` is $G$ and its `jtor` is $-I$ |
| Handedness | $(s,\theta,\zeta)$ is left-handed against $(R,\phi,Z)$ for VMEC and `booz_xform` input: the embedding $\sqrt g = \mathbf r_s\cdot\mathbf r_\theta\times\mathbf r_\zeta < 0$. Volumes use $\lvert\sqrt g\rvert$; triple products $\mathbf b\cdot(\nabla a\times\nabla c)$ carry $\operatorname{sign}\sqrt g$ |
| Angles | $\theta$, $\zeta \in [0, 2\pi)$; $\zeta$ spans the full torus and the cylindrical angle is $\phi = \zeta + p(s,\theta,\zeta)$ |
| Toroidal number | full-torus $n$ (`ixn_b`, a multiple of $N_{fp}$), never $n/N_{fp}$; families are $n_0 + kN_{fp}$ ({func}`fax.mode_family`) |
| Phase | $\phi \propto e^{i(m\theta - n\zeta - \omega t)}$, so $\mathbf B\cdot\nabla\to i k_\parallel B$ with $k_\parallel = (\iota m - n)B/(G + \iota I)$; $\omega > 0$ and $(m, n)\to(-m, -n)$ is the complex conjugate. The sign matters for the drive ($Q_h$, section 8), not for the Hermitian operators $F$, $G$ |
| Units | SI. $\omega$ in rad/s and $\lambda = \omega^2$ in rad²/s² inside the solvers; `frequencies_hz` is $\omega/2\pi$; AE3D prints kHz. $\rho$ is the mass density in kg/m³ ({func}`fax.mass_density`), $\sigma$ in 1/m, pressure in Pa, $\mu_0 p'$ in 1/m² |
| Angle averages | $\langle a\rangle$ is the mean over the grid; the $4\pi^2$ and the common $\psi'$ cancel between $F$ and $G$ |
| Eigenvectors | $F y = \omega^2 G y$ with $y^\dagger G y = 1$; $y_{kj}$ is the amplitude of harmonic $j$ at node $k$ (potential in V, up to a free complex phase) |

{func}`fax.surface_grid` samples these on a uniform $(\theta,\zeta)$ grid
covering the whole torus. Angle derivatives are exact (term by term);
radial derivatives of the harmonics use second-order central differences in
$s$, one-sided at the first and last surface. The position
$\mathbf r = (R\cos\phi, R\sin\phi, Z)$ with $\phi = \zeta + p$ gives the
covariant basis $\mathbf r_i = \partial_i\mathbf r$ for $u^i = (s,\theta,\zeta)$,

$$
g_{ij} = \mathbf r_i\cdot\mathbf r_j,\qquad
g^{ij} = \nabla u^i\cdot\nabla u^j = (g_{ij})^{-1},\qquad
\sqrt g = \mathbf r_s\cdot\mathbf r_\theta\times\mathbf r_\zeta ,
$$

and $|\nabla s|^2 = g^{ss}$. The grid also returns
$\partial_s|B|$, $\partial_\theta|B|$ and $\partial_\zeta|B|$. Two Jacobians
appear below: the continuum uses the Boozer one stored in the file
($\sqrt g_B$, `SurfaceGrid.jacobian`), the global operator the embedding one
($\sqrt g$, `SurfaceGrid.sqrt_g`); they agree up to the truncation of the
Boozer spectrum.

**Radial labels.** `boozmn` files store half-mesh rows `j = jlist - 1`, so
{func}`fax.read_boozmn` sets $s = (j_\mathrm{list} - 1.5)/(n_s - 1)$.
{meth}`fax.Geometry.full_mesh` averages neighbouring half-mesh surfaces onto
the full mesh, as STELLGAP and AE3D do.

**Analytic torus.** {func}`fax.circular_torus` is a concentric circular
tokamak to first order in $\varepsilon = \epsilon_B\sqrt s$:
$|B| = B_0(1 - \varepsilon\cos\theta)$, $G = B_0R_0$, $I = 0$,
$\sqrt g_B = G/B^2 \approx (G/B_0^2)(1 + 2\varepsilon\cos\theta)$, and circles
of radius $r = a\sqrt s$ in the geometric angle $\theta + \varepsilon\sin\theta$
(the $m = 2$ terms of $R$ and $Z$). With $\epsilon_B = a/R_0$ this is the ad hoc
equilibrium of Könies et al. (2018); with $\epsilon_B = 0$ the field is uniform.

## 2. The shear-Alfvén continuum

At low $\beta$, for incompressible shear-Alfvén perturbations, the radial
displacement $E$ of a continuum mode on one surface obeys (Spong, Sanchez &
Weller 2003, eq. 5; Paul et al. 2025, eq. 2.3)

$$
\mu_0\rho\,\omega^2\frac{|\nabla s|^2}{B^2}E
+ \mathbf B\cdot\nabla\!\left(\frac{|\nabla s|^2}{B^2}\,\mathbf B\cdot\nabla E\right) = 0 .
$$ (continuum-ode)

Write $E = \sum_j x_j e_j$ with $e_j = e^{i(m_j\theta - n_j\zeta)}$. By
{eq}`bdotgrad`, $\mathbf B\cdot\nabla e_j = i\,k_{\parallel j} B\, e_j$ with

$$
k_{\parallel j} = \frac{(\iota m_j - n_j)\,B}{G + \iota I}.
$$

Multiply {eq}`continuum-ode` by $\sqrt g_B\,e_i^*$, average over the surface,
$\langle a\rangle = (4\pi^2)^{-1}\oint\oint a\,d\theta\,d\zeta$, and integrate
the second term by parts (the surface operator $\sqrt g_B\,\mathbf B\cdot\nabla$
is anti-self-adjoint, since $\nabla\cdot\mathbf B = 0$). This gives the
generalized Hermitian eigenproblem $K x = \omega^2 M x$ with

$$
K_{ij} = \left\langle \frac{|\sqrt g_B|\,|\nabla s|^2}{\mu_0}\,
         k_{\parallel i}k_{\parallel j}\,e_i^*e_j\right\rangle ,\qquad
M_{ij} = \left\langle \frac{\rho\,|\sqrt g_B|\,|\nabla s|^2}{B^2}\,e_i^*e_j\right\rangle .
$$ (continuum-galerkin)

This is the STELLGAP weighting: the metric factor $|\nabla s|^2/B^2$ and the
Jacobian enter both matrices, and $K$ carries $B^2 k_\parallel^2/\mu_0$
(Spong et al. 2003). {func}`fax.continuum_matrices` evaluates the averages on
the uniform grid (default 32 × 32), which is exact for band-limited
integrands (a test checks 32 × 32 against 64 × 64 to 1e-9), and symmetrizes
both matrices. {func}`fax.continuum` solves each surface with a Cholesky
reduction, $L = \mathrm{chol}(M)$, $L^{-1}KL^{-\dagger}y = \omega^2y$, and
returns $f = \sqrt{\max(\omega^2,0)}/2\pi$ sorted on each surface.

**Uniform field.** For $\epsilon_B = 0$ the matrices are diagonal and
$\omega = |n - m\iota|\,v_A/R_0$, $v_A = B_0/\sqrt{\mu_0\rho}$, exactly.

**Mode families.** In a stellarator with $N_{fp}$ periods, $n$ couples only to
$n + kN_{fp}$, so a family is a list `n = [n0, n0 + Nfp, n0 - Nfp, ...]` in
{func}`fax.mode_family`. Paul et al. (2025) normalize by
$\omega_A = \langle B^2\rangle/((G + \iota I)\sqrt{\mu_0\rho})$.

## 3. Gap formulas

Two cylindrical branches $(m, n)$ and $(m + \Delta m, n + \Delta n N_{fp})$,
$\omega = |n - m\iota|v_A/R$, cross where
$n - m\iota = -(n + \Delta n N_{fp} - (m + \Delta m)\iota)$, i.e.

$$
\iota = \frac{2n + \Delta n\,N_{fp}}{2m + \Delta m},\qquad
\frac{\omega R}{v_A} = \frac{|\Delta m\,n - \Delta n\,m\,N_{fp}|}{2m + \Delta m}.
$$ (gap-location)

The Fourier component of the coupling coefficient with the helicity
$(\Delta m, \Delta n N_{fp})$ splits the crossing into a gap:

| Gap | $(\Delta m, \Delta n)$ | coupling | example |
|---|---|---|---|
| TAE | $(1, 0)$ | toroidicity $\cos\theta$ | $\iota = 2n/(2m+1)$, $\omega R/v_A = n/(2m+1)$ |
| EAE | $(2, 0)$ | ellipticity $\cos2\theta$ | $\iota = n/(m+1)$ |
| MAE | $(0, \neq 0)$ | mirror $\cos N_{fp}\zeta$ | $\iota = (2n + \Delta n N_{fp})/2m$ |
| HAE | both $\neq 0$ | helical $\cos(\theta - N_{fp}\zeta)$ | |

To first order the gap is centred on the crossing frequency with full width
$2\omega_0|\epsilon_{\Delta m,\Delta n}|$, where $\epsilon$ is the
complex Fourier coefficient of the relative coupling coefficient
(half its cosine amplitude) (Paul et al. 2025,
eqs. 4.5 and 4.8; Kolesnichenko et al. 2001, eq. 38). {func}`fax.gap_location`
returns $(\iota, \omega R/v_A)$ of {eq}`gap-location`; {func}`fax.gaps` returns
the centre and width of the widest gap between adjacent sorted branches on
each surface (any family, any gap type).

## 4. Global eigenmodes: the AE3D weak form

AE3D (Spong, D'Azevedo & Todo 2010, eqs. 3–7) writes the reduced ideal-MHD
shear-Alfvén equation for the electrostatic potential $\phi$ in variational
form. With $u = \mathbf B\cdot\nabla\phi/B$ and a test function $\tilde\phi$,

$$
\omega^2\int \frac{\mu_0\rho}{B^2}\,\nabla\tilde\phi^*\cdot\nabla\phi\,dV
= \int \nabla\tilde u^*\cdot\nabla u\,dV
\;[\,+\,K + P\,].
$$ (weak-form)

The left side is the inertia (kinetic energy of the $E\times B$ flow), the
right side the field-line bending energy. The integrals use the full
contravariant metric, so both sides are coordinate-free. The bracket is
optional (`finite_beta=True`: restricted, experimental, off by default; see below).

(finite-beta)=
### Finite-$\beta$ terms (restricted, experimental)

`finite_beta=True` adds the $J_\parallel$ (kink) and pressure-curvature terms of the
ideal-MHD energy for the shear-Alfvén ($E\times B$) displacement. It is a
restricted, experimental option, off by default: there is no sound or
compressional coupling and no full-MHD claim, and $\sigma$, $p'$ and $B_s$ are
reconstructed from $G$, $I$ and $|B|$ alone (below).

**Energy.** The ideal-MHD energy for a displacement $\boldsymbol\xi_\perp$ is
(Freidberg, with $\mathbf Q = \nabla\times(\boldsymbol\xi\times\mathbf B)$,
$\mathbf b = \mathbf B/B$, $\boldsymbol\kappa = \mathbf b\cdot\nabla\mathbf b$)

$$
\delta W = \tfrac12\!\int\!\Big[\frac{|\mathbf Q_\perp|^2}{\mu_0}
 + \frac{B^2}{\mu_0}\big|\nabla\!\cdot\!\boldsymbol\xi_\perp + 2\boldsymbol\xi_\perp\!\cdot\!\boldsymbol\kappa\big|^2
 + \gamma p|\nabla\!\cdot\!\boldsymbol\xi|^2
 - 2(\boldsymbol\xi_\perp\!\cdot\!\nabla p)(\boldsymbol\kappa\!\cdot\!\boldsymbol\xi_\perp^*)
 - J_\parallel(\boldsymbol\xi_\perp^*\!\times\!\mathbf b)\!\cdot\!\mathbf Q_\perp\Big]dV .
$$

Take $\boldsymbol\xi_\perp = (i/\omega)\,\mathbf b\times\nabla\phi/B$ and $E_\parallel = 0$, so
$A_\parallel = u/i\omega$ with $u = \mathbf b\cdot\nabla\phi$ and
$\mathbf Q = \nabla A_\parallel\times\mathbf b + A_\parallel\nabla\times\mathbf b$. For any unit field
$\nabla\times\mathbf b = \sigma\mathbf b + \mathbf b\times\boldsymbol\kappa$, where
$\sigma = \mathbf b\cdot\nabla\times\mathbf b = \mu_0J_\parallel/B$ in an equilibrium, so
$\mathbf Q_\perp = -\mathbf b\times(\nabla_\perp A_\parallel - \boldsymbol\kappa A_\parallel)$. Multiplying by
$2\mu_0\omega^2$ and subtracting the kinetic energy,

$$
\omega^2\!\int\!\frac{\mu_0\rho}{B^2}|\nabla_\perp\phi|^2 =
\int\Big[\,|\nabla_\perp u - \boldsymbol\kappa u|^2
 + \sigma\,\mathbf b\cdot\big(\nabla\phi^*\times(\nabla u - \boldsymbol\kappa u)\big)
 - 2\mu_0\frac{(\mathbf b\times\nabla\phi\cdot\nabla p)(\mathbf b\times\nabla\phi^*\cdot\boldsymbol\kappa)}{B^2}
 + \Big|\sigma u - \frac{\mu_0(\mathbf b\times\nabla p)\cdot\nabla\phi}{B^2}\Big|^2\Big]dV ,
$$ (energy-full)

the last term being the compressional energy
($\nabla\cdot\boldsymbol\xi_\perp + 2\boldsymbol\xi_\perp\cdot\boldsymbol\kappa = (i/\omega)[\sigma u - \mu_0(\mathbf b\times\nabla p)\cdot\nabla\phi/B^2]/B$).

**Retained order.** With $\delta = k_\parallel/k_\perp \sim \sigma/k_\perp \sim \kappa/k_\perp \ll 1$ (the
$k_\parallel/k_\perp$ ordering of Kruger, Hegna & Callen 1998, used by AE3D) the leading
terms are $O(\delta^2)$ relative to $k_\perp^4|\phi|^2$: the bending $|\nabla u|^2$, the kink
term, the inertia and the pressure term (at the ballooning ordering $\mu_0p'\kappa\sim B^2k_\parallel^2$).
Next come the $\boldsymbol\kappa$ terms ($O(\delta^3)$), then $\sigma^2|u|^2$, $\kappa^2|u|^2$, the parallel part of $\nabla u$ and the
compressional energy ($O(\delta^4)$). FAX keeps the first group and drops the others, with the
sound and fast-wave coupling.

**Direct and $\nabla\sigma$ forms.** AE3D's paper (eq. 4, written with $\nabla\zeta$ for $\mathbf b$) has the current term
$-\int u\,\mathbf b\cdot(\nabla\tilde\phi^*\times\nabla\sigma)\,dV$; FAX assembles the direct form
$K = \int\sigma\,\mathbf b\cdot(\nabla\tilde\phi^*\times\nabla u)\,dV$ (the source of AE3D version 31 carries
$J_\parallel/B$ itself, with symmetrized derivative terms, not $\nabla\sigma$). They are not equal. Integrating by parts,

$$
\int\sigma\,\mathbf b\cdot(\nabla\tilde\phi^*\times\nabla u)\,dV
= -\!\int u\,\mathbf b\cdot(\nabla\tilde\phi^*\times\nabla\sigma)\,dV
  - \!\int u\sigma\,(\nabla\times\mathbf b)\cdot\nabla\tilde\phi^*\,dV
  + \oint u\sigma\,(\mathbf b\times\nabla\tilde\phi^*)\cdot d\mathbf S ,
$$ (kink-forms)

where $(\nabla\times\mathbf b)\cdot\nabla\tilde\phi^* = \sigma\tilde u^* + (\mathbf b\times\boldsymbol\kappa)\cdot\nabla\tilde\phi^*$. The omitted term
is not proportional to $\beta$: for the force-free field $\mathbf b = (\cos z, \sin z, 0)$, $\sigma = -1$, $\phi = \sin x$
the direct average is $-1/4$, the $\nabla\sigma$ form gives 0 and the omitted term is $-1/4$
(`test_kink_forms_differ_by_the_curl_b_term`; there $k_\parallel\sim k_\perp\sim\sigma$, so the ordering fails, not the identity).
Boundary terms: at the edge $\tilde\phi = 0$ makes $\nabla\tilde\phi^*\parallel\nabla s$ and
$(\mathbf b\times\nabla s)\cdot\nabla s = 0$; at the axis the surface shrinks while the $m\ne0$ harmonics vanish and the
integrand stays bounded, so the surface integral is zero.

**Restoring $A_\parallel\nabla\times\mathbf b$.** The complete $\mathbf Q_\perp$ changes the bending and kink terms together, and
by {eq}`kink-forms` and $\mathbf b\cdot(\nabla\tilde\phi^*\times\boldsymbol\kappa) = \boldsymbol\kappa\cdot(\mathbf b\times\nabla\tilde\phi^*)$

$$
\int\sigma\,\mathbf b\cdot\big(\nabla\tilde\phi^*\times(\nabla u - \boldsymbol\kappa u)\big)dV
= -\!\int u\,\mathbf b\cdot(\nabla\tilde\phi^*\times\nabla\sigma)\,dV - \!\int\sigma^2u\tilde u^*\,dV ,
$$ (kink-full)

exactly: the $\boldsymbol\kappa$ cross terms cancel (`test_kink_forms_differ_by_the_curl_b_term`, random periodic field,
1e-6), so AE3D's $\nabla\sigma$ form is the complete kink term up to $\sigma^2|u|^2$, which the compressional energy
in {eq}`energy-full` cancels at $p' = 0$. The bending term gains $-\int\boldsymbol\kappa\cdot\nabla|u|^2dV + \int\kappa^2|u|^2dV$.

**Verdict.** At the retained order $O(\delta^2)$ the direct form is the energy-principle term, and it agrees with the
$\nabla\sigma$ form; the two differ at $O(\delta^3)$, where the $\boldsymbol\kappa$ corrections of the bending term enter too. Adding
only one of them would be inconsistent, so FAX keeps the direct form and no correction. On the CTOK potential of
`test_finite_beta_terms_match_real_space_quadrature` ($n = 2$, $m = 3, 4$; real-space quadrature) the bending energy is 8.34,
$K = -1.89$, the $\nabla\sigma$ form $-1.83$, the omitted term $-0.061$, the $\boldsymbol\kappa$ correction of the bending term $+0.006$
and $\int\sigma^2|u|^2$ $+0.004$: the next order moves the energy by about 1% of the bending energy.

FAX assembles the Hermitian part of $K$ and $P$ (the energy principle's real part); it does not copy
AE3D's matrix elements. With $\mathbf B = B_s\nabla s + I\nabla\theta + G\nabla\zeta$,
$\mathbf b\cdot(\nabla a\times\nabla c) = \mathbf B_{\rm cov}\cdot(\partial a\times\partial c)/(B\sqrt g)$
on covariant components, so both terms are tensor contractions like the
bending term, with

$$
K = \int\sigma\,\mathbf b\cdot(\nabla\tilde\phi^*\times\nabla u)\,dV,\qquad
P = -2\mu_0\int p'\,\frac{(\mathbf b\times\nabla\phi\cdot\nabla s)\,
(\mathbf b\times\nabla\tilde\phi^*\cdot\boldsymbol\kappa)}{B^2}\,dV,
$$

$\sigma = \mu_0\mathbf J\cdot\mathbf B/B^2$ and $\boldsymbol\kappa = \nabla_\perp(\mu_0p + B^2/2)/B^2$.

The equilibrium quantities come from $G$, $I$ and $|B|$ alone. The radial
force balance in Boozer coordinates,

$$
(\iota\partial_\theta + \partial_\zeta)B_s = G' + \iota I' + \mu_0p'\,\frac{G + \iota I}{B^2},
$$

fixes $\mu_0p'$ by solvability (its surface average) and $B_s$ (the
Pfirsch–Schlüter part) by dividing by $i(\iota m + n)$ in Fourier space,
skipping resonant harmonics. Then
$\sigma = [I(\partial_\zeta B_s - G') - G(\partial_\theta B_s - I')]/(\sqrt g B^2)$.
Both vanish for a vacuum field ($G$, $I$ constant). {func}`fax.eigen.force_balance_residual` compares $B_s$ with
the embedding's $\mathbf B\cdot\mathbf r_s$ and {func}`fax.geometry.jacobian_check` the embedding Jacobian with the Boozer identity. {func}`fax.circular_torus`
has $I = 0$ unless `current=True` ($I = \iota s a^2B_0/R_0$) or a `pressure`
polynomial is given, which also lowers $G$ so that
$G' + \iota I' = -\mu_0p'R_0/B_0$.

### Fourier–finite-element basis

$$
\phi(s,\theta,\zeta) = \sum_{k=1}^{N}\sum_{j=1}^{H} y_{kj}\,f_k(s)\,e_j(\theta,\zeta),
$$

where $f_k$ are linear hat functions on the nodes $0 < s_1 < \dots < s_N < 1$
(the geometry surfaces, with the axis and the edge added as end nodes). Then

$$
\nabla(f e_j) = f'\,\mathbf X_{0j} + f\,\mathbf X_{1j},\qquad
\mathbf X_{0j} = e_j\nabla s,\quad
\mathbf X_{1j} = i e_j\left(m_j\nabla\theta - n_j\nabla\zeta\right).
$$

For the bending term, {eq}`bdotgrad` gives $u_j = c_j f e_j$ with

$$
c_j = \frac{i(\iota m_j - n_j)\,|B|}{G + \iota I},
$$

and therefore $\nabla(c_jfe_j) = f'\,\mathbf W_{0j} + f\,\mathbf W_{1j}$,

$$
\mathbf W_{0j} = c_je_j\nabla s,\qquad
\mathbf W_{1j} = e_j\left[\partial_sc_j\,\nabla s
 + (\partial_\theta c_j + im_jc_j)\nabla\theta
 + (\partial_\zeta c_j - in_jc_j)\nabla\zeta\right],
$$

$$
\partial_sc_j = \frac{i}{G + \iota I}\Big[\iota'\,m_j|B|
 + (\iota m_j - n_j)\Big(\partial_s|B| - |B|\frac{(G + \iota I)'}{G + \iota I}\Big)\Big],\qquad
\partial_{\theta,\zeta}c_j = \frac{i(\iota m_j - n_j)\,\partial_{\theta,\zeta}|B|}{G + \iota I}.
$$ (shear-term)

The $\iota' m_j$ term is the magnetic shear. It is physical: $u$ depends on
$s$ through $\iota(s)$, and a real-space quadrature of $\int|\nabla u|^2dV$
built from $|B|$, $\iota$ and $g_{ij}$ alone agrees with $F$ to 0.3% with this
term and is off by 9–20% without it (see {doc}`validation`). Radial
derivatives of $\iota$ and $G + \iota I$ use the same finite differences as
the harmonics.

### Coefficient blocks and element matrices

For a weight $w$ and vectors $\mathbf V_0, \mathbf V_1$ (either
$\mathbf X$ with $w = \mu_0\rho|\sqrt g|/B^2$, or $\mathbf W$ with
$w = |\sqrt g|$), each surface gives three $H\times H$ blocks

$$
A_{ij} = \langle w\,\mathbf V_{0i}^*\cdot\mathbf V_{0j}\rangle,\quad
B_{ij} = \langle w\,\mathbf V_{0i}^*\cdot\mathbf V_{1j}\rangle,\quad
D_{ij} = \langle w\,\mathbf V_{1i}^*\cdot\mathbf V_{1j}\rangle ,
$$

with $\mathbf a^*\cdot\mathbf b = a_i^*g^{ij}b_j$ on the
$(\nabla s,\nabla\theta,\nabla\zeta)$ components. As in AE3D, each block is
averaged over a radial element (the mean of its two end surfaces; the axis and
edge elements reuse the first and last surface). On an element of width $h$
with left and right hat functions ($f_L' = -1/h$, $f_R' = 1/h$,
$\int f_L^2 = \int f_R^2 = h/3$, $\int f_Lf_R = h/6$,
$\int f_L'f_L = -\tfrac12$, ...), exact integration gives

$$
\begin{aligned}
E_{LL} &= \frac{A}{h} - \frac{B + B^\dagger}{2} + \frac{Dh}{3}, &
E_{RR} &= \frac{A}{h} + \frac{B + B^\dagger}{2} + \frac{Dh}{3},\\
E_{LR} &= -\frac{A}{h} - \frac{B - B^\dagger}{2} + \frac{Dh}{6}, &
E_{RL} &= -\frac{A}{h} + \frac{B - B^\dagger}{2} + \frac{Dh}{6}.
\end{aligned}
$$

Summing elements gives the Galerkin problem

$$
F y = \omega^2 G y ,
$$ (global-problem)

with $F$ (bending) and $G$ (inertia) Hermitian and block tridiagonal: $N$
diagonal blocks $E_{RR}^{(k)} + E_{LL}^{(k+1)}$ and $N - 1$ off-diagonal
blocks $E_{LR}$ ({func}`fax.global_matrices`, which returns the two
`Blocks`). The angle averages carry no $4\pi^2$, which cancels in
{eq}`global-problem`.

### Boundary conditions

- **Edge:** $\phi = 0$ at $s = 1$ (the edge node is dropped).
- **Axis:** regularity $\phi_m \sim s^{|m|/2}$ is imposed in AE3D's form:
  harmonics with $m \neq 0$ vanish on the axis (the axis node is dropped),
  and $m = 0$ harmonics are flat between the axis and the first surface,
  $\phi(0) = \phi(s_1)$. With $P = \mathrm{diag}(m_j = 0)$ this adds
  $E_{RL}P + PE_{LR} + PE_{LL}P$ of the axis element to the first diagonal
  block.

The singular $f'f'$ blocks $A/h$ reproduce the local continuum of
{eq}`continuum-galerkin`: where $\omega$ meets a continuum branch the
solution develops a jump in $\phi'$ localized to one cell, so the discrete
spectrum fills the continuum with modes of width $\propto 1/N$, and true gap
modes appear as isolated eigenvalues.

## 5. Block-tridiagonal shift-invert solver

The modes of interest lie in the interior of the spectrum, near a target
$\sigma = (2\pi f_\mathrm{target})^2$. {func}`fax.eigenmodes` with
`target_hz` hands the Hermitian block-tridiagonal pencil to
{func}`solvax.generalized_eigenpairs` (SOLVAX owns the numerics;
`fax.eigen.targeted_eigenpairs` is the thin wrapper):

1. Factor $A = F - \sigma G$ by block LU (the Thomas algorithm with
   $H\times H$ blocks), at cost $O(NH^3)$ instead of $O((NH)^3)$. A probe solve
   checks every factorization; if a Schur pivot has broken down (for example
   $\begin{pmatrix}0&1\\1&0\end{pmatrix}$), a dense pivoted LU takes over for
   $NH \le 2048$ and `Status.fallback` is set.
2. Iterate a block of $p = \min(2\,\mathrm{count} + 8, NH)$ vectors,
   $Y \leftarrow \mathrm{qr}\big((F - \sigma G)^{-1} G\,Y\big)$, with Rayleigh–Ritz on
   $(F, G)$ each step, until the `count` values nearest $\sigma$ and the nearest
   excluded one reach the relative residual `tol` (default 1e-10) or `iters`
   (default 100) is exhausted. Eigenvectors are $G$-normalized.

{class}`fax.eigen.Status` reports the worst residual, the $G$-orthogonality, the
iterations, `converged` and the selection `gap` (distance to the target of the
nearest excluded eigenvalue minus that of the farthest selected one; 0 means the
selection splits a tied set). Without a target, `eigenmodes` assembles the dense
matrices and solves all $NH$ modes (SOLVAX has no all-modes dense route, so
`eigenvalue_mean` stays dense). A test checks both solvers agree to 1e-10. Timings of
both solvers at equal accuracy are in {ref}`performance <performance>`.

## 6. Eigenvalue derivatives (Hellmann–Feynman)

For $Fy = \lambda Gy$ with Hermitian $F$, positive-definite $G$ and
$y^\dagger Gy = 1$, differentiate and multiply by $y^\dagger$; the terms in
$\delta y$ cancel by symmetry, leaving

$$
\delta\lambda = y^\dagger\left(\delta F - \lambda\,\delta G\right)y .
$$ (hellmann-feynman)

The derivative is a `custom_jvp`, so none passes through the iteration or the
dense `eigh`:

- `fax.eigen.targeted_eigenvalues` (used by {func}`fax.frequency_hz` with a
  target) applies {eq}`hellmann-feynman` to the SOLVAX pairs. SOLVAX also
  differentiates the eigenvectors (a bordered solve, first order, phase-gauge
  tangent). FAX multiplies every such tangent by NaN unless the solve
  `converged` and `gap > 0`, in forward and reverse mode, so an unconverged
  or tie-splitting solve is rejected rather than silently used;
- {func}`fax.eigenvalue_mean` applies it to dense eigenvalues
  `start <= i < stop` and returns the **mean**,
  $\bar\lambda = \frac1k\sum_{i}\lambda_i$, whose derivative
  $\frac1k\sum_i y_i^\dagger(\delta F - \lambda_i\delta G)y_i$ is the trace of
  the perturbation projected on the cluster. It stays smooth when
  eigenvalues inside the cluster cross or are degenerate (where single
  eigenvalues have kinks and undefined eigenvectors), as long as the cluster
  is separated from the rest of the spectrum.

{func}`fax.frequency_hz` returns $\sqrt{\bar\lambda}/2\pi$: the frequency of
the mean $\omega^2$ of the selected eigenvalues (for one eigenvalue, its
frequency). Derivatives of $F$ and $G$ themselves (with respect to geometry,
$\iota$, density) are ordinary JAX derivatives of the assembly.

(vmex-gradients)=
## 7. vmex boundary gradients

{func}`fax.prepare_design` and {func}`fax.design_geometry` chain

$$
\text{boundary offsets } p \;\to\; x^*(p) \;\to\; \text{Boozer harmonics} \;\to\; (F, G) \;\to\; \omega^2 .
$$

- **Equilibrium.** vmex's `implicit.solve_implicit` returns the converged
  VMEC state $x^*$ with $R(x^*, p) = 0$ (the discrete force balance). Its
  derivative uses the implicit-function theorem,
  $\partial x^*/\partial p = -(\partial_xR)^{-1}\partial_pR$, applied in reverse
  mode as one adjoint solve per scalar objective, so the gradient is that of
  the converged equilibrium, not of a truncated iteration (Antonsen, Paul &
  Landreman 2019 for the shape-gradient idea).
- **Parameters.** Names `rc|rs|zc|zs` + `m` + `n` select `RBC`, `RBS`, `ZBC`,
  `ZBS` entries; `"zs01"` is `ZBS(n=1, m=0)`. Offsets are added to the input's
  coefficients.
- **Boozer transform.** `booz_xform_jax_impl` is traced on the half-mesh
  rows $j \ge 2$, $s = (j - 1/2)/(n_s - 1)$ (row 1 waits for a vmex fix of its
  $R$, $Z$ tables).
- **Radial grid.** {meth}`fax.geometry.Geometry.interpolate` maps that geometry,
  linearly in $s$ and extrapolated to $s = 0, 1$ (harmonics with $m > 0$ follow
  $(s/s_0)^{m/2}$ below the first surface), onto `nodes` uniform FE
  nodes $k/(\text{nodes} + 1)$. The radial domain is always $[0, 1]$, and the
  FE grid refines independently of $n_s$ and `mboz`.
- **Spectrum.** {func}`fax.geometry_frequencies` returns the lowest
  frequencies (dense) or those nearest `target_hz` (shift-invert). Both carry
  the {eq}`hellmann-feynman` JVP, so a smooth objective
  ({func}`fax.window_risk`, {func}`fax.soft_max`) of {func}`fax.design_frequencies`
  is differentiable from the boundary offsets.
- **Gauge.** VMEC's released $m = 1$ constraint combination is an angle gauge.
  The adjoint holds it fixed, while a re-solve lets it drift along its own
  path. Re-solve finite differences therefore differ from the adjoint by the
  response to that drift: 4.4e-4 for $\iota$ wrt `zs01`, about 3e-5 for the
  global frequencies wrt `rc01`, `zs01`, `rc10`, and 2.7e-3 wrt the $m = 1$
  coefficient `zs11`. Adding the drift response closes the $\iota$ gap to 3e-9.

`jax.grad` of any scalar of {func}`fax.design_frequencies` therefore costs one
forward solve plus one adjoint equilibrium solve.

## 8. Perturbative fast-ion drive

The growth rate of a fixed eigenmode is the power the fast ions give to it
over twice its energy (Chen & Zonca 2016, eq. 3.22; NOVA-K and CAS3D-K form),

$$
\gamma_h = -\frac{\operatorname{Im}\delta W_h}{2\omega\,\delta K} = -\frac{P}{2W},\qquad
W = \int\frac{\rho\,|\nabla\phi|^2}{2B^2}\,dV = \frac{2\pi^2}{\mu_0}\,y^\dagger G\,y ,
$$ (gamma-power)

with $G$ the inertia matrix of {func}`fax.global_matrices` (angle-averaged, so
$\int dV = 4\pi^2\int ds\,\langle|\sqrt g|\,\cdot\rangle$) and $y$ the nodal
values of the mode. {func}`fax.growth_rate` evaluates $P$ on the surfaces of a
{class}`fax.Geometry`, for zero orbit width (ZOW), zero Larmor radius and
passing guiding centres, so tokamaks and stellarators share one code path.

**Mode and drift.** $\phi = \sum_h\phi_h(s)e^{i(m_h\theta - n_h\zeta - \omega t)}$,
the convention of {mod}`fax.eigen`. At low $\beta$ the curvature plus
$\nabla B$ drift is
$\mathbf v_d = \frac{m v^2(1 + \xi^2)}{2ZeB^3}\,\mathbf B\times\nabla B$, with
$\xi = v_\parallel/v$. With the Boozer covariant components $B_\theta = I$,
$B_\zeta = G$ ($B_s$ dropped) and the embedding Jacobian $\sqrt g$ of
$(s, \theta, \zeta)$,
$(\mathbf B\times\nabla B)^i = \epsilon^{ijk}B_j\partial_kB/\sqrt g$:

$$
\mathbf v_d\cdot\nabla s \propto \frac{I\partial_\zeta B - G\partial_\theta B}{\sqrt g},\qquad
\mathbf v_d\cdot\nabla\theta \propto \frac{G\,\partial_sB}{\sqrt g},\qquad
\mathbf v_d\cdot\nabla\zeta \propto -\frac{I\,\partial_sB}{\sqrt g},
$$

so the work $Ze\,\mathbf v_d\cdot\delta\mathbf E = -Ze\,\mathbf v_d\cdot\nabla\phi$
of harmonic $h$ is $-m v^2$ times

$$
w_h = \frac{1 + \xi^2}{2B^3\sqrt g}\Big[(I\partial_\zeta B - G\partial_\theta B)\,\phi_h'
  + i\,\partial_sB\,(Gm_h + In_h)\,\phi_h\Big]e^{i(m_h\theta - n_h\zeta)} .
$$

The binormal terms ($\partial_sB$) are the $\cos\theta\,\partial_\theta/r$
part of the large-aspect-ratio vertical drift. Only the overall sign of
$\sqrt g$ is convention, and it cancels in $|w|^2$.

**Orbits and resonances.** A passing guiding centre of pitch
$\lambda = \mu/E \in [0, 1/B_\max)$ follows the field line
$\theta - \iota\zeta = \text{const}$ with
$dt/d\zeta = (G + \iota I)/(v|\xi|B)$, since $B^\zeta = B^2/(G + \iota I)$. Its
mean toroidal transit frequency is
$\omega_\zeta = v/[(G + \iota I)\langle h\rangle]$, $h = 1/(|\xi|B)$ and
$\langle\cdot\rangle$ the Boozer-angle average (the field-line average on an
irrational surface). The angles that advance uniformly in time are
$\vartheta = \theta + \iota\chi$, $\varphi = \zeta + \chi$, with $\chi$ the
periodic solution of the magnetic differential equation

$$
(\iota\partial_\theta + \partial_\zeta)\chi = h/\langle h\rangle - 1,
$$

solved by FFT on the angle grid (`guard` on the small divisors
$\iota k_\theta + k_\zeta$, below). Decomposing the work in $(\vartheta, \varphi)$, the
component $e^{i(j\vartheta - k\varphi)}$ resonates at

$$
\omega = |j\iota - k|\,\omega_\zeta(v, \lambda),\qquad j = m + \ell,\quad k = n + k'N_{fp},
$$ (resonance)

for either sign of $v_\parallel$. The $\ell$ sidebands come from the poloidal
variation of $B$ and of the drift (TAE: $v_\parallel = v_A$, $v_A/3$, ...), the
$k'$ sidebands from the helical harmonics of $B$ (Kolesnichenko et al. 2002;
Heidbrink 2008). `sidebands` (default 3) bounds $|\ell|$ and `helical`
(default 1) bounds $|k'|$; `resolved=True` returns $\gamma$ per $(j, k)$.
The orbit coefficient is

$$
C_{h,jk} = \frac{\langle h\,w_h\,e^{-i(j\vartheta - k\varphi)}\rangle}{\langle h\rangle}.
$$

**Invariants and the drive operator.** The growth rate needs the change of the
distribution's arguments along the perturbed orbit, so the question is which
quantities the unperturbed orbit conserves.

1. *Hamiltonian.* In Boozer coordinates the low-$\beta$ guiding-centre motion
   ($B_s$ dropped) is canonical in $(\theta, \zeta; P_\theta, P_\zeta)$ with
   $P_\theta = Ze\psi + I u$, $P_\zeta = Gu - Ze\psi_p$, $u = mv_\parallel/B$,
   $\psi_p' = \iota\psi'$, and $H_0 = \tfrac12mv_\parallel^2 + \mu B$. $H_0$ depends on
   both angles through $B$, so $P_\zeta$ is conserved only when
   $\partial_\zeta B = 0$. With a helical symmetry $B(\theta - N\zeta)$ the
   conserved momentum is $P_\zeta + NP_\theta$ and nothing is conserved without
   symmetry.
2. *Zero-orbit-width passing orbits* wind the nested tori $\bar\psi = \text{const}$
   quasi-periodically, with uniform angles $(\vartheta, \varphi)$ (above),
   frequencies $\Omega_\vartheta = \iota\Omega_\varphi$ and actions
   $J_\vartheta = Ze\bar\psi + \langle uI\rangle - D\langle u\,\partial_\vartheta\chi\rangle$,
   $D = G + \iota I$: the mean toroidal flux of the orbit up to $O(I/G)$ and
   $O(\varepsilon^2)$ terms that are dropped (they multiply $\omega\,\partial_EJ_\vartheta$
   below, itself $O((j\iota - k)/j)$). $J_\varphi$ is the other action; neither $J_\vartheta$ nor $J_\varphi$ equals $P_\zeta$ in 3-D.
3. *Perturbation.* For an ideal mode ($E_\parallel = 0$, low $\beta$, no $\delta B_\parallel$) the
   perturbed Hamiltonian $H_1 = \sum H_{jk}(J)e^{i(j\vartheta - k\varphi - \omega t)}$ changes the energy at
   $\dot E = \partial_tH_1 = -i\omega H_1$ and the actions at $\dot J_\vartheta = -\partial_\vartheta H_1 = -ijH_1$,
   $\dot J_\varphi = ikH_1$. The harmonic of $\dot E$ is the work $Ze\mathbf v_d\cdot\delta\mathbf E$ computed
   above, $C_{jk} = \sum_hC_{h,jk}$. Hence
   $\dot J_\vartheta = (j/\omega)\dot E$ and $\dot J_\varphi = -(k/\omega)\dot E$.
4. *Distribution.* The ZOW distribution is a function of the invariants,
   $F_0(E, \mu, \bar\psi)$ with the flux label $s$, $\bar\psi = J_\vartheta/Ze$, the choice
   of the benchmark ("a function of the flux label $s$", Könies et al. 2018, section 4.4.1).
   Linearizing along the orbit,
   $\delta\dot F_{jk} + i(j\Omega_\vartheta - k\Omega_\varphi - \omega)\delta F_{jk}$ equals
   minus the change of $F_0$ along the perturbation, which is
   $\dot E\,\partial_EF_0 + \dot{\bar\psi}\,\partial_{\bar\psi}F_0 = \dot E\,Q_{jk}F_0$, with

   $$
   Q_{jk}F = \frac{\partial F}{\partial E} + \frac{j}{\omega Ze\,\psi'}\frac{\partial F}{\partial s}
   $$ (q-operator)

   and $\psi' = \langle\sqrt g\,B^2\rangle/(G + \iota I)$ the toroidal-flux derivative
   (signed with the Jacobian handedness). The Landau prescription
   $\operatorname{Re}[1/(0^+ - i\Delta)] = \pi\delta(\Delta)$ gives the power below.

Consequences, all tested:

- $k$ enters only through the resonance $\omega = (j\iota - k)\Omega_\varphi$, not through $Q$: the
  gradient term follows the *orbit's* poloidal harmonic $j$ and is the same for every
  mode harmonic $h$ contributing to $C_{jk}$. The previous form, $Q_h F = \partial_EF + n_h\partial_sF/(\omega Ze\,\iota\psi')$
  per mode harmonic, used the mode's $n_h$ and is only valid when all $n_h$ are equal (an
  axisymmetric field).
- *Axisymmetry.* $k = n$ and $\omega = (j\iota - n)\Omega_\varphi$, so $j/\psi' = (n/\iota\psi')[1 + \omega/(n\Omega_\varphi)]$ and {eq}`q-operator`
  differs from the tokamak form $\partial_EF - (n/\omega)\partial_{P_\zeta}F$ by $O(\omega/n\Omega_\varphi)$. That is
  the difference between labelling $F$ by the mean flux $\bar\psi$ and by the $P_\zeta$ surface (a shift of order
  $G\rho/\iota$ in the label), the ordering ambiguity of any ZOW drive, and not a numerical error. It lowers the Könies
  TAE ($n = 6$) by 1.3–1.7% and the CTOK TAE ($n = 2$) by 23% (25 keV) and 7% (50 keV).
- *Quasisymmetric helical relabelling* $\theta' = \theta - N\zeta$: $\iota' = \iota - N$, $G' = G + NI$,
  $j' = j$, $k' = k - jN$, $J_\vartheta' = J_\vartheta$, $J_\varphi' = J_\varphi + NJ_\vartheta$, so $\omega$, $C_{jk}$ and
  $\psi'$ are unchanged and so is {eq}`q-operator`. The $n_h/\iota$ form is not (17–20% in the
  test); $P_\zeta + NP_\theta$ is the toroidal momentum of the relabelled tokamak. In a generic stellarator
  there is no such frame, and $j$ with the flux label is the only choice that does not depend on one.
- *Mirrors.* $\theta\to-\theta$ ($\iota\to-\iota$, $m\to-m$, $I\to-I$; $j\to-j$ with $\psi'\to-\psi'$) and
  $\zeta\to-\zeta$ ($\iota\to-\iota$, $n\to-n$, $I\to-I$; $k\to-k$, which swaps co- and counter-passing resonances)
  leave $\gamma$ unchanged to rounding, as do the amplitude and the global phase of $\phi$. A uniform
  distribution has $Q_{jk}F = \partial_EF < 0$: passive.

Not derived: the $O(\omega/n\Omega_\varphi)$ labelling and the dropped $I/G$ and $\varepsilon^2$ terms of $J_\vartheta$, the
reduction from the full guiding-centre phase space to the passing tori (no trapped, no orbit width), and $\delta B_\parallel$.

Linearizing along unperturbed orbits, the phase-space measure
$d^3x\,d^3v = d\psi\,d\theta\,d\zeta\,\sum_\sigma\pi v^2dv\,d\lambda\,(G + \iota I)/(|\xi|B)$
and the delta function integrated exactly over speed at each pitch give

$$
\gamma = \frac{\pi^2\mu_0m_f^2}{2\,y^\dagger Gy}\int|\psi'|\,ds\int d\lambda
  \sum_{jk}\frac{v^6\,Q_{jk}F\,|C_{jk}|^2}{|j\iota - k|\,\hat\omega_\zeta^2}\Big|_{v = v_{jk}},\qquad
C_{jk} = \sum_hC_{h,jk},\quad v_{jk} = \frac{\omega}{|j\iota - k|\,\hat\omega_\zeta},
$$ (gamma-zow)

$\hat\omega_\zeta = \omega_\zeta/v$ ($Q_{jk}F$ with $F$ the distribution times the density). Numerics: Gauss–Legendre (`nodes = 32`) in
$\lambda$ up to $(1 - \texttt{boundary})/B_\max$ (the exact sample maximum, so duplicated
toroidal samples change nothing), the
`ntheta` × `nzeta` Boozer grid of {func}`fax.surface_grid`, central
differences in $s$, the trapezoidal rule in $s$ with $\phi = 0$ at the ends.
There is no artificial broadening, so $\gamma$ is smooth in the geometry, the
mode, $\omega$ and the distribution. In an axisymmetric geometry with
`helical = 0` one toroidal grid point (`nzeta = 1`) is exact.

**Regularizations** are explicit parameters, and $\gamma$ converges as each goes to 0
(`tests/test_drive.py`):

| Parameter | Default | What it regularizes | Convergence of $\gamma$ |
|---|---|---|---|
| `boundary` | 0 | the log-singular trapped–passing edge $\lambda\to1/B_\max$ ($h\sim\xi^{-1}$): pitches stop at $(1-\texttt{boundary})/B_\max$ | first order up to a logarithm: relative error 0.145, 0.018, 1.0e-3, 2e-6 at 0.1, 0.01, 1e-3, 1e-4 (Könies-like, 64 nodes) |
| `nodes` | 32 | Gauss–Legendre pitches | 3.5e-3, 9e-4, 3.5e-4, 2e-5 at 8, 16, 32, 64 |
| `FastIons.birth_width` | 0.02 | the slowing-down birth edge, $\tfrac12\operatorname{erfc}((v-v_b)/(w v_b))$ | $O(w^2)$ (difference ratios 5.0, 4.7 per halving) once $\texttt{nodes}\gtrsim 2/w$ resolves the edge; with too few nodes the edge falls between pitches and the result is silently wrong |
| `guard` | 0.01 | $\kappa\to\kappa/(\kappa^2 + \texttt{guard}^2)$ in the small-divisor inversion of $(\iota\partial_\theta + \partial_\zeta)\chi$; the resonance divisor $|j\iota - k|\hat\omega_\zeta$ has only a $10^{-12}$ floor | no effect in axisymmetry; away from low-order rationals 6e-3, 8e-5, 1e-7 at 0.1, 0.01, 1e-3 |

*Near rational surfaces.* On a surface with $\iota = p/q$ the Boozer-angle average of a
non-axisymmetric field is not the field-line average and the inversion has a true
singularity (islands); `guard` then selects a smoothed value. For $\iota = \tfrac12$ crossed with shear 0.3
($\gamma$ converged to 2e-4 at `guard` 0.01, 1e-5 at 1e-3) the effect is small, but with $\iota$ within
$10^{-3}$ of $\tfrac12$ over the whole mode the error against `guard = 1e-6` is 1–3% and non-monotone down to
`guard` $10^{-3}$ (2e-3 at $10^{-4}$): the drive of a mode sitting on a low-order rational of a 3-D equilibrium is
sensitive to this regularization at the percent level, and `guard` should be refined there.

**Distributions.** {class}`fax.FastIons` is isotropic: a Maxwellian
($F \propto e^{-E/T}$, so $\partial_EF = -F/T$) or a slowing-down
$F \propto 1/(v^3 + v_c^3)$ below the birth speed $v_b$, with the birth edge
smoothed by $\tfrac12\operatorname{erfc}((v - v_b)/0.02v_b)$ so it stays
differentiable ($\partial_EF$ includes the smoothed edge). $F$ is normalized to
$\int F\,d^3v = 1$ with that edge: integrating by parts, the constant is
$3/(4\pi\,\mathbb E[\ln(1 + v^3/v_c^3)])$ with $v \sim \mathcal N(v_b, (0.02v_b)^2/2)$,
done with 24-point Gauss–Hermite.
Any other `kind` raises `ValueError`.
The density may be a profile on $s$; with $\iota\psi' > 0$ an
outward-decreasing density drives $n > 0$, and $\partial_EF < 0$ damps.

**Landau damping.** {func}`fax.ion_landau_damping` is the same integral for
a flat thermal Maxwellian of the bulk ions, with the sign flipped so that a
positive number damps (Fu & Van Dam 1989; Betti & Freidberg 1992). Continuum
damping and electron Landau damping are not computed; subtract them as an
external $\gamma_d$, $\gamma = \gamma_h - \gamma_d$.

**Scope of the number.** Passing-only and perturbative: against the CAS3D-K zero-orbit-width curve of Könies et al. (2018) the
Könies-case drive is 0.58–0.60 (82–191 keV), plausibly the missing trapped share (not established; the curve's content is not
stated), and $\gamma/\omega \le 4\%$ only up to 200 keV ({doc}`validation`).

**Not included.** Trapped and precessing particles (their resonance is
$\omega = k\langle\omega_d\rangle_b + p\,\omega_b$ with bounce averages; not
yet implemented), finite orbit width, $\delta B_\parallel$ and the covariant
$B_s$ in the drift.

## 9. Smooth objectives

Eigenvalues and gaps are not smooth (crossings, `max`), so the objectives are:

| Function | Definition |
|---|---|
| {func}`fax.window_risk` | $\tfrac12\,\mathrm{mean}_k\,\sigma_k^2$, $\sigma_k = \mathrm{sig}\big(\frac{f_k - f_\mathrm{lo}}{\Delta}\big)\,\mathrm{sig}\big(\frac{f_\mathrm{hi} - f_k}{\Delta}\big)$: a soft count of modes inside a frequency window |
| {func}`fax.soft_max` | $\beta^{-1}\log\sum_k e^{\beta x_k}$ (log-sum-exp), e.g. of growth rates |
| {func}`fax.gap_closure` | $\mathrm{mean}_s\,\mathrm{softplus}(-w_s/\Delta)$ with $w_s$ the widest gap on surface $s$: grows as gaps close |

{func}`fax.drive_risk` is the stability target
$\mathrm{softplus}(\mathrm{LSE}(\gamma_k/\gamma_s) - \gamma_\mathrm{th}/\gamma_s)$ (scale
$\gamma_s$ = `scale`, default 10³ s⁻¹) of the growth rates that
{func}`fax.geometry_growth_rates` returns for the modes nearest `target_hz`
(shift-invert eigenvectors, differentiated implicitly by SOLVAX, then
{func}`fax.growth_rate`). Cluster means
({func}`fax.eigenvalue_mean`, `start:stop` in {func}`fax.frequency_hz`)
keep objectives smooth through crossings inside the cluster.

(continuum-objective)=
## 10. Continuum-gap objective

Gaps open where the flux compression $|\nabla\psi|^2$ varies on a surface
(sections 2–3). Paul et al. (2025, eq. 4.28) write the coupling as

$$
\epsilon = \frac{|\nabla s|^2}{\langle|\nabla s|^2\rangle} - 1
         = \sum_{\delta m,\,n} \epsilon_{\delta m,n}\, e^{i(\delta m\,\theta - n\zeta)},
$$

in Boozer angles, with $\langle\cdot\rangle$ the angle average and $n$ the full
toroidal number ($N_P\,\delta n$ in the paper). {func}`fax.compression_harmonics`
takes $|\nabla s|^2$ from {func}`fax.surface_grid` and returns $\epsilon_{\delta m,n}$
from one 2-D FFT per surface. The harmonic $(\delta m, n)$ opens a gap centred at
$\omega^{(0)}/\omega_A = |\iota\,\delta m - n|/2$ of full (edge to edge) width
$2\omega^{(0)}|\epsilon_{\delta m,n}|$ (eqs. 4.5, 4.16; the conjugate $(-\delta m, -n)$ makes the real amplitude $2|\epsilon|$). Gaps above the
quasisymmetry-helicity gap are the ones passing fast ions near birth energy can
resonate with, so Paul's eq. 7.4 penalizes the squared full widths $(2\omega^{(0)}|\epsilon|/\omega_A)^2 = |\epsilon|^2(\iota\delta m - n)^2$ of those:

$$
f_\mathrm{cont} = \sum_s \sum_{|\iota\delta m - n| > |\iota M - N N_P|}
                  |\epsilon_{\delta m,n}|^2\,(\iota\,\delta m - n)^2,
$$

{func}`fax.continuum_cost`, with helicity $(M, N)$: $(1, 0)$ for QA and
$(1, \pm1)$ for QH (the sign follows the Boozer $|B|$ spectrum: LP-QH with
$\iota < 0$ has $|B| \sim \cos(\theta + 4\zeta)$, i.e. $(1, -1)$). Each term is
the square of a full gap width, which grows with the gap frequency, so wide
high-frequency gaps dominate. The sum runs over both $\pm(\delta m, n)$. Paul's cut is a hard mask, which
adds or removes a whole term $|\epsilon|^2k^2$ when $\iota$ moves a harmonic across
$|k| = |\iota M - N N_P|$ (LP-QA, helicity $(1, 0)$, $N_P = 2$, at $\iota = 0.42$:
$(\delta m, n) = (4, 2)$ at $\iota = 0.4$, $(3, 2)$ at $\iota = 0.5$), so the cost jumps there.
{func}`fax.continuum_cost` uses the $C^1$ smoothstep

$$
f_\mathrm{cont} = \sum_s \sum_{\delta m,\,n} m(t)\,|\epsilon_{\delta m,n}|^2 k^2,\qquad
m = 3t^2 - 2t^3,\quad t = \mathrm{clip}\Big(\frac{|k| - |\iota M - N N_P|}{\lambda}, 0, 1\Big),
$$

with $k = \iota\,\delta m - n$ and the stated sharpness $\lambda$ = `softness`
(default 0.05, in units of $|k| = 2\omega^{(0)}/\omega_A$). The membership is 0 up to the cut, so the
helicity harmonic itself ($|k|$ equal to the cut) stays out, and exactly 1 from
$\lambda$ above it. The cost and its derivative are continuous in $\iota$; the
limit $\lambda \to 0$ is the paper's hard cost. The full objective of Paul's eq. 7.1,

$$
f = (A - A^*)^2 + f_{QS} + f_\iota + f_\mathrm{cont},\qquad
f_\iota = \sum_s \min(|\iota| - \iota_\mathrm{min}, 0)^2,
$$

takes $A$ and $f_{QS}$ from vmex (`implicit.aspect_ratio`,
`QuasisymmetryRatioResidual.total_state`) on the same implicit state:
{func}`fax.design_state` returns `(state, runtime)` and
{func}`fax.state_geometry` the Boozer geometry, so one adjoint covers every
term. For the global spectrum, {func}`fax.window_risk` of the frequencies of
{func}`fax.geometry_frequencies` moves targeted modes out of a frequency window
instead (section 9).

### Smooth widest gap

{func}`fax.gap_closure` (section 9) replaces the `argmax` over the gaps of a surface by a
softmax-weighted mean. With $w_k$ the widths of the adjacent sorted branches and
$\Delta$ = `scale_hz` (default $10^4$ Hz),

$$
\bar w = \sum_k p_k w_k,\quad p_k = \frac{e^{w_k/\Delta}}{\sum_j e^{w_j/\Delta}},\qquad
\max_k w_k - \Delta\log K \le \bar w \le \max_k w_k,
$$

and the penalty is $\mathrm{mean}_s\,\mathrm{softplus}(-\bar w_s/\Delta)$. The hard widest gap has a
gradient jump where two gaps trade places; $\bar w$ does not. Smoothing only removes the
selection switch: the sorted branches still cross at a degeneracy, so the gradient is first order there.

### Which modes an objective may act on

A frequency window acts on whatever modes lie in it, and a continuum-like mode is not a property of the
geometry that survives refinement: it is a discrete sample of a branch at the radius where the branch
meets its frequency. {func}`fax.classify_modes` therefore decides, for each selected mode, from three
independent tests on a refinement $N \to 2N + 1$ of the radial grid (nested nodes, $L^2$-prolonged):

1. **Refinement.** The relative frequency shift to the best-matching refined mode and their nodal $L^2$ overlap.
   (The $G$ overlap is dominated by the singular $\phi'$ of a resonant tail and is not used.)
2. **Gap membership.** At the radius $s_\mathrm{peak}$ of the largest $|\phi|^2$, the distance
   $\min(f - f_\mathrm{lo}, f_\mathrm{hi} - f)/f$ to the adjacent branches of the local continuum
   ({func}`fax.continuum`). A continuum-like mode peaks where its frequency meets a branch, so the margin is one cell of
   $\mathrm{d}f/\mathrm{d}s$ at most; a gap mode peaks inside a gap.
3. **Localization.** The growth of the derivative-jump participation count $(\sum_i J_i)^2/\sum_i J_i^2$,
   $J_i = \sum_h|\Delta\phi_{i,h}|^2$, under the refinement: about 1 for a mode pinned to a fixed number of cells (the
   mechanism of the uniform-field spectrum, section 4), about 2 for fixed radial width. It is reported, not tested: a
   gap mode with a weak resonant tail (the CTOK TAE) has a jump count that is dominated by the tail.

A mode is isolated when both solves converged, the shift is below `shift_tol` (1%), the overlap above `overlap_tol` (0.9) and the margin above
`margin_tol` (5%). Only the margin separates the classes on the cases below; the refinement test guards against unconverged or
unresolved modes (a continuum mode converges in $L^2$ trivially, with shift below 1% and overlap above 0.95).

{func}`fax.geometry_frequencies`, {func}`fax.design_frequencies` and {func}`fax.geometry_growth_rates` are
**first order** in the geometry (rule 5 of the plan): they return the solver `Status` and are NaN, values and derivatives, unless the solve
converged and its selection does not split a tied set ($\mathrm{gap} > 10^{-8}\max|\omega^2|$). The classification is a separate, non-differentiable
check that the selected modes are the ones the objective is meant to move.

