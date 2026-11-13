# Alfvén waves in stellarators

This page is the physical picture behind FAX, for a reader who has not worked with Alfvén eigenmodes before. It uses a few
equations and no derivations. The derivations, with every convention and the function that implements each step, are on
{doc}`theory`.

## A wave carried by the field line

A magnetic field line behaves like a stretched string. Push it sideways and its tension pulls it back, while the plasma
frozen onto it supplies the inertia. The result is the shear-Alfvén wave, which travels along the field at the Alfvén speed

$$
v_A = \frac{B}{\sqrt{\mu_0\rho}},
$$

with $\rho$ the mass density. The wave has almost no compression, so it does not feel the pressure of the plasma much; at low
$\beta$ the field-line bending is the whole restoring force. A wave with wavenumber $k_\parallel$ along $\mathbf B$ therefore
oscillates at

$$
\omega = |k_\parallel|\,v_A .
$$

```{figure} _static/physics_schematic.png
:width: 700px
:alt: Left, three field lines displaced sinusoidally along the field direction. Right, a fast-ion distribution against parallel speed with the resonant speeds v_A/3 and v_A marked.

(a) A shear-Alfvén wave bends field lines; the wavelength along $\mathbf B$ sets the frequency. (b) Fast ions whose parallel
speed matches a resonant speed (section on drive below) exchange energy with the wave.
```

In a torus or a stellarator the field lines wind around the magnetic surface, and a perturbation with poloidal number $m$ and
toroidal number $n$ varies along the line as $e^{i(m\theta - n\zeta)}$ in Boozer angles. Moving along a line advances
$\zeta$ and $\theta = \iota\zeta$ together, so the phase advances at a rate set by $\iota m - n$. In the notation of
{doc}`theory`,

$$
k_\parallel = \frac{(\iota m - n)\,B}{G + \iota I}, \qquad \omega = |n - m\iota|\,\frac{v_A}{R}
$$

for a uniform field of major radius $R$. A harmonic whose $\iota m - n$ vanishes on some surface does not bend the field line
there, so its frequency drops to zero at that surface. This is where low-frequency modes live, and why the rotational
transform profile matters so much.

## The continuum

Each flux surface supports its own family of local oscillations. Their frequencies, collected over all surfaces, form the
shear-Alfvén continuum: for every radius there is a set of allowed $\omega$ rather than a single eigenfrequency. A wave launched
at a given frequency finds the surface where it matches the local branch, drives the plasma there, and its radial structure
collapses into a thin layer. The energy ends up in a narrow region and is dissipated by it. This is continuum damping, and it is why
most global perturbations of a plasma do not ring.

{func}`fax.continuum` returns these frequencies on every surface from the surface-averaged low-$\beta$ equation of Spong et al.
(2003), by solving a small Hermitian eigenproblem per surface ({doc}`theory`, section 2). FAX does not compute the damping
itself; it tells you where the continuum is, and a mode that sits inside it is damped.

## Gaps

In a cylinder, branches with different $(m, n)$ cross freely. A real device couples them. In a tokamak the field strength varies
as $\cos\theta$, which links $m$ to $m + 1$ at the same $n$. Where two coupled branches would cross, they repel, and a gap opens
in the continuum (Kolesnichenko et al. 2001; Paul et al. 2025). The crossing of $(m, n)$ and $(m + \Delta m, n + \Delta n N_{fp})$
sits at

$$
\iota = \frac{2n + \Delta n\,N_{fp}}{2m + \Delta m}, \qquad
\frac{\omega R}{v_A} = \frac{|\Delta m\,n - \Delta n\,m\,N_{fp}|}{2m + \Delta m},
$$

and to first order the gap is centred there with full width $2\omega_0|c|$, where $c$ is the Fourier coefficient of the coupling
for that helicity ({doc}`theory`, section 3; {func}`fax.gap_location` evaluates the location).

```{figure} _static/physics_tae_gap.png
:width: 700px
:alt: Two continuum branches for the (m,n) = (1,1) and (2,1) harmonics against the rotational transform. For a uniform field they cross at iota = 2/3; on a torus with epsilon_B = 0.3 they repel and open a gap.

Two coupled harmonics, $(m, n) = (1, 1)$ and $(2, 1)$, on {func}`fax.circular_torus` with $R_0 = 10$ m, $a = 1$ m, $B_0 = 3$ T
and $\iota$ rising from 0.5 to 0.85 across the radius. The dashed lines are the uniform-field branches, which cross at
$\iota = 2/3$, $\omega R_0/v_A = 1/3$. With toroidicity ($\epsilon_B = 0.3$) the branches open a gap there, between
$\omega R_0/v_A = 0.25$ and $0.39$ in this two-harmonic truncation.
```

Which gap opens depends on which harmonic of the geometry provides the coupling.

| Gap | Helicity $(\Delta m, \Delta n)$ | Provided by |
|---|---|---|
| Toroidal (TAE) | $(1, 0)$ | toroidal curvature, $\cos\theta$ in $B$ |
| Elliptical (EAE) | $(2, 0)$ | elongation of the cross-section, $\cos 2\theta$ |
| Mirror (MAE) | $(0, \Delta n \ne 0)$ | the toroidal variation of $B$ with period $N_{fp}$ |
| Helical (HAE) | both nonzero | helical harmonics such as $\cos(\theta - N_{fp}\zeta)$ |

```{figure} _static/physics_gap_families.png
:width: 700px
:alt: Cylindrical Alfven branches against iota, with markers at the crossings colour-coded by gap type: TAE, EAE, MAE and HAE.

Crossings of cylindrical branches $\omega R/v_A = |n - m\iota|$ for the family $n = -1, 1, 3$ with $N_{fp} = 2$ and
$m = 1$ to 4, positioned by {func}`fax.gap_location`. The marker shape and colour give the helicity of the harmonic that would
open the gap.
```

Because a stellarator has $N_{fp}$ periods, $n$ couples only to $n + kN_{fp}$, so the picture above repeats for every family
and the continuum of a stellarator has far more crossings than that of a tokamak. In a quasisymmetric field $|B|$ depends on
a single combination $M\vartheta - NN_{fp}\varphi$, so only harmonics of that helicity remain to open gaps. Paul et al. (2025)
use this to estimate the gap widths of quasisymmetric configurations and to penalize them in optimization.

## Global modes

A mode with a frequency inside a gap does not meet a continuum branch, so nothing damps it through the continuum. It is a
global Alfvén eigenmode, localized around the gap in radius and made of a few dominant harmonics. The TAE is the best known.
FAX finds such modes with {func}`fax.eigenmodes`, which solves the reduced ideal-MHD weak form of Spong, D'Azevedo and Todo
(2010), the one implemented by AE3D, near a target frequency you supply ({doc}`theory`, section 4).

## Energetic particles drive the modes

Fusion-born alphas, neutral-beam ions and ion-cyclotron-heated ions move at speeds comparable to $v_A$ or above. A passing fast
ion follows a field line with parallel speed $v_\parallel$, and sees the wave as a perturbation oscillating at the frequency
with which it transits the torus. When the two match the particle exchanges energy with the wave. For a passing ion the
resonance condition is

$$
\omega = |j\iota - k|\,\omega_\zeta ,
$$

where $\omega_\zeta$ is the toroidal transit frequency and $(j, k)$ label the Fourier component of the wave seen along the
orbit. For a TAE the lowest resonances sit at $v_\parallel = v_A$, then $v_A/3$, and so on (Heidbrink 2008; Chen and Zonca 2016;
{doc}`theory`, section 8). Where the fast-ion distribution gives energy to the wave rather than taking it, the mode grows.

Growth competes with damping. Thermal ions damp the wave by Landau resonance (Fu and Van Dam 1989; Betti and Freidberg 1992),
electrons do the same, and any mode that overlaps the continuum loses energy to it. A mode is unstable when the drive exceeds the
sum of these losses, so a gap mode that is well isolated from the continuum is the dangerous one.

## Why stellarator design cares

Alfvén eigenmodes can expel fast ions before they have heated the plasma, and the confinement of alphas is one requirement a
reactor stellarator must meet. The Alfvén spectrum depends on the shape of the magnetic field, which is the thing an
optimizer controls. Paul et al. (2025) computed the continuum of quasisymmetric configurations and proposed gap widths and
continuum properties as targets for optimization. Knyazev et al. (2026) studied shear-Alfvén-wave-induced fast-ion transport in
optimized stellarators; FAX uses their precise-QA configuration as one of its independent benchmarks ({doc}`validation`).

Evaluating many candidate boundaries needs the spectrum to be cheap and differentiable. FAX is written in JAX so that the
frequency of a mode, a gap width and the drive of a fixed mode are all differentiable with respect to the VMEC boundary
coefficients, through the equilibrium solve in vmex ({doc}`theory`, sections 6 and 7; {doc}`tutorials`, section 5).

## What FAX computes, and what it does not

FAX computes:

- the low-$\beta$ shear-Alfvén continuum on Boozer surfaces, and the gaps in it;
- global modes of the reduced AE3D-form weak form, with optional restricted finite-$\beta$ terms (experimental);
- the perturbative drive of a given mode by a passing fast-ion population, for zero orbit width and zero Larmor radius;
- first-order derivatives of these quantities with respect to the equilibrium.

The continuum can include sound coupling, which opens the BAE and BAAE gaps (pass `pressure` to {func}`fax.continuum`); the global modes do not, so there are no BAE or BAAE eigenmodes. Continuum damping and electron damping are not computed;
a growth rate must be reduced by an external damping rate. The drive covers passing ions only: trapped and precessing ions,
finite orbit width and finite Larmor radius are absent, and the perturbative drive is valid up to about 200 keV in the Könies
case. A drive computed this way is a way to rank designs, and the number it gives is about 0.6 of the CAS3D-K zero-orbit-width
curve, a ratio that is not yet explained. {doc}`validation` lists every gate and the open items under "Not done".

## Where to go next

{doc}`getting-started` installs the package and runs the first continuum. {doc}`tutorials` walks through eight examples, from
the analytic torus to a boundary gradient and a fast-ion drive. {doc}`theory` derives each equation, and {doc}`references` lists
the papers cited here.
