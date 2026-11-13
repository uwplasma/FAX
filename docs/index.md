# FAX

FAX computes shear-Alfvén spectra of stellarators and differentiates them. Give it a Boozer-coordinate equilibrium,
from a `boozmn` file, a VMEC input or an analytic torus, and it returns

- the shear-Alfvén continuum on every flux surface, with the TAE, EAE, MAE and HAE gaps opened by the Boozer spectrum
  of $|B|$;
- the global Alfvén eigenmodes near a target frequency, from the reduced ideal-MHD weak form of AE3D, solved with
  radial finite elements and a block-tridiagonal shift-invert eigensolver from [SOLVAX](https://github.com/uwplasma/SOLVAX);
- the linear drive of a mode by passing energetic ions in the zero-orbit-width limit, and bulk-ion Landau damping;
- first derivatives of all of these with respect to profiles and, through [VMEX](https://github.com/uwplasma/vmex),
  with respect to the Fourier coefficients of the plasma boundary.

The derivatives are what FAX is for. An optimizer that shapes a stellarator for quasisymmetry can carry an Alfvén
objective in the same loop and follow its gradient.

```{image} _static/workflow.svg
:alt: Boundary to VMEX equilibrium to Boozer spectrum to FAX (continuum, global modes, drive), then an objective whose gradient flows back to the boundary
:width: 100%
```

## Five minutes

```console
pip install alfvex                       # imported as fax
python examples/quickstart.py            # continuum, a global TAE and a gradient on an analytic torus
fax run examples/ctok_continuum.toml --plot
```

The PyPI name is `alfvex` because `fax` is taken by an unrelated package. FAX will be renamed ALFVEX.

```{image} _static/ctok_eigenmodes_continuum.png
:alt: CTOK n = 2 continuum with the six global modes nearest 384 kHz
:width: 62%
:align: center
```

The n = 2 continuum of the CTOK tokamak case with the six global modes nearest 384 kHz.

## Where to read next

| If you want to | Read |
|---|---|
| understand Alfvén eigenmodes and why stellarator design cares | {doc}`physics` |
| install FAX and run the first cases | {doc}`getting-started` |
| work through continua, global modes, drive and optimization | {doc}`tutorials` |
| follow the derivations and the numerics | {doc}`theory` |
| see each check against analytic results and other codes | {doc}`validation` |
| know how FAX relates to STELLGAP, AE3D, FAR3D, FIRM3D and others | {doc}`comparisons` |
| know what each function promises about derivatives | {doc}`api` |
| run the tests, contribute or cite FAX | {doc}`development` |

## What FAX does not do

The continuum can include sound coupling (BAE and BAAE gaps); the global model is low-β reduced ideal MHD without it, so there are no BAE or BAAE eigenmodes. The drive is
perturbative, passing ions only, with zero orbit width and no finite Larmor radius. Continuum damping exists as an
unmerged prototype. {doc}`validation` lists every open item with its numbers.

```{toctree}
:hidden:
:maxdepth: 2

physics
getting-started
tutorials
theory
validation
comparisons
api
development
references
```
