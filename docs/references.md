# References

Entries give authors, year, title, journal or arXiv number, and a DOI or link. Items without a DOI are those whose DOI could
not be checked. The sections follow the topics of the code.

## Continuum and gaps

- D. A. Spong, R. Sanchez and A. Weller (2003), "Shear Alfvén continua in stellarators", *Phys. Plasmas* **10**, 3217.
  [doi:10.1063/1.1590316](https://doi.org/10.1063/1.1590316). The continuum model implemented in {func}`fax.continuum`.
- Ya. I. Kolesnichenko, V. V. Lutsenko, H. Wobig, Yu. V. Yakovenko and O. P. Fesenyuk (2001), "Alfvén continuum and
  high-frequency eigenmodes in optimized stellarators", *Phys. Plasmas* **8**, 491.
  [doi:10.1063/1.1339228](https://doi.org/10.1063/1.1339228). Stellarator gap locations.
- E. J. Paul, A. Hyder, E. Rodríguez, R. Jorge and A. Knyazev (2025), "The shear Alfvén continuum of quasisymmetric
  stellarators", *J. Plasma Phys.* **91**, E101, arXiv:2503.03711.
  [doi:10.1017/S0022377825100524](https://doi.org/10.1017/S0022377825100524). Gap widths and the harmonic objective behind
  `continuum_cost`.
- A. Hyder, E. J. Paul and A. Knyazev (2026), "The shear Alfvén continuum of optimized stellarators", arXiv:2611.07182.
  [doi:10.48550/arXiv.2611.07182](https://doi.org/10.48550/arXiv.2611.07182).
- H. L. Berk, J. W. Van Dam, Z. Guo and D. M. Lindberg (1992), "Continuum damping of low-n toroidicity-induced shear Alfvén
  eigenmodes", *Phys. Fluids B* **4**, 1806. [doi:10.1063/1.860455](https://doi.org/10.1063/1.860455).
- G. W. Bowden, M. J. Hole and A. Könies (2015), "Calculation of continuum damping of Alfvén eigenmodes in 2D and 3D cases",
  arXiv:1507.02072. [doi:10.48550/arXiv.1507.02072](https://doi.org/10.48550/arXiv.1507.02072). Target of the continuum-damping
  prototype described under "Not done" in {doc}`validation`.
- G. Bowden, A. Könies, M. Hole, N. Gorelenkov and G. Dennis (2014), "Comparison of methods for numerical calculation of
  continuum damping", arXiv:1402.6389. [doi:10.48550/arXiv.1402.6389](https://doi.org/10.48550/arXiv.1402.6389).

## Global eigenmodes

- D. A. Spong, E. D'Azevedo and Y. Todo (2010), "Clustered frequency analysis of shear Alfvén modes in stellarators", *Phys.
  Plasmas* **17**, 022106. [doi:10.1063/1.3313818](https://doi.org/10.1063/1.3313818). The AE3D-form global operator.
- S. E. Kruger, C. C. Hegna and J. D. Callen (1998), "Generalized reduced magnetohydrodynamic equations", *Phys. Plasmas* **5**,
  4169. [doi:10.1063/1.873152](https://doi.org/10.1063/1.873152). Reduced ordering.
- J. P. Freidberg (2014), *Ideal MHD* (Cambridge University Press). The $\delta W$ form used for the finite-$\beta$ terms.
- A. Könies, S. Briguglio, N. Gorelenkov, T. Fehér, M. Isaev, Ph. Lauber, A. Mishchenko, D. A. Spong, Y. Todo, W. A. Cooper,
  R. Hatzky, R. Kleiber, M. Borchardt, G. Vlad, A. Biancalani, A. Bottino and the ITPA EP TG (2018), "Benchmark of
  gyrokinetic, kinetic MHD and gyrofluid codes for the linear calculation of fast particle driven TAE dynamics", *Nucl.
  Fusion* **58**, 126027. [doi:10.1088/1741-4326/aae4e6](https://doi.org/10.1088/1741-4326/aae4e6). The ITPA TAE benchmark.
- A. R. Knyazev, A. Lachmann, A. G. Goodman, A. S. Hyder, M. Czekanski, D. Spong and E. J. Paul (2026), "On shear Alfvén
  wave-induced energetic ion transport in optimized stellarators", arXiv:2603.03118.
  [doi:10.48550/arXiv.2603.03118](https://doi.org/10.48550/arXiv.2603.03118). Source of the physical-potential vorticity form
  of the weak form. The data archive is [doi:10.5281/zenodo.21299907](https://doi.org/10.5281/zenodo.21299907) (CC-BY-4.0): the
  precise-QA fixture of {doc}`validation`, with its provenance in `tests/data/PROVENANCE.md`. The QA and QI STELLGAP/AE3D
  archive is [doi:10.5281/zenodo.21299908](https://doi.org/10.5281/zenodo.21299908).
- A. Mishchenko, A. Könies and R. Hatzky (2011), "Global particle-in-cell simulations of plasma pressure effects on Alfvénic
  modes", *Phys. Plasmas* **18**, 012504. [doi:10.1063/1.3546021](https://doi.org/10.1063/1.3546021).
- J. Varela, D. Spong, L. Garcia, Y. Ghai, J. Ortiz and the FAR3d project collaborators (2024), "Stability optimization of
  energetic particle driven modes in nuclear fusion devices: the FAR3d gyrofluid code", *Front. Phys.* **12**, 1422411.
  [doi:10.3389/fphy.2024.1422411](https://doi.org/10.3389/fphy.2024.1422411).
- STELLGAP and AE3D: the CTOK test problem and outputs in `tests/data`.

## Kinetic drive and damping

- L. Chen and F. Zonca (2016), "Physics of Alfvén waves and energetic particles in burning plasmas", *Rev. Mod. Phys.* **88**,
  015008. [doi:10.1103/RevModPhys.88.015008](https://doi.org/10.1103/RevModPhys.88.015008).
- W. W. Heidbrink (2008), "Basic physics of Alfvén instabilities driven by energetic particles in toroidally confined plasmas",
  *Phys. Plasmas* **15**, 055501. [doi:10.1063/1.2838239](https://doi.org/10.1063/1.2838239).
- G. Y. Fu and J. W. Van Dam (1989), *Phys. Fluids B* **1**, 1949, and R. Betti and J. P. Freidberg (1992), *Phys. Fluids B* **4**,
  1465: the Maxwellian TAE drive and the Landau damping scaling.
- E. J. Paul, A. Knyazev, M. Czekanski, A. Lachmann, A. Hyder, C. Albert and M. Landreman (2026), "FIRM3D: Fast ion reduced
  models in 3D", arXiv:2605.16734. [doi:10.48550/arXiv.2605.16734](https://doi.org/10.48550/arXiv.2605.16734). The planned
  transport counterpart of the drive.

## Equilibrium and Boozer coordinates

- M. Landreman and E. Paul (2022), "Magnetic fields with precise quasisymmetry for plasma confinement", *Phys. Rev. Lett.*
  **128**, 035001. [doi:10.1103/PhysRevLett.128.035001](https://doi.org/10.1103/PhysRevLett.128.035001). The precise-QA example.

## Optimization

- T. Antonsen Jr., E. J. Paul and M. Landreman (2019), "Adjoint approach to calculating shape gradients for
  three-dimensional magnetic confinement equilibria", *J. Plasma Phys.* **85**, 905850207.
  [doi:10.1017/S0022377819000254](https://doi.org/10.1017/S0022377819000254).
- R. Gaur, S. Buller, M. E. Ruth, M. Landreman, I. G. Abel and W. D. Dorland (2023), "An adjoint-based method for optimising
  MHD equilibria against the infinite-n, ideal ballooning mode", *J. Plasma Phys.* **89**, 905890518.
  [doi:10.1017/S0022377823000995](https://doi.org/10.1017/S0022377823000995).
- R. Gaur, S. Patil, P. Gupta, D. Patch and T. Qian (2026), "AGNI: A differentiable MHD stability solver & optimizer for
  magnetic confinement fusion devices", arXiv:2608.01750.
  [doi:10.48550/arXiv.2608.01750](https://doi.org/10.48550/arXiv.2608.01750).

## Numerics

- SOLVAX (shift-invert and generalized eigenpairs, block-tridiagonal solves) on PyPI:
  [pypi.org/project/solvax](https://pypi.org/project/solvax/).
