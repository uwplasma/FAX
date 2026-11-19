# Changelog

## Unreleased

What changes for users of 1.2.0:

- `continuum`, `eigenmodes` and `frequency_hz` solve each listed harmonic on both signs, $(m, n)$ and $(-m, -n)$, split into
  blocks that do not couple ({doc}`theory`, section 2). A self-conjugate family ($N = 0$, or $N = N_{fp}/2$, or $n = 0$ in a
  tokamak) now returns twice as many branches and modes as harmonics: its cosine and sine sectors. Other families are
  unchanged. `parity="single"` gives the 1.2.0 basis.
- `ntheta` and `nzeta` of `continuum`, `continuum_matrices`, `global_matrices` and `eigenmodes` default to `None`: the
  smallest even grid that does not alias the assembled harmonics, at least 32 ({func}`fax.continuum.grid_size`). Lists that
  fit 32 points keep the 1.2.0 grid; wider ones, which 1.2.0 refused by default, get a finer grid.
- `mode_family` returns a NumPy array instead of a JAX array: harmonics are static host data.
- `growth_rate(..., standing=True)` gives the drive of a cosine- or sine-sector mode, the standing wave of its two
  travelling components; without it the mode is read as a travelling wave, as in 1.2.0.

What stops working:

- An explicit angle grid that resolves the listed harmonics can be refused, since both signs double the span of $m$ and
  $n$: `continuum(geo, mode_family(0, 6, 1), rho, 16, 4)` raises "ntheta=16 aliases the harmonics" (it needs
  `ntheta > 24`, or `parity="single"`).
