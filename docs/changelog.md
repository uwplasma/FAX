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
- The design path (`geometry_frequencies`, `design_frequencies`, `geometry_growth_rates`), `summary` and the case runner use
  the same default basis. A vmex state without `lasym` is stated stellarator symmetric, so the design path solves real
  sectors under `jax.jit`. The quasi-axisymmetric example `examples/qa_vmex.py` moves from 294.11, 294.44, 294.92 kHz to
  245.55, 245.76, 246.06 kHz. The case file key `run.parity` is unset or `"auto"` (default), `"single"`, `1` or `-1`, and
  unset `ntheta` and `nzeta` take the default grid.
- `geometry_growth_rates` drives the modes of cosine and sine sectors as pairs (`fax.drive.pair_rates`): a mode and its
  nearest partner of the other sector form the 2 x 2 ledger of plan 8.5 (frequencies on the diagonal, the drive matrix
  in one wave energy) and get its two rates. Far apart in frequency these are the standing-wave rates; for the LP-QA pair
  near 104.5 kHz, 1412 rad/s apart and coupled at 1.6e4 1/s, they are +15805 and -16173 1/s, the travelling waves.
  Travelling modes go through `fax.eigen.travelling`.
- `mode_families` groups a real potential: one group per $|n|$ in a tokamak and one per pair $\{N, -N\}$ of
  $N = n \bmod N_{fp}$ in a stellarator, instead of one per $n$ and one per $N$.
- `mode_family` returns a NumPy array instead of a JAX array: harmonics are static host data.
- `growth_rate(..., standing=True)` gives the drive of a cosine- or sine-sector mode, the standing wave of its two
  travelling components; without it the mode is read as a travelling wave, as in 1.2.0. The standing rate is the mode's
  rate only when its cosine/sine partner is further away in frequency than the drive couples the two; in an axisymmetric
  field the partners are degenerate and the travelling rates are the physical ones.
- `structures` of a block that holds the families $N$ and $-N$ store a $(-m, -n)$ unknown on the listed $(m, n)$ row as
  its conjugate (the potential at $t = 0$). `fax.eigen.travelling` gives the harmonics and coefficients that
  `growth_rate` reads as a travelling wave.

What stops working:

- An explicit angle grid that resolves the listed harmonics can be refused, since both signs double the span of $m$ and
  $n$: `continuum(geo, mode_family(0, 6, 1), rho, 16, 4)` raises "ntheta=16 aliases the harmonics" (it needs
  `ntheta > 24`, or `parity="single"`).
- A traced geometry with $N_{fp} = 1$ whose axisymmetry is unknown (for example one rebuilt by `dataclasses.replace`
  inside `jax.jit`) is refused by the default basis when the list holds more than one $|n|$: axisymmetry decides whether
  they couple, and the eager call, which knows it, would return a different number of branches (9 against 12 on the
  test field). Build the geometry on the host, or use `parity="single"`.
- A stated symmetry fact (`Geometry.from_arrays(..., stellsym=True)`) that the concrete arrays contradict raises.
- A basis that mixes sector modes with travelling modes is refused by `geometry_growth_rates`.
