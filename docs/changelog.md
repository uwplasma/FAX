# Changelog

## Unreleased

What changes for users of 1.2.0:

- `continuum`, `eigenmodes` and `frequency_hz` solve each listed harmonic on both signs, $(m, n)$ and $(-m, -n)$, split into
  blocks that do not couple ({doc}`theory`, section 2). A self-conjugate family ($N = 0$, or $N = N_{fp}/2$, or $n = 0$ in a
  tokamak) now returns twice as many branches and modes as harmonics: its cosine and sine sectors. Other families are
  unchanged. `parity="single"` gives the 1.2.0 basis.
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
