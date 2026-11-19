# Changelog

## Unreleased

What changes for users of 1.2.0:

- `continuum`, `eigenmodes` and `frequency_hz` solve each listed harmonic on both signs, $(m, n)$ and $(-m, -n)$, split into
  blocks that do not couple ({doc}`theory`, section 2). A self-conjugate family ($N = 0$, or $N = N_{fp}/2$, or $n = 0$ in a
  tokamak) now returns twice as many branches and modes as harmonics: its cosine and sine sectors. Other families are
  unchanged. `parity="single"` gives the 1.2.0 basis.
- `mode_family` returns a NumPy array instead of a JAX array: harmonics are static host data.
- `growth_rate(..., standing=True)` gives the drive of a cosine- or sine-sector mode, the standing wave of its two
  travelling components; without it the mode is read as a travelling wave, as in 1.2.0.
