# Development

## Layout

| File | Contents |
|---|---|
| `src/fax/geometry.py` | Boozer geometry, readers, the vmex/booz_xform_jax path, metric grids |
| `src/fax/continuum.py` | continuum matrices and frequencies, gaps |
| `src/fax/eigen.py` | AE3D operator, a thin wrapper of `solvax.generalized_eigenpairs`, `Status`, derivative rejection |
| `src/fax/drive.py` | perturbative fast-ion drive, Landau damping |
| `src/fax/objectives.py` | smooth objectives, VMEC-boundary design |
| `src/fax/io.py`, `plot.py` | TOML cases, `fax` command, figures |
| `tests/` | one file per module; each test is an equation or a published number |
| `examples/` | the cases of {doc}`tutorials` |

## Tests and checks

Three tiers, selected by two markers (`slow`, `scheduled`) and the workflow that runs them:

| Tier | Contents | Runs |
|---|---|---|
| PR | analytic and contract tests (unmarked) in the Python 3.10–3.13 matrix and on the oldest stack (Python 3.10, JAX 0.4.38, SOLVAX 0.36.0); `slow` (vmex solves, 64-harmonic AE3D band) added in the coverage job; the installed-wheel smoke test; lint, coverage ≥ 95% (statements and branches, `-m 'not scheduled'`), docs, CITATION.cff | every PR and push (`tests.yml`) |
| Scheduled | `scheduled`: frozen-path gradients, resolution budgets, gauge floors; the same minimum stack with the `slow` tests, the minimum vmex stack (Python 3.11, vmex 0.11.10, JAX 0.9.2) and the latest stack (Python 3.13, everything upgraded) | weekly and on demand (`scheduled.yml`) |
| Release | external comparisons and high-resolution runs; see the checklist below | by hand before a tag |

```console
pip install -e ".[dev,vmex]"
python -m pytest -q                                  # PR tier, unmarked tests (about 5 min with vmex installed)
python -m pytest -q -m 'not scheduled'               # what the coverage job runs: adds slow
python -m pytest -q -m scheduled -s                  # scheduled tier (about 30 min)
ruff check . && ruff format --check .
python -m coverage run --branch --source=src/fax -m pytest -q -m 'not scheduled' && python -m coverage report -m
sphinx-build -W -b html docs docs/_build/html
```

The installed-wheel smoke test builds the wheel, installs it into a fresh virtual environment in a temporary directory (no
`tests/`, no `examples/`) and runs `examples/quickstart.py` and `tests/install_smoke.py` there: the uniform-field continuum against $k_\parallel v_A$, a
global eigenmode and an eigenvalue gradient on `fax.circular_torus`. It also fails if the wheel holds anything but
`fax/*.py` and its metadata (36 kB). Locally:

```console
python -m build --outdir /tmp/dist && python -m venv /tmp/venv && /tmp/venv/bin/pip install /tmp/dist/*.whl
cd /tmp && cp <checkout>/tests/install_smoke.py <checkout>/examples/quickstart.py .
venv/bin/python -I install_smoke.py && venv/bin/python -I quickstart.py
```

Release tier, to be run and recorded in the release notes before every tag (not automated):

1. External comparisons: STELLGAP and AE3D CTOK, the Knyazev et al. (2026) precise QA at 64 harmonics, the Könies et al.
   (2018) ITPA TAE and drive (`examples/koenies_tae_drive.py`), with the tolerances of {doc}`validation`.
2. CPU and GPU precision: the CTOK, ITPA and LP-QA gates in float64 on CPU and on one GPU; frequencies agree to the
   validation tolerances and gradients to 1e-6 relative (`jax_enable_x64` is set on import).
3. High resolution: radial, harmonic and angle refinements of the LP-QA gap-mode objective beyond the scheduled sweeps
   and `examples/benchmark.py` at equal accuracy.
4. `python -m build`, `twine check`, the smoke test above on the wheel to be uploaded, then `git tag vX.Y.Z` (the tag must
   equal `version` in `pyproject.toml` and `fax.__version__`): the `Publish` workflow uploads through PyPI trusted
   publishing (environment `pypi`). Update `version` and `date-released` in `CITATION.cff` in the same commit.

CI runs the PR matrix on Python 3.10–3.13 and the vmex path, lint, coverage and docs on Python 3.12. SOLVAX comes
from PyPI (0.36.0 or newer). Keep
Python 3.10 working (the `tomli` fallback, no newer-JAX-only functions).

Coverage is measured with branches on the PR-tier coverage run (unmarked and `slow` tests,
`src/fax` only, no excluded lines or modules): statements 100% (731/731) and
branches 100% (70/70). CI fails if either falls below 95%, checked separately.
The unmarked tests alone reach 94% of statements, because the vmex paths
(`from_vmec_input`, `prepare_design`, `design_geometry`) run only in the slow
tests.

## Reference fixtures

`tests/data/PROVENANCE.md` is the one manifest of every fixture in `tests/data` and `examples/data` (identifier, URL, SHA-256,
model switches, units, license, generation command, tolerance); `tests/test_provenance.py` checks its sizes and hashes, so a
fixture cannot change without an entry. The CTOK `.npz` files are rebuilt from the upstream answer packets by `tests/data/make_ctok_fixtures.py`. The Knyazev et al. (2026) precise-QA fixture (0.67 MB) is rebuilt from the Zenodo archive
(DOI 10.5281/zenodo.21299907, 1.1 GB, CC-BY-4.0) by `examples/knyazev_qa.py`: download `Knyazev_JPP2026.zip` from
zenodo.org, set `ARCHIVE` and `REBUILD = True` at the top of the script and run it. The archive is data, never code: the
script reads members with `zipfile` and parses them as text. The same script holds the comparison helpers the test uses
and, with `STUDY = True`, regenerates the tables and the figure of {doc}`validation`.

## Contributing

- Keep the code small: add physics to the existing modules rather than new
  layers.
- Every new physics claim needs a test against an analytic result or a
  published number, and an entry in {doc}`validation`.
- Every equation in {doc}`theory` must match the code; update both together.
- Figures in `docs/_static` stay compressed (under 200 kB each).

## Citing

If you use FAX, please cite the repository
(<https://github.com/uwplasma/fax>, `CITATION.cff`) and the papers whose models it implements:
Spong, Sanchez & Weller (2003) for the continuum, Spong, D'Azevedo & Todo
(2010) for the global operator, and Könies et al. (2018) for the TAE
benchmark ({doc}`references`).

## License

MIT.
