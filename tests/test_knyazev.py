"""Independent 3-D reference: the precise-QA STELLGAP continuum and AE3D gap band of Knyazev et al. (2026).

The offline fixture and its provenance are in ``tests/data`` (``PROVENANCE.md``); ``examples/knyazev_qa.py`` rebuilds it
and holds the comparison helpers. Tolerances are the convergence budgets of ``docs/validation.md``.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("knyazev_qa", ROOT / "examples" / "knyazev_qa.py")
kq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kq)
FIXTURE = ROOT / "tests" / "data" / "knyazev_qa.npz"


@pytest.fixture(scope="module")
def fx():
    return kq.load(FIXTURE)


def test_fixture_is_small_and_describes_a_dense_spectrum(fx):
    assert FIXTURE.stat().st_size < 1e6
    khz = np.sort(fx["ae_khz"].astype(float))
    gap = np.minimum(np.diff(khz, prepend=-np.inf), np.diff(khz, append=np.inf)) / np.abs(khz)
    assert len(khz) == 7938 and (khz < 0).sum() == 56  # AE3D's own ideal-unstable eigenvalues
    assert ((gap > 0.01) & (khz > 5)).sum() == 3  # no isolated eigenvalue: only 3 of 7938 above 5 kHz are 1% from both neighbours
    assert fx["band_khz"][int(fx["band_global"])] == pytest.approx(kq.GLOBAL_KHZ, abs=1e-3)
    assert len(fx["band_khz"]) == 20 and fx["band_vec"].shape == (20, 49, 96) and fx["global_vec"].shape == (49, 162)


def test_stellgap_continuum_parity(fx):
    """162 branches on every 8th fixture surface, complex basis of the AE3D harmonic list, unaliased 64 x 48 grid."""
    sub = {**fx, "sg_s": fx["sg_s"][::8], "sg_khz": fx["sg_khz"][::8]}
    err, edge, s = kq.continuum_error(sub, 64, 48)
    assert err.size == 13 and err.mean() < 4e-3 and err.max() < 5e-3  # measured 3.0e-3 mean, 4.3e-3 max on all 98
    assert (edge[s > 0.25].max(0) < [1.5e-2, 4e-2]).all()  # branches below and above the archived global mode: 1.0% and 3.0%


def test_ae3d_gap_band_in_the_sine_basis(fx):
    """FAX in AE3D's sine basis at 24 harmonics: positive operator, archived global mode, band count and containment."""
    r = kq.run(fx, 49, 24, 40, 24)
    c = kq.compare(fx, r)
    assert r.khz.min() >= 0 and np.isfinite(r.khz).all() and np.allclose(r.F, r.F.conj().T, atol=1e-6 * np.abs(r.F).max())
    assert abs(c["rq"][c["global"]] / kq.GLOBAL_KHZ - 1) < 2e-2  # the archived mode's FAX energy: +0.9% here, -1.1% at 64 harmonics
    assert 10 <= len(c["band"]) <= 26  # AE3D has 20; FAX 12-14, 22, 21 at 24, 40, 64 harmonics
    assert c["contained"].mean() > 0.15  # the band is shared only in part: 0.22, 0.49, 0.41


@pytest.mark.slow
def test_ae3d_band_at_64_harmonics(fx):
    r = kq.run(fx, 49, 64, 48, 32)
    c = kq.compare(fx, r)
    assert abs(c["rq"][c["global"]] / kq.GLOBAL_KHZ - 1) < 2e-2
    assert 18 <= len(c["band"]) <= 24 and c["contained"].mean() > 0.3
