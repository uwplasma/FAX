"""Independent 3-D reference: the precise-QA STELLGAP continuum and AE3D gap band of Knyazev et al. (2026).

The offline fixture and its provenance are in ``tests/data`` (``PROVENANCE.md``); ``tests/knyazev_ref.py`` rebuilds it
and holds the comparison helpers (``examples/knyazev_qa.py`` uses them). Tolerances are the convergence budgets of ``docs/validation.md``.
"""

from pathlib import Path

import jax
import knyazev_ref as kq
import numpy as np
import pytest

import fax
from fax.continuum import generalized_eigh
from fax.eigen import dense, global_matrices

FIXTURE = Path(__file__).parent / "data" / "knyazev_qa.npz"


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
    """162 branches on every 8th fixture surface, the single-sign basis of the AE3D harmonic list, unaliased 64 x 48 grid."""
    sub = {**fx, "sg_s": fx["sg_s"][::8], "sg_khz": fx["sg_khz"][::8]}
    err, edge, s = kq.continuum_error(sub, 64, 48, parity="single")
    assert err.size == 13 and err.mean() < 4e-3 and err.max() < 5e-3  # measured 3.0e-3 mean, 4.3e-3 max on all 98
    assert (edge[s > 0.25].max(0) < [1.5e-2, 4e-2]).all()  # branches below and above the archived global mode: 1.0% and 3.0%


def test_regression_stellgap_continuum_on_all_surfaces(fx):
    """Regression, labelled as such (plan 8.3 and 10.3): on all 98 STELLGAP surfaces of Knyazev et al. (2026) the relative L2
    error of the cosine sector, STELLGAP's own basis, is 1.617e-3 (the 0.160% of the plan) against 2.975e-3 (0.300%) for
    the single-sign list; measured 2026-11-19, to 0.2%."""
    cos, single = (kq.continuum_error(fx, 112, 72, parity=p)[0] for p in (1, "single"))
    assert cos.size == single.size == 98
    assert cos.mean() == pytest.approx(1.617e-3, rel=2e-3) and single.mean() == pytest.approx(2.975e-3, rel=2e-3)


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


def test_parity_sectors_split_the_two_sign_problem_and_cos_is_stellgaps_basis(fx):
    """The cos and sin sectors together are the problem on (m, n) and (-m, -n); the single-sign basis is neither sector."""
    geo, h = kq.geometry(fx, 12), fx["mn"][fx["order"][:16]].astype(float)
    both = np.concatenate([h, -h])
    full = np.asarray(fax.continuum(geo, both, kq.RHO, 112, 72))
    cos, sin = (np.asarray(fax.continuum(geo, h, kq.RHO, 112, 72, parity=p)) for p in (1, -1))
    np.testing.assert_allclose(np.sort(np.concatenate([cos, sin], 1)), full, rtol=1e-9)
    F, G = global_matrices(geo, both, kq.RHO, 64, 48)
    sectors = [global_matrices(geo, h, kq.RHO, 64, 48, parity=p) for p in (1, -1)]
    lam = np.sort(np.concatenate([generalized_eigh(dense(A), dense(B))[0] for A, B in sectors]))
    np.testing.assert_allclose(lam, generalized_eigh(dense(F), dense(G))[0], rtol=1e-9)
    assert dense(sectors[0][0]).dtype == np.float64 and dense(sectors[0][0]).shape == (12 * 16, 12 * 16)
    single = np.asarray(fax.continuum(geo, h, kq.RHO, 112, 72, parity="single"))
    assert np.abs(single / cos - 1).max() > 1e-2  # the (m, n) x (-m', -n') coupling is missing from the single-sign basis
    sub = {**fx, "sg_s": fx["sg_s"][::8], "sg_khz": fx["sg_khz"][::8]}
    err = kq.continuum_error(sub, 112, 72, parity=1)[0]
    assert err.mean() < 2e-3 and err.max() < 3e-3  # STELLGAP's cos basis: 1.6e-3 mean, 2.6e-3 max on all 98 (single-sign 3.0e-3, 4.3e-3)
    with pytest.raises(ValueError, match="once up to sign|distinct"):
        fax.continuum(geo, both, kq.RHO, 112, 72, parity=1)


def test_one_field_period_of_quadrature_matches_the_full_turn(fx):
    """At nfp = 2 the averages use one field period of the angle grid; the full turn gives the same matrices."""
    from fax.continuum import _continuum_matrices
    from fax.eigen import _global_matrices
    from fax.geometry import field_period, surface_grid

    geo, h = kq.geometry(fx, 8), fx["mn"][fx["order"][:16]].astype(float)
    assert field_period(geo, 48) == 2 and field_period(geo, 47) == 1 and surface_grid(geo, 64, 48, 2).B.shape == (8, 64, 24)
    full = surface_grid(geo, 64, 48)
    K, M = fax.continuum_matrices(geo, h, kq.RHO, 64, 48, pressure=1e3)
    for a, b in zip((K, M), _continuum_matrices(geo, full, h, kq.RHO, 1e3, 5 / 3, 1), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-12 * float(np.abs(b).max()))
    blocks = global_matrices(geo, h, kq.RHO, 64, 48, finite_beta=True), _global_matrices(geo, full, h, kq.RHO, True, 1)
    for a, b in zip(*map(jax.tree_util.tree_leaves, blocks), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-12 * float(np.abs(b).max()))
