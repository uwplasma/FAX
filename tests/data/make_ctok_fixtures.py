"""Rebuild ``stellgap_ctok_continuum.npz`` and ``ae3d_ctok_modes.npz`` from the answer packets of the public test problems.

Usage: ``python make_ctok_fixtures.py STELLGAP_ANSWERS AE3D_ANSWERS OUT_DIR``, where the first two are the unpacked
``stellgap_test_problem_answers.tar.gz`` (``alfven_post``) and ``ae3d_test_problem_answers.tar.gz`` (``egn_mode_asci.dat``,
``egn_values.dat``, ``a_matrix.dat``, ``b_matrix.dat``). Both outputs are reproduced bit for bit (``tests/data/PROVENANCE.md``
lists the packet hashes). The AE3D eigenvalues ``f^2`` (kHz^2) keep their sign: ``egn_values.dat`` stores ``sqrt(f^2)`` for
``f^2 >= 0`` and ``f^2`` for the five negative roots. The matrices are stored as exported, unsymmetrized (float32 blocks).
"""

import sys
from pathlib import Path

import numpy as np

NH, NS = 13, 99
MODES = (1165, 1164, 1152, 1046, 987, 1003, 1006)  # 0-based eigenvalue indices: the 337.9 kHz TAE and six global modes


def main(stellgap, ae3d, out):
    post = np.loadtxt(Path(stellgap) / "alfven_post", skiprows=1)  # columns s, f [kHz], m, n; 13 branches per surface
    s, khz = post[::NH, 0], np.sort(post[:, 1].reshape(-1, NH), axis=1)
    keep = s < 0.99  # the last four surfaces of the packet are not used
    np.savez_compressed(Path(out) / "stellgap_ctok_continuum.npz", s=s[keep], khz=khz[keep])
    raw = np.loadtxt(Path(ae3d) / "egn_mode_asci.dat")  # nmodes, NH, NS, (n, m) pairs, f^2 [kHz^2], s grid, vectors (mode, s, harmonic)
    nm, start = int(raw[0]), 3 + 2 * NH
    vec, lam = raw[start + nm + NS :][: nm * NS * NH].reshape(nm, NS, NH), raw[start : start + nm]
    x = np.loadtxt(Path(ae3d) / "egn_values.dat")[:, 0]
    assert np.allclose(np.where(x >= 0, x**2, x), lam, rtol=2e-6, atol=0) and (lam < 0).sum() == 5
    blocks = {}
    for k in "ab":  # 1-based (row, column, value) of the nonzero entries; block-tridiagonal in the surfaces
        t = np.loadtxt(Path(ae3d) / f"{k}_matrix.dat")
        assert np.abs((t[:, 0] - 1) // NH - (t[:, 1] - 1) // NH).max() == 1
        M = np.zeros((NS * NH, NS * NH))
        M[t[:, 0].astype(int) - 1, t[:, 1].astype(int) - 1] = t[:, 2]
        M, i = M.reshape(NS, NH, NS, NH).swapaxes(1, 2), np.arange(NS)
        blocks |= {f"{k}_diag": M[i, i], f"{k}_upper": M[i[:-1], i[1:]], f"{k}_lower": M[i[1:], i[:-1]]}
    np.savez_compressed(
        Path(out) / "ae3d_ctok_modes.npz",
        khz=np.sqrt(lam[list(MODES)]),
        vectors=vec[list(MODES)].astype(np.float32),
        lam=lam,
        index=np.array(MODES, np.int16),
        **{k: b.astype(np.float32) for k, b in blocks.items()},
    )


if __name__ == "__main__":
    main(*sys.argv[1:4])
