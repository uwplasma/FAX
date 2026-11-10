"""Rebuild ``stellgap_ctok_continuum.npz`` and ``ae3d_ctok_modes.npz`` from the answer packets of the public test problems.

Usage: ``python make_ctok_fixtures.py STELLGAP_ANSWERS AE3D_ANSWERS OUT_DIR``, where the first two are the unpacked
``stellgap_test_problem_answers.tar.gz`` (``alfven_post``) and ``ae3d_test_problem_answers.tar.gz`` (``egn_mode_asci.dat``).
Both outputs are reproduced bit for bit (``tests/data/PROVENANCE.md`` lists the packet hashes).
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
    vec = raw[start + nm + NS :][: nm * NS * NH].reshape(nm, NS, NH)
    np.savez_compressed(
        Path(out) / "ae3d_ctok_modes.npz", khz=np.sqrt(raw[start : start + nm][list(MODES)]), vectors=vec[list(MODES)].astype(np.float32)
    )


if __name__ == "__main__":
    main(*sys.argv[1:4])
