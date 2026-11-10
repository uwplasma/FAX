"""Build ``ae3d_v31_ctok.npz`` from the pinned AE3D v31 CTOK run (docs/validation.md, "pinned block comparison").

Usage: ``python make_ae3d_v31_ctok.py RUN_D RUN_SD OUT`` where ``RUN_D`` and ``RUN_SD`` are the run directories of the instrumented
build (``ae3d_v31_instrumentation.patch``) with ``AE3D_SW=D`` and ``AE3D_SW=SD``; both hold ``ae_metric.dat``, ``blocks.bin``, ``mats.bin``.
"""

import sys
from pathlib import Path

import numpy as np

NS, NT, NZ, H = 99, 80, 80, 13
SELECTED = (44, 45)  # AE3D surfaces 45 and 46, s = 0.45 and 0.46


def blocks(path):
    """``a, b, d, aa, bb, cc, dd`` (surface, i, j) from ``blocks.bin``: int32 sizes and harmonics, 8 profiles, then column-major blocks."""
    raw = np.fromfile(path, np.uint8).tobytes()
    offset = 8 + 8 * H + 8 * NS * 8
    return {
        k: np.frombuffer(raw, "<f8", H * H * NS, offset + 8 * H * H * NS * i).reshape((H, H, NS), order="F").transpose(2, 0, 1)
        for i, k in enumerate(("a", "b", "d", "aa", "bb", "cc", "dd"))
    }


def main(run_d, run_sd, out):
    run_d, run_sd = Path(run_d), Path(run_sd)
    head = np.loadtxt(run_d / "ae_metric.dat", skiprows=1, max_rows=NS)  # iota, iota', jpol, jpol', jtor, jtor', phip, phipp
    grid = np.loadtxt(run_d / "ae_metric.dat", skiprows=1 + NS, max_rows=NS * NT * NZ).reshape(NS, NZ, NT, 11)[
        :, 0
    ]  # theta fast; axisymmetric
    for a in (grid, head):  # AE3D replaces its last point by linear extrapolation
        a[-1] = 2 * a[-2] - a[-3]
    d, sd = blocks(run_d / "blocks.bin"), blocks(run_sd / "blocks.bin")
    F = np.fromfile(run_sd / "mats.bin", "<f8").reshape((2, NS * H, NS * H))[0].T  # mat_bending, column-major; AE3D stores its negative
    sel = list(SELECTED)
    np.savez_compressed(
        out,
        s=np.arange(1, NS + 1) / (NS + 1),
        field_pair=grid[sel],
        profiles_pair=head[sel],
        ae_a=d["a"][sel],
        ae_b=d["b"][sel],
        ae_d=d["d"][sel],
        ae_aa=d["aa"][sel],
        ae_bb0=d["bb"][sel],
        ae_bbS=sd["bb"][sel],
        ae_dd0=d["dd"][sel],
        ae_ddS=sd["dd"][sel],
        field_all=grid[:, ::2].astype(np.float32),
        profiles_all=head,
        ae_F_diag=np.stack([-F[H * k : H * k + H, H * k : H * k + H] for k in range(NS)]).astype(np.float32),
        ae_F_upper=np.stack([-F[H * k : H * k + H, H * k + H : H * k + 2 * H] for k in range(NS - 1)]).astype(np.float32),
        ae3d_commit=np.array("383d327c53bf722026d7e432797ea37c50347b68"),
        ae3d_build=np.array(
            "gfortran 15.2.0 (Homebrew GCC 15.2.0_1): -O2 -ffixed-form -ffixed-line-length-none "
            "-fallow-argument-mismatch; LAPACK Accelerate"
        ),
        ae3d_run=np.array("ae3d_test_problem.tar.gz CTOK, plasma.dat ion_density_0=1e19 amu=2 ion_profile=2; AE3D_SW=D and SD (J, P off)"),
        columns=np.array("rjacob bfield gsssup gttsup gzzsup gstsup gszsup gtzsup bfields bfieldth bfieldze"),
    )


if __name__ == "__main__":
    main(*sys.argv[1:4])
