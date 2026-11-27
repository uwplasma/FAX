"""Rebuild ``stellgap_ctok_edge_continuum.npz``: the STELLGAP CTOK continuum with the outer Boozer half surface restored.

The packet file ``boozmn_ctok.nc`` holds Boozer coefficients on half surfaces ``jlist = 2..100`` of ``ns_b = 101``; the outer
one is absent, and the ORNL metric builder averages it with zeros (its own comment, lines 9-17). This script transforms the
packet's ``wout_ctok.nc`` with the C++ ``booz_xform`` (``mboz = 48``, ``nboz = 1``, all 100 half surfaces), appends only the
final coefficient row to the packet file (the 99 packet rows, ``ixm_b``, ``ixn_b`` and the 101-point profiles are kept), then runs
``xmetric ctok``, ``xstgap 99 300`` and ``xpost`` with the packet's ``plasma.dat`` and ``fourier.dat``.

Usage: ``python make_stellgap_ctok_edge.py STELLGAP BOOZ_XFORM STELLOPT OUT_DIR``: git clones holding the pinned commits below
(sources are read with ``git show`` at the pin, so the checked-out tree does not matter). Needs a Fortran compiler, an MPI Fortran
wrapper, a C++ compiler and NetCDF C and Fortran; override with ``FC``, ``MPIFC``, ``CXX``, ``NC_CONFIG``, ``NF_CONFIG``.
"""

import hashlib
import io
import os
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

STELLGAP, BOOZ, STELLOPT = (
    "9be3aa959bb36bd8c47e8598b6ed283e57a45336",
    "03fa6456711f2b7bb45cadea796494a1c82ebb37",
    "c761def7d7ba6038840a77da6d4c18a503686778",
)
PACKET, BOOZMN, WOUT = (
    "d37a2e3654dce36baffff8958a9d33920e1feb3fdb66885187b525cfadc823bb",  # stellgap_test_problem.tar.gz
    "63d3005cb8796ac64698405ea485b422faf6fc1224bdc84411a7d8584ec43403",  # its boozmn_ctok.nc = tests/data/boozmn_ctok.nc
    "729434ddaa48b8b6098f553d8629a6e96ffb600af3e64775770a2efae94a20bb",  # its wout_ctok.nc
)
LIBSTELL = "Modules/stel_kinds.f Modules/stel_constants.f Modules/mpi_params.f Modules/mpi_inc.f Modules/safe_open_mod.f90"
LIBSTELL += " Miscel/parse_extension.f Ezcdf/handle_err.f90 Ezcdf/ezcdf_opncls.f90 Ezcdf/ezcdf_inqvar.f90 Ezcdf/ezcdf_attrib.f90"
LIBSTELL += " Ezcdf/ezcdf_GenPut.f90 Ezcdf/ezcdf_GenGet.f90 Ezcdf/ezcdf.f90 Modules/read_boozer_mod.f90"
COEFFS = ("bmnc_b", "rmnc_b", "zmns_b", "pmns_b", "gmn_b")
LEGACY = ["-O2", "-std=legacy", "-fallow-argument-mismatch", "-ffixed-line-length-none"]


def run(argv, cwd):
    with open(Path(cwd) / "build.log", "a") as log:
        print("$", shlex.join(map(str, argv)), file=log, flush=True)
        subprocess.run(argv, cwd=cwd, check=True, stdout=log, stderr=subprocess.STDOUT)


def extract(repo, pin, path, target):
    """Write ``path`` of ``repo`` at commit ``pin`` to ``target``; ``path`` ending in ``/`` is a tree."""
    if path.endswith("/"):
        tar = subprocess.run(["git", "-C", repo, "archive", pin, path], check=True, capture_output=True).stdout
        tarfile.open(fileobj=io.BytesIO(tar)).extractall(target, filter="data")
    else:
        Path(target).write_bytes(subprocess.run(["git", "-C", repo, "show", f"{pin}:{path}"], check=True, capture_output=True).stdout)


def checked(data, digest, name):
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"SHA-256 of {name} differs from the pinned packet")
    return data


def build(stellgap, booz, stellopt, out):
    fc, mpifc, cxx = (os.environ.get(k, d) for k, d in (("FC", "gfortran"), ("MPIFC", "mpifort"), ("CXX", "c++")))
    nc, nf = (shlex.split(os.environ.get(k, d)) for k, d in (("NC_CONFIG", "nc-config"), ("NF_CONFIG", "nf-config")))
    cfg = lambda tool, *a: shlex.split(subprocess.check_output([*tool, *a], text=True))  # noqa: E731
    for name in ("stellgap_serial_ver5.f", "Fourier_lib_convolve.f", "fitpack.f", "post_process.f", "metric_element_create_ver8.46.f"):
        extract(stellgap, STELLGAP, name, out / name)
    lapack = ["-framework", "Accelerate"] if sys.platform == "darwin" else ["-llapack", "-lblas"]
    run([fc, *LEGACY, "-o", "xstgap", "stellgap_serial_ver5.f", "Fourier_lib_convolve.f", "fitpack.f", *lapack], out)
    run([fc, *LEGACY, "-o", "xpost", "post_process.f"], out)
    extract(booz, BOOZ, "src/_booz_xform/", out)
    extract(booz, BOOZ, "externalPackages/eigen/", out)
    cpp = sorted(str(p) for p in (out / "src/_booz_xform").glob("*.cpp") if p.name != "bindings.cpp")
    run([cxx, "-O2", "-std=c++14", "-IexternalPackages/eigen", *cfg(nc, "--cflags"), *cpp, *cfg(nc, "--libs"), "-o", "xboozer"], out)
    sources = [Path(p).name for p in LIBSTELL.split()]
    for path, name in zip(LIBSTELL.split(), sources, strict=True):
        extract(stellopt, STELLOPT, f"LIBSTELL/Sources/{path}", out / name)
    flags = [*LEGACY, "-cpp", "-DNETCDF", "-DMPI_OPT", "-ffree-line-length-none", *cfg(nf, "--fflags")]
    run([mpifc, *flags, *sources, "metric_element_create_ver8.46.f", *cfg(nf, "--flibs"), *cfg(nc, "--libs"), "-o", "xmetric"], out)


def complete_edge(packet_file, cpp_file, target):
    """The packet file with ``jlist = 2..101``: its 99 coefficient rows, then the C++ row of half surface 101."""
    with Dataset(packet_file) as src, Dataset(cpp_file) as cpp, Dataset(target, "w", format=src.file_format) as dest:
        assert list(src["jlist"][:]) == list(range(2, 101)) and list(cpp["jlist"][:]) == list(range(2, 102))
        assert all(np.array_equal(src[k][:], cpp[k][:]) for k in ("ixm_b", "ixn_b"))
        for k, dim in src.dimensions.items():
            dest.createDimension(k, len(dim) + (k in ("comput_surfs", "pack_rad")))
        for k, var in src.variables.items():
            dest.createVariable(k, var.datatype, var.dimensions).setncatts(var.__dict__)
            if k in COEFFS:
                dest[k][:] = np.vstack([var[:], cpp[k][-1:]])
            else:
                dest[k][...] = cpp[k][:] if k == "jlist" else var[...]
        dest.setncatts(src.__dict__)
        rows = {k: float(np.linalg.norm(src[k][:] - cpp[k][:-1]) / np.linalg.norm(src[k][:])) for k in COEFFS}
    print("C++ against the 99 packet rows, relative L2:", rows)


def main(stellgap, booz, stellopt, out):
    out = Path(out).resolve()
    out.mkdir()
    build(stellgap, booz, stellopt, out)
    extract(stellgap, STELLGAP, "stellgap_test_problem.tar.gz", out / "packet.tar.gz")
    with tarfile.open(fileobj=io.BytesIO(checked((out / "packet.tar.gz").read_bytes(), PACKET, "packet"))) as packet:
        member = lambda name: packet.extractfile(name).read()  # noqa: E731
        (out / "boozmn_packet.nc").write_bytes(checked(member("boozmn_ctok.nc"), BOOZMN, "boozmn_ctok.nc"))
        (out / "wout_ctok.nc").write_bytes(checked(member("wout_ctok.nc"), WOUT, "wout_ctok.nc"))
        for name in ("plasma.dat", "fourier.dat"):
            (out / name).write_bytes(member(name))
    (out / "in_booz.cpp").write_text("48 1\nctok\n")  # no surface list: all 100 half surfaces, jlist = 2..101
    run(["./xboozer", "in_booz.cpp"], out)
    (out / "boozmn_ctok.nc").rename(out / "boozmn_cpp.nc")
    complete_edge(out / "boozmn_packet.nc", out / "boozmn_cpp.nc", out / "boozmn_ctok.nc")
    run(["./xmetric", "ctok"], out)
    metric = (out / "tae_data_boozer").read_bytes()
    (out / "tae_data_boozer").write_bytes(b"F\n" + metric)  # the non-RFP flag line that xstgap reads first
    run(["./xstgap", "99", "300"], out)
    run(["./xpost"], out)
    post = np.loadtxt(out / "alfven_post", skiprows=1)  # columns s, f [kHz], m, n; 13 branches per surface
    s, khz = post[::13, 0], np.sort(post[:, 1].reshape(-1, 13), axis=1)
    keep = s < 0.99  # as in make_ctok_fixtures.py
    np.savez_compressed(out / "stellgap_ctok_edge_continuum.npz", s=s[keep], khz=khz[keep])


if __name__ == "__main__":
    main(*sys.argv[1:5])
