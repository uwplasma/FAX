"""Helpers for the precise-QA comparison with Knyazev et al. (2026): fixture reader and builder, FAX and AE3D mode comparison.

Data: DOI 10.5281/zenodo.21299907 (version 10.5281/zenodo.21299908), ``Knyazev_JPP2026.zip``, CC-BY-4.0, 1.13 GB; the archive is
data, read with ``zipfile`` and parsed as text. ``build`` reduces it to ``tests/data/knyazev_qa.npz``. The tests and
``examples/knyazev_qa.py`` import this module.

Reference run: precise QA (``nfp = 2``, aspect 6, finite-beta equilibrium), STELLGAP continuum and AE3D with its kink term
removed, a flat mass density of 1.88e-6 kg/m^3 (the decks are not in the archive; the value is the one that reproduces the
STELLGAP continuum, see ``docs/validation.md``). AE3D uses the sine basis ``Phi = sum Phi_mn sin(m theta - n zeta)`` of 162
harmonics on 49 radial nodes ``s = 0.02, ..., 0.98``.
"""

import hashlib
import tempfile
import zipfile
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

import fax
from fax.continuum import generalized_eigh
from fax.eigen import Blocks, dense, global_matrices

FIXTURE = Path(__file__).resolve().parent / "data" / "knyazev_qa.npz"
RHO = 1.88e-6  # kg/m^3
ROOT = "Knyazev_JPP2026/"
BOOZ = ROOT + "simulations_fig4_fig5/QA/all_inputs/boozmn_betaQA.nc"
QA = ROOT + "FIG2-STELLGAP_and_AE3D_for_QI_and_QA/inputs/QA/"
BAND_HARMONICS, SG_STRIDE, BOOZER_CUT = 96, 8, 1e-5
BAND = (26.0, 35.0)  # kHz, the continuum gap that holds the archived global mode
GLOBAL_KHZ = 30.3086  # sqrt of AE3D eigenvalue 918.6117 (the mode the paper follows)


def load(path=FIXTURE):
    """Fixture arrays as a dict (``np.load`` is lazy and cannot be indexed twice cheaply)."""
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def geometry_half(fx):
    """The half-mesh Boozer geometry of the fixture."""
    return fax.Geometry.from_arrays(
        int(fx["g_nfp"]), s=fx["g_s"], xm=fx["g_xm"], xn=fx["g_xn"], iota=fx["g_iota"], G=fx["g_G"], I=fx["g_I"],
        **{k: fx["g_" + k] for k in ("bmnc", "rmnc", "zmns", "pmns", "gmnc")},
    )  # fmt: skip


def geometry(fx, nodes=49):
    """Boozer geometry on ``nodes`` radial nodes ``s_k = k / (nodes + 1)``; 49 is AE3D's grid (the full mesh of the file)."""
    return geometry_half(fx).interpolate(jnp.arange(1, nodes + 1) / (nodes + 1.0))


def continuum_khz(fx, geo, ntheta=64, nzeta=48, sine=False):
    """FAX continuum (kHz, sorted) on every surface of ``geo`` for the 162 AE3D harmonics.

    ``sine=False`` is the complex basis of the ``(m, n)`` list (STELLGAP lists 162 branches); ``sine=True`` restricts the
    ``(m, n)`` and ``(-m, -n)`` basis to the odd subspace, which is AE3D's real sine basis (it needs a finer grid).
    """
    h = jnp.asarray(fx["mn"], dtype=float)
    if not sine:
        return np.asarray(fax.continuum(geo, h, RHO, ntheta, nzeta)) / 1e3
    K, M = fax.continuum_matrices(geo, jnp.concatenate([h, -h]), RHO, ntheta, nzeta)
    q = jnp.concatenate([jnp.eye(len(h)), -jnp.eye(len(h))]) / np.sqrt(2.0)
    odd = lambda A: jnp.einsum("ia,sij,jb->sab", q, A, q)  # noqa: E731
    return np.sqrt(np.maximum(np.asarray(generalized_eigh(odd(K), odd(M))[0]), 0)) / 2e3 / np.pi


def sine_blocks(geo, harmonics, ntheta=64, nzeta=48):
    """AE3D's real sine basis: the FAX operator on ``(m, n)`` and ``(-m, -n)`` restricted to the odd subspace ``(c, -c)``."""
    n = len(harmonics)
    F, G = global_matrices(geo, np.concatenate([harmonics, -harmonics]), RHO, ntheta, nzeta)
    q = jnp.concatenate([jnp.eye(n), -jnp.eye(n)]) / np.sqrt(2.0)
    proj = lambda A: Blocks(jnp.einsum("ia,nij,jb->nab", q, A.diag, q), jnp.einsum("ia,nij,jb->nab", q, A.upper, q))  # noqa: E731
    return proj(F), proj(G)


def solve(F, G):
    """All modes of the pencil: ``(khz, vectors (modes, nodes, harmonics), dense G)``; vectors are ``G``-orthonormal."""
    Gd = dense(G)
    w2, x = generalized_eigh(dense(F), Gd)
    f = np.sqrt(np.maximum(np.asarray(w2), 0.0)) / 2e3 / np.pi
    return f, np.asarray(x).T.reshape(len(f), *F.diag.shape[:2]), np.asarray(Gd)


def radial_weight(y):
    """Radial weight ``sum_h |y|^2`` per node, normalized to 1 (shape ``(modes, nodes)``)."""
    w = (np.abs(y) ** 2).sum(-1)
    return w / w.sum(-1, keepdims=True)


def gap_position(khz, weight, s, sg_s, sg_khz):
    """Mean over the mode's radial weight of the position inside the STELLGAP gap that holds ``khz`` (0 on a branch, 0.5 mid-gap)."""
    C = sg_khz[np.abs(sg_s[None] - (s[:, None] + 0.02)).argmin(1)]  # STELLGAP labels are one cell (0.02) outward
    pos = np.zeros(weight.shape)
    for k, c in enumerate(C):
        i = np.searchsorted(c, khz)
        lo, hi = np.where(i > 0, c[np.maximum(i - 1, 0)], 0.0), np.where(i < len(c), c[np.minimum(i, len(c) - 1)], np.inf)
        pos[:, k] = np.minimum(khz - lo, hi - khz) / (hi - lo)
    return (weight * pos).sum(-1)


def band_modes(khz, weight, s, sg_s, sg_khz):
    """Radially extended gap modes in ``BAND``: inverse participation ratio >= 0.3 of the nodes, mean gap position >= 0.25."""
    ipr = 1 / (weight**2).sum(-1)
    return (khz > BAND[0]) & (khz < BAND[1]) & (ipr >= 0.3 * weight.shape[-1]) & (gap_position(khz, weight, s, sg_s, sg_khz) >= 0.25)


def overlaps(G, y, a):
    """``|<a, y_k>_G|^2 / <a, a>_G`` for the ``G``-orthonormal rows ``y`` and one vector ``a`` of shape ``(nodes, harmonics)``."""
    a, y = a.reshape(-1), y.reshape(len(y), -1)
    return np.abs(y.conj() @ (G @ a)) ** 2 / (a @ G @ a).real


def rayleigh_khz(F, G, a):
    """``sqrt(a.F.a / a.G.a) / 2 pi`` in kHz of a real vector ``a`` of shape ``(nodes, harmonics)``."""
    a = np.asarray(a, dtype=complex).reshape(-1)
    return float(np.sqrt((a.conj() @ F @ a).real / (a.conj() @ G @ a).real) / 2e3 / np.pi)


def continuum_error(fx, ntheta=64, nzeta=48, sine=False):
    """FAX against STELLGAP on the fixture surfaces (physical ``s`` = label - 0.02): per-surface relative L2 error of the
    sorted branches, and the relative error of the two branches that bracket ``GLOBAL_KHZ`` (lower, upper edge of its gap)."""
    s = fx["sg_s"] - 0.02
    ok = (s > fx["g_s"][0]) & (s < fx["g_s"][-1])
    ref = fx["sg_khz"][ok].astype(float)
    half = geometry_half(fx)
    f = continuum_khz(fx, half.interpolate(s[ok]), ntheta, nzeta, sine)
    err = np.linalg.norm(f - ref, axis=1) / np.linalg.norm(ref, axis=1)
    pair = lambda a: np.stack([[r[np.searchsorted(r, GLOBAL_KHZ) + d] for d in (-1, 0)] for r in a])  # noqa: E731
    low, up = pair(f), pair(ref)
    return err, np.abs(low / up - 1), s[ok]


class Run(NamedTuple):
    """FAX odd-sector modes of one resolution: kHz, ``G``-orthonormal vectors, dense ``F`` and ``G``, nodes, harmonics, band mask."""

    khz: np.ndarray
    y: np.ndarray
    F: np.ndarray
    G: np.ndarray
    s: np.ndarray
    harmonics: np.ndarray
    band: np.ndarray


def run(fx, nodes=49, harmonics=64, ntheta=64, nzeta=48):
    """All FAX modes in the sine basis of the ``harmonics`` AE3D harmonics with the largest band energy."""
    geo, h = geometry(fx, nodes), fx["mn"][fx["order"][:harmonics]].astype(float)
    F, G = sine_blocks(geo, h, ntheta, nzeta)
    f, y, Gd = solve(F, G)
    s = np.asarray(geo.s)
    band = band_modes(f, radial_weight(y), s, fx["sg_s"], fx["sg_khz"].astype(float))
    return Run(f, y, np.asarray(dense(F)), Gd, s, h, band)


def on_nodes(fx, r, vectors):
    """AE3D vectors ``(modes, 49, 162 or 96)`` on the nodes and harmonics of run ``r``: linear in ``s``, truncated to its harmonics."""
    cols = [np.flatnonzero((fx["mn"] == m).all(1))[0] for m in r.harmonics.astype(int)]
    full = vectors if vectors.shape[-1] == len(fx["mn"]) else None
    if full is None:  # band vectors keep the first 96 harmonics of ``order``
        cols = [list(fx["order"][: vectors.shape[-1]]).index(c) for c in cols]
    return np.stack([np.stack([np.interp(r.s, fx["ae_s"], a[:, c]) for c in cols], 1) for a in vectors])


def compare(fx, r):
    """The archived band and global mode against run ``r``: counts, Rayleigh quotients, containment, overlaps, localization."""
    g, weight = int(fx["band_global"]), radial_weight(r.y)
    if len(r.harmonics) <= fx["band_vec"].shape[-1]:
        ref = on_nodes(fx, r, fx["band_vec"])
    else:  # only the global mode is stored for all 162 harmonics
        ref, g = on_nodes(fx, r, fx["global_vec"][None]), 0
    rq = np.array([rayleigh_khz(r.F, r.G, a) for a in ref])
    ov = np.stack([overlaps(r.G, r.y, a) for a in ref])  # (AE3D modes, FAX modes)
    best = ov[g].argmax()
    A = ref.reshape(len(ref), -1).T.astype(complex)
    Q = A @ np.linalg.inv(np.linalg.cholesky(A.conj().T @ r.G @ A)).conj().T  # G-orthonormal AE3D subspace
    cos = np.linalg.svd(r.y[r.band].reshape(r.band.sum(), -1).conj() @ r.G @ Q, compute_uv=False)
    w_ae = radial_weight(ref[g][None])[0]
    return {
        "band_all": int(((r.khz > BAND[0]) & (r.khz < BAND[1])).sum()), "band": r.khz[r.band], "rq": rq, "global": g,
        "contained": ov[:, r.band].sum(1), "cos": cos, "best_khz": float(r.khz[best]), "best_ov": float(ov[g, best]),
        "ipr": float(1 / (weight[best] ** 2).sum()), "centroid": float(weight[best] @ r.s),
        "ipr_ae3d": float(1 / (w_ae**2).sum()), "centroid_ae3d": float(w_ae @ r.s), "spread": ov[g],
    }  # fmt: skip


def contained(small, ref, fx):
    """Fraction of the band subspace of run ``small`` inside that of ``ref`` (same harmonics order: ``small`` is zero-padded)."""
    pad = np.zeros((small.band.sum(), len(ref.s), len(ref.harmonics)), complex)
    cols = [np.flatnonzero((ref.harmonics == m).all(1))[0] for m in small.harmonics]
    for i, y in enumerate(small.y[small.band]):
        pad[i][:, cols] = np.stack(
            [np.interp(ref.s, small.s, y[:, c].real) + 1j * np.interp(ref.s, small.s, y[:, c].imag) for c in range(y.shape[1])], 1
        )
    A = pad.reshape(len(pad), -1).T
    Q = A @ np.linalg.inv(np.linalg.cholesky(A.conj().T @ ref.G @ A)).conj().T
    return np.linalg.svd(ref.y[ref.band].reshape(ref.band.sum(), -1).conj() @ ref.G @ Q, compute_uv=False)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def build(archive, out=FIXTURE):
    """Reduce the archive to ``out`` (about 0.6 MB); returns the SHA-256 of the archive and of the members read."""
    sha = {}
    with zipfile.ZipFile(archive) as z:
        digest = hashlib.sha256()
        with open(archive, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                digest.update(chunk)
        sha["Knyazev_JPP2026.zip"] = digest.hexdigest()
        raw = z.read(BOOZ)
        sha[BOOZ] = _sha(raw)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "boozmn.nc").write_bytes(raw)
            geo = fax.read_boozmn(Path(tmp) / "boozmn.nc")
        names = ("bmnc", "rmnc", "zmns", "pmns", "gmnc")
        arrays = {k: np.asarray(getattr(geo, k)) for k in names}
        scale = {k: np.abs(a).max() for k, a in arrays.items()}
        keep = np.any([np.abs(arrays[k]).max(0) > BOOZER_CUT * scale[k] for k in names], axis=0)
        fx = {f"g_{k}": a[:, keep].astype(np.float32) for k, a in arrays.items()}
        fx |= {"g_s": np.asarray(geo.s), "g_xm": np.asarray(geo.xm)[keep], "g_xn": np.asarray(geo.xn)[keep]}
        fx |= {f"g_{k}": np.asarray(getattr(geo, k)) for k in ("iota", "G", "I")} | {"g_nfp": np.asarray(geo.nfp)}
        rows = []
        for i in range(4):
            raw = z.read(QA + f"stellgap/alfven_spec{i}")
            sha[QA + f"stellgap/alfven_spec{i}"] = _sha(raw)
            rows.append(np.loadtxt(raw.decode().splitlines()))
        rows = np.vstack(rows)
        labels = np.unique(rows[:, 0])
        sg = np.array([np.sort(np.sqrt(rows[rows[:, 0] == x, 1] / rows[rows[:, 0] == x, 3])) for x in labels])
        fx |= {"sg_s": labels[::SG_STRIDE], "sg_khz": sg[::SG_STRIDE].astype(np.float32)}
        member = QA + "ae3d/egn_mode_asci.dat"
        digest = hashlib.sha256()
        with z.open(member) as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                digest.update(chunk)
        sha[member] = digest.hexdigest()
        with z.open(member) as f:
            n, h, ns = (int(f.readline()) for _ in range(3))
            mn = np.array([[int(f.readline()), int(f.readline())] for _ in range(h)])
            ev = np.array([float(f.readline()) for _ in range(n)])
            s = np.array([float(f.readline()) for _ in range(ns)])
            khz = np.sqrt(np.abs(ev)) * np.sign(ev)
            near = np.where((khz > BAND[0] - 1) & (khz < BAND[1] + 1))[0]
            vec = np.zeros((len(near), ns, h), np.float32)  # eigenvector j is one line per (node, harmonic), node-major
            for k in range(n):
                block = f.read(16 * n)
                if k in near:
                    vec[np.searchsorted(near, k)] = np.array(block.split(), dtype=float).reshape(ns, h)
    weight = (vec.astype(float) ** 2).sum(-1)
    weight /= weight.sum(-1, keepdims=True)
    chosen = band_modes(khz[near], weight, s, fx["sg_s"], fx["sg_khz"].astype(float))
    pick = near[chosen]
    energy = (vec[chosen].astype(float) ** 2).sum((0, 1))
    order = np.argsort(-energy)
    fx |= {"mn": mn.astype(np.int8), "ae_khz": khz.astype(np.float32), "ae_s": s, "order": order.astype(np.int16)}
    g = int(np.argmin(np.abs(khz[pick] - GLOBAL_KHZ)))
    fx |= {"band_khz": khz[pick].astype(np.float32), "band_vec": vec[chosen][:, :, order[:BAND_HARMONICS]]}
    fx |= {"band_global": np.asarray(g), "global_vec": vec[chosen][g]}
    np.savez_compressed(out, **fx)
    return sha
