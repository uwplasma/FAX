"""Precise-QA comparison with Knyazev et al. (2026): build the offline fixture from the Zenodo archive, then compare FAX.

Data: DOI 10.5281/zenodo.21299907 (version 10.5281/zenodo.21299908), ``Knyazev_JPP2026.zip``, CC-BY-4.0, 1.13 GB.
Download it from zenodo.org yourself, set ``ARCHIVE`` and run ``python examples/knyazev_qa.py``. The archive is data: members are
read with ``zipfile`` and parsed as text, nothing in it is executed. ``REBUILD`` regenerates ``tests/data/knyazev_qa.npz``
(and prints the SHA-256 values for ``tests/data/PROVENANCE.md``); ``STUDY`` runs the FAX comparison from the fixture
(about 15 minutes and 8 GB on one laptop); ``FIGURE`` is the PNG of ``docs/validation.md``.

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

ARCHIVE = Path("Knyazev_JPP2026.zip")
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "data" / "knyazev_qa.npz"
FIGURE = Path(__file__).resolve().parents[1] / "docs" / "_static" / "knyazev_qa.png"
REBUILD, STUDY = False, True
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


def build(archive=ARCHIVE, out=FIXTURE):
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


CONFIGS = [
    (49, 24, 48, 32),
    (49, 40, 48, 32),
    (49, 64, 48, 32),
    (49, 64, 64, 48),
    (49, 96, 48, 32),
    (49, 162, 48, 32),
    (24, 40, 48, 32),
    (99, 40, 48, 32),
]
PR9_KHZ = {"FAX, physical indices": 27.57, "public AE3D formulas": 30.30}  # PR #9 diagnostic reconstruction (not rerun here)


def study(fx):
    """Print the tables of ``docs/validation.md`` and return what the figure needs (about 15 minutes, 8 GB)."""
    err, edge, s = continuum_error(fx, 64, 48)
    err2 = continuum_error(fx, 112, 72, sine=True)[0]
    print(f"continuum, {len(s)} surfaces: complex basis mean {err.mean():.1e} p95 {np.percentile(err, 95):.1e} max {err.max():.1e},")
    print(f"  sine sector mean {err2.mean():.1e} max {err2.max():.1e}, gap edges (s > 0.25) max {edge[s > 0.25].max():.1e}")
    runs, rows = {}, {}
    for cfg in CONFIGS:
        r = run(fx, *cfg)
        c = rows[cfg] = compare(fx, r)
        shift = c["rq"] / (fx["band_khz"] if len(c["rq"]) > 1 else GLOBAL_KHZ) - 1
        print(
            f"nodes {cfg[0]} harmonics {cfg[1]} grid {cfg[2]}x{cfg[3]}: {c['band_all']} modes in {BAND}, {len(c['band'])} band modes",
            end="",
        )
        print(
            f" {c['band'].min():.1f}-{c['band'].max():.1f} kHz; global: shift {shift[c['global']]:+.4f}, best {c['best_khz']:.2f} kHz",
            end="",
        )
        print(
            f" ({c['best_ov']:.2f}), IPR {c['ipr']:.1f} vs {c['ipr_ae3d']:.1f}, s {c['centroid']:.2f} vs {c['centroid_ae3d']:.2f};", end=""
        )
        print(f" band: shift {shift.mean():+.4f}, contained {c['contained'].mean():.2f}")
        runs[cfg] = r._replace(F=None, y=r.y if cfg[1] < 162 else None, G=r.G if cfg[1] < 162 else None)
    base = runs[(49, 64, 48, 32)]
    for name, small, ref in (
        ("grid 48x32 vs 64x48", base, runs[(49, 64, 64, 48)]),
        ("harmonics 40 vs 64", runs[(49, 40, 48, 32)], base),
        ("harmonics 64 vs 96", base, runs[(49, 96, 48, 32)]),
        ("nodes 24 vs 49", runs[(24, 40, 48, 32)], runs[(49, 40, 48, 32)]),
        ("nodes 49 vs 99", runs[(49, 40, 48, 32)], runs[(99, 40, 48, 32)]),
    ):
        cos = contained(small, ref, fx)
        print(f"FAX band subspace in itself, {name}: {(cos**2).sum() / small.band.sum():.2f} (smallest cosine {cos.min():.2f})")
    return runs, rows


def figure(fx, runs, rows, path=FIGURE):
    """Continuum and band (a), the archived global mode in FAX's eigenbasis (b), convergence with the harmonics (c)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
    s = fx["sg_s"] - 0.02
    sg = fx["sg_khz"].astype(float)
    ax[0].plot(np.repeat(s, sg.shape[1]), sg.ravel(), ",", color="0.6", label="STELLGAP")
    f = continuum_khz(fx, geometry_half(fx).interpolate(s))
    ax[0].plot(np.repeat(s, f.shape[1]), f.ravel(), ",", color="C1", alpha=0.5, label="FAX")
    r = runs[(49, 64, 48, 32)]
    w = radial_weight(r.y[r.band])
    ax[0].plot(w @ r.s, r.khz[r.band], "o", ms=3, color="C3", label=f"FAX gap band ({r.band.sum()})")
    ref = fx["band_vec"].astype(float)
    ax[0].plot(radial_weight(ref) @ fx["ae_s"], fx["band_khz"], "kx", ms=4, label=f"AE3D gap band ({len(ref)})")
    ax[0].set(xlabel="$s$", ylabel="kHz", ylim=(0, 60), title="(a) continuum and gap band")
    ax[0].legend(markerscale=4, fontsize=7, loc="upper right")
    for hl, color in ((64, "C0"), (162, "C3")):
        c = rows[(49, hl, 48, 32)]
        ax[1].vlines(runs[(49, hl, 48, 32)].khz, 0, c["spread"], color=color, lw=1.2, label=f"{hl} harmonics")
    ax[1].axvline(GLOBAL_KHZ, color="k", ls="--", lw=1, label="archived AE3D 30.31")
    for (name, v), ls in zip(PR9_KHZ.items(), (":", "-."), strict=True):
        ax[1].axvline(v, color="0.4", ls=ls, lw=1, label=f"{name} {v}")
    ax[1].set(
        xlim=(26, 33),
        xlabel="FAX eigenfrequency (kHz)",
        ylabel="overlap with the archived global mode",
        title="(b) one mode, many FAX modes",
    )
    ax[1].legend(fontsize=6)
    hls = sorted({k[1] for k in rows if k[0] == 49 and k[2] == 48})
    ax[2].plot(
        hls, [100 * (rows[(49, h, 48, 32)]["rq"][rows[(49, h, 48, 32)]["global"]] / GLOBAL_KHZ - 1) for h in hls], "o-", label="global mode"
    )
    ax[2].plot(
        [h for h in hls if h <= 96],
        [100 * (rows[(49, h, 48, 32)]["rq"] / fx["band_khz"] - 1).mean() for h in hls if h <= 96],
        "s-",
        label="band mean",
    )
    ax[2].set(xscale="log", xlabel="harmonics kept", ylabel="Rayleigh-quotient shift (%)", title="(c) energy of the AE3D modes in FAX")
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].legend(fontsize=7)
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    if REBUILD:
        for name, value in build().items():
            print(value, name)
    print("fixture", FIXTURE.stat().st_size, "bytes")
    if STUDY:
        figure(load(), *study(load()))
