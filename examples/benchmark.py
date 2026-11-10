"""Equal-accuracy timings of the FAX eigensolvers on one assembled pencil ``F y = lam G y``.

Run ``python examples/benchmark.py`` on a quiet machine (each job waits up to ``WAIT_MAX`` seconds for a 1-minute load below
``LOAD_MAX``; the loads are reported); edit the settings below. Methods:

* ``targeted``: ``fax.eigenmodes(target_hz, count)``, SOLVAX shift-invert on the block-tridiagonal pencil;
* ``dense-jax``: ``fax.eigenmodes()``, FAX's jitted dense generalized solve of all modes (Cholesky + ``eigh``);
* ``dense-scipy``: ``scipy.linalg.eigh(F, G)`` of all modes, the reference.

Each (case, method) runs in a fresh spawned process, so ``cold`` (compile + first call) and the peak RSS (above the
post-import baseline, compile included) belong to that method alone; ``warm`` is the median of ``REPS`` synchronized calls
(``block_until_ready``). The forward pass includes assembly. ``eta`` is the largest ``||F y - lam G y|| / ((||F|| + |lam| ||G||)
||y||)`` of the requested modes (infinity norm of the matrices), recomputed here from the returned vectors; ``err`` is the
relative eigenvalue error against ``dense-scipy`` and ``gerr`` the relative L2 gradient difference against ``dense-jax``.
``busy thr`` is CPU time over wall time of the warm forward calls: ``THREADS`` pins BLAS/LAPACK, but XLA's own dots may
still use several cores on this jaxlib. Speed-ups are printed only when both methods reach ``TOL``. The gradient is that of
``frequency_hz`` (mean of the requested modes) with respect to every floating-point geometry array.
"""

import os

THREADS, REPS, SLOW, TOL, LOAD_MAX, WAIT_MAX, DENSE_MAX, ITERS = (
    1,
    5,
    30.0,
    1e-10,
    4.0,
    120,
    5000,
    1000,
)  # SLOW: seconds above which REPS falls to 2
CASES = [  # (label, kind, N, H, requested modes); N = 0 takes the file's surfaces (CTOK) or the fixture's (49)
    ("CTOK n=2", "ctok", 0, 13, 6),
    ("Koenies N=200", "koenies", 200, 12, 3),
    ("Koenies N=400", "koenies", 400, 12, 3),
    ("Knyazev QA", "knyazev", 49, 64, 6),
]
BASE = ("koenies", 200, 12, 3)  # scaling sweeps vary one of (N, H, modes) at a time from here
SWEEPS = {"N": (200, 400, 800, 1600, 3200), "H": (12, 16, 24, 32, 48), "modes": (1, 3, 6, 12, 24)}  # N H > 2048: no dense fallback

os.environ.update(  # before numpy and jax load; spawned workers inherit them
    OMP_NUM_THREADS=str(THREADS),
    OPENBLAS_NUM_THREADS=str(THREADS),
    VECLIB_MAXIMUM_THREADS=str(THREADS),
    XLA_FLAGS=f"--xla_cpu_multi_thread_eigen={THREADS > 1} intra_op_parallelism_threads={THREADS}".replace("True", "true").replace(
        "False", "false"
    ),
)

import platform  # noqa: E402
import resource  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jaxlib  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402
import scipy.linalg  # noqa: E402
import solvax  # noqa: E402

import fax  # noqa: E402
from fax.eigen import band, dense, global_matrices  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def build(kind, N, H):
    """``(geometry, harmonics, density, target Hz, grid)`` of a case."""
    if kind == "ctok":
        geo = fax.read_boozmn(ROOT / "tests" / "data" / "boozmn_ctok.nc").full_mesh()
        return geo, fax.mode_family(2, 12), fax.mass_density(1e19, 2 * 1.00727647), 384e3, dict(ntheta=32, nzeta=4)
    if kind == "koenies":  # Koenies et al. (2018) ITPA TAE, n = 6, q = 1.71 + 0.16 r^2; m = 5 .. 4 + H on a grid that resolves them
        s = (jnp.arange(N) + 0.5) / N
        geo = fax.circular_torus(s=s, major_radius=10.0, minor_radius=1.0, B0=3.0, iota=1 / (1.71 + 0.16 * s), epsilon_B=0.1)
        return geo, fax.mode_family(6, 4 + H, 5), fax.mass_density(2e19, 1.00727647), 6.46e4, dict(ntheta=32 * -(-H // 12), nzeta=4)
    from knyazev_qa import RHO, geometry, load  # examples/ is on sys.path

    fx = load()
    return geometry(fx, N), jnp.asarray(fx["mn"][fx["order"][:H]], dtype=float), RHO, 30.3e3, dict(ntheta=64, nzeta=48)


def rss():
    """Peak RSS of this process in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e6 if sys.platform == "darwin" else 1e3)


def cpu():
    """CPU seconds used by this process so far."""
    u = resource.getrusage(resource.RUSAGE_SELF)
    return u.ru_utime + u.ru_stime


def timed(f, *args):
    """``(result, cold, warm, reps, threads)``: compile + first call, the median of ``REPS`` synchronized calls (2 if the
    first is slow) and the mean number of busy threads (CPU time over wall time) of those calls."""
    t = time.perf_counter()
    out = jax.block_until_ready(f(*args))
    cold, runs, c = time.perf_counter() - t, [], cpu()
    for _ in range(REPS if cold < SLOW else 2):
        t = time.perf_counter()
        jax.block_until_ready(f(*args))
        runs.append(time.perf_counter() - t)
    return out, cold, statistics.median(runs), len(runs), (cpu() - c) / sum(runs)


def eta(F, G, lam, y):
    """Largest relative residual of ``F y = lam G y`` for ``y`` of shape ``(modes, N, H)`` (blocks ``F``, ``G``)."""
    y = jnp.moveaxis(y, 0, -1)
    mv = lambda A: solvax.block_tridiag_matvec(*band(A), y)  # noqa: E731
    norm = lambda A: jnp.max(sum(jnp.abs(b).sum(-1) for b in band(A)))  # noqa: E731 - infinity norm
    r = jnp.linalg.norm((mv(F) - lam * mv(G)).reshape(-1, len(lam)), axis=0)
    return float(jnp.max(r / ((norm(F) + jnp.abs(lam) * norm(G)) * jnp.linalg.norm(y.reshape(-1, len(lam)), axis=0))))


def work(job):
    """One (case, method) in this process: forward and gradient timings, peak memory, then accuracy and solve-only diagnostics."""
    kind, N, H, count, method = job
    geo, h, rho, target, grid = build(kind, N, H)
    N, H = geo.s.size, h.shape[0]
    jax.block_until_ready(geo)
    base, w0 = rss(), (2 * np.pi * target) ** 2
    opts = dict(target_hz=target, count=count, iters=ITERS, **grid) if method == "targeted" else grid
    pick = lambda w: np.sort(np.argsort(np.abs(np.asarray(w) - w0))[:count])  # noqa: E731 - the modes nearest the target
    asm = jax.jit(lambda g: tuple(dense(b) for b in global_matrices(g, h, rho, **grid)))
    fwd = {
        "targeted": lambda: jax.jit(lambda g: fax.eigenmodes(g, h, rho, **opts)),
        "dense-jax": lambda: jax.jit(lambda g: fax.eigenmodes(g, h, rho, **opts)),
        "dense-scipy": lambda: lambda g: scipy.linalg.eigh(*map(np.asarray, asm(g))),
    }[method]()
    r, cold, warm, reps, threads = timed(fwd, geo)
    out = dict(N=N, H=H, count=count, base=base, grid=grid, cold=cold, warm=warm, reps=reps, threads=threads, mem=rss() - base)
    if method == "dense-scipy":  # (eigenvalues, eigenvectors): the modes nearest the target
        lam, y = r[0][pick(r[0])], r[1][:, pick(r[0])].T.reshape(count, N, H)
    else:
        idx = slice(None) if method == "targeted" else pick(r.omega2)
        lam, y = r.omega2[idx], r.structures[idx]
        start = int(idx[0]) if method == "dense-jax" else 0  # the nearest modes are contiguous in the sorted spectrum
        grad = jax.jit(jax.grad(lambda g: fax.frequency_hz(g, h, rho, start, start + count, **opts), allow_int=True))
        gr, out["gcold"], out["gwarm"], *_ = timed(grad, geo)
        out["gmem"] = rss() - base
        out["grad"] = np.concatenate([np.ravel(a) for a in jax.tree_util.tree_leaves(gr) if a.dtype.kind == "f"])
        out["temp"] = fwd.lower(geo).compile().memory_analysis().temp_size_in_bytes / 1e6  # XLA static buffers of the forward pass
    F, G = global_matrices(geo, h, rho, **grid)
    out["lam"], out["eta"] = np.asarray(lam), eta(F, G, jnp.asarray(lam), jnp.asarray(y))
    if method == "targeted":
        solve = jax.jit(lambda F, G: fax.eigen.targeted_eigenpairs(F, G, w0, count, ITERS)[0])
        fac = jax.jit(lambda F, G: solvax.block_thomas_factor(*(f - w0 * g for f, g in zip(band(F), band(G), strict=True))))
        out["factor"] = timed(fac, F, G)[2]  # one factorization of F - sigma G: O(N H^3) work, O(N H^2) storage
        out["fmb"] = sum(a.nbytes for a in jax.tree_util.tree_leaves(fac(F, G))) / 1e6
        out["solve"], out["iters"] = timed(solve, F, G)[2], int(r.status.iterations)
        out["stemp"] = solve.lower(F, G).compile().memory_analysis().temp_size_in_bytes / 1e6
    return out


def wait():
    """Block until the 1-minute load average is below ``LOAD_MAX`` (at most ``WAIT_MAX`` seconds); return it."""
    t = time.time()
    while (load := os.getloadavg()[0]) >= LOAD_MAX and time.time() - t < WAIT_MAX:
        time.sleep(10)
    return load


def fit(rows, key, var):
    """Observed exponent of ``key`` against ``var`` (least squares in log-log)."""
    return np.polyfit(np.log([r[var] for r in rows]), np.log([max(r[key], 1e-6) for r in rows]), 1)[0]


def gerr(r, g0):
    """Relative L2 difference of the gradient of ``r`` from the reference gradient ``g0``."""
    return np.linalg.norm(r["grad"] - g0) / np.linalg.norm(g0) if g0 is not None and "grad" in r else np.nan


def show(head, rows):
    """Print a right-aligned table; numbers get three significant digits (``nan`` where not applicable)."""
    cells = [head] + [[c if isinstance(c, str) else f"{c:.3g}" for c in row] for row in rows]
    for row in cells:
        print("  ".join(c.rjust(max(len(x[i]) for x in cells)) if i else c.ljust(max(len(x[0]) for x in cells)) for i, c in enumerate(row)))


if __name__ == "__main__":
    pool, loads = get_context("spawn").Pool(1, maxtasksperchild=1), []

    def run(case, method):
        loads.append(wait())
        return {**pool.apply(work, ((*case, method),)), "load": loads[-1]}

    print(
        f"fax {fax.__version__}, solvax {solvax.__version__}, jax {jax.__version__}, jaxlib {jaxlib.__version__}, "
        f"numpy {np.__version__}, scipy {scipy.__version__}; python {platform.python_version()}, {platform.machine()} {platform.system()}"
    )
    print(
        f"{jax.devices()[0].device_kind} ({jax.default_backend()}), {os.cpu_count()} cores, {THREADS} BLAS thread(s), float64, TOL {TOL:g}"
    )
    nan = np.nan
    for label, *case in CASES:
        res = {
            m: run(case, m) for m in ("dense-scipy", "dense-jax", "targeted") if m == "targeted" or (case[1] or 100) * case[2] <= DENSE_MAX
        }
        ref, g0 = res.get("dense-scipy", res["dense-jax"]), res.get("dense-jax", {}).get("grad")
        r0 = res["targeted"]
        for r in res.values():
            r["err"], r["gerr"] = np.max(np.abs(r["lam"] / ref["lam"] - 1)), gerr(r, g0)
        print(f"\n{label}: N {r0['N']}, H {r0['H']}, {r0['count']} modes, angle grid {r0['grid']}, warm = median of {REPS} (fewer if slow)")
        cols = {
            "eta": lambda r: r["eta"],
            "err": lambda r: r["err"],
            "gerr": lambda r: r["gerr"],
            "fwd cold": lambda r: r["cold"],
            "fwd warm": lambda r: r["warm"],
            "busy thr": lambda r: r["threads"],
            "solve": lambda r: r.get("solve", nan),
            "grad cold": lambda r: r.get("gcold", nan),
            "grad warm": lambda r: r.get("gwarm", nan),
            "RSS f/g MB": lambda r: f"{r['mem']:.0f}/{r.get('gmem', nan):.0f}",
            "XLA MB": lambda r: r.get("temp", nan),
            "iters": lambda r: r.get("iters", 0),
            "load": lambda r: r["load"],
        }
        show(["method", *cols], [[m, *(f(r) for f in cols.values())] for m, r in res.items()])
        if any(r["reps"] < REPS for r in res.values()):
            print("warm forward reps: " + ", ".join(f"{m} {r['reps']}" for m, r in res.items()))
        if "dense-jax" in res and max(res["dense-jax"]["eta"], r0["eta"]) <= TOL:
            d = res["dense-jax"]
            print(
                f"targeted vs dense-jax at eta <= {TOL:g}: forward x{d['warm'] / r0['warm']:.1f}, gradient x{d['gwarm'] / r0['gwarm']:.1f}"
            )
    print(f"\nscaling (targeted, from {BASE[0]} N={BASE[1]} H={BASE[2]} modes={BASE[3]}); factor, solve: SOLVAX on assembled blocks")
    cols = dict(N="N", H="H", modes="count", eta="eta", iters="iters", factor="factor", solve="solve")
    cols |= {"fwd warm": "warm", "grad warm": "gwarm", "factor MB": "fmb", "solve MB": "stemp", "RSS MB": "mem", "load": "load"}
    rows = {
        var: [run((BASE[0], *[v if k == var else b for k, b in zip(SWEEPS, BASE[1:], strict=True)]), "targeted") for v in values]
        for var, values in SWEEPS.items()
    }
    show(["var", *cols], [[var, *(r[c] for c in cols.values())] for var, rs in rows.items() for r in rs])
    for k, rs in rows.items():
        names = {"N": "N", "H": "H", "modes": "count"}
        ok = [r for r in rs if r["eta"] <= TOL]  # only points that reached the tolerance
        fits = {n: fit(ok, cols[n], names[k]) for n in ("factor", "solve", "fwd warm", "grad warm", "factor MB", "solve MB", "RSS MB")}
        print(f"observed exponent in {k} (eta <= TOL only): " + ", ".join(f"{n} {e:.2f}" for n, e in fits.items()))
    print("expected: factorization O(N H^3), factor storage O(N H^2); solve = factor + iterations x O(N H^2 (2 modes + 8))")
    print(f"load before each run: min {min(loads):.1f}, median {np.median(loads):.1f}, max {max(loads):.1f}, {len(loads)} runs,")
    print(f"{sum(x >= LOAD_MAX for x in loads)} not below {LOAD_MAX:g}; RSS is above a baseline of {r0['base']:.0f} MB")
