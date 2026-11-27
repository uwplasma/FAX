"""TOML cases, the :func:`run` driver, and the ``fax`` command line.

A case file has four tables; every key is optional and takes the default shown (``path`` is required for files)::

    [geometry]  source = "circular" | "boozmn" | "wout" | "vmec", path = none, mboz = 8, nboz = 8,
                surfaces = [0.1, ..., 0.9], ns = none, ftol = 1e-12, max_iter = 5000
    [modes]     n = [1], m_max = 8, m_min = 0
    [plasma]    density = 1e19 (ions/m^3), ion_mass = 1.0 (amu), density_profile = [1.0] (polynomial in s)
    [run]       task = "continuum" | "eigenmodes", output = "outputs", ntheta = none, nzeta = none (none: the API
                default grid), target_khz = 0 (0: all modes), count = 8

A scalar is accepted where a list is expected. The full reference: docs/getting-started.md, "TOML reference".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np

try:
    import tomllib
except ImportError:  # Python 3.10
    import tomli as tomllib

from .continuum import continuum, gaps, mass_density, mode_family
from .eigen import eigenmodes
from .geometry import Geometry, circular_torus, from_vmec_input, from_wout, read_boozmn

DEFAULTS: dict[str, dict[str, Any]] = {
    "geometry": {
        "source": "circular",
        "path": None,
        "mboz": 8,
        "nboz": 8,
        "surfaces": [0.1 * i for i in range(1, 10)],
        "ns": None,
        "ftol": 1e-12,
        "max_iter": 5000,
    },
    "modes": {"n": [1], "m_max": 8, "m_min": 0},
    "plasma": {"density": 1.0e19, "ion_mass": 1.0, "density_profile": [1.0]},
    "run": {"task": "continuum", "output": "outputs", "ntheta": None, "nzeta": None, "target_khz": 0.0, "count": 8},
}


def load(case: str | Path | dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Merge a TOML file (or mapping) over :data:`DEFAULTS`, rejecting unknown keys.

    Relative ``geometry.path`` and ``run.output`` are resolved against the case file.
    """
    root = Path.cwd()
    if not isinstance(case, dict):
        root = Path(case).expanduser().resolve().parent
        case = tomllib.loads(Path(case).expanduser().read_text())
    config = {table: dict(values) for table, values in DEFAULTS.items()}
    for table, values in case.items():
        bad = sorted(set(values) - set(config.get(table, {}))) if isinstance(values, dict) else [table]
        if bad or table not in config:
            raise ValueError(f"unknown table or key in [{table}]: {bad}; valid keys: {', '.join(sorted(config.get(table, config)))}")
        for key, value in values.items():
            default = DEFAULTS[table][key]
            if isinstance(default, list):
                value = list(value) if isinstance(value, (list, tuple)) else [value]
            kind = (int, float) if isinstance(default, float) else type(default)
            if default is not None and (not isinstance(value, kind) or isinstance(value, bool) != isinstance(default, bool)):
                raise ValueError(f"[{table}] {key} must be {type(default).__name__}, got {value!r}")
            config[table][key] = value
    for table, key in (("geometry", "path"), ("run", "output")):
        if config[table][key] is not None:
            config[table][key] = str((root / Path(config[table][key]).expanduser()).resolve())
    return config


def geometry(config: dict[str, dict[str, Any]]) -> Geometry:
    """Build the geometry described by ``config["geometry"]``."""
    g = config["geometry"]
    booz = {"mboz": g["mboz"], "nboz": g["nboz"], "surfaces": tuple(g["surfaces"])}
    sources = {
        "boozmn": lambda: read_boozmn(g["path"]),
        "wout": lambda: from_wout(g["path"], **booz),
        "vmec": lambda: from_vmec_input(g["path"], ns=g["ns"], ftol=g["ftol"], max_iter=g["max_iter"], **booz),
        "circular": circular_torus,
    }
    if g["source"] not in sources:
        raise ValueError(f"geometry.source must be one of {sorted(sources)}")
    if g["source"] != "circular" and g["path"] is None:
        raise ValueError(f"geometry.path is required for source = {g['source']!r}")
    return sources[g["source"]]()


def run(case: str | Path | dict[str, Any], *, save: bool = True) -> dict[str, Any]:
    """Run a case and return its results as NumPy arrays (also written to ``run.output``)."""
    config = load(case)
    r, m = config["run"], config["modes"]
    geo = geometry(config)
    harmonics = mode_family(tuple(m["n"]), m["m_max"], m["m_min"])
    p = config["plasma"]
    grid = {key: r[key] for key in ("ntheta", "nzeta") if r[key] is not None}
    rho = mass_density(p["density"] * np.polyval(p["density_profile"][::-1], np.asarray(geo.s)), p["ion_mass"])
    out: dict[str, Any] = {"s": geo.s, "iota": geo.iota, "harmonics": harmonics}
    if r["task"] == "continuum":
        f = continuum(geo, harmonics, rho, **grid)
        center, width = gaps(f)
        out.update(frequencies_hz=f, gap_center_hz=center, gap_width_hz=width)
    elif r["task"] == "eigenmodes":
        target = 1e3 * r["target_khz"] if r["target_khz"] > 0 else None
        modes = eigenmodes(geo, harmonics, rho, target_hz=target, count=r["count"], **grid)
        out.update(
            mode_s=modes.s,
            frequencies_hz=modes.frequencies_hz[: r["count"]],
            structures=modes.structures[: r["count"]],
            continuum_hz=continuum(geo, harmonics, rho, **grid),
        )
    else:
        raise ValueError("run.task must be 'continuum' or 'eigenmodes'")
    out = {key: np.asarray(value) for key, value in out.items()}
    if save:
        folder = Path(r["output"])
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "results.npz", **out)
        (folder / "case.json").write_text(json.dumps(config, indent=2))
    return out


def enable_cache(path: str | Path | None = None) -> str:
    """Persist JAX compilations on disk (``~/.cache/fax/jax`` by default) so a later process skips them; returns the directory.

    A cache directory the user already set (``jax_compilation_cache_dir`` or ``JAX_COMPILATION_CACHE_DIR``) is kept unless ``path``
    is given. Not called at import: ``fax run`` and the examples call it.
    """
    current = jax.config.jax_compilation_cache_dir
    if path is None and current:
        return str(current)
    folder = str(Path(path or "~/.cache/fax/jax").expanduser())
    jax.config.update("jax_compilation_cache_dir", folder)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)  # the many small op-by-op compiles dominate
    return folder


def main(argv: list[str] | None = None) -> int:
    """``fax run case.toml [--plot]`` and ``fax plot outputs/case``."""
    parser = argparse.ArgumentParser(prog="fax", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traceback", action="store_true", help="show the Python traceback of an input error")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="run a TOML case", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    run_parser.add_argument("case")
    run_parser.add_argument("--traceback", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    run_parser.add_argument("--plot", action="store_true", help="also write figures")
    run_parser.add_argument("--cache", nargs="?", metavar="DIR", help="persist JAX compilations in DIR (default ~/.cache/fax/jax, on)")
    run_parser.add_argument("--no-cache", action="store_true", help="do not persist JAX compilations")
    sub.add_parser("plot", help="plot a results folder").add_argument("folder")
    args = parser.parse_args(argv)
    try:
        return _main(args)
    except (ValueError, OSError) as error:  # bad TOML, key, type, value or grid; missing file
        if args.traceback:
            raise
        parser.exit(2, f"fax: error: {error} (--traceback shows where)\n")


def _main(args: argparse.Namespace) -> int:
    if args.command == "run":
        if not args.no_cache:
            enable_cache(args.cache)
        out = run(args.case)
        folder = load(args.case)["run"]["output"]
        print(f"{Path(args.case).name}: {out['frequencies_hz'].shape} frequencies -> {folder}")
    if args.command == "plot" or getattr(args, "plot", False):
        from .plot import plot_results

        for path in plot_results(args.folder if args.command == "plot" else folder):
            print(path)
    return 0
