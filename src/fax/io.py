"""TOML cases, the :func:`run` driver, and the ``fax`` command line.

A case file has four tables (every key is optional except ``geometry.source``)::

    [geometry]  source = "boozmn" | "wout" | "vmec" | "circular", path, mboz, nboz, surfaces, ns, ftol, max_iter
    [modes]     n = [2], m_max = 12, m_min = 0
    [plasma]    density = 1e19 (ions/m^3), ion_mass = 2.0 (proton masses)
    [run]       task = "continuum" | "eigenmodes", output, ntheta, nzeta, target_khz (0: all modes), count
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
    "plasma": {"density": 1.0e19, "ion_mass": 1.0},
    "run": {"task": "continuum", "output": "outputs", "ntheta": 32, "nzeta": 32, "target_khz": 0.0, "count": 8},
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
        if table not in config or set(values) - set(config[table]):
            raise ValueError(f"unknown table or key in [{table}]: {sorted(set(values) - set(config.get(table, {})))}")
        config[table].update(values)
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
        "vmec": lambda: from_vmec_input(g["path"], g["ns"], g["ftol"], g["max_iter"], **booz),
        "circular": lambda: circular_torus(),
    }
    if g["source"] not in sources:
        raise ValueError(f"geometry.source must be one of {sorted(sources)}")
    return sources[g["source"]]()


def run(case: str | Path | dict[str, Any], save: bool = True) -> dict[str, Any]:
    """Run a case and return its results as NumPy arrays (also written to ``run.output``)."""
    config = load(case)
    r, m = config["run"], config["modes"]
    geo = geometry(config)
    harmonics = mode_family(tuple(m["n"]), m["m_max"], m["m_min"])
    rho = mass_density(config["plasma"]["density"], config["plasma"]["ion_mass"])
    out: dict[str, Any] = {"s": geo.s, "iota": geo.iota, "harmonics": harmonics}
    if r["task"] == "continuum":
        f = continuum(geo, harmonics, rho, r["ntheta"], r["nzeta"])
        center, width = gaps(f)
        out.update(frequencies_hz=f, gap_center_hz=center, gap_width_hz=width)
    elif r["task"] == "eigenmodes":
        target = 1e3 * r["target_khz"] if r["target_khz"] > 0 else None
        modes = eigenmodes(geo, harmonics, rho, target_hz=target, count=r["count"], ntheta=r["ntheta"], nzeta=r["nzeta"])
        out.update(
            mode_s=modes.s,
            frequencies_hz=modes.frequencies_hz[: r["count"]],
            structures=modes.structures[: r["count"]],
            continuum_hz=continuum(geo, harmonics, rho, r["ntheta"], r["nzeta"]),
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


def main(argv: list[str] | None = None) -> int:
    """``fax run case.toml [--plot]`` and ``fax plot outputs/case``."""
    parser = argparse.ArgumentParser(prog="fax", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="run a TOML case")
    run_parser.add_argument("case")
    run_parser.add_argument("--plot", action="store_true", help="also write figures")
    run_parser.add_argument(
        "--cache", nargs="?", const="~/.cache/fax/jax", help="persist JAX compilations in DIR (default ~/.cache/fax/jax)"
    )
    sub.add_parser("plot", help="plot a results folder").add_argument("folder")
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.cache:
            jax.config.update("jax_compilation_cache_dir", str(Path(args.cache).expanduser()))
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)  # the many small op-by-op compiles dominate
        out = run(args.case)
        folder = load(args.case)["run"]["output"]
        print(f"{Path(args.case).name}: {out['frequencies_hz'].shape} frequencies -> {folder}")
    if args.command == "plot" or getattr(args, "plot", False):
        from .plot import plot_results

        for path in plot_results(args.folder if args.command == "plot" else folder):
            print(path)
    return 0
