"""FAX: differentiable shear-Alfven continua and eigenmodes for stellarators in JAX."""

import jax

jax.config.update("jax_enable_x64", True)  # eigenvalues of stiff Alfven operators need float64

from .continuum import continuum, continuum_matrices, gap_location, gaps, mass_density, mode_family
from .drive import FastIons, growth_rate, ion_landau_damping
from .eigen import eigenmodes, eigenvalue_mean, frequency_hz, global_matrices
from .geometry import Geometry, circular_torus, from_vmec_input, from_wout, read_boozmn, surface_grid
from .io import load, run
from .objectives import (
    ModeClass,
    classify_modes,
    compression_harmonics,
    continuum_cost,
    design_frequencies,
    design_geometry,
    design_params,
    design_state,
    drive_risk,
    gap_closure,
    geometry_frequencies,
    geometry_growth_rates,
    prepare_design,
    soft_max,
    state_geometry,
    window_risk,
)

__version__ = "1.0.1"
__all__ = [
    "FastIons",
    "Geometry",
    "ModeClass",
    "circular_torus",
    "classify_modes",
    "compression_harmonics",
    "continuum",
    "continuum_cost",
    "continuum_matrices",
    "design_frequencies",
    "design_geometry",
    "design_params",
    "design_state",
    "drive_risk",
    "eigenmodes",
    "eigenvalue_mean",
    "frequency_hz",
    "from_vmec_input",
    "from_wout",
    "gap_closure",
    "geometry_frequencies",
    "geometry_growth_rates",
    "gap_location",
    "gaps",
    "global_matrices",
    "growth_rate",
    "ion_landau_damping",
    "load",
    "mass_density",
    "mode_family",
    "prepare_design",
    "read_boozmn",
    "run",
    "soft_max",
    "state_geometry",
    "surface_grid",
    "window_risk",
]
