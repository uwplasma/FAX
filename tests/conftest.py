"""pytest >= 9.2 walks ``approx`` expected values with ``.flat``, which JAX arrays do not implement: compare via NumPy.

vmex >= 0.15 splits its implicit-solve probes over threads; under ``coverage run`` the threads deadlock in the jitted
probe (reproduced on the Linux runners), so the tests run them on one thread, which is also faster on these sizes.

The shipped ``boozmn_ctok.nc`` lacks its outer half surface, and ``read_boozmn`` warns on every read; that one warning is
ignored per test (a mark, so it also holds under ``-W error``), and ``test_continuum.py`` asserts it with ``pytest.warns``."""

import jax
import numpy as np
import pytest

try:
    from vmex.core import implicit

    implicit._MAX_PROBE_WORKERS = 1
except ImportError:  # vmex is optional
    pass

_approx = pytest.approx


def _numpy_approx(expected, *args, **kwargs):
    return _approx(np.asarray(expected) if isinstance(expected, jax.Array) else expected, *args, **kwargs)


pytest.approx = _numpy_approx

_CTOK_EDGE = pytest.mark.filterwarnings(r"ignore:boozmn_ctok\.nc. the outer half surface .* is not in the file:UserWarning")


def pytest_collection_modifyitems(items):
    """Ignore the known warning of the CTOK fixture in every test."""
    for item in items:
        item.add_marker(_CTOK_EDGE)
