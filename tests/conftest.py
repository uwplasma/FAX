"""pytest >= 9.2 walks ``approx`` expected values with ``.flat``, which JAX arrays do not implement: compare via NumPy.

vmex >= 0.15 splits its implicit-solve probes over threads; under ``coverage run`` the threads deadlock in the jitted
probe (reproduced on the Linux runners), so the tests run them on one thread, which is also faster on these sizes."""

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
