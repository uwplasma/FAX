"""pytest >= 9.2 walks ``approx`` expected values with ``.flat``, which JAX arrays do not implement: compare via NumPy."""

import jax
import numpy as np
import pytest

_approx = pytest.approx


def _numpy_approx(expected, *args, **kwargs):
    return _approx(np.asarray(expected) if isinstance(expected, jax.Array) else expected, *args, **kwargs)


pytest.approx = _numpy_approx
