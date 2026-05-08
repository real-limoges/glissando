"""Shared pytest fixtures for the Python FFI test suite."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_gaussian(rng):
    """Linear Gaussian: y ~ N(2 + 1.5 x, 0.5)."""
    n = 100
    x = np.linspace(0.0, 5.0, n)
    y = 2.0 + 1.5 * x + rng.normal(0.0, 0.5, size=n)
    return {"y": y, "x": x}


@pytest.fixture
def synthetic_poisson(rng):
    """Poisson: y ~ Poisson(exp(0.5 + 0.3 x))."""
    n = 100
    x = np.linspace(0.0, 4.0, n)
    mu = np.exp(0.5 + 0.3 * x)
    y = rng.poisson(mu).astype(float)
    return {"y": y, "x": x}


@pytest.fixture
def gaussian_formula():
    return {
        "mu": [("intercept",), ("linear", "x")],
        "sigma": [("intercept",)],
    }


@pytest.fixture
def poisson_formula():
    return {"mu": [("intercept",), ("linear", "x")]}
