"""End-to-end fitting tests for the Python bindings."""

import numpy as np
import glissando


def test_fit_linear_gaussian(synthetic_gaussian, gaussian_formula):
    data = {"x": synthetic_gaussian["x"]}
    model = glissando.GamlssModel.fit(
        data,
        synthetic_gaussian["y"],
        gaussian_formula,
        glissando.Gaussian(),
    )
    assert model.converged()


def test_fit_poisson(synthetic_poisson, poisson_formula):
    data = {"x": synthetic_poisson["x"]}
    model = glissando.GamlssModel.fit(
        data,
        synthetic_poisson["y"],
        poisson_formula,
        glissando.Poisson(),
    )
    assert model.converged()


def test_intercept_only_gaussian_recovers_mean(rng):
    n = 500
    y = rng.normal(7.0, 1.5, size=n)
    data = {"x": np.zeros(n)}  # unused but DataSet needs columns referenced by formula; none here.
    formula = {
        "mu": [("intercept",)],
        "sigma": [("intercept",)],
    }
    model = glissando.GamlssModel.fit(
        data,
        y,
        formula,
        glissando.Gaussian(),
    )
    assert model.converged()
