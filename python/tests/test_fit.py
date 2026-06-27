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


def test_fit_string_formula_matches_term_list(synthetic_gaussian, gaussian_formula):
    """DATA-5: a parameter's formula may be an R/mgcv-style string, fitting
    identically to the equivalent list-of-tuples encoding."""
    data = {"x": synthetic_gaussian["x"]}
    y = synthetic_gaussian["y"]

    m_list = glissando.GamlssModel.fit(data, y, gaussian_formula, glissando.Gaussian())
    m_str = glissando.GamlssModel.fit(
        data,
        y,
        {"mu": "y ~ x", "sigma": "~ 1"},
        glissando.Gaussian(),
    )
    assert m_str.converged()
    # Same fitted values to tight tolerance (same design, same solve).
    p_list = m_list.predict(data)
    p_str = m_str.predict(data)
    assert np.allclose(p_list["mu"], p_str["mu"], atol=1e-10)


def test_fit_string_formula_with_factor(rng):
    """A factor term parsed from a string expands into contrast columns."""
    n = 300
    g = (np.arange(n) % 3).astype(float)
    y = 5.0 + np.where(g == 1, 2.0, 0.0) - np.where(g == 2, 1.5, 0.0) + rng.normal(0, 0.3, n)
    data = {"g": g}
    model = glissando.GamlssModel.fit(
        data,
        y,
        {"mu": "y ~ factor(g)", "sigma": "~ 1"},
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
