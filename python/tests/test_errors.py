"""Error-mapping tests: bad inputs surface as Python exceptions."""

import numpy as np
import pytest
import glissando


def test_unknown_family_object_rejected(synthetic_gaussian, gaussian_formula):
    data = {"x": synthetic_gaussian["x"]}
    with pytest.raises((ValueError, TypeError)):
        glissando.GamlssModel.fit(
            data,
            synthetic_gaussian["y"],
            gaussian_formula,
            object(),  # deliberately not a distribution
        )


def test_non_finite_response_rejected_with_na_fail(gaussian_formula):
    """Under na_action='fail' a missing response is a hard error."""
    n = 50
    y = np.linspace(0.0, 1.0, n)
    y[5] = np.nan
    data = {"x": np.linspace(0.0, 1.0, n)}
    with pytest.raises(RuntimeError):
        glissando.GamlssModel.fit_with_config(
            data,
            y,
            gaussian_formula,
            glissando.Gaussian(),
            {"na_action": "fail"},
        )


def test_non_finite_response_dropped_by_default(gaussian_formula):
    """The default na_action drops the missing row and fits the rest."""
    n = 50
    y = np.linspace(0.0, 1.0, n)
    y[5] = np.nan
    data = {"x": np.linspace(0.0, 1.0, n)}
    model = glissando.GamlssModel.fit(data, y, gaussian_formula, glissando.Gaussian())
    assert model.converged()


def test_missing_referenced_column_rejected():
    formula = {
        "mu": [("intercept",), ("linear", "missing_col")],
        "sigma": [("intercept",)],
    }
    y = np.array([1.0, 2.0, 3.0])
    data = {"x": np.array([1.0, 2.0, 3.0])}  # note: no "missing_col" here
    with pytest.raises(RuntimeError):
        glissando.GamlssModel.fit(data, y, formula, glissando.Gaussian())


def test_unknown_term_type_rejected():
    formula = {
        "mu": [("nonsense", "x")],
        "sigma": [("intercept",)],
    }
    y = np.array([1.0, 2.0, 3.0])
    data = {"x": np.array([1.0, 2.0, 3.0])}
    with pytest.raises(ValueError):
        glissando.GamlssModel.fit(data, y, formula, glissando.Gaussian())


@pytest.mark.parametrize(
    "options",
    [
        {"bs": "re"},  # not a smooth basis this tuple form builds
        {"k": 20},  # `k` belongs to bs="cr"; a P-spline takes n_splines
        {"bs": "cr", "n_splines": 8},  # and the reverse
    ],
)
def test_unknown_smooth_options_rejected(synthetic_gaussian, options):
    formula = {
        "mu": [("intercept",), ("smooth", "x", options)],
        "sigma": [("intercept",)],
    }
    data = {"x": synthetic_gaussian["x"]}
    with pytest.raises(ValueError):
        glissando.GamlssModel.fit(data, synthetic_gaussian["y"], formula, glissando.Gaussian())


def test_known_smooth_options_accepted(synthetic_gaussian):
    formula = {
        "mu": [("intercept",), ("smooth", "x", {"bs": "cr", "k": 8})],
        "sigma": [("intercept",)],
    }
    data = {"x": synthetic_gaussian["x"]}
    model = glissando.GamlssModel.fit(data, synthetic_gaussian["y"], formula, glissando.Gaussian())
    assert model.converged()


def test_smooth_with_too_few_basis_functions_rejected(synthetic_gaussian):
    formula = {
        "mu": [("intercept",), ("smooth", "x", {"n_splines": 3})],
        "sigma": [("intercept",)],
    }
    data = {"x": synthetic_gaussian["x"]}
    with pytest.raises(RuntimeError):
        glissando.GamlssModel.fit(data, synthetic_gaussian["y"], formula, glissando.Gaussian())


@pytest.mark.parametrize("n_categories", [1, 6])
def test_ocat_out_of_range_rejected(n_categories):
    with pytest.raises(ValueError):
        glissando.Ocat(n_categories)
