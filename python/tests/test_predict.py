"""Prediction tests: shape and finiteness on held-out data."""

import numpy as np
import glissando


def test_predict_shape_matches_input(synthetic_gaussian, gaussian_formula):
    data = {"x": synthetic_gaussian["x"]}
    model = glissando.GamlssModel.fit(
        data,
        synthetic_gaussian["y"],
        gaussian_formula,
        glissando.Gaussian(),
    )

    new_data = {"x": np.array([0.5, 1.0, 1.5, 2.0])}
    preds = model.predict(new_data)

    assert "mu" in preds
    assert "sigma" in preds
    assert len(preds["mu"]) == 4
    assert len(preds["sigma"]) == 4


def test_predict_returns_finite_values(synthetic_gaussian, gaussian_formula):
    data = {"x": synthetic_gaussian["x"]}
    model = glissando.GamlssModel.fit(
        data,
        synthetic_gaussian["y"],
        gaussian_formula,
        glissando.Gaussian(),
    )
    preds = model.predict({"x": np.array([2.5, 3.0])})
    assert np.all(np.isfinite(preds["mu"]))
    assert np.all(np.isfinite(preds["sigma"]))
    assert np.all(preds["sigma"] > 0.0)


def test_predict_tracks_linear_trend(synthetic_gaussian, gaussian_formula):
    data = {"x": synthetic_gaussian["x"]}
    model = glissando.GamlssModel.fit(
        data,
        synthetic_gaussian["y"],
        gaussian_formula,
        glissando.Gaussian(),
    )
    preds = model.predict({"x": np.array([0.0, 5.0])})
    # Slope is positive, so mu(5) should clearly exceed mu(0).
    assert preds["mu"][1] > preds["mu"][0] + 3.0
