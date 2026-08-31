"""Model-selection bindings: gaic, lr_test, ic_table, step_gaic."""

import numpy as np
import pytest
import glissando


def _fit(data, y, formula):
    return glissando.GamlssModel.fit(data, y, formula, glissando.Gaussian())


def test_gaic_matches_aic_and_is_monotone(synthetic_gaussian, gaussian_formula):
    data = {"x": synthetic_gaussian["x"]}
    y = synthetic_gaussian["y"]
    model = _fit(data, y, gaussian_formula)

    n = len(y)
    aic = model.gaic(y, 2.0)
    bic = model.gaic(y, np.log(n))
    assert np.isfinite(aic) and np.isfinite(bic)
    # bigger penalty (BIC) ⇒ bigger GAIC for the same fit.
    assert bic > aic


def test_lr_test_detects_genuine_term(synthetic_gaussian, gaussian_formula):
    data = {"x": synthetic_gaussian["x"]}
    y = synthetic_gaussian["y"]
    null = {"mu": [("intercept",)], "sigma": [("intercept",)]}
    small = _fit(data, y, null)
    big = _fit(data, y, gaussian_formula)

    result = small.lr_test(big, y)
    assert result["df"] > 0.0
    assert result["lr_stat"] > 0.0
    # y is strongly linear in x, so this should come out significant.
    assert result["p_value"] < 0.01

    # and a mis-ordered pair should raise.
    with pytest.raises(ValueError):
        big.lr_test(small, y)


def test_ic_table_ranks_models(synthetic_gaussian, gaussian_formula):
    data = {"x": synthetic_gaussian["x"]}
    y = synthetic_gaussian["y"]
    null = {"mu": [("intercept",)], "sigma": [("intercept",)]}
    m_null = _fit(data, y, null)
    m_x = _fit(data, y, gaussian_formula)

    rows = glissando.GamlssModel.ic_table([("null", m_null), ("with_x", m_x)], y, 2.0)
    assert [r["label"] for r in rows] == ["null", "with_x"]
    # the model with x fits the linear signal better, so lower deviance and GAIC.
    assert rows[1]["global_deviance"] < rows[0]["global_deviance"]
    assert rows[1]["gaic"] < rows[0]["gaic"]


def test_step_gaic_forward_selects_signal(synthetic_gaussian):
    data = {"x": synthetic_gaussian["x"]}
    y = synthetic_gaussian["y"]
    start = {"mu": [("intercept",)], "sigma": [("intercept",)]}
    scope = {"mu": [("linear", "x")]}

    out = glissando.GamlssModel.step_gaic(
        data,
        y,
        glissando.Gaussian(),
        start,
        scope,
        np.log(len(y)),  # BIC penalty
        "forward",
    )
    model = out["model"]
    trace = out["trace"]
    assert model.converged()
    # the genuine linear term should have been added.
    assert len(trace) >= 1
    assert any("x" in step["move"] for step in trace)
    # and the selected model should predict a mean that rises in x.
    preds = model.predict({"x": np.array([0.0, 5.0])})
    assert preds["mu"][1] > preds["mu"][0]


# --- INFER-1 / INFER-2 bindings: quantile_residuals, centiles, quantile_prediction ---


def test_quantile_residuals_gaussian_calibrated(rng):
    n = 500
    x = np.linspace(0.0, 5.0, n)
    y = 2.0 + 1.5 * x + rng.normal(0.0, 0.5, size=n)
    data = {"x": x}
    formula = {"mu": [("intercept",), ("linear", "x")], "sigma": [("intercept",)]}
    model = glissando.GamlssModel.fit(data, y, formula, glissando.Gaussian())

    resid = model.quantile_residuals(y)
    assert resid.shape == (n,)
    assert np.all(np.isfinite(resid))
    assert abs(resid.mean()) < 0.15
    assert abs(resid.std(ddof=1) - 1.0) < 0.2


def test_quantile_residuals_discrete_seed_reproducible(synthetic_poisson, poisson_formula):
    data = {"x": synthetic_poisson["x"]}
    y = synthetic_poisson["y"]
    model = glissando.GamlssModel.fit(data, y, poisson_formula, glissando.Poisson())

    a = model.quantile_residuals(y, seed=7)
    b = model.quantile_residuals(y, seed=7)
    c = model.quantile_residuals(y, seed=8)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


def test_centiles_ordered_and_median_is_mu(synthetic_gaussian, gaussian_formula):
    data = {"x": synthetic_gaussian["x"]}
    y = synthetic_gaussian["y"]
    model = glissando.GamlssModel.fit(data, y, gaussian_formula, glissando.Gaussian())

    curves = model.centiles(data, [10.0, 50.0, 90.0])
    assert set(curves.keys()) == {"C10", "C50", "C90"}
    # Monotone in level, and the 50th centile equals fitted mu.
    assert np.all(curves["C10"] < curves["C50"])
    assert np.all(curves["C50"] < curves["C90"])
    mu = model.predict(data)["mu"]
    assert np.allclose(curves["C50"], mu, atol=1e-6)


def test_quantile_prediction_constant_level(synthetic_gaussian, gaussian_formula):
    data = {"x": synthetic_gaussian["x"]}
    y = synthetic_gaussian["y"]
    model = glissando.GamlssModel.fit(data, y, gaussian_formula, glissando.Gaussian())

    p = np.full(len(y), 0.5)
    q = model.quantile_prediction(data, p)
    mu = model.predict(data)["mu"]
    assert np.allclose(q, mu, atol=1e-6)
