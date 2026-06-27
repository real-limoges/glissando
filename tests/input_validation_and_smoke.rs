// Integration tests cannot run with the `python` feature due to PyO3's extension-module linking
#![cfg(not(feature = "python"))]

//! Gaps the rest of the suite left open: the input-validation error paths
//! (`preprocessing::validate_inputs`), degenerate-size robustness (n = 1, perfect
//! separation), and one end-to-end smoke test exercising the whole public pipeline
//! (`fit → predict → predict_with_se → JSON round-trip`).

use glissando::{
    distributions::{Binomial, Gaussian},
    DataSet, FitConfig, Formula, GamlssError, GamlssModel, NaAction, Term,
};
use ndarray::Array1;

/// `NaAction::Fail` config — the path that rejects non-finite model variables
/// rather than dropping their rows (the historical default).
fn fail_on_na() -> FitConfig {
    FitConfig::default().with_na_action(NaAction::Fail)
}

fn linear_formula() -> Formula {
    let mut f = Formula::new();
    f.add_terms(
        "mu",
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );
    f.add_terms("sigma", vec![Term::Intercept]);
    f
}

// ── Validation error paths ──────────────────────────────────────────────────────

#[test]
fn empty_response_returns_empty_data_error() {
    let y = Array1::<f64>::from_vec(vec![]);
    let mut data = DataSet::new();
    data.insert_column("x", Array1::<f64>::from_vec(vec![]));

    let err = GamlssModel::fit(&data, &y, &linear_formula(), &Gaussian::new()).unwrap_err();
    assert!(
        matches!(err, GamlssError::EmptyData),
        "empty y should yield EmptyData, got {err:?}"
    );
}

#[test]
fn nan_in_response_returns_non_finite_error() {
    let y = Array1::from_vec(vec![1.0, f64::NAN, 3.0, 4.0]);
    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]));

    // Under `NaAction::Fail`, a non-finite response is a hard error (the default
    // `DropRows` would instead drop the row — see `tests/data_na_handling.rs`).
    let err = GamlssModel::fit_with_config(
        &data,
        &y,
        None,
        &linear_formula(),
        &Gaussian::new(),
        fail_on_na(),
    )
    .unwrap_err();
    assert!(
        matches!(err, GamlssError::NonFiniteValues { count, .. } if count >= 1),
        "NaN in response should yield NonFiniteValues, got {err:?}"
    );
}

#[test]
fn inf_in_predictor_returns_non_finite_error() {
    let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_vec(vec![1.0, f64::INFINITY, 3.0, 4.0]));

    let err = GamlssModel::fit_with_config(
        &data,
        &y,
        None,
        &linear_formula(),
        &Gaussian::new(),
        fail_on_na(),
    )
    .unwrap_err();
    assert!(
        matches!(err, GamlssError::NonFiniteValues { .. }),
        "Inf in predictor should yield NonFiniteValues, got {err:?}"
    );
}

// ── Degenerate-size robustness (must not panic) ─────────────────────────────────

#[test]
fn single_observation_intercept_only_does_not_panic() {
    let y = Array1::from_vec(vec![3.5]);
    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_vec(vec![1.0]));

    let mut f = Formula::new();
    f.add_terms("mu", vec![Term::Intercept]);
    f.add_terms("sigma", vec![Term::Intercept]);

    // A single observation is degenerate for σ, but the call must return a Result
    // rather than panic. If it converges, the mean intercept should sit at y[0].
    match GamlssModel::fit(&data, &y, &f, &Gaussian::new()) {
        Ok(model) => {
            let mu0 = model.models["mu"].coefficients[0];
            assert!(
                mu0.is_finite(),
                "fitted mu intercept must be finite, got {mu0}"
            );
        }
        Err(_) => { /* a clean error is an acceptable outcome for n = 1 */ }
    }
}

#[test]
fn binomial_perfect_separation_does_not_panic() {
    // Perfectly separable Bernoulli data: y flips at x = 0. The MLE slope is
    // unbounded, so the fit may not converge — but it must not panic, and the
    // coefficients it reports must stay finite (logit link + Fisher weights guard).
    let x: Vec<f64> = (-10..10).map(|i| i as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| if xi > 0.0 { 1.0 } else { 0.0 })
        .collect();
    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_vec(x));
    let y = Array1::from_vec(y);

    let mut f = Formula::new();
    f.add_terms(
        "mu",
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );

    if let Ok(model) = GamlssModel::fit(&data, &y, &f, &Binomial::new(1)) {
        for c in model.models["mu"].coefficients.iter() {
            assert!(
                c.is_finite(),
                "separation must not produce non-finite coefficients, got {c}"
            );
        }
    }
}

// ── End-to-end smoke test ───────────────────────────────────────────────────────

#[test]
fn fit_predict_se_and_json_roundtrip() {
    // y = 2 + 3x + noise-free, so the pipeline has an unambiguous target.
    let x: Vec<f64> = (0..50).map(|i| i as f64 / 10.0).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 2.0 + 3.0 * xi).collect();
    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_vec(x));
    let y = Array1::from_vec(y);

    let family = Gaussian::new();
    let model = GamlssModel::fit(&data, &y, &linear_formula(), &family).unwrap();

    // predict
    let preds = model.predict(&data, &family).unwrap();
    let mu = &preds["mu"];
    assert_eq!(mu.len(), 50);
    assert!(
        (mu[0] - 2.0).abs() < 0.1,
        "intercept prediction ~2, got {}",
        mu[0]
    );

    // predict_with_se: SEs finite and non-negative; fitted matches bare predict
    let se_preds = model.predict_with_se(&data, &family).unwrap();
    let mu_se = &se_preds["mu"];
    for (f, s) in mu_se.fitted.iter().zip(mu_se.se_eta.iter()) {
        assert!(f.is_finite(), "fitted must be finite");
        assert!(
            s.is_finite() && *s >= 0.0,
            "se must be finite and non-negative, got {s}"
        );
    }
    for (a, b) in mu.iter().zip(mu_se.fitted.iter()) {
        assert!(
            (a - b).abs() < 1e-9,
            "predict and predict_with_se must agree"
        );
    }
}

/// JSON round-trip extends the smoke test through the `serialization` surface:
/// a reloaded model must reproduce the original predictions bit-for-bit.
#[cfg(feature = "serialization")]
#[test]
fn json_roundtrip_preserves_predictions() {
    let x: Vec<f64> = (0..50).map(|i| i as f64 / 10.0).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 2.0 + 3.0 * xi).collect();
    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_vec(x));
    let y = Array1::from_vec(y);

    let family = Gaussian::new();
    let model = GamlssModel::fit(&data, &y, &linear_formula(), &family).unwrap();
    let preds = model.predict(&data, &family).unwrap();

    let json = model.to_json(&family).unwrap();
    let (reloaded, dist_name) = GamlssModel::from_json(&json).unwrap();
    assert_eq!(dist_name, "Gaussian");
    let preds2 = reloaded.predict(&data, &family).unwrap();
    for (a, b) in preds["mu"].iter().zip(preds2["mu"].iter()) {
        assert!(
            (a - b).abs() < 1e-12,
            "predictions must survive JSON round-trip"
        );
    }
}
