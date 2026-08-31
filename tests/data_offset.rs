//! DATA-3 offsets, tested through the public API.
//!
//! An offset enters the linear predictor as `η = X·β + offset` with a fixed
//! coefficient of 1. The load-bearing correctness check is a closed-form
//! equivalence I like: for a Gaussian identity-link model `μ = X·β + o`, fitting
//! `y` with offset `o` is exactly fitting `(y − o)` with no offset.

// Can't run under the `python` feature (PyO3 linking).
#![cfg(not(feature = "python"))]

use glissando::distributions::{Gaussian, Poisson};
use glissando::{DataSet, Formula, GamlssModel, Term};
use ndarray::Array1;

fn gaussian_two_param(with_offset: bool) -> Formula {
    let mut mu = vec![Term::Intercept, Term::linear("x")];
    if with_offset {
        mu.push(Term::offset("o"));
    }
    Formula::new()
        .with_terms("mu", mu)
        .with_terms("sigma", vec![Term::Intercept])
}

/// Gaussian identity link: fitting with an offset equals fitting on the
/// offset-subtracted response, coefficient for coefficient and fitted value for
/// fitted value.
#[test]
fn gaussian_offset_equals_folding_into_response() {
    let n = 200;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * 3.0).collect();
    // A deterministic per-row offset and a nearly noiseless response so the two
    // fits line up to tight tolerance.
    let o: Vec<f64> = x.iter().map(|&xi| 0.5 + 0.3 * (xi * 1.7).sin()).collect();
    let y: Vec<f64> = x
        .iter()
        .zip(&o)
        .enumerate()
        .map(|(i, (&xi, &oi))| {
            // true η = 1.0 + 0.8·x + offset, plus a tiny deterministic wiggle
            1.0 + 0.8 * xi + oi + 0.05 * ((i % 7) as f64 - 3.0)
        })
        .collect();

    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_vec(x.clone()));
    data.insert_column("o", Array1::from_vec(o.clone()));

    // Model A: y with an explicit offset term.
    let model_a = GamlssModel::fit(
        &data,
        &Array1::from_vec(y.clone()),
        &gaussian_two_param(true),
        &Gaussian,
    )
    .unwrap();

    // Model B: (y − o) with no offset.
    let y_folded: Vec<f64> = y.iter().zip(&o).map(|(&yi, &oi)| yi - oi).collect();
    let model_b = GamlssModel::fit(
        &data,
        &Array1::from_vec(y_folded),
        &gaussian_two_param(false),
        &Gaussian,
    )
    .unwrap();

    // μ coefficients (intercept, slope) coincide.
    let beta_a = &model_a.models["mu"].coefficients.0;
    let beta_b = &model_b.models["mu"].coefficients.0;
    for (a, b) in beta_a.iter().zip(beta_b.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "μ coefficient mismatch: {a} (offset fit) vs {b} (folded fit)"
        );
    }

    // Fitted μ from A equals fitted-from-B plus the offset, row by row.
    let pred_a = model_a.predict(&data, &Gaussian).unwrap();
    let pred_b = model_b.predict(&data, &Gaussian).unwrap();
    for ((a, b), oi) in pred_a["mu"].iter().zip(pred_b["mu"].iter()).zip(&o) {
        assert!(
            (a - (b + oi)).abs() < 1e-6,
            "fitted μ mismatch: {a} vs {b} + {oi}"
        );
    }
}

/// An offset never gets silently dropped: pull it out and the fit moves for real.
#[test]
fn offset_changes_the_fit() {
    let n = 150;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * 2.0).collect();
    let o: Vec<f64> = x.iter().map(|&xi| 1.0 + xi).collect();
    let y: Vec<f64> = x
        .iter()
        .zip(&o)
        .map(|(&xi, &oi)| 0.5 + 0.4 * xi + oi)
        .collect();

    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_vec(x));
    data.insert_column("o", Array1::from_vec(o));
    let y = Array1::from_vec(y);

    let with = GamlssModel::fit(&data, &y, &gaussian_two_param(true), &Gaussian).unwrap();
    let without = GamlssModel::fit(&data, &y, &gaussian_two_param(false), &Gaussian).unwrap();

    let int_with = with.models["mu"].coefficients.0[0];
    let int_without = without.models["mu"].coefficients.0[0];
    assert!(
        (int_with - int_without).abs() > 0.5,
        "offset should shift the intercept materially: {int_with} vs {int_without}"
    );
}

/// Poisson rate model: `log(exposure)` as an offset is the canonical use. With
/// the offset present the intercept recovers the underlying log-rate. Double
/// every exposure (offset += log 2) and the fitted intercept drops by ~log 2,
/// while the fitted counts' rate interpretation stays intact.
#[test]
fn poisson_offset_recovers_rate_intercept() {
    let n = 400;
    let log_rate: f64 = -1.0; // true intercept on the rate
    let exposure: Vec<f64> = (0..n).map(|i| 1.0 + (i % 5) as f64).collect();
    // Expected count μ = exposure · exp(log_rate). Generate near-expectation
    // counts deterministically (round) so the test stays noise-free and stable.
    let y: Vec<f64> = exposure
        .iter()
        .map(|&e| (e * log_rate.exp()).round().max(0.0))
        .collect();
    let log_e: Vec<f64> = exposure.iter().map(|&e| e.ln()).collect();

    let mut data = DataSet::new();
    data.insert_column("log_e", Array1::from_vec(log_e));
    let y = Array1::from_vec(y);

    let formula = Formula::new().with_terms("mu", vec![Term::Intercept, Term::offset("log_e")]);
    let model = GamlssModel::fit(&data, &y, &formula, &Poisson).unwrap();
    let intercept = model.models["mu"].coefficients.0[0];
    assert!(
        (intercept - log_rate).abs() < 0.2,
        "Poisson offset model should recover log-rate intercept ≈ {log_rate}, got {intercept}"
    );
}
