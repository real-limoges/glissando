// Integration tests cannot run with the `python` feature due to PyO3's extension-module linking
#![cfg(not(feature = "python"))]

//! Parameter-recovery coverage for a P-spline smooth on a *scale* parameter.
//!
//! `tests/parameter_recovery.rs` and `tests/comprehensive.rs` cover smooths on
//! `mu` and linear terms on `sigma`, but nothing fits a *smooth* on `sigma`.
//! This file pins that case: a smooth on `sigma` must track a known nonlinear
//! `log σ(x)`, not collapse onto the penalty null space (a straight line in
//! `log σ`). A collapsed σ-smooth shows up two ways here — low correlation with
//! the truth, and an effective degrees of freedom pinned near its null-space
//! dimension — and either makes the test fail.

use glissando::{distributions::Gaussian, DataSet, Formula, GamlssModel, Smooth, Term};
use ndarray::Array1;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};

/// Pearson correlation between two equal-length vectors.
fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let da = x - mean_a;
        let db = y - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    cov / (var_a.sqrt() * var_b.sqrt())
}

/// True scale curve: `log σ(x) = -0.7 + 0.8·sin(2πx)` for `x ∈ [0, 1]`.
fn true_log_sigma(x: f64) -> f64 {
    -0.7 + 0.8 * (2.0 * std::f64::consts::PI * x).sin()
}

#[test]
fn sigma_smooth_recovers_nonlinear_scale() {
    let n = 4_000;
    let mut rng = StdRng::seed_from_u64(7);

    // Constant mean; all of the structure lives in the scale parameter.
    let true_mu = 5.0;

    let x_vals: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
    let y_vals: Vec<f64> = x_vals
        .iter()
        .map(|&x| {
            let sigma = true_log_sigma(x).exp();
            Normal::new(true_mu, sigma).unwrap().sample(&mut rng)
        })
        .collect();

    let y = Array1::from_vec(y_vals);
    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_vec(x_vals.clone()));

    let mut formula = Formula::new();
    formula.add_terms("mu".to_string(), vec![Term::Intercept]);
    formula.add_terms(
        "sigma".to_string(),
        vec![
            Term::Intercept,
            Term::Smooth(Smooth::PSpline1D {
                col_name: "x".to_string(),
                n_splines: 15,
                degree: 3,
                penalty_order: 2,
            }),
        ],
    );

    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).expect("Fit failed");

    let sigma_param = &model.models["sigma"];
    let edf_sigma = sigma_param.edf;

    // Predicted σ on the response scale; compare log σ̂ against the truth.
    let preds = model
        .predict(&data, &Gaussian::new())
        .expect("predict failed");
    let fitted_log_sigma: Vec<f64> = preds["sigma"].iter().map(|s| s.ln()).collect();
    let truth_log_sigma: Vec<f64> = x_vals.iter().map(|&x| true_log_sigma(x)).collect();

    let corr = correlation(&fitted_log_sigma, &truth_log_sigma);

    println!("σ-smooth edf = {edf_sigma:.3}, corr(log σ̂, log σ) = {corr:.4}");
    println!("{model}");

    // A smooth on σ that has truly fit the curve correlates strongly with the
    // truth. A null-space collapse (straight line in log σ) cannot follow a full
    // sine period and correlates weakly.
    assert!(
        corr > 0.8,
        "σ-smooth failed to track log σ(x): corr = {corr:.4} (expected > 0.8). \
         A low value means the smooth collapsed toward its linear null space."
    );

    // The smooth block's penalty null space (after sum-to-zero centering of an
    // order-2 P-spline) is the linear direction, so a collapsed σ model has
    // edf ≈ 2 (intercept + linear remainder). Recovering a sine needs visibly
    // more curvature than that.
    assert!(
        edf_sigma > 3.0,
        "σ-smooth edf = {edf_sigma:.3} sits near its null-space dimension; \
         the smoothing parameter has over-penalized the scale curve."
    );
}

#[test]
fn per_term_edf_sums_to_total_and_linear_truth_warns() {
    let n = 3_000;
    let mut rng = StdRng::seed_from_u64(3);

    // A strictly linear mean. A 2nd-order P-spline's penalty null space is the
    // set of straight lines, so the optimal smooth here *is* its null space —
    // REML drives λ up and the smooth collapses to a line, tripping the warning.
    let x_vals: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
    let y_vals: Vec<f64> = x_vals
        .iter()
        .map(|&x| Normal::new(2.0 + 3.0 * x, 0.3).unwrap().sample(&mut rng))
        .collect();

    let y = Array1::from_vec(y_vals);
    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_vec(x_vals));

    let mut formula = Formula::new();
    formula.add_terms(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Smooth(Smooth::PSpline1D {
                col_name: "x".to_string(),
                n_splines: 15,
                degree: 3,
                penalty_order: 2,
            }),
        ],
    );
    formula.add_terms("sigma".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).expect("Fit failed");

    // Per-term EDF must decompose the parameter EDF.
    for (name, fp) in &model.models {
        let sum: f64 = fp.term_edf.iter().sum();
        assert!(
            (sum - fp.edf).abs() < 1e-6,
            "parameter '{name}': per-term EDF {sum} != total EDF {}",
            fp.edf
        );
        assert_eq!(fp.term_edf.len(), fp.terms.len());
    }

    println!(
        "mu term_edf = {:?}, warnings = {:?}",
        model.models["mu"].term_edf, model.diagnostics.warnings
    );
    assert!(
        model.diagnostics.warnings.iter().any(|w| w.contains("mu")),
        "expected a collapse warning for the linear-truth μ-smooth, got {:?}",
        model.diagnostics.warnings
    );
}

#[test]
fn recovered_curve_does_not_warn() {
    // The strong-signal recovery case from `sigma_smooth_recovers_nonlinear_scale`
    // must NOT raise a spurious collapse warning.
    let n = 4_000;
    let mut rng = StdRng::seed_from_u64(7);
    let x_vals: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
    let y_vals: Vec<f64> = x_vals
        .iter()
        .map(|&x| {
            Normal::new(5.0, true_log_sigma(x).exp())
                .unwrap()
                .sample(&mut rng)
        })
        .collect();
    let y = Array1::from_vec(y_vals);
    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_vec(x_vals));

    let mut formula = Formula::new();
    formula.add_terms("mu".to_string(), vec![Term::Intercept]);
    formula.add_terms(
        "sigma".to_string(),
        vec![
            Term::Intercept,
            Term::Smooth(Smooth::PSpline1D {
                col_name: "x".to_string(),
                n_splines: 15,
                degree: 3,
                penalty_order: 2,
            }),
        ],
    );
    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).expect("Fit failed");
    assert!(
        model.diagnostics.warnings.is_empty(),
        "recovered curve should not warn, got {:?}",
        model.diagnostics.warnings
    );
}

/// Control: the *same* nonlinear curve placed on `mu` recovers fine today. This
/// isolates the failure to the scale-parameter path rather than the smooth
/// machinery in general.
#[test]
fn mu_smooth_recovers_nonlinear_mean_control() {
    let n = 4_000;
    let mut rng = StdRng::seed_from_u64(7);

    let x_vals: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
    let y_vals: Vec<f64> = x_vals
        .iter()
        .map(|&x| {
            // log σ curve reused as a mean curve, constant noise.
            let mu = true_log_sigma(x);
            Normal::new(mu, 0.2).unwrap().sample(&mut rng)
        })
        .collect();

    let y = Array1::from_vec(y_vals);
    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_vec(x_vals.clone()));

    let mut formula = Formula::new();
    formula.add_terms(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Smooth(Smooth::PSpline1D {
                col_name: "x".to_string(),
                n_splines: 15,
                degree: 3,
                penalty_order: 2,
            }),
        ],
    );
    formula.add_terms("sigma".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).expect("Fit failed");

    let preds = model
        .predict(&data, &Gaussian::new())
        .expect("predict failed");
    let fitted_mu: Vec<f64> = preds["mu"].to_vec();
    let truth_mu: Vec<f64> = x_vals.iter().map(|&x| true_log_sigma(x)).collect();
    let corr = correlation(&fitted_mu, &truth_mu);

    assert!(
        corr > 0.95,
        "μ-smooth control failed to track the mean curve: corr = {corr:.4}"
    );
}
