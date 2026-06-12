// Prior-weight end-to-end tests.
//
// Validates `GamlssModel::fit_weighted` / `fit_with_config(..., Some(weights), ...)`
// semantics:
//
//   1. Ones-identity: fitting with all-ones weights reproduces the unweighted fit.
//   2. Zero-weight exclusion: rows with weight 0 are effectively dropped.
//   3. Validation errors: wrong length, negative, NaN.
//
// The mgcv parity benchmarks (B1/B2) live in the benchmark harness and are
// run via `benchmark/run_comparison.sh` + `cargo test --test mgcv_reference`.
#![cfg(not(feature = "python"))]
#![cfg(not(target_arch = "wasm32"))]

mod common;

use common::Generator;
use glissando::{
    distributions::{Gaussian, StudentT},
    DataSet, Formula, GamlssModel, Term,
};
use ndarray::Array1;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn gaussian_intercept_formula() -> Formula {
    Formula::new()
        .with_terms("mu", vec![Term::Intercept])
        .with_terms("sigma", vec![Term::Intercept])
}

fn gaussian_linear_formula() -> Formula {
    Formula::new()
        .with_terms(
            "mu",
            vec![
                Term::Intercept,
                Term::Linear {
                    col_name: "x".to_string(),
                },
            ],
        )
        .with_terms("sigma", vec![Term::Intercept])
}

fn studentt_intercept_formula() -> Formula {
    Formula::new()
        .with_terms("mu", vec![Term::Intercept])
        .with_terms("sigma", vec![Term::Intercept])
        .with_terms("nu", vec![Term::Intercept])
}

// ---------------------------------------------------------------------------
// 1. Ones-identity
// ---------------------------------------------------------------------------

/// Fitting with all-ones weights must reproduce the unweighted fit exactly.
/// This is the primary correctness regression guard: if the weights fold in
/// correctly (only into `w_solve`, not into `z`), ones are a no-op.
#[test]
fn prior_weight_ones_identity_gaussian_intercept() {
    let y = Array1::from_vec(vec![1.0, 3.0, 2.0, 4.0, 5.0]);
    let n = y.len();
    let data = DataSet::new();
    let formula = gaussian_intercept_formula();

    let unweighted = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();
    let ones = Array1::ones(n);
    let weighted = GamlssModel::fit_weighted(&data, &y, &ones, &formula, &Gaussian::new()).unwrap();

    let uw_mu = &unweighted.models["mu"].coefficients.0;
    let wt_mu = &weighted.models["mu"].coefficients.0;
    for (u, w) in uw_mu.iter().zip(wt_mu.iter()) {
        assert!(
            (u - w).abs() < 1e-8,
            "mu coefficient mismatch: unweighted={u} weighted={w}"
        );
    }

    let uw_sigma = &unweighted.models["sigma"].coefficients.0;
    let wt_sigma = &weighted.models["sigma"].coefficients.0;
    for (u, w) in uw_sigma.iter().zip(wt_sigma.iter()) {
        assert!(
            (u - w).abs() < 1e-8,
            "sigma coefficient mismatch: unweighted={u} weighted={w}"
        );
    }
}

#[test]
fn prior_weight_ones_identity_gaussian_linear() {
    let mut rng = Generator::new(42);
    let (y, data) = rng.linear_gaussian(60, 1.0, 3.0, 0.5);
    let n = y.len();
    let formula = gaussian_linear_formula();

    let unweighted = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();
    let ones = Array1::ones(n);
    let weighted = GamlssModel::fit_weighted(&data, &y, &ones, &formula, &Gaussian::new()).unwrap();

    let uw_mu = &unweighted.models["mu"].coefficients.0;
    let wt_mu = &weighted.models["mu"].coefficients.0;
    assert_eq!(uw_mu.len(), wt_mu.len());
    for (u, w) in uw_mu.iter().zip(wt_mu.iter()) {
        assert!(
            (u - w).abs() < 1e-7,
            "mu coefficient mismatch: unweighted={u} weighted={w}"
        );
    }
    // EDF should be identical since the weight fold is a no-op.
    let uw_edf = unweighted.models["mu"].edf;
    let wt_edf = weighted.models["mu"].edf;
    assert!(
        (uw_edf - wt_edf).abs() < 1e-6,
        "EDF mismatch: unweighted={uw_edf} weighted={wt_edf}"
    );
}

#[test]
fn prior_weight_ones_identity_student_t() {
    let y = Array1::from_vec(vec![0.5, 1.2, 2.3, 3.0, 4.1, 2.8]);
    let n = y.len();
    let data = DataSet::new();
    let formula = studentt_intercept_formula();

    let unweighted = GamlssModel::fit(&data, &y, &formula, &StudentT::new()).unwrap();
    let ones = Array1::ones(n);
    let weighted = GamlssModel::fit_weighted(&data, &y, &ones, &formula, &StudentT::new()).unwrap();

    for param in ["mu", "sigma", "nu"] {
        let uw_coef = &unweighted.models[param].coefficients.0;
        let wt_coef = &weighted.models[param].coefficients.0;
        for (u, w) in uw_coef.iter().zip(wt_coef.iter()) {
            assert!(
                (u - w).abs() < 1e-7,
                "{param} coefficient mismatch: unweighted={u} weighted={w}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Zero-weight exclusion
// ---------------------------------------------------------------------------

/// Rows with weight 0 contribute nothing — fitting on n rows with the last k
/// zeroed should match fitting on just the first n-k rows.
#[test]
fn prior_weight_zero_excludes_rows() {
    // Use a dataset with a clear split: first 5 rows have one mean, last 5 another.
    // With zero weights on the last 5 the fit should track the first 5 only.
    let y_full = Array1::from_vec(vec![
        2.0, 2.1, 1.9, 2.0, 2.05, // mean ≈ 2
        8.0, 8.1, 7.9, 8.0, 8.05, // mean ≈ 8 (should be invisible)
    ]);
    let y_short = Array1::from_vec(vec![2.0, 2.1, 1.9, 2.0, 2.05]);
    let data = DataSet::new();
    let formula = gaussian_intercept_formula();

    // Weights: 1 for first 5, 0 for last 5.
    let mut weights = Array1::ones(10_usize);
    for i in 5..10 {
        weights[i] = 0.0;
    }

    let weighted =
        GamlssModel::fit_weighted(&data, &y_full, &weights, &formula, &Gaussian::new()).unwrap();
    let short = GamlssModel::fit(&data, &y_short, &formula, &Gaussian::new()).unwrap();

    let wt_mu = weighted.models["mu"].coefficients.0[0];
    let sh_mu = short.models["mu"].coefficients.0[0];
    assert!(
        (wt_mu - sh_mu).abs() < 0.1,
        "weighted(zero-rows) mu={wt_mu} should match short mu={sh_mu}"
    );
    // Fitted mu must be near 2.0, well away from 8.0.
    assert!(
        (wt_mu - 2.0).abs() < 0.5,
        "weighted mu={wt_mu} should be near 2.0 (not influenced by zeroed rows)"
    );
}

// ---------------------------------------------------------------------------
// 3. Validation errors
// ---------------------------------------------------------------------------

#[test]
fn prior_weight_validation_wrong_length() {
    let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let weights = Array1::from_vec(vec![1.0, 1.0]); // length 2, y length 3
    let data = DataSet::new();
    let formula = gaussian_intercept_formula();

    let err =
        GamlssModel::fit_weighted(&data, &y, &weights, &formula, &Gaussian::new()).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("weights") || msg.contains("length"),
        "expected error about weights length, got: {msg}"
    );
}

#[test]
fn prior_weight_validation_negative() {
    let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let weights = Array1::from_vec(vec![1.0, -0.1, 1.0]);
    let data = DataSet::new();
    let formula = gaussian_intercept_formula();

    let err =
        GamlssModel::fit_weighted(&data, &y, &weights, &formula, &Gaussian::new()).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("weights") || msg.contains("non-negative"),
        "expected error about non-negative weights, got: {msg}"
    );
}

#[test]
fn prior_weight_validation_nan() {
    let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let weights = Array1::from_vec(vec![1.0, f64::NAN, 1.0]);
    let data = DataSet::new();
    let formula = gaussian_intercept_formula();

    let err =
        GamlssModel::fit_weighted(&data, &y, &weights, &formula, &Gaussian::new()).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("weights") || msg.contains("finite"),
        "expected error about finite weights, got: {msg}"
    );
}

#[test]
fn prior_weight_validation_infinity() {
    let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let weights = Array1::from_vec(vec![1.0, f64::INFINITY, 1.0]);
    let data = DataSet::new();
    let formula = gaussian_intercept_formula();

    let err =
        GamlssModel::fit_weighted(&data, &y, &weights, &formula, &Gaussian::new()).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("weights") || msg.contains("finite"),
        "expected error about finite weights, got: {msg}"
    );
}
