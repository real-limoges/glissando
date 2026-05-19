// Snapshot-based regression tests, doubling as a backend-equivalence check.
//
// The `.snap` files lock fitted-model output (rounded to 4 significant figures)
// for fixed-seed inputs.  Two distinct purposes share the same fixtures:
//
// 1. REGRESSION — running `cargo test --test regression` after a refactor must
//    reproduce the snapshot byte-for-byte; any drift in coefficients, EDF, or
//    information criteria larger than ~1e-4 trips the test.
// 2. BACKEND EQUIVALENCE — the same `.snap` files are honoured under both
//    `openblas` (default) and `pure-rust`.  CI runs:
//
//        cargo test --test regression
//        cargo test --test regression --no-default-features --features pure-rust,parallel
//
//    Both invocations must pass against the committed snapshot.  4-sig-fig
//    rounding gives the two backends slack to differ in trailing digits while
//    still catching real numerical drift.
//
// First-time creation: `INSTA_UPDATE=auto cargo test --test regression`, then
// `cargo insta accept` to commit the `.snap` files.
#![cfg(not(feature = "python"))]
#![cfg(not(target_arch = "wasm32"))]

mod common;

use common::{linear_intercepts, pspline, Generator};
use glissando::{
    distributions::{Distribution, Gaussian},
    FitConfig, Formula, GamlssModel, SmoothingCriterion, Term,
};

/// Snapshot tests are pinned to GCV so that both linalg backends (openblas and
/// pure-rust) land on the same minimum at 4-sig-fig precision. REML on a
/// near-flat objective surface (e.g. linear-trend data living in the order-2
/// penalty's null space) can converge to numerically distinct λ across backends;
/// that behavior is exercised by `tests/reml.rs` instead.
fn gcv_config() -> FitConfig {
    FitConfig { criterion: SmoothingCriterion::Gcv, ..FitConfig::default() }
}
use ndarray::Array1;
use serde::Serialize;
use std::collections::BTreeMap;

/// Rounded, deterministically-ordered representation of a fitted model.
/// Floats are formatted to 4 significant figures so the snapshot survives
/// trailing-digit differences between linear-algebra backends.
#[derive(Debug, Serialize)]
struct ModelSnapshot {
    converged: bool,
    iterations: usize,
    coefficients: BTreeMap<String, Vec<String>>,
    edf: BTreeMap<String, String>,
    lambdas: BTreeMap<String, Vec<String>>,
    log_likelihood: String,
    aic: String,
    bic: String,
}

fn fmt(x: f64) -> String {
    format!("{:.4e}", x)
}

fn fmt_vec(v: &Array1<f64>) -> Vec<String> {
    v.iter().map(|&x| fmt(x)).collect()
}

impl ModelSnapshot {
    fn from_fit<D: Distribution + ?Sized>(
        model: &GamlssModel,
        family: &D,
        y: &Array1<f64>,
    ) -> Self {
        let diag = model.diagnostics(family, y).unwrap();
        let coefficients: BTreeMap<String, Vec<String>> = model
            .models
            .iter()
            .map(|(k, v)| (k.clone(), fmt_vec(&v.coefficients.0)))
            .collect();
        let edf: BTreeMap<String, String> = model
            .models
            .iter()
            .map(|(k, v)| (k.clone(), fmt(v.edf)))
            .collect();
        let lambdas: BTreeMap<String, Vec<String>> = model
            .models
            .iter()
            .map(|(k, v)| (k.clone(), fmt_vec(&v.lambdas)))
            .collect();
        Self {
            converged: model.converged(),
            iterations: model.diagnostics.iterations,
            coefficients,
            edf,
            lambdas,
            log_likelihood: fmt(diag.log_likelihood),
            aic: fmt(diag.aic),
            bic: fmt(diag.bic),
        }
    }
}

#[test]
fn snapshot_gaussian_linear() {
    let mut rng = Generator::new(42);
    let (y, data) = rng.linear_gaussian(80, 1.0, 5.0, 1.0);
    let formula = linear_intercepts("x", &["mu", "sigma"]);
    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();
    let snap = ModelSnapshot::from_fit(&model, &Gaussian::new(), &y);
    insta::assert_yaml_snapshot!(snap);
}

#[test]
fn snapshot_heteroskedastic_gaussian() {
    let mut rng = Generator::new(42);
    let (y, data) = rng.heteroskedastic_gaussian(120);
    let formula = Formula::new()
        .with_terms(
            "mu",
            vec![
                Term::Intercept,
                Term::Linear {
                    col_name: "x".into(),
                },
            ],
        )
        .with_terms(
            "sigma",
            vec![
                Term::Intercept,
                Term::Linear {
                    col_name: "x".into(),
                },
            ],
        );
    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();
    let snap = ModelSnapshot::from_fit(&model, &Gaussian::new(), &y);
    insta::assert_yaml_snapshot!(snap);
}

#[test]
fn snapshot_gaussian_pspline() {
    let mut rng = Generator::new(42);
    let (y, data) = rng.linear_gaussian(120, 1.0, 5.0, 1.0);
    let formula = Formula::new()
        .with_terms("mu", vec![Term::Intercept, pspline("x", 6)])
        .with_terms("sigma", vec![Term::Intercept]);
    let model =
        GamlssModel::fit_with_config(&data, &y, &formula, &Gaussian::new(), gcv_config()).unwrap();
    let snap = ModelSnapshot::from_fit(&model, &Gaussian::new(), &y);
    insta::assert_yaml_snapshot!(snap);
}

/// Pins the fix for the rank-deficiency / partition-of-unity bug that
/// caused `Intercept + PSpline` to fail convergence under one backend (openblas
/// bouncing along the indeterminate β-direction) while landing in finite iterations
/// under another (pure-rust). The sum-to-zero reparameterization on the spline
/// basis removes the free direction, restoring deterministic convergence on both
/// backends.
#[test]
fn intercept_plus_pspline_converges_for_poisson() {
    let mut rng = Generator::new(42);
    let (y, data) = rng.poisson_data(150, 0.5, 0.3);
    let formula = Formula::new().with_terms("mu", vec![Term::Intercept, pspline("x", 8)]);
    let model = GamlssModel::fit(
        &data,
        &y,
        &formula,
        &glissando::distributions::Poisson::new(),
    )
    .unwrap();
    assert!(model.converged(), "should converge after sum-to-zero fix");
    assert!(model.diagnostics.iterations < 50);
}

/// Closed-form OLS reference: for a Gaussian linear model with identity link on μ
/// and intercept-only σ, the μ coefficients must agree with the analytic OLS
/// solution `(X'X)⁻¹·X'y`.  Catches subtle solver / link / weight issues that the
/// snapshot test would only flag indirectly.
#[test]
fn gaussian_linear_matches_ols_closed_form() {
    let mut rng = Generator::new(42);
    let (y, data) = rng.linear_gaussian(100, 1.0, 5.0, 1.0);
    let x_col = data.column("x").unwrap();

    let n = y.len() as f64;
    let sum_x: f64 = x_col.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x_col.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let sum_xx: f64 = x_col.iter().map(|x| x * x).sum();

    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n;

    let formula = linear_intercepts("x", &["mu", "sigma"]);
    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();
    let beta_mu = &model.models["mu"].coefficients.0;

    assert!(
        (beta_mu[0] - intercept).abs() < 1e-3,
        "intercept: glissando={} ols={}",
        beta_mu[0],
        intercept
    );
    assert!(
        (beta_mu[1] - slope).abs() < 1e-3,
        "slope: glissando={} ols={}",
        beta_mu[1],
        slope
    );
}
