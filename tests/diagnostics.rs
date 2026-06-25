// Integration tests cannot run with the `python` feature due to PyO3's extension-module linking
#![cfg(not(feature = "python"))]

mod common;

use common::{intercept_only, linear_intercepts, Generator};
use glissando::{
    diagnostics::{compute_aic, compute_bic, pearson_residuals, response_residuals, total_edf},
    distributions::{Distribution, Gaussian, Poisson},
    GamlssModel,
};
use ndarray::Array1;
use std::collections::HashMap;

#[test]
fn test_pearson_residuals_gaussian() {
    let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let mu = Array1::from_vec(vec![1.5, 2.0, 2.5, 4.5, 5.0]);
    let sigma = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5, 0.5]);
    let params: HashMap<&str, &Array1<f64>> = HashMap::from([("mu", &mu), ("sigma", &sigma)]);

    let residuals = pearson_residuals(&Gaussian, &y, &params).unwrap();

    // r_i = (y_i - mu_i) / sigma_i
    assert!((residuals[0] - (-1.0)).abs() < 1e-10);
    assert!((residuals[1] - 0.0).abs() < 1e-10);
    assert!((residuals[2] - 1.0).abs() < 1e-10);
    assert!((residuals[3] - (-1.0)).abs() < 1e-10);
    assert!((residuals[4] - 0.0).abs() < 1e-10);
}

#[test]
fn test_pearson_residuals_poisson() {
    let y = Array1::from_vec(vec![0.0, 1.0, 4.0, 9.0, 16.0]);
    let mu = Array1::from_vec(vec![1.0, 1.0, 4.0, 9.0, 16.0]);
    let params: HashMap<&str, &Array1<f64>> = HashMap::from([("mu", &mu)]);

    let residuals = pearson_residuals(&Poisson, &y, &params).unwrap();

    // r_i = (y_i - mu_i) / sqrt(mu_i)
    assert!((residuals[0] - (-1.0)).abs() < 1e-10);
    assert!((residuals[1] - 0.0).abs() < 1e-10);
    assert!((residuals[2] - 0.0).abs() < 1e-10);
    assert!((residuals[3] - 0.0).abs() < 1e-10);
    assert!((residuals[4] - 0.0).abs() < 1e-10);
}

#[test]
fn test_response_residuals() {
    let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let e = Array1::from_vec(vec![1.5, 2.0, 2.5]);

    let residuals = response_residuals(&y, &e);

    assert!((residuals[0] - (-0.5)).abs() < 1e-10);
    assert!((residuals[1] - 0.0).abs() < 1e-10);
    assert!((residuals[2] - 0.5).abs() < 1e-10);
}

#[test]
fn test_loglik_gaussian() {
    // Standard normal at y=mu=0, sigma=1: l = -0.5 log(2π) per obs.
    let y = Array1::from_vec(vec![0.0, 0.0, 0.0]);
    let mu = Array1::from_vec(vec![0.0, 0.0, 0.0]);
    let sigma = Array1::from_vec(vec![1.0, 1.0, 1.0]);
    let params: HashMap<&str, &Array1<f64>> = HashMap::from([("mu", &mu), ("sigma", &sigma)]);

    let ll = Gaussian.loglik(&y, &params).unwrap();
    let expected = 3.0 * (-0.5 * (2.0 * std::f64::consts::PI).ln());
    assert!((ll - expected).abs() < 1e-6);
}

#[test]
fn test_loglik_poisson() {
    let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let mu = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let params: HashMap<&str, &Array1<f64>> = HashMap::from([("mu", &mu)]);

    let ll = Poisson.loglik(&y, &params).unwrap();
    assert!(ll.is_finite());
    assert!(ll < 0.0);
}

#[test]
fn test_aic_bic() {
    let ll = -100.0;
    let edf = 5.0;
    let n = 100;

    let aic = compute_aic(ll, edf);
    let bic = compute_bic(ll, edf, n);

    assert!((aic - 210.0).abs() < 1e-10);
    let expected_bic = 200.0 + (100.0_f64).ln() * 5.0;
    assert!((bic - expected_bic).abs() < 1e-6);
}

#[test]
fn test_total_edf() {
    let mut rng = Generator::new(42);
    let (y, data) = rng.linear_gaussian(100, 1.0, 5.0, 1.0);

    let formula = linear_intercepts("x", &["mu", "sigma"]);

    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

    let edf = total_edf(&model.models);

    // mu has 2 coefficients (intercept + slope), sigma has 1 (intercept). Total ~3.
    assert!(
        edf > 2.5 && edf < 3.5,
        "Total EDF should be ~3, got {}",
        edf
    );
}

#[test]
fn gaic_matches_aic_and_bic_at_canonical_k() {
    let mut rng = Generator::new(7);
    let (y, data) = rng.linear_gaussian(150, 1.5, 4.0, 1.0);
    let formula = linear_intercepts("x", &["mu", "sigma"]);
    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

    let diag = model.diagnostics(&Gaussian::new(), &y).unwrap();
    let aic = model.gaic(&Gaussian::new(), &y, 2.0).unwrap();
    let bic = model
        .gaic(&Gaussian::new(), &y, (y.len() as f64).ln())
        .unwrap();

    // k = 2 ≡ AIC, k = log(n) ≡ BIC.
    assert!(
        (aic - diag.aic).abs() < 1e-9,
        "gaic(2) {} vs aic {}",
        aic,
        diag.aic
    );
    assert!(
        (bic - diag.bic).abs() < 1e-9,
        "gaic(log n) {} vs bic {}",
        bic,
        diag.bic
    );
}

#[test]
fn gaic_works_for_a_discrete_family() {
    // GAIC is family-agnostic: it must produce a finite score for a Poisson fit,
    // and stay consistent with AIC/BIC at the canonical penalties.
    let mut rng = Generator::new(17);
    let (y, data) = rng.poisson_data(200, 0.5, 0.3);
    let formula = intercept_only(&["mu"]);
    let model = GamlssModel::fit(&data, &y, &formula, &Poisson::new()).unwrap();

    let diag = model.diagnostics(&Poisson::new(), &y).unwrap();
    let aic = model.gaic(&Poisson::new(), &y, 2.0).unwrap();
    let bic = model
        .gaic(&Poisson::new(), &y, (y.len() as f64).ln())
        .unwrap();
    assert!(aic.is_finite() && bic.is_finite());
    assert!((aic - diag.aic).abs() < 1e-9);
    assert!((bic - diag.bic).abs() < 1e-9);
}

#[test]
fn gaic_is_monotone_in_k() {
    let mut rng = Generator::new(11);
    let (y, data) = rng.linear_gaussian(120, 1.0, 3.0, 1.0);
    let formula = linear_intercepts("x", &["mu", "sigma"]);
    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

    // edf > 0, so a larger penalty strictly raises GAIC.
    let g2 = model.gaic(&Gaussian::new(), &y, 2.0).unwrap();
    let g4 = model.gaic(&Gaussian::new(), &y, 4.0).unwrap();
    assert!(g4 > g2, "GAIC(4) {} should exceed GAIC(2) {}", g4, g2);
}

#[test]
fn test_diagnostics_with_fitted_model() {
    let mut rng = Generator::new(123);
    let (y, data) = rng.linear_gaussian(200, 2.0, 5.0, 1.0);

    let formula = linear_intercepts("x", &["mu", "sigma"]);

    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();
    let diag = model.diagnostics(&Gaussian::new(), &y).unwrap();

    let pearson_res = &diag.pearson_residuals;
    let mean: f64 = pearson_res.iter().sum::<f64>() / pearson_res.len() as f64;
    let variance: f64 = pearson_res.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
        / (pearson_res.len() - 1) as f64;

    assert!(
        mean.abs() < 0.2,
        "Pearson residuals mean should be ~0, got {}",
        mean
    );
    assert!(
        (variance - 1.0).abs() < 0.3,
        "Pearson residuals variance should be ~1, got {}",
        variance
    );

    assert!(diag.log_likelihood.is_finite());
    assert!(diag.log_likelihood < 0.0);
    assert!(diag.aic > 0.0);
    assert!(diag.bic > diag.aic);
    assert_eq!(diag.n_obs, y.len());
}
