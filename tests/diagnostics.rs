mod common;

use common::Generator;
use gamlss_rs::{
    diagnostics::{
        compute_aic, compute_bic, loglik_gaussian, loglik_poisson, pearson_residuals_gaussian,
        pearson_residuals_poisson, response_residuals, total_edf,
    },
    distributions::Gaussian,
    GamlssModel, Term,
};
use ndarray::Array1;
use std::collections::HashMap;

#[test]
fn test_pearson_residuals_gaussian() {
    let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let mu = Array1::from_vec(vec![1.5, 2.0, 2.5, 4.5, 5.0]);
    let sigma = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5, 0.5]);

    let residuals = pearson_residuals_gaussian(&y, &mu, &sigma);

    // r_i = (y_i - mu_i) / sigma_i
    assert!((residuals[0] - (-1.0)).abs() < 1e-10); // (1 - 1.5) / 0.5 = -1
    assert!((residuals[1] - 0.0).abs() < 1e-10); // (2 - 2) / 0.5 = 0
    assert!((residuals[2] - 1.0).abs() < 1e-10); // (3 - 2.5) / 0.5 = 1
    assert!((residuals[3] - (-1.0)).abs() < 1e-10); // (4 - 4.5) / 0.5 = -1
    assert!((residuals[4] - 0.0).abs() < 1e-10); // (5 - 5) / 0.5 = 0
}

#[test]
fn test_pearson_residuals_poisson() {
    let y = Array1::from_vec(vec![0.0, 1.0, 4.0, 9.0, 16.0]);
    let mu = Array1::from_vec(vec![1.0, 1.0, 4.0, 9.0, 16.0]);

    let residuals = pearson_residuals_poisson(&y, &mu);

    // r_i = (y_i - mu_i) / sqrt(mu_i)
    assert!((residuals[0] - (-1.0)).abs() < 1e-10); // (0 - 1) / 1 = -1
    assert!((residuals[1] - 0.0).abs() < 1e-10); // (1 - 1) / 1 = 0
    assert!((residuals[2] - 0.0).abs() < 1e-10); // (4 - 4) / 2 = 0
    assert!((residuals[3] - 0.0).abs() < 1e-10); // (9 - 9) / 3 = 0
    assert!((residuals[4] - 0.0).abs() < 1e-10); // (16 - 16) / 4 = 0
}

#[test]
fn test_response_residuals() {
    let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let mu = Array1::from_vec(vec![1.5, 2.0, 2.5]);

    let residuals = response_residuals(&y, &mu);

    assert!((residuals[0] - (-0.5)).abs() < 1e-10);
    assert!((residuals[1] - 0.0).abs() < 1e-10);
    assert!((residuals[2] - 0.5).abs() < 1e-10);
}

#[test]
fn test_loglik_gaussian() {
    // Simple test: standard normal with y = mu (perfect fit)
    let y = Array1::from_vec(vec![0.0, 0.0, 0.0]);
    let mu = Array1::from_vec(vec![0.0, 0.0, 0.0]);
    let sigma = Array1::from_vec(vec![1.0, 1.0, 1.0]);

    let ll = loglik_gaussian(&y, &mu, &sigma);

    // For standard normal at y=0: l = -0.5*log(2*pi) - 0 = -0.9189...
    // Total for 3 observations: 3 * -0.9189 = -2.7568
    let expected = 3.0 * (-0.5 * (2.0 * std::f64::consts::PI).ln());
    assert!((ll - expected).abs() < 1e-6);
}

#[test]
fn test_loglik_poisson() {
    // Test with y = mu (perfect fit)
    let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let mu = Array1::from_vec(vec![1.0, 2.0, 3.0]);

    let ll = loglik_poisson(&y, &mu);

    // Should be finite and negative (since it's a log-likelihood)
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

    // AIC = -2*(-100) + 2*5 = 200 + 10 = 210
    assert!((aic - 210.0).abs() < 1e-10);

    // BIC = -2*(-100) + log(100)*5 = 200 + 4.605*5 = 200 + 23.03 = 223.03
    let expected_bic = 200.0 + (100.0_f64).ln() * 5.0;
    assert!((bic - expected_bic).abs() < 1e-6);
}

#[test]
fn test_total_edf() {
    let mut rng = Generator::new(42);
    let df = rng.linear_gaussian(100, 1.0, 5.0, 1.0);

    let mut formula = HashMap::new();
    formula.insert(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );
    formula.insert("sigma".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&df, "y", &formula, &Gaussian::new()).unwrap();

    let edf = total_edf(&model.models);

    // mu has 2 coefficients (intercept + slope), sigma has 1 (intercept)
    // Total should be close to 3
    assert!(
        edf > 2.5 && edf < 3.5,
        "Total EDF should be ~3, got {}",
        edf
    );
}

#[test]
fn test_diagnostics_with_fitted_model() {
    let mut rng = Generator::new(123);
    let df = rng.linear_gaussian(200, 2.0, 5.0, 1.0);

    let mut formula = HashMap::new();
    formula.insert(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );
    formula.insert("sigma".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&df, "y", &formula, &Gaussian::new()).unwrap();

    // Extract fitted values
    let mu = &model.models["mu"].fitted_values;
    let sigma = &model.models["sigma"].fitted_values;

    // Get y values
    let y_series = df.column("y").unwrap();
    let y_vec: Vec<f64> = y_series.f64().unwrap().into_no_null_iter().collect();
    let y = Array1::from_vec(y_vec);

    // Compute residuals
    let pearson_res = pearson_residuals_gaussian(&y, mu, sigma);

    // Pearson residuals should have mean close to 0 and variance close to 1
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

    // Compute log-likelihood and information criteria
    let ll = loglik_gaussian(&y, mu, sigma);
    let edf = total_edf(&model.models);
    let aic = compute_aic(ll, edf);
    let bic = compute_bic(ll, edf, y.len());

    // Basic sanity checks
    assert!(ll.is_finite(), "Log-likelihood should be finite");
    assert!(ll < 0.0, "Log-likelihood should be negative");
    assert!(aic > 0.0, "AIC should be positive");
    assert!(bic > aic, "BIC should be > AIC for n > e^2");
}
