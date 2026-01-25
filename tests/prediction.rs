mod common;

use common::Generator;
use gamlss_rs::{
    distributions::{Gaussian, Poisson},
    GamlssModel, Smooth, Term,
};
use polars::prelude::*;
use rand::Rng;
use std::collections::HashMap;

#[test]
fn test_predict_on_training_data() {
    // Predictions on training data should match fitted values
    let mut rng = Generator::new(42);
    let df = rng.linear_gaussian(100, 2.0, 5.0, 1.0);

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

    // Predict on the same data
    let predictions = model.predict(&df, &Gaussian::new()).unwrap();

    // Check that predictions match fitted values
    let mu_pred = &predictions["mu"];
    let mu_fitted = &model.models["mu"].fitted_values;

    for i in 0..mu_pred.len() {
        let diff = (mu_pred[i] - mu_fitted[i]).abs();
        assert!(
            diff < 1e-10,
            "Prediction should match fitted at index {}: {} vs {}",
            i,
            mu_pred[i],
            mu_fitted[i]
        );
    }
}

#[test]
fn test_predict_on_new_data() {
    // Test prediction on new data points
    let mut rng = Generator::new(123);
    let df_train = rng.linear_gaussian(200, 2.0, 5.0, 1.0);

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

    let model = GamlssModel::fit(&df_train, "y", &formula, &Gaussian::new()).unwrap();

    // Create new data
    let df_new = df!(
        "x" => [0.0, 50.0, 100.0, 150.0, 200.0]
    )
    .unwrap();

    let predictions = model.predict(&df_new, &Gaussian::new()).unwrap();
    let mu_pred = &predictions["mu"];

    // For linear model: mu = intercept + slope * x
    // Predictions should follow this pattern
    let coeffs = &model.models["mu"].coefficients.0;
    let intercept = coeffs[0];
    let slope = coeffs[1];

    for (i, &x) in [0.0, 50.0, 100.0, 150.0, 200.0].iter().enumerate() {
        let expected = intercept + slope * x;
        let diff = (mu_pred[i] - expected).abs();
        assert!(
            diff < 1e-10,
            "Prediction at x={} should be {}, got {}",
            x,
            expected,
            mu_pred[i]
        );
    }
}

#[test]
fn test_predict_with_se() {
    let mut rng = Generator::new(456);
    let df = rng.linear_gaussian(100, 2.0, 5.0, 1.0);

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

    let results = model.predict_with_se(&df, &Gaussian::new()).unwrap();

    let mu_result = &results["mu"];

    // Check that standard errors are positive
    for i in 0..mu_result.se_eta.len() {
        assert!(
            mu_result.se_eta[i] >= 0.0,
            "SE should be non-negative at index {}",
            i
        );
    }

    // For Gaussian with identity link, fitted should equal eta
    for i in 0..mu_result.fitted.len() {
        let diff = (mu_result.fitted[i] - mu_result.eta[i]).abs();
        assert!(diff < 1e-10, "For identity link, fitted should equal eta");
    }
}

#[test]
fn test_predict_poisson() {
    let mut rng = Generator::new(789);

    let n = 300;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * 2.0).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mu = (1.0 + 0.5 * xi).exp();
            let dist = rand_distr::Poisson::new(mu).unwrap();
            rng.rng.sample(dist)
        })
        .collect();

    let df = df!("x" => x, "y" => y).unwrap();

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

    let model = GamlssModel::fit(&df, "y", &formula, &Poisson::new()).unwrap();

    // Predict on training data
    let predictions = model.predict(&df, &Poisson::new()).unwrap();
    let mu_pred = &predictions["mu"];

    // All predictions should be positive (Poisson has log link)
    for i in 0..mu_pred.len() {
        assert!(
            mu_pred[i] > 0.0,
            "Poisson predictions should be positive, got {} at index {}",
            mu_pred[i],
            i
        );
    }
}

#[test]
fn test_posterior_samples() {
    let mut rng = Generator::new(999);
    let df = rng.linear_gaussian(100, 2.0, 5.0, 1.0);

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

    // Get posterior samples for mu
    let samples = model.posterior_samples("mu", 100);

    assert_eq!(samples.len(), 100, "Should have 100 samples");

    // Each sample should have 2 coefficients (intercept + slope)
    for (i, sample) in samples.iter().enumerate() {
        assert_eq!(sample.0.len(), 2, "Sample {} should have 2 coefficients", i);
    }

    // Sample mean should be close to fitted coefficients
    let fitted_coeffs = &model.models["mu"].coefficients.0;
    let mut mean_intercept = 0.0;
    let mut mean_slope = 0.0;
    for sample in &samples {
        mean_intercept += sample.0[0];
        mean_slope += sample.0[1];
    }
    mean_intercept /= samples.len() as f64;
    mean_slope /= samples.len() as f64;

    assert!(
        (mean_intercept - fitted_coeffs[0]).abs() < 1.0,
        "Sample mean intercept should be close to fitted"
    );
    assert!(
        (mean_slope - fitted_coeffs[1]).abs() < 0.1,
        "Sample mean slope should be close to fitted"
    );
}

#[test]
fn test_predict_samples() {
    let mut rng = Generator::new(111);
    let df = rng.linear_gaussian(50, 2.0, 5.0, 1.0);

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

    // Get prediction samples
    let pred_samples = model.predict_samples(&df, &Gaussian::new(), 50).unwrap();

    // Check mu predictions
    let mu_samples = &pred_samples["mu"];
    assert_eq!(mu_samples.len(), 50, "Should have 50 prediction samples");

    // Each sample should have predictions for all observations
    for sample in mu_samples {
        assert_eq!(sample.len(), 50, "Each sample should have 50 predictions");
    }
}

#[test]
fn test_predict_with_smooth() {
    let mut rng = Generator::new(222);

    let n = 200;
    let x: Vec<f64> = (0..n)
        .map(|i| i as f64 / n as f64 * 2.0 * std::f64::consts::PI)
        .collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| xi.sin() + rng.rng.sample::<f64, _>(rand_distr::StandardNormal) * 0.2)
        .collect();

    let df = df!("x" => x, "y" => y).unwrap();

    let mut formula = HashMap::new();
    formula.insert(
        "mu".to_string(),
        vec![Term::Smooth(Smooth::PSpline1D {
            col_name: "x".to_string(),
            n_splines: 10,
            degree: 3,
            penalty_order: 2,
        })],
    );
    formula.insert("sigma".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&df, "y", &formula, &Gaussian::new()).unwrap();

    // Predict on training data
    let predictions = model.predict(&df, &Gaussian::new()).unwrap();
    let mu_pred = &predictions["mu"];

    // Predictions should capture the sinusoidal pattern
    // Check that predictions at 0, pi, 2*pi are roughly 0, 0, 0 (sin values)
    let idx_0 = 0;
    let idx_pi = n / 2;
    let idx_2pi = n - 1;

    // At x=0, sin(0) = 0
    assert!(
        mu_pred[idx_0].abs() < 0.5,
        "Prediction at x=0 should be near 0, got {}",
        mu_pred[idx_0]
    );

    // At x=pi, sin(pi) = 0
    assert!(
        mu_pred[idx_pi].abs() < 0.5,
        "Prediction at x=pi should be near 0, got {}",
        mu_pred[idx_pi]
    );

    // At x=2*pi, sin(2*pi) = 0
    assert!(
        mu_pred[idx_2pi].abs() < 0.5,
        "Prediction at x=2*pi should be near 0, got {}",
        mu_pred[idx_2pi]
    );
}

#[test]
fn test_predict_missing_column_error() {
    let mut rng = Generator::new(333);
    let df_train = rng.linear_gaussian(100, 2.0, 5.0, 1.0);

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

    let model = GamlssModel::fit(&df_train, "y", &formula, &Gaussian::new()).unwrap();

    // Try to predict on data missing the 'x' column
    let df_bad = df!("z" => [1.0, 2.0, 3.0]).unwrap();

    let result = model.predict(&df_bad, &Gaussian::new());
    assert!(result.is_err(), "Should error when column is missing");
}
