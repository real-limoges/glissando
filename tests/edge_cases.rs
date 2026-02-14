mod common;

use common::Generator;
use gamlss_rs::{
    distributions::{Beta, Binomial, Gamma, Gaussian, NegativeBinomial, Poisson, StudentT},
    DataSet, Formula, GamlssModel, Smooth, Term,
};
use ndarray::Array1;
use rand::{Rng, RngExt};

// Helper to sample from Negative Binomial using Gamma-Poisson mixture
// NB(mu, sigma) where r = 1/sigma, Var(Y) = mu + sigma*mu^2
fn sample_negative_binomial(rng: &mut impl Rng, mu: f64, sigma: f64) -> f64 {
    let r = 1.0 / sigma; // size parameter
                         // Sample lambda ~ Gamma(shape=r, scale=mu/r)
    let gamma_dist = rand_distr::Gamma::new(r, mu / r).unwrap();
    let lambda: f64 = rng.sample(gamma_dist);
    // Sample y ~ Poisson(lambda)
    let poisson_dist = rand_distr::Poisson::new(lambda.max(1e-10)).unwrap();
    rng.sample(poisson_dist)
}

#[test]
fn test_poisson_with_smooth() {
    let mut rng = Generator::new(42);

    let n = 300;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * 4.0).collect();
    let y_vec: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mu = (1.0 + 0.5 * xi.sin()).exp();
            let dist = rand_distr::Poisson::new(mu).unwrap();
            rng.rng.sample(dist)
        })
        .collect();

    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("x".to_string(), Array1::from_vec(x));

    let mut formula = Formula::new();
    formula.add_terms(
        "mu".to_string(),
        vec![Term::Smooth(Smooth::PSpline1D {
            col_name: "x".to_string(),
            n_splines: 10,
            degree: 3,
            penalty_order: 2,
        })],
    );

    let model = GamlssModel::fit(&y, &data, &formula, &Poisson::new()).unwrap();

    let edf = model.models["mu"].edf;
    assert!(edf > 2.0, "EDF too low for nonlinear Poisson: {}", edf);
    assert!(edf < 10.0, "EDF too high: {}", edf);
}

#[test]
fn test_student_t_linear() {
    let mut rng = Generator::new(123);

    let n = 200;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let y_vec: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mu = 5.0 + 2.0 * xi;
            let t_sample: f64 = rng.rng.sample(rand_distr::StudentT::new(5.0).unwrap());
            mu + t_sample * 0.5
        })
        .collect();

    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("x".to_string(), Array1::from_vec(x));

    let mut formula = Formula::new();
    formula.add_terms(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );
    formula.add_terms("sigma".to_string(), vec![Term::Intercept]);
    formula.add_terms("nu".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&y, &data, &formula, &StudentT::new()).unwrap();

    let mu_coeffs = &model.models["mu"].coefficients;
    assert!(
        (mu_coeffs[0] - 5.0).abs() < 0.5,
        "Intercept should be ~5, got {}",
        mu_coeffs[0]
    );
    assert!(
        (mu_coeffs[1] - 2.0).abs() < 0.5,
        "Slope should be ~2, got {}",
        mu_coeffs[1]
    );
}

#[test]
fn test_different_spline_configs() {
    let mut rng = Generator::new(999);
    let (y, data) = rng.linear_gaussian(200, 1.0, 5.0, 1.0);

    for n_splines in [5, 10, 20] {
        let mut formula = Formula::new();
        formula.add_terms(
            "mu".to_string(),
            vec![Term::Smooth(Smooth::PSpline1D {
                col_name: "x".to_string(),
                n_splines,
                degree: 3,
                penalty_order: 2,
            })],
        );
        formula.add_terms("sigma".to_string(), vec![Term::Intercept]);

        let model = GamlssModel::fit(&y, &data, &formula, &Gaussian::new());
        assert!(model.is_ok(), "Failed with n_splines={}", n_splines);
    }
}

#[test]
fn test_penalty_order_1_vs_2() {
    let mut rng = Generator::new(42);

    let n = 200;
    let x: Vec<f64> = (0..n)
        .map(|i| i as f64 / n as f64 * 2.0 * std::f64::consts::PI)
        .collect();
    let y_vec: Vec<f64> = x
        .iter()
        .map(|&xi| xi.sin() + rng.rng.random_range(-0.1..0.1))
        .collect();

    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("x".to_string(), Array1::from_vec(x));

    // penalizes first differences
    let mut formula1 = Formula::new();
    formula1.add_terms(
        "mu".to_string(),
        vec![Term::Smooth(Smooth::PSpline1D {
            col_name: "x".to_string(),
            n_splines: 15,
            degree: 3,
            penalty_order: 1,
        })],
    );
    formula1.add_terms("sigma".to_string(), vec![Term::Intercept]);

    // penalizes second differences - curvature
    let mut formula2 = Formula::new();
    formula2.add_terms(
        "mu".to_string(),
        vec![Term::Smooth(Smooth::PSpline1D {
            col_name: "x".to_string(),
            n_splines: 15,
            degree: 3,
            penalty_order: 2,
        })],
    );
    formula2.add_terms("sigma".to_string(), vec![Term::Intercept]);

    let model1 = GamlssModel::fit(&y, &data, &formula1, &Gaussian::new()).unwrap();
    let model2 = GamlssModel::fit(&y, &data, &formula2, &Gaussian::new()).unwrap();

    assert!(model1.models["mu"].edf > 2.0);
    assert!(model2.models["mu"].edf > 2.0);
}

#[test]
fn test_very_noisy_data() {
    let mut rng = Generator::new(42);

    let n = 500;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let y_vec: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mu = 1.0 + 2.0 * xi;
            mu + rng.rng.random_range(-5.0..5.0) // Large noise
        })
        .collect();

    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("x".to_string(), Array1::from_vec(x));

    let mut formula = Formula::new();
    formula.add_terms(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );
    formula.add_terms("sigma".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&y, &data, &formula, &Gaussian::new()).unwrap();

    // Should still recover approximate slope despite noise
    let slope = model.models["mu"].coefficients[1];
    assert!(
        (slope - 2.0).abs() < 1.0,
        "Slope should be roughly ~2 even with noise, got {}",
        slope
    );
}

#[test]
fn test_perfect_linear_fit() {
    // perfect linear relationship
    let y = Array1::from_vec(vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]);
    let mut data = DataSet::new();
    data.insert_column(
        "x".to_string(),
        Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
    );

    let mut formula = Formula::new();
    formula.add_terms(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );
    formula.add_terms("sigma".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&y, &data, &formula, &Gaussian::new()).unwrap();

    let coeffs = &model.models["mu"].coefficients;
    assert!(
        coeffs[0].abs() < 1e-6,
        "Intercept should be ~0, got {}",
        coeffs[0]
    );
    assert!(
        (coeffs[1] - 2.0).abs() < 1e-6,
        "Slope should be exactly 2, got {}",
        coeffs[1]
    );
}

#[test]
fn test_lambdas_positive() {
    let mut rng = Generator::new(42);
    let (y, data) = rng.linear_gaussian(200, 1.0, 5.0, 1.0);

    let mut formula = Formula::new();
    formula.add_terms(
        "mu".to_string(),
        vec![Term::Smooth(Smooth::PSpline1D {
            col_name: "x".to_string(),
            n_splines: 10,
            degree: 3,
            penalty_order: 2,
        })],
    );
    formula.add_terms("sigma".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&y, &data, &formula, &Gaussian::new()).unwrap();

    // Smoothing parameters should be positive
    for &lambda in model.models["mu"].lambdas.iter() {
        assert!(lambda > 0.0, "Lambda should be positive, got {}", lambda);
    }
}

#[test]
fn test_covariance_symmetric() {
    let mut rng = Generator::new(42);
    let (y, data) = rng.linear_gaussian(100, 1.0, 5.0, 1.0);

    let mut formula = Formula::new();
    formula.add_terms(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );
    formula.add_terms("sigma".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&y, &data, &formula, &Gaussian::new()).unwrap();

    let cov = &model.models["mu"].covariance.0;
    let (n, m) = cov.dim();

    assert_eq!(n, m, "Covariance should be square");

    for i in 0..n {
        for j in 0..m {
            let diff = (cov[[i, j]] - cov[[j, i]]).abs();
            assert!(diff < 1e-10, "Covariance should be symmetric");
        }
    }
}

#[test]
fn test_fitted_values_match_eta_transform() {
    let mut rng = Generator::new(42);
    let (y, data) = rng.linear_gaussian(100, 1.0, 5.0, 1.0);

    let mut formula = Formula::new();
    formula.add_terms(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );
    formula.add_terms("sigma".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&y, &data, &formula, &Gaussian::new()).unwrap();

    // For Gaussian with identity link, fitted_values should equal eta
    let mu = &model.models["mu"];
    for i in 0..mu.eta.len() {
        let diff = (mu.fitted_values[i] - mu.eta[i]).abs();
        assert!(diff < 1e-10, "For identity link, fitted should equal eta");
    }

    // For sigma with log link, fitted_values should equal exp(eta)
    let sigma = &model.models["sigma"];
    for i in 0..sigma.eta.len() {
        let expected = sigma.eta[i].exp();
        let diff = (sigma.fitted_values[i] - expected).abs();
        assert!(diff < 1e-10, "For log link, fitted should equal exp(eta)");
    }
}

#[test]
fn test_random_effect_basic() {
    // Groups encoded as numeric indices: 0.0 = group A, 1.0 = group B, 2.0 = group C
    let group = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    let y = Array1::from_vec(vec![1.0, 1.2, 0.8, 5.0, 5.1, 4.9, 3.0, 3.1, 2.9]);

    let mut data = DataSet::new();
    data.insert_column("group".to_string(), group);

    let mut formula = Formula::new();
    formula.add_terms(
        "mu".to_string(),
        vec![Term::Smooth(Smooth::RandomEffect {
            col_name: "group".to_string(),
        })],
    );
    formula.add_terms("sigma".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&y, &data, &formula, &Gaussian::new()).unwrap();

    assert_eq!(
        model.models["mu"].coefficients.len(),
        3,
        "Should have one coefficient per group"
    );
}

#[test]
fn test_wide_data_more_predictors() {
    let mut rng = Generator::new(42);

    let n = 100;
    let x1: Vec<f64> = (0..n).map(|_| rng.rng.random::<f64>()).collect();
    let x2: Vec<f64> = (0..n).map(|_| rng.rng.random::<f64>()).collect();
    let x3: Vec<f64> = (0..n).map(|_| rng.rng.random::<f64>()).collect();
    let x4: Vec<f64> = (0..n).map(|_| rng.rng.random::<f64>()).collect();

    let y_vec: Vec<f64> = (0..n)
        .map(|i| 1.0 + x1[i] + 2.0 * x2[i] - x3[i] + 0.5 * x4[i] + rng.rng.random_range(-0.1..0.1))
        .collect();

    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("x1".to_string(), Array1::from_vec(x1));
    data.insert_column("x2".to_string(), Array1::from_vec(x2));
    data.insert_column("x3".to_string(), Array1::from_vec(x3));
    data.insert_column("x4".to_string(), Array1::from_vec(x4));

    let mut formula = Formula::new();
    formula.add_terms(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x1".to_string(),
            },
            Term::Linear {
                col_name: "x2".to_string(),
            },
            Term::Linear {
                col_name: "x3".to_string(),
            },
            Term::Linear {
                col_name: "x4".to_string(),
            },
        ],
    );
    formula.add_terms("sigma".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&y, &data, &formula, &Gaussian::new()).unwrap();

    let coeffs = &model.models["mu"].coefficients;
    assert_eq!(coeffs.len(), 5, "Should have 5 coefficients");

    // check recovery
    assert!((coeffs[1] - 1.0).abs() < 0.3, "x1 coef should be ~1");
    assert!((coeffs[2] - 2.0).abs() < 0.3, "x2 coef should be ~2");
    assert!((coeffs[3] - (-1.0)).abs() < 0.3, "x3 coef should be ~-1");
    assert!((coeffs[4] - 0.5).abs() < 0.3, "x4 coef should be ~0.5");
}

// ============================================================================
// Poisson Distribution Tests
// ============================================================================

#[test]
fn test_poisson_multiple_predictors() {
    let mut rng = Generator::new(555);

    let n = 500;
    let x1: Vec<f64> = (0..n).map(|_| rng.rng.random::<f64>() * 2.0).collect();
    let x2: Vec<f64> = (0..n).map(|_| rng.rng.random::<f64>() * 2.0).collect();

    // True model: log(mu) = 1.0 + 0.5*x1 - 0.3*x2
    let y_vec: Vec<f64> = (0..n)
        .map(|i| {
            let log_mu = 1.0 + 0.5 * x1[i] - 0.3 * x2[i];
            let mu = log_mu.exp();
            let dist = rand_distr::Poisson::new(mu).unwrap();
            rng.rng.sample(dist)
        })
        .collect();

    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("x1".to_string(), Array1::from_vec(x1));
    data.insert_column("x2".to_string(), Array1::from_vec(x2));

    let mut formula = Formula::new();
    formula.add_terms(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x1".to_string(),
            },
            Term::Linear {
                col_name: "x2".to_string(),
            },
        ],
    );

    let model = GamlssModel::fit(&y, &data, &formula, &Poisson::new()).unwrap();

    let coeffs = &model.models["mu"].coefficients;
    assert!(
        (coeffs[0] - 1.0).abs() < 0.15,
        "Poisson intercept should be ~1.0, got {}",
        coeffs[0]
    );
    assert!(
        (coeffs[1] - 0.5).abs() < 0.15,
        "Poisson x1 coef should be ~0.5, got {}",
        coeffs[1]
    );
    assert!(
        (coeffs[2] - (-0.3)).abs() < 0.15,
        "Poisson x2 coef should be ~-0.3, got {}",
        coeffs[2]
    );
}

#[test]
fn test_poisson_high_rate() {
    // Test Poisson with high mean values (numerical stability)
    let mut rng = Generator::new(777);

    let n = 400;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * 2.0).collect();

    // True model: log(mu) = 3.0 + 1.0*x => mu ranges from ~20 to ~109
    let y_vec: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mu = (3.0 + 1.0 * xi).exp();
            let dist = rand_distr::Poisson::new(mu).unwrap();
            rng.rng.sample(dist)
        })
        .collect();

    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("x".to_string(), Array1::from_vec(x));

    let mut formula = Formula::new();
    formula.add_terms(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );

    let model = GamlssModel::fit(&y, &data, &formula, &Poisson::new()).unwrap();

    let coeffs = &model.models["mu"].coefficients;
    assert!(
        (coeffs[0] - 3.0).abs() < 0.15,
        "High-rate Poisson intercept should be ~3.0, got {}",
        coeffs[0]
    );
    assert!(
        (coeffs[1] - 1.0).abs() < 0.15,
        "High-rate Poisson slope should be ~1.0, got {}",
        coeffs[1]
    );
}

#[test]
fn test_poisson_smooth_nonlinear() {
    // Test Poisson with a nonlinear smooth relationship
    let mut rng = Generator::new(888);

    let n = 400;
    let x: Vec<f64> = (0..n)
        .map(|i| i as f64 / n as f64 * 2.0 * std::f64::consts::PI)
        .collect();

    // True model: log(mu) = 2.0 + 0.5*sin(x)
    let y_vec: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mu = (2.0 + 0.5 * xi.sin()).exp();
            let dist = rand_distr::Poisson::new(mu).unwrap();
            rng.rng.sample(dist)
        })
        .collect();

    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("x".to_string(), Array1::from_vec(x));

    let mut formula = Formula::new();
    formula.add_terms(
        "mu".to_string(),
        vec![Term::Smooth(Smooth::PSpline1D {
            col_name: "x".to_string(),
            n_splines: 12,
            degree: 3,
            penalty_order: 2,
        })],
    );

    let model = GamlssModel::fit(&y, &data, &formula, &Poisson::new()).unwrap();

    let edf = model.models["mu"].edf;
    // Should capture some curvature but not overfit
    assert!(
        edf > 3.0,
        "Poisson smooth EDF too low for sinusoidal pattern: {}",
        edf
    );
    assert!(edf < 10.0, "Poisson smooth EDF too high: {}", edf);
}

#[test]
fn test_poisson_low_counts() {
    // Test Poisson with very low counts (edge case)
    let mut rng = Generator::new(111);

    let n = 300;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();

    // True model: log(mu) = -0.5 + 1.0*x => mu ranges from ~0.6 to ~1.6
    let y_vec: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mu = (-0.5 + 1.0 * xi).exp();
            let dist = rand_distr::Poisson::new(mu).unwrap();
            rng.rng.sample(dist)
        })
        .collect();

    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("x".to_string(), Array1::from_vec(x));

    let mut formula = Formula::new();
    formula.add_terms(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );

    let model = GamlssModel::fit(&y, &data, &formula, &Poisson::new()).unwrap();

    let coeffs = &model.models["mu"].coefficients;
    // Lower precision for low-count data
    assert!(
        (coeffs[0] - (-0.5)).abs() < 0.3,
        "Low-count Poisson intercept should be ~-0.5, got {}",
        coeffs[0]
    );
    assert!(
        (coeffs[1] - 1.0).abs() < 0.3,
        "Low-count Poisson slope should be ~1.0, got {}",
        coeffs[1]
    );
}

// ============================================================================
// Student-t Distribution Tests
// ============================================================================

#[test]
fn test_student_t_smooth_mu() {
    // Test StudentT with smooth mu relationship
    let mut rng = Generator::new(222);

    let n = 500;
    let nu = 5.0; // degrees of freedom
    let sigma = 0.5;

    let x: Vec<f64> = (0..n)
        .map(|i| i as f64 / n as f64 * 2.0 * std::f64::consts::PI)
        .collect();

    let y_vec: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mu = 3.0 * xi.sin(); // sinusoidal mean
            let t_sample: f64 = rng.rng.sample(rand_distr::StudentT::new(nu).unwrap());
            mu + sigma * t_sample
        })
        .collect();

    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("x".to_string(), Array1::from_vec(x));

    let mut formula = Formula::new();
    formula.add_terms(
        "mu".to_string(),
        vec![Term::Smooth(Smooth::PSpline1D {
            col_name: "x".to_string(),
            n_splines: 15,
            degree: 3,
            penalty_order: 2,
        })],
    );
    formula.add_terms("sigma".to_string(), vec![Term::Intercept]);
    formula.add_terms("nu".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&y, &data, &formula, &StudentT::new()).unwrap();

    let edf = model.models["mu"].edf;
    assert!(
        edf > 3.0,
        "StudentT smooth mu EDF too low for sinusoidal: {}",
        edf
    );
    assert!(edf < 15.0, "StudentT smooth mu EDF too high: {}", edf);
}

#[test]
fn test_student_t_heteroskedastic() {
    // Test StudentT with varying sigma
    let mut rng = Generator::new(333);

    let n = 800;
    let nu = 6.0;

    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * 3.0).collect();

    // True model:
    // mu = 5.0 + 2.0*x
    // log(sigma) = -1.0 + 0.5*x => sigma varies from ~0.37 to ~0.82
    let y_vec: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mu = 5.0 + 2.0 * xi;
            let sigma = (-1.0 + 0.5 * xi).exp();
            let t_sample: f64 = rng.rng.sample(rand_distr::StudentT::new(nu).unwrap());
            mu + sigma * t_sample
        })
        .collect();

    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("x".to_string(), Array1::from_vec(x));

    let mut formula = Formula::new();
    formula.add_terms(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );
    formula.add_terms(
        "sigma".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );
    formula.add_terms("nu".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&y, &data, &formula, &StudentT::new()).unwrap();

    let mu_coeffs = &model.models["mu"].coefficients;
    let sigma_coeffs = &model.models["sigma"].coefficients;

    // Check mu recovery
    assert!(
        (mu_coeffs[0] - 5.0).abs() < 0.5,
        "StudentT hetero mu intercept should be ~5.0, got {}",
        mu_coeffs[0]
    );
    assert!(
        (mu_coeffs[1] - 2.0).abs() < 0.3,
        "StudentT hetero mu slope should be ~2.0, got {}",
        mu_coeffs[1]
    );

    // Check sigma recovery (log link)
    assert!(
        (sigma_coeffs[0] - (-1.0)).abs() < 0.4,
        "StudentT hetero sigma intercept should be ~-1.0, got {}",
        sigma_coeffs[0]
    );
    assert!(
        (sigma_coeffs[1] - 0.5).abs() < 0.3,
        "StudentT hetero sigma slope should be ~0.5, got {}",
        sigma_coeffs[1]
    );
}

#[test]
fn test_student_t_heavy_tails() {
    // Test StudentT with very low degrees of freedom (heavy tails)
    let mut rng = Generator::new(444);

    let n = 1000;
    let true_nu = 3.0; // Heavy tails
    let true_mu = 10.0;
    let true_sigma = 1.0;

    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();

    let y_vec: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mu = true_mu + 2.0 * xi;
            let t_sample: f64 = rng.rng.sample(rand_distr::StudentT::new(true_nu).unwrap());
            mu + true_sigma * t_sample
        })
        .collect();

    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("x".to_string(), Array1::from_vec(x));

    let mut formula = Formula::new();
    formula.add_terms(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );
    formula.add_terms("sigma".to_string(), vec![Term::Intercept]);
    formula.add_terms("nu".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&y, &data, &formula, &StudentT::new()).unwrap();

    let nu_coeff = model.models["nu"].coefficients[0];
    let fitted_nu = nu_coeff.exp();

    // Nu estimation is noisy but should be in reasonable range for heavy tails
    assert!(
        fitted_nu < 10.0,
        "StudentT should detect heavy tails (low nu), got nu={}",
        fitted_nu
    );
    assert!(
        fitted_nu > 1.5,
        "Fitted nu too low (unstable), got nu={}",
        fitted_nu
    );
}

#[test]
fn test_student_t_multiple_predictors() {
    // Test StudentT with multiple linear predictors
    let mut rng = Generator::new(666);

    let n = 600;
    let nu = 5.0;
    let sigma = 0.8;

    let x1: Vec<f64> = (0..n).map(|_| rng.rng.random::<f64>() * 2.0).collect();
    let x2: Vec<f64> = (0..n).map(|_| rng.rng.random::<f64>() * 2.0).collect();
    let x3: Vec<f64> = (0..n).map(|_| rng.rng.random::<f64>() * 2.0).collect();

    // True model: mu = 2.0 + 1.5*x1 - 0.8*x2 + 0.5*x3
    let y_vec: Vec<f64> = (0..n)
        .map(|i| {
            let mu = 2.0 + 1.5 * x1[i] - 0.8 * x2[i] + 0.5 * x3[i];
            let t_sample: f64 = rng.rng.sample(rand_distr::StudentT::new(nu).unwrap());
            mu + sigma * t_sample
        })
        .collect();

    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("x1".to_string(), Array1::from_vec(x1));
    data.insert_column("x2".to_string(), Array1::from_vec(x2));
    data.insert_column("x3".to_string(), Array1::from_vec(x3));

    let mut formula = Formula::new();
    formula.add_terms(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x1".to_string(),
            },
            Term::Linear {
                col_name: "x2".to_string(),
            },
            Term::Linear {
                col_name: "x3".to_string(),
            },
        ],
    );
    formula.add_terms("sigma".to_string(), vec![Term::Intercept]);
    formula.add_terms("nu".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&y, &data, &formula, &StudentT::new()).unwrap();

    let coeffs = &model.models["mu"].coefficients;
    assert!(
        (coeffs[0] - 2.0).abs() < 0.4,
        "StudentT intercept should be ~2.0, got {}",
        coeffs[0]
    );
    assert!(
        (coeffs[1] - 1.5).abs() < 0.3,
        "StudentT x1 coef should be ~1.5, got {}",
        coeffs[1]
    );
    assert!(
        (coeffs[2] - (-0.8)).abs() < 0.3,
        "StudentT x2 coef should be ~-0.8, got {}",
        coeffs[2]
    );
    assert!(
        (coeffs[3] - 0.5).abs() < 0.3,
        "StudentT x3 coef should be ~0.5, got {}",
        coeffs[3]
    );
}

#[test]
fn test_student_t_near_gaussian() {
    // Test StudentT with high degrees of freedom (should behave like Gaussian)
    let mut rng = Generator::new(999);

    let n = 500;
    let true_nu = 30.0; // High nu => nearly Gaussian
    let true_mu_int = 5.0;
    let true_mu_slope = 3.0;
    let true_sigma = 1.0;

    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * 2.0).collect();

    let y_vec: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mu = true_mu_int + true_mu_slope * xi;
            let t_sample: f64 = rng.rng.sample(rand_distr::StudentT::new(true_nu).unwrap());
            mu + true_sigma * t_sample
        })
        .collect();

    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("x".to_string(), Array1::from_vec(x));

    let mut formula = Formula::new();
    formula.add_terms(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );
    formula.add_terms("sigma".to_string(), vec![Term::Intercept]);
    formula.add_terms("nu".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&y, &data, &formula, &StudentT::new()).unwrap();

    let mu_coeffs = &model.models["mu"].coefficients;

    // With high nu, should recover parameters similar to Gaussian
    assert!(
        (mu_coeffs[0] - true_mu_int).abs() < 0.3,
        "Near-Gaussian StudentT intercept should be ~{}, got {}",
        true_mu_int,
        mu_coeffs[0]
    );
    assert!(
        (mu_coeffs[1] - true_mu_slope).abs() < 0.3,
        "Near-Gaussian StudentT slope should be ~{}, got {}",
        true_mu_slope,
        mu_coeffs[1]
    );

    // Fitted nu should be reasonably high (nu estimation is noisy for near-Gaussian data
    // since there's little tail information to distinguish moderate from high nu)
    let fitted_nu = model.models["nu"].coefficients[0].exp();
    assert!(
        fitted_nu > 5.0,
        "Near-Gaussian StudentT should have moderate-to-high nu, got {}",
        fitted_nu
    );
}

// ============================================================================
// Gamma Distribution Tests
// ============================================================================

#[test]
fn test_gamma_linear_mu() {
    // Test Gamma with linear relationship for mu
    let mut rng = Generator::new(1001);

    let n = 500;
    let true_sigma = 0.5; // CV = 0.5
    let shape = 1.0 / (true_sigma * true_sigma); // alpha = 4

    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * 2.0).collect();

    // True model: log(mu) = 1.0 + 0.5*x => mu from ~2.7 to ~7.4
    let y_vec: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mu = (1.0 + 0.5 * xi).exp();
            let scale = mu / shape; // theta = mu * sigma^2 = mu / alpha
            let gamma_dist = rand_distr::Gamma::new(shape, scale).unwrap();
            rng.rng.sample(gamma_dist)
        })
        .collect();

    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("x".to_string(), Array1::from_vec(x));

    let mut formula = Formula::new();
    formula.add_terms(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );
    formula.add_terms("sigma".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&y, &data, &formula, &Gamma::new()).unwrap();

    let mu_coeffs = &model.models["mu"].coefficients;
    // Coefficients are on log scale due to log link
    assert!(
        (mu_coeffs[0] - 1.0).abs() < 0.2,
        "Gamma mu intercept should be ~1.0, got {}",
        mu_coeffs[0]
    );
    assert!(
        (mu_coeffs[1] - 0.5).abs() < 0.2,
        "Gamma mu slope should be ~0.5, got {}",
        mu_coeffs[1]
    );
}

#[test]
fn test_gamma_heteroscedastic() {
    // Test Gamma with varying sigma (coefficient of variation)
    let mut rng = Generator::new(1002);

    let n = 600;

    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * 2.0).collect();

    // True model:
    // log(mu) = 2.0 + 0.3*x
    // log(sigma) = -1.0 + 0.4*x => sigma varies from ~0.37 to ~0.82
    let y_vec: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mu = (2.0 + 0.3 * xi).exp();
            let sigma = (-1.0 + 0.4 * xi).exp();
            let shape = 1.0 / (sigma * sigma);
            let scale = mu / shape;
            let gamma_dist = rand_distr::Gamma::new(shape, scale).unwrap();
            rng.rng.sample(gamma_dist)
        })
        .collect();

    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("x".to_string(), Array1::from_vec(x));

    let mut formula = Formula::new();
    formula.add_terms(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );
    formula.add_terms(
        "sigma".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );

    let model = GamlssModel::fit(&y, &data, &formula, &Gamma::new()).unwrap();

    let mu_coeffs = &model.models["mu"].coefficients;
    let sigma_coeffs = &model.models["sigma"].coefficients;

    assert!(
        (mu_coeffs[0] - 2.0).abs() < 0.3,
        "Gamma hetero mu intercept should be ~2.0, got {}",
        mu_coeffs[0]
    );
    assert!(
        (mu_coeffs[1] - 0.3).abs() < 0.2,
        "Gamma hetero mu slope should be ~0.3, got {}",
        mu_coeffs[1]
    );
    assert!(
        (sigma_coeffs[0] - (-1.0)).abs() < 0.4,
        "Gamma hetero sigma intercept should be ~-1.0, got {}",
        sigma_coeffs[0]
    );
    assert!(
        (sigma_coeffs[1] - 0.4).abs() < 0.3,
        "Gamma hetero sigma slope should be ~0.4, got {}",
        sigma_coeffs[1]
    );
}

#[test]
fn test_gamma_smooth_mu() {
    // Test Gamma with smooth mu relationship
    let mut rng = Generator::new(1003);

    let n = 400;
    let sigma = 0.4;
    let shape = 1.0 / (sigma * sigma);

    let x: Vec<f64> = (0..n)
        .map(|i| i as f64 / n as f64 * 2.0 * std::f64::consts::PI)
        .collect();

    // True model: log(mu) = 2.0 + 0.3*sin(x)
    let y_vec: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mu = (2.0 + 0.3 * xi.sin()).exp();
            let scale = mu / shape;
            let gamma_dist = rand_distr::Gamma::new(shape, scale).unwrap();
            rng.rng.sample(gamma_dist)
        })
        .collect();

    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("x".to_string(), Array1::from_vec(x));

    let mut formula = Formula::new();
    formula.add_terms(
        "mu".to_string(),
        vec![Term::Smooth(Smooth::PSpline1D {
            col_name: "x".to_string(),
            n_splines: 12,
            degree: 3,
            penalty_order: 2,
        })],
    );
    formula.add_terms("sigma".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&y, &data, &formula, &Gamma::new()).unwrap();

    let edf = model.models["mu"].edf;
    assert!(
        edf > 2.0,
        "Gamma smooth mu EDF too low for sinusoidal: {}",
        edf
    );
    assert!(edf < 12.0, "Gamma smooth mu EDF too high: {}", edf);
}

// ============================================================================
// Negative Binomial Distribution Tests
// ============================================================================

#[test]
fn test_negative_binomial_linear() {
    // Test Negative Binomial with linear mu relationship
    let mut rng = Generator::new(2001);

    let n = 500;
    let true_sigma = 0.5; // overdispersion parameter

    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * 2.0).collect();

    // True model: log(mu) = 1.5 + 0.5*x => mu from ~4.5 to ~12.2
    let y_vec: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mu = (1.5 + 0.5 * xi).exp();
            sample_negative_binomial(&mut rng.rng, mu, true_sigma)
        })
        .collect();

    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("x".to_string(), Array1::from_vec(x));

    let mut formula = Formula::new();
    formula.add_terms(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );
    formula.add_terms("sigma".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&y, &data, &formula, &NegativeBinomial::new()).unwrap();

    let mu_coeffs = &model.models["mu"].coefficients;
    assert!(
        (mu_coeffs[0] - 1.5).abs() < 0.3,
        "NB mu intercept should be ~1.5, got {}",
        mu_coeffs[0]
    );
    assert!(
        (mu_coeffs[1] - 0.5).abs() < 0.2,
        "NB mu slope should be ~0.5, got {}",
        mu_coeffs[1]
    );
}

#[test]
fn test_negative_binomial_overdispersed() {
    // Test NB with high overdispersion (distinct from Poisson)
    let mut rng = Generator::new(2002);

    let n = 600;
    let true_sigma = 1.0; // high overdispersion

    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * 2.0).collect();

    // True model: log(mu) = 2.0 + 0.3*x
    let y_vec: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mu = (2.0 + 0.3 * xi).exp();
            sample_negative_binomial(&mut rng.rng, mu, true_sigma)
        })
        .collect();

    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("x".to_string(), Array1::from_vec(x));

    let mut formula = Formula::new();
    formula.add_terms(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );
    formula.add_terms("sigma".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&y, &data, &formula, &NegativeBinomial::new()).unwrap();

    let mu_coeffs = &model.models["mu"].coefficients;
    assert!(
        (mu_coeffs[0] - 2.0).abs() < 0.3,
        "NB overdispersed mu intercept should be ~2.0, got {}",
        mu_coeffs[0]
    );
    assert!(
        (mu_coeffs[1] - 0.3).abs() < 0.2,
        "NB overdispersed mu slope should be ~0.3, got {}",
        mu_coeffs[1]
    );

    // Check that sigma is estimated reasonably
    let sigma_coeff = model.models["sigma"].coefficients[0];
    let fitted_sigma = sigma_coeff.exp();
    assert!(
        fitted_sigma > 0.3,
        "NB should detect overdispersion, got sigma={}",
        fitted_sigma
    );
}

#[test]
fn test_negative_binomial_smooth() {
    // Test NB with smooth mu relationship
    let mut rng = Generator::new(2003);

    let n = 400;
    let sigma = 0.3;

    let x: Vec<f64> = (0..n)
        .map(|i| i as f64 / n as f64 * 2.0 * std::f64::consts::PI)
        .collect();

    // True model: log(mu) = 2.5 + 0.5*sin(x)
    let y_vec: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mu = (2.5 + 0.5 * xi.sin()).exp();
            sample_negative_binomial(&mut rng.rng, mu, sigma)
        })
        .collect();

    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("x".to_string(), Array1::from_vec(x));

    let mut formula = Formula::new();
    formula.add_terms(
        "mu".to_string(),
        vec![Term::Smooth(Smooth::PSpline1D {
            col_name: "x".to_string(),
            n_splines: 12,
            degree: 3,
            penalty_order: 2,
        })],
    );
    formula.add_terms("sigma".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&y, &data, &formula, &NegativeBinomial::new()).unwrap();

    let edf = model.models["mu"].edf;
    assert!(
        edf > 2.0,
        "NB smooth mu EDF too low for sinusoidal: {}",
        edf
    );
    assert!(edf < 12.0, "NB smooth mu EDF too high: {}", edf);
}

#[test]
fn test_negative_binomial_multiple_predictors() {
    // Test NB with multiple linear predictors
    let mut rng = Generator::new(2004);

    let n = 600;
    let sigma = 0.4;

    let x1: Vec<f64> = (0..n).map(|_| rng.rng.random::<f64>() * 2.0).collect();
    let x2: Vec<f64> = (0..n).map(|_| rng.rng.random::<f64>() * 2.0).collect();

    // True model: log(mu) = 1.0 + 0.5*x1 - 0.3*x2
    let y_vec: Vec<f64> = (0..n)
        .map(|i| {
            let mu = (1.0 + 0.5 * x1[i] - 0.3 * x2[i]).exp();
            sample_negative_binomial(&mut rng.rng, mu, sigma)
        })
        .collect();

    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("x1".to_string(), Array1::from_vec(x1));
    data.insert_column("x2".to_string(), Array1::from_vec(x2));

    let mut formula = Formula::new();
    formula.add_terms(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x1".to_string(),
            },
            Term::Linear {
                col_name: "x2".to_string(),
            },
        ],
    );
    formula.add_terms("sigma".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&y, &data, &formula, &NegativeBinomial::new()).unwrap();

    let mu_coeffs = &model.models["mu"].coefficients;
    assert!(
        (mu_coeffs[0] - 1.0).abs() < 0.3,
        "NB intercept should be ~1.0, got {}",
        mu_coeffs[0]
    );
    assert!(
        (mu_coeffs[1] - 0.5).abs() < 0.2,
        "NB x1 coef should be ~0.5, got {}",
        mu_coeffs[1]
    );
    assert!(
        (mu_coeffs[2] - (-0.3)).abs() < 0.2,
        "NB x2 coef should be ~-0.3, got {}",
        mu_coeffs[2]
    );
}

// ============================================================================
// Beta Distribution Tests
// ============================================================================

// Helper to sample from Beta distribution
fn sample_beta(rng: &mut impl Rng, alpha: f64, beta: f64) -> f64 {
    let beta_dist = rand_distr::Beta::new(alpha, beta).unwrap();
    rng.sample(beta_dist)
}

#[test]
fn test_beta_linear_mu() {
    // Test Beta with linear relationship for mu (on logit scale)
    let mut rng = Generator::new(3001);

    let n = 500;
    let true_phi = 10.0; // precision parameter

    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * 2.0 - 1.0).collect(); // x in [-1, 1]

    // True model: logit(mu) = 0.0 + 0.5*x => mu varies around 0.5
    let y_vec: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let eta = 0.0 + 0.5 * xi;
            let mu = 1.0 / (1.0 + (-eta).exp()); // inverse logit
            let alpha = mu * true_phi;
            let beta_param = (1.0 - mu) * true_phi;
            sample_beta(&mut rng.rng, alpha, beta_param)
        })
        .collect();

    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("x".to_string(), Array1::from_vec(x));

    let mut formula = Formula::new();
    formula.add_terms(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );
    formula.add_terms("phi".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&y, &data, &formula, &Beta::new()).unwrap();

    let mu_coeffs = &model.models["mu"].coefficients;
    // Coefficients are on logit scale
    assert!(
        mu_coeffs[0].abs() < 0.3,
        "Beta mu intercept should be ~0.0, got {}",
        mu_coeffs[0]
    );
    assert!(
        (mu_coeffs[1] - 0.5).abs() < 0.3,
        "Beta mu slope should be ~0.5, got {}",
        mu_coeffs[1]
    );
}

#[test]
fn test_beta_varying_precision() {
    // Test Beta with varying phi (precision)
    let mut rng = Generator::new(3002);

    let n = 600;

    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * 2.0).collect();

    // True model:
    // logit(mu) = 0.0 (constant mu = 0.5)
    // log(phi) = 1.0 + 0.5*x => phi varies from ~2.7 to ~7.4
    let y_vec: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mu = 0.5;
            let phi = (1.0 + 0.5 * xi).exp();
            let alpha = mu * phi;
            let beta_param = (1.0 - mu) * phi;
            sample_beta(&mut rng.rng, alpha, beta_param)
        })
        .collect();

    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("x".to_string(), Array1::from_vec(x));

    let mut formula = Formula::new();
    formula.add_terms("mu".to_string(), vec![Term::Intercept]);
    formula.add_terms(
        "phi".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );

    let model = GamlssModel::fit(&y, &data, &formula, &Beta::new()).unwrap();

    let phi_coeffs = &model.models["phi"].coefficients;
    assert!(
        (phi_coeffs[0] - 1.0).abs() < 0.4,
        "Beta phi intercept should be ~1.0, got {}",
        phi_coeffs[0]
    );
    assert!(
        (phi_coeffs[1] - 0.5).abs() < 0.3,
        "Beta phi slope should be ~0.5, got {}",
        phi_coeffs[1]
    );
}

#[test]
fn test_beta_smooth_mu() {
    // Test Beta with smooth mu relationship
    let mut rng = Generator::new(3003);

    let n = 400;
    let phi = 15.0;

    let x: Vec<f64> = (0..n)
        .map(|i| i as f64 / n as f64 * 2.0 * std::f64::consts::PI)
        .collect();

    // True model: logit(mu) = 0.3*sin(x) => mu oscillates around 0.5
    let y_vec: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let eta = 0.3 * xi.sin();
            let mu = 1.0 / (1.0 + (-eta).exp());
            let alpha = mu * phi;
            let beta_param = (1.0 - mu) * phi;
            sample_beta(&mut rng.rng, alpha, beta_param)
        })
        .collect();

    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("x".to_string(), Array1::from_vec(x));

    let mut formula = Formula::new();
    formula.add_terms(
        "mu".to_string(),
        vec![Term::Smooth(Smooth::PSpline1D {
            col_name: "x".to_string(),
            n_splines: 12,
            degree: 3,
            penalty_order: 2,
        })],
    );
    formula.add_terms("phi".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&y, &data, &formula, &Beta::new()).unwrap();

    let edf = model.models["mu"].edf;
    assert!(edf > 2.0, "Beta smooth mu EDF too low: {}", edf);
    assert!(edf < 12.0, "Beta smooth mu EDF too high: {}", edf);
}

#[test]
fn test_beta_high_precision() {
    // Test Beta with high precision (low variance, data clustered around mean)
    let mut rng = Generator::new(3004);

    let n = 400;
    let true_phi = 50.0; // high precision

    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();

    // True model: logit(mu) = -0.5 + 1.0*x
    let y_vec: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let eta = -0.5 + 1.0 * xi;
            let mu = 1.0 / (1.0 + (-eta).exp());
            let alpha = mu * true_phi;
            let beta_param = (1.0 - mu) * true_phi;
            sample_beta(&mut rng.rng, alpha, beta_param)
        })
        .collect();

    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("x".to_string(), Array1::from_vec(x));

    let mut formula = Formula::new();
    formula.add_terms(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );
    formula.add_terms("phi".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&y, &data, &formula, &Beta::new()).unwrap();

    let mu_coeffs = &model.models["mu"].coefficients;
    assert!(
        (mu_coeffs[0] - (-0.5)).abs() < 0.3,
        "Beta high-precision mu intercept should be ~-0.5, got {}",
        mu_coeffs[0]
    );
    assert!(
        (mu_coeffs[1] - 1.0).abs() < 0.3,
        "Beta high-precision mu slope should be ~1.0, got {}",
        mu_coeffs[1]
    );

    // Check that phi is estimated as high
    let phi_coeff = model.models["phi"].coefficients[0];
    let fitted_phi = phi_coeff.exp();
    assert!(
        fitted_phi > 20.0,
        "Beta should detect high precision, got phi={}",
        fitted_phi
    );
}

// ============================================================================
// Binomial Distribution Tests
// ============================================================================

#[test]
fn test_binomial_linear() {
    let mut rng = Generator::new(42);

    let n = 300;
    let n_trials = 20; // Number of trials per observation
    let true_intercept = -0.5; // logit scale
    let true_slope = 2.0;

    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let y_vec: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let eta = true_intercept + true_slope * xi;
            let mu = 1.0 / (1.0 + (-eta).exp()); // inverse logit
            let dist = rand_distr::Binomial::new(n_trials as u64, mu).unwrap();
            rng.rng.sample(dist) as f64
        })
        .collect();

    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("x".to_string(), Array1::from_vec(x));

    let mut formula = Formula::new();
    formula.add_terms(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );

    let model = GamlssModel::fit(&y, &data, &formula, &Binomial::new(n_trials)).unwrap();

    let mu_coeffs = &model.models["mu"].coefficients;
    assert!(
        (mu_coeffs[0] - true_intercept).abs() < 0.5,
        "Binomial intercept should be ~{}, got {}",
        true_intercept,
        mu_coeffs[0]
    );
    assert!(
        (mu_coeffs[1] - true_slope).abs() < 0.5,
        "Binomial slope should be ~{}, got {}",
        true_slope,
        mu_coeffs[1]
    );
}

#[test]
fn test_binomial_high_probability() {
    // Test with high success probability
    let mut rng = Generator::new(123);

    let n = 200;
    let n_trials = 50;
    let true_mu = 0.8; // High probability

    let y_vec: Vec<f64> = (0..n)
        .map(|_| {
            let dist = rand_distr::Binomial::new(n_trials as u64, true_mu).unwrap();
            rng.rng.sample(dist) as f64
        })
        .collect();

    let y = Array1::from_vec(y_vec);
    let data = DataSet::new();

    let mut formula = Formula::new();
    formula.add_terms("mu".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&y, &data, &formula, &Binomial::new(n_trials)).unwrap();

    // Check fitted probability is close to true value
    let mu_coeff = model.models["mu"].coefficients[0];
    let fitted_mu = 1.0 / (1.0 + (-mu_coeff).exp()); // inverse logit
    assert!(
        (fitted_mu - true_mu).abs() < 0.1,
        "Binomial should recover mu ~{}, got {}",
        true_mu,
        fitted_mu
    );
}

#[test]
fn test_binomial_multiple_predictors() {
    // Test Binomial with multiple linear predictors
    let mut rng = Generator::new(456);

    let n = 300;
    let n_trials = 30;

    let x1: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let x2: Vec<f64> = (0..n).map(|_| rng.rng.random::<f64>()).collect();

    let y_vec: Vec<f64> = x1
        .iter()
        .zip(x2.iter())
        .map(|(&xi1, &xi2)| {
            let eta = -0.5 + 1.5 * xi1 + 0.8 * xi2;
            let mu: f64 = 1.0 / (1.0 + (-eta).exp());
            let dist = rand_distr::Binomial::new(n_trials as u64, mu.clamp(0.05, 0.95)).unwrap();
            rng.rng.sample(dist) as f64
        })
        .collect();

    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("x1".to_string(), Array1::from_vec(x1));
    data.insert_column("x2".to_string(), Array1::from_vec(x2));

    let mut formula = Formula::new();
    formula.add_terms(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x1".to_string(),
            },
            Term::Linear {
                col_name: "x2".to_string(),
            },
        ],
    );

    let model = GamlssModel::fit(&y, &data, &formula, &Binomial::new(n_trials)).unwrap();

    let mu_coeffs = &model.models["mu"].coefficients;
    assert_eq!(mu_coeffs.0.len(), 3, "Should have 3 coefficients");

    // Check that fitted values are valid probabilities
    let mu_fitted = &model.models["mu"].fitted_values;
    assert!(
        mu_fitted.iter().all(|&v| v > 0.0 && v < 1.0),
        "All fitted probabilities should be in (0, 1)"
    );
}
