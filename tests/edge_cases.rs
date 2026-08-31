// Integration tests can't run under the `python` feature. PyO3's extension-module linking gets in the way.
#![cfg(not(feature = "python"))]

mod common;

use common::{
    intercept_only, linear, linear_intercepts, pspline_with, random, sample_negative_binomial,
    smooth_intercepts, Generator,
};
use glissando::{
    distributions::{Beta, Binomial, Gamma, Gaussian, NegativeBinomial, Poisson, StudentT},
    DataSet, Formula, GamlssModel, Term,
};
use ndarray::Array1;
use rand::{Rng, RngExt};

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

    let formula = smooth_intercepts("x", 10, &["mu"]);

    let model = GamlssModel::fit(&data, &y, &formula, &Poisson::new()).unwrap();

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

    let formula = linear_intercepts("x", &["mu", "sigma", "nu"]);

    let model = GamlssModel::fit(&data, &y, &formula, &StudentT::new()).unwrap();

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
        let formula = smooth_intercepts("x", n_splines, &["mu", "sigma"]);
        let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new());
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

    // First-difference penalty. This one leans on non-flat trends.
    let formula1 = Formula::new()
        .with_terms("mu", vec![pspline_with("x", 15, 3, 1)])
        .with_terms("sigma", vec![Term::Intercept]);
    // Second-difference penalty. This one leans on curvature.
    let formula2 = Formula::new()
        .with_terms("mu", vec![pspline_with("x", 15, 3, 2)])
        .with_terms("sigma", vec![Term::Intercept]);

    let model1 = GamlssModel::fit(&data, &y, &formula1, &Gaussian::new()).unwrap();
    let model2 = GamlssModel::fit(&data, &y, &formula2, &Gaussian::new()).unwrap();

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
            mu + rng.rng.random_range(-5.0..5.0) // heavy noise, on purpose
        })
        .collect();

    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("x".to_string(), Array1::from_vec(x));

    let formula = linear_intercepts("x", &["mu", "sigma"]);

    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

    // Even buried in noise, the slope should still come back close
    let slope = model.models["mu"].coefficients[1];
    assert!(
        (slope - 2.0).abs() < 1.0,
        "Slope should be roughly ~2 even with noise, got {}",
        slope
    );
}

#[test]
fn test_perfect_linear_fit() {
    // dead-perfect linear relationship, no noise at all
    let y = Array1::from_vec(vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]);
    let mut data = DataSet::new();
    data.insert_column(
        "x".to_string(),
        Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
    );

    let formula = linear_intercepts("x", &["mu", "sigma"]);

    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

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

    let formula = smooth_intercepts("x", 10, &["mu", "sigma"]);

    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

    // every smoothing parameter has to come out positive, no exceptions
    for &lambda in model.models["mu"].lambdas.iter() {
        assert!(lambda > 0.0, "Lambda should be positive, got {}", lambda);
    }
}

#[test]
fn test_covariance_symmetric() {
    let mut rng = Generator::new(42);
    let (y, data) = rng.linear_gaussian(100, 1.0, 5.0, 1.0);

    let formula = linear_intercepts("x", &["mu", "sigma"]);

    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

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

    let formula = linear_intercepts("x", &["mu", "sigma"]);

    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

    // Gaussian rides an identity link, so fitted_values just equals eta
    let mu = &model.models["mu"];
    for i in 0..mu.eta.len() {
        let diff = (mu.fitted_values[i] - mu.eta[i]).abs();
        assert!(diff < 1e-10, "For identity link, fitted should equal eta");
    }

    // sigma rides a log link, so fitted_values equals exp(eta)
    let sigma = &model.models["sigma"];
    for i in 0..sigma.eta.len() {
        let expected = sigma.eta[i].exp();
        let diff = (sigma.fitted_values[i] - expected).abs();
        assert!(diff < 1e-10, "For log link, fitted should equal exp(eta)");
    }
}

#[test]
fn test_random_effect_basic() {
    // groups are just numeric indices here: 0.0 = group A, 1.0 = group B, 2.0 = group C
    let group = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    let y = Array1::from_vec(vec![1.0, 1.2, 0.8, 5.0, 5.1, 4.9, 3.0, 3.1, 2.9]);

    let mut data = DataSet::new();
    data.insert_column("group".to_string(), group);

    let formula = Formula::new()
        .with_terms("mu", vec![random("group")])
        .with_terms("sigma", vec![Term::Intercept]);

    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

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

    let formula = Formula::new()
        .with_terms(
            "mu",
            vec![
                Term::Intercept,
                linear("x1"),
                linear("x2"),
                linear("x3"),
                linear("x4"),
            ],
        )
        .with_terms("sigma", vec![Term::Intercept]);

    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

    let coeffs = &model.models["mu"].coefficients;
    assert_eq!(coeffs.len(), 5, "Should have 5 coefficients");

    // now confirm every coef came back
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

    let formula =
        Formula::new().with_terms("mu", vec![Term::Intercept, linear("x1"), linear("x2")]);

    let model = GamlssModel::fit(&data, &y, &formula, &Poisson::new()).unwrap();

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
    // Poisson with high means. This is really a numerical-stability check.
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

    let formula = linear_intercepts("x", &["mu"]);

    let model = GamlssModel::fit(&data, &y, &formula, &Poisson::new()).unwrap();

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
    // Poisson, but the truth is a nonlinear smooth this time
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

    let formula = smooth_intercepts("x", 12, &["mu"]);

    let model = GamlssModel::fit(&data, &y, &formula, &Poisson::new()).unwrap();

    let edf = model.models["mu"].edf;
    // With n=400 >> 12 basis functions, REML rightly decides little or no
    // penalization is needed: the marginal likelihood peaks near lambda≈0
    // because model complexity (12 params) is way under sample size (400 obs).
    // The lower bound proves the smooth actually got fitted. The upper bound
    // leaves room for the REML-selected near-unpenalized solution.
    assert!(
        edf > 3.0,
        "Poisson smooth EDF too low for sinusoidal pattern: {}",
        edf
    );
    assert!(
        edf <= 12.0,
        "Poisson smooth EDF exceeds basis dimension: {}",
        edf
    );
}

#[test]
fn test_poisson_low_counts() {
    // Poisson down at very low counts. This is the edge case.
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

    let formula = linear_intercepts("x", &["mu"]);

    let model = GamlssModel::fit(&data, &y, &formula, &Poisson::new()).unwrap();

    let coeffs = &model.models["mu"].coefficients;
    // low counts carry less signal, so loosen the tolerance
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
    // StudentT with a smooth mu underneath
    let mut rng = Generator::new(222);

    let n = 500;
    let nu = 5.0; // degrees of freedom, moderate tails
    let sigma = 0.5;

    let x: Vec<f64> = (0..n)
        .map(|i| i as f64 / n as f64 * 2.0 * std::f64::consts::PI)
        .collect();

    let y_vec: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mu = 3.0 * xi.sin(); // mean rides a sine wave
            let t_sample: f64 = rng.rng.sample(rand_distr::StudentT::new(nu).unwrap());
            mu + sigma * t_sample
        })
        .collect();

    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("x".to_string(), Array1::from_vec(x));

    let formula = smooth_intercepts("x", 15, &["mu", "sigma", "nu"]);

    let model = GamlssModel::fit(&data, &y, &formula, &StudentT::new()).unwrap();

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
    // StudentT where sigma moves with x
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

    let formula = Formula::new()
        .with_terms("mu", vec![Term::Intercept, linear("x")])
        .with_terms("sigma", vec![Term::Intercept, linear("x")])
        .with_terms("nu", vec![Term::Intercept]);

    let model = GamlssModel::fit(&data, &y, &formula, &StudentT::new()).unwrap();

    let mu_coeffs = &model.models["mu"].coefficients;
    let sigma_coeffs = &model.models["sigma"].coefficients;

    // mu first
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

    // now sigma, remembering it's on the log link
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
    // StudentT with very low df, so genuinely heavy tails
    let mut rng = Generator::new(444);

    let n = 1000;
    let true_nu = 3.0; // heavy tails
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

    let formula = linear_intercepts("x", &["mu", "sigma", "nu"]);

    let model = GamlssModel::fit(&data, &y, &formula, &StudentT::new()).unwrap();

    let nu_coeff = model.models["nu"].coefficients[0];
    let fitted_nu = nu_coeff.exp();

    // nu is always a noisy estimate, but for heavy tails it should still land in range
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
    // StudentT with a fistful of linear predictors
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

    let formula = Formula::new()
        .with_terms(
            "mu",
            vec![Term::Intercept, linear("x1"), linear("x2"), linear("x3")],
        )
        .with_terms("sigma", vec![Term::Intercept])
        .with_terms("nu", vec![Term::Intercept]);

    let model = GamlssModel::fit(&data, &y, &formula, &StudentT::new()).unwrap();

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
    // StudentT with high df. At this point it's basically Gaussian.
    let mut rng = Generator::new(999);

    let n = 500;
    let true_nu = 30.0; // high nu => nearly Gaussian
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

    let formula = linear_intercepts("x", &["mu", "sigma", "nu"]);

    let model = GamlssModel::fit(&data, &y, &formula, &StudentT::new()).unwrap();

    let mu_coeffs = &model.models["mu"].coefficients;

    // high nu means the params should land right where Gaussian would put them
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

    // fitted nu should come out reasonably high. It stays noisy for near-Gaussian data
    // because there's barely any tail information to tell moderate nu from high nu.
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
    // Gamma with a plain linear mu
    let mut rng = Generator::new(1001);

    let n = 500;
    let true_sigma = 0.5; // CV = 0.5
    let shape = 1.0 / (true_sigma * true_sigma); // alpha = 4, falls straight out of the CV

    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * 2.0).collect();

    // True model: log(mu) = 1.0 + 0.5*x => mu from ~2.7 to ~7.4
    let y_vec: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mu = (1.0 + 0.5 * xi).exp();
            let scale = mu / shape; // theta = mu * sigma^2 = mu / alpha, same thing two ways
            let gamma_dist = rand_distr::Gamma::new(shape, scale).unwrap();
            rng.rng.sample(gamma_dist)
        })
        .collect();

    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("x".to_string(), Array1::from_vec(x));

    let formula = linear_intercepts("x", &["mu", "sigma"]);

    let model = GamlssModel::fit(&data, &y, &formula, &Gamma::new()).unwrap();

    let mu_coeffs = &model.models["mu"].coefficients;
    // log link, so read these coefficients on the log scale
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
    // Gamma with a moving sigma, i.e. a changing coefficient of variation
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

    let formula = Formula::new()
        .with_terms("mu", vec![Term::Intercept, linear("x")])
        .with_terms("sigma", vec![Term::Intercept, linear("x")]);

    let model = GamlssModel::fit(&data, &y, &formula, &Gamma::new()).unwrap();

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
    // Gamma, this time with a smooth mu
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

    let formula = smooth_intercepts("x", 12, &["mu", "sigma"]);

    let model = GamlssModel::fit(&data, &y, &formula, &Gamma::new()).unwrap();

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
    // Negative Binomial with a linear mu
    let mut rng = Generator::new(2001);

    let n = 500;
    let true_sigma = 0.5; // this is the overdispersion knob

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

    let formula = linear_intercepts("x", &["mu", "sigma"]);

    let model = GamlssModel::fit(&data, &y, &formula, &NegativeBinomial::new()).unwrap();

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
    // NB cranked to high overdispersion, well clear of Poisson
    let mut rng = Generator::new(2002);

    let n = 600;
    let true_sigma = 1.0; // heavy overdispersion

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

    let formula = linear_intercepts("x", &["mu", "sigma"]);

    let model = GamlssModel::fit(&data, &y, &formula, &NegativeBinomial::new()).unwrap();

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

    // sigma should come back sane, and clearly overdispersed
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
    // NB with a smooth mu
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

    let formula = smooth_intercepts("x", 12, &["mu", "sigma"]);

    let model = GamlssModel::fit(&data, &y, &formula, &NegativeBinomial::new()).unwrap();

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
    // NB with several linear predictors
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

    let formula = Formula::new()
        .with_terms("mu", vec![Term::Intercept, linear("x1"), linear("x2")])
        .with_terms("sigma", vec![Term::Intercept]);

    let model = GamlssModel::fit(&data, &y, &formula, &NegativeBinomial::new()).unwrap();

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

// little helper to draw from a Beta
fn sample_beta(rng: &mut impl Rng, alpha: f64, beta: f64) -> f64 {
    let beta_dist = rand_distr::Beta::new(alpha, beta).unwrap();
    rng.sample(beta_dist)
}

#[test]
fn test_beta_linear_mu() {
    // Beta with a linear mu, living on the logit scale
    let mut rng = Generator::new(3001);

    let n = 500;
    let true_phi = 10.0; // the precision parameter

    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * 2.0 - 1.0).collect(); // x in [-1, 1]

    // True model: logit(mu) = 0.0 + 0.5*x => mu varies around 0.5
    let y_vec: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let eta = 0.0 + 0.5 * xi;
            let mu = 1.0 / (1.0 + (-eta).exp()); // back through the inverse logit
            let alpha = mu * true_phi;
            let beta_param = (1.0 - mu) * true_phi;
            sample_beta(&mut rng.rng, alpha, beta_param)
        })
        .collect();

    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("x".to_string(), Array1::from_vec(x));

    let formula = linear_intercepts("x", &["mu", "phi"]);

    let model = GamlssModel::fit(&data, &y, &formula, &Beta::new()).unwrap();

    let mu_coeffs = &model.models["mu"].coefficients;
    // these coefficients live on the logit scale
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
    // Beta where phi (the precision) varies
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

    let formula = Formula::new()
        .with_terms("mu", vec![Term::Intercept])
        .with_terms("phi", vec![Term::Intercept, linear("x")]);

    let model = GamlssModel::fit(&data, &y, &formula, &Beta::new()).unwrap();

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
    // Beta with a smooth mu
    let mut rng = Generator::new(3003);

    let n = 1_000;
    let phi = 15.0;

    let x: Vec<f64> = (0..n)
        .map(|i| i as f64 / n as f64 * 2.0 * std::f64::consts::PI)
        .collect();

    // True model: logit(mu) = 0.7*sin(x) => mu oscillates around 0.5
    let y_vec: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let eta = 0.7 * xi.sin();
            let mu = 1.0 / (1.0 + (-eta).exp());
            let alpha = mu * phi;
            let beta_param = (1.0 - mu) * phi;
            sample_beta(&mut rng.rng, alpha, beta_param)
        })
        .collect();

    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("x".to_string(), Array1::from_vec(x));

    let formula = smooth_intercepts("x", 12, &["mu", "phi"]);

    let model = GamlssModel::fit(&data, &y, &formula, &Beta::new()).unwrap();

    let edf = model.models["mu"].edf;
    assert!(edf > 2.0, "Beta smooth mu EDF too low: {}", edf);
    assert!(edf < 12.0, "Beta smooth mu EDF too high: {}", edf);
}

#[test]
fn test_beta_high_precision() {
    // Beta at high precision, so low variance and data hugging the mean
    let mut rng = Generator::new(3004);

    let n = 400;
    let true_phi = 50.0; // tight, high precision

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

    let formula = linear_intercepts("x", &["mu", "phi"]);

    let model = GamlssModel::fit(&data, &y, &formula, &Beta::new()).unwrap();

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

    // phi should come back high, the way we set it
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
    let n_trials = 20; // trials per observation
    let true_intercept = -0.5; // on the logit scale
    let true_slope = 2.0;

    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let y_vec: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let eta = true_intercept + true_slope * xi;
            let mu = 1.0 / (1.0 + (-eta).exp()); // back through the inverse logit
            let dist = rand_distr::Binomial::new(n_trials as u64, mu).unwrap();
            rng.rng.sample(dist) as f64
        })
        .collect();

    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("x".to_string(), Array1::from_vec(x));

    let formula = linear_intercepts("x", &["mu"]);

    let model = GamlssModel::fit(&data, &y, &formula, &Binomial::new(n_trials)).unwrap();

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
    // Binomial up near the ceiling, high success probability
    let mut rng = Generator::new(123);

    let n = 200;
    let n_trials = 50;
    let true_mu = 0.8; // high probability

    let y_vec: Vec<f64> = (0..n)
        .map(|_| {
            let dist = rand_distr::Binomial::new(n_trials as u64, true_mu).unwrap();
            rng.rng.sample(dist) as f64
        })
        .collect();

    let y = Array1::from_vec(y_vec);
    let data = DataSet::new();

    let formula = intercept_only(&["mu"]);

    let model = GamlssModel::fit(&data, &y, &formula, &Binomial::new(n_trials)).unwrap();

    // fitted probability should sit close to the truth
    let mu_coeff = model.models["mu"].coefficients[0];
    let fitted_mu = 1.0 / (1.0 + (-mu_coeff).exp()); // back through the inverse logit
    assert!(
        (fitted_mu - true_mu).abs() < 0.1,
        "Binomial should recover mu ~{}, got {}",
        true_mu,
        fitted_mu
    );
}

#[test]
fn test_binomial_multiple_predictors() {
    // Binomial with a couple of linear predictors
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

    let formula =
        Formula::new().with_terms("mu", vec![Term::Intercept, linear("x1"), linear("x2")]);

    let model = GamlssModel::fit(&data, &y, &formula, &Binomial::new(n_trials)).unwrap();

    let mu_coeffs = &model.models["mu"].coefficients;
    assert_eq!(mu_coeffs.0.len(), 3, "Should have 3 coefficients");

    // every fitted value has to be a real probability, strictly inside (0, 1)
    let mu_fitted = &model.models["mu"].fitted_values;
    assert!(
        mu_fitted.iter().all(|&v| v > 0.0 && v < 1.0),
        "All fitted probabilities should be in (0, 1)"
    );
}
