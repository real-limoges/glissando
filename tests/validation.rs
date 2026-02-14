// Integration tests cannot run with the `python` feature due to PyO3's extension-module linking
#![cfg(not(feature = "python"))]

mod common;

use common::Generator;
use gamlss_rs::{
    distributions::{Gaussian, Poisson},
    DataSet, Formula, GamlssError, GamlssModel, Term,
};
use ndarray::Array1;
use rand::RngExt;

#[test]
fn test_missing_variable_in_data() {
    let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let data = DataSet::new(); // empty data, no "x" column

    let mut formulas = Formula::new();
    formulas.add_terms(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );

    let result = GamlssModel::fit(&data, &y, &formulas, &Poisson::new());

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(err, GamlssError::MissingVariable { .. }),
        "Expected MissingVariable error for missing predictor column, got {:?}",
        err
    );
}

#[test]
fn test_missing_formula_for_parameter() {
    let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_vec(vec![1.0, 2.0, 3.0]));

    let mut formulas = Formula::new();
    formulas.add_terms("mu", vec![Term::Intercept]);
    // Missing "sigma" formula for Gaussian

    let result = GamlssModel::fit(&data, &y, &formulas, &Gaussian::new());

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(err, GamlssError::MissingFormula { .. }),
        "Expected MissingFormula error for missing formula, got {:?}",
        err
    );
}

#[test]
fn test_small_dataset() {
    let y = Array1::from_vec(vec![2.1, 4.0, 5.9, 8.1, 10.0]);
    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]));

    let mut formulas = Formula::new();
    formulas.add_terms(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );
    formulas.add_terms("sigma", vec![Term::Intercept]);

    let model = GamlssModel::fit(&data, &y, &formulas, &Gaussian::new()).unwrap();

    let mu_coeffs = &model.models["mu"].coefficients;
    // Should recover approximately y = 2x (intercept ~0, slope ~2)
    assert!(
        mu_coeffs[1] > 1.5 && mu_coeffs[1] < 2.5,
        "Slope should be ~2, got {}",
        mu_coeffs[1]
    );
}

#[test]
fn test_intercept_only_model() {
    let mut rand_gen = Generator::new(999);
    let (y, data) = rand_gen.linear_gaussian(200, 0.0, 10.0, 1.0); // slope=0, intercept=10

    let mut formulas = Formula::new();
    formulas.add_terms("mu".to_string(), vec![Term::Intercept]);
    formulas.add_terms("sigma".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&data, &y, &formulas, &Gaussian::new()).unwrap();

    let mu_intercept = model.models["mu"].coefficients[0];
    assert!(
        (mu_intercept - 10.0).abs() < 0.5,
        "Intercept should be ~10, got {}",
        mu_intercept
    );
}

#[test]
fn test_large_coefficients() {
    // Test that the model can handle data with large scale
    let y = Array1::from_vec(vec![
        1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0,
    ]);
    let mut data = DataSet::new();
    data.insert_column(
        "x".to_string(),
        Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
    );

    let mut formulas = Formula::new();
    formulas.add_terms(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );
    formulas.add_terms("sigma".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&data, &y, &formulas, &Gaussian::new()).unwrap();

    let mu_coeffs = &model.models["mu"].coefficients;
    // Should recover y = 1000 + 100*x
    assert!(
        (mu_coeffs[0] - 1000.0).abs() < 10.0,
        "Intercept should be ~1000, got {}",
        mu_coeffs[0]
    );
    assert!(
        (mu_coeffs[1] - 100.0).abs() < 5.0,
        "Slope should be ~100, got {}",
        mu_coeffs[1]
    );
}

#[test]
fn test_negative_response_gaussian() {
    let y = Array1::from_vec(vec![-10.0, -8.0, -6.0, -4.0, -2.0]);
    let mut data = DataSet::new();
    data.insert_column(
        "x".to_string(),
        Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
    );

    let mut formulas = Formula::new();
    formulas.add_terms(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );
    formulas.add_terms("sigma".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&data, &y, &formulas, &Gaussian::new()).unwrap();

    let mu_coeffs = &model.models["mu"].coefficients;
    // Should recover y = -12 + 2*x
    assert!(
        (mu_coeffs[1] - 2.0).abs() < 0.5,
        "Slope should be ~2, got {}",
        mu_coeffs[1]
    );
}

#[test]
fn test_multiple_linear_terms() {
    let mut rand_gen = Generator::new(42);

    let n = 500;
    // Use independent random predictors to avoid collinearity
    let x1: Vec<f64> = (0..n).map(|_| rand_gen.rng.random::<f64>()).collect();
    let x2: Vec<f64> = (0..n).map(|_| rand_gen.rng.random::<f64>()).collect();
    let y: Vec<f64> = x1
        .iter()
        .zip(x2.iter())
        .map(|(&a, &b)| {
            let mu = 1.0 + 2.0 * a + 3.0 * b;
            mu + rand_gen.rng.random_range(-0.1..0.1)
        })
        .collect();

    let y = Array1::from_vec(y);
    let mut data = DataSet::new();
    data.insert_column("x1".to_string(), Array1::from_vec(x1));
    data.insert_column("x2".to_string(), Array1::from_vec(x2));

    let mut formulas = Formula::new();
    formulas.add_terms(
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
    formulas.add_terms("sigma".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&data, &y, &formulas, &Gaussian::new()).unwrap();

    let mu_coeffs = &model.models["mu"].coefficients;
    // Should recover intercept ~1, x1 coef ~2, x2 coef ~3
    assert!(
        (mu_coeffs[0] - 1.0).abs() < 0.3,
        "Intercept should be ~1, got {}",
        mu_coeffs[0]
    );
    assert!(
        (mu_coeffs[1] - 2.0).abs() < 0.3,
        "x1 coef should be ~2, got {}",
        mu_coeffs[1]
    );
    assert!(
        (mu_coeffs[2] - 3.0).abs() < 0.3,
        "x2 coef should be ~3, got {}",
        mu_coeffs[2]
    );
}

#[test]
fn test_spline_smooth_recovery() {
    let mut rand_gen = Generator::new(123);

    let n = 300;
    let x: Vec<f64> = (0..n)
        .map(|i| i as f64 / n as f64 * 2.0 * std::f64::consts::PI)
        .collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mu = xi.sin();
            mu + rand_gen.rng.random_range(-0.1..0.1)
        })
        .collect();

    let y = Array1::from_vec(y);
    let mut data = DataSet::new();
    data.insert_column("x".to_string(), Array1::from_vec(x.clone()));

    let mut formulas = Formula::new();
    formulas.add_terms(
        "mu".to_string(),
        vec![Term::Smooth(gamlss_rs::Smooth::PSpline1D {
            col_name: "x".to_string(),
            n_splines: 15,
            degree: 3,
            penalty_order: 2,
        })],
    );
    formulas.add_terms("sigma".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&data, &y, &formulas, &Gaussian::new()).unwrap();

    // Check that fitted values roughly follow sin(x)
    let fitted = &model.models["mu"].fitted_values;
    let mut mse = 0.0;
    for (i, &xi) in x.iter().enumerate() {
        let true_val = xi.sin();
        mse += (fitted[i] - true_val).powi(2);
    }
    mse /= n as f64;

    assert!(
        mse < 0.05,
        "MSE should be small for smooth recovery, got {}",
        mse
    );
}

#[test]
fn test_edf_reasonable() {
    let mut rand_gen = Generator::new(42);
    let (y, data) = rand_gen.linear_gaussian(200, 1.0, 5.0, 1.0);

    let mut formulas = Formula::new();
    formulas.add_terms(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );
    formulas.add_terms("sigma".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&data, &y, &formulas, &Gaussian::new()).unwrap();

    // For intercept + linear, EDF should be ~2
    let mu_edf = model.models["mu"].edf;
    assert!(
        mu_edf > 1.5 && mu_edf < 2.5,
        "EDF for linear model should be ~2, got {}",
        mu_edf
    );

    // For intercept only, EDF should be ~1
    let sigma_edf = model.models["sigma"].edf;
    assert!(
        sigma_edf > 0.5 && sigma_edf < 1.5,
        "EDF for intercept-only should be ~1, got {}",
        sigma_edf
    );
}
