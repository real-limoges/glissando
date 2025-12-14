use gamlss_rs::{
    GamlssModel,
    Smooth,
    Term,
    distributions::{Gaussian, Poisson},
};
use polars::prelude::*;
use std::collections::HashMap;
use rand::prelude::*;
use rand_distr::{
    Distribution,
    Poisson as PoissonDist,
    Normal
};

// Test Poisson (Counts)
#[test]
fn test_poisson_recovery() {
    let n = 1000;
    let mut rng = StdRng::seed_from_u64(123);

    let true_intercept: f64 = 1.5;
    let true_slope: f64 = 0.5;

    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * 4.0).collect();
    let y: Vec<f64> = x.iter().map(|&x| {
        let mu = (true_intercept + true_slope * x).exp();
        let dist = PoissonDist::new(mu).unwrap();
        dist.sample(&mut rng)
    }).collect();

    let df = df! (
        "x" => x,
        "y" => y,
    ).unwrap();

    let mut formulas = HashMap::new();
    formulas.insert("mu".to_string(), vec![
        Term::Intercept,
        Term::Linear { col_name: "x".to_string() }
    ]);

    let model = GamlssModel::fit(&df, "y", &formulas, &Poisson::new()).expect("Poisson Fit Failed!");

    let coeffs = &model.models["mu"].coefficients;

    println!("Poisson Coeffs: {:?}", coeffs);

    assert!((coeffs[0] - true_intercept).abs() < 0.1, "Poisson Intercept Failed!");
    assert!((coeffs[1] - true_slope).abs() < 0.1, "Poisson Slope Failed!");
}

// Test Heteroskedastic Gaussian
#[test]
fn test_heteroskedastic_gaussian() {
    let n = 2000;
    let mut rng = StdRng::seed_from_u64(456);

    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * 3.0).collect();
    let y: Vec<f64> = x.iter().map(|&x| {
        let mu = 10.0 + 2.0 * x;
        let sigma = (-1.0 + 0.5 * x).exp();
        let dist = Normal::new(mu, sigma).unwrap();
        dist.sample(&mut rng)
    }).collect();

    let df = df! (
        "x" => x,
        "y" => y,
    ).unwrap();

    let mut formulas = HashMap::new();
    formulas.insert("mu".to_string(),vec![
        Term::Intercept,
        Term::Linear { col_name: "x".to_string() }
    ]);
    formulas.insert("sigma".to_string(), vec![
        Term::Intercept,
        Term::Linear { col_name: "x".to_string() }
    ]);

    let model = GamlssModel::fit(&df, "y", &formulas, &Gaussian::new()).expect("Gaussian Fit Failed!");

    let mu_coeffs = &model.models["mu"].coefficients;
    let sigma_coeffs = &model.models["sigma"].coefficients;

    println!("Mu Coeffs: {:?}", mu_coeffs);
    println!("Sigma Coeffs: {:?}", sigma_coeffs);

    assert!((mu_coeffs[0] - 10.0).abs() < 0.1);
    assert!((mu_coeffs[1] - 2.0).abs() < 0.1);

    assert!((sigma_coeffs[0] - (-1.0)).abs() < 0.15);
    assert!((sigma_coeffs[1] - 0.5).abs() < 0.15);
}

// P Splines
// #[test]
// fn test_p_spline_smoothing() {
//     let n = 200;
//     let mut rng = StdRng::seed_from_u64(123);
// }

// Test Kroeneker Product
#[test]
fn test_tensor_product() {
    let n = 400;
    let mut rng = StdRng::seed_from_u64(123);

    // two independent variables so we can test the product
    let mut x1 = Vec::new();
    let mut x2 = Vec::new();
    let mut y = Vec::new();

    for _ in 0..n {
        let v1 = rng.random::<f64>();
        let v2 = rng.random::<f64>();

        let dist_sq = (v1 - 0.5).powi(2) + (v2 - 0.5).powi(2);
        let mu = (-dist_sq * 5.0).exp();

        let noise = rng.random_range(-0.1..0.1);

        x1.push(v1);
        x2.push(v2);
        y.push(mu + noise);
    }

    let df = df! (
        "x1" => x1,
        "x2" => x2,
        "y" => y,
    ).unwrap();

    let mut formulas = HashMap::new();
    formulas.insert("mu".to_string(), vec![
        Term::Smooth(Smooth::TensorProduct {
            col_name_1: "x1".to_string(),
            n_splines_1: 5,
            penalty_order_1: 2,
            col_name_2: "x2".to_string(),
            n_splines_2: 5,
            penalty_order_2: 2,
            degree: 3
        })
    ]);
    formulas.insert("sigma".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&df, "y", &formulas, &Gaussian::new()).expect("Fit Failed!");

    println!("{:?}", model.models["mu"].edf);

    assert!(model.models["mu"].edf > 5.0);
    assert!(model.models["mu"].edf < 24.0);
}