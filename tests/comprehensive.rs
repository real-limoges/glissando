use gamlss_rs::{
    GamlssModel,
    Smooth,
    Term,
    distributions::{Gaussian},
};
use polars::prelude::*;
use std::collections::HashMap;
use ndarray::Array1;
use rand::prelude::*;
use rand_distr::{
    Distribution,
    Poisson as PoissonDist,
    Normal
};

fn create_df(x: &[f64], y: &[f64]) -> DataFrame {
    df! {
        "x" => x,
        "y" => y,
    }.unwrap()
}

// Test Poisson (Counts)
#[test]
fn test_poisson_recovery() {
    let n = 1000;
    let mut rng = StdRng::seed_from_u64(123);

}

// Test Heteroskedastic Gaussian
#[test]
fn test_heteroskedastic_gaussian() {
    let n = 2000;
    let mut rng = StdRng::seed_from_u64(123);
}

// P Splines
#[test]
fn test_p_spline_smoothing() {
    let n = 200;
    let mut rng = StdRng::seed_from_u64(123);
}

// Test Kroeneker Product
#[test]
fn test_tensor_product() {
    let n = 400;
    let mut rng = StdRng::seed_from_u64(123);

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
        Term::Intercept,
        Term::Smooth(Smooth::TensorProduct {
            col_name_1: "x1".to_string(),
            n_splines_1: 1,
            penalty_order_1: 2,
            col_name_2: "x2".to_string(),
            n_splines_2: 5,
            penalty_order_2: 2,
            degree: 3
        })
    ]);
    formulas.insert("sigma".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&df, "y", &formulas, &Gaussian::new()).expect("Fit Failed!");

    assert!(model.models["mu"].edf > 5.0);
    assert!(model.models["mu"].edf < 24.0);
}