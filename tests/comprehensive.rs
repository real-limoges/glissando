mod common;

use common::Generator;
use gamlss_rs::{
    distributions::{Gaussian, Poisson},
    GamlssModel, Smooth, Term,
};
use polars::prelude::{PlSmallStr, UInt32Chunked};
use rand::prelude::SliceRandom;
use std::collections::HashMap;

#[test]
fn test_poisson_recovery() {
    let mut rand_gen = Generator::new(123);
    let (true_int, true_slope) = (1.5, 0.5);
    let df = rand_gen.poisson_data(1000, true_int, true_slope);

    let mut formulas = HashMap::new();
    formulas.insert(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );

    let model =
        GamlssModel::fit(&df, "y", &formulas, &Poisson::new()).expect("Poisson Fit Failed!");

    let coeffs = &model.models["mu"].coefficients;

    // Recovery assertions
    assert!(
        (coeffs[0] - true_int).abs() < 0.1,
        "Intercept recovery failed"
    );
    assert!(
        (coeffs[1] - true_slope).abs() < 0.1,
        "Slope recovery failed"
    );
}

#[test]
fn test_heteroskedastic_gaussian_recovery() {
    let mut rand_gen = Generator::new(456);
    let df = rand_gen.heteroskedastic_gaussian(2000);

    let mut formulas = HashMap::new();
    formulas.insert(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );
    formulas.insert(
        "sigma".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );

    let model =
        GamlssModel::fit(&df, "y", &formulas, &Gaussian::new()).expect("Gaussian Fit Failed!");

    let mu = &model.models["mu"].coefficients;
    let sigma = &model.models["sigma"].coefficients;

    // Mu recovery (Truth: 10.0, 2.0)
    assert!((mu[0] - 10.0).abs() < 0.15);
    assert!((mu[1] - 2.0).abs() < 0.15);

    // Sigma recovery (Truth: -1.0, 0.5)
    assert!((sigma[0] - (-1.0)).abs() < 0.2);
    assert!((sigma[1] - 0.5).abs() < 0.2);
}

#[test]
fn test_tensor_product_complexity() {
    let mut rand_gen = Generator::new(123);
    let df = rand_gen.tensor_surface(400);

    let mut formulas = HashMap::new();
    formulas.insert(
        "mu".to_string(),
        vec![Term::Smooth(Smooth::TensorProduct {
            col_name_1: "x1".to_string(),
            n_splines_1: 5,
            penalty_order_1: 2,
            col_name_2: "x2".to_string(),
            n_splines_2: 5,
            penalty_order_2: 2,
            degree: 3,
        })],
    );
    formulas.insert("sigma".to_string(), vec![Term::Intercept]);

    let model =
        GamlssModel::fit(&df, "y", &formulas, &Gaussian::new()).expect("Tensor Fit Failed!");

    let edf = model.models["mu"].edf;

    // check that smoothing actually happened
    // should neither be a flat plane (EDF ~3) nor unpenalized (EDF 25)

    assert!(edf > 4.0, "Model is over-smoothed (EDF: {})", edf);
    assert!(edf < 20.0, "Model is under-smoothed (EDF: {})", edf);
}

#[test]
fn test_model_convergence_invariants() {
    let mut rand_gen = Generator::new(42);
    let df = rand_gen.linear_gaussian(500, 1.0, 5.0, 1.0);

    let mut formulas = HashMap::new();
    formulas.insert(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );
    formulas.insert("sigma".to_string(), vec![Term::Intercept]);

    let model_1 = GamlssModel::fit(&df, "y", &formulas, &Gaussian::new()).unwrap();

    // Use sample_n with the number of rows to effectively shuffle the whole set.
    let n = df.height();
    let mut indices: Vec<u32> = (0..n as u32).collect();
    indices.shuffle(&mut rand_gen.rng); // Use your common generator's rng

    let idx_ca = UInt32Chunked::from_vec(PlSmallStr::from_static("idx"), indices);
    let df_shuffled = df.take(&idx_ca).unwrap();

    let model_2 = GamlssModel::fit(&df_shuffled, "y", &formulas, &Gaussian::new()).unwrap();

    // verify coefficients are identical regardless of row order
    let b1 = &model_1.models["mu"].coefficients;
    let b2 = &model_2.models["mu"].coefficients;

    assert!(
        (b1[0] - b2[0]).abs() < 1e-6,
        "Intercept shifted after shuffle"
    );
    assert!((b1[1] - b2[1]).abs() < 1e-6, "Slope shifted after shuffle");
}

#[test]
fn test_spline_partition_of_unity() {
    let mut rand_gen = Generator::new(42);

    let df = rand_gen.linear_gaussian(100, 1.0, 0.0, 1.0);

    let n_splines = 10;
    let term = Term::Smooth(Smooth::PSpline1D {
        col_name: "x".to_string(),
        n_splines,
        degree: 3,
        penalty_order: 2,
    });

    let (mm, _, _) =
        gamlss_rs::fitting::assembler::assemble_model_matrices(&df, 100, &[term]).unwrap();

    // Check each row of the spline basis part of the ModelMatrix. each row sums to 1-ish
    for row in mm.0.rows() {
        let row_sum: f64 = row.sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-10,
            "Spline basis does not sum to 1.0 at a point!"
        );
    }
}
