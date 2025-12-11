use gamlss_rs::distributions::StudentT;
use gamlss_rs::{GamlssError, GamlssModel, Smooth, Term};
use polars::prelude::*;
use rand::Rng; // Needed for the .random_range trait
use std::collections::HashMap;

fn main() -> Result<(), GamlssError> {
    // Generate Synthetic Data

    let mut rng = rand::rng();
    let n = 200;

    // one issue is that it isn't numerically stable right now
    let x_vals: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1).collect();

    let y_vals: Vec<f64> = x_vals
        .iter()
        .map(|&x| {
            let mu = x.sin();
            let sigma = 0.5 + 0.1 * x;

            // rand 0.9 syntax for range
            let noise: f64 = rng.random_range(-1.0..1.0);
            mu + sigma * noise
        })
        .collect();

    let df = df! {
        "x" => &x_vals,
        "y" => &y_vals,
    }?;

    // the formula hashmap
    let mut formulas = HashMap::new();

    // Mu: Smooth P-Spline
    formulas.insert(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Smooth(Smooth::PSpline1D {
                col_name: "x".to_string(),
                n_splines: 20,
                degree: 3,
                penalty_order: 2,
            }),
        ],
    );

    // Sigma: Linear
    formulas.insert(
        "sigma".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );

    // Nu: Constant
    formulas.insert("nu".to_string(), vec![Term::Intercept]);

    // Fit
    println!("Fitting GAMLSS model...");
    let model = GamlssModel::fit(&df, "y", &formulas, &StudentT::new())?;
    println!("Successfully Trained GAMLSS Model!");

    // 5. Inspect Results
    let mu_model = &model.models["mu"];
    let sigma_model = &model.models["sigma"];
    let nu_model = &model.models["nu"];

    println!("--- Results ---");
    println!("Mu coefficients count: {}", mu_model.coefficients.len());
    println!("Sigma coefficients: {:?}", sigma_model.coefficients);
    println!("Nu coefficients: {:?}", nu_model.coefficients);

    Ok(())
}
