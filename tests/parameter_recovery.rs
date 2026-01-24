use gamlss_rs::{distributions::StudentT, GamlssModel, Term};
use polars::prelude::*;
use rand::prelude::*;
use rand_distr::{Distribution, StudentT as StudentTDist};
use std::collections::HashMap; // We need the actual distribution generator

#[test]
fn test_student_t_recovery() {
    let n = 2_000;
    let mut rng = StdRng::seed_from_u64(42);

    // expected
    // log(sigma) = -0.7 + 0.0 * x  => sigma = exp(-0.7) approx 0.5
    // log(nu)    =  1.6 + 0.0 * x  => nu = exp(1.6) approx 4.95
    let true_mu_intercept: f64 = 10.0;
    let true_mu_slope: f64 = 2.0;

    let true_sigma_log: f64 = -0.7; // exp(-0.7) ~ 0.496
    let true_nu_log: f64 = 1.6; // exp(1.6) ~ 4.953

    // actuals
    let t_dist = StudentTDist::new(true_nu_log.exp()).unwrap();

    let x_vals: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * 10.0).collect();
    let y_vals: Vec<f64> = x_vals
        .iter()
        .map(|&x| {
            let mu = true_mu_intercept + true_mu_slope * x;
            let sigma = true_sigma_log.exp();

            // this is actually students t noise
            let noise = t_dist.sample(&mut rng);
            mu + sigma * noise
        })
        .collect();

    let df = df! {
        "x" => &x_vals,
        "y" => &y_vals,
    }
    .unwrap();

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
    formulas.insert("nu".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&df, "y", &formulas, &StudentT::new()).expect("Fit failed");

    // assertions
    let mu_coeffs = &model.models["mu"].coefficients;
    let sigma_coeffs = &model.models["sigma"].coefficients;
    let nu_coeffs = &model.models["nu"].coefficients;

    println!("Fitted Mu: {:?}", mu_coeffs);
    println!("Fitted Sigma: {:?}", sigma_coeffs);
    println!("Fitted Nu: {:?}", nu_coeffs);

    let tolerance = 0.2;

    // check params (mu is linear)
    assert!(
        (mu_coeffs[0] - true_mu_intercept).abs() < tolerance,
        "Mu Intercept failed"
    );
    assert!(
        (mu_coeffs[1] - true_mu_slope).abs() < tolerance,
        "Mu Slope failed"
    );

    // check sigma (Log Link)
    assert!(
        (sigma_coeffs[0] - true_sigma_log).abs() < tolerance,
        "Sigma Intercept failed"
    );

    // check nu (Log Link)
    // nu gets more slack because it's pretty hard to estimate
    assert!(
        (nu_coeffs[0] - true_nu_log).abs() < 0.7,
        "Nu Intercept failed"
    );
}
