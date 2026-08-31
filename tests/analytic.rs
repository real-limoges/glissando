// Integration tests can't run under the `python` feature. PyO3's extension-module linking gets in the way.
#![cfg(not(feature = "python"))]

//! Closed-form anchors. I want these to prove the iterative fitter recovers the
//! exact analytic solution on problems where one exists, independent of any
//! snapshot or regression comparison against past runs.

use glissando::{distributions::Gaussian, DataSet, Formula, GamlssModel, Term};
use ndarray::Array1;

/// Tight closed-form anchor. Simple linear regression has an analytic least-squares
/// solution, so with `Gaussian + Linear + Intercept` on data with a known true linear
/// relationship plus tiny jitter (the jitter keeps sigma off 0, which would destabilize
/// the IRLS loop), the fitter has to recover `(α, β)` to high precision.
#[test]
fn gaussian_linear_recovers_ols_to_floating_point() {
    let n = 50;
    let true_alpha = 2.5;
    let true_beta = -1.75;

    let x: Array1<f64> = Array1::from_iter((0..n).map(|i| i as f64 / 10.0));
    // Pseudo-random jitter, deterministic off the index, just to keep sigma > 0.
    let y: Array1<f64> = (0..n)
        .map(|i| {
            let jitter = ((i as f64 * 0.731).sin()) * 1e-3;
            true_alpha + true_beta * x[i] + jitter
        })
        .collect();

    let mut data = DataSet::new();
    data.insert_column("x", x.clone());

    let formula = Formula::new()
        .with_terms(
            "mu",
            vec![
                Term::Intercept,
                Term::Linear {
                    col_name: "x".into(),
                },
            ],
        )
        .with_terms("sigma", vec![Term::Intercept]);

    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

    // Compute the analytic OLS solution on the *same* y (jitter and all) and check
    // the fitter matches it. That match is the real closed-form anchor, and it
    // holds regardless of whether the RS outer loop's eps-criterion flagged "converged".
    let x_bar = x.mean().unwrap();
    let y_bar = y.mean().unwrap();
    let sxx: f64 = x.iter().map(|xi| (xi - x_bar).powi(2)).sum();
    let sxy: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - x_bar) * (yi - y_bar))
        .sum();
    let beta_ols = sxy / sxx;
    let alpha_ols = y_bar - beta_ols * x_bar;

    let mu_coefs = &model.models["mu"].coefficients;
    assert!(
        (mu_coefs[0] - alpha_ols).abs() < 1e-6,
        "intercept: OLS {:.8}, fit {:.8}",
        alpha_ols,
        mu_coefs[0]
    );
    assert!(
        (mu_coefs[1] - beta_ols).abs() < 1e-6,
        "slope: OLS {:.8}, fit {:.8}",
        beta_ols,
        mu_coefs[1]
    );
}

/// Intercept-only Gaussian is just the sample mean. Exact closed form, no wiggle room.
#[test]
fn gaussian_intercept_only_recovers_sample_mean() {
    let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let y_bar: f64 = y.mean().unwrap();

    let data = DataSet::new();
    let formula = Formula::new()
        .with_terms("mu", vec![Term::Intercept])
        .with_terms("sigma", vec![Term::Intercept]);

    // intercept-only models fit fine with no data columns. The dummy column is
    // only here so n_obs detection has something to count.
    let mut data = data;
    data.insert_column("_unused", Array1::ones(y.len()));

    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();
    let mu_intercept = model.models["mu"].coefficients[0];
    assert!(
        (mu_intercept - y_bar).abs() < 1e-6,
        "intercept should equal ȳ = {:.6}, got {:.6}",
        y_bar,
        mu_intercept
    );
}
