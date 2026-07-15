//! STRUCT-2 integration tests: a `Truncated` wrapper fits end-to-end, recovers
//! the parameters of a left-truncated Gaussian, and reduces to the base family
//! over the full range.

#![cfg(not(feature = "python"))]

use glissando::distributions::{Distribution, Gaussian, Truncated};
use glissando::{DataSet, Formula, GamlssModel, Term};
use ndarray::Array1;

fn intercept_only() -> Formula {
    Formula::new()
        .with_terms("mu", vec![Term::Intercept])
        .with_terms("sigma", vec![Term::Intercept])
}

fn dummy_data(n: usize) -> DataSet {
    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_iter((0..n).map(|i| i as f64)));
    data
}

/// Ideal order statistics of a Gaussian left-truncated at `lo`, i.e. samples from
/// `N(mu, sigma) | Y > lo`, built deterministically.
fn truncated_latent(mu: f64, sigma: f64, lo: f64, n: usize) -> Array1<f64> {
    let owned = [
        ("mu", Array1::from_elem(n, mu)),
        ("sigma", Array1::from_elem(n, sigma)),
    ];
    let view = owned.iter().map(|(k, v)| (*k, v)).collect();
    let f_lo = Gaussian
        .cdf(&Array1::from_elem(n, lo), &view)
        .unwrap()
        .to_vec();
    // p mapped into the truncated tail: F(lo) + u·(1 − F(lo)).
    let p = Array1::from_iter((0..n).map(|i| {
        let u = (i as f64 + 0.5) / n as f64;
        f_lo[i] + u * (1.0 - f_lo[i])
    }));
    Gaussian.quantile(&p, &view).unwrap()
}

#[test]
fn full_range_matches_plain_gaussian_fit() {
    let n = 80;
    let owned = [
        ("mu", Array1::from_elem(n, 3.0)),
        ("sigma", Array1::from_elem(n, 1.5)),
    ];
    let view = owned.iter().map(|(k, v)| (*k, v)).collect();
    let p = Array1::from_iter((0..n).map(|i| (i as f64 + 0.5) / n as f64));
    let y = Gaussian.quantile(&p, &view).unwrap();

    let data = dummy_data(n);
    let formula = intercept_only();
    let plain = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

    let lo = Array1::from_elem(n, f64::NEG_INFINITY);
    let hi = Array1::from_elem(n, f64::INFINITY);
    let trunc = Truncated::new(Box::new(Gaussian::new()), lo, hi);
    let trunc_fit = GamlssModel::fit(&data, &y, &formula, &trunc).unwrap();

    let plain_mu = plain.models["mu"].coefficients.0[0];
    let trunc_mu = trunc_fit.models["mu"].coefficients.0[0];
    assert!(
        (plain_mu - trunc_mu).abs() < 1e-6,
        "(−∞,∞) truncation mu {trunc_mu} should match plain {plain_mu}"
    );
}

#[test]
fn left_truncation_recovers_parameters() {
    // True N(2, 1.5) observed only above lo = 1.0.
    let n = 300;
    let (true_mu, true_sigma, lo) = (2.0, 1.5, 1.0);
    let y = truncated_latent(true_mu, true_sigma, lo, n);
    assert!(y.iter().all(|&v| v > lo), "all samples lie in the support");

    let data = dummy_data(n);
    let formula = intercept_only();
    let lower = Array1::from_elem(n, lo);
    let upper = Array1::from_elem(n, f64::INFINITY);
    let trunc = Truncated::new(Box::new(Gaussian::new()), lower, upper);
    let fit = GamlssModel::fit(&data, &y, &formula, &trunc).unwrap();

    let mu_hat = fit.models["mu"].coefficients.0[0];
    let sigma_hat = fit.models["sigma"].fitted_values[0];
    assert!(
        (mu_hat - true_mu).abs() < 0.4,
        "truncated MLE mu {mu_hat} should recover ≈ {true_mu}"
    );
    assert!(
        (sigma_hat - true_sigma).abs() < 0.4,
        "truncated MLE sigma {sigma_hat} should recover ≈ {true_sigma}"
    );
    // The truncated sample mean is biased high (the low tail is missing); the
    // wrapper corrects for it.
    let naive_mean = y.mean().unwrap();
    assert!(
        naive_mean > true_mu + 0.2,
        "naive mean {naive_mean} biased high"
    );
}
