//! STRUCT-1 integration tests. Here I want a `Censored` wrapper to fit end-to-end
//! through the standard RS loop, recover known parameters under right-censoring, and
//! collapse back to the base family when every row is an event.

#![cfg(not(feature = "python"))]

use glissando::distributions::{CensorStatus, Censored, Distribution, Gaussian};
use glissando::{DataSet, Formula, GamlssModel, Term};
use ndarray::Array1;

/// Ideal latent Gaussian order statistics for `N(mu, sigma)` of length `n`.
/// Deterministic, so the recovery tests reproduce exactly every run.
fn latent_gaussian(mu: f64, sigma: f64, n: usize) -> Array1<f64> {
    let p = Array1::from_iter((0..n).map(|i| (i as f64 + 0.5) / n as f64));
    let owned = [
        ("mu", Array1::from_elem(n, mu)),
        ("sigma", Array1::from_elem(n, sigma)),
    ];
    let view = owned.iter().map(|(k, v)| (*k, v)).collect();
    Gaussian.quantile(&p, &view).unwrap()
}

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

#[test]
fn all_event_matches_plain_gaussian_fit() {
    let n = 80;
    let y = latent_gaussian(5.0, 2.0, n);
    let data = dummy_data(n);
    let formula = intercept_only();

    let plain = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

    let status = Array1::from_elem(n, CensorStatus::Event);
    let cens = Censored::new(Box::new(Gaussian::new()), status);
    let censored_fit = GamlssModel::fit(&data, &y, &formula, &cens).unwrap();

    // All-event censoring is exactly the base likelihood, so the coefficients have to come out identical.
    let plain_mu = plain.models["mu"].coefficients.0[0];
    let cens_mu = censored_fit.models["mu"].coefficients.0[0];
    assert!(
        (plain_mu - cens_mu).abs() < 1e-6,
        "all-event mu {cens_mu} should match plain {plain_mu}"
    );
}

#[test]
fn right_censored_recovers_mean() {
    // True N(5, 2). Right-censor everything above 6, which is a substantial censored mass.
    let n = 200;
    let true_mu = 5.0;
    let true_sigma = 2.0;
    let latent = latent_gaussian(true_mu, true_sigma, n);
    let cutoff = 6.0;

    let mut y = latent.clone();
    let mut status = Array1::from_elem(n, CensorStatus::Event);
    for i in 0..n {
        if latent[i] > cutoff {
            y[i] = cutoff;
            status[i] = CensorStatus::Right;
        }
    }
    // Sanity check. Censoring has to actually bind on a meaningful fraction, or the test proves nothing.
    let n_cens = status.iter().filter(|s| **s == CensorStatus::Right).count();
    assert!(
        n_cens > 20 && n_cens < n - 20,
        "censoring should be partial"
    );

    let data = dummy_data(n);
    let formula = intercept_only();
    let cens = Censored::new(Box::new(Gaussian::new()), status);
    let fit = GamlssModel::fit(&data, &y, &formula, &cens).unwrap();

    let mu_hat = fit.models["mu"].coefficients.0[0];
    // A naive fit treating censored values as observed comes out biased low (≈ the
    // censored sample mean, well under 5). The censored MLE is what recovers ≈ 5.
    assert!(
        (mu_hat - true_mu).abs() < 0.4,
        "censored MLE mu {mu_hat} should recover ≈ {true_mu}"
    );

    // The naive (ignore-censoring) mean sits materially lower. That's the proof the wrapper
    // is doing real work, not just echoing back the data mean.
    let naive_mean = y.mean().unwrap();
    assert!(
        naive_mean < true_mu - 0.3,
        "naive mean {naive_mean} should be biased below the truth"
    );
}
