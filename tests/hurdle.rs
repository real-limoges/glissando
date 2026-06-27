//! STRUCT-3 integration tests: a `Hurdle` wrapper fits end-to-end, recovering
//! both the zero-atom probability and the positive-part parameters.

#![cfg(not(feature = "python"))]

use glissando::distributions::{Distribution, Gamma, Hurdle};
use glissando::{DataSet, Formula, GamlssModel, Term};
use ndarray::Array1;

fn dummy_data(n: usize) -> DataSet {
    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_iter((0..n).map(|i| i as f64)));
    data
}

/// Ideal Gamma order statistics for `(mu, sigma)` of length `m`.
fn gamma_latent(mu: f64, sigma: f64, m: usize) -> Array1<f64> {
    let p = Array1::from_iter((0..m).map(|i| (i as f64 + 0.5) / m as f64));
    let owned = [
        ("mu", Array1::from_elem(m, mu)),
        ("sigma", Array1::from_elem(m, sigma)),
    ];
    let view = owned.iter().map(|(k, v)| (*k, v)).collect();
    Gamma.quantile(&p, &view).unwrap()
}

#[test]
fn hurdle_recovers_zero_fraction_and_positive_mean() {
    // 40% structural zeros, 60% positive Gamma(mu=3, sigma=0.5).
    let n_zero = 120;
    let n_pos = 180;
    let n = n_zero + n_pos;
    let true_xi = n_zero as f64 / n as f64;

    let pos = gamma_latent(3.0, 0.5, n_pos);
    let mut vals = vec![0.0; n_zero];
    vals.extend(pos.iter().copied());
    let y = Array1::from_vec(vals);

    let data = dummy_data(n);
    let formula = Formula::new()
        .with_terms("mu", vec![Term::Intercept])
        .with_terms("sigma", vec![Term::Intercept])
        .with_terms("xi", vec![Term::Intercept]);

    let hurdle = Hurdle::new(Box::new(Gamma::new()));
    let fit = GamlssModel::fit(&data, &y, &formula, &hurdle).unwrap();

    // Fitted xi (logit link) ⇒ inverse-link the intercept.
    let xi_hat = fit.models["xi"].fitted_values[0];
    assert!(
        (xi_hat - true_xi).abs() < 0.05,
        "hurdle xi {xi_hat} should recover the zero fraction ≈ {true_xi}"
    );

    // Positive-part mean recovered near 3 (zeros excluded from the μ fit).
    let mu_hat = fit.models["mu"].fitted_values[0];
    assert!(
        (mu_hat - 3.0).abs() < 0.5,
        "positive-part mu {mu_hat} should recover ≈ 3.0"
    );
}
