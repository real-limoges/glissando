//! Hurdle integration tests: a `Hurdle` wrapper fits end-to-end and recovers
//! both the zero-atom probability and the positive-part parameters.

use glissando::distributions::{Distribution, Gamma, Hurdle};
use glissando::Param;
use glissando::{DataSet, Formula, GamlssModel, Term};
use ndarray::Array1;

fn dummy_data(n: usize) -> DataSet {
    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_iter((0..n).map(|i| i as f64)));
    data
}

/// Ideal Gamma order statistics for `(mu, sigma)` of length `m`. These are
/// evenly spaced quantiles with no sampling noise, so the recovery target is
/// exact and needs no slack.
fn gamma_latent(mu: f64, sigma: f64, m: usize) -> Array1<f64> {
    let p = Array1::from_iter((0..m).map(|i| (i as f64 + 0.5) / m as f64));
    let owned = [
        (Param::Mu, Array1::from_elem(m, mu)),
        (Param::Sigma, Array1::from_elem(m, sigma)),
    ];
    let view = owned.iter().map(|(k, v)| (*k, v)).collect();
    Gamma.quantile(&p, &view).unwrap()
}

/// Intercept-only hurdle-Gamma fit: 40% zeros, the rest Gamma(3, 0.5).
fn fit_hurdle_gamma() -> (GamlssModel, Array1<f64>, Hurdle) {
    let (n_zero, n_pos) = (120, 180);
    let mut vals = vec![0.0; n_zero];
    vals.extend(gamma_latent(3.0, 0.5, n_pos).iter().copied());
    let y = Array1::from_vec(vals);
    let formula = Formula::new()
        .with_terms(Param::Mu, vec![Term::Intercept])
        .with_terms(Param::Sigma, vec![Term::Intercept])
        .with_terms(Param::Xi, vec![Term::Intercept]);
    let hurdle = Hurdle::new(Box::new(Gamma::new()));
    let fit = GamlssModel::fit(&dummy_data(y.len()), &y, &formula, &hurdle).unwrap();
    (fit, y, hurdle)
}

/// The fitted mean counts the zeros.
#[test]
fn hurdle_expected_value_counts_the_zeros() {
    let (fit, y, hurdle) = fit_hurdle_gamma();
    let diagnostics = fit.diagnostics(&hurdle, &y).unwrap();
    let mean_fit = (&y - &diagnostics.response_residuals)[0];
    let mean_sample = y.mean().unwrap();
    assert!(
        (mean_fit - mean_sample).abs() < 0.1,
        "fitted mean {mean_fit} should match sample mean {mean_sample}"
    );
}

/// With the zero atom randomized, residuals of a correct fit look N(0, 1).
#[test]
fn hurdle_quantile_residuals_are_standard_normal() {
    let (fit, y, hurdle) = fit_hurdle_gamma();
    let r = fit.quantile_residuals(&hurdle, &y, Some(11)).unwrap();
    let mean = r.mean().unwrap();
    let sd = r.std(1.0);
    assert!(mean.abs() < 0.2, "residual mean {mean}");
    assert!((sd - 1.0).abs() < 0.2, "residual sd {sd}");

    let zeros: Vec<f64> = r.iter().take(120).copied().collect();
    let spread = zeros.iter().cloned().fold(f64::MIN, f64::max)
        - zeros.iter().cloned().fold(f64::MAX, f64::min);
    assert!(
        spread > 1.0,
        "zero-row residuals are lumped: spread {spread}"
    );
}

/// Centiles inside the atom are zero.
#[test]
fn hurdle_centiles_respect_the_atom() {
    let (fit, _y, hurdle) = fit_hurdle_gamma();
    let c = fit
        .centiles(&dummy_data(1), &hurdle, &[10.0, 90.0])
        .unwrap();
    assert_eq!(c["C10"][0], 0.0, "the 10th centile sits in the 40% atom");
    assert!(
        c["C90"][0] > 3.0,
        "the 90th centile is in the positive part"
    );
}

#[test]
fn hurdle_recovers_zero_fraction_and_positive_mean() {
    // 40% structural zeros, the other 60% positive Gamma(mu=3, sigma=0.5).
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
        .with_terms(Param::Mu, vec![Term::Intercept])
        .with_terms(Param::Sigma, vec![Term::Intercept])
        .with_terms(Param::Xi, vec![Term::Intercept]);

    let hurdle = Hurdle::new(Box::new(Gamma::new()));
    let fit = GamlssModel::fit(&data, &y, &formula, &hurdle).unwrap();

    // Fitted xi uses a logit link. Inverse-link the intercept to read it on the probability scale.
    let xi_hat = fit.models[&Param::Xi].fitted_values[0];
    assert!(
        (xi_hat - true_xi).abs() < 0.05,
        "hurdle xi {xi_hat} should recover the zero fraction ≈ {true_xi}"
    );

    // Positive-part mean lands near 3; in a hurdle model the zeros do not enter the μ fit.
    let mu_hat = fit.models[&Param::Mu].fitted_values[0];
    assert!(
        (mu_hat - 3.0).abs() < 0.5,
        "positive-part mu {mu_hat} should recover ≈ 3.0"
    );
}
