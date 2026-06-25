// Integration tests cannot run with the `python` feature due to PyO3's extension-module linking.
#![cfg(not(feature = "python"))]

//! INFER-1 — randomized normalized quantile residuals.

mod common;

use common::{linear_intercepts, Generator};
use glissando::{
    distributions::{Distribution, Gaussian, Poisson},
    GamlssModel,
};
use ndarray::Array1;

fn mean(v: &Array1<f64>) -> f64 {
    v.sum() / v.len() as f64
}

fn variance(v: &Array1<f64>) -> f64 {
    let m = mean(v);
    v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64
}

/// Pearson correlation of sorted residuals against standard-normal order
/// statistics — the Filliben / Q-Q correlation (≈1 when residuals are normal).
fn filliben_correlation(resid: &Array1<f64>) -> f64 {
    use statrs::function::erf::erf_inv;
    let n = resid.len();
    let mut sorted: Vec<f64> = resid.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // Theoretical N(0,1) quantiles at plotting positions (i-0.5)/n.
    let theo: Vec<f64> = (0..n)
        .map(|i| {
            let p = (i as f64 + 0.5) / n as f64;
            std::f64::consts::SQRT_2 * erf_inv(2.0 * p - 1.0)
        })
        .collect();
    let mean_s = sorted.iter().sum::<f64>() / n as f64;
    let mean_t = theo.iter().sum::<f64>() / n as f64;
    let mut cov = 0.0;
    let mut var_s = 0.0;
    let mut var_t = 0.0;
    for i in 0..n {
        let ds = sorted[i] - mean_s;
        let dt = theo[i] - mean_t;
        cov += ds * dt;
        var_s += ds * ds;
        var_t += dt * dt;
    }
    cov / (var_s.sqrt() * var_t.sqrt())
}

#[test]
fn quantile_residuals_gaussian_are_standard_normal() {
    let mut rng = Generator::new(42);
    let (y, data) = rng.linear_gaussian(500, 1.0, 3.0, 1.0);
    let formula = linear_intercepts("x", &["mu", "sigma"]);
    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

    // Continuous family: seed is ignored; residuals are u = Φ⁻¹(F(y)).
    let resid = model
        .quantile_residuals(&Gaussian::new(), &y, None)
        .unwrap();
    assert_eq!(resid.len(), y.len());
    assert!(resid.iter().all(|r| r.is_finite()));
    assert!(mean(&resid).abs() < 0.15, "mean {}", mean(&resid));
    assert!(
        (variance(&resid) - 1.0).abs() < 0.3,
        "variance {}",
        variance(&resid)
    );
    assert!(
        filliben_correlation(&resid) > 0.99,
        "Filliben corr {}",
        filliben_correlation(&resid)
    );
}

#[test]
fn quantile_residuals_poisson_are_calibrated() {
    let mut rng = Generator::new(7);
    let (y, data) = rng.poisson_data(500, 0.5, 0.3);
    let formula = linear_intercepts("x", &["mu"]);
    let model = GamlssModel::fit(&data, &y, &formula, &Poisson::new()).unwrap();

    let resid = model
        .quantile_residuals(&Poisson::new(), &y, Some(123))
        .unwrap();
    assert!(resid.iter().all(|r| r.is_finite()));
    // Randomized PIT makes even a discrete fit map to ≈N(0,1).
    assert!(mean(&resid).abs() < 0.2, "mean {}", mean(&resid));
    assert!(
        (variance(&resid) - 1.0).abs() < 0.4,
        "variance {}",
        variance(&resid)
    );
    assert!(
        filliben_correlation(&resid) > 0.97,
        "Filliben corr {}",
        filliben_correlation(&resid)
    );
}

#[test]
fn quantile_residuals_continuous_ignore_seed() {
    let mut rng = Generator::new(1);
    let (y, data) = rng.linear_gaussian(100, 1.0, 2.0, 1.0);
    let formula = linear_intercepts("x", &["mu", "sigma"]);
    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

    let a = model
        .quantile_residuals(&Gaussian::new(), &y, Some(1))
        .unwrap();
    let b = model
        .quantile_residuals(&Gaussian::new(), &y, Some(999))
        .unwrap();
    let c = model
        .quantile_residuals(&Gaussian::new(), &y, None)
        .unwrap();
    // u = F(y) is deterministic for a continuous family — seed has no effect.
    for i in 0..y.len() {
        assert!((a[i] - b[i]).abs() < 1e-15);
        assert!((a[i] - c[i]).abs() < 1e-15);
    }
}

#[test]
fn quantile_residuals_discrete_seed_controls_randomization() {
    let mut rng = Generator::new(3);
    let (y, data) = rng.poisson_data(200, 0.5, 0.3);
    let formula = linear_intercepts("x", &["mu"]);
    let model = GamlssModel::fit(&data, &y, &formula, &Poisson::new()).unwrap();

    let same_a = model
        .quantile_residuals(&Poisson::new(), &y, Some(42))
        .unwrap();
    let same_b = model
        .quantile_residuals(&Poisson::new(), &y, Some(42))
        .unwrap();
    let diff = model
        .quantile_residuals(&Poisson::new(), &y, Some(43))
        .unwrap();

    // Same seed ⇒ identical; different seed ⇒ at least one residual differs.
    for i in 0..y.len() {
        assert!((same_a[i] - same_b[i]).abs() < 1e-15);
    }
    assert!(
        (0..y.len()).any(|i| (same_a[i] - diff[i]).abs() > 1e-9),
        "different seeds should change the randomized residuals"
    );
}

#[test]
fn quantile_residuals_discrete_respect_cdf_bracket() {
    use statrs::function::erf::erf;
    let mut rng = Generator::new(5);
    let (y, data) = rng.poisson_data(150, 0.5, 0.3);
    let formula = linear_intercepts("x", &["mu"]);
    let family = Poisson::new();
    let model = GamlssModel::fit(&data, &y, &formula, &family).unwrap();

    let resid = model.quantile_residuals(&family, &y, Some(11)).unwrap();

    // Reconstruct the fitted params and the jump interval [F(y−1), F(y)].
    let params_owned = model.predict(&data, &family).unwrap();
    let params: std::collections::HashMap<&str, &Array1<f64>> =
        params_owned.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let upper = family.cdf(&y, &params).unwrap();
    let lower = family.cdf(&(&y - 1.0), &params).unwrap();

    // Invert the residual: u = Φ(r). Every u must sit inside its jump interval.
    for i in 0..y.len() {
        let u = 0.5 * (1.0 + erf(resid[i] / std::f64::consts::SQRT_2));
        assert!(
            u >= lower[i] - 1e-9 && u <= upper[i] + 1e-9,
            "obs {}: u={} not in [{}, {}]",
            i,
            u,
            lower[i],
            upper[i]
        );
    }
}
