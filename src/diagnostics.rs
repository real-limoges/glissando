use crate::fitting::FittedParameter;
use ndarray::Array1;
use std::collections::HashMap;

/// Minimum positive value to prevent division by zero or log(0)
const MIN_POSITIVE: f64 = 1e-10;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ModelDiagnostics {
    pub pearson_residuals: Array1<f64>,
    pub response_residuals: Array1<f64>,
    pub total_edf: f64,
    pub aic: f64,
    pub bic: f64,
    pub log_likelihood: f64,
    pub n_obs: usize,
}

pub fn pearson_residuals_gaussian(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    sigma: &Array1<f64>,
) -> Array1<f64> {
    (y - mu) / &sigma.mapv(|s| s.max(MIN_POSITIVE))
}

pub fn pearson_residuals_poisson(y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
    (y - mu) / &mu.mapv(|m| m.max(MIN_POSITIVE).sqrt())
}

pub fn pearson_residuals_gamma(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    sigma: &Array1<f64>,
) -> Array1<f64> {
    let sd = (mu * sigma).mapv(|v| v.max(MIN_POSITIVE));
    (y - mu) / &sd
}

pub fn pearson_residuals_negative_binomial(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    sigma: &Array1<f64>,
) -> Array1<f64> {
    use ndarray::Zip;
    let mut variance = Array1::zeros(mu.len());
    Zip::from(&mut variance)
        .and(mu)
        .and(sigma)
        .for_each(|v, &m, &s| {
            *v = (m + s * m * m).max(MIN_POSITIVE).sqrt();
        });
    (y - mu) / &variance
}

pub fn pearson_residuals_beta(y: &Array1<f64>, mu: &Array1<f64>, phi: &Array1<f64>) -> Array1<f64> {
    use ndarray::Zip;
    let mut sd = Array1::zeros(mu.len());
    Zip::from(&mut sd).and(mu).and(phi).for_each(|v, &m, &p| {
        let variance = m * (1.0 - m) / (1.0 + p);
        *v = variance.max(MIN_POSITIVE).sqrt();
    });
    (y - mu) / &sd
}

pub fn pearson_residuals_binomial(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    n: &Array1<f64>,
) -> Array1<f64> {
    use ndarray::Zip;
    let mut sd = Array1::zeros(mu.len());
    Zip::from(&mut sd).and(mu).and(n).for_each(|v, &m, &ni| {
        // Variance of binomial: n * mu * (1 - mu)
        let variance = ni * m * (1.0 - m);
        *v = variance.max(MIN_POSITIVE).sqrt();
    });
    // Residuals: (y - n*mu) / sqrt(variance)
    let expected = mu * n;
    (y - &expected) / &sd
}

pub fn loglik_gaussian(y: &Array1<f64>, mu: &Array1<f64>, sigma: &Array1<f64>) -> f64 {
    use ndarray::Zip;
    let log_2pi = (2.0 * std::f64::consts::PI).ln();
    let mut ll = 0.0;
    Zip::from(y).and(mu).and(sigma).for_each(|&yi, &mui, &si| {
        let s = si.max(MIN_POSITIVE);
        let z = (yi - mui) / s;
        ll += -0.5 * log_2pi - s.ln() - 0.5 * z * z;
    });
    ll
}

pub fn loglik_poisson(y: &Array1<f64>, mu: &Array1<f64>) -> f64 {
    use ndarray::Zip;
    use statrs::function::gamma::ln_gamma;
    let mut ll = 0.0;
    Zip::from(y).and(mu).for_each(|&yi, &mui| {
        ll += yi * mui.max(MIN_POSITIVE).ln() - mui - ln_gamma(yi + 1.0);
    });
    ll
}

pub fn loglik_binomial(y: &Array1<f64>, mu: &Array1<f64>, n: &Array1<f64>) -> f64 {
    use ndarray::Zip;
    use statrs::function::gamma::ln_gamma;
    let mut ll = 0.0;
    Zip::from(y).and(mu).and(n).for_each(|&yi, &mui, &ni| {
        let m = mui.clamp(MIN_POSITIVE, 1.0 - MIN_POSITIVE);
        // log(C(n,y)) + y*log(mu) + (n-y)*log(1-mu)
        ll += ln_gamma(ni + 1.0) - ln_gamma(yi + 1.0) - ln_gamma(ni - yi + 1.0)
            + yi * m.ln()
            + (ni - yi) * (1.0 - m).ln();
    });
    ll
}

pub fn loglik_gamma(y: &Array1<f64>, mu: &Array1<f64>, sigma: &Array1<f64>) -> f64 {
    use ndarray::Zip;
    use statrs::function::gamma::ln_gamma;
    let mut ll = 0.0;
    Zip::from(y).and(mu).and(sigma).for_each(|&yi, &mui, &si| {
        let s = si.max(MIN_POSITIVE);
        let alpha = 1.0 / (s * s);
        let theta = mui * s * s;
        ll += (alpha - 1.0) * yi.max(MIN_POSITIVE).ln()
            - yi / theta
            - alpha * theta.ln()
            - ln_gamma(alpha);
    });
    ll
}

pub fn loglik_negative_binomial(y: &Array1<f64>, mu: &Array1<f64>, sigma: &Array1<f64>) -> f64 {
    use ndarray::Zip;
    use statrs::function::gamma::ln_gamma;
    let mut ll = 0.0;
    Zip::from(y).and(mu).and(sigma).for_each(|&yi, &mui, &si| {
        let r = 1.0 / si.max(MIN_POSITIVE);
        let p = r / (r + mui);
        ll += ln_gamma(yi + r) - ln_gamma(r) - ln_gamma(yi + 1.0)
            + r * p.max(MIN_POSITIVE).ln()
            + yi * (1.0 - p).max(MIN_POSITIVE).ln();
    });
    ll
}

pub fn loglik_beta(y: &Array1<f64>, mu: &Array1<f64>, phi: &Array1<f64>) -> f64 {
    use ndarray::Zip;
    use statrs::function::gamma::ln_gamma;
    let mut ll = 0.0;
    Zip::from(y).and(mu).and(phi).for_each(|&yi, &mui, &phii| {
        let alpha = mui * phii;
        let beta = (1.0 - mui) * phii;
        let y_clamped = yi.clamp(1e-10, 1.0 - 1e-10);
        ll += ln_gamma(phii) - ln_gamma(alpha) - ln_gamma(beta)
            + (alpha - 1.0) * y_clamped.ln()
            + (beta - 1.0) * (1.0 - y_clamped).ln();
    });
    ll
}

pub fn compute_aic(log_likelihood: f64, total_edf: f64) -> f64 {
    -2.0 * log_likelihood + 2.0 * total_edf
}

pub fn compute_bic(log_likelihood: f64, total_edf: f64, n_obs: usize) -> f64 {
    -2.0 * log_likelihood + (n_obs as f64).ln() * total_edf
}

pub fn total_edf(fitted_params: &HashMap<String, FittedParameter>) -> f64 {
    fitted_params.values().map(|p| p.edf).sum()
}

pub fn response_residuals(y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
    y - mu
}
