use crate::fitting::FittedParameter;
use ndarray::Array1;
use std::collections::HashMap;

#[derive(Debug, Clone)]
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
    (y - mu) / &sigma.mapv(|s| s.max(1e-10))
}

pub fn pearson_residuals_poisson(y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
    (y - mu) / &mu.mapv(|m| m.max(1e-10).sqrt())
}

pub fn pearson_residuals_gamma(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    sigma: &Array1<f64>,
) -> Array1<f64> {
    let sd = (mu * sigma).mapv(|v| v.max(1e-10));
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
            *v = (m + s * m * m).max(1e-10).sqrt();
        });
    (y - mu) / &variance
}

pub fn pearson_residuals_beta(y: &Array1<f64>, mu: &Array1<f64>, phi: &Array1<f64>) -> Array1<f64> {
    use ndarray::Zip;
    let mut sd = Array1::zeros(mu.len());
    Zip::from(&mut sd).and(mu).and(phi).for_each(|v, &m, &p| {
        let variance = m * (1.0 - m) / (1.0 + p);
        *v = variance.max(1e-10).sqrt();
    });
    (y - mu) / &sd
}

pub fn loglik_gaussian(y: &Array1<f64>, mu: &Array1<f64>, sigma: &Array1<f64>) -> f64 {
    use ndarray::Zip;
    let log_2pi = (2.0 * std::f64::consts::PI).ln();
    let mut ll = 0.0;
    Zip::from(y).and(mu).and(sigma).for_each(|&yi, &mui, &si| {
        let s = si.max(1e-10);
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
        ll += yi * mui.max(1e-10).ln() - mui - ln_gamma(yi + 1.0);
    });
    ll
}

pub fn loglik_gamma(y: &Array1<f64>, mu: &Array1<f64>, sigma: &Array1<f64>) -> f64 {
    use ndarray::Zip;
    use statrs::function::gamma::ln_gamma;
    let mut ll = 0.0;
    Zip::from(y).and(mu).and(sigma).for_each(|&yi, &mui, &si| {
        let s = si.max(1e-10);
        let alpha = 1.0 / (s * s);
        let theta = mui * s * s;
        ll +=
            (alpha - 1.0) * yi.max(1e-10).ln() - yi / theta - alpha * theta.ln() - ln_gamma(alpha);
    });
    ll
}

pub fn loglik_negative_binomial(y: &Array1<f64>, mu: &Array1<f64>, sigma: &Array1<f64>) -> f64 {
    use ndarray::Zip;
    use statrs::function::gamma::ln_gamma;
    let mut ll = 0.0;
    Zip::from(y).and(mu).and(sigma).for_each(|&yi, &mui, &si| {
        let r = 1.0 / si.max(1e-10);
        let p = r / (r + mui);
        ll += ln_gamma(yi + r) - ln_gamma(r) - ln_gamma(yi + 1.0)
            + r * p.max(1e-10).ln()
            + yi * (1.0 - p).max(1e-10).ln();
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
