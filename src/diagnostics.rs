//! Model diagnostic functions: residuals, log-likelihoods, and information criteria.

use crate::fitting::FittedParameter;
use ndarray::Array1;
use std::collections::HashMap;

const MIN_POSITIVE: f64 = 1e-10;

/// Aggregated model diagnostics including residuals, EDF, and information criteria.
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

/// Computes Pearson residuals for a Gaussian model: (y - mu) / sigma.
pub fn pearson_residuals_gaussian(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    sigma: &Array1<f64>,
) -> Array1<f64> {
    (y - mu) / &sigma.mapv(|s| s.max(MIN_POSITIVE))
}

/// Computes Pearson residuals for a Poisson model: (y - mu) / sqrt(mu).
pub fn pearson_residuals_poisson(y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
    (y - mu) / &mu.mapv(|m| m.max(MIN_POSITIVE).sqrt())
}

/// Computes Pearson residuals for a Gamma model: (y - mu) / (mu * sigma).
pub fn pearson_residuals_gamma(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    sigma: &Array1<f64>,
) -> Array1<f64> {
    let sd = (mu * sigma).mapv(|v| v.max(MIN_POSITIVE));
    (y - mu) / &sd
}

/// Computes Pearson residuals for a Negative Binomial model: (y - mu) / sqrt(mu + sigma*mu^2).
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

/// Computes Pearson residuals for a Beta model: (y - mu) / sqrt(mu*(1-mu)/(1+phi)).
pub fn pearson_residuals_beta(y: &Array1<f64>, mu: &Array1<f64>, phi: &Array1<f64>) -> Array1<f64> {
    use ndarray::Zip;
    let mut sd = Array1::zeros(mu.len());
    Zip::from(&mut sd).and(mu).and(phi).for_each(|v, &m, &p| {
        let variance = m * (1.0 - m) / (1.0 + p);
        *v = variance.max(MIN_POSITIVE).sqrt();
    });
    (y - mu) / &sd
}

/// Computes Pearson residuals for a Binomial model: (y - n*mu) / sqrt(n*mu*(1-mu)).
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

/// Computes Gaussian log-likelihood: Σ[-0.5*log(2π) - log(σ) - 0.5*((y-μ)/σ)²].
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

/// Computes Poisson log-likelihood: Σ[y*log(μ) - μ - log(y!)].
pub fn loglik_poisson(y: &Array1<f64>, mu: &Array1<f64>) -> f64 {
    use ndarray::Zip;
    use statrs::function::gamma::ln_gamma;
    let mut ll = 0.0;
    Zip::from(y).and(mu).for_each(|&yi, &mui| {
        ll += yi * mui.max(MIN_POSITIVE).ln() - mui - ln_gamma(yi + 1.0);
    });
    ll
}

/// Computes Binomial log-likelihood: Σ[log(C(n,y)) + y*log(μ) + (n-y)*log(1-μ)].
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

/// Computes Gamma log-likelihood using (mu, sigma) parameterization where α = 1/σ².
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

/// Computes Negative Binomial log-likelihood (NB2 parameterization, r = 1/σ).
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

/// Computes Beta log-likelihood with (mu, phi) parameterization where α = μφ, β = (1-μ)φ.
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

/// Computes Akaike Information Criterion: -2*loglik + 2*EDF.
pub fn compute_aic(log_likelihood: f64, total_edf: f64) -> f64 {
    -2.0 * log_likelihood + 2.0 * total_edf
}

/// Computes Bayesian Information Criterion: -2*loglik + log(n)*EDF.
pub fn compute_bic(log_likelihood: f64, total_edf: f64, n_obs: usize) -> f64 {
    -2.0 * log_likelihood + (n_obs as f64).ln() * total_edf
}

/// Sums effective degrees of freedom across all fitted parameters.
pub fn total_edf(fitted_params: &HashMap<String, FittedParameter>) -> f64 {
    fitted_params.values().map(|p| p.edf).sum()
}

/// Computes raw response residuals: y - mu.
pub fn response_residuals(y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
    y - mu
}
