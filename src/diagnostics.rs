// Model diagnostics for GAMLSS
// Provides residuals, information criteria, and model summaries

use crate::fitting::FittedParameter;
use ndarray::Array1;
use std::collections::HashMap;

/// Diagnostic results for a fitted GAMLSS model
#[derive(Debug, Clone)]
pub struct ModelDiagnostics {
    /// Pearson residuals: (y - mu) / sqrt(Var(y))
    pub pearson_residuals: Array1<f64>,
    /// Response residuals: y - mu
    pub response_residuals: Array1<f64>,
    /// Total effective degrees of freedom across all parameters
    pub total_edf: f64,
    /// AIC: -2*loglik + 2*edf
    pub aic: f64,
    /// BIC: -2*loglik + log(n)*edf
    pub bic: f64,
    /// Log-likelihood at fitted values
    pub log_likelihood: f64,
    /// Number of observations
    pub n_obs: usize,
}

/// Compute Pearson residuals for Gaussian distribution
/// r_i = (y_i - mu_i) / sigma_i
pub fn pearson_residuals_gaussian(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    sigma: &Array1<f64>,
) -> Array1<f64> {
    let n = y.len();
    let mut residuals = Array1::zeros(n);
    for i in 0..n {
        residuals[i] = (y[i] - mu[i]) / sigma[i].max(1e-10);
    }
    residuals
}

/// Compute Pearson residuals for Poisson distribution
/// r_i = (y_i - mu_i) / sqrt(mu_i)
pub fn pearson_residuals_poisson(y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
    let n = y.len();
    let mut residuals = Array1::zeros(n);
    for i in 0..n {
        residuals[i] = (y[i] - mu[i]) / mu[i].max(1e-10).sqrt();
    }
    residuals
}

/// Compute Pearson residuals for Gamma distribution
/// r_i = (y_i - mu_i) / (mu_i * sigma_i)
pub fn pearson_residuals_gamma(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    sigma: &Array1<f64>,
) -> Array1<f64> {
    let n = y.len();
    let mut residuals = Array1::zeros(n);
    for i in 0..n {
        let sd = mu[i] * sigma[i]; // SD = mu * CV
        residuals[i] = (y[i] - mu[i]) / sd.max(1e-10);
    }
    residuals
}

/// Compute Pearson residuals for Negative Binomial distribution
/// r_i = (y_i - mu_i) / sqrt(mu_i + sigma_i * mu_i^2)
pub fn pearson_residuals_negative_binomial(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    sigma: &Array1<f64>,
) -> Array1<f64> {
    let n = y.len();
    let mut residuals = Array1::zeros(n);
    for i in 0..n {
        let variance = mu[i] + sigma[i] * mu[i].powi(2);
        residuals[i] = (y[i] - mu[i]) / variance.max(1e-10).sqrt();
    }
    residuals
}

/// Compute Pearson residuals for Beta distribution
/// r_i = (y_i - mu_i) / sqrt(mu_i * (1 - mu_i) / (1 + phi_i))
pub fn pearson_residuals_beta(y: &Array1<f64>, mu: &Array1<f64>, phi: &Array1<f64>) -> Array1<f64> {
    let n = y.len();
    let mut residuals = Array1::zeros(n);
    for i in 0..n {
        let variance = mu[i] * (1.0 - mu[i]) / (1.0 + phi[i]);
        residuals[i] = (y[i] - mu[i]) / variance.max(1e-10).sqrt();
    }
    residuals
}

/// Compute log-likelihood for Gaussian distribution
pub fn loglik_gaussian(y: &Array1<f64>, mu: &Array1<f64>, sigma: &Array1<f64>) -> f64 {
    let n = y.len();
    let mut ll = 0.0;
    let log_2pi = (2.0 * std::f64::consts::PI).ln();
    for i in 0..n {
        let z = (y[i] - mu[i]) / sigma[i].max(1e-10);
        ll += -0.5 * log_2pi - sigma[i].max(1e-10).ln() - 0.5 * z * z;
    }
    ll
}

/// Compute log-likelihood for Poisson distribution
pub fn loglik_poisson(y: &Array1<f64>, mu: &Array1<f64>) -> f64 {
    use statrs::function::gamma::ln_gamma;
    let n = y.len();
    let mut ll = 0.0;
    for i in 0..n {
        // l = y*log(mu) - mu - log(y!)
        ll += y[i] * mu[i].max(1e-10).ln() - mu[i] - ln_gamma(y[i] + 1.0);
    }
    ll
}

/// Compute log-likelihood for Gamma distribution
pub fn loglik_gamma(y: &Array1<f64>, mu: &Array1<f64>, sigma: &Array1<f64>) -> f64 {
    use statrs::function::gamma::ln_gamma;
    let n = y.len();
    let mut ll = 0.0;
    for i in 0..n {
        let alpha = 1.0 / sigma[i].max(1e-10).powi(2); // shape
        let theta = mu[i] * sigma[i].max(1e-10).powi(2); // scale
                                                         // l = (alpha-1)*log(y) - y/theta - alpha*log(theta) - log(Gamma(alpha))
        ll += (alpha - 1.0) * y[i].max(1e-10).ln()
            - y[i] / theta
            - alpha * theta.ln()
            - ln_gamma(alpha);
    }
    ll
}

/// Compute log-likelihood for Negative Binomial distribution
pub fn loglik_negative_binomial(y: &Array1<f64>, mu: &Array1<f64>, sigma: &Array1<f64>) -> f64 {
    use statrs::function::gamma::ln_gamma;
    let n = y.len();
    let mut ll = 0.0;
    for i in 0..n {
        let r = 1.0 / sigma[i].max(1e-10); // size parameter
        let p = r / (r + mu[i]); // probability
                                 // l = log(Gamma(y+r)) - log(Gamma(r)) - log(y!) + r*log(p) + y*log(1-p)
        ll += ln_gamma(y[i] + r) - ln_gamma(r) - ln_gamma(y[i] + 1.0)
            + r * p.max(1e-10).ln()
            + y[i] * (1.0 - p).max(1e-10).ln();
    }
    ll
}

/// Compute log-likelihood for Beta distribution
pub fn loglik_beta(y: &Array1<f64>, mu: &Array1<f64>, phi: &Array1<f64>) -> f64 {
    use statrs::function::gamma::ln_gamma;
    let n = y.len();
    let mut ll = 0.0;
    for i in 0..n {
        let alpha = mu[i] * phi[i];
        let beta = (1.0 - mu[i]) * phi[i];
        let y_clamped = y[i].clamp(1e-10, 1.0 - 1e-10);
        // l = log(Gamma(phi)) - log(Gamma(alpha)) - log(Gamma(beta))
        //     + (alpha-1)*log(y) + (beta-1)*log(1-y)
        ll += ln_gamma(phi[i]) - ln_gamma(alpha) - ln_gamma(beta)
            + (alpha - 1.0) * y_clamped.ln()
            + (beta - 1.0) * (1.0 - y_clamped).ln();
    }
    ll
}

/// Compute AIC: -2*loglik + 2*edf
pub fn compute_aic(log_likelihood: f64, total_edf: f64) -> f64 {
    -2.0 * log_likelihood + 2.0 * total_edf
}

/// Compute BIC: -2*loglik + log(n)*edf
pub fn compute_bic(log_likelihood: f64, total_edf: f64, n_obs: usize) -> f64 {
    -2.0 * log_likelihood + (n_obs as f64).ln() * total_edf
}

/// Compute total effective degrees of freedom from fitted parameters
pub fn total_edf(fitted_params: &HashMap<String, FittedParameter>) -> f64 {
    fitted_params.values().map(|p| p.edf).sum()
}

/// Compute response residuals (y - fitted_mu)
pub fn response_residuals(y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
    y - mu
}
