//! Model diagnostic functions: residuals, log-likelihoods, and information criteria.

use crate::fitting::FittedParameter;
use ndarray::Array1;
use std::collections::HashMap;

const MIN_POSITIVE: f64 = 1e-10;

/// Aggregated model diagnostics including residuals, EDF, and information criteria.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ModelDiagnostics {
    /// Pearson residuals (standardized by distribution variance).
    pub pearson_residuals: Array1<f64>,
    /// Raw response residuals (y - fitted mu).
    pub response_residuals: Array1<f64>,
    /// Total effective degrees of freedom across all parameters.
    pub total_edf: f64,
    /// Akaike Information Criterion.
    pub aic: f64,
    /// Bayesian Information Criterion.
    pub bic: f64,
    /// Model log-likelihood evaluated at fitted parameters.
    pub log_likelihood: f64,
    /// Number of observations.
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
///
/// # Examples
///
/// ```
/// use glissando::diagnostics::compute_aic;
///
/// // Two models with the same log-likelihood: lower-EDF model has lower AIC.
/// let aic_simple = compute_aic(-100.0, 3.0);
/// let aic_complex = compute_aic(-100.0, 5.0);
/// assert!(aic_simple < aic_complex);
/// ```
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terms::Term;
    use crate::types::{Coefficients, CovarianceMatrix};
    use ndarray::{array, Array2};
    #[cfg(not(target_arch = "wasm32"))]
    use proptest::prelude::*;

    // --- Pearson residuals: zero when y == mu ---

    #[test]
    fn pearson_residuals_zero_when_y_equals_mu_gaussian() {
        let y = array![1.0, 2.0, 3.0];
        let r = pearson_residuals_gaussian(&y, &y, &array![1.0, 1.0, 1.0]);
        assert!(r.iter().all(|&v| v.abs() < 1e-12));
    }

    #[test]
    fn pearson_residuals_zero_when_y_equals_mu_poisson() {
        let y = array![1.0, 2.0, 3.0];
        let r = pearson_residuals_poisson(&y, &y);
        assert!(r.iter().all(|&v| v.abs() < 1e-12));
    }

    #[test]
    fn pearson_residuals_zero_when_y_equals_mu_gamma() {
        let y = array![1.0, 2.0, 3.0];
        let r = pearson_residuals_gamma(&y, &y, &array![0.5, 0.5, 0.5]);
        assert!(r.iter().all(|&v| v.abs() < 1e-12));
    }

    #[test]
    fn pearson_residuals_zero_when_y_equals_mu_negative_binomial() {
        let y = array![1.0, 4.0, 9.0];
        let r = pearson_residuals_negative_binomial(&y, &y, &array![0.5, 0.5, 0.5]);
        assert!(r.iter().all(|&v| v.abs() < 1e-12));
    }

    #[test]
    fn pearson_residuals_zero_when_y_equals_mu_beta() {
        let y = array![0.25, 0.5, 0.75];
        let r = pearson_residuals_beta(&y, &y, &array![10.0, 10.0, 10.0]);
        assert!(r.iter().all(|&v| v.abs() < 1e-12));
    }

    #[test]
    fn pearson_residuals_zero_when_y_equals_n_mu_binomial() {
        // Binomial expected counts are n * mu; residuals zero when y matches.
        let mu = array![0.3, 0.5, 0.7];
        let n = array![10.0, 10.0, 10.0];
        let y = &mu * &n;
        let r = pearson_residuals_binomial(&y, &mu, &n);
        assert!(r.iter().all(|&v| v.abs() < 1e-12));
    }

    #[test]
    fn pearson_residuals_handle_zero_mu_gracefully() {
        // mu=0 in Poisson would divide by zero; MIN_POSITIVE clamp must keep it finite.
        let y = array![0.0];
        let mu = array![0.0];
        let r = pearson_residuals_poisson(&y, &mu);
        assert!(r.iter().all(|v| v.is_finite()));
    }

    // --- Log-likelihoods: known values ---

    #[test]
    fn loglik_gaussian_matches_manual_formula() {
        let y = array![0.0];
        let mu = array![0.0];
        let sigma = array![1.0];
        let ll = loglik_gaussian(&y, &mu, &sigma);
        let expected = -0.5 * (2.0 * std::f64::consts::PI).ln(); // -0.5 log(2π)
        assert!((ll - expected).abs() < 1e-12);
    }

    #[test]
    fn loglik_poisson_matches_manual() {
        // l = y log(μ) − μ − log Γ(y+1). y=0, μ=1 → 0 − 1 − 0 = −1.
        let ll = loglik_poisson(&array![0.0], &array![1.0]);
        assert!((ll - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn loglik_binomial_matches_manual() {
        // n=2, y=1, mu=0.5 → log C(2,1) + 1·log(0.5) + 1·log(0.5) = log 2 + 2 log 0.5 = log 2 − log 4
        let ll = loglik_binomial(&array![1.0], &array![0.5], &array![2.0]);
        let expected = 2.0_f64.ln() + 2.0 * 0.5_f64.ln();
        assert!((ll - expected).abs() < 1e-9);
    }

    #[test]
    fn loglik_gamma_finite_on_typical_inputs() {
        let y = array![1.0, 2.0, 5.0];
        let mu = array![2.0, 2.0, 4.0];
        let sigma = array![0.5, 0.4, 0.3];
        let ll = loglik_gamma(&y, &mu, &sigma);
        assert!(ll.is_finite());
    }

    #[test]
    fn loglik_negative_binomial_finite() {
        let y = array![0.0, 5.0, 10.0];
        let mu = array![1.0, 4.0, 8.0];
        let sigma = array![0.5, 0.5, 0.5];
        let ll = loglik_negative_binomial(&y, &mu, &sigma);
        assert!(ll.is_finite());
    }

    #[test]
    fn loglik_beta_finite() {
        let y = array![0.1, 0.5, 0.9];
        let mu = array![0.2, 0.5, 0.8];
        let phi = array![10.0, 10.0, 10.0];
        let ll = loglik_beta(&y, &mu, &phi);
        assert!(ll.is_finite());
    }

    #[cfg(not(target_arch = "wasm32"))]
    proptest! {
        #[test]
        fn loglik_gaussian_matches_naive_loop(
            n in 1usize..20,
            mu_val in -5.0f64..5.0,
            sigma_val in 0.1f64..3.0,
        ) {
            let y = Array1::from_iter((0..n).map(|i| i as f64 * 0.1));
            let mu = Array1::from_elem(n, mu_val);
            let sigma = Array1::from_elem(n, sigma_val);
            let actual = loglik_gaussian(&y, &mu, &sigma);
            let log_2pi = (2.0 * std::f64::consts::PI).ln();
            let expected: f64 = (0..n).map(|i| {
                let z = (y[i] - mu_val) / sigma_val;
                -0.5 * log_2pi - sigma_val.ln() - 0.5 * z * z
            }).sum();
            prop_assert!((actual - expected).abs() < 1e-9);
        }
    }

    // --- compute_aic, compute_bic ---

    #[test]
    fn compute_aic_formula() {
        assert!((compute_aic(-100.0, 5.0) - 210.0).abs() < 1e-12);
        assert!((compute_aic(0.0, 0.0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn compute_bic_formula() {
        let n = 100usize;
        let bic = compute_bic(-100.0, 5.0, n);
        let expected = 200.0 + (n as f64).ln() * 5.0;
        assert!((bic - expected).abs() < 1e-12);
    }

    #[test]
    fn bic_exceeds_aic_for_n_above_e_squared() {
        // log(n) > 2 ⇒ BIC penalty is heavier than AIC's, so BIC > AIC for the same model.
        let ll = -50.0;
        let edf = 4.0;
        let aic = compute_aic(ll, edf);
        let bic = compute_bic(ll, edf, 100);
        assert!(bic > aic);
    }

    // --- response_residuals ---

    #[test]
    fn response_residuals_subtracts() {
        let y = array![3.0, 5.0, 7.0];
        let mu = array![1.0, 2.0, 4.0];
        let r = response_residuals(&y, &mu);
        assert_eq!(r, array![2.0, 3.0, 3.0]);
    }

    // --- total_edf ---

    fn dummy_fitted_param(edf: f64) -> FittedParameter {
        FittedParameter {
            coefficients: Coefficients(array![0.0]),
            covariance: CovarianceMatrix(Array2::<f64>::zeros((1, 1))),
            terms: vec![Term::Intercept],
            lambdas: array![],
            eta: array![0.0],
            fitted_values: array![0.0],
            edf,
        }
    }

    #[test]
    fn total_edf_sums_per_parameter_edf() {
        let mut params = HashMap::new();
        params.insert("mu".to_string(), dummy_fitted_param(3.5));
        params.insert("sigma".to_string(), dummy_fitted_param(1.2));
        assert!((total_edf(&params) - 4.7).abs() < 1e-12);
    }

    #[test]
    fn total_edf_empty_returns_zero() {
        let params: HashMap<String, FittedParameter> = HashMap::new();
        assert_eq!(total_edf(&params), 0.0);
    }
}
