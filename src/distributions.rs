//! Probability distributions for GAMLSS models.
//!
//! This module provides standard distributions commonly used in statistical regression.
//! Each distribution defines:
//! - Parameters (e.g., μ, σ, ν)
//! - Default link functions for each parameter
//! - Derivatives (score and Fisher information) for the RS fitting algorithm
//!
//! All derivatives are computed in batched form (vectorized across observations) for efficiency.

use crate::error::GamlssError;
use crate::math::{digamma_batch, trigamma_batch};
use ndarray::Array1;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt::Debug;

/// Threshold for using parallel computation (below this, sequential is faster)
#[cfg(feature = "parallel")]
const PARALLEL_THRESHOLD: usize = 10_000;

/// Minimum value for positive parameters (mu, sigma, etc.) to avoid log(0) or division by zero
const MIN_POSITIVE: f64 = 1e-10;

/// Maximum linear predictor value for log/logit links to prevent overflow
const MAX_ETA: f64 = 30.0;

/// Minimum linear predictor value for log/logit links to prevent underflow
const MIN_ETA: f64 = -30.0;

/// Minimum Fisher information weight to ensure positive definiteness
const MIN_WEIGHT: f64 = 1e-6;

/// Link function trait for GLM/GAMLSS models.
///
/// A link function g maps the mean (mu) to the linear predictor (eta): η = g(μ).
/// The inverse link maps back: μ = g⁻¹(η).
pub trait Link: Debug + Send + Sync {
    /// Apply the link function: η = g(μ).
    fn link(&self, mu: f64) -> f64;
    /// Apply the inverse link function: μ = g⁻¹(η).
    fn inv_link(&self, eta: f64) -> f64;
}

/// Identity link: η = μ.
///
/// Used for unbounded continuous responses (e.g., Gaussian mean).
#[derive(Debug, Clone, Copy, Default)]
pub struct IdentityLink;
impl Link for IdentityLink {
    fn link(&self, mu: f64) -> f64 {
        mu
    }
    fn inv_link(&self, eta: f64) -> f64 {
        eta
    }
}

/// Log link: η = log(μ).
///
/// Used for positive continuous responses (e.g., Poisson mean, Gamma).
/// Clamps eta to [-30, 30] to prevent numerical overflow.
#[derive(Debug, Clone, Copy, Default)]
pub struct LogLink;
impl Link for LogLink {
    fn link(&self, mu: f64) -> f64 {
        mu.ln().max(MIN_ETA)
    }
    fn inv_link(&self, eta: f64) -> f64 {
        eta.min(MAX_ETA).exp()
    }
}

/// Logit link: η = log(μ / (1 - μ)).
///
/// Used for probability parameters bounded in (0, 1) (e.g., Binomial, Beta).
/// Clamps mu to [1e-10, 1-1e-10] and eta to [-30, 30] for numerical stability.
#[derive(Debug, Clone, Copy, Default)]
pub struct LogitLink;
impl Link for LogitLink {
    fn link(&self, mu: f64) -> f64 {
        // logit(mu) = log(mu / (1 - mu))
        let mu_clamped = mu.clamp(MIN_POSITIVE, 1.0 - MIN_POSITIVE);
        (mu_clamped / (1.0 - mu_clamped)).ln()
    }
    fn inv_link(&self, eta: f64) -> f64 {
        // inverse logit = 1 / (1 + exp(-eta))
        let eta_clamped = eta.clamp(MIN_ETA, MAX_ETA);
        1.0 / (1.0 + (-eta_clamped).exp())
    }
}

/// Result type for batched derivatives computation.
/// Maps parameter names to (score, Fisher info) array pairs.
pub type DerivativesResult = Result<HashMap<String, (Array1<f64>, Array1<f64>)>, GamlssError>;

/// A statistical distribution for GAMLSS, defining parameters, link functions, and derivatives.
///
/// Implementations must provide score vectors and Fisher information for each parameter
/// to drive the IRLS algorithm. See [`Gaussian`] for a reference implementation.
pub trait Distribution: Debug + Send + Sync {
    /// Returns the names of the distribution's parameters (e.g., `["mu", "sigma"]`).
    fn parameters(&self) -> &[&'static str];
    /// Returns the default link function for the given parameter name.
    ///
    /// # Errors
    ///
    /// Returns `GamlssError::UnknownParameter` if the parameter name is not recognized.
    fn default_link(&self, param: &str) -> Result<Box<dyn Link>, GamlssError>;
    /// Computes score (u) and Fisher information (w) for each distribution parameter.
    ///
    /// Returns a HashMap mapping parameter names to (u, w) array pairs where:
    /// - u: score vector (derivative of log-likelihood w.r.t. linear predictor)
    /// - w: Fisher information (weight for IRLS)
    fn derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult;
    /// Returns the distribution name as a static string (e.g., `"Gaussian"`).
    fn name(&self) -> &'static str;

    /// Returns initial value for a parameter on the response scale.
    ///
    /// Default implementation uses y.mean() for mu, y.std() for sigma, etc.
    /// Override for distributions where the response isn't directly the mean
    /// (e.g., Binomial where y is counts but mu is probability).
    fn initial_value(&self, param: &str, y: &Array1<f64>) -> f64 {
        match param {
            "mu" => y.mean().unwrap_or(0.5),
            "sigma" => {
                let s = y.std(1.0);
                if s < 1e-4 {
                    1.0
                } else {
                    s
                }
            }
            "nu" => 5.0,
            "phi" => 1.0,
            _ => 0.1,
        }
    }
}

/// Poisson distribution for count data.
///
/// Single parameter: mu (mean/rate) with log link.
/// Variance equals mean: Var(Y) = μ.
#[derive(Debug, Clone, Copy, Default)]
pub struct Poisson;
impl Poisson {
    /// Create a new Poisson distribution.
    pub fn new() -> Self {
        Self
    }
}

impl Distribution for Poisson {
    fn parameters(&self) -> &[&'static str] {
        &["mu"]
    }
    fn default_link(&self, param: &str) -> Result<Box<dyn Link>, GamlssError> {
        match param {
            "mu" => Ok(Box::new(LogLink)),
            _ => Err(GamlssError::UnknownParameter {
                distribution: self.name().to_string(),
                param: param.to_string(),
            }),
        }
    }
    fn derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        // Poisson log-likelihood: l = y*log(mu) - mu
        // Score (dl/dmu): u = y/mu - 1 = (y - mu)/mu
        // Fisher information: E[-d²l/dmu²] = 1/mu
        // For IRLS with log link, working weight w = mu (since Var(Y) = mu)
        let mu = *params
            .get("mu")
            .ok_or_else(|| GamlssError::UnknownParameter {
                distribution: self.name().to_string(),
                param: "mu".to_string(),
            })?;

        let deriv_u = y - mu;
        let deriv_w = mu.clone();

        Ok(HashMap::from([("mu".to_string(), (deriv_u, deriv_w))]))
    }
    fn name(&self) -> &'static str {
        "Poisson"
    }
}

/// Gaussian (Normal) distribution for continuous data.
///
/// Parameters: mu (mean, identity link) and sigma (std dev, log link).
/// Var(Y) = σ².
#[derive(Debug, Clone, Copy, Default)]
pub struct Gaussian;
impl Gaussian {
    /// Create a new Gaussian distribution.
    pub fn new() -> Self {
        Self
    }
}

impl Distribution for Gaussian {
    fn parameters(&self) -> &[&'static str] {
        &["mu", "sigma"]
    }
    fn default_link(&self, param: &str) -> Result<Box<dyn Link>, GamlssError> {
        match param {
            "mu" => Ok(Box::new(IdentityLink)),
            "sigma" => Ok(Box::new(LogLink)),
            _ => Err(GamlssError::UnknownParameter {
                distribution: self.name().to_string(),
                param: param.to_string(),
            }),
        }
    }
    fn derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        // Gaussian log-likelihood: l = -0.5*log(2*pi) - log(sigma) - (y-mu)^2/(2*sigma^2)
        //
        // For mu (identity link):
        //   dl/dmu = (y-mu)/sigma^2,  Fisher info = 1/sigma^2
        //
        // For sigma (log link, so we work with eta = log(sigma)):
        //   Chain rule gives u_sigma = [(y-mu)^2 - sigma^2] / sigma^2
        //   Fisher info for eta: w = 2 (since Var[(Y-mu)^2/sigma^2] = 2 for normal)
        // See docs/mathematics.md for full derivation.
        let mu = *params
            .get("mu")
            .ok_or_else(|| GamlssError::UnknownParameter {
                distribution: self.name().to_string(),
                param: "mu".to_string(),
            })?;
        let sigma = *params
            .get("sigma")
            .ok_or_else(|| GamlssError::UnknownParameter {
                distribution: self.name().to_string(),
                param: "sigma".to_string(),
            })?;

        let n = y.len();
        let sigma_sq = sigma.mapv(|s| s.powi(2));

        let u_mu = (y - mu) / &sigma_sq;
        let w_mu = sigma_sq.mapv(|s2| 1.0 / s2);

        let residual_sq = (y - mu).mapv(|r| r.powi(2));
        let u_sigma = (&residual_sq - &sigma_sq) / &sigma_sq;
        let w_sigma = Array1::from_elem(n, 2.0);

        Ok(HashMap::from([
            ("mu".to_string(), (u_mu, w_mu)),
            ("sigma".to_string(), (u_sigma, w_sigma)),
        ]))
    }
    fn name(&self) -> &'static str {
        "Gaussian"
    }
}

/// Student's t distribution for heavy-tailed continuous data.
///
/// Parameters: mu (location, identity link), sigma (scale, log link),
/// and nu (degrees of freedom, log link). As nu → ∞, approaches Gaussian.
#[derive(Debug, Clone, Copy, Default)]
pub struct StudentT;
impl StudentT {
    /// Create a new Student's t distribution.
    pub fn new() -> Self {
        Self
    }
}
impl Distribution for StudentT {
    fn parameters(&self) -> &[&'static str] {
        &["mu", "sigma", "nu"]
    }
    fn default_link(&self, param: &str) -> Result<Box<dyn Link>, GamlssError> {
        match param {
            "mu" => Ok(Box::new(IdentityLink)),
            "sigma" => Ok(Box::new(LogLink)),
            "nu" => Ok(Box::new(LogLink)),
            _ => Err(GamlssError::UnknownParameter {
                distribution: self.name().to_string(),
                param: param.to_string(),
            }),
        }
    }
    fn derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        // Student-t log-likelihood (location-scale parameterization).
        // See docs/mathematics.md for the full derivation of all derivatives.
        let mu = *params
            .get("mu")
            .ok_or_else(|| GamlssError::UnknownParameter {
                distribution: self.name().to_string(),
                param: "mu".to_string(),
            })?;
        let sigma = *params
            .get("sigma")
            .ok_or_else(|| GamlssError::UnknownParameter {
                distribution: self.name().to_string(),
                param: "sigma".to_string(),
            })?;
        let nu = *params
            .get("nu")
            .ok_or_else(|| GamlssError::UnknownParameter {
                distribution: self.name().to_string(),
                param: "nu".to_string(),
            })?;

        #[cfg(feature = "parallel")]
        let n = y.len();
        let z = (y - mu) / sigma;
        let z_sq = z.mapv(|v| v.powi(2));

        // w_robust = (nu+1)/(nu+z^2) appears in all derivatives.
        // This "robustifying weight" downweights outliers (large |z|).
        // As nu -> infinity, w_robust -> 1 and we recover Gaussian behavior.
        #[cfg(feature = "parallel")]
        let w_robust: Array1<f64> = if n < PARALLEL_THRESHOLD {
            nu.iter()
                .zip(z_sq.iter())
                .map(|(&nu_i, &z2_i)| (nu_i + 1.0) / (nu_i + z2_i))
                .collect()
        } else {
            let nu_slice = nu.as_slice().expect("nu array not contiguous");
            let z_sq_slice = z_sq.as_slice().expect("z_sq array not contiguous");
            Array1::from_vec(
                nu_slice
                    .par_iter()
                    .zip(z_sq_slice.par_iter())
                    .map(|(&nu_i, &z2_i)| (nu_i + 1.0) / (nu_i + z2_i))
                    .collect(),
            )
        };

        #[cfg(not(feature = "parallel"))]
        let w_robust: Array1<f64> = nu
            .iter()
            .zip(z_sq.iter())
            .map(|(&nu_i, &z2_i)| (nu_i + 1.0) / (nu_i + z2_i))
            .collect();

        // --- mu derivatives (identity link) ---
        let u_mu = (&w_robust * &z) / sigma;
        let w_mu = &w_robust / sigma.mapv(|s| s.powi(2));

        // --- sigma derivatives (log link) ---
        // Chain rule: dl/d_eta = sigma * dl/d_sigma = w_robust*z^2 - 1
        let u_sigma = &w_robust * &z_sq - 1.0;
        let w_sigma: Array1<f64> = nu.mapv(|nu_i| (2.0 * nu_i) / (nu_i + 3.0));

        // --- nu derivatives (log link) ---
        // The score involves digamma functions (derivative of log-gamma).
        // Use batch digamma for vectorized computation
        let nu_plus_1_half = nu.mapv(|nu_i| (nu_i + 1.0) / 2.0);
        let nu_half = nu.mapv(|nu_i| nu_i / 2.0);
        let d1 = digamma_batch(&nu_plus_1_half);
        let d2 = digamma_batch(&nu_half);

        #[cfg(feature = "parallel")]
        let (term3, term4): (Array1<f64>, Array1<f64>) = if n < PARALLEL_THRESHOLD {
            let t3: Array1<f64> = nu
                .iter()
                .zip(z_sq.iter())
                .map(|(&nu_i, &z2_i)| (1.0 + z2_i / nu_i).ln())
                .collect();
            let t4: Array1<f64> = nu
                .iter()
                .zip(w_robust.iter())
                .zip(z_sq.iter())
                .map(|((&nu_i, &w_i), &z2_i)| (w_i * z2_i - 1.0) / nu_i)
                .collect();
            (t3, t4)
        } else {
            let nu_slice = nu.as_slice().expect("nu array not contiguous");
            let z_sq_slice = z_sq.as_slice().expect("z_sq array not contiguous");
            let w_robust_slice = w_robust.as_slice().expect("w_robust array not contiguous");

            let t3: Vec<f64> = nu_slice
                .par_iter()
                .zip(z_sq_slice.par_iter())
                .map(|(&nu_i, &z2_i)| (1.0 + z2_i / nu_i).ln())
                .collect();
            let t4: Vec<f64> = (0..n)
                .into_par_iter()
                .map(|i| (w_robust_slice[i] * z_sq_slice[i] - 1.0) / nu_slice[i])
                .collect();
            (Array1::from_vec(t3), Array1::from_vec(t4))
        };

        #[cfg(not(feature = "parallel"))]
        let (term3, term4): (Array1<f64>, Array1<f64>) = {
            let t3: Array1<f64> = nu
                .iter()
                .zip(z_sq.iter())
                .map(|(&nu_i, &z2_i)| (1.0 + z2_i / nu_i).ln())
                .collect();
            let t4: Array1<f64> = nu
                .iter()
                .zip(w_robust.iter())
                .zip(z_sq.iter())
                .map(|((&nu_i, &w_i), &z2_i)| (w_i * z2_i - 1.0) / nu_i)
                .collect();
            (t3, t4)
        };

        let dl_dnu = 0.5 * (&d1 - &d2 - &term3 + &term4);

        // Chain rule for log link: u_eta = nu * dl/dnu
        let u_nu = &dl_dnu * nu;

        // Fisher information for nu involves trigamma functions (second derivative of log-gamma).
        // Use batch trigamma for vectorized computation
        let t1 = trigamma_batch(&nu_half);
        let t2 = trigamma_batch(&nu_plus_1_half);
        let t3: Array1<f64> = nu.mapv(|nu_i| (2.0 * (nu_i + 3.0)) / (nu_i * (nu_i + 1.0)));
        // Note: The sign is + not - because t3 subtracts from the negative Hessian
        let i_nu = 0.25 * (&t1 - &t2 + &t3);
        // Floor at MIN_WEIGHT to ensure positive definiteness of the weight matrix.
        // For log link: W_eta = I_nu * nu^2
        #[cfg(feature = "parallel")]
        let w_nu: Array1<f64> = if n < PARALLEL_THRESHOLD {
            i_nu.iter()
                .zip(nu.iter())
                .map(|(&i, &nu_i)| (i * nu_i.powi(2)).abs().max(MIN_WEIGHT))
                .collect()
        } else {
            let i_nu_slice = i_nu.as_slice().expect("i_nu array not contiguous");
            let nu_slice = nu.as_slice().expect("nu array not contiguous");
            Array1::from_vec(
                i_nu_slice
                    .par_iter()
                    .zip(nu_slice.par_iter())
                    .map(|(&i, &nu_i)| (i * nu_i.powi(2)).abs().max(MIN_WEIGHT))
                    .collect(),
            )
        };

        #[cfg(not(feature = "parallel"))]
        let w_nu: Array1<f64> = i_nu
            .iter()
            .zip(nu.iter())
            .map(|(&i, &nu_i)| (i * nu_i.powi(2)).abs().max(MIN_WEIGHT))
            .collect();

        Ok(HashMap::from([
            ("mu".to_string(), (u_mu, w_mu)),
            ("sigma".to_string(), (u_sigma, w_sigma)),
            ("nu".to_string(), (u_nu, w_nu)),
        ]))
    }
    fn name(&self) -> &'static str {
        "StudentT"
    }
}

/// Gamma distribution for positive continuous data.
///
/// Parameters: mu (mean, log link) and sigma (coefficient of variation, log link).
/// Parameterization: Shape α = 1/σ², Scale θ = μσ². Var(Y) = μ²σ².
#[derive(Debug, Clone, Copy, Default)]
pub struct Gamma;

impl Gamma {
    /// Creates a new Gamma distribution.
    pub fn new() -> Self {
        Self
    }
}

impl Distribution for Gamma {
    fn parameters(&self) -> &[&'static str] {
        &["mu", "sigma"]
    }

    fn default_link(&self, param: &str) -> Result<Box<dyn Link>, GamlssError> {
        match param {
            "mu" => Ok(Box::new(LogLink)),
            "sigma" => Ok(Box::new(LogLink)),
            _ => Err(GamlssError::UnknownParameter {
                distribution: self.name().to_string(),
                param: param.to_string(),
            }),
        }
    }

    fn derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        // Gamma log-likelihood with (mu, sigma) parameterization:
        // alpha = 1/sigma^2 (shape), theta = mu*sigma^2 (scale)
        // l = -alpha*log(theta) - log(Gamma(alpha)) + (alpha-1)*log(y) - y/theta
        //
        // For mu (log link, eta = log(mu)):
        //   dl/dmu = (y - mu) / (mu^2 * sigma^2)
        //   dl/deta = mu * dl/dmu = (y - mu) / (mu * sigma^2)
        //   Fisher info = 1/sigma^2
        //
        // For sigma (log link, eta = log(sigma)):
        //   Score involves digamma function. See docs/mathematics.md for derivation.
        let mu = *params
            .get("mu")
            .ok_or_else(|| GamlssError::UnknownParameter {
                distribution: self.name().to_string(),
                param: "mu".to_string(),
            })?;
        let sigma = *params
            .get("sigma")
            .ok_or_else(|| GamlssError::UnknownParameter {
                distribution: self.name().to_string(),
                param: "sigma".to_string(),
            })?;

        let mu_safe = mu.mapv(|m| m.max(MIN_POSITIVE));
        let sigma_safe = sigma.mapv(|s| s.max(MIN_POSITIVE));
        let sigma_sq = sigma_safe.mapv(|s| s.powi(2));
        let alpha = sigma_sq.mapv(|s2| 1.0 / s2);

        // mu derivatives (log link)
        let u_mu = (y - &mu_safe) / (&mu_safe * &sigma_sq);
        let w_mu = sigma_sq.mapv(|s2| 1.0 / s2);

        // sigma derivatives (log link)
        // For log link eta = log(sigma), the score is:
        // dl/deta = (2/sigma^2) * [digamma(1/sigma^2) + 2*log(sigma) - log(y/mu) + y/mu - 1]
        let psi_alpha = digamma_batch(&alpha);
        let log_sigma = sigma_safe.mapv(|s| s.ln());
        let log_y_over_mu = (y / &mu_safe).mapv(|v| v.ln());
        let y_over_mu = y / &mu_safe;

        let u_sigma =
            (2.0 / &sigma_sq) * (&psi_alpha + 2.0 * &log_sigma - &log_y_over_mu + &y_over_mu - 1.0);

        // Fisher info for sigma involves trigamma
        // I_sigma = (4/sigma^4) * trigamma(1/sigma^2) - 2/sigma^2
        let psi_prime_alpha = trigamma_batch(&alpha);
        let sigma_sq_sq = sigma_sq.mapv(|s2| s2.powi(2));
        let w_sigma = ((4.0 / &sigma_sq_sq) * &psi_prime_alpha - 2.0 / &sigma_sq)
            .mapv(|v| v.abs().max(MIN_WEIGHT));

        Ok(HashMap::from([
            ("mu".to_string(), (u_mu, w_mu)),
            ("sigma".to_string(), (u_sigma, w_sigma)),
        ]))
    }

    fn name(&self) -> &'static str {
        "Gamma"
    }
}

/// Negative Binomial distribution for overdispersed count data (NB2 parameterization).
///
/// Parameters: mu (mean, log link) and sigma (overdispersion, log link).
/// Var(Y) = μ + σμ². As σ → 0, approaches Poisson.
#[derive(Debug, Clone, Copy, Default)]
pub struct NegativeBinomial;

impl NegativeBinomial {
    /// Create a new Negative Binomial distribution.
    pub fn new() -> Self {
        Self
    }
}

impl Distribution for NegativeBinomial {
    fn parameters(&self) -> &[&'static str] {
        &["mu", "sigma"]
    }

    fn default_link(&self, param: &str) -> Result<Box<dyn Link>, GamlssError> {
        match param {
            "mu" => Ok(Box::new(LogLink)),
            "sigma" => Ok(Box::new(LogLink)),
            _ => Err(GamlssError::UnknownParameter {
                distribution: self.name().to_string(),
                param: param.to_string(),
            }),
        }
    }

    fn derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        // Negative Binomial (NB2) log-likelihood:
        // l = log(Gamma(y + 1/sigma)) - log(Gamma(1/sigma)) - log(y!)
        //     + (1/sigma)*log(1/(1+sigma*mu)) + y*log(sigma*mu/(1+sigma*mu))
        //
        // For mu (log link):
        //   dl/deta = (y - mu) / (1 + sigma*mu)
        //   Fisher info = mu / (1 + sigma*mu)
        //
        // For sigma (log link):
        //   Score involves digamma differences. See docs/mathematics.md.
        let mu = *params
            .get("mu")
            .ok_or_else(|| GamlssError::UnknownParameter {
                distribution: self.name().to_string(),
                param: "mu".to_string(),
            })?;
        let sigma = *params
            .get("sigma")
            .ok_or_else(|| GamlssError::UnknownParameter {
                distribution: self.name().to_string(),
                param: "sigma".to_string(),
            })?;

        #[cfg(feature = "parallel")]
        let n = y.len();
        let mu_safe = mu.mapv(|m| m.max(MIN_POSITIVE));
        let sigma_safe = sigma.mapv(|s| s.max(MIN_POSITIVE));

        #[cfg(feature = "parallel")]
        let one_plus_sigma_mu: Array1<f64> = if n < PARALLEL_THRESHOLD {
            sigma_safe
                .iter()
                .zip(mu_safe.iter())
                .map(|(&s, &m)| 1.0 + s * m)
                .collect()
        } else {
            let sigma_slice = sigma_safe.as_slice().expect("sigma array not contiguous");
            let mu_slice = mu_safe.as_slice().expect("mu array not contiguous");
            Array1::from_vec(
                sigma_slice
                    .par_iter()
                    .zip(mu_slice.par_iter())
                    .map(|(&s, &m)| 1.0 + s * m)
                    .collect(),
            )
        };

        #[cfg(not(feature = "parallel"))]
        let one_plus_sigma_mu: Array1<f64> = sigma_safe
            .iter()
            .zip(mu_safe.iter())
            .map(|(&s, &m)| 1.0 + s * m)
            .collect();

        // mu derivatives (log link)
        let u_mu = (y - &mu_safe) / &one_plus_sigma_mu;
        let w_mu = &mu_safe / &one_plus_sigma_mu;

        // sigma derivatives (log link)
        // dl/dsigma = (-1/sigma^2) * [digamma(y + r) - digamma(r) - log(1+sigma*mu) + (y-mu)/(1+sigma*mu)]
        // dl/deta = sigma * dl/dsigma
        let r = sigma_safe.mapv(|s| 1.0 / s);
        let y_plus_r = y + &r;
        let psi_y_r = digamma_batch(&y_plus_r);
        let psi_r = digamma_batch(&r);
        let log_term = one_plus_sigma_mu.mapv(|v| v.ln());
        let ratio_term = (y - &mu_safe) / &one_plus_sigma_mu;

        let u_sigma = (-1.0 / &sigma_safe) * (&psi_y_r - &psi_r - &log_term + &ratio_term);

        // Fisher info for sigma: involves E[digamma(Y + r)]
        // Using approximation based on trigamma(r)
        // w_sigma = (1/sigma^2) * trigamma(1/sigma) (simplified)
        let psi_prime_r = trigamma_batch(&r);
        let sigma_sq = sigma_safe.mapv(|s| s.powi(2));
        let w_sigma: Array1<f64> = (&psi_prime_r / &sigma_sq).mapv(|v| v.abs().max(MIN_WEIGHT));

        Ok(HashMap::from([
            ("mu".to_string(), (u_mu, w_mu)),
            ("sigma".to_string(), (u_sigma, w_sigma)),
        ]))
    }

    fn name(&self) -> &'static str {
        "NegativeBinomial"
    }
}

/// Beta distribution for proportions/rates in (0, 1).
///
/// Parameters: mu (mean, logit link) and phi (precision, log link).
/// Shape α = μφ, β = (1-μ)φ. Var(Y) = μ(1-μ)/(1+φ).
#[derive(Debug, Clone, Copy, Default)]
pub struct Beta;

impl Beta {
    /// Create a new Beta distribution.
    pub fn new() -> Self {
        Self
    }
}

impl Distribution for Beta {
    fn parameters(&self) -> &[&'static str] {
        &["mu", "phi"]
    }

    fn default_link(&self, param: &str) -> Result<Box<dyn Link>, GamlssError> {
        match param {
            "mu" => Ok(Box::new(LogitLink)),
            "phi" => Ok(Box::new(LogLink)),
            _ => Err(GamlssError::UnknownParameter {
                distribution: self.name().to_string(),
                param: param.to_string(),
            }),
        }
    }

    fn derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        // Beta log-likelihood with (mu, phi) parameterization:
        // alpha = mu * phi, beta = (1 - mu) * phi
        // l = log(Gamma(phi)) - log(Gamma(alpha)) - log(Gamma(beta))
        //     + (alpha - 1)*log(y) + (beta - 1)*log(1 - y)
        //
        // For mu (logit link, eta = logit(mu)):
        //   Score and Fisher info involve digamma/trigamma functions
        //
        // For phi (log link, eta = log(phi)):
        //   Similar derivation with digamma/trigamma
        let mu = *params
            .get("mu")
            .ok_or_else(|| GamlssError::UnknownParameter {
                distribution: self.name().to_string(),
                param: "mu".to_string(),
            })?;
        let phi = *params
            .get("phi")
            .ok_or_else(|| GamlssError::UnknownParameter {
                distribution: self.name().to_string(),
                param: "phi".to_string(),
            })?;

        let mu_safe = mu.mapv(|m| m.clamp(MIN_POSITIVE, 1.0 - MIN_POSITIVE));
        let phi_safe = phi.mapv(|p| p.max(MIN_POSITIVE));

        // Clamp y to valid range
        let y_clamped = y.mapv(|v| v.clamp(MIN_POSITIVE, 1.0 - MIN_POSITIVE));

        // Compute alpha = mu * phi, beta_param = (1 - mu) * phi
        let alpha = &mu_safe * &phi_safe;
        let one_minus_mu = mu_safe.mapv(|m| 1.0 - m);
        let beta_param = &one_minus_mu * &phi_safe;

        let log_y = y_clamped.mapv(|v| v.ln());
        let log_1_minus_y = y_clamped.mapv(|v| (1.0 - v).ln());

        // Batch digamma values
        let psi_alpha = digamma_batch(&alpha);
        let psi_beta = digamma_batch(&beta_param);
        let psi_phi = digamma_batch(&phi_safe);

        // Batch trigamma values
        let psi_prime_alpha = trigamma_batch(&alpha);
        let psi_prime_beta = trigamma_batch(&beta_param);
        let psi_prime_phi = trigamma_batch(&phi_safe);

        // mu derivatives (logit link)
        // dl/d_mu = phi * [log(y) - log(1-y) - digamma(alpha) + digamma(beta)]
        // For logit link: dl/d_eta = mu*(1-mu) * dl/d_mu
        let dl_dmu = &phi_safe * (&log_y - &log_1_minus_y - &psi_alpha + &psi_beta);
        let mu_1_minus_mu = &mu_safe * &one_minus_mu;
        let u_mu = &mu_1_minus_mu * &dl_dmu;

        // Fisher info for mu with logit link
        // I_mu = phi^2 * [trigamma(alpha) + trigamma(beta)]
        // For logit link: w_mu = [mu*(1-mu)]^2 * I_mu
        let phi_sq = phi_safe.mapv(|p| p.powi(2));
        let i_mu = &phi_sq * (&psi_prime_alpha + &psi_prime_beta);
        let mu_1_minus_mu_sq = mu_1_minus_mu.mapv(|v| v.powi(2));
        let w_mu = (&mu_1_minus_mu_sq * &i_mu).mapv(|v| v.max(MIN_WEIGHT));

        // phi derivatives (log link)
        // dl/d_phi = digamma(phi) - mu*digamma(alpha) - (1-mu)*digamma(beta)
        //            + mu*log(y) + (1-mu)*log(1-y)
        // For log link: dl/d_eta = phi * dl/d_phi
        let dl_dphi = &psi_phi - &mu_safe * &psi_alpha - &one_minus_mu * &psi_beta
            + &mu_safe * &log_y
            + &one_minus_mu * &log_1_minus_y;
        let u_phi = &phi_safe * &dl_dphi;

        // Fisher info for phi with log link
        // I_phi = trigamma(phi) - mu^2*trigamma(alpha) - (1-mu)^2*trigamma(beta)
        // For log link: w_phi = phi^2 * I_phi
        let mu_sq = mu_safe.mapv(|m| m.powi(2));
        let one_minus_mu_sq = one_minus_mu.mapv(|v| v.powi(2));
        let i_phi = &psi_prime_phi - &mu_sq * &psi_prime_alpha - &one_minus_mu_sq * &psi_prime_beta;
        let w_phi = (&phi_sq * &i_phi).mapv(|v| v.abs().max(MIN_WEIGHT));

        Ok(HashMap::from([
            ("mu".to_string(), (u_mu, w_mu)),
            ("phi".to_string(), (u_phi, w_phi)),
        ]))
    }

    fn name(&self) -> &'static str {
        "Beta"
    }
}

/// Binomial distribution for modeling count data with a known number of trials.
///
/// The response y represents the number of successes out of n trials.
/// The probability parameter mu uses a logit link by default.
#[derive(Debug, Clone)]
pub struct Binomial {
    /// Number of trials per observation (can be uniform or varying)
    n_trials: Array1<f64>,
}

impl Binomial {
    /// Create a Binomial distribution with a fixed number of trials for all observations.
    pub fn new(n_trials: usize) -> Self {
        Self {
            n_trials: Array1::from_elem(1, n_trials as f64),
        }
    }

    /// Create a Binomial distribution with varying number of trials per observation.
    pub fn with_trials(n_trials: Array1<f64>) -> Self {
        Self { n_trials }
    }

    /// Get the number of trials, broadcasting if necessary.
    fn get_n(&self, n_obs: usize) -> Array1<f64> {
        if self.n_trials.len() == 1 {
            Array1::from_elem(n_obs, self.n_trials[0])
        } else {
            self.n_trials.clone()
        }
    }
}

impl Distribution for Binomial {
    fn parameters(&self) -> &[&'static str] {
        &["mu"]
    }

    fn default_link(&self, param: &str) -> Result<Box<dyn Link>, GamlssError> {
        match param {
            "mu" => Ok(Box::new(LogitLink)),
            _ => Err(GamlssError::UnknownParameter {
                distribution: self.name().to_string(),
                param: param.to_string(),
            }),
        }
    }

    fn derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        // Binomial log-likelihood: l = y*log(mu) + (n-y)*log(1-mu) + log(C(n,y))
        //
        // Score (dl/dmu): (y - n*mu) / (mu*(1-mu))
        // Fisher info: n / (mu*(1-mu))
        //
        // With logit link η = logit(mu), dmu/deta = mu*(1-mu):
        //   Score on eta: u_eta = y - n*mu
        //   Fisher info on eta: w_eta = n * mu * (1-mu)
        let mu = *params
            .get("mu")
            .ok_or_else(|| GamlssError::UnknownParameter {
                distribution: self.name().to_string(),
                param: "mu".to_string(),
            })?;

        let n_obs = y.len();
        let n = self.get_n(n_obs);

        // Clamp mu to valid probability range
        let mu_safe = mu.mapv(|m| m.clamp(MIN_POSITIVE, 1.0 - MIN_POSITIVE));

        // Score on eta scale: u = y - n*mu
        let u_mu = y - &(&n * &mu_safe);

        // Fisher info on eta scale: w = n * mu * (1 - mu)
        let mu_1_minus_mu = &mu_safe * &mu_safe.mapv(|m| 1.0 - m);
        let w_mu = (&n * &mu_1_minus_mu).mapv(|v| v.max(MIN_WEIGHT));

        Ok(HashMap::from([("mu".to_string(), (u_mu, w_mu))]))
    }

    fn name(&self) -> &'static str {
        "Binomial"
    }

    fn initial_value(&self, param: &str, y: &Array1<f64>) -> f64 {
        match param {
            "mu" => {
                // y is counts, so divide by n to get probability
                let n = self.n_trials[0];
                let p = y.mean().unwrap_or(n / 2.0) / n;
                p.clamp(0.1, 0.9) // Avoid extreme starting values
            }
            _ => 0.1,
        }
    }
}
