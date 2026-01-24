use crate::error::GamlssError;
use crate::math::{digamma_batch, trigamma_batch};
use ndarray::Array1;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt::Debug;

/// Threshold for using parallel computation (below this, sequential is faster)
const PARALLEL_THRESHOLD: usize = 10_000;

// These traits help make sure the actual distributions are implemented correctly
pub trait Link: Debug + Send + Sync {
    fn link(&self, mu: f64) -> f64;
    fn inv_link(&self, eta: f64) -> f64;
}

// Concrete Links

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

#[derive(Debug, Clone, Copy, Default)]
pub struct LogLink;
impl Link for LogLink {
    fn link(&self, mu: f64) -> f64 {
        mu.ln().max(-30.0)
    }
    fn inv_link(&self, eta: f64) -> f64 {
        eta.min(30.0).exp()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LogitLink;
impl Link for LogitLink {
    fn link(&self, mu: f64) -> f64 {
        // logit(mu) = log(mu / (1 - mu))
        let mu_clamped = mu.clamp(1e-10, 1.0 - 1e-10);
        (mu_clamped / (1.0 - mu_clamped)).ln()
    }
    fn inv_link(&self, eta: f64) -> f64 {
        // inverse logit = 1 / (1 + exp(-eta))
        let eta_clamped = eta.clamp(-30.0, 30.0);
        1.0 / (1.0 + (-eta_clamped).exp())
    }
}

/// Result type for batched derivatives computation.
/// Maps parameter names to (score, Fisher info) array pairs.
pub type DerivativesResult = Result<HashMap<String, (Array1<f64>, Array1<f64>)>, GamlssError>;

pub trait Distribution: Debug + Send + Sync {
    fn parameters(&self) -> &[&'static str];
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
    fn name(&self) -> &'static str;
}

// Distributions
#[derive(Debug, Clone, Copy, Default)]
pub struct Poisson;
impl Poisson {
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

#[derive(Debug, Clone, Copy, Default)]
pub struct Gaussian;
impl Gaussian {
    pub fn new() -> Self {
        Self
    }
}

impl Distribution for Gaussian {
    // Gaussian has two parameters: mu and sigma. Mu is the mean, sigma is the standard deviation.
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

#[derive(Debug, Clone, Copy, Default)]
pub struct StudentT;
impl StudentT {
    pub fn new() -> Self {
        Self
    }
}
impl Distribution for StudentT {
    // StudentT has three parameters: mu, sigma and nu.
    // Mu is the mean, sigma is the standard deviation and nu is the degrees of freedom.
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

        let n = y.len();
        let z = (y - mu) / sigma;
        let z_sq = z.mapv(|v| v.powi(2));

        // w_robust = (nu+1)/(nu+z^2) appears in all derivatives.
        // This "robustifying weight" downweights outliers (large |z|).
        // As nu -> infinity, w_robust -> 1 and we recover Gaussian behavior.
        let w_robust: Array1<f64> = if n < PARALLEL_THRESHOLD {
            nu.iter()
                .zip(z_sq.iter())
                .map(|(&nu_i, &z2_i)| (nu_i + 1.0) / (nu_i + z2_i))
                .collect()
        } else {
            let nu_slice = nu.as_slice().unwrap();
            let z_sq_slice = z_sq.as_slice().unwrap();
            Array1::from_vec(
                nu_slice
                    .par_iter()
                    .zip(z_sq_slice.par_iter())
                    .map(|(&nu_i, &z2_i)| (nu_i + 1.0) / (nu_i + z2_i))
                    .collect(),
            )
        };

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
            let nu_slice = nu.as_slice().unwrap();
            let z_sq_slice = z_sq.as_slice().unwrap();
            let w_robust_slice = w_robust.as_slice().unwrap();

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

        let dl_dnu = 0.5 * (&d1 - &d2 - &term3 + &term4);

        // Chain rule for log link: u_eta = nu * dl/dnu
        let u_nu = &dl_dnu * nu;

        // Fisher information for nu involves trigamma functions (second derivative of log-gamma).
        // Use batch trigamma for vectorized computation
        let t1 = trigamma_batch(&nu_half);
        let t2 = trigamma_batch(&nu_plus_1_half);
        let t3: Array1<f64> = nu.mapv(|nu_i| (2.0 * (nu_i + 3.0)) / (nu_i * (nu_i + 1.0)));
        let i_nu = 0.25 * (&t1 - &t2 - &t3);
        // Floor at 1e-6 to ensure positive definiteness of the weight matrix.
        // For log link: W_eta = I_nu * nu^2
        let w_nu: Array1<f64> = if n < PARALLEL_THRESHOLD {
            i_nu.iter()
                .zip(nu.iter())
                .map(|(&i, &nu_i)| (i * nu_i.powi(2)).abs().max(1e-6))
                .collect()
        } else {
            let i_nu_slice = i_nu.as_slice().unwrap();
            let nu_slice = nu.as_slice().unwrap();
            Array1::from_vec(
                i_nu_slice
                    .par_iter()
                    .zip(nu_slice.par_iter())
                    .map(|(&i, &nu_i)| (i * nu_i.powi(2)).abs().max(1e-6))
                    .collect(),
            )
        };

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

// Gamma Distribution
// Parameterization: mu = mean, sigma = coefficient of variation (sqrt(Var/mu^2))
// Shape alpha = 1/sigma^2, Scale theta = mu * sigma^2
// Var(Y) = mu^2 * sigma^2
#[derive(Debug, Clone, Copy, Default)]
pub struct Gamma;

impl Gamma {
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

        let mu_safe = mu.mapv(|m| m.max(1e-10));
        let sigma_safe = sigma.mapv(|s| s.max(1e-10));
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
        let w_sigma =
            ((4.0 / &sigma_sq_sq) * &psi_prime_alpha - 2.0 / &sigma_sq).mapv(|v| v.abs().max(1e-6));

        Ok(HashMap::from([
            ("mu".to_string(), (u_mu, w_mu)),
            ("sigma".to_string(), (u_sigma, w_sigma)),
        ]))
    }

    fn name(&self) -> &'static str {
        "Gamma"
    }
}

// Negative Binomial Distribution (NB2 parameterization)
// Parameterization: mu = mean, sigma = overdispersion parameter
// Var(Y) = mu + sigma * mu^2
// When sigma -> 0, approaches Poisson
// size (r) = 1/sigma, prob p = 1/(1 + sigma*mu)
#[derive(Debug, Clone, Copy, Default)]
pub struct NegativeBinomial;

impl NegativeBinomial {
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

        let n = y.len();
        let mu_safe = mu.mapv(|m| m.max(1e-10));
        let sigma_safe = sigma.mapv(|s| s.max(1e-10));

        let one_plus_sigma_mu: Array1<f64> = if n < PARALLEL_THRESHOLD {
            sigma_safe
                .iter()
                .zip(mu_safe.iter())
                .map(|(&s, &m)| 1.0 + s * m)
                .collect()
        } else {
            let sigma_slice = sigma_safe.as_slice().unwrap();
            let mu_slice = mu_safe.as_slice().unwrap();
            Array1::from_vec(
                sigma_slice
                    .par_iter()
                    .zip(mu_slice.par_iter())
                    .map(|(&s, &m)| 1.0 + s * m)
                    .collect(),
            )
        };

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
        let w_sigma: Array1<f64> = (&psi_prime_r / &sigma_sq).mapv(|v| v.abs().max(1e-6));

        Ok(HashMap::from([
            ("mu".to_string(), (u_mu, w_mu)),
            ("sigma".to_string(), (u_sigma, w_sigma)),
        ]))
    }

    fn name(&self) -> &'static str {
        "NegativeBinomial"
    }
}

// Beta Distribution
// Parameterization: mu = mean (0 < mu < 1), phi = precision (phi > 0)
// Shape parameters: alpha = mu * phi, beta = (1 - mu) * phi
// Var(Y) = mu * (1 - mu) / (1 + phi)
#[derive(Debug, Clone, Copy, Default)]
pub struct Beta;

impl Beta {
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

        let mu_safe = mu.mapv(|m| m.clamp(1e-10, 1.0 - 1e-10));
        let phi_safe = phi.mapv(|p| p.max(1e-10));

        // Clamp y to valid range
        let y_clamped = y.mapv(|v| v.clamp(1e-10, 1.0 - 1e-10));

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
        let w_mu = (&mu_1_minus_mu_sq * &i_mu).mapv(|v| v.max(1e-6));

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
        let w_phi = (&phi_sq * &i_phi).mapv(|v| v.abs().max(1e-6));

        Ok(HashMap::from([
            ("mu".to_string(), (u_mu, w_mu)),
            ("phi".to_string(), (u_phi, w_phi)),
        ]))
    }

    fn name(&self) -> &'static str {
        "Beta"
    }
}

// TODO: Add Binomial distribution (will need special handling for n parameter)
