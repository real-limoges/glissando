//! Probability distributions for GAMLSS models.
//!
//! Each distribution defines its parameter names (μ, σ, ν, …), default link functions,
//! and the score / Fisher-information pairs that drive the Rigby–Stasinopoulos
//! IRLS update. Derivatives are batched (vectorized over observations).

use crate::error::GamlssError;
use crate::math::{digamma_batch, par_zip3_map, par_zip_map, trigamma_batch};
use ndarray::Array1;
use statrs::function::gamma::ln_gamma;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::Debug;

/// Floor for positive parameters (μ, σ, …) to avoid log(0) or division by zero.
const MIN_POSITIVE: f64 = 1e-10;
/// Linear-predictor ceiling for log/logit links (prevents `exp` overflow).
const MAX_ETA: f64 = 30.0;
/// Linear-predictor floor for log/logit links (prevents `exp` underflow).
const MIN_ETA: f64 = -30.0;
/// Lower bound on Fisher-information weights to keep `W` positive definite.
const MIN_WEIGHT: f64 = 1e-6;

// ============================================================================
// Link functions
// ============================================================================

/// A link function `g` mapping the response-scale parameter `μ` to the linear predictor `η = g(μ)`.
pub trait Link: Debug + Send + Sync {
    /// Apply the link: `η = g(μ)`.
    fn link(&self, mu: f64) -> f64;
    /// Apply the inverse link: `μ = g⁻¹(η)`.
    fn inv_link(&self, eta: f64) -> f64;
}

/// Identity link: `η = μ`. Used for unbounded continuous parameters (e.g. Gaussian mean).
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

/// Log link: `η = log(μ)`. Used for positive parameters (Poisson rate, Gamma mean).
///
/// `link` clamps `log(μ)` to [`MIN_ETA`]; `inv_link` clamps `η` to [`MAX_ETA`].
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

/// Logit link: `η = log(μ / (1 - μ))`. Used for `(0, 1)` probability parameters.
#[derive(Debug, Clone, Copy, Default)]
pub struct LogitLink;

impl Link for LogitLink {
    fn link(&self, mu: f64) -> f64 {
        let m = mu.clamp(MIN_POSITIVE, 1.0 - MIN_POSITIVE);
        (m / (1.0 - m)).ln()
    }
    fn inv_link(&self, eta: f64) -> f64 {
        let e = eta.clamp(MIN_ETA, MAX_ETA);
        1.0 / (1.0 + (-e).exp())
    }
}

// ============================================================================
// Distribution trait
// ============================================================================

/// Score / Fisher-information pairs keyed by distribution-parameter name.
pub type DerivativesResult = Result<HashMap<String, (Array1<f64>, Array1<f64>)>, GamlssError>;

/// A statistical distribution for GAMLSS, defining parameters, link functions, and
/// score / Fisher-information pairs that drive the IRLS algorithm.
pub trait Distribution: Debug + Send + Sync {
    /// Distribution-parameter names (e.g. `["mu", "sigma"]`).
    fn parameters(&self) -> &[&'static str];

    /// Default link function for the named parameter.
    ///
    /// # Errors
    ///
    /// Returns `GamlssError::UnknownParameter` if the name is not one of [`Self::parameters`].
    fn default_link(&self, param: &str) -> Result<Box<dyn Link>, GamlssError>;

    /// Score (`u`) and Fisher information (`w`) on the linear-predictor scale, for each
    /// parameter, evaluated against the current `params` snapshot.
    fn derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult;

    /// Per-observation log-density `log f(y_i | params_i)`, used to assemble the model
    /// log-likelihood and observation-level diagnostics (WAIC, leave-one-out, etc.).
    fn loglik_pointwise(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError>;

    /// Marginal `Var(Y_i | params_i)` on the response scale, used for Pearson residuals.
    ///
    /// Distinct from the Fisher-information weight returned by [`Self::derivatives`],
    /// which is on the linear-predictor scale.
    fn variance(&self, params: &HashMap<&str, &Array1<f64>>) -> Result<Array1<f64>, GamlssError>;

    /// Marginal `E[Y_i | params_i]` on the response scale.
    ///
    /// Default returns `params["mu"]` cloned. Distributions where `mu` is not the
    /// expected value of `Y` (e.g. [`Binomial`] where `E[Y] = n·μ`) override.
    fn expected_value(
        &self,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        Ok(require(self, params, "mu")?.to_owned())
    }

    /// Total model log-likelihood: `Σ log f(y_i | params_i)`.
    fn loglik(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<f64, GamlssError> {
        Ok(self.loglik_pointwise(y, params)?.sum())
    }

    /// Stable distribution name (e.g. `"Gaussian"`); used in error messages and
    /// for the WASM `from_name` lookup.
    fn name(&self) -> &'static str;

    /// Initial response-scale value for a parameter, used to seed the IRLS loop.
    /// Override for distributions where `y` is not directly a sample of the parameter.
    fn initial_value(&self, param: &str, y: &Array1<f64>) -> f64 {
        // `validate_inputs` rejects empty `y` before fitting, so `mean` returning `None`
        // is unreachable on the public path. The `unwrap_or` below keeps the fn pure.
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

    /// Build a fresh `UnknownParameter` error tagged with this distribution's name.
    fn unknown_param(&self, param: &str) -> GamlssError {
        GamlssError::UnknownParameter {
            distribution: self.name().to_string(),
            param: param.to_string(),
        }
    }
}

/// Look up `param` in a derivatives-input map or yield an `UnknownParameter` error.
fn require<'a, D: Distribution + ?Sized>(
    dist: &D,
    params: &HashMap<&str, &'a Array1<f64>>,
    name: &str,
) -> Result<&'a Array1<f64>, GamlssError> {
    params
        .get(name)
        .copied()
        .ok_or_else(|| dist.unknown_param(name))
}

/// Construct a stateless distribution from its name (e.g. for WASM JSON I/O).
///
/// Excludes [`Binomial`] because it requires `n_trials` state that cannot be recovered
/// from the name alone.
///
/// # Errors
///
/// Returns `GamlssError::Input` if `name` does not match a supported stateless distribution.
///
/// # Examples
///
/// ```
/// use glissando::distributions::{from_name, Distribution};
///
/// let d = from_name("Gaussian").unwrap();
/// assert_eq!(d.name(), "Gaussian");
/// assert_eq!(d.parameters(), &["mu", "sigma"]);
///
/// assert!(from_name("Wishart").is_err());
/// ```
pub fn from_name(name: &str) -> Result<Box<dyn Distribution>, GamlssError> {
    match name {
        "Gaussian" => Ok(Box::new(Gaussian)),
        "Poisson" => Ok(Box::new(Poisson)),
        "StudentT" => Ok(Box::new(StudentT)),
        "Gamma" => Ok(Box::new(Gamma)),
        "NegativeBinomial" => Ok(Box::new(NegativeBinomial)),
        "Beta" => Ok(Box::new(Beta)),
        other => Err(GamlssError::Input(format!(
            "Unknown distribution: '{}'. Supported: Gaussian, Poisson, StudentT, Gamma, NegativeBinomial, Beta",
            other
        ))),
    }
}

// ============================================================================
// Poisson
// ============================================================================

/// Poisson distribution for count data.
///
/// Single parameter `μ` (mean / rate) with log link. `Var(Y) = μ`.
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
            other => Err(self.unknown_param(other)),
        }
    }

    fn derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        // Log-likelihood: l = y·log(μ) − μ.
        // Score on η = log(μ): u = y − μ.   Fisher info: w = μ.
        let mu = require(self, params, "mu")?;
        let u = y - mu;
        let w = mu.to_owned();
        Ok(HashMap::from([("mu".to_string(), (u, w))]))
    }

    fn loglik_pointwise(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        Ok(par_zip_map(y, mu, |yi, mui| {
            yi * mui.max(MIN_POSITIVE).ln() - mui - ln_gamma(yi + 1.0)
        }))
    }

    fn variance(&self, params: &HashMap<&str, &Array1<f64>>) -> Result<Array1<f64>, GamlssError> {
        Ok(require(self, params, "mu")?.to_owned())
    }

    fn name(&self) -> &'static str {
        "Poisson"
    }
}

// ============================================================================
// Gaussian
// ============================================================================

/// Gaussian (Normal) distribution.
///
/// Parameters: `μ` (mean, identity link) and `σ` (std dev, log link). `Var(Y) = σ²`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Gaussian;

impl Gaussian {
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
            other => Err(self.unknown_param(other)),
        }
    }

    fn derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        // Gaussian log-likelihood:  l = −0.5·log(2π) − log(σ) − (y−μ)²/(2σ²).
        //   μ (identity link):  u = (y−μ)/σ²,                w = 1/σ².
        //   σ (log link, η = log σ):  u = ((y−μ)² − σ²)/σ²,  w = 2.
        // Full derivation in docs/mathematics.md.
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;

        let sigma_sq = sigma.mapv(|s| s * s);
        let residual = y - mu;
        let residual_sq = residual.mapv(|r| r * r);

        let u_mu = &residual / &sigma_sq;
        let w_mu = sigma_sq.mapv(|s2| 1.0 / s2);

        let u_sigma = (&residual_sq - &sigma_sq) / &sigma_sq;
        let w_sigma = Array1::from_elem(y.len(), 2.0);

        Ok(HashMap::from([
            ("mu".to_string(), (u_mu, w_mu)),
            ("sigma".to_string(), (u_sigma, w_sigma)),
        ]))
    }

    fn loglik_pointwise(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        let log_2pi = (2.0 * std::f64::consts::PI).ln();
        Ok(par_zip3_map(y, mu, sigma, |yi, mui, si| {
            let s = si.max(MIN_POSITIVE);
            let z = (yi - mui) / s;
            -0.5 * log_2pi - s.ln() - 0.5 * z * z
        }))
    }

    fn variance(&self, params: &HashMap<&str, &Array1<f64>>) -> Result<Array1<f64>, GamlssError> {
        let sigma = require(self, params, "sigma")?;
        Ok(sigma.mapv(|s| s * s))
    }

    fn name(&self) -> &'static str {
        "Gaussian"
    }
}

// ============================================================================
// Student's t
// ============================================================================

/// Student's t distribution for heavy-tailed continuous data.
///
/// Parameters: `μ` (location, identity), `σ` (scale, log), `ν` (degrees of freedom, log).
/// As `ν → ∞` the distribution approaches Gaussian.
#[derive(Debug, Clone, Copy, Default)]
pub struct StudentT;

impl StudentT {
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
            "sigma" | "nu" => Ok(Box::new(LogLink)),
            other => Err(self.unknown_param(other)),
        }
    }

    fn derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        // Student-t log-likelihood, location-scale parameterization. Full derivation
        // in docs/mathematics.md.
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        let nu = require(self, params, "nu")?;

        let z = (y - mu) / sigma;
        let z_sq = z.mapv(|v| v * v);

        // The "robustifying weight" w = (ν+1)/(ν+z²) downweights outliers (large |z|).
        // It → 1 as ν → ∞, recovering Gaussian behavior.
        let w_robust = par_zip_map(nu, &z_sq, |nu_i, z2_i| (nu_i + 1.0) / (nu_i + z2_i));

        // μ derivatives (identity link).
        let u_mu = (&w_robust * &z) / sigma;
        let w_mu = &w_robust / sigma.mapv(|s| s * s);

        // σ derivatives (log link). Chain rule: dl/dη = σ · dl/dσ = w·z² − 1.
        let u_sigma = &w_robust * &z_sq - 1.0;
        let w_sigma: Array1<f64> = nu.mapv(|nu_i| (2.0 * nu_i) / (nu_i + 3.0));

        // ν derivatives (log link). Score involves digamma differences.
        let nu_plus_1_half = nu.mapv(|nu_i| (nu_i + 1.0) / 2.0);
        let nu_half = nu.mapv(|nu_i| nu_i / 2.0);
        let d1 = digamma_batch(&nu_plus_1_half);
        let d2 = digamma_batch(&nu_half);

        let term3 = par_zip_map(nu, &z_sq, |nu_i, z2_i| (1.0 + z2_i / nu_i).ln());
        let term4 = par_zip3_map(nu, &w_robust, &z_sq, |nu_i, w_i, z2_i| {
            (w_i * z2_i - 1.0) / nu_i
        });

        let dl_dnu = 0.5 * (&d1 - &d2 - &term3 + &term4);
        // Chain rule for log link: u_η = ν · dl/dν.
        let u_nu = &dl_dnu * nu;

        // Fisher information for ν uses trigamma (the second derivative of log-Γ).
        let t1 = trigamma_batch(&nu_half);
        let t2 = trigamma_batch(&nu_plus_1_half);
        let t3: Array1<f64> = nu.mapv(|nu_i| (2.0 * (nu_i + 3.0)) / (nu_i * (nu_i + 1.0)));
        // The `+ t3` term subtracts from the negative Hessian — sign is correct.
        let i_nu = 0.25 * (&t1 - &t2 + &t3);
        // For log link `W_η = I_ν · ν²`, floored to keep the weight matrix positive definite.
        let w_nu = par_zip_map(&i_nu, nu, |i, nu_i| (i * nu_i * nu_i).abs().max(MIN_WEIGHT));

        Ok(HashMap::from([
            ("mu".to_string(), (u_mu, w_mu)),
            ("sigma".to_string(), (u_sigma, w_sigma)),
            ("nu".to_string(), (u_nu, w_nu)),
        ]))
    }

    fn loglik_pointwise(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        let nu = require(self, params, "nu")?;
        let n = y.len();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let s = sigma[i].max(MIN_POSITIVE);
            let nu_i = nu[i].max(MIN_POSITIVE);
            let z = (y[i] - mu[i]) / s;
            out[i] = ln_gamma((nu_i + 1.0) / 2.0)
                - ln_gamma(nu_i / 2.0)
                - 0.5 * (std::f64::consts::PI * nu_i).ln()
                - s.ln()
                - 0.5 * (nu_i + 1.0) * (1.0 + z * z / nu_i).ln();
        }
        Ok(out)
    }

    /// `Var(Y) = σ²·ν/(ν−2)` for `ν > 2`. For `ν ≤ 2` the variance is undefined; we
    /// clamp the denominator at `MIN_POSITIVE` so Pearson residuals stay finite.
    fn variance(&self, params: &HashMap<&str, &Array1<f64>>) -> Result<Array1<f64>, GamlssError> {
        let sigma = require(self, params, "sigma")?;
        let nu = require(self, params, "nu")?;
        Ok(par_zip_map(sigma, nu, |s, nu_i| {
            let denom = (nu_i - 2.0).max(MIN_POSITIVE);
            s * s * nu_i / denom
        }))
    }

    fn name(&self) -> &'static str {
        "StudentT"
    }
}

// ============================================================================
// Gamma
// ============================================================================

/// Gamma distribution for positive continuous data.
///
/// Parameters: `μ` (mean, log link) and `σ` (coefficient of variation, log link).
/// Parameterization: shape `α = 1/σ²`, scale `θ = μσ²`. `Var(Y) = μ²σ²`.
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
            "mu" | "sigma" => Ok(Box::new(LogLink)),
            other => Err(self.unknown_param(other)),
        }
    }

    fn derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        // Gamma (μ, σ) parameterization: α = 1/σ², θ = μσ².
        // l = −α·log(θ) − log Γ(α) + (α−1)·log(y) − y/θ.
        // μ (log link, η = log μ):  u = (y−μ)/(μσ²),  w = 1/σ².
        // σ (log link, η = log σ):  u = (2/σ²)·[ψ(α) + 2 log σ − log(y/μ) + y/μ − 1].
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;

        let mu_safe = mu.mapv(|m| m.max(MIN_POSITIVE));
        let sigma_safe = sigma.mapv(|s| s.max(MIN_POSITIVE));
        let sigma_sq = sigma_safe.mapv(|s| s * s);
        let alpha = sigma_sq.mapv(|s2| 1.0 / s2);

        let u_mu = (y - &mu_safe) / (&mu_safe * &sigma_sq);
        let w_mu = sigma_sq.mapv(|s2| 1.0 / s2);

        let psi_alpha = digamma_batch(&alpha);
        let log_sigma = sigma_safe.mapv(|s| s.ln());
        let y_over_mu = y / &mu_safe;
        let log_y_over_mu = y_over_mu.mapv(|v| v.ln());
        let u_sigma =
            (2.0 / &sigma_sq) * (&psi_alpha + 2.0 * &log_sigma - &log_y_over_mu + &y_over_mu - 1.0);

        // Fisher info for σ: I_σ = (4/σ⁴)·ψ'(α) − 2/σ². Floored at MIN_WEIGHT.
        let psi_prime_alpha = trigamma_batch(&alpha);
        let sigma_sq_sq = sigma_sq.mapv(|s2| s2 * s2);
        let w_sigma = ((4.0 / &sigma_sq_sq) * &psi_prime_alpha - 2.0 / &sigma_sq)
            .mapv(|v| v.abs().max(MIN_WEIGHT));

        Ok(HashMap::from([
            ("mu".to_string(), (u_mu, w_mu)),
            ("sigma".to_string(), (u_sigma, w_sigma)),
        ]))
    }

    fn loglik_pointwise(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        Ok(par_zip3_map(y, mu, sigma, |yi, mui, si| {
            let s = si.max(MIN_POSITIVE);
            let alpha = 1.0 / (s * s);
            let theta = mui * s * s;
            (alpha - 1.0) * yi.max(MIN_POSITIVE).ln()
                - yi / theta
                - alpha * theta.ln()
                - ln_gamma(alpha)
        }))
    }

    fn variance(&self, params: &HashMap<&str, &Array1<f64>>) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        Ok(par_zip_map(mu, sigma, |m, s| m * m * s * s))
    }

    fn name(&self) -> &'static str {
        "Gamma"
    }
}

// ============================================================================
// Negative Binomial (NB2)
// ============================================================================

/// Negative Binomial distribution (NB2) for overdispersed count data.
///
/// Parameters: `μ` (mean, log link) and `σ` (overdispersion, log link).
/// `Var(Y) = μ + σμ²`. As `σ → 0` it approaches Poisson.
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
            "mu" | "sigma" => Ok(Box::new(LogLink)),
            other => Err(self.unknown_param(other)),
        }
    }

    fn derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        // NB2 log-likelihood:
        //   l = log Γ(y + 1/σ) − log Γ(1/σ) − log y!
        //       + (1/σ)·log(1/(1+σμ)) + y·log(σμ/(1+σμ)).
        // μ (log link):  u = (y−μ)/(1+σμ),  w = μ/(1+σμ).
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;

        let mu_safe = mu.mapv(|m| m.max(MIN_POSITIVE));
        let sigma_safe = sigma.mapv(|s| s.max(MIN_POSITIVE));
        let one_plus_sigma_mu = par_zip_map(&sigma_safe, &mu_safe, |s, m| 1.0 + s * m);

        let u_mu = (y - &mu_safe) / &one_plus_sigma_mu;
        let w_mu = &mu_safe / &one_plus_sigma_mu;

        // σ (log link, r = 1/σ):
        //   dl/dr = ψ(y+r) − ψ(r) − log(1+σμ) + (μ−y)/(r+μ)
        //   dl/dσ = −(1/σ²)·dl/dr,   dl/dη = σ·dl/dσ = −(1/σ)·dl/dr.
        let r = sigma_safe.mapv(|s| 1.0 / s);
        let y_plus_r = y + &r;
        let psi_y_r = digamma_batch(&y_plus_r);
        let psi_r = digamma_batch(&r);
        let log_term = one_plus_sigma_mu.mapv(|v| v.ln());
        let r_plus_mu = &r + &mu_safe;
        let ratio_term = (&mu_safe - y) / &r_plus_mu;

        let u_sigma = (-1.0 / &sigma_safe) * (&psi_y_r - &psi_r - &log_term + &ratio_term);

        // Fisher info for σ ≈ ψ'(r)/σ², floored at MIN_WEIGHT.
        let psi_prime_r = trigamma_batch(&r);
        let sigma_sq = sigma_safe.mapv(|s| s * s);
        let w_sigma = (&psi_prime_r / &sigma_sq).mapv(|v| v.abs().max(MIN_WEIGHT));

        Ok(HashMap::from([
            ("mu".to_string(), (u_mu, w_mu)),
            ("sigma".to_string(), (u_sigma, w_sigma)),
        ]))
    }

    fn loglik_pointwise(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        Ok(par_zip3_map(y, mu, sigma, |yi, mui, si| {
            let r = 1.0 / si.max(MIN_POSITIVE);
            let p = r / (r + mui);
            ln_gamma(yi + r) - ln_gamma(r) - ln_gamma(yi + 1.0)
                + r * p.max(MIN_POSITIVE).ln()
                + yi * (1.0 - p).max(MIN_POSITIVE).ln()
        }))
    }

    fn variance(&self, params: &HashMap<&str, &Array1<f64>>) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        Ok(par_zip_map(mu, sigma, |m, s| m + s * m * m))
    }

    fn name(&self) -> &'static str {
        "NegativeBinomial"
    }
}

// ============================================================================
// Beta
// ============================================================================

/// Beta distribution for proportions on `(0, 1)`.
///
/// Parameters: `μ` (mean, logit link) and `φ` (precision, log link).
/// Shape `α = μφ`, `β = (1−μ)φ`. `Var(Y) = μ(1−μ)/(1+φ)`.
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
            other => Err(self.unknown_param(other)),
        }
    }

    fn derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        // Beta (μ, φ) parameterization: α = μφ, β = (1−μ)φ.
        // l = log Γ(φ) − log Γ(α) − log Γ(β) + (α−1)·log(y) + (β−1)·log(1−y).
        let mu = require(self, params, "mu")?;
        let phi = require(self, params, "phi")?;

        let mu_safe = mu.mapv(|m| m.clamp(MIN_POSITIVE, 1.0 - MIN_POSITIVE));
        let phi_safe = phi.mapv(|p| p.max(MIN_POSITIVE));
        let y_clamped = y.mapv(|v| v.clamp(MIN_POSITIVE, 1.0 - MIN_POSITIVE));

        let one_minus_mu = mu_safe.mapv(|m| 1.0 - m);
        let alpha = &mu_safe * &phi_safe;
        let beta_param = &one_minus_mu * &phi_safe;

        let log_y = y_clamped.mapv(|v| v.ln());
        let log_1_minus_y = y_clamped.mapv(|v| (1.0 - v).ln());

        let psi_alpha = digamma_batch(&alpha);
        let psi_beta = digamma_batch(&beta_param);
        let psi_phi = digamma_batch(&phi_safe);
        let psi_prime_alpha = trigamma_batch(&alpha);
        let psi_prime_beta = trigamma_batch(&beta_param);
        let psi_prime_phi = trigamma_batch(&phi_safe);

        // μ (logit link). dl/dμ = φ·[log(y) − log(1−y) − ψ(α) + ψ(β)].
        // Chain rule: dl/dη = μ(1−μ)·dl/dμ.
        let dl_dmu = &phi_safe * (&log_y - &log_1_minus_y - &psi_alpha + &psi_beta);
        let mu_1_minus_mu = &mu_safe * &one_minus_mu;
        let u_mu = &mu_1_minus_mu * &dl_dmu;

        // Fisher info for μ on η-scale: w = (μ(1−μ))² · φ²·(ψ'(α) + ψ'(β)).
        let phi_sq = phi_safe.mapv(|p| p * p);
        let i_mu = &phi_sq * (&psi_prime_alpha + &psi_prime_beta);
        let mu_1_minus_mu_sq = mu_1_minus_mu.mapv(|v| v * v);
        let w_mu = (&mu_1_minus_mu_sq * &i_mu).mapv(|v| v.max(MIN_WEIGHT));

        // φ (log link). dl/dφ = ψ(φ) − μ·ψ(α) − (1−μ)·ψ(β) + μ·log(y) + (1−μ)·log(1−y).
        // Chain rule: dl/dη = φ · dl/dφ.
        let dl_dphi = &psi_phi - &mu_safe * &psi_alpha - &one_minus_mu * &psi_beta
            + &mu_safe * &log_y
            + &one_minus_mu * &log_1_minus_y;
        let u_phi = &phi_safe * &dl_dphi;

        // Fisher info for φ on η-scale: w = φ²·(ψ'(φ) − μ²·ψ'(α) − (1−μ)²·ψ'(β)).
        let mu_sq = mu_safe.mapv(|m| m * m);
        let one_minus_mu_sq = one_minus_mu.mapv(|v| v * v);
        let i_phi = &psi_prime_phi - &mu_sq * &psi_prime_alpha - &one_minus_mu_sq * &psi_prime_beta;
        let w_phi = (&phi_sq * &i_phi).mapv(|v| v.abs().max(MIN_WEIGHT));

        Ok(HashMap::from([
            ("mu".to_string(), (u_mu, w_mu)),
            ("phi".to_string(), (u_phi, w_phi)),
        ]))
    }

    fn loglik_pointwise(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let phi = require(self, params, "phi")?;
        Ok(par_zip3_map(y, mu, phi, |yi, mui, phii| {
            let alpha = mui * phii;
            let beta = (1.0 - mui) * phii;
            let yc = yi.clamp(MIN_POSITIVE, 1.0 - MIN_POSITIVE);
            ln_gamma(phii) - ln_gamma(alpha) - ln_gamma(beta)
                + (alpha - 1.0) * yc.ln()
                + (beta - 1.0) * (1.0 - yc).ln()
        }))
    }

    fn variance(&self, params: &HashMap<&str, &Array1<f64>>) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let phi = require(self, params, "phi")?;
        Ok(par_zip_map(mu, phi, |m, p| {
            let m_clamped = m.clamp(MIN_POSITIVE, 1.0 - MIN_POSITIVE);
            m_clamped * (1.0 - m_clamped) / (1.0 + p.max(MIN_POSITIVE))
        }))
    }

    fn name(&self) -> &'static str {
        "Beta"
    }
}

// ============================================================================
// Binomial
// ============================================================================

/// Binomial distribution: response `y` is the count of successes out of `n` trials.
///
/// Single parameter `μ` ∈ `(0, 1)` (success probability) with logit link.
#[derive(Debug, Clone)]
pub struct Binomial {
    /// Trials per observation. Length 1 broadcasts; otherwise must match `y.len()`.
    n_trials: Array1<f64>,
}

impl Binomial {
    /// Construct a Binomial with a constant number of trials shared across observations.
    pub fn new(n_trials: usize) -> Self {
        Self {
            n_trials: Array1::from_elem(1, n_trials as f64),
        }
    }

    /// Construct a Binomial with per-observation trial counts.
    pub fn with_trials(n_trials: Array1<f64>) -> Self {
        Self { n_trials }
    }

    /// Returns trials broadcast to `n_obs`. Borrows when length already matches; otherwise allocates.
    fn trials(&self, n_obs: usize) -> Cow<'_, Array1<f64>> {
        if self.n_trials.len() == 1 {
            Cow::Owned(Array1::from_elem(n_obs, self.n_trials[0]))
        } else {
            Cow::Borrowed(&self.n_trials)
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
            other => Err(self.unknown_param(other)),
        }
    }

    fn derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        // Binomial log-likelihood: l = y·log(μ) + (n−y)·log(1−μ) + log C(n, y).
        // With logit link η = logit(μ) and dμ/dη = μ(1−μ):
        //   u_η = y − n·μ,    w_η = n·μ·(1−μ)  (floored at MIN_WEIGHT).
        let mu = require(self, params, "mu")?;
        let n = self.trials(y.len());

        let mu_safe = mu.mapv(|m| m.clamp(MIN_POSITIVE, 1.0 - MIN_POSITIVE));
        let u_mu = y - &(n.as_ref() * &mu_safe);

        let mu_1_minus_mu = &mu_safe * &mu_safe.mapv(|m| 1.0 - m);
        let w_mu = (n.as_ref() * &mu_1_minus_mu).mapv(|v| v.max(MIN_WEIGHT));

        Ok(HashMap::from([("mu".to_string(), (u_mu, w_mu))]))
    }

    fn loglik_pointwise(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let n = self.trials(y.len());
        Ok(par_zip3_map(y, mu, n.as_ref(), |yi, mui, ni| {
            let m = mui.clamp(MIN_POSITIVE, 1.0 - MIN_POSITIVE);
            ln_gamma(ni + 1.0) - ln_gamma(yi + 1.0) - ln_gamma(ni - yi + 1.0)
                + yi * m.ln()
                + (ni - yi) * (1.0 - m).ln()
        }))
    }

    fn variance(&self, params: &HashMap<&str, &Array1<f64>>) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let n = self.trials(mu.len());
        Ok(par_zip_map(n.as_ref(), mu, |ni, mi| {
            let m = mi.clamp(MIN_POSITIVE, 1.0 - MIN_POSITIVE);
            ni * m * (1.0 - m)
        }))
    }

    fn expected_value(
        &self,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let n = self.trials(mu.len());
        Ok(n.as_ref() * mu)
    }

    fn name(&self) -> &'static str {
        "Binomial"
    }

    fn initial_value(&self, param: &str, y: &Array1<f64>) -> f64 {
        match param {
            "mu" => {
                // y is counts; convert to a probability via the first trial count.
                // `validate_inputs` rejects empty `y`, so the `unwrap_or` is unreachable
                // on the public path.
                let n = self.n_trials[0];
                let p = y.mean().unwrap_or(n / 2.0) / n;
                // Clamp away from {0, 1} so the IRLS loop has a well-conditioned start.
                p.clamp(0.1, 0.9)
            }
            _ => 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;
    #[cfg(not(target_arch = "wasm32"))]
    use proptest::prelude::*;

    // --- Links ---

    #[test]
    fn identity_link_is_noop() {
        let l = IdentityLink;
        assert_eq!(l.link(0.0), 0.0);
        assert_eq!(l.link(3.5), 3.5);
        assert_eq!(l.inv_link(-2.0), -2.0);
    }

    #[test]
    fn log_link_clamps_underflow_and_overflow() {
        let l = LogLink;
        assert!(l.link(0.0).is_finite());
        assert!(l.link(0.0) <= MIN_ETA + 1e-9);
        assert!(l.inv_link(1e6).is_finite());
        assert!(l.inv_link(1e6) <= MAX_ETA.exp() + 1.0);
    }

    #[test]
    fn logit_link_handles_boundary_probabilities() {
        let l = LogitLink;
        assert!(l.link(0.0).is_finite());
        assert!(l.link(1.0).is_finite());
        assert!((0.0..=1.0).contains(&l.inv_link(-1e6)));
        assert!((0.0..=1.0).contains(&l.inv_link(1e6)));
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn close(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps * (1.0 + a.abs().max(b.abs()))
    }

    #[cfg(not(target_arch = "wasm32"))]
    proptest! {
        #[test]
        fn identity_link_round_trip(x in -1e6f64..1e6) {
            let l = IdentityLink;
            prop_assert!(close(l.link(l.inv_link(x)), x, 1e-9));
        }

        #[test]
        fn log_link_round_trip_in_safe_range(eta in (MIN_ETA + 1.0) .. (MAX_ETA - 1.0)) {
            let l = LogLink;
            prop_assert!(close(l.link(l.inv_link(eta)), eta, 1e-9));
        }

        #[test]
        fn logit_link_round_trip_in_safe_range(eta in -20.0f64..20.0) {
            let l = LogitLink;
            prop_assert!(close(l.link(l.inv_link(eta)), eta, 1e-6));
        }

        #[test]
        fn log_link_inv_link_strictly_positive(eta in -50.0f64..50.0) {
            let l = LogLink;
            prop_assert!(l.inv_link(eta) > 0.0);
        }

        #[test]
        fn logit_link_inv_link_in_unit_interval(eta in -1e6f64..1e6) {
            let l = LogitLink;
            let p = l.inv_link(eta);
            prop_assert!((0.0..=1.0).contains(&p));
        }
    }

    // --- from_name ---

    #[test]
    fn from_name_returns_expected_variants() {
        for name in &[
            "Gaussian",
            "Poisson",
            "StudentT",
            "Gamma",
            "NegativeBinomial",
            "Beta",
        ] {
            let d = from_name(name).unwrap();
            assert_eq!(d.name(), *name);
        }
    }

    #[test]
    fn from_name_unknown_returns_input_error() {
        let err = from_name("Wishart").unwrap_err();
        assert!(matches!(err, GamlssError::Input(_)));
    }

    // --- Per-distribution derivatives shape + finiteness ---

    fn finite_array(a: &Array1<f64>) -> bool {
        a.iter().all(|v| v.is_finite())
    }

    fn derivative_keys_match_parameters<D: Distribution>(
        d: &D,
        params: HashMap<&str, &Array1<f64>>,
        y: &Array1<f64>,
    ) {
        let derivs = d.derivatives(y, &params).unwrap();
        let mut keys: Vec<&str> = derivs.keys().map(String::as_str).collect();
        keys.sort();
        let mut expected: Vec<&str> = d.parameters().to_vec();
        expected.sort();
        assert_eq!(keys, expected);
        for (u, w) in derivs.values() {
            assert_eq!(u.len(), y.len());
            assert_eq!(w.len(), y.len());
            assert!(finite_array(u));
            assert!(finite_array(w));
            // Fisher info should be non-negative.
            assert!(w.iter().all(|&v| v >= 0.0));
        }
    }

    #[test]
    fn poisson_derivatives() {
        let y = array![0.0, 1.0, 5.0, 10.0];
        let mu = array![1.0, 2.0, 4.0, 9.0];
        let mut p = HashMap::new();
        p.insert("mu", &mu);
        derivative_keys_match_parameters(&Poisson, p, &y);
    }

    #[test]
    fn poisson_score_zero_when_y_equals_mu() {
        let y = array![1.0, 2.0, 4.0];
        let mu = y.clone();
        let mut p = HashMap::new();
        p.insert("mu", &mu);
        let derivs = Poisson.derivatives(&y, &p).unwrap();
        let (u, _) = &derivs["mu"];
        assert!(u.iter().all(|&v| v.abs() < 1e-12));
    }

    #[test]
    fn poisson_unknown_parameter_errors() {
        let y = array![1.0];
        let p: HashMap<&str, &Array1<f64>> = HashMap::new();
        let err = Poisson.derivatives(&y, &p).unwrap_err();
        assert!(matches!(err, GamlssError::UnknownParameter { .. }));
    }

    #[test]
    fn gaussian_derivatives() {
        let y = array![0.0, 1.0, -1.0, 2.5];
        let mu = array![0.0, 0.5, -0.5, 2.0];
        let sigma = array![1.0, 1.0, 2.0, 0.5];
        let mut p = HashMap::new();
        p.insert("mu", &mu);
        p.insert("sigma", &sigma);
        derivative_keys_match_parameters(&Gaussian, p, &y);
    }

    #[test]
    fn gaussian_mu_score_zero_when_y_equals_mu() {
        let y = array![0.5, 1.5, -2.0];
        let mu = y.clone();
        let sigma = array![1.0, 1.0, 1.0];
        let mut p = HashMap::new();
        p.insert("mu", &mu);
        p.insert("sigma", &sigma);
        let derivs = Gaussian.derivatives(&y, &p).unwrap();
        let (u_mu, w_mu) = &derivs["mu"];
        assert!(u_mu.iter().all(|&v| v.abs() < 1e-12));
        // w_mu = 1/sigma^2 = 1.0
        assert!(w_mu.iter().all(|&v| (v - 1.0).abs() < 1e-12));
    }

    #[test]
    fn gaussian_sigma_fisher_info_constant() {
        let y = array![0.0, 1.0, 2.0];
        let mu = array![0.0, 0.0, 0.0];
        let sigma = array![1.0, 2.0, 3.0];
        let mut p = HashMap::new();
        p.insert("mu", &mu);
        p.insert("sigma", &sigma);
        let derivs = Gaussian.derivatives(&y, &p).unwrap();
        let (_, w_sigma) = &derivs["sigma"];
        assert!(w_sigma.iter().all(|&v| (v - 2.0).abs() < 1e-12));
    }

    #[test]
    fn studentt_derivatives() {
        let y = array![0.0, 1.0, 2.0, -1.5];
        let mu = array![0.0, 0.5, 1.5, -1.0];
        let sigma = array![1.0, 1.0, 0.8, 1.2];
        let nu = array![5.0, 10.0, 4.0, 8.0];
        let mut p = HashMap::new();
        p.insert("mu", &mu);
        p.insert("sigma", &sigma);
        p.insert("nu", &nu);
        derivative_keys_match_parameters(&StudentT, p, &y);
    }

    #[test]
    fn gamma_derivatives() {
        let y = array![0.5, 1.5, 3.0, 7.0];
        let mu = array![1.0, 2.0, 4.0, 6.0];
        let sigma = array![0.5, 0.4, 0.3, 0.6];
        let mut p = HashMap::new();
        p.insert("mu", &mu);
        p.insert("sigma", &sigma);
        derivative_keys_match_parameters(&Gamma, p, &y);
    }

    #[test]
    fn negative_binomial_derivatives() {
        let y = array![0.0, 3.0, 10.0, 25.0];
        let mu = array![1.0, 4.0, 8.0, 20.0];
        let sigma = array![0.5, 0.5, 0.5, 0.5];
        let mut p = HashMap::new();
        p.insert("mu", &mu);
        p.insert("sigma", &sigma);
        derivative_keys_match_parameters(&NegativeBinomial, p, &y);
    }

    #[test]
    fn beta_derivatives() {
        let y = array![0.1, 0.5, 0.9, 0.25];
        let mu = array![0.2, 0.5, 0.8, 0.3];
        let phi = array![5.0, 10.0, 15.0, 8.0];
        let mut p = HashMap::new();
        p.insert("mu", &mu);
        p.insert("phi", &phi);
        derivative_keys_match_parameters(&Beta, p, &y);
    }

    #[test]
    fn binomial_derivatives_constant_trials() {
        let bin = Binomial::new(20);
        let y = array![5.0, 10.0, 15.0, 8.0];
        let mu = array![0.25, 0.5, 0.7, 0.4];
        let mut p = HashMap::new();
        p.insert("mu", &mu);
        derivative_keys_match_parameters(&bin, p, &y);
    }

    #[test]
    fn binomial_per_observation_trials() {
        let trials = array![10.0, 20.0, 5.0];
        let bin = Binomial::with_trials(trials);
        let y = array![3.0, 10.0, 2.0];
        let mu = array![0.3, 0.5, 0.4];
        let mut p = HashMap::new();
        p.insert("mu", &mu);
        let derivs = bin.derivatives(&y, &p).unwrap();
        let (u_mu, _) = &derivs["mu"];
        assert_relative_eq!(u_mu[0], 3.0 - 10.0 * 0.3, epsilon = 1e-12);
        assert_relative_eq!(u_mu[1], 10.0 - 20.0 * 0.5, epsilon = 1e-12);
    }

    #[test]
    fn binomial_score_zero_when_y_equals_n_mu() {
        let bin = Binomial::new(10);
        let y = array![3.0, 5.0, 7.0];
        let mu = array![0.3, 0.5, 0.7];
        let mut p = HashMap::new();
        p.insert("mu", &mu);
        let derivs = bin.derivatives(&y, &p).unwrap();
        let (u, _) = &derivs["mu"];
        assert!(u.iter().all(|&v| v.abs() < 1e-12));
    }

    // --- initial_value ---

    #[test]
    fn initial_value_finite_for_typical_parameters() {
        let y = array![1.0, 2.0, 3.0, 4.0];
        for d in [
            from_name("Gaussian").unwrap(),
            from_name("Poisson").unwrap(),
            from_name("StudentT").unwrap(),
            from_name("Gamma").unwrap(),
            from_name("Beta").unwrap(),
            from_name("NegativeBinomial").unwrap(),
        ] {
            for p in d.parameters() {
                let v = d.initial_value(p, &y);
                assert!(
                    v.is_finite(),
                    "{}::{} initial_value not finite: {}",
                    d.name(),
                    p,
                    v
                );
            }
        }
    }

    #[test]
    fn binomial_initial_value_clamped_in_unit_interval() {
        let bin = Binomial::new(10);
        // All-zero counts → naive 0.0, must be clamped up.
        let y_zero = array![0.0, 0.0];
        let v0 = bin.initial_value("mu", &y_zero);
        assert!((0.1..=0.9).contains(&v0));
        // All-max counts → naive 1.0, must be clamped down.
        let y_full = array![10.0, 10.0];
        let v1 = bin.initial_value("mu", &y_full);
        assert!((0.1..=0.9).contains(&v1));
    }

    // --- unknown_param helper ---

    #[test]
    fn unknown_param_carries_distribution_name() {
        let err = Gaussian.unknown_param("zeta");
        match err {
            GamlssError::UnknownParameter {
                distribution,
                param,
            } => {
                assert_eq!(distribution, "Gaussian");
                assert_eq!(param, "zeta");
            }
            other => panic!("expected UnknownParameter, got {:?}", other),
        }
    }

    // --- loglik / variance / expected_value per distribution ---

    /// Build a `params` view from owned arrays for test ergonomics.
    fn params_view<'a>(
        owned: &'a [(&'static str, Array1<f64>)],
    ) -> HashMap<&'a str, &'a Array1<f64>> {
        owned.iter().map(|(k, v)| (*k, v)).collect()
    }

    #[test]
    fn loglik_gaussian_matches_manual_formula() {
        let owned = [("mu", array![0.0]), ("sigma", array![1.0])];
        let p = params_view(&owned);
        let ll = Gaussian.loglik(&array![0.0], &p).unwrap();
        let expected = -0.5 * (2.0 * std::f64::consts::PI).ln();
        assert!((ll - expected).abs() < 1e-12);
    }

    #[test]
    fn loglik_poisson_matches_manual() {
        // l = y log(μ) − μ − log Γ(y+1). y=0, μ=1 → −1.
        let owned = [("mu", array![1.0])];
        let p = params_view(&owned);
        let ll = Poisson.loglik(&array![0.0], &p).unwrap();
        assert!((ll - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn loglik_binomial_matches_manual() {
        // n=2, y=1, mu=0.5 → log C(2,1) + 1·log(0.5) + 1·log(0.5) = log 2 + 2 log 0.5
        let bin = Binomial::new(2);
        let owned = [("mu", array![0.5])];
        let p = params_view(&owned);
        let ll = bin.loglik(&array![1.0], &p).unwrap();
        let expected = 2.0_f64.ln() + 2.0 * 0.5_f64.ln();
        assert!((ll - expected).abs() < 1e-9);
    }

    #[test]
    fn loglik_gamma_finite_on_typical_inputs() {
        let owned = [
            ("mu", array![2.0, 2.0, 4.0]),
            ("sigma", array![0.5, 0.4, 0.3]),
        ];
        let p = params_view(&owned);
        let ll = Gamma.loglik(&array![1.0, 2.0, 5.0], &p).unwrap();
        assert!(ll.is_finite());
    }

    #[test]
    fn loglik_negative_binomial_finite() {
        let owned = [
            ("mu", array![1.0, 4.0, 8.0]),
            ("sigma", array![0.5, 0.5, 0.5]),
        ];
        let p = params_view(&owned);
        let ll = NegativeBinomial
            .loglik(&array![0.0, 5.0, 10.0], &p)
            .unwrap();
        assert!(ll.is_finite());
    }

    #[test]
    fn loglik_beta_finite() {
        let owned = [
            ("mu", array![0.2, 0.5, 0.8]),
            ("phi", array![10.0, 10.0, 10.0]),
        ];
        let p = params_view(&owned);
        let ll = Beta.loglik(&array![0.1, 0.5, 0.9], &p).unwrap();
        assert!(ll.is_finite());
    }

    #[test]
    fn loglik_studentt_matches_cauchy_at_zero() {
        // Student-t with ν=1, μ=0, σ=1 is standard Cauchy. Density at y=0 is 1/π.
        let owned = [
            ("mu", array![0.0]),
            ("sigma", array![1.0]),
            ("nu", array![1.0]),
        ];
        let p = params_view(&owned);
        let ll = StudentT.loglik(&array![0.0], &p).unwrap();
        let expected = -std::f64::consts::PI.ln();
        assert!((ll - expected).abs() < 1e-12);
    }

    #[test]
    fn loglik_studentt_finite_on_typical_inputs() {
        let owned = [
            ("mu", array![0.0, 1.0, 2.0]),
            ("sigma", array![1.0, 1.5, 0.5]),
            ("nu", array![5.0, 10.0, 4.0]),
        ];
        let p = params_view(&owned);
        let ll = StudentT.loglik(&array![0.5, 0.5, 1.5], &p).unwrap();
        assert!(ll.is_finite());
    }

    #[test]
    fn variance_studentt_uses_sigma_sq_nu_over_nu_minus_two() {
        let owned = [
            ("mu", array![0.0]),
            ("sigma", array![1.0]),
            ("nu", array![4.0]),
        ];
        let p = params_view(&owned);
        // σ²·ν/(ν−2) = 1·4/2 = 2.
        let v = StudentT.variance(&p).unwrap();
        assert!((v[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn variance_studentt_clamps_at_low_nu() {
        // ν ≤ 2 is undefined; clamp keeps the value finite for downstream Pearson math.
        let owned = [("sigma", array![1.0]), ("nu", array![1.5])];
        let p = params_view(&owned);
        let v = StudentT.variance(&p).unwrap();
        assert!(v[0].is_finite());
        assert!(v[0] > 0.0);
    }

    #[test]
    fn variance_gamma_is_mu_squared_sigma_squared() {
        let owned = [("mu", array![2.0, 3.0]), ("sigma", array![0.5, 0.5])];
        let p = params_view(&owned);
        let v = Gamma.variance(&p).unwrap();
        // μ²σ² = 4·0.25 = 1; 9·0.25 = 2.25.
        assert!((v[0] - 1.0).abs() < 1e-12);
        assert!((v[1] - 2.25).abs() < 1e-12);
    }

    #[test]
    fn variance_gaussian_is_sigma_squared() {
        let owned = [("mu", array![0.0, 0.0]), ("sigma", array![2.0, 3.0])];
        let p = params_view(&owned);
        let v = Gaussian.variance(&p).unwrap();
        assert!((v[0] - 4.0).abs() < 1e-12);
        assert!((v[1] - 9.0).abs() < 1e-12);
    }

    #[test]
    fn variance_poisson_is_mu() {
        let owned = [("mu", array![1.0, 4.0, 9.0])];
        let p = params_view(&owned);
        let v = Poisson.variance(&p).unwrap();
        assert_eq!(v, array![1.0, 4.0, 9.0]);
    }

    #[test]
    fn variance_negative_binomial_is_mu_plus_sigma_mu_squared() {
        let owned = [("mu", array![2.0]), ("sigma", array![0.5])];
        let p = params_view(&owned);
        let v = NegativeBinomial.variance(&p).unwrap();
        // 2 + 0.5·4 = 4
        assert!((v[0] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn variance_beta_uses_mu_one_minus_mu_over_one_plus_phi() {
        let owned = [("mu", array![0.5]), ("phi", array![3.0])];
        let p = params_view(&owned);
        let v = Beta.variance(&p).unwrap();
        // 0.5·0.5/(1+3) = 0.0625
        assert!((v[0] - 0.0625).abs() < 1e-12);
    }

    #[test]
    fn variance_binomial_is_n_mu_one_minus_mu() {
        let bin = Binomial::new(10);
        let owned = [("mu", array![0.3, 0.5])];
        let p = params_view(&owned);
        let v = bin.variance(&p).unwrap();
        assert!((v[0] - 10.0 * 0.3 * 0.7).abs() < 1e-12);
        assert!((v[1] - 10.0 * 0.5 * 0.5).abs() < 1e-12);
    }

    #[test]
    fn expected_value_default_is_mu() {
        let owned = [("mu", array![1.0, 2.0]), ("sigma", array![1.0, 1.0])];
        let p = params_view(&owned);
        let e = Gaussian.expected_value(&p).unwrap();
        assert_eq!(e, array![1.0, 2.0]);
    }

    #[test]
    fn expected_value_binomial_is_n_times_mu() {
        let bin = Binomial::new(10);
        let owned = [("mu", array![0.3, 0.5])];
        let p = params_view(&owned);
        let e = bin.expected_value(&p).unwrap();
        assert!((e[0] - 3.0).abs() < 1e-12);
        assert!((e[1] - 5.0).abs() < 1e-12);
    }

    // --- gradient consistency: analytic score u vs central-difference of loglik ---
    //
    // For each (distribution, parameter), perturb that parameter on the η-scale by ±ε
    // (round-tripping through inv_link), evaluate `loglik_pointwise` at both points,
    // and assert the central difference matches the analytic score returned by
    // `derivatives()`. Catches sign errors and chain-rule slips that integration tests
    // would only detect indirectly via slow / wrong convergence.

    fn check_score_via_finite_diff<D: Distribution + ?Sized>(
        family: &D,
        y: &Array1<f64>,
        owned: &[(&'static str, Array1<f64>)],
        target: &str,
        tol: f64,
    ) {
        let p: HashMap<&str, &Array1<f64>> = owned.iter().map(|(k, v)| (*k, v)).collect();
        let derivs = family.derivatives(y, &p).unwrap();
        let analytic_u = derivs.get(target).unwrap().0.clone();

        let link = family.default_link(target).unwrap();
        let eps: f64 = 1e-6;
        let idx = owned.iter().position(|(k, _)| *k == target).unwrap();

        let mut perturbed: Vec<(&'static str, Array1<f64>)> =
            owned.iter().map(|(k, v)| (*k, v.clone())).collect();

        for i in 0..y.len() {
            let mu_orig = owned[idx].1[i];
            let eta = link.link(mu_orig);

            perturbed[idx].1[i] = link.inv_link(eta + eps);
            let p_plus: HashMap<&str, &Array1<f64>> =
                perturbed.iter().map(|(k, v)| (*k, v)).collect();
            let l_plus = family.loglik_pointwise(y, &p_plus).unwrap()[i];

            perturbed[idx].1[i] = link.inv_link(eta - eps);
            let p_minus: HashMap<&str, &Array1<f64>> =
                perturbed.iter().map(|(k, v)| (*k, v)).collect();
            let l_minus = family.loglik_pointwise(y, &p_minus).unwrap()[i];

            perturbed[idx].1[i] = mu_orig;

            let numeric_u = (l_plus - l_minus) / (2.0 * eps);
            let scale = analytic_u[i].abs().max(1.0);
            assert!(
                (analytic_u[i] - numeric_u).abs() / scale < tol,
                "{}::{} obs {}: analytic u={:.6e}, numeric u={:.6e}",
                family.name(),
                target,
                i,
                analytic_u[i],
                numeric_u
            );
        }
    }

    #[test]
    fn score_matches_finite_diff_poisson() {
        let y = array![0.0, 3.0, 7.0, 12.0];
        let owned = [("mu", array![1.0, 3.5, 6.0, 10.0])];
        check_score_via_finite_diff(&Poisson, &y, &owned, "mu", 1e-5);
    }

    #[test]
    fn score_matches_finite_diff_gaussian() {
        let y = array![-1.0, 0.0, 1.0, 2.0];
        let owned = [
            ("mu", array![-0.5, 0.5, 0.5, 1.5]),
            ("sigma", array![1.0, 1.5, 0.8, 1.2]),
        ];
        check_score_via_finite_diff(&Gaussian, &y, &owned, "mu", 1e-5);
        check_score_via_finite_diff(&Gaussian, &y, &owned, "sigma", 1e-5);
    }

    #[test]
    fn score_matches_finite_diff_studentt() {
        let y = array![-1.0, 0.5, 2.0];
        let owned = [
            ("mu", array![0.0, 0.5, 1.0]),
            ("sigma", array![1.0, 1.2, 0.8]),
            ("nu", array![5.0, 8.0, 4.0]),
        ];
        check_score_via_finite_diff(&StudentT, &y, &owned, "mu", 1e-5);
        check_score_via_finite_diff(&StudentT, &y, &owned, "sigma", 1e-5);
        check_score_via_finite_diff(&StudentT, &y, &owned, "nu", 1e-5);
    }

    #[test]
    fn score_matches_finite_diff_gamma() {
        let y = array![1.0, 2.5, 5.0];
        let owned = [
            ("mu", array![1.5, 2.0, 4.0]),
            ("sigma", array![0.5, 0.4, 0.3]),
        ];
        check_score_via_finite_diff(&Gamma, &y, &owned, "mu", 1e-5);
        check_score_via_finite_diff(&Gamma, &y, &owned, "sigma", 1e-5);
    }

    #[test]
    fn score_matches_finite_diff_negative_binomial() {
        let y = array![0.0, 4.0, 10.0];
        let owned = [
            ("mu", array![1.0, 4.0, 8.0]),
            ("sigma", array![0.5, 0.3, 0.4]),
        ];
        check_score_via_finite_diff(&NegativeBinomial, &y, &owned, "mu", 1e-5);
        check_score_via_finite_diff(&NegativeBinomial, &y, &owned, "sigma", 1e-5);
    }

    #[test]
    fn score_matches_finite_diff_beta() {
        let y = array![0.2, 0.5, 0.85];
        let owned = [
            ("mu", array![0.3, 0.5, 0.7]),
            ("phi", array![10.0, 12.0, 8.0]),
        ];
        check_score_via_finite_diff(&Beta, &y, &owned, "mu", 1e-5);
        check_score_via_finite_diff(&Beta, &y, &owned, "phi", 1e-5);
    }

    #[test]
    fn score_matches_finite_diff_binomial() {
        let bin = Binomial::new(10);
        let y = array![3.0, 5.0, 8.0];
        let owned = [("mu", array![0.3, 0.5, 0.7])];
        check_score_via_finite_diff(&bin, &y, &owned, "mu", 1e-5);
    }

    #[cfg(not(target_arch = "wasm32"))]
    proptest! {
        #[test]
        fn loglik_gaussian_pointwise_matches_naive(
            n in 1usize..20,
            mu_val in -5.0f64..5.0,
            sigma_val in 0.1f64..3.0,
        ) {
            let y = Array1::from_iter((0..n).map(|i| i as f64 * 0.1));
            let mu = Array1::from_elem(n, mu_val);
            let sigma = Array1::from_elem(n, sigma_val);
            let owned = [("mu", mu), ("sigma", sigma)];
            let p = params_view(&owned);
            let actual = Gaussian.loglik(&y, &p).unwrap();
            let log_2pi = (2.0 * std::f64::consts::PI).ln();
            let expected: f64 = (0..n).map(|i| {
                let z = (y[i] - mu_val) / sigma_val;
                -0.5 * log_2pi - sigma_val.ln() - 0.5 * z * z
            }).sum();
            prop_assert!((actual - expected).abs() < 1e-9);
        }
    }
}
