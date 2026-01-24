use crate::error::GamlssError;
use crate::math::trigamma;
use statrs::function::gamma::digamma;
use std::collections::HashMap;
use std::fmt::Debug;

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

pub trait Distribution: Debug + Send + Sync {
    fn parameters(&self) -> &[&'static str];
    fn default_link(&self, param: &str) -> Result<Box<dyn Link>, GamlssError>;
    fn derivatives(&self, y: f64, params: &HashMap<String, f64>) -> HashMap<String, (f64, f64)>;
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
    fn derivatives(&self, y: f64, params: &HashMap<String, f64>) -> HashMap<String, (f64, f64)> {
        // Poisson log-likelihood: l = y*log(mu) - mu
        // Score (dl/dmu): u = y/mu - 1 = (y - mu)/mu
        // Fisher information: E[-d²l/dmu²] = 1/mu
        // For IRLS with log link, working weight w = mu (since Var(Y) = mu)
        let mu = params["mu"];
        let deriv_u = y - mu;
        let deriv_w = mu;

        HashMap::from([("mu".to_string(), (deriv_u, deriv_w))])
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
    fn derivatives(&self, y: f64, params: &HashMap<String, f64>) -> HashMap<String, (f64, f64)> {
        // Gaussian log-likelihood: l = -0.5*log(2*pi) - log(sigma) - (y-mu)^2/(2*sigma^2)
        //
        // For mu (identity link):
        //   dl/dmu = (y-mu)/sigma^2,  Fisher info = 1/sigma^2
        //
        // For sigma (log link, so we work with eta = log(sigma)):
        //   Chain rule gives u_sigma = [(y-mu)^2 - sigma^2] / sigma^2
        //   Fisher info for eta: w = 2 (since Var[(Y-mu)^2/sigma^2] = 2 for normal)
        // See docs/mathematics.md for full derivation.
        let mu = params["mu"];
        let sigma = params["sigma"];
        let sigma_sq = sigma.powi(2);

        let u_mu = (y - mu) / sigma_sq;
        let w_mu = 1.0 / sigma_sq;

        let u_sigma = ((y - mu).powi(2) - sigma_sq) / sigma_sq;
        let w_sigma = 2.0;

        HashMap::from([
            ("mu".to_string(), (u_mu, w_mu)),
            ("sigma".to_string(), (u_sigma, w_sigma)),
        ])
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
    fn derivatives(&self, y: f64, params: &HashMap<String, f64>) -> HashMap<String, (f64, f64)> {
        // Student-t log-likelihood (location-scale parameterization).
        // See docs/mathematics.md for the full derivation of all derivatives.
        let mu = params.get("mu").copied().unwrap_or(0.0);
        let sigma = params.get("sigma").copied().unwrap_or(1.0);
        let nu = params.get("nu").copied().unwrap_or(10.0);

        let z = (y - mu) / sigma;
        let z_sq = z.powi(2);

        // w_robust = (nu+1)/(nu+z^2) appears in all derivatives.
        // This "robustifying weight" downweights outliers (large |z|).
        // As nu -> infinity, w_robust -> 1 and we recover Gaussian behavior.
        let w_robust = (nu + 1.0) / (nu + z_sq);

        // --- mu derivatives (identity link) ---
        let u_mu = (w_robust * z) / sigma;
        let w_mu = w_robust / sigma.powi(2);

        // --- sigma derivatives (log link) ---
        // Chain rule: dl/d_eta = sigma * dl/d_sigma = w_robust*z^2 - 1
        let u_sigma = w_robust * z_sq - 1.0;
        let w_sigma = (2.0 * nu) / (nu + 3.0);

        // --- nu derivatives (log link) ---
        // The score involves digamma functions (derivative of log-gamma).
        let d1 = digamma((nu + 1.0) / 2.0);
        let d2 = digamma(nu / 2.0);
        let term3 = (1.0 + z_sq / nu).ln();
        let term4 = (w_robust * z_sq - 1.0) / nu;

        let dl_dnu = 0.5 * (d1 - d2 - term3 + term4);

        // Chain rule for log link: u_eta = nu * dl/dnu
        let u_nu = dl_dnu * nu;

        // Fisher information for nu involves trigamma functions (second derivative of log-gamma).
        let t1 = trigamma(nu / 2.0);
        let t2 = trigamma((nu + 1.0) / 2.0);
        let t3 = (2.0 * (nu + 3.0)) / (nu * (nu + 1.0));

        let i_nu = 0.25 * (t1 - t2 - t3);

        // Floor at 1e-6 to ensure positive definiteness of the weight matrix.
        // For log link: W_eta = I_nu * nu^2
        let w_nu = (i_nu * nu.powi(2)).abs().max(1e-6);

        HashMap::from([
            ("mu".to_string(), (u_mu, w_mu)),
            ("sigma".to_string(), (u_sigma, w_sigma)),
            ("nu".to_string(), (u_nu, w_nu)),
        ])
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

    fn derivatives(&self, y: f64, params: &HashMap<String, f64>) -> HashMap<String, (f64, f64)> {
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
        let mu = params.get("mu").copied().unwrap_or(1.0).max(1e-10);
        let sigma = params.get("sigma").copied().unwrap_or(1.0).max(1e-10);
        let sigma_sq = sigma.powi(2);
        let alpha = 1.0 / sigma_sq;

        // mu derivatives (log link)
        let u_mu = (y - mu) / (mu * sigma_sq);
        let w_mu = 1.0 / sigma_sq;

        // sigma derivatives (log link)
        // For log link eta = log(sigma), the score is:
        // dl/deta = (2/sigma^2) * [digamma(1/sigma^2) + 2*log(sigma) - log(y/mu) + y/mu - 1]
        let log_y_over_mu = (y / mu).ln();
        let psi_alpha = digamma(alpha);

        let u_sigma =
            (2.0 / sigma_sq) * (psi_alpha + 2.0 * sigma.ln() - log_y_over_mu + y / mu - 1.0);

        // Fisher info for sigma involves trigamma
        // I_sigma = (4/sigma^4) * trigamma(1/sigma^2) - 2/sigma^2
        let psi_prime_alpha = trigamma(alpha);
        let w_sigma = ((4.0 / sigma_sq.powi(2)) * psi_prime_alpha - 2.0 / sigma_sq)
            .abs()
            .max(1e-6);

        HashMap::from([
            ("mu".to_string(), (u_mu, w_mu)),
            ("sigma".to_string(), (u_sigma, w_sigma)),
        ])
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

    fn derivatives(&self, y: f64, params: &HashMap<String, f64>) -> HashMap<String, (f64, f64)> {
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
        let mu = params.get("mu").copied().unwrap_or(1.0).max(1e-10);
        let sigma = params.get("sigma").copied().unwrap_or(0.1).max(1e-10);
        let sigma_sq = sigma.powi(2);

        let one_plus_sigma_mu = 1.0 + sigma * mu;
        let r = 1.0 / sigma; // size parameter

        // mu derivatives (log link)
        let u_mu = (y - mu) / one_plus_sigma_mu;
        let w_mu = mu / one_plus_sigma_mu;

        // sigma derivatives (log link)
        // dl/dsigma = (-1/sigma^2) * [digamma(y + r) - digamma(r) - log(1+sigma*mu) + (y-mu)/(1+sigma*mu)]
        // dl/deta = sigma * dl/dsigma
        let psi_y_r = digamma(y + r);
        let psi_r = digamma(r);
        let log_term = one_plus_sigma_mu.ln();
        let ratio_term = (y - mu) / one_plus_sigma_mu;

        let u_sigma = (-1.0 / sigma) * (psi_y_r - psi_r - log_term + ratio_term);

        // Fisher info for sigma: involves E[digamma(Y + r)]
        // Using approximation based on trigamma(r)
        // w_sigma = (1/sigma^2) * trigamma(1/sigma) (simplified)
        let psi_prime_r = trigamma(r);
        let w_sigma = (psi_prime_r / sigma_sq).abs().max(1e-6);

        HashMap::from([
            ("mu".to_string(), (u_mu, w_mu)),
            ("sigma".to_string(), (u_sigma, w_sigma)),
        ])
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

    fn derivatives(&self, y: f64, params: &HashMap<String, f64>) -> HashMap<String, (f64, f64)> {
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
        let mu = params
            .get("mu")
            .copied()
            .unwrap_or(0.5)
            .clamp(1e-10, 1.0 - 1e-10);
        let phi = params.get("phi").copied().unwrap_or(1.0).max(1e-10);

        let alpha = mu * phi;
        let beta_param = (1.0 - mu) * phi;

        // Clamp y to valid range
        let y_clamped = y.clamp(1e-10, 1.0 - 1e-10);
        let log_y = y_clamped.ln();
        let log_1_minus_y = (1.0 - y_clamped).ln();

        // Digamma values
        let psi_alpha = digamma(alpha);
        let psi_beta = digamma(beta_param);
        let psi_phi = digamma(phi);

        // Trigamma values
        let psi_prime_alpha = trigamma(alpha);
        let psi_prime_beta = trigamma(beta_param);
        let psi_prime_phi = trigamma(phi);

        // mu derivatives (logit link)
        // dl/d_mu = phi * [log(y) - log(1-y) - digamma(alpha) + digamma(beta)]
        // For logit link: dl/d_eta = mu*(1-mu) * dl/d_mu
        let dl_dmu = phi * (log_y - log_1_minus_y - psi_alpha + psi_beta);
        let u_mu = mu * (1.0 - mu) * dl_dmu;

        // Fisher info for mu with logit link
        // I_mu = phi^2 * [trigamma(alpha) + trigamma(beta)]
        // For logit link: w_mu = [mu*(1-mu)]^2 * I_mu
        let i_mu = phi * phi * (psi_prime_alpha + psi_prime_beta);
        let w_mu = (mu * (1.0 - mu)).powi(2) * i_mu;
        let w_mu = w_mu.max(1e-6);

        // phi derivatives (log link)
        // dl/d_phi = digamma(phi) - mu*digamma(alpha) - (1-mu)*digamma(beta)
        //            + mu*log(y) + (1-mu)*log(1-y)
        // For log link: dl/d_eta = phi * dl/d_phi
        let dl_dphi = psi_phi - mu * psi_alpha - (1.0 - mu) * psi_beta
            + mu * log_y
            + (1.0 - mu) * log_1_minus_y;
        let u_phi = phi * dl_dphi;

        // Fisher info for phi with log link
        // I_phi = trigamma(phi) - mu^2*trigamma(alpha) - (1-mu)^2*trigamma(beta)
        // For log link: w_phi = phi^2 * I_phi
        let i_phi =
            psi_prime_phi - mu * mu * psi_prime_alpha - (1.0 - mu) * (1.0 - mu) * psi_prime_beta;
        let w_phi = (phi * phi * i_phi).abs().max(1e-6);

        HashMap::from([
            ("mu".to_string(), (u_mu, w_mu)),
            ("phi".to_string(), (u_phi, w_phi)),
        ])
    }

    fn name(&self) -> &'static str {
        "Beta"
    }
}

// TODO: Add Binomial distribution (will need special handling for n parameter)
