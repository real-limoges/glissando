use crate::error::GamlssError;
use crate::math::trigamma;
use ndarray::Array1;
use statrs::function::gamma::digamma;
use std::collections::HashMap;
use std::fmt::Debug;

// These traits help make sure the actual distributions are implemented correctly
// I implemented Poisson, Gaussian and StudentT - each more complex than the last
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
    fn name(&self) -> &'static str {
        "StudentT"
    }
    fn derivatives(&self, y: f64, params: &HashMap<String, f64>) -> HashMap<String, (f64, f64)> {
        // nu is set to 10 to prevent a panic
        let mu = params.get("mu").copied().unwrap_or(0.0);
        let sigma = params.get("sigma").copied().unwrap_or(1.0);
        let nu = params.get("nu").copied().unwrap_or(10.0);

        let z = (y - mu) / sigma;
        let z_sq = z.powi(2);
        let w_robust = (nu + 1.0) / (nu + z_sq);

        // so the pattern here is to find u and w for each parameter
        // then combine them into a HashMap. Nu is a bit tricky.

        // mu
        let u_mu = (w_robust * z) / sigma;
        let w_mu = w_robust / sigma.powi(2);

        // sigma
        let u_sigma = w_robust * z_sq - 1.0;
        let w_sigma = (2.0 * nu) / (nu + 3.0);

        // nu
        // dl/dnu
        let d1 = digamma((nu + 1.0) / 2.0);
        let d2 = digamma(nu / 2.0);
        let term3 = (1.0 + z_sq / nu).ln();
        let term4 = (w_robust * z_sq - 1.0) / nu;

        let dl_dnu = 0.5 * (d1 - d2 - term3 + term4);

        // log link chain rule: dl/d_eta = dl/dnu * nu
        let u_nu = dl_dnu * nu;

        // Expected Information for Nu
        // I wrote the trigamma because I couldn't find it elsewhere
        let t1 = trigamma(nu / 2.0);
        let t2 = trigamma((nu + 1.0) / 2.0);
        let t3 = (2.0 * (nu + 3.0)) / (nu * (nu + 1.0));

        let i_nu = 0.25 * (t1 - t2 - t3);

        // log link chain rule: W_eta = I_nu * nu^2
        // positive definiteness
        let w_nu = (i_nu * nu.powi(2)).abs().max(1e-6);

        HashMap::from([
            ("mu".to_string(), (u_mu, w_mu)),
            ("sigma".to_string(), (u_sigma, w_sigma)),
            ("nu".to_string(), (u_nu, w_nu)),
        ])
    }
}
