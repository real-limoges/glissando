use std::collections::HashMap;
use std::fmt::Debug;
use ndarray::Array1;
use statrs::function::gamma::digamma;
use crate::math::trigamma;

// ----- These traits help make sure the actual distributions are implemented correctly
// ----- I have chosen Poisson and Normal/Gaussian, because they are easy.
pub trait Link: Debug + Send + Sync {
    fn link(&self, mu:f64) -> f64;
    fn inv_link(&self, eta:f64) -> f64;
}

// Concrete Links

#[derive(Debug, Clone, Copy, Default)]
pub struct IdentityLink;
impl Link for IdentityLink {
    fn link(&self, mu:f64) -> f64 {
        mu
    }
    fn inv_link(&self, eta:f64) -> f64 {
        eta
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LogLink;
impl Link for LogLink {
    fn link(&self, mu:f64) -> f64 {
        mu.ln().max(-30.0)
    }
    fn inv_link(&self, eta:f64) -> f64 {
        eta.min(30.0).exp()
    }
}


pub trait Distribution: Debug + Send + Sync {
    fn parameters(&self) -> &[&'static str];
    fn default_link(&self, param: &str) -> Box<dyn Link>;
    fn derivatives(&self, y: f64, params: &HashMap<String, f64>) -> HashMap<String,(f64,f64)>;
}


// Distributions


#[derive(Debug, Clone, Copy, Default)]
pub struct Poisson;
impl Poisson {
    pub fn new() -> Self { Self }
}

impl Distribution for Poisson {
    fn parameters(&self) -> &[&'static str] {
        &["mu"]
    }
    fn default_link(&self, param: &str) -> Box<dyn Link> {
        match param {
            "mu" => Box::new(LogLink),
            _ => panic!("Unknown parameter: {}", param)
        }
    }
    fn derivatives(&self, y: f64, params: &HashMap<String, f64>) -> HashMap<String, (f64, f64)> {
        let mu = params["mu"];
        let deriv_u = y - mu;
        let deriv_w = mu;

        HashMap::from([("mu".to_string(), (deriv_u, deriv_w))])
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Gaussian;
impl Gaussian {
    pub fn new() -> Self { Self }
}

impl Distribution for Gaussian {
    fn parameters(&self) -> &[&'static str] {
        &["mu", "sigma"]
    }
    fn default_link(&self, param: &str) -> Box<dyn Link> {
        match param {
            "mu" => Box::new(IdentityLink),
            "sigma" => Box::new(LogLink),
            _ => panic!("Unknown parameter: {}", param),
        }
    }
    fn derivatives(&self, y: f64, params: &HashMap<String, f64>) -> HashMap<String, (f64, f64)> {
        let mu = params["mu"];
        let sigma = params["sigma"];
        let sigma_sq = sigma.powi(2);

        let u_mu = (y - mu) / sigma_sq;
        let w_mu = 1.0 / sigma_sq;

        let u_sigma = ((y - mu).powi(2) - sigma.powi(2)) / sigma_sq;
        let w_sigma = 2.0;

        HashMap::from(
            [("mu".to_string(), (u_mu, w_mu)),
            ("sigma".to_string(), (sigma, u_sigma)),
        ])
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct StudentT;
impl StudentT {
    pub fn new() -> Self { Self }
}
impl Distribution for StudentT {
    fn parameters(&self) -> &[&'static str] {
        &["mu", "sigma", "nu"]
    }
    fn default_link(&self, param: &str) -> Box<dyn Link> {
        match param {
            "mu" => Box::new(IdentityLink),
            "sigma" => Box::new(LogLink),
            "nu" => Box::new(LogLink),
            _ => panic!("Unknown parameter: {}", param),
        }
    }
    fn derivatives(&self, y: f64, params: &HashMap<String, f64>) -> HashMap<String,(f64,f64)> {
        let mu = params["mu"];
        let sigma = params["sigma"];
        let nu = params["nu"];

        let z = (y - mu) / sigma;
        let z_sq = z.powi(2);
        let w_robust = (nu + 1.0) / (nu + z_sq);

        // mu
        let u_mu = (w_robust * z) / sigma;
        let w_mu = (nu + 1.0) / ((nu + 3.0) * sigma.powi(2));
        
        // sigma
        let u_sigma = (w_robust * z_sq) - 1.0;
        let w_sigma = (2.0 * w_robust) / (nu + 3.0);
        
        // nu
        let d1 = digamma((nu + 1.0) / 2.0);
        let d2 = digamma(nu / 2.0);
        let term3 = (1.0 + z_sq / nu).ln();
        let term4 = (w_robust * z - 1.0) / nu;

        let dl_dnu = 0.5 * (d1 - d2 - term3 + term4);
        let u_nu = dl_dnu * nu;

        let t1 = trigamma(nu / 2.0);
        let t2 = trigamma((nu + 1.0) / 2.0);
        let t3 = (2.0 * (nu + 3.0)) / (nu * (nu + 1.0));

        let i_nu = 0.25 * (t1 - t2 - t3);
        let w_nu = (i_nu * nu.powi(2)).abs().max(1e-6);

        HashMap::from([
            ("mu".to_string(), (u_mu, w_mu)),
            ("sigma".to_string(), (u_sigma, w_sigma)),
            ("nu".to_string(), (u_nu, w_nu)),
        ])
    }
}
