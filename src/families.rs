use std::collections::HashMap;
use std::fmt::Debug;
use ndarray::Array1;

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

        HashMap::from([("mu", (deriv_u, deriv_w))])
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
    fn derivatives(&self, y: f64, params: &HashMap<String, f64>) -> HashMap<&'static str, (f64, f64)> {
        let mu = params["mu"];
        let sigma = params["sigma"];
        let sigma_sq = sigma.powi(2);

        let u_mu = (y - mu) / sigma_sq;
        let w_mu = 1.0 / sigma_sq;

        let u_sigma = ((y - mu).powi(2) - sigma ^ 2) / sigma_sq;
        let w_sigma = 2.0;

        HashMap::from([("mu", (u_mu, w_mu)),
            ("sigma", (sigma, u_sigma)),
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
        let _mu = params["mu"];
        let _sigma = params["sigma"];
        let _nu = params["nu"];

        todo!("Will fill in derivatives later!")
    }
}
