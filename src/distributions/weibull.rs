//! Weibull distribution for positive continuous data

use super::{
    require, DerivativesResult, Distribution, GamlssError, Link, LogLink, MIN_POSITIVE, MIN_WEIGHT,
};
use crate::math::{par_zip3_map, par_zip_map};
use ndarray::Array1;
use statrs::function::gamma::ln_gamma;
use std::collections::HashMap;

/// Weibull distribution (gamlss `WEI` parameterization).
///
/// Parameters: `μ` (scale, log link) and `σ` (shape, log link). Support `y > 0`.
/// With `z = (y/μ)^σ`, `Var(Y) = μ²·[Γ(1+2/σ) − Γ(1+1/σ)²]` and the mean is
/// `μ·Γ(1+1/σ)` — neither equals `μ`, so both moment methods are overridden.
#[derive(Debug, Clone, Copy, Default)]
pub struct Weibull;

impl Weibull {
    pub fn new() -> Self {
        Self
    }
}

impl Distribution for Weibull {
    fn parameters(&self) -> &[&'static str] {
        &["mu", "sigma"]
    }

    fn default_link(&self, param: &str) -> Result<Box<dyn Link>, GamlssError> {
        match param {
            "mu" | "sigma" => Ok(Box::new(LogLink)),
            other => Err(self.unknown_param(other)),
        }
    }

    /// σ is the Weibull shape; the default `y.std()` seed is meaningless for it.
    /// Seed σ = 1 (Exponential), where the scale μ ≈ mean(y); RS refines both.
    fn initial_value(&self, param: &str, y: &Array1<f64>) -> f64 {
        match param {
            "mu" => y.mean().expect("validate_inputs rejects empty y"),
            "sigma" => 1.0,
            _ => 0.1,
        }
    }

    fn derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        // z = (y/μ)^σ ~ Exp(1) at the truth.
        // μ (η = ln μ): u = σ(z−1),               w = σ².
        // σ (η = ln σ): u = 1 + σ·ln(y/μ)·(1−z),  w = π²/6 + (1−γ)² (constant).
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;

        const EULER: f64 = 0.577_215_664_901_532_9;
        let w_sigma_const = std::f64::consts::PI.powi(2) / 6.0 + (1.0 - EULER).powi(2);

        let u_mu = par_zip3_map(y, mu, sigma, |yi, mui, si| {
            let m = mui.max(MIN_POSITIVE);
            let z = (yi.max(MIN_POSITIVE) / m).powf(*si);
            si * (z - 1.0)
        });
        let w_mu = sigma.mapv(|s| (s * s).max(MIN_WEIGHT));

        let u_sigma = par_zip3_map(y, mu, sigma, |yi, mui, si| {
            let m = mui.max(MIN_POSITIVE);
            let r = yi.max(MIN_POSITIVE) / m;
            let z = r.powf(*si);
            1.0 + si * r.ln() * (1.0 - z)
        });
        let w_sigma = Array1::from_elem(y.len(), w_sigma_const.max(MIN_WEIGHT));

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
            let yv = yi.max(MIN_POSITIVE);
            let m = mui.max(MIN_POSITIVE);
            let z = (yv / m).powf(*si);
            si.ln() - si * m.ln() + (si - 1.0) * yv.ln() - z
        }))
    }

    fn expected_value(
        &self,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        // E[Y] = μ·Γ(1 + 1/σ)
        Ok(par_zip_map(mu, sigma, |m, s| {
            m * ln_gamma(1.0 + 1.0 / s.max(MIN_POSITIVE)).exp()
        }))
    }

    fn variance(&self, params: &HashMap<&str, &Array1<f64>>) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        // V[Y] = μ²·[Γ(1+2/σ) − Γ(1+1/σ)²]
        Ok(par_zip_map(mu, sigma, |m, s| {
            let s = s.max(MIN_POSITIVE);
            let g1 = ln_gamma(1.0 + 1.0 / s).exp();
            let g2 = ln_gamma(1.0 + 2.0 / s).exp();
            m * m * (g2 - g1 * g1)
        }))
    }
}
