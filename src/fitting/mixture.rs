//! Finite mixture models (STRUCT-4): the EM capstone over the weighted RS fit.
//!
//! A `K`-component mixture `f(y) = Σ_k w_k · g_k(y)` couples the observations
//! through the components' responsibilities, so — unlike the per-row likelihood
//! wrappers — it cannot be expressed as a single [`Distribution`]. Instead it
//! wraps the existing fit in an EM outer loop:
//!
//! - **E-step**: posterior responsibilities `r_ik = w_k g_k(y_i) / Σ_j w_j g_j(y_i)`.
//! - **M-step**: refit each component via the prior-weighted RS fit
//!   ([`GamlssModel::fit_with_config`] with `weights = r[:,k]`), then set
//!   `w_k = mean_i r_ik`.
//!
//! Iteration stops when the mixture log-likelihood `Σ_i log Σ_k w_k g_k(y_i)`
//! stops improving. Oracle: R `gamlss.mx` (`gamlssMX`).

use super::diagnostics;
use crate::distributions::{Distribution, FamilyDescriptor};
use crate::error::GamlssError;
use crate::model::GamlssModel;
use crate::types::{DataSet, Formula};
use crate::FitConfig;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// Maximum EM outer iterations.
const EM_MAX_ITER: usize = 200;
/// Every responsibility is floored to this (then rows renormalized) before the
/// M-step, so no component can lose all its mass and collapse the weighted fit.
const RESP_FLOOR: f64 = 1e-3;
/// Mixture densities are summed with this floor to keep `log Σ` finite.
const DENS_FLOOR: f64 = 1e-300;

/// A fitted `K`-component finite mixture: the component models, their mixing
/// weights, and the EM convergence summary.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MixtureModel {
    /// One fitted [`GamlssModel`] per component, in component order.
    pub components: Vec<GamlssModel>,
    /// Mixing weights `w_k`, summing to 1, aligned with `components`.
    pub weights: Vec<f64>,
    /// Mixture log-likelihood `Σ_i log Σ_k w_k g_k(y_i)` at convergence.
    pub log_likelihood: f64,
    /// Whether the EM loop met the relative log-likelihood tolerance.
    pub converged: bool,
    /// Number of EM outer iterations performed.
    pub iterations: usize,
    /// Number of observations the mixture was fit on.
    pub n_obs: usize,
    /// Description of the shared component family, so the mixture round-trips
    /// through [`to_json`](MixtureModel::to_json) / [`from_json`](MixtureModel::from_json).
    pub family: FamilyDescriptor,
}

impl MixtureModel {
    /// Total effective degrees of freedom: the summed component EDF plus the
    /// `K − 1` free mixing weights.
    pub fn total_edf(&self) -> f64 {
        let comp_edf: f64 = self
            .components
            .iter()
            .map(|c| diagnostics::total_edf(&c.models))
            .sum();
        comp_edf + (self.components.len().saturating_sub(1)) as f64
    }

    /// Akaike Information Criterion: `−2·loglik + 2·EDF`.
    pub fn aic(&self) -> f64 {
        diagnostics::compute_aic(self.log_likelihood, self.total_edf())
    }

    /// Bayesian Information Criterion: `−2·loglik + ln(n)·EDF`.
    pub fn bic(&self) -> f64 {
        diagnostics::compute_bic(self.log_likelihood, self.total_edf(), self.n_obs)
    }

    /// Serialize the mixture (components, weights, and family descriptor) to JSON.
    ///
    /// # Errors
    ///
    /// Returns [`GamlssError::Input`] if serialization fails.
    #[cfg(feature = "serde")]
    pub fn to_json(&self) -> Result<String, GamlssError> {
        serde_json::to_string(self).map_err(|e| GamlssError::Input(e.to_string()))
    }

    /// Deserialize a mixture from JSON. Reconstruct the shared family with
    /// `model.family.build()`.
    ///
    /// # Errors
    ///
    /// Returns [`GamlssError::Input`] if deserialization fails.
    #[cfg(feature = "serde")]
    pub fn from_json(json: &str) -> Result<Self, GamlssError> {
        serde_json::from_str(json).map_err(|e| GamlssError::Input(e.to_string()))
    }

    /// Mixture mean on new data: `Σ_k w_k · E_k[Y]`, where each component's
    /// expected value is evaluated through `family` at its predicted parameters.
    ///
    /// # Errors
    ///
    /// Propagates any error from component prediction or the family's
    /// `expected_value`.
    pub fn predict_expected_value<D: Distribution + ?Sized>(
        &self,
        new_data: &DataSet,
        family: &D,
    ) -> Result<Array1<f64>, GamlssError> {
        let n = new_data
            .n_obs()
            .ok_or_else(|| GamlssError::Input("new_data has no columns".into()))?;
        let mut out = Array1::<f64>::zeros(n);
        for (comp, &w) in self.components.iter().zip(self.weights.iter()) {
            let params = comp.predict(new_data, family)?;
            let view = params.iter().map(|(k, v)| (k.as_str(), v)).collect();
            let ek = family.expected_value(&view)?;
            out = out + w * &ek;
        }
        Ok(out)
    }
}

/// Fit a `K`-component finite mixture of `family` by EM.
///
/// All components share the same `family` and `formula` but fit independent
/// parameters. `seed` makes the randomized responsibility initialization
/// reproducible; pass `None` for an entropy-seeded start.
///
/// # Errors
///
/// - [`GamlssError::Input`] if `k < 2`, or if a component fit drops rows so the
///   fitted values no longer align with `y` (mixture needs complete-case data).
/// - Any error from the underlying [`GamlssModel::fit_with_config`].
pub fn fit_mixture<D: Distribution + ?Sized>(
    data: &DataSet,
    y: &Array1<f64>,
    formula: &Formula,
    family: &D,
    k: usize,
    config: &FitConfig,
    seed: Option<u64>,
) -> Result<MixtureModel, GamlssError> {
    if k < 2 {
        return Err(GamlssError::Input(format!(
            "fit_mixture: need at least 2 components, got {k}"
        )));
    }
    let n = y.len();
    let tol = config.gd_tolerance;

    let mut rng = StdRng::seed_from_u64(seed.unwrap_or_else(|| rand::rng().random()));
    let mut resp = init_responsibilities(y, k, &mut rng);

    let mut components: Vec<GamlssModel> = Vec::with_capacity(k);
    let mut weights = vec![1.0 / k as f64; k];
    let mut prev_ll = f64::NEG_INFINITY;
    let mut log_likelihood = f64::NEG_INFINITY;
    let mut converged = false;
    let mut iterations = 0;

    for iter in 0..EM_MAX_ITER {
        iterations = iter + 1;
        floor_and_normalize_rows(&mut resp);

        // M-step: refit each component with its responsibility column as weights.
        components.clear();
        let mut raw_weights = Vec::with_capacity(k);
        for j in 0..k {
            let wj = resp.column(j).to_owned();
            let comp =
                GamlssModel::fit_with_config(data, y, Some(&wj), formula, family, config.clone())?;
            // Mixture math indexes fitted values against y; a row-drop would break that.
            let fitted_len = comp
                .models
                .values()
                .next()
                .map(|p| p.fitted_values.len())
                .unwrap_or(0);
            if fitted_len != n {
                return Err(GamlssError::Input(
                    "fit_mixture: a component fit changed the row count (missing data?); \
                     mixtures require complete-case data"
                        .into(),
                ));
            }
            raw_weights.push(wj.sum() / n as f64);
            components.push(comp);
        }
        let wsum: f64 = raw_weights.iter().sum();
        weights = raw_weights.iter().map(|w| w / wsum).collect();

        // E-step: weighted component densities → responsibilities + mixture loglik.
        let mut dens = Array2::<f64>::zeros((n, k));
        for (j, comp) in components.iter().enumerate() {
            let params = diagnostics::fitted_params_view(&comp.models);
            let ll_pt = family.loglik_pointwise(y, &params)?;
            for i in 0..n {
                dens[[i, j]] = weights[j] * ll_pt[i].exp();
            }
        }

        log_likelihood = 0.0;
        for i in 0..n {
            let row_sum: f64 = (0..k).map(|j| dens[[i, j]]).sum::<f64>().max(DENS_FLOOR);
            log_likelihood += row_sum.ln();
            for j in 0..k {
                resp[[i, j]] = dens[[i, j]] / row_sum;
            }
        }

        let rel = (log_likelihood - prev_ll).abs() / (log_likelihood.abs() + 0.1);
        if iter > 0 && rel < tol {
            converged = true;
            break;
        }
        prev_ll = log_likelihood;
    }

    Ok(MixtureModel {
        components,
        weights,
        log_likelihood,
        converged,
        iterations,
        n_obs: n,
        family: family.descriptor(),
    })
}

/// Separating initialization: draw `k` distinct observations as seeds and assign
/// each row hard to its nearest seed in `y` (a 1-D k-means seeding). Unlike a
/// purely random per-row assignment — which hands every component a
/// representative sample of the *whole* response and leaves EM stuck at the
/// symmetric "all components identical" fixed point — this gives the components
/// genuinely different starting regions. The later floor + renormalize keeps
/// every component non-empty.
fn init_responsibilities(y: &Array1<f64>, k: usize, rng: &mut StdRng) -> Array2<f64> {
    let n = y.len();
    // Distinct random seed rows where possible (n ≥ k on any real fit); bounded
    // attempts then top up with repeats so we always return k seeds.
    let mut seeds: Vec<usize> = Vec::with_capacity(k);
    let mut attempts = 0;
    while seeds.len() < k && attempts < 20 * k {
        let idx = rng.random_range(0..n);
        if !seeds.contains(&idx) {
            seeds.push(idx);
        }
        attempts += 1;
    }
    while seeds.len() < k {
        seeds.push(rng.random_range(0..n));
    }

    let mut resp = Array2::<f64>::zeros((n, k));
    for i in 0..n {
        let mut best = 0usize;
        let mut best_d = f64::INFINITY;
        for (j, &s) in seeds.iter().enumerate() {
            let d = (y[i] - y[s]).abs();
            if d < best_d {
                best_d = d;
                best = j;
            }
        }
        resp[[i, best]] = 1.0;
    }
    resp
}

/// Floor every responsibility to [`RESP_FLOOR`] and renormalize each row to sum
/// to 1, guaranteeing no component column is entirely zero.
fn floor_and_normalize_rows(resp: &mut Array2<f64>) {
    let (n, k) = resp.dim();
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..k {
            let v = resp[[i, j]].max(RESP_FLOOR);
            resp[[i, j]] = v;
            sum += v;
        }
        for j in 0..k {
            resp[[i, j]] /= sum;
        }
    }
}
