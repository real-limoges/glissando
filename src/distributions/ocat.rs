//! Ordered-categorical (ocat) distribution: proportional-odds / cumulative-logit.
//!
//! Models an ordinal response y ∈ {1, …, R} via a single latent linear predictor
//! `mu` (identity link) and R−1 estimated threshold parameters θ₁ < … < θ_{R-1}.
//! Monotonicity is enforced by parameterising as positive increments:
//!   θ₁ = δ₁  (identity link, unconstrained)
//!   θ_k = θ_{k-1} + exp(δ_k)  for k ≥ 2  (log link, increment always > 0)
//!
//! The fitting loop treats each threshold as an intercept-only formula; no new
//! machinery is needed beyond what already drives the RS update for scalar parameters.
//!
//! # Supported category counts
//! R = 2, 3, 4, 5.  R = 4 is the B3 use case.
//!
//! # Exclusion from name-based routes
//! Like `Binomial`, `Ocat` carries state (`n_categories`) that cannot be recovered
//! from a name string alone.  It is excluded from `from_name` and from the
//! WASM / JSON string-dispatch routes.  Construct it explicitly in Rust or via the
//! Python `Ocat(n_categories=4)` class.

use super::{
    require, DerivativesResult, Distribution, GamlssError, IdentityLink, Link, LinkContext, LogLink,
};
use crate::distributions::{MAX_ETA, MIN_ETA};
use ndarray::Array1;
use std::collections::HashMap;

/// Floor for cumulative/category probabilities. Deliberately larger (and kept
/// local to this file) than the shared `distributions::PROB_EPS` (`1e-12`):
/// probabilities here are summed and renormalized across up to 5 categories, so
/// a `1e-12` floor would still underflow after normalization.
const MIN_PROB: f64 = 1e-10;

/// Ordered-categorical GAMLSS distribution with `R` levels.
///
/// Parameters: `"mu"` (latent linear predictor, identity link) plus `"delta_1"` …
/// `"delta_{R-1}"` (threshold / increment parameters).
#[derive(Debug, Clone)]
pub struct Ocat {
    n_categories: usize,
}

impl Ocat {
    /// Create an `Ocat` family with `n_categories` ordered response levels.
    ///
    /// # Panics
    ///
    /// Panics if `n_categories < 2` or `n_categories > 5`.
    pub fn new(n_categories: usize) -> Self {
        assert!(
            (2..=5).contains(&n_categories),
            "Ocat: n_categories must be 2–5, got {}",
            n_categories
        );
        Self { n_categories }
    }

    /// Number of ordered response levels.
    pub fn n_categories(&self) -> usize {
        self.n_categories
    }

    fn n_thresholds(&self) -> usize {
        self.n_categories - 1
    }

    /// Static parameter name for the k-th threshold (k = 1..=4).
    pub(crate) fn threshold_param_name(k: usize) -> &'static str {
        match k {
            1 => "delta_1",
            2 => "delta_2",
            3 => "delta_3",
            4 => "delta_4",
            // Unreachable: `Ocat::new` enforces `n_categories ∈ 2..=5`, so callers
            // only ever pass `k ∈ 1..=n_thresholds = 1..=4`.
            _ => {
                unreachable!("Ocat: threshold index {k} out of range (n_categories ≤ 5 invariant)")
            }
        }
    }

    fn logistic(x: f64) -> f64 {
        1.0 / (1.0 + (-x.clamp(MIN_ETA, MAX_ETA)).exp())
    }

    fn logistic_density(x: f64) -> f64 {
        let f = Self::logistic(x);
        f * (1.0 - f)
    }

    /// Reconstruct thresholds θ₁ < … < θ_{n_thresholds} from the per-observation
    /// response-scale delta params at observation index `i`.
    ///
    /// `params["delta_1"][i]` = θ₁ (identity link).
    /// `params["delta_k"][i]` = θ_k − θ_{k-1} = exp(η_k) > 0  for k ≥ 2  (log link).
    ///
    /// Calling with a specific `i` is necessary for finite-difference derivative
    /// checking, which perturbs one observation at a time.  During the fitting loop
    /// all elements are equal, so the result is the same for every `i`.
    pub(crate) fn compute_thresholds_at(
        params: &HashMap<&str, &Array1<f64>>,
        n_thresholds: usize,
        i: usize,
    ) -> Result<Vec<f64>, GamlssError> {
        let mut thresholds = Vec::with_capacity(n_thresholds);
        for k in 1..=n_thresholds {
            let name = Self::threshold_param_name(k);
            let val = params
                .get(name)
                .ok_or_else(|| GamlssError::Input(format!("ocat: missing parameter '{name}'")))?[i];
            if k == 1 {
                thresholds.push(val);
            } else {
                // response-scale value is the positive increment; enforce > 0
                thresholds.push(thresholds[k - 2] + val.max(MIN_PROB));
            }
        }
        Ok(thresholds)
    }

    /// P(y = r | η_μ, θ) for r = 1 … R.  Returns a normalised Vec of length R.
    ///
    /// P(y ≤ k | η_μ, θ) = logistic(θ_k − η_μ)  (proportional-odds model).
    pub fn category_probs(eta_mu: f64, thresholds: &[f64]) -> Vec<f64> {
        let r = thresholds.len() + 1;
        let cum: Vec<f64> = thresholds
            .iter()
            .map(|&t| Self::logistic(t - eta_mu))
            .collect();
        let mut probs = Vec::with_capacity(r);
        probs.push(cum[0].max(MIN_PROB));
        for k in 1..thresholds.len() {
            probs.push((cum[k] - cum[k - 1]).max(MIN_PROB));
        }
        probs.push((1.0 - cum[thresholds.len() - 1]).max(MIN_PROB));
        let total: f64 = probs.iter().sum();
        probs.iter().map(|&p| p / total).collect()
    }
}

impl Distribution for Ocat {
    fn parameters(&self) -> &[&'static str] {
        match self.n_categories {
            2 => &["mu", "delta_1"],
            3 => &["mu", "delta_1", "delta_2"],
            4 => &["mu", "delta_1", "delta_2", "delta_3"],
            5 => &["mu", "delta_1", "delta_2", "delta_3", "delta_4"],
            // Unreachable: `Ocat::new` rejects any `n_categories` outside `2..=5`.
            n => {
                unreachable!("Ocat: unsupported n_categories {n} (n_categories ∈ 2..=5 invariant)")
            }
        }
    }

    fn default_link(&self, param: &str) -> Result<Box<dyn Link>, GamlssError> {
        match param {
            "mu" | "delta_1" => Ok(Box::new(IdentityLink)),
            "delta_2" | "delta_3" | "delta_4" => Ok(Box::new(LogLink)),
            other => Err(self.unknown_param(other)),
        }
    }

    fn name(&self) -> &'static str {
        "Ocat"
    }

    fn descriptor(&self) -> super::FamilyDescriptor {
        super::FamilyDescriptor::Ocat {
            n_categories: self.n_categories,
        }
    }

    fn initial_value(&self, param: &str, _y: &Array1<f64>) -> f64 {
        match param {
            "mu" => 0.0,      // latent predictor starts at 0
            "delta_1" => 0.0, // first threshold at 0; identity link → η₁ = 0
            _ => 0.5,         // increments start at 0.5; log link seeds η_k = ln(0.5)
        }
    }

    // -------------------------------------------------------------------------
    // Score and Fisher information on the η-scale (the RS IRLS weights)
    //
    // For observation y_i = r ∈ {1, …, R}:
    //
    //   F_k  = logistic(θ_k − η_μ_i)         cumulative prob P(y ≤ k)
    //   f_k  = F_k·(1−F_k)                   logistic density at θ_k
    //   π_r  = F_r − F_{r-1}                 category probability
    //
    // Score for mu (identity link):
    //   u_μ  = (f_{r-1} − f_r) / π_r         (f_0 = f_R = 0 by boundary convention)
    //
    // Score for delta_k:
    //   u_k  = jac_k · (f_r·1_{r≤R-1} − f_{r-1}·1_{r>k}) / π_r  if r ≥ k
    //        = 0                                                    if r < k
    //
    //   jac_k = 1            for k=1 (identity link: ∂θ_j/∂η_1 = 1 for all j ≥ 1)
    //   jac_k = exp(η_k)     for k≥2 (log link: ∂θ_j/∂η_k = exp(η_k) for j ≥ k)
    //           stored as the response-scale value params["delta_k"][i]
    //
    // Fisher info: E_y[(u_param)²] summed over categories.
    // -------------------------------------------------------------------------
    /// Overrides the η-scale adapter directly, permanently.
    ///
    /// Ocat's thresholds are a *cumulative reparameterization*
    /// `θ_k = δ₁ + Σ_{j≤k} exp(η_j)`, so the map from η to the natural parameters has
    /// a lower-triangular Jacobian rather than the `diag(mu_eta)` that
    /// [`chain_to_eta`](crate::distributions::chain_to_eta) assumes. There is no
    /// separable `(∂l/∂θ, i_θ)` for the generic rule to lift.
    ///
    /// Relatedly, this family's `params["mu"]` already holds **η**, not μ, and
    /// `jac_k` below is `exp(η_k)` only under the log link. Overriding any of its
    /// links is therefore unsupported; see `docs/planning/ALTITUDE-1-phases.md`.
    fn eta_derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
        _ctx: &LinkContext,
    ) -> DerivativesResult {
        let eta_mu = require(self, params, "mu")?;
        let n_obs = y.len();
        let n_thresh = self.n_thresholds();

        let mut u_mu = Array1::zeros(n_obs);
        let mut w_mu = Array1::zeros(n_obs);
        let mut u_thresh: Vec<Array1<f64>> = (0..n_thresh).map(|_| Array1::zeros(n_obs)).collect();
        let mut w_thresh: Vec<Array1<f64>> = (0..n_thresh).map(|_| Array1::zeros(n_obs)).collect();

        for i in 0..n_obs {
            let eta_i = eta_mu[i];
            let y_r = (y[i] as usize).clamp(1, self.n_categories);

            // Reconstruct thresholds at this observation (for finite-diff correctness).
            let thresholds = Self::compute_thresholds_at(params, n_thresh, i)?;

            // f_k = logistic density at θ_k − η_i  (0-indexed: f_vals[k] = f(θ_{k+1}))
            // cum[k] = F_{k+1} = P(y ≤ k+1)
            let f_vals: Vec<f64> = thresholds
                .iter()
                .map(|&t| Self::logistic_density(t - eta_i))
                .collect();
            let cum: Vec<f64> = thresholds
                .iter()
                .map(|&t| Self::logistic(t - eta_i))
                .collect();

            // π_r (1-indexed, r=1..R)
            let mut pi = Vec::with_capacity(self.n_categories);
            pi.push(cum[0].max(MIN_PROB));
            for k in 1..n_thresh {
                pi.push((cum[k] - cum[k - 1]).max(MIN_PROB));
            }
            pi.push((1.0 - cum[n_thresh - 1]).max(MIN_PROB));
            let total: f64 = pi.iter().sum();
            let pi: Vec<f64> = pi.iter().map(|&p| p / total).collect();

            let pi_r = pi[y_r - 1];

            // Boundary-aware f values for the observed category y_r:
            //   f_upper = f(θ_{y_r})   if y_r ≤ n_thresh, else 0 (upper boundary)
            //   f_lower = f(θ_{y_r-1}) if y_r ≥ 2, else 0         (lower boundary)
            let f_upper = if y_r <= n_thresh {
                f_vals[y_r - 1]
            } else {
                0.0
            };
            let f_lower = if y_r >= 2 { f_vals[y_r - 2] } else { 0.0 };

            // ── Score and Fisher info for mu ──────────────────────────────────
            u_mu[i] = (f_lower - f_upper) / pi_r;

            w_mu[i] = (0..self.n_categories)
                .map(|r| {
                    let fl = if r >= 1 { f_vals[r - 1] } else { 0.0 };
                    let fu = if r < n_thresh { f_vals[r] } else { 0.0 };
                    (fl - fu).powi(2) / pi[r].max(MIN_PROB)
                })
                .sum::<f64>();

            // ── Score and Fisher info for each threshold delta_k ──────────────
            for k in 1..=n_thresh {
                let k0 = k - 1; // 0-indexed slot

                // Jacobian on the η-scale:
                //   k=1: identity link → jac = 1
                //   k≥2: log link → jac = exp(η_k) = response-scale value
                let jac_k = if k == 1 {
                    1.0
                } else {
                    params[Self::threshold_param_name(k)][i].max(MIN_PROB)
                };

                u_thresh[k0][i] = if y_r < k {
                    0.0
                } else {
                    // r ≥ k: upper bound θ_{y_r} contributes; lower f_{y_r-1} only if r > k.
                    let eff_lower = if y_r > k { f_lower } else { 0.0 };
                    jac_k * (f_upper - eff_lower) / pi_r
                };

                // Fisher info: sum over r = k..R of (jac·A_r)² / π_r
                w_thresh[k0][i] = (k..=self.n_categories)
                    .map(|r| {
                        let fu = if r <= n_thresh { f_vals[r - 1] } else { 0.0 };
                        let eff_fl = if r > k { f_vals[r - 2] } else { 0.0 };
                        let a = jac_k * (fu - eff_fl);
                        a.powi(2) / pi[r - 1].max(MIN_PROB)
                    })
                    .sum::<f64>();
            }
        }

        let mut result: HashMap<String, (Array1<f64>, Array1<f64>)> = HashMap::new();
        result.insert("mu".to_string(), (u_mu, w_mu));
        for (k0, (u_k, w_k)) in u_thresh.into_iter().zip(w_thresh).enumerate() {
            result.insert(Self::threshold_param_name(k0 + 1).to_string(), (u_k, w_k));
        }
        Ok(result)
    }

    fn loglik_pointwise(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let eta_mu = require(self, params, "mu")?;
        let n_thresh = self.n_thresholds();

        let ll: Result<Array1<f64>, GamlssError> = (0..y.len())
            .map(|i| {
                let r = (y[i] as usize).clamp(1, self.n_categories);
                let thresholds = Self::compute_thresholds_at(params, n_thresh, i)?;
                let eta_i = eta_mu[i];
                let cum_lower = if r >= 2 {
                    Self::logistic(thresholds[r - 2] - eta_i)
                } else {
                    0.0
                };
                let cum_upper = if r <= n_thresh {
                    Self::logistic(thresholds[r - 1] - eta_i)
                } else {
                    1.0
                };
                Ok((cum_upper - cum_lower).max(MIN_PROB).ln())
            })
            .collect();
        ll
    }

    fn variance(&self, params: &HashMap<&str, &Array1<f64>>) -> Result<Array1<f64>, GamlssError> {
        let eta_mu = require(self, params, "mu")?;
        let n_thresh = self.n_thresholds();
        // Use element 0 for thresholds (all elements equal in fitted context).
        let thresholds = Self::compute_thresholds_at(params, n_thresh, 0)?;

        Ok((0..eta_mu.len())
            .map(|i| {
                let probs = Self::category_probs(eta_mu[i], &thresholds);
                let e_y: f64 = probs
                    .iter()
                    .enumerate()
                    .map(|(r, &p)| (r + 1) as f64 * p)
                    .sum();
                let e_y2: f64 = probs
                    .iter()
                    .enumerate()
                    .map(|(r, &p)| ((r + 1) as f64).powi(2) * p)
                    .sum();
                (e_y2 - e_y.powi(2)).max(0.0)
            })
            .collect())
    }

    fn expected_value(
        &self,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let eta_mu = require(self, params, "mu")?;
        let n_thresh = self.n_thresholds();
        let thresholds = Self::compute_thresholds_at(params, n_thresh, 0)?;

        Ok((0..eta_mu.len())
            .map(|i| {
                let probs = Self::category_probs(eta_mu[i], &thresholds);
                probs
                    .iter()
                    .enumerate()
                    .map(|(r, &p)| (r + 1) as f64 * p)
                    .sum::<f64>()
            })
            .collect())
    }

    fn is_discrete(&self) -> bool {
        true
    }

    fn cdf(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        // Right-continuous step CDF at the level ⌊y⌋. Proportional-odds model:
        // P(Y ≤ r) = logistic(θ_r − η) for r < R, and 1 at r = R — the same
        // cumulative the per-category mass in `loglik_pointwise` differences.
        let eta_mu = require(self, params, "mu")?;
        let n_thresh = self.n_thresholds();
        let r_max = self.n_categories;
        (0..y.len())
            .map(|i| {
                let level = y[i].floor();
                if level < 1.0 {
                    return Ok(0.0);
                }
                if level >= r_max as f64 {
                    return Ok(1.0);
                }
                let thresholds = Self::compute_thresholds_at(params, n_thresh, i)?;
                Ok(Self::logistic(thresholds[level as usize - 1] - eta_mu[i]))
            })
            .collect()
    }

    fn quantile(
        &self,
        p: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        // Smallest level r ∈ {1, …, R} whose cumulative prob ≥ p.
        let eta_mu = require(self, params, "mu")?;
        let n_thresh = self.n_thresholds();
        let r_max = self.n_categories;
        (0..p.len())
            .map(|i| {
                let thresholds = Self::compute_thresholds_at(params, n_thresh, i)?;
                let pi = p[i].clamp(0.0, 1.0);
                for r in 1..r_max {
                    if Self::logistic(thresholds[r - 1] - eta_mu[i]) >= pi {
                        return Ok(r as f64);
                    }
                }
                Ok(r_max as f64)
            })
            .collect()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::test_helpers::{
        check_discrete_cdf_matches_pmf, check_score_via_finite_diff,
        derivative_keys_match_parameters, params_view,
    };
    use approx::assert_relative_eq;
    use ndarray::array;

    fn make_params_r4(
        mu: Array1<f64>,
        d1: f64,
        d2: f64,
        d3: f64,
    ) -> Vec<(&'static str, Array1<f64>)> {
        let n = mu.len();
        vec![
            ("mu", mu),
            ("delta_1", Array1::from_elem(n, d1)),
            // For k≥2 the response-scale value is the positive increment exp(η_k).
            ("delta_2", Array1::from_elem(n, d2)),
            ("delta_3", Array1::from_elem(n, d3)),
        ]
    }

    #[test]
    fn category_probs_sum_to_one() {
        let thresholds = vec![-1.0, 0.0, 1.2];
        for eta in [-3.0, -1.0, 0.0, 1.0, 3.0] {
            let probs = Ocat::category_probs(eta, &thresholds);
            assert_eq!(probs.len(), 4);
            let total: f64 = probs.iter().sum();
            assert!((total - 1.0).abs() < 1e-12, "sum={total} for eta={eta}");
        }
    }

    #[test]
    fn monotone_thresholds_from_positive_increments() {
        let n = 3;
        let owned = vec![
            ("delta_1", Array1::from_elem(n, 0.5_f64)),
            ("delta_2", Array1::from_elem(n, 1.0_f64)), // increment = 1.0
            ("delta_3", Array1::from_elem(n, 0.8_f64)), // increment = 0.8
        ];
        let params = params_view(&owned);
        let thresholds = Ocat::compute_thresholds_at(&params, 3, 0).unwrap();
        assert_eq!(thresholds.len(), 3);
        assert!(thresholds[0] < thresholds[1]);
        assert!(thresholds[1] < thresholds[2]);
        assert_relative_eq!(thresholds[0], 0.5, epsilon = 1e-12);
        assert_relative_eq!(thresholds[1], 1.5, epsilon = 1e-12); // 0.5 + 1.0
        assert_relative_eq!(thresholds[2], 2.3, epsilon = 1e-12); // 1.5 + 0.8
    }

    #[test]
    fn derivative_keys_match_all_parameters_r4() {
        let ocat = Ocat::new(4);
        let y = array![1.0, 2.0, 3.0, 4.0];
        let owned = make_params_r4(
            array![0.0, 0.5, -0.5, 1.0],
            0.0, // theta_1 = 0
            1.0, // increment_2 = 1.0 → theta_2 = 1.0
            1.0, // increment_3 = 1.0 → theta_3 = 2.0
        );
        let p = params_view(&owned);
        derivative_keys_match_parameters(&ocat, p, &y);
    }

    #[test]
    fn score_via_finite_diff_mu() {
        let ocat = Ocat::new(4);
        let y = array![1.0, 2.0, 3.0, 4.0, 1.0, 3.0];
        let owned = make_params_r4(array![-1.5, -0.5, 0.0, 0.5, 1.5, -1.0], -0.5, 1.0, 1.0);
        check_score_via_finite_diff(&ocat, &y, &owned, "mu", 1e-5);
    }

    #[test]
    fn score_via_finite_diff_delta_1() {
        let ocat = Ocat::new(4);
        let y = array![1.0, 2.0, 3.0, 4.0, 2.0, 1.0];
        let owned = make_params_r4(array![0.1, -0.3, 0.5, -0.5, 0.2, 0.8], -0.5, 1.0, 1.0);
        check_score_via_finite_diff(&ocat, &y, &owned, "delta_1", 1e-5);
    }

    #[test]
    fn score_via_finite_diff_delta_2() {
        let ocat = Ocat::new(4);
        // Use non-trivial increment values; log-link so stored value is exp(η₂).
        let y = array![1.0, 2.0, 3.0, 4.0, 3.0, 2.0];
        let owned = make_params_r4(array![0.1, -0.3, 0.5, -0.5, 0.2, 0.8], -0.5, 0.7, 1.2);
        check_score_via_finite_diff(&ocat, &y, &owned, "delta_2", 1e-5);
    }

    #[test]
    fn score_via_finite_diff_delta_3() {
        let ocat = Ocat::new(4);
        let y = array![1.0, 2.0, 3.0, 4.0, 4.0, 1.0];
        let owned = make_params_r4(array![0.1, -0.3, 0.5, -0.5, 0.2, 0.8], -0.5, 0.7, 1.2);
        check_score_via_finite_diff(&ocat, &y, &owned, "delta_3", 1e-5);
    }

    #[test]
    fn loglik_pointwise_matches_manual() {
        // θ = (0.0, 1.0, 2.0): thresholds at 0, 1, 2.
        // η_μ = 0.  π_2 = F(1-0) - F(0-0) = logistic(1) - logistic(0) = 0.7311 - 0.5 = 0.2311
        let ocat = Ocat::new(4);
        let y = array![2.0]; // category 2
        let n = 1;
        let owned = vec![
            ("mu", array![0.0_f64]),
            ("delta_1", Array1::from_elem(n, 0.0_f64)), // theta_1 = 0
            ("delta_2", Array1::from_elem(n, 1.0_f64)), // increment → theta_2 = 1
            ("delta_3", Array1::from_elem(n, 1.0_f64)), // increment → theta_3 = 2
        ];
        let p = params_view(&owned);
        let ll = ocat.loglik_pointwise(&y, &p).unwrap();
        let expected = (Ocat::logistic(1.0) - Ocat::logistic(0.0)).ln();
        assert!(
            (ll[0] - expected).abs() < 1e-9,
            "ll={} expected={}",
            ll[0],
            expected
        );
    }

    #[test]
    fn cdf_matches_pmf_ocat() {
        // cdf(k) − cdf(k−1) must equal the per-category mass exp(loglik).
        let ocat = Ocat::new(4);
        let ks = array![1.0, 2.0, 3.0, 4.0];
        let owned = make_params_r4(array![0.2, 0.2, 0.2, 0.2], -0.5, 1.0, 1.0);
        check_discrete_cdf_matches_pmf(&ocat, &ks, &owned, 1e-9);
    }

    #[test]
    fn cdf_monotone_and_endpoints_ocat() {
        let ocat = Ocat::new(4);
        // One parameter row per level on the grid (thresholds are per-observation).
        let levels = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let owned = make_params_r4(array![0.3, 0.3, 0.3, 0.3, 0.3], -0.5, 1.0, 1.0);
        let p = params_view(&owned);
        let f = ocat.cdf(&levels, &p).unwrap();
        assert_eq!(f[0], 0.0); // below the lowest level
        assert_eq!(f[4], 1.0); // at the top level
        for w in f.windows(2) {
            assert!(w[1] >= w[0] - 1e-12, "cdf not monotone: {:?}", w);
        }
    }

    #[test]
    fn quantile_returns_valid_levels_ocat() {
        let ocat = Ocat::new(4);
        let owned = make_params_r4(array![0.0], -0.5, 1.0, 1.0);
        let p = params_view(&owned);
        for &prob in &[0.01, 0.3, 0.6, 0.99] {
            let q = ocat.quantile(&array![prob], &p).unwrap()[0];
            assert!((1.0..=4.0).contains(&q), "level {q} out of 1..=4");
            // The returned level's CDF must reach p (smallest such level).
            let f_q = ocat.cdf(&array![q], &p).unwrap()[0];
            assert!(f_q >= prob - 1e-12, "F(q)={f_q} < p={prob}");
        }
    }
}
