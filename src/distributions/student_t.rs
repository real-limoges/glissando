//! Student's t distribution for heavy-tailed continuous data.

use super::{
    chain_to_eta, clamp_prob, require, DerivativesResult, Distribution, FlooredLogLink,
    GamlssError, IdentityLink, Link, LinkContext, LogLink, DENOM_FLOOR, MIN_POSITIVE,
};
use crate::math::{
    digamma_batch, median, median_abs_deviation, par_zip3_map, par_zip_map, trigamma_batch,
};
use ndarray::Array1;
use statrs::distribution::{ContinuousCDF, StudentsT};
use statrs::function::gamma::ln_gamma;
use std::collections::HashMap;

/// Lower bound on the degrees-of-freedom `ν`, enforced via [`FlooredLogLink`].
/// Keeps `ν > 2` so the variance `σ²·ν/(ν−2)` stays finite while the optimizer
/// explores the heavy-tail region. Never binds when the true `ν` is well above 2.
const NU_FLOOR: f64 = 2.0;

/// Starting value for `ν`. A fixed moderate seed is deliberately preferred over a
/// sample-kurtosis estimate: for regression data the *marginal* kurtosis reflects the
/// spread of the mean structure, not the noise tails, so a kurtosis inversion biases
/// `ν` and (in the multi-smooth weighted case) can seed the optimizer into a degenerate
/// over-smoothed basin. 5 is a standard heavy-tail default, well clear of the `ν > 2`
/// finite-variance boundary.
const NU_INIT: f64 = 5.0;

/// Student's t distribution for heavy-tailed continuous data.
///
/// Parameters: `μ` (location, identity), `σ` (scale, log), `ν` (degrees of freedom,
/// floored log link with `ν ≥ 2`). As `ν → ∞` the distribution approaches Gaussian.
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
            "sigma" => Ok(Box::new(LogLink)),
            "nu" => Ok(Box::new(FlooredLogLink { floor: NU_FLOOR })),
            other => Err(self.unknown_param(other)),
        }
    }

    /// `nu` refuses a link override; `mu` and `sigma` accept one.
    ///
    /// ν keeps a hand-written `eta_derivatives` branch because its floor is
    /// enforced by a KKT-style aggregate projection that only makes sense
    /// against [`FlooredLogLink`], whose `mu_eta` is a hard zero below the
    /// floor. Under any other link the projection's freeze branch would fire on
    /// the wrong condition, which is exactly the lift-off case it exists to
    /// handle. μ and σ go through `chain_to_eta` like any other family.
    fn allows_link_override(&self, param: &str) -> bool {
        param != "nu"
    }

    /// Robust IRLS seeds for heavy-tailed data. The trait default (sample mean,
    /// sample SD) is non-robust: under heavy tails the mean is pulled by outliers and
    /// the SD overestimates the scale `σ`. Instead:
    /// - `μ` = median(y),
    /// - `σ` = 1.4826·MAD(y) (the MAD-to-σ consistency factor for a normal core),
    /// - `ν` = `NU_INIT` = 5 (a fixed moderate seed; see its doc for why a kurtosis
    ///   estimate is avoided).
    fn initial_value(&self, param: &str, y: &Array1<f64>) -> f64 {
        match param {
            "mu" => median(y),
            "sigma" => {
                let s = 1.4826 * median_abs_deviation(y);
                if s < 1e-4 {
                    1.0
                } else {
                    s
                }
            }
            "nu" => NU_INIT,
            other => {
                debug_assert!(
                    matches!(other, "mu" | "sigma" | "nu"),
                    "StudentT has no parameter '{other}'"
                );
                NU_INIT
            }
        }
    }

    /// Hybrid: μ and σ go through the generic chain rule, ν does not.
    ///
    /// ν keeps a hand-written η-scale entry permanently, for two reasons that no
    /// natural-scale form can express. Its KKT boundary projection is a *cross-row
    /// aggregate* (`u_ν[i]` depends on a sum of scores over every pinned row), so it
    /// is not element-wise, and [`FlooredLogLink::mu_eta`] returns 0 below the
    /// floor, so the generic rule would force the freeze branch unconditionally,
    /// which is wrong in exactly the lift-off case the projection exists to handle.
    ///
    /// Consequently `allows_link_override("nu")` is false; see the
    /// `[CHAIN-GENERIC]` section of `docs/math/mathematics.md`.
    fn eta_derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
        ctx: &LinkContext,
    ) -> DerivativesResult {
        // Build the standardized-residual block once and hand it to both halves.
        // `theta_derivatives` and the ν block each need `(z², w_robust)`, and recomputing
        // it in the second cost two extra O(n) passes plus n divisions on every
        // Fisher-scoring step. Worse, it disagreed with the first: the ν block
        // divided by a raw σ where this body uses the `DENOM_FLOOR`-guarded
        // reciprocal, so at σ = 0 the μ/σ pairs stayed finite while the ν score alone
        // came out NaN (`z² = ∞` → `w_robust = 0` → `(0·∞ − 1)/ν`).
        let shared = Standardized::new(self, y, params)?;
        let mut out = chain_to_eta(self.mu_sigma_derivatives(&shared, params)?, ctx)?;
        out.insert("nu".to_string(), self.nu_eta_derivatives(&shared, params)?);
        Ok(out)
    }

    /// Natural-scale score and expected information for **μ and σ only**.
    ///
    /// ν is deliberately absent: it has no separable natural scale (see
    /// [`Self::eta_derivatives`]). Callers that need all three parameters must go
    /// through `eta_derivatives`, which is what the scoring loop does.
    fn theta_derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        self.mu_sigma_derivatives(&Standardized::new(self, y, params)?, params)
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

    fn cdf(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        // Location-scale t: F(y) = T_ν((y−μ)/σ). ν varies per observation, so build
        // one StudentsT per row (mirrors the indexed loglik_pointwise loop above).
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        let nu = require(self, params, "nu")?;
        let n = y.len();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let s = sigma[i].max(MIN_POSITIVE);
            let nu_i = nu[i].max(MIN_POSITIVE);
            out[i] = StudentsT::new(mu[i], s, nu_i)
                .expect("valid StudentsT params")
                .cdf(y[i]);
        }
        Ok(out)
    }

    fn cdf_theta_derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> super::CdfThetaResult {
        // Natural-scale (Altitude #1) location-scale derivatives of F = T_ν(z),
        // z = (y−μ)/σ, with standardized t-pdf g and g'(z) = −g·(ν+1)z/(ν+z²).
        // ∂z/∂μ = −1/σ and ∂z/∂σ = −z/σ, so:
        //   μ:  ∂F/∂μ = −g/σ,    ∂²F/∂μ² = g'/σ².
        //   σ:  ∂F/∂σ = −zg/σ,   ∂²F/∂σ² = (2zg + z²g')/σ².
        // The caller chains to η. Under the default links (identity, log) that
        // recovers the previous η-scale forms exactly: μ has mu_eta = 1 and
        // mu_eta2 = 0, so it is unchanged; σ has mu_eta = mu_eta2 = σ, giving
        // σ·(−zg/σ) = −zg and (2zg + z²g') − zg = zg + z²g'.
        // ν has no elementary CDF derivative (incomplete-beta shape derivative) and
        // is left to the wrapper's numeric fallback.
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        let nu = require(self, params, "nu")?;

        let mut d1_mu = Array1::<f64>::zeros(y.len());
        let mut d2_mu = Array1::<f64>::zeros(y.len());
        let mut d1_sigma = Array1::<f64>::zeros(y.len());
        let mut d2_sigma = Array1::<f64>::zeros(y.len());
        for i in 0..y.len() {
            if !y[i].is_finite() {
                continue; // ±∞ bound: F saturates, all derivatives vanish
            }
            let s = sigma[i].max(MIN_POSITIVE);
            let nu_i = nu[i].max(MIN_POSITIVE);
            let z = (y[i] - mu[i]) / s;
            // As in `gaussian.rs`: every derivative below has limit 0 as |z| → ∞, and
            // taking that limit explicitly is the only way to get it. z overflows to
            // infinity once the bound and σ are far enough apart (σ on its
            // `MIN_POSITIVE` floor against a far-out censoring bound), and then
            // `g' = −g(ν+1)z/(ν+z²)` is ∞/∞ = NaN rather than the 0 it tends to.
            if !z.is_finite() {
                continue;
            }
            // standardized t density g(z)
            let log_g = ln_gamma((nu_i + 1.0) / 2.0)
                - ln_gamma(nu_i / 2.0)
                - 0.5 * (std::f64::consts::PI * nu_i).ln()
                - 0.5 * (nu_i + 1.0) * (1.0 + z * z / nu_i).ln();
            let g = log_g.exp();
            let g_prime = -g * (nu_i + 1.0) * z / (nu_i + z * z);
            let z_sq = z * z;
            d1_mu[i] = -g / s;
            d2_mu[i] = g_prime / (s * s);
            d1_sigma[i] = -z * g / s;
            // z² still overflows on its own around |z| ≈ 1e154, where z is finite but
            // `g'` has already underflowed to 0 and `z²·g'` is `∞ · 0`. The guard
            // fires only where the unguarded expression is NaN, leaving every in-range
            // value untouched.
            d2_sigma[i] = if z_sq.is_finite() {
                (2.0 * z * g + z_sq * g_prime) / (s * s)
            } else {
                0.0
            };
        }
        Ok(HashMap::from([
            ("mu".to_string(), (d1_mu, d2_mu)),
            ("sigma".to_string(), (d1_sigma, d2_sigma)),
        ]))
    }

    fn quantile(
        &self,
        p: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        let nu = require(self, params, "nu")?;
        let n = p.len();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let s = sigma[i].max(MIN_POSITIVE);
            let nu_i = nu[i].max(MIN_POSITIVE);
            out[i] = StudentsT::new(mu[i], s, nu_i)
                .expect("valid StudentsT params")
                .inverse_cdf(clamp_prob(p[i]));
        }
        Ok(out)
    }

    fn name(&self) -> &'static str {
        "StudentT"
    }
}

/// The standardized-residual block every Student-t derivative body starts from.
///
/// Built once per [`Distribution::eta_derivatives`] call and shared by the μ/σ and
/// ν halves, which both need `(z², w_robust)`. Keeping it in one place is also what
/// keeps the two halves *agreeing*: the σ guard below has to be the same on both
/// sides or a saturated σ leaves one block finite and the other NaN.
struct Standardized {
    z: Array1<f64>,
    z_sq: Array1<f64>,
    /// `w = (ν+1)/(ν+z²)`, the robustifying weight that downweights outliers
    /// (large `|z|`). It → 1 as ν → ∞, recovering Gaussian behavior.
    w_robust: Array1<f64>,
    inv_sigma: Array1<f64>,
    inv_sigma_sq: Array1<f64>,
}

impl Standardized {
    fn new(
        family: &StudentT,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Self, GamlssError> {
        let mu = require(family, params, "mu")?;
        let sigma = require(family, params, "sigma")?;
        let nu = require(family, params, "nu")?;

        // Guard each reciprocal at the power it is used at, rather than clamping σ:
        // raising an already-guarded reciprocal to a power would overflow to infinity
        // for a σ the log link can still underflow to, and `inf · 0` is NaN. μ's
        // score and weight divided by a raw σ before Phase 2b; guarding them here
        // costs nothing and keeps the whole family finite at a saturated σ.
        let inv_sigma = sigma.mapv(|s| 1.0 / s.max(DENOM_FLOOR));
        let inv_sigma_sq = sigma.mapv(|s| 1.0 / (s * s).max(DENOM_FLOOR));

        let z = (y - mu) * &inv_sigma;
        let z_sq = z.mapv(|v| v * v);
        let w_robust = par_zip_map(nu, &z_sq, |nu_i, z2_i| (nu_i + 1.0) / (nu_i + z2_i));

        Ok(Self {
            z,
            z_sq,
            w_robust,
            inv_sigma,
            inv_sigma_sq,
        })
    }
}

impl StudentT {
    /// Natural-scale score and expected information for μ and σ.
    ///
    /// Student-t log-likelihood, location-scale parameterization. Full derivation
    /// in docs/math/mathematics.md.
    fn mu_sigma_derivatives(
        &self,
        s: &Standardized,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        let nu = require(self, params, "nu")?;

        // μ derivatives (identity link, so the chain rule leaves these untouched).
        // The score uses the robustifying weight (that IS dl/dμ); the working weight
        // uses the *expected* information I_μ = (ν+1)/((ν+3)·σ²), the same convention
        // as gamlss TF's d2ldm2, rather than the data-dependent w_robust/σ², so the
        // PWLS subproblem (and hence λ selection, EDF, and SEs) matches the RS oracle.
        let u_mu = &s.w_robust * &s.z * &s.inv_sigma;
        let i_mu = par_zip_map(nu, &s.inv_sigma_sq, |nu_i, iss| {
            (nu_i + 1.0) / (nu_i + 3.0) * iss
        });

        // σ derivatives, natural scale: ∂l/∂σ = (w·z² − 1)/σ, i_σ = [2ν/(ν+3)]/σ².
        // Under the default log link `chain_to_eta` (mu_eta = σ) recovers the
        // previous `u_η = w·z² − 1` and `w_η = 2ν/(ν+3)`. Returned unfloored.
        let u_sigma = (&s.w_robust * &s.z_sq - 1.0) * &s.inv_sigma;
        let i_sigma: Array1<f64> = par_zip_map(nu, &s.inv_sigma_sq, |nu_i, iss| {
            (2.0 * nu_i) / (nu_i + 3.0) * iss
        });

        Ok(HashMap::from([
            ("mu".to_string(), (u_mu, i_mu)),
            ("sigma".to_string(), (u_sigma, i_sigma)),
        ]))
    }

    /// The η-scale `(u_ν, w_ν)` pair, deliberately outside the generic chain rule.
    ///
    /// See [`Distribution::eta_derivatives`] on this type for why ν has no
    /// separable natural scale. Nothing here changed in Phase 2b beyond being
    /// lifted out of `theta_derivatives` and having its `MIN_WEIGHT` floor removed, so
    /// that `scoring::step` stays the single place any weight is floored.
    fn nu_eta_derivatives(
        &self,
        s: &Standardized,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<(Array1<f64>, Array1<f64>), GamlssError> {
        let nu = require(self, params, "nu")?;
        let z_sq = &s.z_sq;
        let w_robust = &s.w_robust;

        // ν score involves digamma differences.
        let nu_plus_1_half = nu.mapv(|nu_i| (nu_i + 1.0) / 2.0);
        let nu_half = nu.mapv(|nu_i| nu_i / 2.0);
        let d1 = digamma_batch(&nu_plus_1_half);
        let d2 = digamma_batch(&nu_half);

        let term3 = par_zip_map(nu, z_sq, |nu_i, z2_i| (1.0 + z2_i / nu_i).ln());
        let term4 = par_zip3_map(nu, w_robust, z_sq, |nu_i, w_i, z2_i| {
            (w_i * z2_i - 1.0) / nu_i
        });

        let dl_dnu = 0.5 * (&d1 - &d2 - &term3 + &term4);
        // Chain rule for log link: u_η = ν · dl/dν, with an *aggregate* boundary
        // projection at the ν-floor. Where `FlooredLogLink` binds (ν pinned at
        // NU_FLOOR), dν/dη is genuinely 0, so per-row scores must not be forwarded
        // blindly: a negative aggregate walks η_ν downward forever (Δβ never
        // converges), while a per-row one-sided projection biases the aggregate
        // upward and produces a limit cycle of lift-off/fall-back at the boundary.
        // The KKT-correct rule uses the *summed* score over the pinned rows: if it
        // is ≤ 0 the constrained optimum is at the boundary: freeze those rows
        // (u = 0) so the block reports a zero step and the loop converges; if it
        // is > 0 the fit should re-enter the interior: forward the full chain
        // rule so the aggregate pull is preserved.
        //
        // Scope: the single summed score is the exact KKT test only when η_ν is
        // an intercept (the standard TF usage, and all this crate's ν formulas
        // in practice). Under a covariate/smooth model on ν the pinned rows
        // load on different coefficients and a per-coefficient projected
        // gradient X'g⁺ would be needed; theta_derivatives() has no design-matrix
        // access, so that refinement belongs in the scoring layer if ν
        // covariates become a supported pattern.
        let pinned_tol = NU_FLOOR * (1.0 + 1e-9);
        let pinned_score: f64 = nu
            .iter()
            .zip(dl_dnu.iter())
            .filter(|(&nu_i, _)| nu_i <= pinned_tol)
            .map(|(_, &g_i)| g_i)
            .sum();
        let boundary_frozen = pinned_score <= 0.0;
        let u_nu = par_zip_map(nu, &dl_dnu, |nu_i, g_i| {
            if boundary_frozen && nu_i <= pinned_tol {
                0.0
            } else {
                nu_i * g_i
            }
        });

        // Expected Fisher information for ν (Lange–Little–Taylor 1989; identical to
        // gamlss TF's d2ldv2):
        //   I_ν = ¼·[ψ'(ν/2) − ψ'((ν+1)/2) − 2(ν+5)/(ν(ν+1)(ν+3))].
        // Verified against Monte-Carlo E[(ν·dl/dν)²]. The rational term is a small
        // correction to a near-cancellation: I_ν decays like O(1/ν³), so a sign or
        // degree error there inflates the weight by orders of magnitude and
        // effectively freezes ν at its seed.
        let t1 = trigamma_batch(&nu_half);
        let t2 = trigamma_batch(&nu_plus_1_half);
        let t3: Array1<f64> =
            nu.mapv(|nu_i| (2.0 * (nu_i + 5.0)) / (nu_i * (nu_i + 1.0) * (nu_i + 3.0)));
        let i_nu = 0.25 * (&t1 - &t2 - &t3);
        // For the log link `w_η = I_ν · ν²`. Returned unfloored, like every other
        // weight: `scoring::step` applies `MIN_WEIGHT` once, and its `<` test catches
        // the negative values the trigamma near-cancellation can produce as well.
        let w_nu = par_zip_map(&i_nu, nu, |i, nu_i| i * nu_i * nu_i);

        Ok((u_nu, w_nu))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::test_helpers::{
        check_cdf_monotone_in_unit, check_cdf_pdf_consistency, check_cdf_quantile_roundtrip,
        check_cdf_theta_derivatives_via_finite_diff, check_eta_score_via_finite_diff,
        check_score_via_finite_diff, default_link_derivatives, derivative_keys_match_parameters,
        finite_array, no_nan_array, params_view,
    };
    use crate::distributions::{InverseLink, SqrtLink};
    use ndarray::array;

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
    fn initial_value_is_robust_to_outliers() {
        // A clean core around 10 with a few gross outliers. The non-robust trait
        // default (mean/SD) would be dragged toward the outliers; median/MAD resist.
        let y = array![9.8, 10.1, 9.9, 10.2, 10.0, 9.7, 10.3, 9.95, 10.05, 1000.0, -800.0];
        let mu0 = StudentT.initial_value("mu", &y);
        assert!(
            (mu0 - 10.0).abs() < 0.5,
            "median seed should sit near the core (got {mu0})"
        );
        let sigma0 = StudentT.initial_value("sigma", &y);
        assert!(
            sigma0 > 0.0 && sigma0 < 2.0,
            "MAD-based scale seed should reflect the core spread, not the outliers (got {sigma0})"
        );
        let nu0 = StudentT.initial_value("nu", &y);
        assert_eq!(
            nu0, NU_INIT,
            "ν seed is a fixed moderate default, not derived from the (outlier-sensitive) kurtosis"
        );
    }

    #[test]
    fn nu_link_floors_below_two() {
        // The floored log link must keep ν ≥ 2 regardless of how negative η drifts,
        // so the variance σ²ν/(ν−2) stays finite during iteration.
        let link = StudentT.default_link("nu").unwrap();
        assert!(link.inv_link(-50.0) >= NU_FLOOR - 1e-12);
        assert!(link.inv_link(-1.0) >= NU_FLOOR - 1e-12);
        // Above the floor it behaves like a plain log link.
        assert!((link.inv_link(2.0_f64.ln()) - 2.0).abs() < 1e-9);
        assert!((link.inv_link(10.0_f64.ln()) - 10.0).abs() < 1e-9);
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
    fn score_matches_finite_diff_under_non_default_links() {
        // The Altitude #1 gate, for the two parameters that went through the generic
        // chain rule. μ is identity-linked, so the default-link check above is
        // vacuous for it; a log link is what makes it bite (μ is positive here so
        // the link is well defined).
        //
        // ν is deliberately absent: it keeps its hand-written η-scale entry against
        // `FlooredLogLink`, and overriding its link is rejected as unsupported (see
        // `Distribution::eta_derivatives` on this type).
        let y = array![-1.0, 0.5, 2.0];
        let owned = [
            ("mu", array![0.5, 0.75, 1.0]),
            ("sigma", array![1.0, 1.2, 0.8]),
            ("nu", array![5.0, 8.0, 4.0]),
        ];
        check_eta_score_via_finite_diff(&StudentT, &y, &owned, "mu", &LogLink, 1e-5);
        check_eta_score_via_finite_diff(&StudentT, &y, &owned, "mu", &SqrtLink, 1e-5);
        check_eta_score_via_finite_diff(&StudentT, &y, &owned, "sigma", &SqrtLink, 1e-5);
        check_eta_score_via_finite_diff(&StudentT, &y, &owned, "sigma", &InverseLink, 1e-5);
    }

    #[test]
    fn the_nu_block_shares_the_guarded_sigma_with_the_mu_sigma_block() {
        // The ν block used to rebuild `z` from a raw σ where `mu_sigma_derivatives`
        // divides by the `DENOM_FLOOR`-guarded reciprocal. On an exactly-fitting row
        // (y = μ) with a collapsed σ that is `0/0` = NaN against the guarded form's
        // `0 · 1e300` = 0, so the ν score alone went NaN while μ and σ stayed finite.
        let y = array![1.0, 2.0];
        let owned = [
            ("mu", array![1.0, 2.0]),
            ("sigma", array![0.0, 1.0]),
            ("nu", array![5.0, 5.0]),
        ];
        let p = params_view(&owned);
        let chained = default_link_derivatives(&StudentT, &y, &p).unwrap();
        for name in ["mu", "sigma", "nu"] {
            let (u, w) = &chained[name];
            assert!(
                finite_array(u) && finite_array(w),
                "{name}: u={u:?} w={w:?}"
            );
        }
    }

    #[test]
    fn cdf_theta_derivatives_stay_finite_at_an_overflowing_z() {
        // As in `gaussian.rs`: at a far-out bound against a floored σ, z overflows and
        // `g' = −g(ν+1)z/(ν+z²)` is ∞/∞ = NaN rather than the 0 it tends to.
        let bounds = array![1e300, -1e300, 1e200];
        let owned = [
            ("mu", array![0.0, 0.0, 0.0]),
            ("sigma", array![1e-10, 1e-10, 1.0]),
            ("nu", array![5.0, 5.0, 5.0]),
        ];
        let p = params_view(&owned);
        let d = StudentT.cdf_theta_derivatives(&bounds, &p).unwrap();
        for name in ["mu", "sigma"] {
            let (d1, d2) = &d[name];
            assert!(
                finite_array(d1) && finite_array(d2),
                "{name}: {d1:?} {d2:?}"
            );
            assert_eq!((d1[0], d2[0]), (0.0, 0.0), "{name} row 0");
        }
    }

    #[test]
    fn derivatives_stay_finite_at_a_saturated_sigma() {
        // Un-folding introduces `1/σ` and `1/σ²` that the previous η-scale forms
        // cancelled.
        let y = array![-1.0, 0.5, 2.0];
        let owned = [
            ("mu", array![0.0, 0.5, 1.0]),
            // Spans well past `exp(MIN_ETA) ≈ 9.4e-14`, the smallest σ a log link
            // reaches inside its own η clamp. σ = 0 exactly is excluded: there
            // `z = (y−μ)/σ` overflows and `z²` becomes infinite, so `w_robust · z²`
            // is `0 · ∞ = NaN`. That predates Altitude #1 (the old `z` overflowed
            // identically) and is a separate fix.
            ("sigma", array![1e-100, 1e-13, 1e-8]),
            ("nu", array![5.0, 8.0, 4.0]),
        ];
        let p = params_view(&owned);
        let natural = StudentT.theta_derivatives(&y, &p).unwrap();
        let chained = default_link_derivatives(&StudentT, &y, &p).unwrap();
        for name in ["mu", "sigma"] {
            let (u_n, i_n) = &natural[name];
            assert!(no_nan_array(u_n) && no_nan_array(i_n), "natural {name}");
        }
        for name in ["mu", "sigma", "nu"] {
            let (u, w) = &chained[name];
            assert!(finite_array(u) && finite_array(w), "chained {name}: {u:?}");
        }
    }

    #[test]
    fn natural_derivatives_omit_nu_but_eta_derivatives_supply_it() {
        // The hybrid contract: `theta_derivatives` covers only the two separable
        // parameters, and `eta_derivatives` fills ν back in. A family that silently
        // dropped ν from the η-scale map would freeze the ν block with no error.
        let y = array![-1.0, 0.5, 2.0];
        let owned = [
            ("mu", array![0.0, 0.5, 1.0]),
            ("sigma", array![1.0, 1.2, 0.8]),
            ("nu", array![5.0, 8.0, 4.0]),
        ];
        let p = params_view(&owned);
        let natural = StudentT.theta_derivatives(&y, &p).unwrap();
        let mut keys: Vec<&str> = natural.keys().map(String::as_str).collect();
        keys.sort();
        assert_eq!(keys, ["mu", "sigma"]);

        let chained = default_link_derivatives(&StudentT, &y, &p).unwrap();
        let mut keys: Vec<&str> = chained.keys().map(String::as_str).collect();
        keys.sort();
        assert_eq!(keys, ["mu", "nu", "sigma"]);
    }

    #[test]
    fn cdf_theta_derivatives_match_finite_diff_studentt() {
        // μ and σ are analytic; ν is intentionally absent (numeric fallback).
        let y = array![-2.0, 0.3, 1.4, 3.0];
        let owned = [
            ("mu", array![0.0, 0.5, 1.0, 2.0]),
            ("sigma", array![1.0, 1.2, 0.9, 1.4]),
            ("nu", array![5.0, 8.0, 6.0, 12.0]),
        ];
        check_cdf_theta_derivatives_via_finite_diff(&StudentT, &y, &owned, "mu", 2e-4);
        check_cdf_theta_derivatives_via_finite_diff(&StudentT, &y, &owned, "sigma", 2e-4);
        // ν must not be supplied analytically.
        let p = params_view(&owned);
        let derivs = StudentT.cdf_theta_derivatives(&y, &p).unwrap();
        assert!(!derivs.contains_key("nu"));
    }

    #[test]
    fn cdf_theta_derivatives_stay_finite_at_a_saturated_sigma() {
        // Same exposure as Gaussian's: un-folding σ introduced a `1/σ` and a `1/σ²`
        // the η-scale forms did not have. ν is swept alongside σ because `g` and
        // `g'` both carry it. σ = 0 exactly is excluded here for the reason Phase 2b
        // recorded: `z = (y−μ)/σ` overflows there and `w_robust · z²` is a `0 · ∞`
        // NaN, a fragility that predates this work and is unchanged by it.
        let y = array![0.0, 1.0, 2.0, -1.0];
        let owned = [
            ("mu", array![0.0, 0.0, 0.0, 0.0]),
            ("sigma", array![1e-320, 1e-8, 1e13, 1.0]),
            ("nu", array![2.0, 1e-8, 1e13, 2.0]),
        ];
        let p = params_view(&owned);
        let d = StudentT.cdf_theta_derivatives(&y, &p).unwrap();
        for name in ["mu", "sigma"] {
            let (d1, d2) = &d[name];
            assert!(
                finite_array(d1) && finite_array(d2),
                "{name}: {d1:?} {d2:?}"
            );
        }
    }

    #[test]
    fn cdf_quantile_roundtrip_studentt() {
        let y = array![-3.0, -0.5, 0.0, 1.2, 4.0];
        let owned = [
            ("mu", array![0.0, 0.5, 1.0, 2.0, 1.5]),
            ("sigma", array![1.0, 1.5, 0.8, 2.0, 1.2]),
            ("nu", array![5.0, 10.0, 4.0, 8.0, 30.0]),
        ];
        check_cdf_quantile_roundtrip(&StudentT, &y, &owned, 1e-6);
        check_cdf_pdf_consistency(&StudentT, &y, &owned, 1e-4, 1e-3);
    }

    #[test]
    fn cdf_monotone_studentt_and_median_is_mu() {
        let grid = Array1::from_iter((0..60).map(|i| -8.0 + i as f64 * 0.25));
        let owned = [
            ("mu", array![1.0]),
            ("sigma", array![1.3]),
            ("nu", array![6.0]),
        ];
        check_cdf_monotone_in_unit(&StudentT, &grid, &owned);
        let p = params_view(&owned);
        let med = StudentT.quantile(&array![0.5], &p).unwrap();
        assert!((med[0] - 1.0).abs() < 1e-7);
    }
}
