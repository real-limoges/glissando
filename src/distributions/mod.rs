//! Probability distributions for GAMLSS models.
//!
//! Each distribution defines its parameter names (μ, σ, ν, …), default link functions,
//! and the score / Fisher-information pairs that drive the Rigby–Stasinopoulos
//! IRLS update. Derivatives are batched (vectorized over observations).

use crate::error::GamlssError;
use ndarray::Array1;
use std::collections::HashMap;
use std::fmt::Debug;

mod links;
pub use links::{
    link_from_name, CauchitLink, CloglogLink, FlooredLogLink, IdentityLink, InverseLink,
    InverseSquareLink, Link, LogLink, LogitLink, ProbitLink, SqrtLink,
};
// Re-export at crate-internal scope so submodules can `use super::MIN_POSITIVE`
// after the move without breaking. MAX_ETA/MIN_ETA are link-internal today and
// only re-exported here so any future submodule can opt in without a separate edit.
#[allow(unused_imports)]
pub(crate) use links::{MAX_ETA, MIN_ETA, MIN_POSITIVE};

/// Lower bound on Fisher-information weights to keep `W` positive definite.
pub(crate) const MIN_WEIGHT: f64 = 1e-6;

// ============================================================================
// Distribution trait
// ============================================================================

/// Score / Fisher-information pairs keyed by distribution-parameter name.
pub type DerivativesResult = Result<HashMap<String, (Array1<f64>, Array1<f64>)>, GamlssError>;

/// Per-parameter `(∂F/∂η, ∂²F/∂η²)` pairs keyed by parameter name — the same
/// shape as a derivatives map, used by the structural wrappers (SER-1 / STRUCT).
pub type CdfEtaMap = HashMap<String, (Array1<f64>, Array1<f64>)>;
/// Result wrapper around [`CdfEtaMap`], returned by [`Distribution::cdf_eta_derivatives`].
pub type CdfEtaResult = Result<CdfEtaMap, GamlssError>;

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

    /// Whether the response is discrete (counts / categories) rather than
    /// continuous. Drives the randomized branch of quantile residuals (INFER-1),
    /// which needs both `F(y)` and `F(y−1)` to de-lump each atom. Default: continuous.
    fn is_discrete(&self) -> bool {
        false
    }

    /// Density (continuous) or mass (discrete) `f(y_i | params_i)`.
    ///
    /// Default: `exp(loglik_pointwise)`, correct for every family whose
    /// `loglik_pointwise` returns a true (normalized) log-density/-mass.
    fn pdf(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        Ok(self.loglik_pointwise(y, params)?.mapv(f64::exp))
    }

    /// Cumulative distribution `F(y_i | params_i) = P(Y ≤ y_i)`, vectorized over rows.
    ///
    /// For discrete families this is the right-continuous step CDF evaluated at
    /// `⌊y⌋`, so that `cdf(k) − cdf(k−1) = pdf(k)` at integer support points.
    fn cdf(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError>;

    /// Analytic first and second derivatives of the CDF `F(y_i)` with respect to
    /// each parameter's linear predictor η, returned per parameter as
    /// `(∂F/∂η, ∂²F/∂η²)`, vectorized over rows.
    ///
    /// Only parameters with a closed form are included — the default returns an
    /// empty map. The structural wrappers ([`Censored`] / [`Truncated`]) build
    /// the censoring / truncation score and observed-information weight from
    /// these, and fall back to a central difference of [`Self::cdf`] (perturbed
    /// on the parameter's η via its [`Self::default_link`]) for any parameter a
    /// family omits. Location/scale parameters are analytic (Gaussian μ/σ,
    /// Student-t μ/σ, Gamma μ); shape parameters whose CDF derivative is
    /// non-elementary (Gamma σ, Student-t ν, both Beta params) are left to the
    /// numeric fallback. See the structural-likelihoods guide.
    fn cdf_eta_derivatives(
        &self,
        _y: &Array1<f64>,
        _params: &HashMap<&str, &Array1<f64>>,
    ) -> CdfEtaResult {
        Ok(HashMap::new())
    }

    /// Inverse CDF / quantile: smallest `y` with `F(y) ≥ p_i`, element-wise over
    /// `p ∈ (0, 1)`. Returns response-scale values.
    fn quantile(
        &self,
        p: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError>;

    /// Stable distribution name (e.g. `"Gaussian"`); used in error messages and
    /// for the WASM `from_name` lookup.
    fn name(&self) -> &'static str;

    /// A serializable description of this family, sufficient to rebuild it via
    /// [`FamilyDescriptor::build`] (SER-1).
    ///
    /// The default — `FamilyDescriptor::Named(self.name())` — round-trips every
    /// stateless family through [`from_name`]. Stateful families ([`Binomial`],
    /// [`Ocat`]) and the structural wrappers ([`Censored`] / [`Truncated`] /
    /// [`Hurdle`]), which a bare name cannot reconstruct, override it to carry
    /// their per-observation state and (recursively) their base family.
    fn descriptor(&self) -> FamilyDescriptor {
        FamilyDescriptor::Named(self.name().to_string())
    }

    /// Initial response-scale value for a parameter, used to seed the IRLS loop.
    /// Override for distributions where `y` is not directly a sample of the parameter.
    fn initial_value(&self, param: &str, y: &Array1<f64>) -> f64 {
        // `validate_inputs` rejects empty `y` before fitting, so `mean` returning `None`
        // is unreachable on the public path. `expect` makes the precondition explicit
        // rather than masking it with a Gaussian-flavored fallback that would corrupt
        // IRLS init for non-Gaussian families (Poisson, Gamma, etc.).
        match param {
            "mu" => y.mean().expect("validate_inputs rejects empty y"),
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
pub(crate) fn require<'a, D: Distribution + ?Sized>(
    dist: &D,
    params: &HashMap<&str, &'a Array1<f64>>,
    name: &str,
) -> Result<&'a Array1<f64>, GamlssError> {
    params
        .get(name)
        .copied()
        .ok_or_else(|| dist.unknown_param(name))
}

/// Smallest non-negative integer `k` (returned as `f64`) with `cdf_at(k) ≥ p`,
/// for a monotone discrete CDF. Doubling bracket then bisection keeps it
/// `O(log k)` even for large means. Shared by the discrete families' `quantile`.
pub(crate) fn discrete_quantile(p: f64, cdf_at: impl Fn(u64) -> f64) -> f64 {
    if p <= cdf_at(0) {
        return 0.0;
    }
    let (mut lo, mut hi) = (0u64, 1u64);
    while cdf_at(hi) < p {
        lo = hi;
        hi = hi.saturating_mul(2);
        // Guard against an unreachable target (p ~ 1 with a capped CDF): once `hi`
        // stops growing we have bracketed as far as the integer range allows.
        if hi == lo {
            return hi as f64;
        }
    }
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if cdf_at(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi as f64
}

// ============================================================================
// Distribution implementations
// ============================================================================

mod bccg;
mod bcpe;
mod bct;
mod beta;
mod binomial;
mod boxcox;
mod censored;
mod descriptor;
mod gamma;
mod gaussian;
mod hurdle;
mod negative_binomial;
mod ocat;
mod poisson;
mod structural;
mod student_t;
mod truncated;
mod weibull;

pub use bccg::BCCG;
pub use bcpe::BCPE;
pub use bct::BCT;
pub use beta::Beta;
pub use binomial::Binomial;
pub use censored::{CensorStatus, Censored};
pub use descriptor::FamilyDescriptor;
pub use gamma::Gamma;
pub use gaussian::Gaussian;
pub use hurdle::Hurdle;
pub use negative_binomial::NegativeBinomial;
pub use ocat::Ocat;
pub use poisson::Poisson;
pub use student_t::StudentT;
pub use truncated::Truncated;
pub use weibull::Weibull;

/// Construct a stateless distribution from its name (e.g. for WASM JSON I/O).
///
/// Excludes [`Binomial`] and [`Ocat`] because they carry state (`n_trials` /
/// `n_categories`) that cannot be recovered from the name string alone.
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
        "BCCG" => Ok(Box::new(BCCG)),
        "BCT" => Ok(Box::new(BCT)),
        "BCPE" => Ok(Box::new(BCPE)),
        other => Err(GamlssError::Input(format!(
            "Unknown distribution: '{}'. Supported: Gaussian, Poisson, StudentT, Gamma, NegativeBinomial, Beta, BCCG, BCT, BCPE",
            other
        ))),
    }
}

#[cfg(test)]
pub(crate) mod test_helpers {
    use super::*;

    pub fn finite_array(a: &Array1<f64>) -> bool {
        a.iter().all(|v| v.is_finite())
    }

    pub fn derivative_keys_match_parameters<D: Distribution>(
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

    /// Build a `params` view from owned arrays for test ergonomics.
    pub fn params_view<'a>(
        owned: &'a [(&'static str, Array1<f64>)],
    ) -> HashMap<&'a str, &'a Array1<f64>> {
        owned.iter().map(|(k, v)| (*k, v)).collect()
    }

    /// Check that the analytic score `u` returned by `derivatives()` matches the
    /// central difference of `loglik_pointwise` for `target` on the η-scale.
    pub fn check_score_via_finite_diff<D: Distribution + ?Sized>(
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

    /// Check the analytic `(∂F/∂η, ∂²F/∂η²)` returned by [`Distribution::cdf_eta_derivatives`]
    /// for `target` against central differences of [`Distribution::cdf`] on the
    /// η-scale. Skips silently if the family does not supply `target` analytically
    /// (the wrapper's numeric fallback covers those parameters).
    pub fn check_cdf_eta_derivatives_via_finite_diff<D: Distribution + ?Sized>(
        family: &D,
        y: &Array1<f64>,
        owned: &[(&'static str, Array1<f64>)],
        target: &str,
        tol: f64,
    ) {
        let p: HashMap<&str, &Array1<f64>> = owned.iter().map(|(k, v)| (*k, v)).collect();
        let derivs = family.cdf_eta_derivatives(y, &p).unwrap();
        let Some((analytic_d1, analytic_d2)) = derivs.get(target) else {
            return; // family leaves this parameter to the numeric fallback
        };

        let link = family.default_link(target).unwrap();
        let eps: f64 = 1e-5;
        let idx = owned.iter().position(|(k, _)| *k == target).unwrap();
        let mut perturbed: Vec<(&'static str, Array1<f64>)> =
            owned.iter().map(|(k, v)| (*k, v.clone())).collect();

        for i in 0..y.len() {
            let mu_orig = owned[idx].1[i];
            let eta = link.link(mu_orig);

            let f0 = {
                let pv: HashMap<&str, &Array1<f64>> =
                    perturbed.iter().map(|(k, v)| (*k, v)).collect();
                family.cdf(y, &pv).unwrap()[i]
            };
            perturbed[idx].1[i] = link.inv_link(eta + eps);
            let f_plus = {
                let pv: HashMap<&str, &Array1<f64>> =
                    perturbed.iter().map(|(k, v)| (*k, v)).collect();
                family.cdf(y, &pv).unwrap()[i]
            };
            perturbed[idx].1[i] = link.inv_link(eta - eps);
            let f_minus = {
                let pv: HashMap<&str, &Array1<f64>> =
                    perturbed.iter().map(|(k, v)| (*k, v)).collect();
                family.cdf(y, &pv).unwrap()[i]
            };
            perturbed[idx].1[i] = mu_orig;

            let numeric_d1 = (f_plus - f_minus) / (2.0 * eps);
            let numeric_d2 = (f_plus - 2.0 * f0 + f_minus) / (eps * eps);

            let s1 = analytic_d1[i].abs().max(1.0);
            assert!(
                (analytic_d1[i] - numeric_d1).abs() / s1 < tol,
                "{}::{} obs {}: ∂F/∂η analytic={:.6e} numeric={:.6e}",
                family.name(),
                target,
                i,
                analytic_d1[i],
                numeric_d1
            );
            let s2 = analytic_d2[i].abs().max(1.0);
            assert!(
                (analytic_d2[i] - numeric_d2).abs() / s2 < tol,
                "{}::{} obs {}: ∂²F/∂η² analytic={:.6e} numeric={:.6e}",
                family.name(),
                target,
                i,
                analytic_d2[i],
                numeric_d2
            );
        }
    }

    /// Continuous round-trip: `Q(F(y)) ≈ y` element-wise.
    pub fn check_cdf_quantile_roundtrip<D: Distribution + ?Sized>(
        d: &D,
        y: &Array1<f64>,
        owned: &[(&'static str, Array1<f64>)],
        tol: f64,
    ) {
        let p = params_view(owned);
        let u = d.cdf(y, &p).unwrap();
        let back = d.quantile(&u, &p).unwrap();
        for i in 0..y.len() {
            assert!(
                (back[i] - y[i]).abs() < tol,
                "{} obs {}: Q(F(y))={:.6e} y={:.6e} (F={:.6e})",
                d.name(),
                i,
                back[i],
                y[i],
                u[i]
            );
        }
    }

    /// Continuous CDF↔pdf consistency: central difference `dF/dy ≈ pdf(y)`.
    pub fn check_cdf_pdf_consistency<D: Distribution + ?Sized>(
        d: &D,
        y: &Array1<f64>,
        owned: &[(&'static str, Array1<f64>)],
        h: f64,
        tol: f64,
    ) {
        let p = params_view(owned);
        let pdf = d.pdf(y, &p).unwrap();
        let f_plus = d.cdf(&(y + h), &p).unwrap();
        let f_minus = d.cdf(&(y - h), &p).unwrap();
        for i in 0..y.len() {
            let numeric = (f_plus[i] - f_minus[i]) / (2.0 * h);
            let scale = pdf[i].abs().max(1.0);
            assert!(
                (numeric - pdf[i]).abs() / scale < tol,
                "{} obs {}: dF/dy={:.6e} pdf={:.6e}",
                d.name(),
                i,
                numeric,
                pdf[i]
            );
        }
    }

    /// Discrete consistency at integer support: `cdf(k) − cdf(k−1) ≈ pdf(k)`.
    pub fn check_discrete_cdf_matches_pmf<D: Distribution + ?Sized>(
        d: &D,
        ks: &Array1<f64>,
        owned: &[(&'static str, Array1<f64>)],
        tol: f64,
    ) {
        let p = params_view(owned);
        let pmf = d.pdf(ks, &p).unwrap();
        let hi = d.cdf(ks, &p).unwrap();
        let lo = d.cdf(&(ks - 1.0), &p).unwrap();
        for i in 0..ks.len() {
            let jump = hi[i] - lo[i];
            assert!(
                (jump - pmf[i]).abs() < tol,
                "{} obs {} (k={}): cdf(k)−cdf(k−1)={:.6e} pmf={:.6e}",
                d.name(),
                i,
                ks[i],
                jump,
                pmf[i]
            );
        }
    }

    /// CDF range/monotonicity: `cdf ∈ [0,1]` and non-decreasing in `y` over the
    /// supplied (ascending, single-observation) grid, plus tail anchors.
    pub fn check_cdf_monotone_in_unit<D: Distribution + ?Sized>(
        d: &D,
        grid: &Array1<f64>,
        owned: &[(&'static str, Array1<f64>)],
    ) {
        let n = grid.len();
        // Broadcast the single-row params across the grid.
        let broadcast: Vec<(&'static str, Array1<f64>)> = owned
            .iter()
            .map(|(k, v)| (*k, Array1::from_elem(n, v[0])))
            .collect();
        let p = params_view(&broadcast);
        let f = d.cdf(grid, &p).unwrap();
        for i in 0..n {
            assert!(
                (-1e-9..=1.0 + 1e-9).contains(&f[i]),
                "{} obs {}: cdf={} out of [0,1]",
                d.name(),
                i,
                f[i]
            );
            if i > 0 {
                assert!(
                    f[i] >= f[i - 1] - 1e-9,
                    "{} cdf not monotone at {}: {} < {}",
                    d.name(),
                    i,
                    f[i],
                    f[i - 1]
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Link-function tests live alongside the link impls in `links.rs`.

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
            "BCCG",
            "BCT",
            "BCPE",
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

    // --- initial_value (cross-distribution) ---

    #[test]
    fn initial_value_finite_for_typical_parameters() {
        let y = ndarray::array![1.0, 2.0, 3.0, 4.0];
        for d in [
            from_name("Gaussian").unwrap(),
            from_name("Poisson").unwrap(),
            from_name("StudentT").unwrap(),
            from_name("Gamma").unwrap(),
            from_name("Beta").unwrap(),
            from_name("NegativeBinomial").unwrap(),
            from_name("BCCG").unwrap(),
            from_name("BCT").unwrap(),
            from_name("BCPE").unwrap(),
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
}
