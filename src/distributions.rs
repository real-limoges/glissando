//! Probability distributions for GAMLSS models.
//!
//! Each distribution defines its parameter names (μ, σ, ν, …), default link functions,
//! and the score / Fisher-information pairs that drive the Rigby–Stasinopoulos
//! IRLS update. Derivatives are batched (vectorized over observations).

use crate::error::GamlssError;
use ndarray::{Array1, Zip};
use std::collections::HashMap;
use std::fmt::Debug;

mod links;
pub use links::{
    link_from_name, CauchitLink, CloglogLink, FlooredLogLink, IdentityLink, InverseLink,
    InverseSquareLink, Link, LinkContext, LogLink, LogitLink, ProbitLink, SqrtLink,
};
// Re-exported at crate-internal scope so submodules can `use super::MIN_POSITIVE`
// after the move without breaking. MAX_ETA/MIN_ETA are link-internal today. I
// re-export them here anyway so a future submodule can opt in without a separate edit.
#[allow(unused_imports)]
pub(crate) use links::{MAX_ETA, MIN_ETA, MIN_POSITIVE};

/// Lower bound on Fisher-information weights to keep `W` positive definite.
pub(crate) const MIN_WEIGHT: f64 = 1e-6;

/// Floor for a denominator that the η-scale chain rule multiplies straight back out.
///
/// Natural-scale [`Distribution::theta_derivatives`] bodies divide by quantities the old
/// folded η-scale forms canceled algebraically (`1/μ`, `1/σ`, `1/(μ(1−μ))`). Those
/// divisions have to stay finite, but here is the catch: the guard must never *bind*
/// for any θ a link can actually produce. If it did, it would disagree with the
/// `mu_eta` that [`chain_to_eta`] multiplies back in, and the product would no longer
/// telescope. That is why this is a floor on the *denominator* and not a clamp on θ,
/// and why it is so much smaller than [`MIN_POSITIVE`].
///
/// `1e-300` leaves both margins comfortable. It sits roughly 290 orders of
/// magnitude below anything a built-in link yields inside its own η clamp
/// (`exp(MIN_ETA) ≈ 9.4e-14`), and still leaves about seven decades of headroom
/// before `numerator / DENOM_FLOOR` overflows for a numerator of realistic size.
pub(crate) const DENOM_FLOOR: f64 = 1e-300;

/// Shared probability/mass floor: values are clamped into `[PROB_EPS, 1 − PROB_EPS]`
/// before a logarithm or division, so a saturated tail can't produce `−inf` / NaN.
/// Used by the structural wrappers ([`Censored`]/[`Truncated`]/[`Hurdle`]) and by
/// quantile functions that invert a `[0, 1]` probability. [`Ocat`]'s `MIN_PROB` is
/// a distinct, larger floor kept local to that file; see its doc comment.
pub(crate) const PROB_EPS: f64 = 1e-12;

/// Clamp a probability into `[PROB_EPS, 1 − PROB_EPS]` so its logarithm or
/// reciprocal stays finite.
pub(crate) fn clamp_prob(p: f64) -> f64 {
    p.clamp(PROB_EPS, 1.0 - PROB_EPS)
}

/// Floor for a Gamma-function argument whose *trigamma* is taken.
///
/// `ψ'(x) ~ 1/x²`, so it overflows to infinity below `x ≈ 1e-154` even though
/// `ψ(x) ~ −1/x` is still finite there. Families whose natural scale feeds a
/// shape parameter into `trigamma_batch` ([`Beta`]'s `α = μφ` and `β = (1−μ)φ`)
/// floor the *argument* here rather than clamping μ, for the same reason
/// [`DENOM_FLOOR`] is a floor on a denominator: at `1e-150` it sits ~140 orders
/// of magnitude below anything a link produces inside its own η clamp, so it
/// cannot bind where the chain rule still has to telescope.
pub(crate) const TRIGAMMA_FLOOR: f64 = 1e-150;

/// Replace an overflowed `±∞` with the largest finite `f64` of the same sign.
///
/// A natural-scale derivative can genuinely diverge at a saturated θ (Weibull's
/// `σ(z−1)/μ` as `μ → 0`), and there is no representable value to return. A large
/// finite magnitude drives the step in the right direction and lets step-halving
/// in `scoring::step` pull the iterate back; an infinity instead reaches the PWLS
/// solve as `inf/inf` and produces NaN.
///
/// NaN is deliberately passed through. [`chain_to_eta`] removes the one way a
/// well-formed family body produces one (`inf · 0`, below), so a NaN arriving here
/// is a bug in that body. I want it to surface, not get papered over.
fn saturate(v: f64) -> f64 {
    if v.is_infinite() {
        f64::MAX.copysign(v)
    } else {
        v
    }
}

// ============================================================================
// Distribution trait
// ============================================================================

/// Score / Fisher-information pairs keyed by distribution-parameter name.
pub type DerivativesResult = Result<HashMap<String, (Array1<f64>, Array1<f64>)>, GamlssError>;

/// Per-parameter `(∂F/∂η, ∂²F/∂η²)` pairs keyed by parameter name: the same
/// shape as a derivatives map, used by the structural wrappers (SER-1 / STRUCT).
///
/// This is the *chained* map, produced by `structural::cdf_eta_grads`. A family's
/// own [`Distribution::cdf_theta_derivatives`] returns the natural-scale
/// [`CdfThetaMap`] instead.
pub type CdfEtaMap = HashMap<String, (Array1<f64>, Array1<f64>)>;
/// Result wrapper around [`CdfEtaMap`].
pub type CdfEtaResult = Result<CdfEtaMap, GamlssError>;

/// Per-parameter `(∂F/∂θ, ∂²F/∂θ²)` pairs: the *natural-scale* counterpart of
/// [`CdfEtaMap`], returned by [`Distribution::cdf_theta_derivatives`].
///
/// Structurally identical to [`CdfEtaMap`]; the two are distinct names so a
/// reader can tell at a glance which side of the chain rule a value sits on.
pub type CdfThetaMap = HashMap<String, (Array1<f64>, Array1<f64>)>;
/// Result wrapper around [`CdfThetaMap`], returned by
/// [`Distribution::cdf_theta_derivatives`].
pub type CdfThetaResult = Result<CdfThetaMap, GamlssError>;

/// Map a natural-scale derivatives map onto the linear-predictor scale:
/// `u_η = mu_eta · ∂l/∂θ` and `w_η = mu_eta² · i_θ`, per parameter.
///
/// This is the generic chain rule every family with a separable natural scale
/// delegates to from [`Distribution::eta_derivatives`], so that a link override
/// selected through [`FitConfig::with_link`](crate::FitConfig::with_link) is
/// honored rather than silently ignored.
///
/// **The returned `w_η` is unfloored, and must stay that way.** `MIN_WEIGHT` is
/// applied exactly once, downstream, in the scoring loop, because
/// `max(mu_eta²·i_θ, F) ≠ mu_eta²·max(i_θ, F)`. Flooring `i_θ` inside a family body
/// first is not a rounding difference: Student-t's `i_ν` decays like `O(ν⁻³)` while
/// `mu_eta² = ν²`, so at ν ≈ 1e6 a pre-floored weight comes out twelve orders of
/// magnitude too large and freezes a block that should be free to drift.
///
/// # Errors
///
/// Returns [`GamlssError::Internal`] if `ctx` holds no entry for one of the map's
/// parameters, or if the link derivatives and the derivative arrays disagree on
/// length.
pub fn chain_to_eta(
    natural: HashMap<String, (Array1<f64>, Array1<f64>)>,
    ctx: &LinkContext,
) -> DerivativesResult {
    natural
        .into_iter()
        .map(|(name, (mut u, mut i))| {
            let mu_eta = ctx.mu_eta(&name)?;
            if mu_eta.len() != u.len() || mu_eta.len() != i.len() {
                return Err(GamlssError::Internal(format!(
                    "chain_to_eta length mismatch for '{}': mu_eta {}, u {}, i {}",
                    name,
                    mu_eta.len(),
                    u.len(),
                    i.len()
                )));
            }
            Zip::from(&mut u)
                .and(&mut i)
                .and(mu_eta)
                .for_each(|u_out, i_out, &me| {
                    if me == 0.0 {
                        // `dμ/dη = 0` freezes the observation. No move in η changes
                        // its μ, so its score and information are exactly zero
                        // however large the natural-scale pair is. Take the product
                        // literally and you get `inf · 0` = NaN for a family whose
                        // natural-scale derivative diverges at a saturated θ, and one
                        // NaN row poisons the entire PWLS solve: `scoring::step`'s
                        // `w < MIN_WEIGHT` and `step > MAX_STEP` tests are both false
                        // for NaN, so nothing downstream catches it. And a hard zero
                        // is reachable, not hypothetical: `SqrtLink::mu_eta(0.0)` and
                        // `LogLink::mu_eta(η ≤ −745)` are both exactly 0.
                        *u_out = 0.0;
                        *i_out = 0.0;
                    } else {
                        *u_out = saturate(*u_out * me);
                        *i_out = saturate(*i_out * me * me);
                    }
                });
            Ok((name, (u, i)))
        })
        .collect()
}

/// Map one parameter's natural-scale CDF derivatives `(∂F/∂θ, ∂²F/∂θ²)` onto the
/// linear-predictor scale.
///
/// This is the **second-order** chain rule, and it is deliberately *not*
/// [`chain_to_eta`]:
///
/// ```text
/// ∂F/∂η   = mu_eta · ∂F/∂θ
/// ∂²F/∂η² = mu_eta² · ∂²F/∂θ² + mu_eta2 · ∂F/∂θ
/// ```
///
/// [`chain_to_eta`] transforms a *score and expected information* pair, where the
/// `mu_eta2 · ∂l/∂θ` term drops out because `E[∂l/∂θ] = 0`. A CDF is not a
/// likelihood, so here the term stays, and dropping it would move every censored,
/// truncated and hurdle model's weights (and with them its standard errors, EDF
/// and λ selection). See [`Link::mu_eta2`].
///
/// Applied per parameter by `structural::cdf_eta_grads`, which is the sole caller.
fn chain_cdf_to_eta(
    d1: &mut Array1<f64>,
    d2: &mut Array1<f64>,
    mu_eta: &Array1<f64>,
    mu_eta2: &Array1<f64>,
    param: &str,
) -> Result<(), GamlssError> {
    if mu_eta.len() != d1.len() || mu_eta.len() != d2.len() {
        return Err(GamlssError::Internal(format!(
            "chain_cdf_to_eta length mismatch for '{}': mu_eta {}, d1 {}, d2 {}",
            param,
            mu_eta.len(),
            d1.len(),
            d2.len()
        )));
    }
    Zip::from(d1)
        .and(d2)
        .and(mu_eta)
        .and(mu_eta2)
        .for_each(|d1_out, d2_out, &me, &me2| {
            // No `me == 0` shortcut here, unlike `chain_to_eta`: the `mu_eta2 · ∂F/∂θ`
            // term survives a zero `mu_eta` and dropping it would be wrong. Both
            // outputs are saturated instead, which is enough because a family's
            // `cdf_theta_derivatives` is required to return finite values.
            *d2_out = saturate(me * me * *d2_out + me2 * *d1_out);
            *d1_out = saturate(*d1_out * me);
        });
    Ok(())
}

/// Implement [`Distribution::eta_derivatives`] by chaining the family's
/// natural-scale [`Distribution::theta_derivatives`] through the generic rule.
///
/// Every family with a separable natural scale uses this; it is the whole adapter.
/// The three that do not ([`Ocat`], [`StudentT`] for its `ν`, and the structural
/// wrappers) write `eta_derivatives` out by hand and document why.
macro_rules! eta_derivatives_via_chain {
    () => {
        fn eta_derivatives(
            &self,
            y: &::ndarray::Array1<f64>,
            params: &::std::collections::HashMap<&str, &::ndarray::Array1<f64>>,
            ctx: &$crate::distributions::LinkContext,
        ) -> $crate::distributions::DerivativesResult {
            $crate::distributions::chain_to_eta(self.theta_derivatives(y, params)?, ctx)
        }
    };
}

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

    /// Whether this parameter accepts a link override from
    /// [`FitConfig::links`](crate::FitConfig::links). Default: `true`.
    ///
    /// A family returns `false` for a parameter whose [`Self::eta_derivatives`]
    /// is written against one specific link and cannot be expressed through the
    /// generic chain rule. Honoring an override there would silently compute the
    /// score and weight for a different link than the one the fit uses for
    /// `η → μ`, so `fit_gamlss` rejects the override instead of accepting it and
    /// producing wrong estimates.
    ///
    /// Only two built-ins refuse: every [`Ocat`] parameter (its `params["mu"]`
    /// holds η rather than μ, and its threshold Jacobian is `exp(η_k)` only
    /// under the log link) and [`StudentT`]'s `nu` (its ν-floor KKT projection
    /// is written against [`FlooredLogLink`], whose
    /// `mu_eta` is a hard zero below the floor).
    ///
    /// A `param` outside [`Self::parameters`] is not this method's problem;
    /// `fit_gamlss` checks membership first, so implementors may answer
    /// arbitrarily for an unknown name.
    fn allows_link_override(&self, _param: &str) -> bool {
        true
    }

    /// Score `∂l/∂θ` and expected information `i_θ` on the **natural parameter
    /// scale**, evaluated against the current `params` snapshot.
    ///
    /// **This is not the whole model's derivative map, and it is not every family's.**
    /// The returned map covers only the parameters with a separable natural scale, so
    /// it may be partial ([`StudentT`] returns μ and σ but not ν) or absent entirely:
    /// the default body is an error, which is what [`Ocat`] and the structural
    /// wrappers rely on so they do not have to invent a natural scale they do not
    /// have. [`Self::eta_derivatives`] is the complete one, and is what the scoring
    /// loop calls.
    ///
    /// It carried the plain name `derivatives` and returned *η-scale* pairs until the
    /// generic-chain-rule refactor (Altitude #1). The rename is the point: an
    /// embedder calling the old name against the new contract would otherwise have
    /// read natural-scale numbers as η-scale ones, or hit the error default at
    /// runtime, with nothing failing at compile time. Same reasoning as the
    /// deliberately-absent default body on [`Self::eta_derivatives`].
    ///
    /// **The returned `i_θ` must be unfloored.** `MIN_WEIGHT` is applied exactly
    /// once, downstream in `scoring::step`, after the chain rule; see
    /// [`chain_to_eta`] for why flooring here instead is not a rounding difference.
    fn theta_derivatives(
        &self,
        _y: &Array1<f64>,
        _params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        Err(GamlssError::Internal(format!(
            "{} has no separable natural-scale derivative; it implements \
             eta_derivatives directly",
            self.name()
        )))
    }

    /// Score `u` and IRLS weight `w` on the linear-predictor scale, for each
    /// parameter, given the link derivatives in `ctx`.
    ///
    /// This is what the Fisher-scoring step calls. Families with a separable
    /// natural scale implement it as
    /// `chain_to_eta(self.theta_derivatives(y, params)?, ctx)`; the rest build the η-scale
    /// quantities directly.
    ///
    /// **There is deliberately no default body.** [`Distribution`] is public, so an
    /// external implementor written against the old η-scale `theta_derivatives` contract
    /// would keep compiling against a defaulted adapter and silently double-chain
    /// to `mu_eta⁴ · i_θ`, with no error at compile time or run time. Requiring the
    /// method turns that into a compile error.
    ///
    /// The returned `w` must be **unfloored**: `MIN_WEIGHT` is applied once,
    /// downstream, in the scoring loop. See [`chain_to_eta`] for why the order
    /// matters.
    fn eta_derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
        ctx: &LinkContext,
    ) -> DerivativesResult;

    /// Whether [`Self::eta_derivatives`] reads [`LinkContext::mu_eta2`]. Default: `false`.
    ///
    /// Only the structural wrappers ([`Censored`] / [`Truncated`] / [`Hurdle`]) do:
    /// they chain a *CDF* rather than a likelihood, and there the `mu_eta2 · ∂F/∂θ`
    /// term does not drop out (see `chain_cdf_to_eta`). Answering `false` lets the
    /// scoring loop build a [`LinkContext::first_order`] context and skip computing a
    /// `d²μ/dη²` array per parameter per step that nothing would read.
    fn needs_second_order_links(&self) -> bool {
        false
    }

    /// Per-observation log-density `log f(y_i | params_i)`, used to assemble the model
    /// log-likelihood and observation-level diagnostics (WAIC, leave-one-out, etc.).
    fn loglik_pointwise(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError>;

    /// Marginal `Var(Y_i | params_i)` on the response scale, used for Pearson residuals.
    ///
    /// Distinct from the Fisher-information weight returned by [`Self::theta_derivatives`],
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
    /// each parameter **on its own natural scale θ**, returned per parameter as
    /// `(∂F/∂θ, ∂²F/∂θ²)`, vectorized over rows.
    ///
    /// **This is a natural-scale contract, not an η-scale one.** Return the
    /// derivative w.r.t. the parameter itself and let the caller apply the link;
    /// `structural::cdf_eta_grads` chains to η generically via
    /// [`Link::mu_eta`] and [`Link::mu_eta2`], so an overridden link is honored
    /// rather than silently ignored (Altitude #1). Baking a default-link chain
    /// rule in here is exactly the bug this contract replaced.
    ///
    /// Only parameters with a closed form are included; the default returns an
    /// empty map. The structural wrappers ([`Censored`] / [`Truncated`] /
    /// [`Hurdle`]) build the censoring / truncation score and
    /// observed-information weight from these, and fall back to a central
    /// difference of [`Self::cdf`] (perturbed on the parameter's live η) for any
    /// parameter a family omits. Location/scale parameters are analytic
    /// (Gaussian μ/σ, Student-t μ/σ, Gamma μ); shape parameters whose CDF
    /// derivative is non-elementary (Gamma σ, Student-t ν, both Beta params) are
    /// left to the numeric fallback. See the structural-likelihoods guide.
    fn cdf_theta_derivatives(
        &self,
        _y: &Array1<f64>,
        _params: &HashMap<&str, &Array1<f64>>,
    ) -> CdfThetaResult {
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
    /// The default, `FamilyDescriptor::Named(self.name())`, round-trips every
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
        "Weibull" => Ok(Box::new(Weibull)),
        "BCCG" => Ok(Box::new(BCCG)),
        "BCT" => Ok(Box::new(BCT)),
        "BCPE" => Ok(Box::new(BCPE)),
        other => Err(GamlssError::Input(format!(
            "Unknown distribution: '{}'. Supported: Gaussian, Poisson, StudentT, Gamma, NegativeBinomial, Beta, Weibull, BCCG, BCT, BCPE",
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

    /// No NaN: the invariant a *natural-scale* derivative map has to hold.
    ///
    /// Finiteness is deliberately not required of
    /// [`Distribution::theta_derivatives`]: a natural-scale score genuinely diverges
    /// as its parameter collapses (Gamma's `(2/σ³)·[… + y/μ]` at μ → 0, Weibull's
    /// `σ(z−1)/μ`), past what an f64 can represent. The old bodies only looked finite
    /// there because they clamped θ, and that clamp is exactly what this refactor
    /// removed to keep the chain rule telescoping. [`chain_to_eta`] is the single
    /// place finiteness is enforced, which is why the *chained* half of each
    /// saturated-parameter test still asserts [`finite_array`].
    pub fn no_nan_array(a: &Array1<f64>) -> bool {
        a.iter().all(|v| !v.is_nan())
    }

    /// Either a link the helper boxed itself or one the caller supplied.
    enum LinkSlot<'a> {
        Owned(Box<dyn Link>),
        Borrowed(&'a dyn Link),
    }

    impl LinkSlot<'_> {
        fn get(&self) -> &dyn Link {
            match self {
                LinkSlot::Owned(b) => b.as_ref(),
                LinkSlot::Borrowed(r) => *r,
            }
        }
    }

    /// Holds each parameter's link and the η implied by a `params` snapshot, so that
    /// a [`LinkContext`] can borrow from it.
    ///
    /// [`Distribution::eta_derivatives`] takes a context, and a context borrows the
    /// links and η it reports on; a free function cannot return one without also
    /// returning what it borrows from. Hence the two-step
    /// `let links = ParamLinks::defaults(..); let ctx = links.context();`.
    ///
    /// η is reconstructed as `g(μ)` from the fixture's μ. That is exact for every
    /// family except [`Ocat`], whose `params["mu"]` already holds η. Harmless there,
    /// because `Ocat` reparameterizes its thresholds and ignores the context.
    pub struct ParamLinks<'a> {
        names: Vec<&'static str>,
        links: Vec<LinkSlot<'a>>,
        etas: Vec<Array1<f64>>,
    }

    impl<'a> ParamLinks<'a> {
        /// Every parameter on its family default link.
        pub fn defaults<D: Distribution + ?Sized>(
            family: &D,
            params: &HashMap<&str, &Array1<f64>>,
        ) -> Self {
            Self::with_override(family, params, "", None)
        }

        /// Every parameter on its family default link, except `target`, which uses
        /// `link`. This is the post-refactor contract: the score must be `∂l/∂η` for
        /// whichever link the caller selected, not only the family's default.
        pub fn overriding<D: Distribution + ?Sized>(
            family: &D,
            params: &HashMap<&str, &Array1<f64>>,
            target: &str,
            link: &'a dyn Link,
        ) -> Self {
            Self::with_override(family, params, target, Some(link))
        }

        fn with_override<D: Distribution + ?Sized>(
            family: &D,
            params: &HashMap<&str, &Array1<f64>>,
            target: &str,
            override_link: Option<&'a dyn Link>,
        ) -> Self {
            let mut names = Vec::new();
            let mut links = Vec::new();
            let mut etas = Vec::new();
            for &name in family.parameters() {
                let slot = match override_link {
                    Some(l) if name == target => LinkSlot::Borrowed(l),
                    _ => LinkSlot::Owned(
                        family
                            .default_link(name)
                            .unwrap_or_else(|e| panic!("{}::{}: {}", family.name(), name, e)),
                    ),
                };
                let mu = params.get(name).unwrap_or_else(|| {
                    panic!("{}: fixture has no '{}' entry", family.name(), name)
                });
                etas.push(mu.mapv(|m| slot.get().link(m)));
                links.push(slot);
                names.push(name);
            }
            Self { names, links, etas }
        }

        pub fn context(&self) -> LinkContext<'_> {
            LinkContext::new(
                self.names
                    .iter()
                    .zip(&self.links)
                    .zip(&self.etas)
                    .map(|((&name, slot), eta)| (name, slot.get(), eta)),
            )
        }
    }

    /// Evaluate `eta_derivatives` under every parameter's default link.
    pub fn default_link_derivatives<D: Distribution + ?Sized>(
        family: &D,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        let links = ParamLinks::defaults(family, params);
        family.eta_derivatives(y, params, &links.context())
    }

    /// Keys, lengths, finiteness, and `w ≥ 0`, on the family's default links.
    ///
    /// The non-negativity assertion is what makes this the *expected*-information
    /// variant. Use [`derivative_keys_match_parameters_observed_info`] for a family
    /// that returns observed information, which may legitimately be negative.
    pub fn derivative_keys_match_parameters<D: Distribution>(
        d: &D,
        params: HashMap<&str, &Array1<f64>>,
        y: &Array1<f64>,
    ) {
        keys_and_arrays_are_well_formed(d, params, y, true)
    }

    /// As [`derivative_keys_match_parameters`], but without the `w ≥ 0` assertion.
    ///
    /// For the structural wrappers ([`Censored`] / [`Truncated`] / [`Hurdle`]),
    /// whose censoring / normalizer rows carry *observed* information
    /// `−∂²l/∂η²` rather than expected Fisher information. Observed information is
    /// not a variance and is genuinely allowed to be negative: a left-censored
    /// Gaussian row at z ≈ 0.125 gives `w_σ = −d2/F + (d1/F)² ≈ −0.08`, because the
    /// curvature term dominates the squared-score term there.
    ///
    /// This does not reach the solver. `scoring::step` floors with `w < MIN_WEIGHT`,
    /// which catches negatives as well as small positives, and it is the only floor
    /// in the pipeline (the floor-once rule of Altitude #1). Until Phase 3 the
    /// wrappers pre-floored these rows themselves, which produced the same number
    /// but hid them from `weight_floor_hits`; that is the Altitude #4 half of the
    /// same work.
    pub fn derivative_keys_match_parameters_observed_info<D: Distribution>(
        d: &D,
        params: HashMap<&str, &Array1<f64>>,
        y: &Array1<f64>,
    ) {
        keys_and_arrays_are_well_formed(d, params, y, false)
    }

    fn keys_and_arrays_are_well_formed<D: Distribution>(
        d: &D,
        params: HashMap<&str, &Array1<f64>>,
        y: &Array1<f64>,
        require_non_negative_weights: bool,
    ) {
        let derivs = default_link_derivatives(d, y, &params).unwrap();
        let mut keys: Vec<&str> = derivs.keys().map(String::as_str).collect();
        keys.sort();
        let mut expected: Vec<&str> = d.parameters().to_vec();
        expected.sort();
        assert_eq!(keys, expected);
        for (name, (u, w)) in &derivs {
            assert_eq!(u.len(), y.len());
            assert_eq!(w.len(), y.len());
            assert!(finite_array(u));
            assert!(finite_array(w));
            if require_non_negative_weights {
                assert!(
                    w.iter().all(|&v| v >= 0.0),
                    "{}::{}: expected information must be non-negative, got {:?}",
                    d.name(),
                    name,
                    w.to_vec()
                );
            }
        }
    }

    /// Build a `params` view from owned arrays for test ergonomics.
    pub fn params_view<'a>(
        owned: &'a [(&'static str, Array1<f64>)],
    ) -> HashMap<&'a str, &'a Array1<f64>> {
        owned.iter().map(|(k, v)| (*k, v)).collect()
    }

    /// Check that the analytic score `u` returned by `theta_derivatives()` matches the
    /// central difference of `loglik_pointwise` for `target` on the η-scale,
    /// under the family's *default* link.
    ///
    /// Thin wrapper over [`check_eta_score_via_finite_diff`], which takes the
    /// link explicitly.
    pub fn check_score_via_finite_diff<D: Distribution + ?Sized>(
        family: &D,
        y: &Array1<f64>,
        owned: &[(&'static str, Array1<f64>)],
        target: &str,
        tol: f64,
    ) {
        let link = family.default_link(target).unwrap();
        check_eta_score_via_finite_diff(family, y, owned, target, link.as_ref(), tol);
    }

    /// Check the analytic η-scale score `u` for `target` against a central
    /// difference of `loglik_pointwise` taken on the η of an **explicitly
    /// supplied** link.
    ///
    /// This is the contract the fitting loop actually needs: `u` must be
    /// `∂l/∂η` for *whichever* link the caller selected via
    /// [`FitConfig::with_link`](crate::FitConfig::with_link), not only the
    /// family's default. Taking the link as a parameter does two things the
    /// default-link-only version cannot:
    ///
    /// 1. It gives the seven identity-link parameters (Gaussian μ, StudentT μ,
    ///    BCCG/BCT/BCPE ν, Ocat μ and `delta_1`) a non-trivial check at all:
    ///    under the identity link `∂l/∂η ≡ ∂l/∂θ`, so the default-link check is
    ///    vacuously satisfied by any correct natural-scale score.
    /// 2. It is the post-refactor contract for the generic chain rule
    ///    (Altitude #1), so call sites written against it stay valid across the
    ///    change.
    ///
    /// Note this asserts on the *score* only. The Fisher weight has no generic
    /// finite-difference oracle (several families deliberately return expected
    /// information or a squared-score surrogate rather than `−∂²l/∂η²`), and is
    /// covered by the golden characterization tables in
    /// `tests/derivative_golden.rs` instead.
    pub fn check_eta_score_via_finite_diff<D: Distribution + ?Sized>(
        family: &D,
        y: &Array1<f64>,
        owned: &[(&'static str, Array1<f64>)],
        target: &str,
        link: &dyn Link,
        tol: f64,
    ) {
        let p: HashMap<&str, &Array1<f64>> = owned.iter().map(|(k, v)| (*k, v)).collect();
        let links = ParamLinks::overriding(family, &p, target, link);
        let derivs = family.eta_derivatives(y, &p, &links.context()).unwrap();
        let analytic_u = derivs.get(target).unwrap().0.clone();

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

    /// Check the analytic `(∂F/∂θ, ∂²F/∂θ²)` returned by
    /// [`Distribution::cdf_theta_derivatives`] for `target` against central
    /// differences of [`Distribution::cdf`] on the parameter's **own natural
    /// scale**.
    ///
    /// The step is *relative*: `h = eps·max(|θ|, 1)`. For a positive parameter
    /// that is numerically the same perturbation the previous η-scale version took
    /// through a log link (`σ·e^{±eps} ≈ σ(1 ± eps)`), so the callers' tolerances
    /// carry over; for an identity-linked parameter it is unchanged. An *absolute*
    /// step would put the minus side at or below zero for any θ ≲ eps.
    ///
    /// **Fails** if the family does not supply `target` analytically. A silent
    /// skip was the previous behavior and it was a coverage hole: a family that
    /// stopped emitting an analytic entry would quietly degrade to the
    /// structural wrappers' numeric fallback with every test still green.
    pub fn check_cdf_theta_derivatives_via_finite_diff<D: Distribution + ?Sized>(
        family: &D,
        y: &Array1<f64>,
        owned: &[(&'static str, Array1<f64>)],
        target: &str,
        tol: f64,
    ) {
        let p: HashMap<&str, &Array1<f64>> = owned.iter().map(|(k, v)| (*k, v)).collect();
        let derivs = family.cdf_theta_derivatives(y, &p).unwrap();
        let Some((analytic_d1, analytic_d2)) = derivs.get(target) else {
            panic!(
                "{}::{} supplies no analytic cdf_theta_derivatives entry, so this \
                 check would be vacuous. The structural wrappers fall back to a \
                 central difference for such parameters; cover it there, or \
                 stop asserting on it here.",
                family.name(),
                target
            )
        };

        let eps: f64 = 1e-5;
        let idx = owned.iter().position(|(k, _)| *k == target).unwrap();
        let mut perturbed: Vec<(&'static str, Array1<f64>)> =
            owned.iter().map(|(k, v)| (*k, v.clone())).collect();

        for i in 0..y.len() {
            let theta = owned[idx].1[i];
            let h = eps * theta.abs().max(1.0);

            let f0 = {
                let pv: HashMap<&str, &Array1<f64>> =
                    perturbed.iter().map(|(k, v)| (*k, v)).collect();
                family.cdf(y, &pv).unwrap()[i]
            };
            perturbed[idx].1[i] = theta + h;
            let f_plus = {
                let pv: HashMap<&str, &Array1<f64>> =
                    perturbed.iter().map(|(k, v)| (*k, v)).collect();
                family.cdf(y, &pv).unwrap()[i]
            };
            perturbed[idx].1[i] = theta - h;
            let f_minus = {
                let pv: HashMap<&str, &Array1<f64>> =
                    perturbed.iter().map(|(k, v)| (*k, v)).collect();
                family.cdf(y, &pv).unwrap()[i]
            };
            perturbed[idx].1[i] = theta;

            let numeric_d1 = (f_plus - f_minus) / (2.0 * h);
            let numeric_d2 = (f_plus - 2.0 * f0 + f_minus) / (h * h);

            let s1 = analytic_d1[i].abs().max(1.0);
            assert!(
                (analytic_d1[i] - numeric_d1).abs() / s1 < tol,
                "{}::{} obs {}: ∂F/∂θ analytic={:.6e} numeric={:.6e}",
                family.name(),
                target,
                i,
                analytic_d1[i],
                numeric_d1
            );
            let s2 = analytic_d2[i].abs().max(1.0);
            assert!(
                (analytic_d2[i] - numeric_d2).abs() / s2 < tol,
                "{}::{} obs {}: ∂²F/∂θ² analytic={:.6e} numeric={:.6e}",
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
    use ndarray::array;

    // Link-function tests live alongside the link impls in `links.rs`.

    // --- chain_to_eta ---

    #[test]
    fn chain_to_eta_applies_first_and_squared_link_derivatives() {
        // σ under a log link: mu_eta = σ, so u_η = σ·∂l/∂σ and w_η = σ²·i_σ.
        // Values chosen so the expected result is exact in binary floating point.
        let eta = array![0.0, f64::ln(2.0)]; // σ = 1, 2
        let log = LogLink;
        let ctx = LinkContext::new([("sigma", &log as &dyn Link, &eta)]);

        let natural = HashMap::from([(
            "sigma".to_string(),
            (array![3.0, 5.0], array![4.0, 0.5]), // ∂l/∂σ, i_σ
        )]);
        let out = chain_to_eta(natural, &ctx).unwrap();
        let (u, w) = &out["sigma"];

        assert!((u[0] - 3.0).abs() < 1e-12); // 1·3
        assert!((u[1] - 10.0).abs() < 1e-12); // 2·5
        assert!((w[0] - 4.0).abs() < 1e-12); // 1²·4
        assert!((w[1] - 2.0).abs() < 1e-12); // 2²·0.5
    }

    #[test]
    fn chain_to_eta_leaves_weights_unfloored() {
        // The floor-once rule: a weight far below MIN_WEIGHT must survive this
        // function untouched, because flooring before the multiply is what turns a
        // drifting Student-t ν block into a frozen one.
        let eta = array![0.0];
        let id = IdentityLink;
        let ctx = LinkContext::new([("mu", &id as &dyn Link, &eta)]);

        let natural = HashMap::from([("mu".to_string(), (array![1.0], array![1e-18]))]);
        let out = chain_to_eta(natural, &ctx).unwrap();
        assert_eq!(out["mu"].1[0], 1e-18);
        assert!(out["mu"].1[0] < MIN_WEIGHT);
    }

    #[test]
    fn chain_to_eta_rejects_a_parameter_the_context_lacks() {
        let eta = array![0.0];
        let id = IdentityLink;
        let ctx = LinkContext::new([("mu", &id as &dyn Link, &eta)]);

        let natural = HashMap::from([("sigma".to_string(), (array![1.0], array![1.0]))]);
        let err = chain_to_eta(natural, &ctx).unwrap_err();
        assert!(matches!(err, GamlssError::Internal(_)), "{err:?}");
    }

    #[test]
    fn chain_to_eta_rejects_a_length_mismatch() {
        let eta = array![0.0, 1.0];
        let id = IdentityLink;
        let ctx = LinkContext::new([("mu", &id as &dyn Link, &eta)]);

        let natural = HashMap::from([("mu".to_string(), (array![1.0], array![1.0]))]);
        let err = chain_to_eta(natural, &ctx).unwrap_err();
        assert!(matches!(err, GamlssError::Internal(_)), "{err:?}");
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
            "Weibull",
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
            from_name("Weibull").unwrap(),
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

    // --- Altitude #1: the non-default-link contract ---
    //
    // The derivative-level gate for this lives with each family, not here: every
    // family file has a `score_matches_finite_diff_under_non_default_links` test
    // asserting its `eta_derivatives` agrees with a finite difference on an
    // *overridden* link's η. Three characterization tests used to sit at this spot
    // asserting the opposite (that the score DISAGREED) to pin the pre-refactor
    // behavior; the generic chain rule landed, so they were replaced by the
    // per-family gates rather than inverted in place.
    //
    // The end-to-end counterparts are `tests/link_selection.rs` and the independent
    // MLE oracle in `tests/link_mle_oracle.rs`.
}
