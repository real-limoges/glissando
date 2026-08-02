//! Link functions mapping a response-scale parameter `μ` to the linear predictor
//! `η = g(μ)`. Used by the IRLS scoring step (via `inv_link` to materialise `μ`
//! from `η`) and by the design-matrix code (via `link` to seed initial values).

use crate::error::GamlssError;
use crate::math::{std_normal_cdf, std_normal_pdf, std_normal_quantile};
use ndarray::Array1;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::fmt::Debug;

/// Floor for positive parameters (μ, σ, …) to avoid `log(0)` or division by zero.
pub(crate) const MIN_POSITIVE: f64 = 1e-10;
/// Linear-predictor ceiling for log/logit links (prevents `exp` overflow).
pub(crate) const MAX_ETA: f64 = 30.0;
/// Linear-predictor floor for log/logit links (prevents `exp` underflow).
pub(crate) const MIN_ETA: f64 = -30.0;

/// A link function `g` mapping the response-scale parameter `μ` to the linear predictor `η = g(μ)`.
pub trait Link: Debug + Send + Sync {
    /// Apply the link: `η = g(μ)`.
    fn link(&self, mu: f64) -> f64;
    /// Apply the inverse link: `μ = g⁻¹(η)`.
    fn inv_link(&self, eta: f64) -> f64;

    /// Derivative of the inverse link wrt the linear predictor: `dμ/dη = (g⁻¹)'(η)`.
    ///
    /// The Fisher-scoring inner loop works on the η-scale, where the generic IRLS
    /// weight is `mu_eta(η)² · i_θ` (with `i_θ` the Fisher information on the
    /// natural scale). Supplying this analytically lets the scoring step build
    /// η-scale weights for *any* link: the prerequisite for the link expansion
    /// (probit, cloglog, …).
    ///
    /// The default is a symmetric finite difference so external `Link` impls keep
    /// compiling; the built-in links override it with their closed forms.
    fn mu_eta(&self, eta: f64) -> f64 {
        let h = 1e-6;
        (self.inv_link(eta + h) - self.inv_link(eta - h)) / (2.0 * h)
    }

    /// Second derivative of the inverse link wrt the linear predictor:
    /// `d²μ/dη² = (g⁻¹)''(η)`.
    ///
    /// Needed by the structural wrappers (`Censored`/`Truncated`/`Hurdle`), whose
    /// IRLS weights are *observed* information rather than expected information.
    /// Observed information is not link-invariant: `d²/dη²[−log D]` carries a
    /// `mu_eta2 · ∂l/∂θ` term with no `mu_eta²` factor, so a wrapper's η-weight is
    /// provably not of the form `mu_eta² × (anything on the natural scale)`.
    ///
    /// **Clamp convention.** Every built-in override evaluates the closed-form
    /// second derivative at the *same clamped* `η` that its [`Link::mu_eta`] uses,
    /// so the two form a consistent pair (both are derivatives of the same
    /// unclamped inverse link, sampled at the same point) and both stay finite.
    /// They are deliberately *not* the derivatives of the clamped `inv_link`
    /// itself, which would be identically zero in a saturated region and would
    /// freeze a saturated row instead of letting it walk back out.
    /// [`FlooredLogLink`] is the one exception: its hard zero below the floor is a
    /// modeling constraint that the Student-t ν-floor KKT projection depends on.
    ///
    /// The default is a symmetric finite difference of `mu_eta` so external `Link`
    /// impls keep compiling. It differences `mu_eta` rather than `inv_link` twice:
    /// a second difference of `inv_link` loses roughly half the available digits.
    fn mu_eta2(&self, eta: f64) -> f64 {
        let h = 1e-6;
        (self.mu_eta(eta + h) - self.mu_eta(eta - h)) / (2.0 * h)
    }

    /// Canonical lowercase name of this link, used to persist it in a fitted model
    /// and to select it by string (see [`link_from_name`]).
    ///
    /// Defaults to `"custom"` so external `Link` impls keep compiling; every
    /// built-in link overrides it with a name `link_from_name` round-trips.
    fn name(&self) -> &'static str {
        "custom"
    }
}

/// Identity link: `η = μ`. Used for unbounded continuous parameters (e.g. Gaussian mean).
#[derive(Debug, Clone, Copy, Default)]
pub struct IdentityLink;

impl Link for IdentityLink {
    fn link(&self, mu: f64) -> f64 {
        mu
    }
    fn inv_link(&self, eta: f64) -> f64 {
        eta
    }
    fn mu_eta(&self, _eta: f64) -> f64 {
        // μ = η ⇒ dμ/dη = 1.
        1.0
    }
    fn mu_eta2(&self, _eta: f64) -> f64 {
        // dμ/dη is constant ⇒ d²μ/dη² = 0.
        0.0
    }
    fn name(&self) -> &'static str {
        "identity"
    }
}

/// Log link: `η = log(μ)`. Used for positive parameters (Poisson rate, Gamma mean).
///
/// `link` clamps `log(μ)` from below at `MIN_ETA = -30.0`; `inv_link` clamps `η`
/// from above at `MAX_ETA = 30.0` to keep `exp(η)` finite.
#[derive(Debug, Clone, Copy, Default)]
pub struct LogLink;

impl Link for LogLink {
    fn link(&self, mu: f64) -> f64 {
        // Clamp both sides so link/inv_link round-trip consistently: inv_link
        // caps η at MAX_ETA, so an uncapped ln(μ) for extreme μ would break
        // g(g⁻¹(η)) = η at the top end.
        mu.ln().clamp(MIN_ETA, MAX_ETA)
    }
    fn inv_link(&self, eta: f64) -> f64 {
        eta.min(MAX_ETA).exp()
    }
    fn mu_eta(&self, eta: f64) -> f64 {
        // μ = e^η ⇒ dμ/dη = μ. Clamp η to match `inv_link`.
        eta.min(MAX_ETA).exp()
    }
    fn mu_eta2(&self, eta: f64) -> f64 {
        // μ = e^η ⇒ d²μ/dη² = μ, clamped at the same η as `mu_eta`.
        eta.min(MAX_ETA).exp()
    }
    fn name(&self) -> &'static str {
        "log"
    }
}

/// Log link with a lower bound applied after inversion: `μ = max(exp(η), floor)`.
///
/// Used for the Student-t degrees-of-freedom `ν`, floored at 2 so the variance
/// `σ²·ν/(ν−2)` (and the mean) stay finite as the optimizer explores the heavy-tail
/// region. The floor keeps `ν` out of the degenerate `ν ≤ 2` zone during iteration;
/// on data whose true `ν` is well above 2 it never binds, so parity with an unfloored
/// fit is preserved.
#[derive(Debug, Clone, Copy)]
pub struct FlooredLogLink {
    pub floor: f64,
}

impl Link for FlooredLogLink {
    fn link(&self, mu: f64) -> f64 {
        mu.max(self.floor).ln().clamp(MIN_ETA, MAX_ETA)
    }
    fn inv_link(&self, eta: f64) -> f64 {
        eta.min(MAX_ETA).exp().max(self.floor)
    }
    fn mu_eta(&self, eta: f64) -> f64 {
        // μ = max(e^η, floor): the derivative is e^η in the active region and 0
        // where the floor binds (μ is constant there).
        let mu = eta.min(MAX_ETA).exp();
        if mu <= self.floor {
            0.0
        } else {
            mu
        }
    }
    fn mu_eta2(&self, eta: f64) -> f64 {
        // Same shape as `mu_eta`: e^η where the floor is slack, 0 where it binds.
        // The hard zero is deliberate here (unlike the other links), because the
        // ν-floor KKT projection in `student_t.rs` keys off a vanishing derivative
        // to detect a pinned row.
        let mu = eta.min(MAX_ETA).exp();
        if mu <= self.floor {
            0.0
        } else {
            mu
        }
    }
    fn name(&self) -> &'static str {
        "floored_log"
    }
}

/// Logit link: `η = log(μ / (1 - μ))`. Used for `(0, 1)` probability parameters.
#[derive(Debug, Clone, Copy, Default)]
pub struct LogitLink;

impl Link for LogitLink {
    fn link(&self, mu: f64) -> f64 {
        let m = mu.clamp(MIN_POSITIVE, 1.0 - MIN_POSITIVE);
        (m / (1.0 - m)).ln()
    }
    fn inv_link(&self, eta: f64) -> f64 {
        let e = eta.clamp(MIN_ETA, MAX_ETA);
        1.0 / (1.0 + (-e).exp())
    }
    fn mu_eta(&self, eta: f64) -> f64 {
        // μ = σ(η) ⇒ dμ/dη = μ(1−μ).
        let mu = self.inv_link(eta);
        mu * (1.0 - mu)
    }
    fn mu_eta2(&self, eta: f64) -> f64 {
        // d/dη[μ(1−μ)] = (1−2μ)·μ(1−μ). Vanishes at μ=½ (the inflection of σ).
        let mu = self.inv_link(eta);
        mu * (1.0 - mu) * (1.0 - 2.0 * mu)
    }
    fn name(&self) -> &'static str {
        "logit"
    }
}

/// Probit link: `η = Φ⁻¹(μ)`. A bounded `(0, 1)` link (the latent-normal alternative
/// to logit), common in econometrics and bioassay. `mu_eta` is the standard-normal
/// PDF `φ(η)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ProbitLink;

impl Link for ProbitLink {
    fn link(&self, mu: f64) -> f64 {
        // Φ⁻¹ on a clamped probability; std_normal_quantile clamps internally too.
        std_normal_quantile(mu.clamp(MIN_POSITIVE, 1.0 - MIN_POSITIVE))
    }
    fn inv_link(&self, eta: f64) -> f64 {
        std_normal_cdf(eta.clamp(MIN_ETA, MAX_ETA))
    }
    fn mu_eta(&self, eta: f64) -> f64 {
        // μ = Φ(η) ⇒ dμ/dη = φ(η).
        std_normal_pdf(eta.clamp(MIN_ETA, MAX_ETA))
    }
    fn mu_eta2(&self, eta: f64) -> f64 {
        // φ'(η) = −η·φ(η), at the same clamped η as `mu_eta`.
        let e = eta.clamp(MIN_ETA, MAX_ETA);
        -e * std_normal_pdf(e)
    }
    fn name(&self) -> &'static str {
        "probit"
    }
}

/// Complementary log-log link: `η = log(−log(1 − μ))`. A bounded `(0, 1)` link, the
/// standard choice for discrete-time survival / complementary-risk and asymmetric
/// rare-event data.
#[derive(Debug, Clone, Copy, Default)]
pub struct CloglogLink;

impl Link for CloglogLink {
    fn link(&self, mu: f64) -> f64 {
        let m = mu.clamp(MIN_POSITIVE, 1.0 - MIN_POSITIVE);
        (-(1.0 - m).ln()).ln().clamp(MIN_ETA, MAX_ETA)
    }
    fn inv_link(&self, eta: f64) -> f64 {
        // μ = 1 − exp(−exp(η)).
        let e = eta.clamp(MIN_ETA, MAX_ETA).exp();
        1.0 - (-e).exp()
    }
    fn mu_eta(&self, eta: f64) -> f64 {
        // dμ/dη = exp(η − exp(η)).
        let ec = eta.clamp(MIN_ETA, MAX_ETA);
        (ec - ec.exp()).exp()
    }
    fn mu_eta2(&self, eta: f64) -> f64 {
        // d²μ/dη² = exp(η − exp(η))·(1 − exp(η)); sign flips at η = 0.
        let ec = eta.clamp(MIN_ETA, MAX_ETA);
        let ee = ec.exp();
        (ec - ee).exp() * (1.0 - ee)
    }
    fn name(&self) -> &'static str {
        "cloglog"
    }
}

/// Inverse link: `η = 1/μ`. The canonical link for the Gamma family; `μ` and `η`
/// share a sign and are kept away from zero.
#[derive(Debug, Clone, Copy, Default)]
pub struct InverseLink;

impl Link for InverseLink {
    fn link(&self, mu: f64) -> f64 {
        1.0 / signed_floor(mu)
    }
    fn inv_link(&self, eta: f64) -> f64 {
        1.0 / signed_floor(eta)
    }
    fn mu_eta(&self, eta: f64) -> f64 {
        // μ = 1/η ⇒ dμ/dη = −1/η².
        let e = signed_floor(eta);
        -1.0 / (e * e)
    }
    fn mu_eta2(&self, eta: f64) -> f64 {
        // μ = 1/η ⇒ d²μ/dη² = 2/η³, sign-following like `mu_eta`.
        let e = signed_floor(eta);
        2.0 / (e * e * e)
    }
    fn name(&self) -> &'static str {
        "inverse"
    }
}

/// Inverse-square link: `η = 1/μ²`. Used for positive responses (e.g. an
/// inverse-Gaussian mean); `η` stays positive.
#[derive(Debug, Clone, Copy, Default)]
pub struct InverseSquareLink;

impl Link for InverseSquareLink {
    fn link(&self, mu: f64) -> f64 {
        let m = mu.abs().max(MIN_POSITIVE);
        1.0 / (m * m)
    }
    fn inv_link(&self, eta: f64) -> f64 {
        // μ = η^(−1/2), η > 0.
        eta.max(MIN_POSITIVE).powf(-0.5)
    }
    fn mu_eta(&self, eta: f64) -> f64 {
        // dμ/dη = −½·η^(−3/2).
        -0.5 * eta.max(MIN_POSITIVE).powf(-1.5)
    }
    fn mu_eta2(&self, eta: f64) -> f64 {
        // d²μ/dη² = ¾·η^(−5/2), at the same floored η as `mu_eta`.
        0.75 * eta.max(MIN_POSITIVE).powf(-2.5)
    }
    fn name(&self) -> &'static str {
        "inverse_square"
    }
}

/// Square-root link: `η = √μ`. The Poisson variance-stabilizing link.
#[derive(Debug, Clone, Copy, Default)]
pub struct SqrtLink;

impl Link for SqrtLink {
    fn link(&self, mu: f64) -> f64 {
        mu.max(0.0).sqrt()
    }
    fn inv_link(&self, eta: f64) -> f64 {
        eta * eta
    }
    fn mu_eta(&self, eta: f64) -> f64 {
        // μ = η² ⇒ dμ/dη = 2η.
        2.0 * eta
    }
    fn mu_eta2(&self, _eta: f64) -> f64 {
        // μ = η² ⇒ d²μ/dη² = 2, everywhere.
        2.0
    }
    fn name(&self) -> &'static str {
        "sqrt"
    }
}

/// Cauchit link: `η = tan(π(μ − ½))`. A bounded `(0, 1)` link with heavier tails than
/// logit/probit, more robust to extreme points in the linear predictor.
#[derive(Debug, Clone, Copy, Default)]
pub struct CauchitLink;

impl Link for CauchitLink {
    fn link(&self, mu: f64) -> f64 {
        let m = mu.clamp(MIN_POSITIVE, 1.0 - MIN_POSITIVE);
        (PI * (m - 0.5)).tan().clamp(MIN_ETA, MAX_ETA)
    }
    fn inv_link(&self, eta: f64) -> f64 {
        // μ = ½ + atan(η)/π ∈ (0, 1) for all finite η.
        0.5 + eta.atan() / PI
    }
    fn mu_eta(&self, eta: f64) -> f64 {
        // dμ/dη = 1/(π(1 + η²)).
        1.0 / (PI * (1.0 + eta * eta))
    }
    fn mu_eta2(&self, eta: f64) -> f64 {
        // d²μ/dη² = −2η/(π(1 + η²)²).
        let d = 1.0 + eta * eta;
        -2.0 * eta / (PI * d * d)
    }
    fn name(&self) -> &'static str {
        "cauchit"
    }
}

// ============================================================================
// LinkContext
// ============================================================================

/// Per-parameter link derivatives evaluated at the current linear predictor.
///
/// Built once per Fisher-scoring step from the model's live `η` and resolved links,
/// and handed to [`Distribution::eta_derivatives`](crate::distributions::Distribution)
/// so a family can map its natural-scale score and information onto the η scale:
/// `u_η = mu_eta · ∂l/∂θ`, `w_η = mu_eta² · i_θ`.
///
/// Both derivative arrays are materialized eagerly, for every parameter, at
/// construction. Laziness would need interior mutability, which makes the type
/// `!Sync` and therefore unusable inside the families' rayon closures under the
/// `parallel` feature. The cost is two `O(n)` passes per parameter against an
/// `O(np²)` solve.
///
/// Raw `η` is deliberately absent from the public surface.
/// [`Ocat`](crate::distributions::Ocat) already carries η in its `params["mu"]`
/// slot and that pattern should not spread;
/// the crate-internal `LinkContext::link_and_eta` exists only for the structural
/// wrappers' numeric CDF fallback, which must perturb the actual linear predictor.
#[derive(Debug)]
pub struct LinkContext<'a> {
    entries: HashMap<&'a str, LinkEntry<'a>>,
}

#[derive(Debug)]
struct LinkEntry<'a> {
    // Read only through `link_and_eta`, whose one production caller is the
    // structural wrappers' numeric CDF fallback (`structural::numeric_cdf_grad`).
    link: &'a dyn Link,
    eta: &'a Array1<f64>,
    mu_eta: Array1<f64>,
    mu_eta2: Array1<f64>,
}

impl<'a> LinkContext<'a> {
    /// Build a context from `(parameter name, link, η)` triples.
    ///
    /// Takes an iterator rather than the fitting layer's parameter map so that this
    /// module stays below `fitting/` in the dependency order.
    pub fn new<I>(entries: I) -> Self
    where
        I: IntoIterator<Item = (&'a str, &'a dyn Link, &'a Array1<f64>)>,
    {
        let entries = entries
            .into_iter()
            .map(|(name, link, eta)| {
                let entry = LinkEntry {
                    link,
                    eta,
                    mu_eta: eta.mapv(|e| link.mu_eta(e)),
                    mu_eta2: eta.mapv(|e| link.mu_eta2(e)),
                };
                (name, entry)
            })
            .collect();
        Self { entries }
    }

    /// `dμ/dη` for `param`, evaluated at the current η.
    ///
    /// # Errors
    ///
    /// Returns [`GamlssError::Internal`] if the context holds no entry for `param`.
    pub fn mu_eta(&self, param: &str) -> Result<&Array1<f64>, GamlssError> {
        Ok(&self.entry(param)?.mu_eta)
    }

    /// `d²μ/dη²` for `param`, evaluated at the current η.
    ///
    /// # Errors
    ///
    /// Returns [`GamlssError::Internal`] if the context holds no entry for `param`.
    pub fn mu_eta2(&self, param: &str) -> Result<&Array1<f64>, GamlssError> {
        Ok(&self.entry(param)?.mu_eta2)
    }

    /// The link and linear predictor backing `param`.
    ///
    /// Crate-internal on purpose: the only legitimate consumer is the structural
    /// wrappers' numeric CDF fallback (`structural::numeric_cdf_grad`), which
    /// finite-differences `base.cdf` on η itself rather than on θ, and so needs the
    /// actual linear predictor instead of `link.link(θ)`, a round trip that is not
    /// the identity under `sqrt` for η < 0, or `inverse_square` at all.
    pub(crate) fn link_and_eta(&self, param: &str) -> Option<(&dyn Link, &Array1<f64>)> {
        self.entries.get(param).map(|e| (e.link, e.eta))
    }

    fn entry(&self, param: &str) -> Result<&LinkEntry<'a>, GamlssError> {
        self.entries.get(param).ok_or_else(|| {
            GamlssError::Internal(format!(
                "LinkContext has no entry for parameter '{}'",
                param
            ))
        })
    }
}

/// Clamp a value away from zero while preserving its sign, so `1/x` stays finite for
/// the inverse link near the origin.
#[inline]
fn signed_floor(x: f64) -> f64 {
    if x < 0.0 {
        x.min(-MIN_POSITIVE)
    } else {
        x.max(MIN_POSITIVE)
    }
}

/// Construct a boxed [`Link`] from its canonical name (see [`Link::name`]).
///
/// This is the string→`Link` registry that backs per-parameter link selection in
/// [`FitConfig`](crate::FitConfig) and the JSON/WASM/Python surfaces. The internal
/// `floored_log` link is intentionally excluded; it is never user-selectable.
///
/// # Errors
///
/// Returns [`GamlssError::Input`] for an unknown name.
pub fn link_from_name(name: &str) -> Result<Box<dyn Link>, GamlssError> {
    match name {
        "identity" => Ok(Box::new(IdentityLink)),
        "log" => Ok(Box::new(LogLink)),
        "logit" => Ok(Box::new(LogitLink)),
        "probit" => Ok(Box::new(ProbitLink)),
        "cloglog" => Ok(Box::new(CloglogLink)),
        "inverse" => Ok(Box::new(InverseLink)),
        "inverse_square" => Ok(Box::new(InverseSquareLink)),
        "sqrt" => Ok(Box::new(SqrtLink)),
        "cauchit" => Ok(Box::new(CauchitLink)),
        other => Err(GamlssError::Input(format!(
            "Unknown link: '{}'. Supported: identity, log, logit, probit, cloglog, \
             inverse, inverse_square, sqrt, cauchit",
            other
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(target_arch = "wasm32"))]
    use proptest::prelude::*;

    #[test]
    fn identity_link_is_noop() {
        let l = IdentityLink;
        assert_eq!(l.link(0.0), 0.0);
        assert_eq!(l.link(3.5), 3.5);
        assert_eq!(l.inv_link(-2.0), -2.0);
    }

    #[test]
    fn log_link_clamps_underflow_and_overflow() {
        let l = LogLink;
        assert!(l.link(0.0).is_finite());
        assert!(l.link(0.0) <= MIN_ETA + 1e-9);
        assert!(l.inv_link(1e6).is_finite());
        assert!(l.inv_link(1e6) <= MAX_ETA.exp() + 1.0);
    }

    #[test]
    fn logit_link_handles_boundary_probabilities() {
        let l = LogitLink;
        assert!(l.link(0.0).is_finite());
        assert!(l.link(1.0).is_finite());
        assert!((0.0..=1.0).contains(&l.inv_link(-1e6)));
        assert!((0.0..=1.0).contains(&l.inv_link(1e6)));
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn close(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps * (1.0 + a.abs().max(b.abs()))
    }

    /// Symmetric finite-difference of the inverse link: the trait's default impl,
    /// recomputed here as an independent oracle for the analytic overrides.
    fn fd_mu_eta(link: &dyn Link, eta: f64) -> f64 {
        let h = 1e-6;
        (link.inv_link(eta + h) - link.inv_link(eta - h)) / (2.0 * h)
    }

    /// Symmetric finite-difference of `mu_eta`: the oracle for the `mu_eta2`
    /// overrides. A larger step than `fd_mu_eta` uses: the analytic `mu_eta` this
    /// differences is exact, so `h` trades only truncation against cancellation and
    /// `1e-4` sits near the optimum for a first difference of a smooth function.
    fn fd_mu_eta2(link: &dyn Link, eta: f64) -> f64 {
        let h = 1e-4;
        (link.mu_eta(eta + h) - link.mu_eta(eta - h)) / (2.0 * h)
    }

    #[test]
    fn mu_eta_identity_is_one_everywhere() {
        let l = IdentityLink;
        for &eta in &[-10.0, -1.0, 0.0, 2.5, 100.0] {
            assert_eq!(l.mu_eta(eta), 1.0);
        }
    }

    #[test]
    fn mu_eta_logit_peaks_at_zero() {
        let l = LogitLink;
        // dμ/dη = μ(1−μ) is maximised at η=0 where μ=0.5 ⇒ 0.25.
        assert!((l.mu_eta(0.0) - 0.25).abs() < 1e-12);
        // Symmetric and strictly smaller away from 0.
        assert!(l.mu_eta(1.0) < 0.25);
        assert!((l.mu_eta(1.0) - l.mu_eta(-1.0)).abs() < 1e-12);
    }

    #[test]
    fn mu_eta_analytic_matches_finite_difference() {
        // Each analytic override agrees with the finite-difference default across a
        // grid of η (inside the clamp regions to avoid the saturated endpoints).
        //
        // Saturated regions (|η| ≥ MAX_ETA, and the FlooredLogLink floor) are
        // *deliberately* excluded rather than accidentally uncovered: there the
        // clamped `inv_link` is constant, so a finite difference of it is 0, while
        // the overrides return the unclamped closed form sampled at the clamped η.
        // That divergence is the documented convention (see `Link::mu_eta2`), not a
        // defect, so a finite-difference oracle cannot check it.
        //
        // Links whose `inv_link` is smooth across the full η grid (incl. 0 and
        // negatives). The inverse / inverse-square links have a positive-η domain
        // and are checked separately in `mu_eta_inverse_links_match_finite_difference`.
        let links: [&dyn Link; 8] = [
            &IdentityLink,
            &LogLink,
            &FlooredLogLink { floor: 2.0 },
            &LogitLink,
            &ProbitLink,
            &CloglogLink,
            &CauchitLink,
            &SqrtLink,
        ];
        for link in links {
            for &eta in &[-5.0, -2.0, -0.5, 0.0, 0.5, 2.0, 5.0] {
                let analytic = link.mu_eta(eta);
                let numeric = fd_mu_eta(link, eta);
                assert!(
                    (analytic - numeric).abs() <= 1e-5 * (1.0 + analytic.abs()),
                    "{:?} at η={}: analytic {} vs fd {}",
                    link,
                    eta,
                    analytic,
                    numeric
                );
            }
        }
    }

    #[test]
    fn mu_eta2_analytic_matches_finite_difference() {
        // Mirrors `mu_eta_analytic_matches_finite_difference`, one derivative up,
        // over the same η grid and with the same deliberate exclusion of saturated
        // regions (see the note there).
        let links: [&dyn Link; 8] = [
            &IdentityLink,
            &LogLink,
            &FlooredLogLink { floor: 2.0 },
            &LogitLink,
            &ProbitLink,
            &CloglogLink,
            &CauchitLink,
            &SqrtLink,
        ];
        for link in links {
            // The FlooredLogLink kink sits at η = ln(2) ≈ 0.693; the grid straddles
            // it without landing on it, so every point has a one-sided-constant or
            // fully-active neighborhood wider than the difference step.
            for &eta in &[-5.0, -2.0, -0.5, 0.0, 0.5, 2.0, 5.0] {
                let analytic = link.mu_eta2(eta);
                let numeric = fd_mu_eta2(link, eta);
                assert!(
                    (analytic - numeric).abs() <= 1e-4 * (1.0 + analytic.abs()),
                    "{:?} at η={}: analytic {} vs fd {}",
                    link,
                    eta,
                    analytic,
                    numeric
                );
            }
        }
    }

    #[test]
    fn mu_eta2_inverse_links_match_finite_difference() {
        // Same domain split as `mu_eta_inverse_links_match_finite_difference`:
        // inverse is smooth away from 0 (both signs), inverse-square is positive-η
        // only. Both blow up near the origin, so the tolerance is looser.
        for &eta in &[-5.0, -2.0, -0.5, 0.5, 2.0, 5.0] {
            let l = InverseLink;
            let (a, n) = (l.mu_eta2(eta), fd_mu_eta2(&l, eta));
            assert!(
                (a - n).abs() <= 1e-3 * (1.0 + a.abs()),
                "inverse at η={eta}: analytic {a} vs fd {n}"
            );
        }
        for &eta in &[0.5, 1.0, 2.0, 5.0, 10.0] {
            let l = InverseSquareLink;
            let (a, n) = (l.mu_eta2(eta), fd_mu_eta2(&l, eta));
            assert!(
                (a - n).abs() <= 1e-3 * (1.0 + a.abs()),
                "inverse_square at η={eta}: analytic {a} vs fd {n}"
            );
        }
    }

    #[test]
    fn mu_eta2_closed_forms_at_known_points() {
        // Fixed points where the second derivative has an exact value, so the test
        // is independent of the finite-difference oracle above.
        assert_eq!(IdentityLink.mu_eta2(3.7), 0.0);
        assert_eq!(SqrtLink.mu_eta2(-2.0), 2.0);
        // log: d²μ/dη² = μ = e^η.
        assert!((LogLink.mu_eta2(1.5) - 1.5_f64.exp()).abs() < 1e-12);
        // logit: (1−2μ) vanishes at μ = ½, i.e. the inflection at η = 0.
        assert!(LogitLink.mu_eta2(0.0).abs() < 1e-15);
        // probit: −η·φ(η) likewise vanishes at η = 0, and is negative for η > 0.
        assert!(ProbitLink.mu_eta2(0.0).abs() < 1e-15);
        assert!(ProbitLink.mu_eta2(1.0) < 0.0);
        // cloglog: (1 − e^η) changes sign at η = 0.
        assert!(CloglogLink.mu_eta2(-0.5) > 0.0);
        assert!(CloglogLink.mu_eta2(0.5) < 0.0);
        // cauchit: −2η/(π(1+η²)²) is odd.
        assert!((CauchitLink.mu_eta2(1.3) + CauchitLink.mu_eta2(-1.3)).abs() < 1e-15);
    }

    #[test]
    fn mu_eta2_default_impl_matches_an_analytic_override() {
        // An external `Link` with no `mu_eta2` override falls back to the trait's
        // finite difference. Wrap SqrtLink so the default body is exercised.
        #[derive(Debug)]
        struct BareSqrt;
        impl Link for BareSqrt {
            fn link(&self, mu: f64) -> f64 {
                SqrtLink.link(mu)
            }
            fn inv_link(&self, eta: f64) -> f64 {
                SqrtLink.inv_link(eta)
            }
            fn mu_eta(&self, eta: f64) -> f64 {
                SqrtLink.mu_eta(eta)
            }
        }
        for &eta in &[-2.0, 0.0, 1.0, 7.5] {
            assert!(
                (BareSqrt.mu_eta2(eta) - 2.0).abs() < 1e-6,
                "default mu_eta2 at η={eta}: {}",
                BareSqrt.mu_eta2(eta)
            );
        }
    }

    #[test]
    fn mu_eta2_floored_log_is_zero_below_floor() {
        let l = FlooredLogLink { floor: 2.0 };
        // Below the floor μ is pinned, so both derivatives vanish. This hard zero is
        // the one deliberate exception to the clamp convention (see `Link::mu_eta2`).
        assert_eq!(l.mu_eta2(0.0), 0.0);
        let eta = 2.0_f64;
        assert!((l.mu_eta2(eta) - eta.exp()).abs() < 1e-12);
    }

    #[test]
    fn mu_eta_floored_log_is_zero_below_floor() {
        let l = FlooredLogLink { floor: 2.0 };
        // Where exp(η) < floor, μ is pinned at the floor and the derivative is 0.
        assert_eq!(l.mu_eta(0.0), 0.0); // exp(0)=1 < 2
                                        // Above the floor it equals exp(η).
        let eta = 2.0_f64; // exp(2) ≈ 7.39 > 2
        assert!((l.mu_eta(eta) - eta.exp()).abs() < 1e-12);
    }

    #[test]
    fn mu_eta_reconstructs_hardcoded_gaussian_weights() {
        // Regression target for the future generic-weight refactor: the chain-rule
        // reconstruction u_η = mu_eta·∂l/∂θ, w_η = mu_eta²·i_θ must equal the
        // weights gaussian.rs hard-codes today.
        let (y, mu, sigma) = (1.3_f64, 0.4_f64, 0.8_f64);

        // μ: identity link, mu_eta = 1. Natural scale: ∂l/∂μ = (y−μ)/σ², i_μ = 1/σ².
        let me_mu = IdentityLink.mu_eta(0.4);
        let dl_dmu = (y - mu) / (sigma * sigma);
        let i_mu = 1.0 / (sigma * sigma);
        let u_mu = me_mu * dl_dmu;
        let w_mu = me_mu * me_mu * i_mu;
        assert!((u_mu - (y - mu) / (sigma * sigma)).abs() < 1e-12);
        assert!((w_mu - 1.0 / (sigma * sigma)).abs() < 1e-12);

        // σ: log link, mu_eta = σ. Natural scale: ∂l/∂σ = ((y−μ)²−σ²)/σ³, i_σ = 2/σ².
        let me_sigma = LogLink.mu_eta(sigma.ln());
        assert!((me_sigma - sigma).abs() < 1e-9);
        let dl_dsigma = ((y - mu).powi(2) - sigma * sigma) / sigma.powi(3);
        let i_sigma = 2.0 / (sigma * sigma);
        let u_sigma = me_sigma * dl_dsigma;
        let w_sigma = me_sigma * me_sigma * i_sigma;
        // Matches gaussian.rs: u_sigma = ((y−μ)²−σ²)/σ², w_sigma = 2.
        assert!((u_sigma - ((y - mu).powi(2) - sigma * sigma) / (sigma * sigma)).abs() < 1e-12);
        assert!((w_sigma - 2.0).abs() < 1e-12);
    }

    #[test]
    fn mu_eta_inverse_links_match_finite_difference() {
        // Inverse is smooth away from 0 (both signs); inverse-square is positive-η only.
        for &eta in &[-5.0, -2.0, -0.5, 0.5, 2.0, 5.0] {
            let l = InverseLink;
            let (a, n) = (l.mu_eta(eta), fd_mu_eta(&l, eta));
            assert!(
                (a - n).abs() <= 1e-4 * (1.0 + a.abs()),
                "inverse at η={eta}: analytic {a} vs fd {n}"
            );
        }
        for &eta in &[0.5, 1.0, 2.0, 5.0, 10.0] {
            let l = InverseSquareLink;
            let (a, n) = (l.mu_eta(eta), fd_mu_eta(&l, eta));
            assert!(
                (a - n).abs() <= 1e-4 * (1.0 + a.abs()),
                "inverse_square at η={eta}: analytic {a} vs fd {n}"
            );
        }
    }

    #[test]
    fn bounded_links_handle_boundaries() {
        // probit / cloglog / cauchit all map to (0,1); link(0), link(1) stay finite
        // and inv_link saturates inside [0,1].
        for l in [
            &ProbitLink as &dyn Link,
            &CloglogLink as &dyn Link,
            &CauchitLink as &dyn Link,
        ] {
            assert!(l.link(0.0).is_finite(), "{l:?} link(0)");
            assert!(l.link(1.0).is_finite(), "{l:?} link(1)");
            assert!(
                (0.0..=1.0).contains(&l.inv_link(-1e6)),
                "{l:?} inv_link(-1e6)"
            );
            assert!(
                (0.0..=1.0).contains(&l.inv_link(1e6)),
                "{l:?} inv_link(1e6)"
            );
        }
    }

    #[test]
    fn positive_links_stay_finite_at_extremes() {
        for l in [
            &InverseLink as &dyn Link,
            &InverseSquareLink as &dyn Link,
            &SqrtLink as &dyn Link,
        ] {
            assert!(l.inv_link(1e-12).is_finite(), "{l:?} inv_link(1e-12)");
            assert!(l.inv_link(1e6).is_finite(), "{l:?} inv_link(1e6)");
            assert!(l.link(MIN_POSITIVE).is_finite(), "{l:?} link(tiny)");
        }
    }

    #[test]
    fn link_context_materializes_both_derivatives_per_parameter() {
        let eta_mu = Array1::from(vec![-1.0, 0.0, 2.0]);
        let eta_sigma = Array1::from(vec![0.5, 0.5, 0.5]);
        let (id, log) = (IdentityLink, LogLink);
        let ctx = LinkContext::new([
            ("mu", &id as &dyn Link, &eta_mu),
            ("sigma", &log as &dyn Link, &eta_sigma),
        ]);

        // Identity: dμ/dη ≡ 1, d²μ/dη² ≡ 0.
        assert_eq!(ctx.mu_eta("mu").unwrap(), &Array1::from(vec![1.0; 3]));
        assert_eq!(ctx.mu_eta2("mu").unwrap(), &Array1::from(vec![0.0; 3]));

        // Log: both derivatives equal μ = e^η.
        for (&got, &eta) in ctx.mu_eta("sigma").unwrap().iter().zip(eta_sigma.iter()) {
            assert!((got - eta.exp()).abs() < 1e-12);
        }
        for (&got, &eta) in ctx.mu_eta2("sigma").unwrap().iter().zip(eta_sigma.iter()) {
            assert!((got - eta.exp()).abs() < 1e-12);
        }
    }

    #[test]
    fn link_context_reports_a_missing_parameter() {
        let eta = Array1::from(vec![0.0]);
        let id = IdentityLink;
        let ctx = LinkContext::new([("mu", &id as &dyn Link, &eta)]);

        let err = ctx.mu_eta("nu").unwrap_err().to_string();
        assert!(err.contains("nu"), "unexpected message: {err}");
        assert!(ctx.mu_eta2("nu").is_err());
        assert!(ctx.link_and_eta("nu").is_none());

        let (link, got_eta) = ctx.link_and_eta("mu").expect("present");
        assert_eq!(link.name(), "identity");
        assert_eq!(got_eta, &eta);
    }

    #[test]
    fn link_from_name_round_trips_every_selectable_link() {
        for l in [
            &IdentityLink as &dyn Link,
            &LogLink,
            &LogitLink,
            &ProbitLink,
            &CloglogLink,
            &InverseLink,
            &InverseSquareLink,
            &SqrtLink,
            &CauchitLink,
        ] {
            let rebuilt = link_from_name(l.name()).expect("registered link");
            assert_eq!(rebuilt.name(), l.name());
        }
        assert!(link_from_name("nonexistent").is_err());
        // The internal floored-log link is deliberately not user-selectable.
        assert!(link_from_name("floored_log").is_err());
    }

    #[cfg(not(target_arch = "wasm32"))]
    proptest! {
        #[test]
        fn identity_link_round_trip(x in -1e6f64..1e6) {
            let l = IdentityLink;
            prop_assert!(close(l.link(l.inv_link(x)), x, 1e-9));
        }

        #[test]
        fn log_link_round_trip_in_safe_range(eta in (MIN_ETA + 1.0) .. (MAX_ETA - 1.0)) {
            let l = LogLink;
            prop_assert!(close(l.link(l.inv_link(eta)), eta, 1e-9));
        }

        #[test]
        fn logit_link_round_trip_in_safe_range(eta in -20.0f64..20.0) {
            let l = LogitLink;
            prop_assert!(close(l.link(l.inv_link(eta)), eta, 1e-6));
        }

        #[test]
        fn log_link_inv_link_strictly_positive(eta in -50.0f64..50.0) {
            let l = LogLink;
            prop_assert!(l.inv_link(eta) > 0.0);
        }

        #[test]
        fn logit_link_inv_link_in_unit_interval(eta in -1e6f64..1e6) {
            let l = LogitLink;
            let p = l.inv_link(eta);
            prop_assert!((0.0..=1.0).contains(&p));
        }

        #[test]
        fn probit_link_round_trip_in_safe_range(eta in -5.0f64..5.0) {
            let l = ProbitLink;
            prop_assert!(close(l.link(l.inv_link(eta)), eta, 1e-4));
        }

        #[test]
        fn cloglog_link_round_trip_in_safe_range(eta in -5.0f64..2.0) {
            // Above η≈2 the inv_link saturates to 1.0 and the round trip is lost.
            let l = CloglogLink;
            prop_assert!(close(l.link(l.inv_link(eta)), eta, 1e-4));
        }

        #[test]
        fn cauchit_link_round_trip_in_safe_range(eta in -25.0f64..25.0) {
            let l = CauchitLink;
            prop_assert!(close(l.link(l.inv_link(eta)), eta, 1e-6));
        }

        #[test]
        fn inverse_link_round_trip(eta in 0.5f64..50.0) {
            let l = InverseLink;
            prop_assert!(close(l.link(l.inv_link(eta)), eta, 1e-9));
        }

        #[test]
        fn inverse_square_link_round_trip(eta in 0.5f64..50.0) {
            let l = InverseSquareLink;
            prop_assert!(close(l.link(l.inv_link(eta)), eta, 1e-6));
        }

        #[test]
        fn sqrt_link_round_trip(eta in 0.01f64..50.0) {
            let l = SqrtLink;
            prop_assert!(close(l.link(l.inv_link(eta)), eta, 1e-9));
        }

        #[test]
        fn bounded_links_inv_link_in_unit_interval(eta in -1e6f64..1e6) {
            for p in [ProbitLink.inv_link(eta), CloglogLink.inv_link(eta), CauchitLink.inv_link(eta)] {
                prop_assert!((0.0..=1.0).contains(&p));
            }
        }
    }
}
