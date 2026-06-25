//! Link functions mapping a response-scale parameter `μ` to the linear predictor
//! `η = g(μ)`. Used by the IRLS scoring step (via `inv_link` to materialise `μ`
//! from `η`) and by the design-matrix code (via `link` to seed initial values).

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
    /// η-scale weights for *any* link — the prerequisite for the link expansion
    /// (probit, cloglog, …).
    ///
    /// The default is a symmetric finite difference so external `Link` impls keep
    /// compiling; the built-in links override it with their closed forms.
    fn mu_eta(&self, eta: f64) -> f64 {
        let h = 1e-6;
        (self.inv_link(eta + h) - self.inv_link(eta - h)) / (2.0 * h)
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
}

/// Log link: `η = log(μ)`. Used for positive parameters (Poisson rate, Gamma mean).
///
/// `link` clamps `log(μ)` from below at `MIN_ETA = -30.0`; `inv_link` clamps `η`
/// from above at `MAX_ETA = 30.0` to keep `exp(η)` finite.
#[derive(Debug, Clone, Copy, Default)]
pub struct LogLink;

impl Link for LogLink {
    fn link(&self, mu: f64) -> f64 {
        mu.ln().max(MIN_ETA)
    }
    fn inv_link(&self, eta: f64) -> f64 {
        eta.min(MAX_ETA).exp()
    }
    fn mu_eta(&self, eta: f64) -> f64 {
        // μ = e^η ⇒ dμ/dη = μ. Clamp η to match `inv_link`.
        eta.min(MAX_ETA).exp()
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
        mu.max(self.floor).ln().max(MIN_ETA)
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

    /// Symmetric finite-difference of the inverse link — the trait's default impl,
    /// recomputed here as an independent oracle for the analytic overrides.
    fn fd_mu_eta(link: &dyn Link, eta: f64) -> f64 {
        let h = 1e-6;
        (link.inv_link(eta + h) - link.inv_link(eta - h)) / (2.0 * h)
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
        let links: [&dyn Link; 4] = [
            &IdentityLink,
            &LogLink,
            &FlooredLogLink { floor: 2.0 },
            &LogitLink,
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
    }
}
