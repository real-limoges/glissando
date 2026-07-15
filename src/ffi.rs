//! Shared infrastructure for the `python` and `wasm` binding layers.
//!
//! Currently this module exists to host [`FamilyType`], a concrete enum that both
//! bindings use to dispatch into the [`Distribution`](crate::distributions::Distribution)
//! trait. Defining it once here keeps the two FFI surfaces in lockstep about which
//! distributions are supported.

#![cfg(any(feature = "python", feature = "wasm"))]

use crate::distributions::{
    Beta, Binomial, Distribution, Gamma, Gaussian, NegativeBinomial, Ocat, Poisson, StudentT,
    Weibull, BCCG, BCPE, BCT,
};
use crate::error::GamlssError;

/// Concrete distribution dispatched from a binding-layer payload (PyAny / JSON name).
///
/// Owning the concrete distribution (instead of a `Box<dyn Distribution>`) lets the
/// binding layer keep the family alive across multiple `fit` / `predict` calls and
/// avoids re-allocating a trait object on every call.
pub(crate) enum FamilyType {
    Gaussian(Gaussian),
    Poisson(Poisson),
    // Binomial and Ocat are only constructed by the python binding (their state can't be
    // serialized into the wasm name-based JSON dispatch). Suppress dead_code warnings
    // when only the wasm feature is enabled.
    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    Binomial(Binomial),
    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    Ocat(Ocat),
    Gamma(Gamma),
    NegativeBinomial(NegativeBinomial),
    Beta(Beta),
    StudentT(StudentT),
    Weibull(Weibull),
    BCCG(BCCG),
    BCT(BCT),
    BCPE(BCPE),
}

impl FamilyType {
    /// View the inner distribution as a trait object.
    pub(crate) fn as_distribution(&self) -> &dyn Distribution {
        match self {
            FamilyType::Gaussian(d) => d,
            FamilyType::Poisson(d) => d,
            FamilyType::Binomial(d) => d,
            FamilyType::Ocat(d) => d,
            FamilyType::Gamma(d) => d,
            FamilyType::NegativeBinomial(d) => d,
            FamilyType::Beta(d) => d,
            FamilyType::StudentT(d) => d,
            FamilyType::Weibull(d) => d,
            FamilyType::BCCG(d) => d,
            FamilyType::BCT(d) => d,
            FamilyType::BCPE(d) => d,
        }
    }

    /// Return a reference to the inner `Ocat` if this is an `Ocat` variant.
    #[cfg(feature = "python")]
    pub(crate) fn as_ocat(&self) -> Option<&Ocat> {
        if let FamilyType::Ocat(d) = self {
            Some(d)
        } else {
            None
        }
    }

    /// Construct a stateless distribution from its name.
    ///
    /// Mirrors [`crate::distributions::from_name`] but yields the concrete
    /// `FamilyType` enum used by the binding layers. Excludes [`Binomial`] because
    /// it requires `n_trials` state that cannot be recovered from a name alone —
    /// bindings that support Binomial must construct that variant directly.
    #[cfg_attr(not(feature = "wasm"), allow(dead_code))]
    pub(crate) fn from_name(name: &str) -> Result<Self, GamlssError> {
        match name {
            "Gaussian" => Ok(FamilyType::Gaussian(Gaussian::new())),
            "Poisson" => Ok(FamilyType::Poisson(Poisson::new())),
            "StudentT" => Ok(FamilyType::StudentT(StudentT::new())),
            "Gamma" => Ok(FamilyType::Gamma(Gamma::new())),
            "NegativeBinomial" => Ok(FamilyType::NegativeBinomial(NegativeBinomial::new())),
            "Beta" => Ok(FamilyType::Beta(Beta::new())),
            "Weibull" => Ok(FamilyType::Weibull(Weibull::new())),
            "BCCG" => Ok(FamilyType::BCCG(BCCG::new())),
            "BCT" => Ok(FamilyType::BCT(BCT::new())),
            "BCPE" => Ok(FamilyType::BCPE(BCPE::new())),
            other => Err(GamlssError::Input(format!(
                "Unknown distribution: '{}'. Supported: Gaussian, Poisson, StudentT, Gamma, NegativeBinomial, Beta, Weibull, BCCG, BCT, BCPE",
                other
            ))),
        }
    }
}
