//! Scale-tagged derivative pairs.
//!
//! A family's score and information can live on two scales: the parameter's own
//! **natural** scale θ, or the **linear-predictor** scale η that the Fisher-scoring
//! loop works on. The numbers are the same shape either way, so before these types
//! existed a natural-scale value read as an η-scale one (or chained twice) compiled
//! and ran, and only the estimates were wrong.
//!
//! [`ScoreInfo`] and [`CdfGrad`] carry the scale as a type parameter, [`Natural`] or
//! [`Eta`]. A value can only be built through a constructor, so its scale is always a
//! deliberate choice:
//!
//! - [`ScoreInfo::new`] / [`CdfGrad::new`] build natural-scale values, which is what a
//!   family returns from [`Distribution::theta_derivatives`] and
//!   [`Distribution::cdf_theta_derivatives`].
//! - [`chain_to_eta`] is the generic way from natural to η.
//! - [`ScoreInfo::computed_on_eta`] / [`CdfGrad::computed_on_eta`] build η-scale values
//!   directly, for the few places that genuinely compute on η (Ocat, Student-t's ν,
//!   the numeric CDF fallback). The long name is the point: every such site is easy
//!   to find and to question in review.
//!
//! The fields are public, so values can be mutated in place; changing the numbers
//! never changes which scale they are on.
//!
//! A natural-scale map is rejected where an η-scale one is expected:
//!
//! ```compile_fail
//! use glissando::distributions::{DerivativeMap, Eta, Natural, ScoreInfo};
//! use glissando::Param;
//! use glissando::ndarray::array;
//!
//! fn wants_eta(_: &DerivativeMap<Eta>) {}
//!
//! let mut natural: DerivativeMap<Natural> = DerivativeMap::new();
//! natural.insert(Param::Mu, ScoreInfo::new(array![1.0], array![1.0]));
//! wants_eta(&natural); // mismatched types: Natural is not Eta
//! ```
//!
//! [`Distribution::theta_derivatives`]: super::Distribution::theta_derivatives
//! [`Distribution::cdf_theta_derivatives`]: super::Distribution::cdf_theta_derivatives
//! [`chain_to_eta`]: super::chain_to_eta

use crate::Param;
use ndarray::Array1;
use std::collections::HashMap;
use std::marker::PhantomData;

/// Type tag: the parameter's own natural scale θ (μ, σ, ν, …).
#[derive(Debug, Clone, Copy)]
pub enum Natural {}

/// Type tag: the linear-predictor scale η = g(θ).
#[derive(Debug, Clone, Copy)]
pub enum Eta {}

/// One parameter's per-observation score and expected information on scale `S`.
///
/// On [`Natural`] these are `∂l/∂θ` and `i_θ`; on [`Eta`] they are `u_η` and the
/// IRLS weight `w_η`. Either way `info` is **unfloored**: `MIN_WEIGHT` is applied
/// exactly once, downstream, in the scoring loop.
#[derive(Debug, Clone)]
pub struct ScoreInfo<S> {
    /// `∂l/∂θ` on [`Natural`], `u_η` on [`Eta`].
    pub score: Array1<f64>,
    /// `i_θ` on [`Natural`], `w_η` on [`Eta`].
    pub info: Array1<f64>,
    scale: PhantomData<fn() -> S>,
}

impl ScoreInfo<Natural> {
    /// A natural-scale score and expected information.
    pub fn new(score: Array1<f64>, info: Array1<f64>) -> Self {
        Self {
            score,
            info,
            scale: PhantomData,
        }
    }
}

impl ScoreInfo<Eta> {
    /// An η-scale score and weight computed directly, not chained from a natural-scale
    /// pair. Use [`chain_to_eta`](super::chain_to_eta) whenever the family has a
    /// separable natural scale; this is for the cases that do not.
    pub fn computed_on_eta(score: Array1<f64>, weight: Array1<f64>) -> Self {
        Self {
            score,
            info: weight,
            scale: PhantomData,
        }
    }
}

/// One parameter's per-observation CDF derivatives `(∂F/∂·, ∂²F/∂·²)` on scale `S`.
#[derive(Debug, Clone)]
pub struct CdfGrad<S> {
    /// First derivative of the CDF with respect to θ ([`Natural`]) or η ([`Eta`]).
    pub d1: Array1<f64>,
    /// Second derivative of the CDF with respect to θ ([`Natural`]) or η ([`Eta`]).
    pub d2: Array1<f64>,
    scale: PhantomData<fn() -> S>,
}

impl CdfGrad<Natural> {
    /// Natural-scale CDF derivatives `(∂F/∂θ, ∂²F/∂θ²)`.
    pub fn new(d1: Array1<f64>, d2: Array1<f64>) -> Self {
        Self {
            d1,
            d2,
            scale: PhantomData,
        }
    }
}

impl CdfGrad<Eta> {
    /// η-scale CDF derivatives `(∂F/∂η, ∂²F/∂η²)` computed directly (the numeric
    /// fallback differences on η), not chained from a natural-scale pair.
    pub fn computed_on_eta(d1: Array1<f64>, d2: Array1<f64>) -> Self {
        Self {
            d1,
            d2,
            scale: PhantomData,
        }
    }
}

/// Score / information pairs keyed by distribution parameter, on scale `S`.
pub type DerivativeMap<S> = HashMap<Param, ScoreInfo<S>>;

/// CDF derivative pairs keyed by distribution parameter, on scale `S`.
pub type CdfMap<S> = HashMap<Param, CdfGrad<S>>;
