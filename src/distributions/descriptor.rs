//! [`FamilyDescriptor`] — a serializable description of a distribution family,
//! the descriptor-aware successor to [`from_name`](super::from_name) (SER-1).
//!
//! A bare name round-trips the stateless families, but [`Binomial`] / [`Ocat`]
//! carry per-observation state, and the structural wrappers carry both state and
//! a (recursively described) base family. This enum captures all of them so
//! [`GamlssModel::to_json`](crate::GamlssModel::to_json) /
//! [`from_json`](crate::GamlssModel::from_json) can rebuild any model's family.

use super::{
    from_name, Binomial, CensorStatus, Censored, Distribution, GamlssError, Hurdle, Ocat, Truncated,
};
use ndarray::Array1;

/// A reconstructable description of a [`Distribution`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FamilyDescriptor {
    /// A stateless family addressed by name (see [`from_name`](super::from_name)).
    Named(String),
    /// [`Binomial`] with per-observation trial counts.
    Binomial { n_trials: Vec<f64> },
    /// [`Ocat`] with its category count.
    Ocat { n_categories: usize },
    /// [`Censored`] over a base family, with per-row status and interval upper
    /// bounds (infinite bounds are sentinel-encoded; see [`encode_bound`]).
    Censored {
        base: Box<FamilyDescriptor>,
        status: Vec<CensorStatus>,
        upper: Vec<f64>,
    },
    /// [`Truncated`] over a base family, with per-row `(lower, upper)` bounds.
    Truncated {
        base: Box<FamilyDescriptor>,
        lower: Vec<f64>,
        upper: Vec<f64>,
    },
    /// [`Hurdle`] over a base family (the positive part).
    Hurdle { base: Box<FamilyDescriptor> },
}

impl FamilyDescriptor {
    /// Reconstruct the boxed [`Distribution`] this descriptor describes.
    ///
    /// # Errors
    ///
    /// Returns [`GamlssError::Input`] if a [`Named`](FamilyDescriptor::Named)
    /// variant names an unknown stateless family.
    pub fn build(&self) -> Result<Box<dyn Distribution>, GamlssError> {
        Ok(match self {
            FamilyDescriptor::Named(name) => from_name(name)?,
            FamilyDescriptor::Binomial { n_trials } => {
                Box::new(Binomial::with_trials(Array1::from_vec(n_trials.clone())))
            }
            FamilyDescriptor::Ocat { n_categories } => Box::new(Ocat::new(*n_categories)),
            FamilyDescriptor::Censored {
                base,
                status,
                upper,
            } => Box::new(Censored::with_upper(
                base.build()?,
                Array1::from_vec(status.clone()),
                decode_bounds(upper),
            )),
            FamilyDescriptor::Truncated { base, lower, upper } => Box::new(Truncated::new(
                base.build()?,
                decode_bounds(lower),
                decode_bounds(upper),
            )),
            FamilyDescriptor::Hurdle { base } => Box::new(Hurdle::new(base.build()?)),
        })
    }
}

/// Map an infinite truncation/interval bound to a finite, JSON-safe sentinel.
///
/// `serde_json` cannot represent `±∞` (it emits `null`, which then fails to parse
/// back into `f64`), so `+∞ → f64::MAX` and `−∞ → f64::MIN` on the way out;
/// [`decode_bound`] reverses it. Finite bounds pass through untouched.
pub(crate) fn encode_bound(v: f64) -> f64 {
    if v == f64::INFINITY {
        f64::MAX
    } else if v == f64::NEG_INFINITY {
        f64::MIN
    } else {
        v
    }
}

/// Inverse of [`encode_bound`]: the sentinels `f64::MAX` / `f64::MIN` decode back
/// to `±∞` so the rebuilt wrapper sees true infinities (and its `is_finite`
/// fast-path fires).
fn decode_bound(v: f64) -> f64 {
    if v >= f64::MAX {
        f64::INFINITY
    } else if v <= f64::MIN {
        f64::NEG_INFINITY
    } else {
        v
    }
}

/// Encode a bound array for serialization.
pub(crate) fn encode_bounds(b: &Array1<f64>) -> Vec<f64> {
    b.iter().copied().map(encode_bound).collect()
}

/// Decode a serialized bound vector back into an array.
fn decode_bounds(b: &[f64]) -> Array1<f64> {
    Array1::from_iter(b.iter().copied().map(decode_bound))
}
