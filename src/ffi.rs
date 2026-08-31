//! Shared infrastructure for the `python` binding layer.
//!
//! Home of [`FamilyHandle`], which turns a
//! [`FamilyDescriptor`](crate::distributions::FamilyDescriptor) into a boxed
//! [`Distribution`](crate::distributions::Distribution) and keeps it alive across
//! however many `fit` / `predict` calls the binding makes. The point of going
//! through `FamilyDescriptor` instead of a hand-maintained enum of concrete family
//! types is coverage: every family the descriptor already knows about (`Binomial`,
//! `Ocat`, and the `Censored`/`Truncated`/`Hurdle` structural wrappers) is
//! reachable here for free, with no second name roster that could quietly fall out
//! of sync.

#![cfg(feature = "python")]

use crate::distributions::{Distribution, FamilyDescriptor, Ocat};
use crate::error::GamlssError;

/// A concrete distribution built from a [`FamilyDescriptor`], kept alive across
/// multiple `fit` / `predict` calls from the binding layer.
pub(crate) struct FamilyHandle {
    descriptor: FamilyDescriptor,
    distribution: Box<dyn Distribution>,
}

impl FamilyHandle {
    /// Build a handle from a descriptor, reconstructing the boxed distribution
    /// (recursively, for the structural wrappers).
    pub(crate) fn from_descriptor(descriptor: FamilyDescriptor) -> Result<Self, GamlssError> {
        let distribution = descriptor.build()?;
        Ok(Self {
            descriptor,
            distribution,
        })
    }

    /// View the inner distribution as a trait object.
    pub(crate) fn as_distribution(&self) -> &dyn Distribution {
        self.distribution.as_ref()
    }

    /// Rebuild the concrete `Ocat` this handle describes, if that's what it is.
    ///
    /// `Ocat` carries nothing but `n_categories`, so it's cheap to just rebuild.
    /// That lets this hand back an owned value rather than fighting a downcast out
    /// of the boxed trait object.
    pub(crate) fn as_ocat(&self) -> Option<Ocat> {
        match &self.descriptor {
            FamilyDescriptor::Ocat { n_categories } => Some(Ocat::new(*n_categories)),
            _ => None,
        }
    }
}
