//! Shared infrastructure for the `python` binding layer.
//!
//! Hosts [`FamilyHandle`], which builds a boxed
//! [`Distribution`](crate::distributions::Distribution) from a
//! [`FamilyDescriptor`](crate::distributions::FamilyDescriptor) and keeps it
//! alive across multiple `fit` / `predict` calls. Routing through
//! `FamilyDescriptor` rather than a hand-maintained enum of concrete family
//! types means every family `FamilyDescriptor` already supports — including
//! `Binomial`/`Ocat` and the `Censored`/`Truncated`/`Hurdle` structural
//! wrappers — is reachable here with no separate name roster to keep in sync.

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

    /// Rebuild the concrete `Ocat` this handle describes, if it is one.
    ///
    /// `Ocat` is cheap to reconstruct (it carries only `n_categories`), so this
    /// returns an owned value instead of needing a downcast from the boxed
    /// trait object.
    pub(crate) fn as_ocat(&self) -> Option<Ocat> {
        match &self.descriptor {
            FamilyDescriptor::Ocat { n_categories } => Some(Ocat::new(*n_categories)),
            _ => None,
        }
    }
}
