//! Type-safe wrappers and core data structures used throughout GAMLSS fitting.
//!
//! Submodules:
//! - [`newtypes`]: `Coefficients`, `CovarianceMatrix`, `ModelMatrix`, … with
//!   `Deref` to the underlying ndarray types.
//! - [`dataset`]: the [`DataSet`] column-collection with length invariants.
//! - [`formula`]: the [`Formula`] parameter → terms map.
//! - [`param`]: the [`Param`] enum of distribution-parameter names.

mod dataset;
mod formula;
mod newtypes;
mod param;
mod parse;

pub use dataset::DataSet;
pub use formula::Formula;
pub use newtypes::{Coefficients, CovarianceMatrix};
pub(crate) use newtypes::{ModelMatrix, PenaltyMatrix};
pub use param::Param;
pub use parse::parse_formula_string;
