//! Type-safe wrappers and core data structures used throughout GAMLSS fitting.
//!
//! Submodules:
//! - [`newtypes`]: `Coefficients`, `CovarianceMatrix`, `ModelMatrix`, … with
//!   `Deref` to the underlying ndarray types.
//! - [`dataset`]: the [`DataSet`] column-collection with length invariants.
//! - [`formula`]: the [`Formula`] parameter → terms map.

mod dataset;
mod formula;
mod newtypes;
mod parse;

pub use dataset::DataSet;
pub use formula::Formula;
pub use newtypes::{Coefficients, CovarianceMatrix};
pub(crate) use newtypes::{ModelMatrix, PenaltyMatrix};
pub use parse::parse_formula_string;
