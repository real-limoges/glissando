#![recursion_limit = "1024"]
//! Generalized Additive Models for Location, Scale, and Shape (GAMLSS) in Rust.
//!
//! GAMLSS extends traditional regression by modeling multiple distribution parameters
//! (mean, variance, shape) as functions of predictors using the Rigby-Stasinopoulos
//! algorithm with penalized B-splines for nonlinear effects.
//!
//! # Quick start
//!
//! ```
//! use glissando::{GamlssModel, DataSet, Formula, Term};
//! use glissando::distributions::Gaussian;
//! use ndarray::Array1;
//!
//! let y = Array1::from_vec(vec![2.1, 4.0, 5.9, 8.1, 10.0]);
//! let mut data = DataSet::new();
//! data.insert_column("x", Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]));
//!
//! let formula = Formula::new()
//!     .with_terms("mu", vec![Term::Intercept, Term::Linear { col_name: "x".to_string() }])
//!     .with_terms("sigma", vec![Term::Intercept]);
//!
//! let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();
//! assert!(model.converged());
//! ```

pub mod distributions;
mod error;
#[cfg(feature = "python")]
mod ffi;
pub mod fitting;
#[cfg(feature = "serialization")]
pub mod json;
mod linalg;
mod math;
mod model;
pub mod preprocessing;
#[cfg(feature = "python")]
mod python;
mod splines;
mod terms;
mod types;
#[cfg(feature = "wasm")]
pub mod wasm;

/// Re-export of the exact `ndarray` major this crate is built against. Construct
/// or consume `Array1`/`Array2` through `glissando::ndarray::…` so the types
/// unify with this crate's public API (`predict`, `predict_with_se`) without
/// pinning a matching `ndarray` version yourself.
pub use ndarray;

pub use error::GamlssError;
pub use fitting::diagnostics::{self, ModelDiagnostics};
pub use fitting::selection::{self, Direction, IcRow, LrTest, StepRecord, StepResult, StepScope};
pub use fitting::{FitConfig, FitDiagnostics, ParamDiagnostic, SmoothingCriterion};
pub use model::{GamlssModel, PredictionResult};
pub use terms::{Smooth, Term};
pub use types::{Coefficients, CovarianceMatrix, DataSet, Formula};
