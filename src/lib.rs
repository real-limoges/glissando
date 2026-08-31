#![recursion_limit = "1024"]
//! Generalized Additive Models for Location, Scale, and Shape (GAMLSS) in Rust.
//!
//! Ordinary regression models the mean and stops there. GAMLSS models the whole
//! shape of the response: the mean, yes, but also the spread, the skew, and the
//! kurtosis, each as its own function of the predictors. It does that through the
//! Rigby-Stasinopoulos algorithm, with penalized B-splines carrying the nonlinear
//! effects.
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

/// The exact `ndarray` major this crate is built against, re-exported. Build and
/// read your `Array1`/`Array2` through `glissando::ndarray::…` and they unify
/// with this crate's public API (`predict`, `predict_with_se`) for free. Skip it
/// and pin the wrong `ndarray` version yourself, and you get a type-mismatch
/// error that reads like the two arrays are unrelated when they only differ by a
/// version number.
pub use ndarray;

pub use error::GamlssError;
pub use fitting::diagnostics::{self, ModelDiagnostics};
pub use fitting::mixture::{fit_mixture, MixtureModel};
pub use fitting::selection::{self, Direction, IcRow, LrTest, StepRecord, StepResult, StepScope};
pub use fitting::{FitConfig, FitDiagnostics, NaAction, ParamDiagnostic, SmoothingCriterion};
pub use model::{GamlssModel, PredictionResult};
pub use terms::{Contrast, Smooth, Term};
pub use types::{parse_formula_string, Coefficients, CovarianceMatrix, DataSet, Formula};
