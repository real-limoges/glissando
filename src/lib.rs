#![recursion_limit = "1024"]
//! Generalized Additive Models for Location, Scale, and Shape (GAMLSS) in Rust.
//!
//! GAMLSS models each parameter of the response distribution (mean, spread,
//! skewness, kurtosis) as its own function of the predictors. Fitting uses the
//! Rigby-Stasinopoulos algorithm, with penalized smooths (P-splines, cubic
//! regression splines, tensor products, random effects) for nonlinear effects and
//! REML, GCV, or Fellner-Schall choosing the smoothing parameters.
//!
//! # What ships
//!
//! - Twelve families in [`distributions`]: Gaussian, Poisson, Student-t, Gamma,
//!   negative binomial, Beta, Weibull, BCCG, BCT, BCPE, Binomial, and ordered
//!   categorical (`Ocat`).
//! - Structural-likelihood wrappers over any base family:
//!   [`Censored`](distributions::Censored), [`Truncated`](distributions::Truncated),
//!   and [`Hurdle`](distributions::Hurdle).
//! - Finite mixtures fitted by EM: [`MixtureModel`] and [`fit_mixture`].
//! - Nine links selectable per parameter through [`FitConfig::with_link`].
//! - Formulas built from [`Term`]s or parsed from R/mgcv-style strings
//!   ([`parse_formula_string`]).
//! - Diagnostics and model selection in [`diagnostics`] and [`selection`].
//!
//! # Feature flags
//!
//! `openblas` (default) and `pure-rust` pick the linear-algebra backend and are
//! mutually exclusive. `parallel` (default) enables Rayon. `serialization` adds
//! serde support and the `json` facade. `wasm` and `python` build the
//! JavaScript and Python bindings; `wasm` needs `--no-default-features`.
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

/// The exact `ndarray` major this crate is built against, re-exported. Arrays
/// built through `glissando::ndarray::…` unify with this crate's public API
/// (`predict`, `predict_with_se`). Pinning a different `ndarray` version yourself
/// produces a type-mismatch error that makes the two array types look unrelated
/// when they differ only by version number.
pub use ndarray;

pub use error::GamlssError;
pub use fitting::diagnostics::{self, ModelDiagnostics};
pub use fitting::mixture::{fit_mixture, MixtureModel};
pub use fitting::selection::{self, Direction, IcRow, LrTest, StepRecord, StepResult, StepScope};
pub use fitting::{FitConfig, FitDiagnostics, NaAction, ParamDiagnostic, SmoothingCriterion};
pub use model::{GamlssModel, PredictionResult};
pub use terms::{Contrast, Smooth, Term};
pub use types::{parse_formula_string, Coefficients, CovarianceMatrix, DataSet, Formula};
