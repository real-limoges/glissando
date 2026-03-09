//! Error types for the glissando library.
//!
//! Defines [`GamlssError`], the unified error enum covering input validation,
//! numerical failures, convergence issues, and internal logic errors.

use ndarray::ShapeError;
use thiserror::Error;

/// Errors that can occur during GAMLSS model fitting, prediction, or serialization.
#[derive(Debug, Error)]
pub enum GamlssError {
    /// L-BFGS or other optimizer failed.
    #[error("Optimization failed: {0}")]
    Optimization(String),

    /// Linear algebra operation failed (e.g., singular matrix).
    #[cfg(feature = "openblas")]
    #[error("Linear algebra error: {0}")]
    Linalg(#[from] ndarray_linalg::error::LinalgError),

    /// Linear algebra operation failed (e.g., singular matrix).
    #[cfg(feature = "pure-rust")]
    #[error("Linear algebra error: {0}")]
    Linalg(String),

    /// Array shape mismatch.
    #[error("Array shape error: {0}")]
    Shape(String),

    /// RS algorithm did not converge within the iteration limit.
    #[error("PIRLS algorithm failed to converge after {0} iterations")]
    Convergence(usize),

    /// Invalid user input.
    #[error("Invalid input: {0}")]
    Input(String),

    /// Computation error from ndarray shape operations.
    #[error("ShapeError (Private): {0}")]
    ComputationError(String),

    /// Requested parameter not defined by the distribution.
    #[error("Unknown parameter '{param}' for distribution '{distribution}'")]
    UnknownParameter { distribution: String, param: String },

    /// Internal logic error (indicates a bug).
    #[error("Internal error: {0}")]
    Internal(String),

    /// Variable referenced in formula not found in dataset.
    #[error("Variable '{name}' not found in data")]
    MissingVariable { name: String },

    /// Variable contains NaN or infinite values.
    #[error("Variable '{name}' contains {count} non-finite values (NaN or Inf)")]
    NonFiniteValues { name: String, count: usize },

    /// No formula terms specified for a required parameter.
    #[error("Formula missing terms for distribution parameter '{param}'")]
    MissingFormula { param: String },

    /// Dataset has zero observations.
    #[error("Empty dataset: no observations provided")]
    EmptyData,
}

impl From<argmin::core::Error> for GamlssError {
    fn from(e: argmin::core::Error) -> Self {
        GamlssError::Optimization(e.to_string())
    }
}
impl From<ShapeError> for GamlssError {
    fn from(err: ShapeError) -> Self {
        GamlssError::Shape(err.to_string())
    }
}
