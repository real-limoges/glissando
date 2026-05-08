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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn display_optimization() {
        let s = format!("{}", GamlssError::Optimization("lbfgs failed".into()));
        assert!(s.contains("Optimization failed"));
        assert!(s.contains("lbfgs failed"));
    }

    #[test]
    fn display_shape() {
        let s = format!("{}", GamlssError::Shape("bad dim".into()));
        assert!(s.contains("Array shape error"));
    }

    #[test]
    fn display_convergence() {
        let s = format!("{}", GamlssError::Convergence(42));
        assert!(s.contains("42"));
    }

    #[test]
    fn display_input() {
        let s = format!("{}", GamlssError::Input("bad".into()));
        assert!(s.contains("Invalid input"));
    }

    #[test]
    fn display_unknown_parameter() {
        let s = format!(
            "{}",
            GamlssError::UnknownParameter {
                distribution: "Gaussian".into(),
                param: "zeta".into(),
            }
        );
        assert!(s.contains("zeta"));
        assert!(s.contains("Gaussian"));
    }

    #[test]
    fn display_missing_variable() {
        let s = format!("{}", GamlssError::MissingVariable { name: "x".into() });
        assert!(s.contains("'x'"));
    }

    #[test]
    fn display_non_finite_values() {
        let s = format!(
            "{}",
            GamlssError::NonFiniteValues {
                name: "y".into(),
                count: 3
            }
        );
        assert!(s.contains("3"));
        assert!(s.contains("'y'"));
    }

    #[test]
    fn display_missing_formula() {
        let s = format!(
            "{}",
            GamlssError::MissingFormula {
                param: "sigma".into()
            }
        );
        assert!(s.contains("sigma"));
    }

    #[test]
    fn display_empty_data() {
        let s = format!("{}", GamlssError::EmptyData);
        assert!(s.contains("Empty"));
    }

    #[test]
    fn from_shape_error() {
        // Trigger a real ShapeError by reshaping incompatibly.
        let res: Result<_, _> = Array2::<f64>::zeros((2, 3)).into_shape_with_order((4, 4));
        let err = res.unwrap_err();
        let g: GamlssError = err.into();
        assert!(matches!(g, GamlssError::Shape(_)));
    }
}
