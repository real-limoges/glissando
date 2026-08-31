//! Error types for the glissando library.
//!
//! Defines [`GamlssError`], the unified error enum covering input validation,
//! numerical failures, convergence issues, and internal logic errors.

use ndarray::ShapeError;
use thiserror::Error;

/// Errors that can occur during GAMLSS model fitting, prediction, or serialization.
#[derive(Debug, Error)]
pub enum GamlssError {
    #[error("Optimization failed: {0}")]
    Optimization(String),

    /// The payload is stringified at the backend boundary on purpose, so a
    /// `match` on this variant reads the same under both `openblas` and
    /// `pure-rust`. The two backends raise different concrete error types; only
    /// the string survives to here.
    #[error("Linear algebra error: {0}")]
    Linalg(String),

    /// Cholesky factorization of the posterior covariance failed. Usually that
    /// means a rank-deficient design, or a parameter sitting right at the
    /// boundary of its support.
    #[error("Posterior covariance is not positive definite (Cholesky failed)")]
    PosteriorNotPositiveDefinite,

    #[error("Array shape error: {0}")]
    Shape(String),

    #[error("RS algorithm failed to converge after {0} iterations")]
    Convergence(usize),

    #[error("Invalid input: {0}")]
    Input(String),

    #[error("Unknown parameter '{param}' for distribution '{distribution}'")]
    UnknownParameter { distribution: String, param: String },

    #[error("Family mismatch: model was fit with {expected} but predict was called with a different family ({actual})")]
    FamilyMismatch { expected: String, actual: String },

    /// This one is on us, not you. It means a bug in the library, not bad input.
    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Variable '{name}' not found in data")]
    MissingVariable { name: String },

    #[error("Variable '{name}' contains {count} non-finite values (NaN or Inf)")]
    NonFiniteValues { name: String, count: usize },

    #[error("Formula missing terms for distribution parameter '{param}'")]
    MissingFormula { param: String },

    #[error("Empty dataset: no observations provided")]
    EmptyData,
}

impl From<argmin::core::Error> for GamlssError {
    fn from(e: argmin::core::Error) -> Self {
        // `argmin::core::Error` is really `anyhow::Error`. `{:#}` walks the full source
        // chain (e.g. "L-BFGS step failed: line search did not converge"). Plain
        // `.to_string()` gives you the top-level message and throws the rest away, which
        // is exactly the part you needed.
        GamlssError::Optimization(format!("{:#}", e))
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
        // Force a real ShapeError by reshaping to a size that can't fit.
        let res: Result<_, _> = Array2::<f64>::zeros((2, 3)).into_shape_with_order((4, 4));
        let err = res.unwrap_err();
        let g: GamlssError = err.into();
        assert!(matches!(g, GamlssError::Shape(_)));
    }
}
