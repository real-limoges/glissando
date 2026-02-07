use ndarray::ShapeError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GamlssError {
    #[error("Optimization failed: {0}")]
    Optimization(String),

    #[cfg(feature = "openblas")]
    #[error("Linear algebra error: {0}")]
    Linalg(#[from] ndarray_linalg::error::LinalgError),

    #[cfg(feature = "pure-rust")]
    #[error("Linear algebra error: {0}")]
    Linalg(String),

    #[error("Array shape error: {0}")]
    Shape(String),

    #[error("PIRLS algorithm failed to converge after {0} iterations")]
    Convergence(usize),

    #[error("Invalid input: {0}")]
    Input(String),

    #[error("ShapeError (Private): {0}")]
    ComputationError(String),

    #[error("Unknown parameter '{param}' for distribution '{distribution}'")]
    UnknownParameter { distribution: String, param: String },

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
        GamlssError::Optimization(e.to_string())
    }
}
impl From<ShapeError> for GamlssError {
    fn from(err: ShapeError) -> Self {
        GamlssError::Shape(err.to_string())
    }
}
