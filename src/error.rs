use ndarray::ShapeError;
use polars::prelude::PolarsError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GamlssError {
    #[error("Optimization failed: {0}")]
    Optimization(String),

    #[error("Linear algebra error: {0}")]
    Linalg(#[from] ndarray_linalg::error::LinalgError),

    #[error("Array shape error: {0}")]
    Shape(String),

    #[error("PIRLS algorithm failed to converge after {0} iterations")]
    Convergence(usize),

    #[error("Invalid input: {0}")]
    Input(String),

    #[error("Polars error: {0}")]
    Polars(#[from] PolarsError),

    #[error("ShapeError (Private): {0}")]
    ComputationError(String),

    #[error("Unknown parameter '{param}' for distribution '{distribution}'")]
    UnknownParameter { distribution: String, param: String },

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Column '{column}' not found in DataFrame")]
    MissingColumn { column: String },

    #[error("Column '{column}' contains {count} non-finite values (NaN or Inf)")]
    NonFiniteValues { column: String, count: usize },

    #[error("Formula missing terms for distribution parameter '{param}'")]
    MissingFormula { param: String },

    #[error("Empty dataset: DataFrame has no rows")]
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
