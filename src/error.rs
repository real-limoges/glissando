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
