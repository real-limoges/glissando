use ndarray::ShapeError;
use thiserror::Error;
use polars::prelude::PolarsError;
// I haven't tried thiserror before.
// I've only written web stuff in Rust, so it had a different pattern
// I kind of like the other pattern beter....
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
// impl From<ShapeError> for GamlssError {
//     fn from(err: ShapeError) -> Self {
//         GamlssError::Shape(err.to_string())
//     }
// }
impl From<ShapeError> for GamlssError {
    fn from(err: ShapeError) -> Self {
        GamlssError::Shape(err.to_string())
    }
}