//! Linear algebra backend abstraction layer.
//!
//! This module provides a unified interface for linear algebra operations,
//! supporting multiple backends:
//! - `openblas`: Uses ndarray-linalg with OpenBLAS (default, highest performance)
//! - `pure-rust`: Uses nalgebra for pure Rust implementation (WASM-compatible, no relaxed SIMD)
//!
//! The backend is selected at compile time via feature flags.

use crate::GamlssError;
use ndarray::{Array1, Array2};

/// Result type for linear algebra operations.
pub type Result<T> = std::result::Result<T, GamlssError>;

// =============================================================================
// OpenBLAS Backend (default)
// =============================================================================

#[cfg(feature = "openblas")]
pub fn solve(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    use ndarray_linalg::Solve;
    Ok(a.solve(b)?)
}

#[cfg(feature = "openblas")]
pub fn inv(a: &Array2<f64>) -> Result<Array2<f64>> {
    use ndarray_linalg::Inverse;
    Ok(a.inv()?)
}

#[cfg(feature = "openblas")]
pub fn cholesky_lower(a: &Array2<f64>) -> Result<Array2<f64>> {
    use ndarray_linalg::{Cholesky, UPLO};
    Ok(a.cholesky(UPLO::Lower)?)
}

// =============================================================================
// Pure Rust Backend (nalgebra) - WASM compatible, no relaxed SIMD
// =============================================================================

#[cfg(feature = "pure-rust")]
pub fn solve(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    let a_na = to_dmatrix(a);
    let b_na = nalgebra::DVector::from_iterator(b.len(), b.iter().copied());
    let x = a_na.lu().solve(&b_na).ok_or_else(|| {
        GamlssError::Linalg("Linear system is singular or ill-conditioned".to_string())
    })?;
    Ok(Array1::from_iter(x.iter().copied()))
}

#[cfg(feature = "pure-rust")]
pub fn inv(a: &Array2<f64>) -> Result<Array2<f64>> {
    let a_na = to_dmatrix(a);
    let inv = a_na.try_inverse().ok_or_else(|| {
        GamlssError::Linalg("Matrix is singular, cannot compute inverse".to_string())
    })?;
    Ok(from_dmatrix(&inv))
}

#[cfg(feature = "pure-rust")]
pub fn cholesky_lower(a: &Array2<f64>) -> Result<Array2<f64>> {
    let a_na = to_dmatrix(a);
    let chol = a_na.cholesky().ok_or_else(|| {
        GamlssError::Linalg(
            "Cholesky decomposition failed (matrix not positive definite)".to_string(),
        )
    })?;
    Ok(from_dmatrix(&chol.l()))
}

#[cfg(feature = "pure-rust")]
fn to_dmatrix(arr: &Array2<f64>) -> nalgebra::DMatrix<f64> {
    let (nrows, ncols) = arr.dim();
    nalgebra::DMatrix::from_fn(nrows, ncols, |i, j| arr[[i, j]])
}

#[cfg(feature = "pure-rust")]
fn from_dmatrix(mat: &nalgebra::DMatrix<f64>) -> Array2<f64> {
    Array2::from_shape_fn((mat.nrows(), mat.ncols()), |(i, j)| mat[(i, j)])
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_solve() {
        let a = array![[4.0, 2.0], [2.0, 3.0]];
        let b = array![8.0, 7.0];

        let x = solve(&a, &b).unwrap();

        // Check Ax ≈ b
        let ax = a.dot(&x);
        assert!((ax[0] - b[0]).abs() < 1e-10);
        assert!((ax[1] - b[1]).abs() < 1e-10);
    }

    #[test]
    fn test_inv() {
        let a = array![[4.0, 2.0], [2.0, 3.0]];

        let a_inv = inv(&a).unwrap();

        // Check A * A^-1 ≈ I
        let identity = a.dot(&a_inv);
        assert!((identity[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((identity[[1, 1]] - 1.0).abs() < 1e-10);
        assert!(identity[[0, 1]].abs() < 1e-10);
        assert!(identity[[1, 0]].abs() < 1e-10);
    }

    #[test]
    fn test_cholesky() {
        // Positive definite matrix
        let a = array![[4.0, 2.0], [2.0, 3.0]];

        let l = cholesky_lower(&a).unwrap();

        // Check L * L^T ≈ A
        let lt = l.t().to_owned();
        let reconstructed = l.dot(&lt);
        assert!((reconstructed[[0, 0]] - a[[0, 0]]).abs() < 1e-10);
        assert!((reconstructed[[0, 1]] - a[[0, 1]]).abs() < 1e-10);
        assert!((reconstructed[[1, 0]] - a[[1, 0]]).abs() < 1e-10);
        assert!((reconstructed[[1, 1]] - a[[1, 1]]).abs() < 1e-10);
    }

    #[test]
    fn test_cholesky_not_positive_definite() {
        // Not positive definite (negative eigenvalue)
        let a = array![[1.0, 2.0], [2.0, 1.0]];

        let result = cholesky_lower(&a);
        assert!(result.is_err());
    }
}
