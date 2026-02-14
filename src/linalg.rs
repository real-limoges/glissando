//! Linear algebra backend abstraction layer.
//!
//! This module provides a unified interface for linear algebra operations,
//! supporting multiple backends:
//! - `openblas`: Uses ndarray-linalg with OpenBLAS (default, highest performance)
//! - `pure-rust`: Uses faer for pure Rust implementation (WASM-compatible)
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
// Pure Rust Backend (faer) - WASM compatible
// =============================================================================

#[cfg(feature = "pure-rust")]
pub fn solve(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    use faer::linalg::solvers::Solve;

    // Convert ndarray -> faer
    let a_faer = ndarray_to_faer_mat(a);
    let b_faer = ndarray_to_faer_col(b);

    // Solve using LU decomposition with partial pivoting
    let plu = a_faer.partial_piv_lu();
    let x_faer = plu.solve(&b_faer);

    // Convert back to ndarray
    faer_col_to_ndarray(&x_faer)
}

#[cfg(feature = "pure-rust")]
pub fn inv(a: &Array2<f64>) -> Result<Array2<f64>> {
    use faer::linalg::solvers::DenseSolveCore;

    // Convert ndarray -> faer
    let a_faer = ndarray_to_faer_mat(a);

    // Compute inverse using LU decomposition
    let plu = a_faer.partial_piv_lu();
    let inv_faer = plu.inverse();

    // Convert back to ndarray
    faer_mat_to_ndarray(&inv_faer)
}

#[cfg(feature = "pure-rust")]
pub fn cholesky_lower(a: &Array2<f64>) -> Result<Array2<f64>> {
    // Convert ndarray -> faer
    let a_faer = ndarray_to_faer_mat(a);

    // Compute Cholesky decomposition (LLT)
    // llt() decomposes A = LL^H where L is lower triangular
    let chol = a_faer.llt(faer::Side::Lower).map_err(|_| {
        GamlssError::Linalg(
            "Cholesky decomposition failed (matrix not positive definite)".to_string(),
        )
    })?;

    // Extract the lower triangular matrix
    let l_ref = chol.L();

    // Convert to owned Mat and then to ndarray
    let l_faer = l_ref.to_owned();
    faer_mat_to_ndarray(&l_faer)
}

// =============================================================================
// Conversion Helpers: ndarray <-> faer
// =============================================================================

#[cfg(feature = "pure-rust")]
fn ndarray_to_faer_mat(arr: &Array2<f64>) -> faer::Mat<f64> {
    use faer::Mat;

    let (nrows, ncols) = arr.dim();

    // Note: ndarray is row-major, faer is column-major
    // We need to transpose during conversion
    Mat::from_fn(nrows, ncols, |i, j| arr[[i, j]])
}

#[cfg(feature = "pure-rust")]
fn ndarray_to_faer_col(arr: &Array1<f64>) -> faer::Col<f64> {
    use faer::Col;

    let n = arr.len();
    Col::from_fn(n, |i| arr[i])
}

#[cfg(feature = "pure-rust")]
fn faer_mat_to_ndarray(mat: &faer::Mat<f64>) -> Result<Array2<f64>> {
    let (nrows, ncols) = (mat.nrows(), mat.ncols());
    let result = Array2::from_shape_fn((nrows, ncols), |(i, j)| mat[(i, j)]);
    Ok(result)
}

#[cfg(feature = "pure-rust")]
fn faer_col_to_ndarray(col: &faer::Col<f64>) -> Result<Array1<f64>> {
    let n = col.nrows();
    let result = Array1::from_shape_fn(n, |i| col[i]);
    Ok(result)
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
