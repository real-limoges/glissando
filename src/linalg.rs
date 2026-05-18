//! Linear algebra backend abstraction.
//!
//! Two mutually exclusive backends are selected at compile time:
//! - `openblas` (default): `ndarray-linalg` with system OpenBLAS — fastest on native targets.
//! - `pure-rust`: `nalgebra` — no system dependencies, WASM-compatible.
//!
//! Each backend module exposes the same `solve`, `inv`, and `cholesky_lower` functions,
//! and the file-level `pub use` re-exports the active backend's set.

use crate::GamlssError;

/// Result type for linear algebra operations.
pub type Result<T> = std::result::Result<T, GamlssError>;

#[cfg(feature = "openblas")]
mod backend {
    use super::Result;
    use crate::GamlssError;
    use ndarray::{Array1, Array2};
    use ndarray_linalg::{Cholesky, Inverse, Solve, UPLO};

    fn lin<E: std::fmt::Display>(e: E) -> GamlssError {
        GamlssError::Linalg(e.to_string())
    }

    pub fn solve(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
        a.solve(b).map_err(lin)
    }

    pub fn inv(a: &Array2<f64>) -> Result<Array2<f64>> {
        a.inv().map_err(lin)
    }

    pub fn cholesky_lower(a: &Array2<f64>) -> Result<Array2<f64>> {
        a.cholesky(UPLO::Lower).map_err(lin)
    }
}

#[cfg(feature = "pure-rust")]
mod backend {
    use super::Result;
    use crate::GamlssError;
    use nalgebra::{DMatrix, DVector};
    use ndarray::{Array1, Array2};

    /// Convert an `Array2<f64>` (row-major in standard layout) to a column-major nalgebra `DMatrix`.
    ///
    /// Uses the underlying contiguous slice when available (a memcpy plus a transpose);
    /// falls back to element-wise iteration for non-standard layouts.
    fn to_dmatrix(arr: &Array2<f64>) -> DMatrix<f64> {
        let (nrows, ncols) = arr.dim();
        match arr.as_slice() {
            // `as_slice` returns row-major data; `from_row_slice` matches that layout.
            Some(s) => DMatrix::from_row_slice(nrows, ncols, s),
            None => DMatrix::from_iterator(nrows, ncols, arr.t().iter().copied()),
        }
    }

    /// Convert a column-major nalgebra `DMatrix` back to a row-major `Array2<f64>`.
    fn from_dmatrix(mat: &DMatrix<f64>) -> Array2<f64> {
        let (nrows, ncols) = (mat.nrows(), mat.ncols());
        // `mat.transpose().as_slice()` materializes the matrix in row-major order
        // (transposing column-major → column-major-of-transpose = row-major-of-original).
        // `from_shape_vec` then takes that contiguous slice without a per-element copy.
        let transposed = mat.transpose();
        Array2::from_shape_vec((nrows, ncols), transposed.as_slice().to_vec())
            .expect("dim of transposed nalgebra slice matches (nrows, ncols)")
    }

    pub fn solve(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
        let a_na = to_dmatrix(a);
        let b_na = DVector::from_iterator(b.len(), b.iter().copied());
        let x = a_na.lu().solve(&b_na).ok_or_else(|| {
            GamlssError::Linalg("Linear system is singular or ill-conditioned".to_string())
        })?;
        Ok(Array1::from_iter(x.iter().copied()))
    }

    pub fn inv(a: &Array2<f64>) -> Result<Array2<f64>> {
        let a_na = to_dmatrix(a);
        let inv = a_na.try_inverse().ok_or_else(|| {
            GamlssError::Linalg("Matrix is singular, cannot compute inverse".to_string())
        })?;
        Ok(from_dmatrix(&inv))
    }

    pub fn cholesky_lower(a: &Array2<f64>) -> Result<Array2<f64>> {
        let a_na = to_dmatrix(a);
        let chol = a_na.cholesky().ok_or_else(|| {
            GamlssError::Linalg(
                "Cholesky decomposition failed (matrix not positive definite)".to_string(),
            )
        })?;
        Ok(from_dmatrix(&chol.l()))
    }
}

pub use backend::{cholesky_lower, inv, solve};

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_solve() {
        let a = array![[4.0, 2.0], [2.0, 3.0]];
        let b = array![8.0, 7.0];
        let x = solve(&a, &b).unwrap();
        let ax = a.dot(&x);
        assert!((ax[0] - b[0]).abs() < 1e-10);
        assert!((ax[1] - b[1]).abs() < 1e-10);
    }

    #[test]
    fn test_inv() {
        let a = array![[4.0, 2.0], [2.0, 3.0]];
        let a_inv = inv(&a).unwrap();
        let identity = a.dot(&a_inv);
        assert!((identity[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((identity[[1, 1]] - 1.0).abs() < 1e-10);
        assert!(identity[[0, 1]].abs() < 1e-10);
        assert!(identity[[1, 0]].abs() < 1e-10);
    }

    #[test]
    fn test_cholesky() {
        let a = array![[4.0, 2.0], [2.0, 3.0]];
        let l = cholesky_lower(&a).unwrap();
        let lt = l.t().to_owned();
        let reconstructed = l.dot(&lt);
        assert!((reconstructed[[0, 0]] - a[[0, 0]]).abs() < 1e-10);
        assert!((reconstructed[[0, 1]] - a[[0, 1]]).abs() < 1e-10);
        assert!((reconstructed[[1, 0]] - a[[1, 0]]).abs() < 1e-10);
        assert!((reconstructed[[1, 1]] - a[[1, 1]]).abs() < 1e-10);
    }

    #[test]
    fn test_cholesky_not_positive_definite() {
        // Indefinite matrix (eigenvalues -1 and 3).
        let a = array![[1.0, 2.0], [2.0, 1.0]];
        assert!(cholesky_lower(&a).is_err());
    }
}
