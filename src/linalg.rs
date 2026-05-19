//! Linear algebra backend abstraction.
//!
//! Two mutually exclusive backends are selected at compile time:
//! - `openblas` (default): `ndarray-linalg` with system OpenBLAS — fastest on native targets.
//! - `pure-rust`: `nalgebra` — no system dependencies, WASM-compatible.
//!
//! Each backend module exposes the same `solve`, `inv`, `cholesky_lower`,
//! `log_det_via_cholesky`, and `symmetric_eigh` functions, and the file-level
//! `pub use` re-exports the active backend's set.

// `openblas` and `pure-rust` define `mod backend { … }` with `#[cfg]` gates that
// assume exactly one is active. Cargo's feature-unification across workspace
// members can quietly activate both (e.g. `cargo --workspace --features pure-rust`
// while the `benchmark` member forces `openblas`), producing a cryptic E0428
// "name `backend` defined multiple times" instead of a useful diagnostic.
#[cfg(all(feature = "openblas", feature = "pure-rust"))]
compile_error!(
    "Features `openblas` and `pure-rust` are mutually exclusive — pick one linear-algebra backend. \
     If this fired while you ran `cargo --workspace … --features pure-rust`, the `benchmark` crate \
     is unioning `openblas` on top of your override; run cargo on the library directly instead: \
     `cargo test -p glissando --no-default-features --features pure-rust`."
);

use crate::GamlssError;

/// Result type for linear algebra operations.
pub type Result<T> = std::result::Result<T, GamlssError>;

#[cfg(feature = "openblas")]
mod backend {
    use super::Result;
    use crate::GamlssError;
    use ndarray::{Array1, Array2};
    use ndarray_linalg::{Cholesky, Eigh, Inverse, Solve, UPLO};

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

    /// log|A| via Cholesky: 2·Σ log diag(L) where L = chol(A, lower).
    /// Returns an error if A is not positive definite.
    pub fn log_det_via_cholesky(a: &Array2<f64>) -> Result<f64> {
        let l = cholesky_lower(a)?;
        let mut acc = 0.0;
        for i in 0..l.nrows() {
            acc += l[[i, i]].ln();
        }
        Ok(2.0 * acc)
    }

    /// Symmetric eigendecomposition. Returns `(eigvals_ascending, eigvecs_as_columns)`
    /// such that `A = Q · diag(d) · Qᵀ`.
    pub fn symmetric_eigh(a: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>)> {
        // ndarray-linalg's `eigh` already returns eigenvalues in ascending order.
        a.eigh(UPLO::Lower).map_err(lin)
    }
}

// `not(feature = "openblas")` keeps this mod gone when both features unify, so
// only the `compile_error!` above fires — no cryptic E0428 alongside it.
#[cfg(all(feature = "pure-rust", not(feature = "openblas")))]
mod backend {
    use super::Result;
    use crate::GamlssError;
    use nalgebra::{DMatrix, DVector, SymmetricEigen};
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

    /// log|A| via Cholesky: 2·Σ log diag(L) where L = chol(A, lower).
    /// Returns an error if A is not positive definite.
    pub fn log_det_via_cholesky(a: &Array2<f64>) -> Result<f64> {
        let l = cholesky_lower(a)?;
        let mut acc = 0.0;
        for i in 0..l.nrows() {
            acc += l[[i, i]].ln();
        }
        Ok(2.0 * acc)
    }

    /// Symmetric eigendecomposition. Returns `(eigvals_ascending, eigvecs_as_columns)`
    /// such that `A = Q · diag(d) · Qᵀ`.
    pub fn symmetric_eigh(a: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>)> {
        let a_na = to_dmatrix(a);
        let eig = SymmetricEigen::new(a_na);
        // nalgebra returns eigenvalues unsorted; sort ascending and apply permutation to vectors.
        let n = eig.eigenvalues.len();
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| {
            eig.eigenvalues[a]
                .partial_cmp(&eig.eigenvalues[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let eigvals = Array1::from_iter(idx.iter().map(|&i| eig.eigenvalues[i]));
        let mut eigvecs = Array2::<f64>::zeros((n, n));
        for (new_col, &old_col) in idx.iter().enumerate() {
            for row in 0..n {
                eigvecs[[row, new_col]] = eig.eigenvectors[(row, old_col)];
            }
        }
        Ok((eigvals, eigvecs))
    }
}

pub use backend::{cholesky_lower, inv, log_det_via_cholesky, solve, symmetric_eigh};

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

    #[test]
    fn test_log_det_via_cholesky() {
        // det([[4,2],[2,3]]) = 12 - 4 = 8, so log|A| = ln(8).
        let a = array![[4.0, 2.0], [2.0, 3.0]];
        let lhs = log_det_via_cholesky(&a).unwrap();
        let expected = 8.0_f64.ln();
        assert!((lhs - expected).abs() < 1e-10, "got {}, want {}", lhs, expected);
    }

    #[test]
    fn test_log_det_via_cholesky_not_pd_errors() {
        let a = array![[1.0, 2.0], [2.0, 1.0]];
        assert!(log_det_via_cholesky(&a).is_err());
    }

    #[test]
    fn test_symmetric_eigh_reconstruction_and_ordering() {
        // Symmetric matrix with known eigenvalues {1, 4} (eigenvectors of [[2.5,1.5],[1.5,2.5]]).
        let a = array![[2.5, 1.5], [1.5, 2.5]];
        let (d, q) = symmetric_eigh(&a).unwrap();

        // Eigenvalues ascending.
        assert!(d[0] <= d[1]);
        assert!((d[0] - 1.0).abs() < 1e-10);
        assert!((d[1] - 4.0).abs() < 1e-10);

        // Q · diag(d) · Qᵀ ≈ A.
        let qd = &q * &d.view().insert_axis(ndarray::Axis(0));
        let reconstructed = qd.dot(&q.t());
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (reconstructed[[i, j]] - a[[i, j]]).abs() < 1e-10,
                    "reconstruction mismatch at ({},{}): got {}, want {}",
                    i, j, reconstructed[[i, j]], a[[i, j]]
                );
            }
        }

        // Qᵀ Q ≈ I (orthonormal columns).
        let qtq = q.t().dot(&q);
        for i in 0..2 {
            for j in 0..2 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((qtq[[i, j]] - want).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_symmetric_eigh_rank_deficient() {
        // Rank-1 matrix: eigenvalues should be (0, 2).
        let a = array![[1.0, 1.0], [1.0, 1.0]];
        let (d, _q) = symmetric_eigh(&a).unwrap();
        assert!(d[0].abs() < 1e-10, "got d[0]={}", d[0]);
        assert!((d[1] - 2.0).abs() < 1e-10);
    }
}
