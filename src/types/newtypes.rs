//! Type-safe newtype wrappers around the linear-algebra primitives the fitter and
//! solver lean on: `Coefficients`, `CovarianceMatrix`, `ModelMatrix`, …
//!
//! Each wrapper carries an `Array1`/`Array2` and provides `Deref`, so callers can
//! use it as if it were the bare ndarray type, granted once via the
//! `impl_deref_for_vector_wrapper!` / `impl_deref_for_matrix_wrapper!` macros.

use ndarray::{Array1, Array2};
use std::ops::{Deref, DerefMut};

/// Regression coefficient vector. Derefs to `Array1<f64>`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct Coefficients(pub Array1<f64>);

macro_rules! impl_deref_for_vector_wrapper {
    ($t:ty) => {
        impl Deref for $t {
            type Target = Array1<f64>;
            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl DerefMut for $t {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }
    };
}

impl_deref_for_vector_wrapper!(Coefficients);

/// Design matrix (n_obs x n_coeffs). Derefs to `Array2<f64>`.
#[derive(Debug, Clone)]
pub(crate) struct ModelMatrix(pub Array2<f64>);

/// Penalty matrix for a smooth term. Stores only its own contiguous coefficient
/// block (never the full model width), plus the offset at which that block sits in
/// the full coefficient vector.
#[derive(Debug, Clone)]
pub(crate) struct PenaltyMatrix {
    pub(crate) offset: usize,
    pub(crate) block: Array2<f64>,
}

impl PenaltyMatrix {
    /// Inclusive `[start, end]` range in the full coefficient space this
    /// penalty's non-zero entries occupy. Matches the convention already
    /// used by `PenaltyGroups`/`group_penalties`.
    pub(crate) fn block_range(&self) -> (usize, usize) {
        (self.offset, self.offset + self.block.nrows() - 1)
    }
}

/// Covariance matrix of coefficient estimates, V = (X'WX + Σλ·S)⁻¹. Derefs to `Array2<f64>`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct CovarianceMatrix(pub Array2<f64>);

macro_rules! impl_deref_for_matrix_wrapper {
    ($t:ty) => {
        impl Deref for $t {
            type Target = Array2<f64>;
            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }
        impl DerefMut for $t {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }
    };
}

impl_deref_for_matrix_wrapper!(CovarianceMatrix);
impl_deref_for_matrix_wrapper!(ModelMatrix);

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn coefficients_deref_to_array1() {
        let a = Coefficients(array![1.0, 2.0]);
        // Access via Deref → Array1
        assert_eq!(a.len(), 2);
        let mut b = Coefficients(array![5.0, 6.0]);
        b[0] = 99.0; // DerefMut
        assert_eq!(b.0[0], 99.0);
    }

    // --- Newtype wrapper deref ---

    #[test]
    fn matrix_wrappers_deref_to_array2() {
        let m = ModelMatrix(Array2::from_shape_fn((2, 3), |(i, j)| (i + j) as f64));
        assert_eq!(m.dim(), (2, 3));
        let c = CovarianceMatrix(Array2::<f64>::zeros((2, 2)));
        assert_eq!(c.dim(), (2, 2));
    }

    #[test]
    fn penalty_matrix_block_range() {
        let p = PenaltyMatrix {
            offset: 2,
            block: Array2::<f64>::eye(3),
        };
        assert_eq!(p.block.dim(), (3, 3));
        assert_eq!(p.block_range(), (2, 4));
    }

    // --- Serialization round-trip ---

    #[cfg(feature = "serde")]
    #[test]
    fn coefficients_json_round_trip() {
        let c = Coefficients(array![1.5, 2.5, 3.5]);
        let s = serde_json::to_string(&c).unwrap();
        let back: Coefficients = serde_json::from_str(&s).unwrap();
        assert_eq!(back.0, c.0);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn covariance_matrix_json_round_trip() {
        let m = CovarianceMatrix(ndarray::arr2(&[[1.0, 0.0], [0.0, 1.0]]));
        let s = serde_json::to_string(&m).unwrap();
        let back: CovarianceMatrix = serde_json::from_str(&s).unwrap();
        assert_eq!(back.0, m.0);
    }
}
