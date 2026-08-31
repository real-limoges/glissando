//! Type-safe newtype wrappers around the linear-algebra primitives the fitter and
//! solver lean on: `Coefficients`, `CovarianceMatrix`, `ModelMatrix`, …
//!
//! Each wrapper carries an `Array1`/`Array2` and provides `Deref`, so callers can
//! use it as if it were the bare ndarray type. `LogLambdas` additionally
//! implements the suite of `argmin-math` traits L-BFGS needs for the
//! smoothing-parameter optimization, defined once via the
//! `impl_argmin_math_for_vector_wrapper!` macro. `Coefficients` is never used
//! as an argmin `Param`/`Gradient` (only `LogLambdas` is), so it only gets the
//! `Deref`/`DerefMut` half via `impl_deref_for_vector_wrapper!`.

use argmin_math::ArgminScaledSub;
use argmin_math::{
    ArgminAdd, ArgminDot, ArgminL1Norm, ArgminL2Norm, ArgminMinMax, ArgminMul, ArgminScaledAdd,
    ArgminSignum, ArgminSub, ArgminZeroLike,
};
use ndarray::{Array1, Array2};
use std::ops::{Deref, DerefMut};

/// Regression coefficient vector. Derefs to `Array1<f64>`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct Coefficients(pub Array1<f64>);

/// Log-space smoothing parameters for L-BFGS optimization. Derefs to `Array1<f64>`.
#[derive(Clone, Debug)]
pub(crate) struct LogLambdas(pub Array1<f64>);

macro_rules! impl_argmin_math_for_vector_wrapper {
    ($t:ty) => {
        impl ArgminAdd<Self, Self> for $t {
            fn add(&self, other: &Self) -> Self {
                Self(&self.0 + &other.0)
            }
        }

        impl ArgminSub<Self, Self> for $t {
            fn sub(&self, other: &Self) -> Self {
                Self(&self.0 - &other.0)
            }
        }

        impl ArgminMul<f64, Self> for $t {
            fn mul(&self, scalar: &f64) -> Self {
                Self(&self.0 * *scalar)
            }
        }

        impl ArgminDot<Self, f64> for $t {
            fn dot(&self, other: &Self) -> f64 {
                self.0.dot(&other.0)
            }
        }

        impl ArgminL1Norm<f64> for $t {
            fn l1_norm(&self) -> f64 {
                self.0.mapv(|x| x.abs()).sum()
            }
        }

        impl ArgminL2Norm<f64> for $t {
            fn l2_norm(&self) -> f64 {
                self.0.mapv(|x| x * x).sum().sqrt()
            }
        }

        impl ArgminSignum for $t {
            fn signum(self) -> Self {
                Self(self.0.mapv(|x| x.signum()))
            }
        }

        impl ArgminMinMax for $t {
            fn min(x: &Self, y: &Self) -> Self {
                Self(
                    ndarray::Zip::from(&x.0)
                        .and(&y.0)
                        .map_collect(|a, b| a.min(*b)),
                )
            }

            fn max(x: &Self, y: &Self) -> Self {
                Self(
                    ndarray::Zip::from(&x.0)
                        .and(&y.0)
                        .map_collect(|a, b| a.max(*b)),
                )
            }
        }

        impl ArgminZeroLike for $t {
            fn zero_like(&self) -> Self {
                Self(Array1::zeros(self.0.len()))
            }
        }

        impl ArgminScaledAdd<Self, f64, Self> for $t {
            fn scaled_add(&self, alpha: &f64, y: &Self) -> Self {
                Self(&self.0 + &(y.0.mapv(|yi| yi * alpha)))
            }
        }

        impl ArgminScaledSub<Self, f64, Self> for $t {
            fn scaled_sub(&self, alpha: &f64, y: &Self) -> Self {
                Self(&self.0 - &(y.0.mapv(|yi| yi * alpha)))
            }
        }
        impl ArgminAdd<f64, $t> for $t {
            fn add(&self, scalar: &f64) -> $t {
                Self(self.0.mapv(|a| a + scalar))
            }
        }

        impl ArgminSub<f64, $t> for $t {
            fn sub(&self, scalar: &f64) -> $t {
                Self(self.0.mapv(|a| a - scalar))
            }
        }

        impl ArgminMul<Self, Self> for $t {
            fn mul(&self, other: &Self) -> Self {
                // ndarray's * operator on two arrays is element-wise
                Self(&self.0 * &other.0)
            }
        }
    };
}

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
impl_deref_for_vector_wrapper!(LogLambdas);
impl_argmin_math_for_vector_wrapper!(LogLambdas);

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

    // --- LogLambdas argmin-math impls ---
    //
    // Coefficients does not implement argmin-math traits: it is never used as an
    // argmin `Param`/`Gradient` (only `LogLambdas` is, for the L-BFGS
    // smoothing-parameter optimization), so it only gets Deref/DerefMut.

    #[test]
    fn loglambdas_argmin_add_sub() {
        let a = LogLambdas(array![1.0, 2.0, 3.0]);
        let b = LogLambdas(array![10.0, 20.0, 30.0]);
        let s = ArgminAdd::add(&a, &b);
        assert_eq!(s.0.to_vec(), vec![11.0, 22.0, 33.0]);
        let d = ArgminSub::sub(&b, &a);
        assert_eq!(d.0.to_vec(), vec![9.0, 18.0, 27.0]);
    }

    #[test]
    fn loglambdas_scalar_ops() {
        let a = LogLambdas(array![1.0, 2.0, 3.0]);
        let m: LogLambdas = ArgminMul::mul(&a, &2.0);
        assert_eq!(m.0.to_vec(), vec![2.0, 4.0, 6.0]);
        let plus: LogLambdas = ArgminAdd::add(&a, &10.0);
        assert_eq!(plus.0.to_vec(), vec![11.0, 12.0, 13.0]);
        let minus: LogLambdas = ArgminSub::sub(&a, &1.0);
        assert_eq!(minus.0.to_vec(), vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn loglambdas_dot_l1_l2() {
        let a = LogLambdas(array![3.0, 4.0]);
        let b = LogLambdas(array![1.0, 2.0]);
        assert_eq!(ArgminDot::dot(&a, &b), 11.0);
        assert_eq!(ArgminL1Norm::l1_norm(&LogLambdas(array![-3.0, 4.0])), 7.0);
        assert_eq!(ArgminL2Norm::l2_norm(&LogLambdas(array![3.0, 4.0])), 5.0);
    }

    #[test]
    fn loglambdas_signum_and_zero_like() {
        let a = LogLambdas(array![-1.0, -0.5, 2.0]);
        let s = ArgminSignum::signum(a.clone());
        assert_eq!(s.0.to_vec(), vec![-1.0, -1.0, 1.0]);
        let z = ArgminZeroLike::zero_like(&a);
        assert_eq!(z.0.to_vec(), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn loglambdas_minmax_elementwise() {
        let a = LogLambdas(array![1.0, 5.0, 3.0]);
        let b = LogLambdas(array![2.0, 4.0, 3.0]);
        let mn = ArgminMinMax::min(&a, &b);
        let mx = ArgminMinMax::max(&a, &b);
        assert_eq!(mn.0.to_vec(), vec![1.0, 4.0, 3.0]);
        assert_eq!(mx.0.to_vec(), vec![2.0, 5.0, 3.0]);
    }

    #[test]
    fn loglambdas_scaled_add_sub() {
        let a = LogLambdas(array![1.0, 2.0, 3.0]);
        let y = LogLambdas(array![10.0, 20.0, 30.0]);
        let s = ArgminScaledAdd::scaled_add(&a, &0.1, &y);
        assert_eq!(s.0.to_vec(), vec![2.0, 4.0, 6.0]);
        let sd = ArgminScaledSub::scaled_sub(&a, &0.1, &y);
        assert_eq!(sd.0.to_vec(), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn loglambdas_elementwise_mul() {
        let a = LogLambdas(array![1.0, 2.0, 3.0]);
        let b = LogLambdas(array![10.0, 20.0, 30.0]);
        let p: LogLambdas = ArgminMul::mul(&a, &b);
        assert_eq!(p.0.to_vec(), vec![10.0, 40.0, 90.0]);
    }

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
