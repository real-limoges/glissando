use argmin_math::ArgminScaledSub;
use argmin_math::{
    ArgminAdd, ArgminDot, ArgminL1Norm, ArgminL2Norm, ArgminMinMax, ArgminMul, ArgminScaledAdd,
    ArgminSignum, ArgminSub, ArgminZeroLike,
};
use ndarray::{Array1, Array2};
use std::ops::{Deref, DerefMut};

// ----- Newtypes for Safety (Vectors)
#[derive(Debug, Clone)]
pub struct Coefficients(pub Array1<f64>);

#[derive(Clone, Debug)]
pub struct LogLambdas(pub Array1<f64>);

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

impl_argmin_math_for_vector_wrapper!(Coefficients);
impl_argmin_math_for_vector_wrapper!(LogLambdas);

// ----- Newtypes for Safety (Matrices)
#[derive(Debug, Clone)]
pub struct ModelMatrix(pub Array2<f64>);

#[derive(Debug, Clone)]
pub struct PenaltyMatrix(pub Array2<f64>);

#[derive(Debug, Clone)]
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
impl_deref_for_matrix_wrapper!(PenaltyMatrix);
impl_deref_for_matrix_wrapper!(ModelMatrix);
