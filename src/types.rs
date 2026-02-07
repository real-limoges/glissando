use argmin_math::ArgminScaledSub;
use argmin_math::{
    ArgminAdd, ArgminDot, ArgminL1Norm, ArgminL2Norm, ArgminMinMax, ArgminMul, ArgminScaledAdd,
    ArgminSignum, ArgminSub, ArgminZeroLike,
};
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use crate::terms::Term;

// ----- Newtypes for Safety (Vectors)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
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
impl_deref_for_matrix_wrapper!(PenaltyMatrix);
impl_deref_for_matrix_wrapper!(ModelMatrix);

// ----- Newtypes for Input Data and Formula

/// A dataset of named columns, wrapping `HashMap<String, Array1<f64>>`.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct DataSet(pub HashMap<String, Array1<f64>>);

impl DataSet {
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    pub fn column(&self, name: &str) -> Option<&Array1<f64>> {
        self.0.get(name)
    }

    pub fn n_obs(&self) -> Option<usize> {
        self.0.values().next().map(|v| v.len())
    }

    pub fn n_columns(&self) -> usize {
        self.0.len()
    }

    pub fn insert_column(&mut self, name: impl Into<String>, values: Array1<f64>) {
        self.0.insert(name.into(), values);
    }

    /// Create a DataSet from a HashMap of Vec<f64>, converting each to Array1<f64>.
    pub fn from_vecs(data: HashMap<String, Vec<f64>>) -> Self {
        let mut ds = Self::new();
        for (name, values) in data {
            ds.insert_column(name, Array1::from_vec(values));
        }
        ds
    }
}

impl Deref for DataSet {
    type Target = HashMap<String, Array1<f64>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for DataSet {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<HashMap<String, Array1<f64>>> for DataSet {
    fn from(map: HashMap<String, Array1<f64>>) -> Self {
        Self(map)
    }
}

/// A model formula mapping parameter names to term vectors,
/// wrapping `HashMap<String, Vec<Term>>`.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct Formula(pub HashMap<String, Vec<Term>>);

impl Formula {
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    pub fn with_terms(mut self, param: impl Into<String>, terms: Vec<Term>) -> Self {
        self.0.insert(param.into(), terms);
        self
    }

    pub fn add_terms(&mut self, param: impl Into<String>, terms: Vec<Term>) {
        self.0.insert(param.into(), terms);
    }

    pub fn param_names(&self) -> Vec<&String> {
        self.0.keys().collect()
    }
}

impl Deref for Formula {
    type Target = HashMap<String, Vec<Term>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Formula {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<HashMap<String, Vec<Term>>> for Formula {
    fn from(map: HashMap<String, Vec<Term>>) -> Self {
        Self(map)
    }
}
