#![allow(dead_code)]

use glissando::{DataSet, Formula, Smooth, Term};
use ndarray::Array1;
use rand::prelude::*;
use rand_distr::{Distribution, Normal, Poisson};

// ----------------------------------------------------------------------------
// Term and formula builders
//
// Most tests want one of a handful of formula shapes. These helpers cut the
// `Formula::new() + add_terms(...)` boilerplate down to a single line.
// ----------------------------------------------------------------------------

/// Default cubic P-spline (degree 3, second-order penalty) on `col`.
pub fn pspline(col: &str, n_splines: usize) -> Term {
    Term::Smooth(Smooth::PSpline1D {
        col_name: col.to_string(),
        n_splines,
        degree: 3,
        penalty_order: 2,
    })
}

/// P-spline with explicit degree and penalty order.
pub fn pspline_with(col: &str, n_splines: usize, degree: usize, penalty_order: usize) -> Term {
    Term::Smooth(Smooth::PSpline1D {
        col_name: col.to_string(),
        n_splines,
        degree,
        penalty_order,
    })
}

/// Linear effect on `col`.
pub fn linear(col: &str) -> Term {
    Term::Linear {
        col_name: col.to_string(),
    }
}

/// Random-effect term on `col`.
pub fn random(col: &str) -> Term {
    Term::Smooth(Smooth::RandomEffect {
        col_name: col.to_string(),
    })
}

/// Tensor product of two P-splines on `(col1, col2)`.
pub fn tensor(col1: &str, col2: &str, n1: usize, n2: usize) -> Term {
    Term::Smooth(Smooth::TensorProduct {
        col_name_1: col1.to_string(),
        n_splines_1: n1,
        penalty_order_1: 2,
        col_name_2: col2.to_string(),
        n_splines_2: n2,
        penalty_order_2: 2,
        degree: 3,
    })
}

/// Formula with `Intercept` for every named parameter.
pub fn intercept_only(params: &[&str]) -> Formula {
    let mut f = Formula::new();
    for p in params {
        f.add_terms((*p).to_string(), vec![Term::Intercept]);
    }
    f
}

/// Formula with `Intercept + Linear(col)` for the first parameter and `Intercept` for the rest.
pub fn linear_intercepts(col: &str, params: &[&str]) -> Formula {
    let mut f = Formula::new();
    let (head, rest) = params.split_first().expect("params must be non-empty");
    f.add_terms((*head).to_string(), vec![Term::Intercept, linear(col)]);
    for p in rest {
        f.add_terms((*p).to_string(), vec![Term::Intercept]);
    }
    f
}

/// Formula with a P-spline on `col` for the first parameter and `Intercept` for the rest.
pub fn smooth_intercepts(col: &str, n_splines: usize, params: &[&str]) -> Formula {
    let mut f = Formula::new();
    let (head, rest) = params.split_first().expect("params must be non-empty");
    f.add_terms((*head).to_string(), vec![pspline(col, n_splines)]);
    for p in rest {
        f.add_terms((*p).to_string(), vec![Term::Intercept]);
    }
    f
}

// ----------------------------------------------------------------------------
// Synthetic-data generators
// ----------------------------------------------------------------------------

pub struct Generator {
    pub rng: StdRng,
}

impl Generator {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Single-column dataset with `x` evenly spaced on `[0, 4]` and `y ~ Poisson(exp(a + b·x))`.
    pub fn poisson_data(&mut self, n: usize, intercept: f64, slope: f64) -> (Array1<f64>, DataSet) {
        let x: Vec<f64> = (0..n).map(|i| (i as f64 / n as f64) * 4.0).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&x_val| {
                let mu = (intercept + slope * x_val).exp();
                Poisson::new(mu).unwrap().sample(&mut self.rng)
            })
            .collect();

        let mut data = DataSet::new();
        data.insert_column("x", Array1::from_vec(x));
        (Array1::from_vec(y), data)
    }

    /// Heteroskedastic Gaussian: `y ~ N(10 + 2x, exp(-1 + 0.5x))` on `x ∈ [0, 3]`.
    pub fn heteroskedastic_gaussian(&mut self, n: usize) -> (Array1<f64>, DataSet) {
        let x: Vec<f64> = (0..n).map(|i| (i as f64 / n as f64) * 3.0).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&x_val| {
                let mu = 10.0 + 2.0 * x_val;
                let sigma = (-1.0 + 0.5 * x_val).exp();
                Normal::new(mu, sigma).unwrap().sample(&mut self.rng)
            })
            .collect();

        let mut data = DataSet::new();
        data.insert_column("x", Array1::from_vec(x));
        (Array1::from_vec(y), data)
    }

    /// 2D Gaussian-bump surface for tensor-product smooth tests.
    pub fn tensor_surface(&mut self, n: usize) -> (Array1<f64>, DataSet) {
        let mut x1 = Vec::with_capacity(n);
        let mut x2 = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);

        for _ in 0..n {
            let v1: f64 = self.rng.random();
            let v2: f64 = self.rng.random();
            let dist_sq = (v1 - 0.5).powi(2) + (v2 - 0.5).powi(2);
            let mu = (-dist_sq * 5.0).exp();
            let noise = self.rng.random_range(-0.1..0.1);

            x1.push(v1);
            x2.push(v2);
            y.push(mu + noise);
        }

        let mut data = DataSet::new();
        data.insert_column("x1", Array1::from_vec(x1));
        data.insert_column("x2", Array1::from_vec(x2));
        (Array1::from_vec(y), data)
    }

    /// Linear Gaussian with explicit intercept, slope, and noise scale: `y ~ N(a + b·x, σ)`.
    pub fn linear_gaussian(
        &mut self,
        n: usize,
        slope: f64,
        intercept: f64,
        sigma: f64,
    ) -> (Array1<f64>, DataSet) {
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&x_val| {
                let mu = intercept + slope * x_val;
                Normal::new(mu, sigma).unwrap().sample(&mut self.rng)
            })
            .collect();

        let mut data = DataSet::new();
        data.insert_column("x", Array1::from_vec(x));
        (Array1::from_vec(y), data)
    }
}

/// Sample one observation from `NB(mu, sigma)` via a Gamma-Poisson mixture, where
/// `r = 1/sigma` so `Var(Y) = mu + sigma·mu²`.
pub fn sample_negative_binomial(rng: &mut impl Rng, mu: f64, sigma: f64) -> f64 {
    let r = 1.0 / sigma;
    let lambda: f64 = rng.sample(rand_distr::Gamma::new(r, mu / r).unwrap());
    rng.sample(rand_distr::Poisson::new(lambda.max(1e-10)).unwrap())
}
