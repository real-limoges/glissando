use super::{Coefficients, CovarianceMatrix};
use ndarray::{Array1, Array2};
use ndarray_linalg::{Cholesky, UPLO};
use rand::rng;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

pub fn sample_posterior(
    beta_hat: &Coefficients,
    v_beta: &CovarianceMatrix,
    n_samples: usize,
) -> Vec<Array1<f64>> {
    let l_factor = match v_beta.0.cholesky(UPLO::Lower) {
        Ok(cholesky) => cholesky,
        Err(_) => return vec![],
    };

    let mut rng_rs = rng();

    sample_from_cholesky(&beta_hat.0, &l_factor, n_samples, &mut rng_rs)
}

pub(crate) fn sample_from_cholesky(
    mean: &Array1<f64>,
    l_factor: &Array2<f64>,
    n_samples: usize,
    rng: &mut impl Rng,
) -> Vec<Array1<f64>> {
    let dim = mean.len();

    (0..n_samples)
        .map(|_| {
            let z = Array1::<f64>::from_shape_fn(dim, |_| StandardNormal.sample(rng));
            mean + l_factor.dot(&z)
        })
        .collect()
}
