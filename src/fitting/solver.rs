use super::{Coefficients, CovarianceMatrix, GamlssError, LogLambdas, ModelMatrix, PenaltyMatrix};
use crate::Term;
use crate::distributions;
use crate::distributions::{Distribution, Gaussian, Link};
use argmin::core::Gradient;
use argmin::core::{CostFunction, Error, Executor};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use ndarray::prelude::*;
use ndarray_linalg::{Inverse, Solve};
use polars::prelude::DataFrame;
use std::marker::PhantomData;

const H: f64 = 1.49e-8;
pub(crate) const PIRLS_TOLERANCE: f64 = 1e-6;

pub(crate) struct GamlssCost<'a, D: Distribution> {
    pub(crate) x_matrix: &'a ModelMatrix,
    pub(crate) z: &'a Array1<f64>,
    pub(crate) w: &'a Array1<f64>,
    pub(crate) penalty_matrices: &'a Vec<PenaltyMatrix>,

    pub _marker: PhantomData<D>,
}

impl<'a, D: Distribution> CostFunction for GamlssCost<'a, D> {
    type Param = LogLambdas;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let lambdas = param.mapv(f64::exp);

        let (beta, _, edf) = fit_pwls(
            self.x_matrix,
            self.z,
            self.w,
            self.penalty_matrices,
            &lambdas,
        )
        .map_err(|e| Error::new(e))?;

        let n = self.z.len() as f64;

        let fitted_z = self.x_matrix.0.dot(&beta.0);
        let residuals_z = self.z - &fitted_z;
        let rss = (&residuals_z * &residuals_z * self.w).sum();

        let denominator = (n - edf).powi(2);
        if denominator.abs() < 1e-10 {
            return Ok(f64::MAX);
        }
        let gcv_score = (n * rss) / denominator;

        Ok(gcv_score)
    }
}

impl<'a, D: Distribution> Gradient for GamlssCost<'a, D> {
    type Param = LogLambdas;
    type Gradient = LogLambdas;
    fn gradient(&self, param: &Self::Param) -> Result<Self::Param, Error> {
        let n = param.0.len();
        let mut grad_vec = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut param_plus = param.0.clone();
            param_plus[i] += H;

            let mut param_minus = param.0.clone();
            param_minus[i] -= H;

            let cost_plus = self.cost(&LogLambdas(param_plus))?;
            let cost_minus = self.cost(&LogLambdas(param_minus))?;

            grad_vec[i] = (cost_plus - cost_minus) / (2.0 * H);
        }

        Ok(LogLambdas(grad_vec))
    }
}

pub(crate) fn run_optimization<D: Distribution>(
    x_model: &ModelMatrix,
    z: &Array1<f64>,
    w: &Array1<f64>,
    penalty_matrices: &Vec<PenaltyMatrix>,
) -> Result<Array1<f64>, GamlssError> {
    let cost_function: GamlssCost<'_, D> = GamlssCost::<D> {
        x_matrix: x_model,
        z,
        w,
        penalty_matrices,
        _marker: PhantomData,
    };

    let initial_log_lambdas = LogLambdas(Array1::<f64>::zeros(penalty_matrices.len()));
    let linesearch = MoreThuenteLineSearch::new();
    let solver = LBFGS::new(linesearch, 7);

    let res = Executor::new(cost_function, solver)
        .configure(|state| state.param(initial_log_lambdas).max_iters(100))
        .run()?;

    let best_log_lambdas = res.state.best_param.ok_or_else(|| {
        GamlssError::Optimization("Optimizer failed to find best parameters".to_string())
    })?;
    let best_lambdas = best_log_lambdas.mapv(f64::exp);

    Ok(best_lambdas)
}

pub(crate) fn fit_pwls(
    x_matrix: &ModelMatrix,
    z: &Array1<f64>,
    w_diag: &Array1<f64>,
    penalty_matrices: &[PenaltyMatrix],
    lambdas: &Array1<f64>,
) -> Result<(Coefficients, CovarianceMatrix, f64), GamlssError> {
    // lifted some pirls code out of a numerical analysis book

    let x = &x_matrix.0;

    let (_n_obs, n_coeffs) = x.dim();

    let mut s_lambda = Array2::<f64>::zeros((n_coeffs, n_coeffs));
    for (i, s_j) in penalty_matrices.iter().enumerate() {
        s_lambda.scaled_add(lambdas[i], &s_j.0);
    }

    // Use sqrt-weighted approach to avoid creating n×n diagonal matrix.
    // X'WX = (√W·X)'(√W·X) and X'Wz = (√W·X)'(√W·z)
    // This reduces memory from O(n²) to O(n·p).
    let sqrt_w = w_diag.mapv(f64::sqrt);

    // Scale each row i of X by sqrt_w[i]
    let x_weighted = x * &sqrt_w.view().insert_axis(Axis(1));
    let z_weighted = z * &sqrt_w;

    // X'WX and X'Wz without the n×n matrix
    let x_t_w_x = x_weighted.t().dot(&x_weighted);
    let x_t_w_z = x_weighted.t().dot(&z_weighted);

    let lhs = &x_t_w_x + &s_lambda;
    let rhs = x_t_w_z;

    let beta_arr = lhs.solve(&rhs).map_err(GamlssError::Linalg);
    let beta = Coefficients(beta_arr?);

    let inv_lhs = lhs.inv().map_err(GamlssError::Linalg)?;

    let v_beta_unscaled = CovarianceMatrix(inv_lhs);

    let edf = v_beta_unscaled.0.dot(&x_t_w_x).diag().sum();

    Ok((beta, v_beta_unscaled, edf))
}
