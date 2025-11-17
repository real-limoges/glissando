use argmin::solver::linesearch::MoreThuenteLineSearch;
// use argmin::solver::quasinewton::LBFGS;
use crate::Term;
use crate::families::{Link, Distribution, Gaussian};
use argmin::core::Gradient;
use super::{GamlssError, ModelMatrix, PenaltyMatrix, Coefficients, CovarianceMatrix, LogLambdas};
use ndarray::{Array1, Array2};
use ndarray_linalg::{Inverse, Solve, UPLO};
use argmin::core::{CostFunction, Error, Executor};
use polars::prelude::{DataFrame};

pub(crate) const PIRLS_TOLERANCE: f64 = 1e-6;

pub(crate) struct GamlssCost<'a, D: Distribution> {
    pub(crate) x_matrix: &'a ModelMatrix,
    pub(crate) z: &'a Array1<f64>,
    pub(crate) w: &'a Array1<f64>,
    pub(crate) penalty_matrices: &'a Vec<PenaltyMatrix>,
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
        ).map_err(|e| Error::new(e))?;

        let n = self.z.len() as f64;

        let fitted_z = self.x_model.dot(&beta);
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
        const H: f64 = 1.4901161193847656e-8; // This seemingly random choice is (f64::EPSILON.sqrt())
        let n = param.0.len();
        let mut grad_vec = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut param_plus_h_vec = param.0.clone();
            param_plus_h_vec += H;

            let mut param_minus_h_vec = param.0.clone();
            param_minus_h_vec -= H;

            let cost_plus = self.cost(&LogLambdas(param_plus_h_vec))?;
            let cost_minus = self.cost(&LogLambdas(param_minus_h_vec))?;

            grad_vec[i] = (cost_plus - cost_minus) / (2.0 * H);
        }
        Ok(LogLambdas(grad_vec))
    }
}

pub(crate) fn run_optimization<D>(
    x_model: &ModelMatrix,
    z: &Array1<f64>,
    w: &Array1<f64>,
    penalty_matrices: &Vec<PenaltyMatrix>
) -> Result<Array1<f64>, GamlssError> {
    let cost_function = GamlssCost {
        x_matrix: &x_model,
        z,
        w,
        penalty_matrices: &penalty_matrices
    };

    let initial_log_lambdas = LogLambdas(Array1::<f64>::zeros(penalty_matrices.len()));
    let solver = Lbfgs::new();

    let res = Executor::new(cost_function, solver)
        .configure(|state| state.param(initial_log_lambdas).max_iters(100))
        .run()?;

    let best_log_lambdas = res.state.best_param.unwrap();
    let best_lambdas = best_log_lambdas.mapv(f64::exp);

    Ok(best_lambdas)
}

pub(crate) fn fit_pwls(
    x_matrix: &ModelMatrix,
    z: &Array1<f64>,
    w_diag: &Array1<f64>,
    penalty_matrices: &[PenaltyMatrix],
    lambdas: &Array1<f64>
) -> Result<(Coefficients, CovarianceMatrix, f64), GamlssError> {
    // lifted some pirls code out of a numerical analysis book

    let (n_obs, n_coeffs) = x_matrix.dim();

    let mut s_lambda = Array2::<f64>::zeros((n_coeffs, n_coeffs));
    for (i, s_j) in penalty_matrices.iter().enumerate() {
        s_lambda.scaled_add(lambdas[i], s_j);
    }

    let w = Array2::<f64>::from_diag(&w_diag);

    let x_t_w = x_matrix.t().dot(&w);
    let lhs = x_t_w.dot(x_matrix.0) + &s_lambda;
    let rhs = x_t_w.dot(&z);

    let beta = Coefficients(lhs.solve(&rhs)?);

    let v_beta_unscaled = CovarianceMatrix(lhs.inv()?);

    let x_t_w_x = x_t_w.dot(&x_matrix.0);
    let edf = v_beta_unscaled.dot(&x_t_w_x).diag().sum();

    Ok((beta, v_beta_unscaled, edf))
}