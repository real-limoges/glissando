use super::{Coefficients, CovarianceMatrix, GamlssError, LogLambdas, ModelMatrix, PenaltyMatrix};
use crate::distributions::Distribution;
use argmin::core::Gradient;
use argmin::core::{CostFunction, Error, Executor};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use ndarray::prelude::*;
use ndarray_linalg::{Inverse, Solve};
use std::marker::PhantomData;

/// Result from PWLS fitting that includes gradient computation info.
struct PwlsGradientInfo {
    beta: Coefficients,
    v_matrix: Array2<f64>,
    edf: f64,
    x_t_w_x: Array2<f64>,
    x_t_w_r: Array1<f64>,
    rss: f64,
}

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

    /// Computes Generalized Cross-Validation (GCV) score for smoothing parameter selection.
    ///
    /// GCV approximates leave-one-out CV without refitting n times:
    ///   GCV(λ) = n * RSS / (n - EDF)²
    ///
    /// where RSS is weighted residual sum of squares and EDF is effective degrees of freedom.
    /// Minimizing GCV balances fit (low RSS) against complexity (high EDF).
    /// We optimize in log-space (log λ) for numerical stability and unconstrained optimization.
    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let lambdas = param.mapv(f64::exp);

        let (beta, _, edf) = fit_pwls(
            self.x_matrix,
            self.z,
            self.w,
            self.penalty_matrices,
            &lambdas,
        )
        .map_err(Error::new)?;

        let n = self.z.len() as f64;

        let fitted_z = self.x_matrix.0.dot(&beta.0);
        let residuals_z = self.z - &fitted_z;
        let rss = (&residuals_z * &residuals_z * self.w).sum();

        // Guard against division by zero when EDF approaches n (overfit)
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

    /// Computes the gradient of GCV with respect to log(lambda) for quasi-Newton optimization.
    ///
    /// The key insight is that beta depends on lambda through the penalized normal equations.
    /// See docs/mathematics.md for the full derivation of dRSS/dlambda and dEDF/dlambda.
    fn gradient(&self, param: &Self::Param) -> Result<Self::Param, Error> {
        let lambdas = param.mapv(f64::exp);
        let n_penalties = lambdas.len();

        if n_penalties == 0 {
            return Ok(LogLambdas(Array1::zeros(0)));
        }

        let info = fit_pwls_with_grad_info(
            self.x_matrix,
            self.z,
            self.w,
            self.penalty_matrices,
            &lambdas,
        )
        .map_err(Error::new)?;

        let n = self.z.len() as f64;
        let denom = n - info.edf;

        if denom.abs() < 1e-10 {
            return Ok(LogLambdas(Array1::zeros(n_penalties)));
        }

        let mut grad_vec = Array1::zeros(n_penalties);

        for j in 0..n_penalties {
            let s_j = &self.penalty_matrices[j].0;

            // dRSS/dlambda_j = 2 * (X'Wr)' * V * Sj * beta
            let v_sj_beta = info.v_matrix.dot(&s_j.dot(&info.beta.0));
            let d_rss = 2.0 * info.x_t_w_r.dot(&v_sj_beta);

            // dEDF/dlambda_j = -tr(V * Sj * V * X'WX)
            let v_sj = info.v_matrix.dot(s_j);
            let v_sj_v = v_sj.dot(&info.v_matrix);
            let d_edf = -v_sj_v.dot(&info.x_t_w_x).diag().sum();

            // Quotient rule: dGCV/dlambda_j = n * [dRSS*(n-EDF) + 2*RSS*dEDF] / (n-EDF)^3
            let d_gcv = n * (d_rss * denom + 2.0 * info.rss * d_edf) / denom.powi(3);

            // Chain rule for log-space: d/d(log lambda) = lambda * d/dlambda
            grad_vec[j] = lambdas[j] * d_gcv;
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

/// Solves the penalized weighted least squares problem:
///   minimize  (z - X*beta)'W(z - X*beta) + sum_j lambda_j * beta'*S_j*beta
///
/// The solution satisfies the penalized normal equations:
///   (X'WX + sum_j lambda_j*S_j) * beta = X'Wz
///
/// Returns coefficients beta, covariance matrix V = (X'WX + sum lambda*S)^-1, and
/// effective degrees of freedom EDF = tr(V * X'WX).
pub(crate) fn fit_pwls(
    x_matrix: &ModelMatrix,
    z: &Array1<f64>,
    w_diag: &Array1<f64>,
    penalty_matrices: &[PenaltyMatrix],
    lambdas: &Array1<f64>,
) -> Result<(Coefficients, CovarianceMatrix, f64), GamlssError> {
    let info = fit_pwls_with_grad_info(x_matrix, z, w_diag, penalty_matrices, lambdas)?;
    Ok((info.beta, CovarianceMatrix(info.v_matrix), info.edf))
}

fn fit_pwls_with_grad_info(
    x_matrix: &ModelMatrix,
    z: &Array1<f64>,
    w_diag: &Array1<f64>,
    penalty_matrices: &[PenaltyMatrix],
    lambdas: &Array1<f64>,
) -> Result<PwlsGradientInfo, GamlssError> {
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

    let beta_arr = lhs.solve(&x_t_w_z).map_err(GamlssError::Linalg)?;
    let beta = Coefficients(beta_arr);

    let v = lhs.inv().map_err(GamlssError::Linalg)?;

    // EDF (effective degrees of freedom) measures model complexity.
    // EDF = tr(H) where H = X(X'WX + sum lambda*S)^-1 X'W is the hat matrix.
    // Equivalently, EDF = tr(V * X'WX). Ranges from 0 (lambda->inf) to p (lambda->0).
    let edf = v.dot(&x_t_w_x).diag().sum();

    let fitted = x.dot(&beta.0);
    let residuals = z - &fitted;
    let rss = (&residuals * &residuals * w_diag).sum();

    // X'Wr = (√W·X)' * (√W·r) - needed for gradient computation
    let x_t_w_r = x_weighted.t().dot(&(&residuals * &sqrt_w));

    Ok(PwlsGradientInfo {
        beta,
        v_matrix: v,
        edf,
        x_t_w_x,
        x_t_w_r,
        rss,
    })
}
