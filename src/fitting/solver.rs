//! Penalized weighted least squares (PWLS) solver and GCV smoothing parameter optimization.
//!
//! Uses Cholesky decomposition for solving the PWLS system and L-BFGS (via argmin)
//! for optimizing smoothing parameters (lambda) by minimizing the GCV score.

use super::{Coefficients, CovarianceMatrix, GamlssError, LogLambdas, ModelMatrix, PenaltyMatrix};
use crate::linalg;
use argmin::core::Gradient;
use argmin::core::{CostFunction, Error, Executor};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use ndarray::prelude::*;

/// Minimum denominator value to prevent division by zero in GCV computation
const MIN_DENOMINATOR: f64 = 1e-10;

/// Minimum lambda value for log-space conversion
const MIN_LAMBDA: f64 = 1e-10;

/// Relative tolerance for declaring a penalty eigenvalue numerically zero.
/// τ = REML_RANK_TOL_EPS · max(eigval); below this the direction is treated as
/// part of the penalty null space when computing the pseudo-determinant.
const REML_RANK_TOL_EPS: f64 = 1e-8;

/// Clamp on |log λ| applied to REML's optimized output before exponentiation.
/// Wide enough that no real solution is constrained; narrow enough to keep
/// L-BFGS from wandering into numerically pathological regions.
const LOG_LAMBDA_CLAMP: f64 = 30.0;

/// Max Fellner-Schall iterations before returning the current λ.
const FS_MAX_ITERS: usize = 50;

/// Fellner-Schall convergence threshold on max |log λ_j_new − log λ_j_old|.
const FS_TOL: f64 = 1e-3;

/// Relative floor on the Fellner-Schall multiplicative-update numerator
/// `tr(S_λ⁺ S_j) − tr(V S_j)`. Wood-Fasiolo §3 discusses pathological signs
/// (the numerator should be non-negative under the smoothness prior); a small
/// relative floor keeps the update bounded when it dips.
const FS_NUMERATOR_FLOOR_REL: f64 = 1e-12;

/// Floor on the Fellner-Schall denominator `β̂ᵀ S_j β̂`, which is near-zero
/// when β̂ sits in the null space of S_j. Prevents division-by-zero.
const FS_DENOMINATOR_FLOOR: f64 = 1e-12;

/// Result from PWLS fitting that includes gradient computation info.
struct PwlsGradientInfo {
    beta: Coefficients,
    v_matrix: Array2<f64>,
    edf: f64,
    x_t_w_x: Array2<f64>,
    x_t_w_r: Array1<f64>,
    rss: f64,
}

pub(crate) struct GamlssCost<'a> {
    pub(crate) x_matrix: &'a ModelMatrix,
    pub(crate) z: &'a Array1<f64>,
    pub(crate) w: &'a Array1<f64>,
    pub(crate) penalty_matrices: &'a [PenaltyMatrix],
}

impl<'a> CostFunction for GamlssCost<'a> {
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
        if denominator.abs() < MIN_DENOMINATOR {
            return Ok(f64::MAX);
        }
        let gcv_score = (n * rss) / denominator;

        Ok(gcv_score)
    }
}

impl<'a> Gradient for GamlssCost<'a> {
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

        if denom.abs() < MIN_DENOMINATOR {
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

/// Laplace-Approximate Marginal Likelihood (LAML / REML) cost for smoothing-
/// parameter selection (Wood 2011), applied to a single distributional
/// parameter's converged PWLS subproblem (working response `z`, weights `w`,
/// scale φ = 1).
///
/// Minimizes `−V_r(λ)` where
///   V_r = ℓ(β̂) − ½·β̂ᵀS_λβ̂ + ½·log|S_λ|_+ − ½·log|H + S_λ| + (M_p/2)·log(2π)
/// with H = X'WX, ℓ = −½·RSS (the Gaussian working log-likelihood; constants
/// in z, w independent of λ drop out of the optimization).
pub(crate) struct RemlCost<'a> {
    pub(crate) x_matrix: &'a ModelMatrix,
    pub(crate) z: &'a Array1<f64>,
    pub(crate) w: &'a Array1<f64>,
    pub(crate) penalty_matrices: &'a [PenaltyMatrix],
}

impl<'a> CostFunction for RemlCost<'a> {
    type Param = LogLambdas;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let lambdas = param.mapv(f64::exp);

        let info = fit_pwls_with_grad_info(
            self.x_matrix,
            self.z,
            self.w,
            self.penalty_matrices,
            &lambdas,
        )
        .map_err(Error::new)?;

        let s_lambda = weighted_penalty_sum(&lambdas, self.penalty_matrices);
        let eig = penalty_eigen(&s_lambda, REML_RANK_TOL_EPS).map_err(Error::new)?;

        let lhs = &info.x_t_w_x + &s_lambda;
        let log_det_lhs = linalg::log_det_via_cholesky(&lhs).map_err(Error::new)?;

        let beta_s_beta = info.beta.0.dot(&s_lambda.dot(&info.beta.0));
        let m_p = eig.null_dim as f64;

        // V_r = ℓ − ½·βᵀS_λβ + ½·log|S_λ|_+ − ½·log|H+S_λ| + (M_p/2)·log(2π)
        // ℓ_partial = −½·RSS (constants in the working-likelihood independent of λ cancel).
        let v_r = -0.5 * info.rss
            - 0.5 * beta_s_beta
            + 0.5 * eig.log_pdet
            - 0.5 * log_det_lhs
            + 0.5 * m_p * (2.0 * std::f64::consts::PI).ln();

        Ok(-v_r)
    }
}

impl<'a> Gradient for RemlCost<'a> {
    type Param = LogLambdas;
    type Gradient = LogLambdas;

    /// Analytic gradient of −V_r with respect to ρ = log λ.
    ///
    /// ∂V_r/∂ρ_j = −(λ_j/2)·β̂ᵀS_jβ̂ + (λ_j/2)·tr(S_λ⁺ S_j) − (λ_j/2)·tr(V S_j)
    ///
    /// V, S_λ⁺, and S_j are symmetric, so each trace reduces to a Hadamard-sum.
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

        let s_lambda = weighted_penalty_sum(&lambdas, self.penalty_matrices);
        let eig = penalty_eigen(&s_lambda, REML_RANK_TOL_EPS).map_err(Error::new)?;

        let mut grad = Array1::<f64>::zeros(n_penalties);
        for j in 0..n_penalties {
            let s_j = &self.penalty_matrices[j].0;
            let bsb = s_j.dot(&info.beta.0).dot(&info.beta.0);
            let tr_v_s = (&info.v_matrix * s_j).sum();
            let tr_pinv_s = (&eig.pinv * s_j).sum();
            // ∂V_r/∂ρ_j; minimize −V_r so negate.
            let dvr = 0.5 * lambdas[j] * (-bsb + tr_pinv_s - tr_v_s);
            grad[j] = -dvr;
        }

        Ok(LogLambdas(grad))
    }
}

/// Runs L-BFGS optimization to find optimal smoothing parameters (lambdas).
///
/// Uses warm-starting from previous lambdas when available for faster convergence.
/// Skips optimization entirely when there are no penalty matrices.
pub(crate) fn run_optimization(
    x_model: &ModelMatrix,
    z: &Array1<f64>,
    w: &Array1<f64>,
    penalty_matrices: &[PenaltyMatrix],
    initial_lambdas: Option<&Array1<f64>>,
) -> Result<Array1<f64>, GamlssError> {
    let n_penalties = penalty_matrices.len();

    // Fast path: no penalties means no smoothing parameters to optimize
    if n_penalties == 0 {
        return Ok(Array1::zeros(0));
    }

    let cost_function = GamlssCost {
        x_matrix: x_model,
        z,
        w,
        penalty_matrices,
    };

    // Warm-start from previous lambdas (in log-space) if available
    let initial_log_lambdas = match initial_lambdas {
        Some(prev) if prev.len() == n_penalties => {
            LogLambdas(prev.mapv(|l| l.max(MIN_LAMBDA).ln()))
        }
        _ => LogLambdas(Array1::<f64>::zeros(n_penalties)),
    };

    // Check if starting point is already good enough.
    // Intentional fallback: if cost computation fails (e.g., singular matrix at initial lambdas),
    // treat it as worst-case so we proceed with optimization rather than aborting.
    let initial_cost = cost_function.cost(&initial_log_lambdas).unwrap_or(f64::MAX);

    // If we have a warm start and the cost is very low, skip optimization
    if initial_lambdas.is_some() && initial_cost < 1e-6 {
        return Ok(initial_log_lambdas.mapv(f64::exp));
    }

    let linesearch = MoreThuenteLineSearch::new();
    let solver = LBFGS::new(linesearch, 7);

    let res = Executor::new(cost_function, solver)
        .configure(|state| {
            state
                .param(initial_log_lambdas)
                .max_iters(50)
                .target_cost(MIN_DENOMINATOR) // Early exit if GCV is very small
        })
        .run()?;

    let best_log_lambdas = res.state.best_param.ok_or_else(|| {
        GamlssError::Optimization("Optimizer failed to find best parameters".to_string())
    })?;
    let best_lambdas = best_log_lambdas.mapv(f64::exp);

    Ok(best_lambdas)
}

/// REML/LAML analogue of `run_optimization`.
///
/// Differences vs the GCV path:
/// - cold start uses `initial_log_lambda` (mgcv-style heuristic) instead of zeros;
/// - no `target_cost` early exit — `−V_r` is not bounded below by zero;
/// - returned log λ is clamped to `[-LOG_LAMBDA_CLAMP, LOG_LAMBDA_CLAMP]` before
///   exponentiation, so a runaway L-BFGS step cannot produce non-finite λ.
pub(crate) fn run_optimization_reml(
    x_model: &ModelMatrix,
    z: &Array1<f64>,
    w: &Array1<f64>,
    penalty_matrices: &[PenaltyMatrix],
    initial_lambdas: Option<&Array1<f64>>,
) -> Result<Array1<f64>, GamlssError> {
    let n_penalties = penalty_matrices.len();
    if n_penalties == 0 {
        return Ok(Array1::zeros(0));
    }

    let cost_function = RemlCost {
        x_matrix: x_model,
        z,
        w,
        penalty_matrices,
    };

    let initial_log_lambdas = match initial_lambdas {
        Some(prev) if prev.len() == n_penalties => {
            LogLambdas(prev.mapv(|l| {
                l.max(MIN_LAMBDA)
                    .ln()
                    .clamp(-LOG_LAMBDA_CLAMP, LOG_LAMBDA_CLAMP)
            }))
        }
        _ => {
            // Cold start: build X'WX once to seed the heuristic.
            let sqrt_w = w.mapv(f64::sqrt);
            let x_weighted = &x_model.0 * &sqrt_w.view().insert_axis(Axis(1));
            let x_t_w_x = x_weighted.t().dot(&x_weighted);
            LogLambdas(initial_log_lambda(&x_t_w_x, penalty_matrices))
        }
    };

    let linesearch = MoreThuenteLineSearch::new();
    let solver = LBFGS::new(linesearch, 7);

    let res = Executor::new(cost_function, solver)
        .configure(|state| state.param(initial_log_lambdas).max_iters(50))
        .run()?;

    let best_log_lambdas = res.state.best_param.ok_or_else(|| {
        GamlssError::Optimization("REML optimizer failed to find best parameters".to_string())
    })?;
    let clamped = best_log_lambdas.mapv(|l| l.clamp(-LOG_LAMBDA_CLAMP, LOG_LAMBDA_CLAMP));
    Ok(clamped.mapv(f64::exp))
}

/// Fellner-Schall (Wood & Fasiolo 2017) multiplicative fixed-point optimizer
/// for the same LAML target that `run_optimization_reml` minimizes.
///
/// Per-iteration update for each penalty `j`:
///
/// ```text
///                   tr(S_λ⁺ S_j) − tr(V S_j)
///   λ_j  ←  λ_j ·  ─────────────────────────
///                          β̂ᵀ S_j β̂
/// ```
///
/// where `V = (X'WX + S_λ)⁻¹` and `β̂` come from a PIRLS solve at the current
/// `λ`. Wood & Fasiolo prove monotone improvement of the LAML score under mild
/// regularity for any quadratically penalized smooth log-likelihood.
///
/// Compared to `run_optimization_reml`:
/// - no outer L-BFGS, no line search → deterministic across linalg backends;
/// - first-order convergence (slower asymptotically) but no Hessian / no
///   step-size tuning;
/// - shares every helper with the REML path (`fit_pwls_with_grad_info`,
///   `weighted_penalty_sum`, `penalty_eigen`).
pub(crate) fn run_optimization_fellner_schall(
    x_model: &ModelMatrix,
    z: &Array1<f64>,
    w: &Array1<f64>,
    penalty_matrices: &[PenaltyMatrix],
    initial_lambdas: Option<&Array1<f64>>,
) -> Result<Array1<f64>, GamlssError> {
    let n_penalties = penalty_matrices.len();
    if n_penalties == 0 {
        return Ok(Array1::zeros(0));
    }

    // Initialize λ: warm-start from previous, else mgcv-style heuristic.
    let mut lambdas: Array1<f64> = match initial_lambdas {
        Some(prev) if prev.len() == n_penalties => prev.mapv(|l| l.max(MIN_LAMBDA)),
        _ => {
            let sqrt_w = w.mapv(f64::sqrt);
            let x_weighted = &x_model.0 * &sqrt_w.view().insert_axis(Axis(1));
            let x_t_w_x = x_weighted.t().dot(&x_weighted);
            initial_log_lambda(&x_t_w_x, penalty_matrices).mapv(f64::exp)
        }
    };

    let lambda_ceiling = LOG_LAMBDA_CLAMP.exp();

    for _iter in 0..FS_MAX_ITERS {
        let info = fit_pwls_with_grad_info(x_model, z, w, penalty_matrices, &lambdas)?;
        let s_lambda = weighted_penalty_sum(&lambdas, penalty_matrices);
        let eig = penalty_eigen(&s_lambda, REML_RANK_TOL_EPS)?;

        let mut max_log_change: f64 = 0.0;
        let mut new_lambdas = lambdas.clone();

        for j in 0..n_penalties {
            let s_j = &penalty_matrices[j].0;
            let tr_pinv_s = (&eig.pinv * s_j).sum();
            let tr_v_s = (&info.v_matrix * s_j).sum();
            let bsb = s_j.dot(&info.beta.0).dot(&info.beta.0);

            let numerator = (tr_pinv_s - tr_v_s).max(FS_NUMERATOR_FLOOR_REL * lambdas[j]);
            let denominator = bsb.max(FS_DENOMINATOR_FLOOR);
            let lambda_new = (lambdas[j] * (numerator / denominator))
                .max(MIN_LAMBDA)
                .min(lambda_ceiling);

            let log_change = (lambda_new.ln() - lambdas[j].ln()).abs();
            if log_change > max_log_change {
                max_log_change = log_change;
            }
            new_lambdas[j] = lambda_new;
        }

        lambdas = new_lambdas;
        if max_log_change < FS_TOL {
            break;
        }
    }

    Ok(lambdas)
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

    let beta_arr = linalg::solve(&lhs, &x_t_w_z)?;
    let beta = Coefficients(beta_arr);

    let v = linalg::inv(&lhs)?;

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

/// Forms `S_λ = Σ_j λ_j · S_j`.
fn weighted_penalty_sum(lambdas: &Array1<f64>, penalty_matrices: &[PenaltyMatrix]) -> Array2<f64> {
    debug_assert!(!penalty_matrices.is_empty());
    let p = penalty_matrices[0].0.nrows();
    let mut s = Array2::<f64>::zeros((p, p));
    for (i, s_j) in penalty_matrices.iter().enumerate() {
        s.scaled_add(lambdas[i], &s_j.0);
    }
    s
}

/// Eigendecomposition products of S_λ that REML needs:
/// the log pseudo-determinant log|S_λ|_+, the null-space dimension M_p,
/// and the Moore-Penrose pseudo-inverse S_λ⁺ used in the gradient term tr(S_λ⁺ S_j).
struct PenaltyEigen {
    log_pdet: f64,
    null_dim: usize,
    pinv: Array2<f64>,
}

/// Symmetric eigendecomposition of `S_λ` with rank-aware accumulation of
/// log|S_λ|_+ and construction of the pseudo-inverse.
///
/// τ = `eps · max(|eigval|, 1)`; eigenvalues at or below τ contribute to the
/// null-space dimension and are zeroed in the pseudo-inverse.
fn penalty_eigen(s_lambda: &Array2<f64>, eps: f64) -> Result<PenaltyEigen, GamlssError> {
    let (eigvals, eigvecs) = linalg::symmetric_eigh(s_lambda)?;

    let d_max = eigvals
        .iter()
        .cloned()
        .fold(0.0_f64, |acc, x| acc.max(x.abs()));
    let tau = eps * d_max.max(1.0);

    let n = eigvals.len();
    let mut log_pdet = 0.0;
    let mut null_dim = 0usize;
    let mut d_inv = Array1::<f64>::zeros(n);
    for (i, &di) in eigvals.iter().enumerate() {
        if di > tau {
            log_pdet += di.ln();
            d_inv[i] = 1.0 / di;
        } else {
            null_dim += 1;
        }
    }

    // pinv = Q · diag(d_inv) · Qᵀ. Multiply each column j of Q by d_inv[j],
    // then post-multiply by Qᵀ.
    let q_scaled = &eigvecs * &d_inv.view().insert_axis(Axis(0));
    let pinv = q_scaled.dot(&eigvecs.t());

    Ok(PenaltyEigen {
        log_pdet,
        null_dim,
        pinv,
    })
}

/// Cold-start heuristic for log λ when no warm start is available.
///
/// `λ_j^(0) = tr(X'WX) / max(tr(S_j), MIN_LAMBDA)`, in log space and clamped to
/// `[-LOG_LAMBDA_CLAMP, LOG_LAMBDA_CLAMP]`. This is mgcv's `initial.sp` heuristic
/// in spirit — pick λ so each smooth lands in the interior of the EDF range,
/// not pinned at 0 or p.
fn initial_log_lambda(
    x_t_w_x: &Array2<f64>,
    penalty_matrices: &[PenaltyMatrix],
) -> Array1<f64> {
    let tr_xwx = x_t_w_x.diag().sum().max(MIN_LAMBDA);
    Array1::from_iter(penalty_matrices.iter().map(|s_j| {
        let tr_sj = s_j.0.diag().sum().max(MIN_LAMBDA);
        (tr_xwx / tr_sj)
            .ln()
            .clamp(-LOG_LAMBDA_CLAMP, LOG_LAMBDA_CLAMP)
    }))
}

#[cfg(test)]
mod reml_tests {
    use super::*;
    use crate::splines::create_penalty_matrix;

    #[test]
    fn penalty_eigen_rank_deficient_pspline() {
        // Order-2 difference penalty on basis size 10 has null space of dim 2
        // (constants and lines).
        let p = create_penalty_matrix(10, 2);
        let eig = penalty_eigen(&p, REML_RANK_TOL_EPS).unwrap();
        assert_eq!(eig.null_dim, 2, "second-order P-spline penalty should have null dim 2");
        assert!(eig.log_pdet.is_finite());

        // Symmetric pseudo-inverse.
        for i in 0..10 {
            for j in (i + 1)..10 {
                assert!(
                    (eig.pinv[[i, j]] - eig.pinv[[j, i]]).abs() < 1e-10,
                    "pinv should be symmetric at ({},{})", i, j
                );
            }
        }

        // P · pinv · P ≈ P (Moore-Penrose property A·A⁺·A = A).
        let recon = p.dot(&eig.pinv).dot(&p);
        for i in 0..10 {
            for j in 0..10 {
                assert!(
                    (recon[[i, j]] - p[[i, j]]).abs() < 1e-8,
                    "P·P⁺·P mismatch at ({},{}): got {}, want {}",
                    i, j, recon[[i, j]], p[[i, j]]
                );
            }
        }
    }

    #[test]
    fn weighted_penalty_sum_combines_linearly() {
        let p1 = PenaltyMatrix(arr2(&[[1.0_f64, 0.0], [0.0, 1.0]]));
        let p2 = PenaltyMatrix(arr2(&[[2.0_f64, 1.0], [1.0, 2.0]]));
        let lambdas = arr1(&[0.5_f64, 3.0]);
        let s = weighted_penalty_sum(&lambdas, &[p1, p2]);
        // 0.5·I + 3·[[2,1],[1,2]] = [[6.5, 3], [3, 6.5]]
        assert!((s[[0, 0]] - 6.5).abs() < 1e-12);
        assert!((s[[0, 1]] - 3.0).abs() < 1e-12);
        assert!((s[[1, 0]] - 3.0).abs() < 1e-12);
        assert!((s[[1, 1]] - 6.5).abs() < 1e-12);
    }

    #[test]
    fn initial_log_lambda_in_clamp_bounds() {
        // tr(X'WX) = 100, tr(S_j) = 4 → log(25) ≈ 3.22
        let xwx = arr2(&[[50.0_f64, 0.0], [0.0, 50.0]]);
        let p = PenaltyMatrix(arr2(&[[2.0_f64, 0.0], [0.0, 2.0]]));
        let init = initial_log_lambda(&xwx, &[p]);
        assert_eq!(init.len(), 1);
        assert!(init[0].abs() <= LOG_LAMBDA_CLAMP);
        assert!((init[0] - 25.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn initial_log_lambda_clamps_extremes() {
        // tr(S_j) tiny → λ would be huge → clamp.
        let xwx = arr2(&[[1.0_f64, 0.0], [0.0, 1.0]]);
        let p = PenaltyMatrix(arr2(&[[1e-30, 0.0], [0.0, 1e-30]]));
        let init = initial_log_lambda(&xwx, &[p]);
        assert!(init[0] <= LOG_LAMBDA_CLAMP);
    }

    /// Build a small synthetic P-spline problem for REML cost/gradient tests.
    /// Uses a real centered B-spline basis with a matching order-2 difference
    /// penalty so the objective surface is well-conditioned (the penalty
    /// genuinely measures wiggliness in the basis).
    fn synthetic_pwls_problem() -> (ModelMatrix, Array1<f64>, Array1<f64>, Vec<PenaltyMatrix>) {
        use crate::splines::create_basis_matrix;
        let n = 40;
        let n_splines = 8;
        let x_coord = Array1::from_iter((0..n).map(|i| i as f64 / (n as f64 - 1.0)));
        let basis = create_basis_matrix(&x_coord, n_splines, 3); // cubic B-splines
        let penalty = create_penalty_matrix(n_splines, 2);
        let z = x_coord.mapv(|t| (2.0 * std::f64::consts::PI * t).sin());
        let w = Array1::from_elem(n, 1.0_f64);
        (
            ModelMatrix(basis),
            z,
            w,
            vec![PenaltyMatrix(penalty)],
        )
    }

    #[test]
    fn reml_cost_finite_on_simple_problem() {
        let (x, z, w, ps) = synthetic_pwls_problem();
        let cost = RemlCost {
            x_matrix: &x,
            z: &z,
            w: &w,
            penalty_matrices: &ps,
        };
        // Evaluate at several log-λ to catch obvious blow-ups.
        for log_lambda in [-5.0, -1.0, 0.0, 1.0, 5.0] {
            let val = cost.cost(&LogLambdas(arr1(&[log_lambda]))).unwrap();
            assert!(val.is_finite(), "REML cost not finite at log λ = {}: got {}", log_lambda, val);
        }
    }

    /// Critical correctness gate for Fellner-Schall: capture λ at each F-S
    /// iteration and verify the LAML score (−V_r, as evaluated by `RemlCost::cost`)
    /// is monotonically non-increasing along the trajectory. Wood & Fasiolo 2017
    /// prove this property; if our implementation breaks it, the bug lives in
    /// the update formula or the helpers it consumes.
    #[test]
    fn fellner_schall_monotone_improves_laml() {
        let (x, z, w, ps) = synthetic_pwls_problem();
        let cost = RemlCost {
            x_matrix: &x,
            z: &z,
            w: &w,
            penalty_matrices: &ps,
        };

        // Reproduce the F-S loop here so we can snapshot λ at each step
        // (run_optimization_fellner_schall doesn't expose intermediates).
        let n_penalties = ps.len();
        let sqrt_w = w.mapv(f64::sqrt);
        let x_weighted = &x.0 * &sqrt_w.view().insert_axis(Axis(1));
        let x_t_w_x = x_weighted.t().dot(&x_weighted);
        let mut lambdas = initial_log_lambda(&x_t_w_x, &ps).mapv(f64::exp);

        let lambda_ceiling = LOG_LAMBDA_CLAMP.exp();
        let mut scores: Vec<f64> = Vec::new();
        for _iter in 0..20 {
            let log_lambdas = LogLambdas(lambdas.mapv(f64::ln));
            let score = cost.cost(&log_lambdas).unwrap();
            scores.push(score);

            let info = fit_pwls_with_grad_info(&x, &z, &w, &ps, &lambdas).unwrap();
            let s_lambda = weighted_penalty_sum(&lambdas, &ps);
            let eig = penalty_eigen(&s_lambda, REML_RANK_TOL_EPS).unwrap();
            let mut new_lambdas = lambdas.clone();
            for j in 0..n_penalties {
                let s_j = &ps[j].0;
                let tr_pinv_s = (&eig.pinv * s_j).sum();
                let tr_v_s = (&info.v_matrix * s_j).sum();
                let bsb = s_j.dot(&info.beta.0).dot(&info.beta.0);
                let numerator = (tr_pinv_s - tr_v_s).max(FS_NUMERATOR_FLOOR_REL * lambdas[j]);
                let denominator = bsb.max(FS_DENOMINATOR_FLOOR);
                new_lambdas[j] = (lambdas[j] * (numerator / denominator))
                    .max(MIN_LAMBDA)
                    .min(lambda_ceiling);
            }
            lambdas = new_lambdas;
        }

        // −V_r should be non-increasing along the F-S trajectory.
        // Allow a tiny numerical slack (1e-6) for floating-point round-off in
        // the eigendecomposition.
        for window in scores.windows(2) {
            let (prev, next) = (window[0], window[1]);
            assert!(
                next <= prev + 1e-6,
                "F-S violated LAML monotonicity: score went {} → {}",
                prev, next
            );
        }
    }

    #[test]
    fn reml_gradient_matches_finite_diff() {
        // Critical correctness gate: analytic ∂(−V_r)/∂ρ_j must match central diff.
        let (x, z, w, ps) = synthetic_pwls_problem();
        let cost = RemlCost {
            x_matrix: &x,
            z: &z,
            w: &w,
            penalty_matrices: &ps,
        };

        let h = 1e-5;
        // Single-penalty case here; loop kept so the test extends easily.
        for &log_lambda in &[-2.0_f64, 0.0, 2.0] {
            let p0 = LogLambdas(arr1(&[log_lambda]));
            let analytic = cost.gradient(&p0).unwrap();

            let mut p_plus = p0.clone();
            p_plus.0[0] += h;
            let mut p_minus = p0.clone();
            p_minus.0[0] -= h;
            let fd = (cost.cost(&p_plus).unwrap() - cost.cost(&p_minus).unwrap()) / (2.0 * h);

            let rel_err = (analytic.0[0] - fd).abs() / fd.abs().max(1e-6);
            assert!(
                rel_err < 1e-4,
                "gradient mismatch at log λ = {}: analytic = {}, fd = {}, rel_err = {}",
                log_lambda, analytic.0[0], fd, rel_err
            );
        }
    }
}
