//! Penalized weighted least squares (PWLS) solver and GCV smoothing parameter optimization.
//!
//! Uses Cholesky decomposition for solving the PWLS system and L-BFGS (via argmin)
//! for optimizing smoothing parameters (lambda) by minimizing the GCV score.

use super::{
    Coefficients, CovarianceMatrix, GamlssError, LogLambdas, ModelMatrix, PenaltyMatrix,
    SmoothingCriterion,
};
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
pub(super) const LOG_LAMBDA_CLAMP: f64 = 30.0;

/// Decades (in natural-log λ) below the cold-start heuristic to seed the
/// collapse-guarded restart. The cold start sits *past* the interior LAML
/// optimum on the slope toward the flat high-λ shelf; subtracting this offset
/// lands the restart safely below both the optimum and the shelf, so a
/// gradient/fixed-point optimizer descends into the (unimodal) interior optimum
/// instead of staying pinned to the penalty null space.
const RESTART_LOG_OFFSET: f64 = 8.0;

/// Max Fellner-Schall iterations before returning the current λ.
const FS_MAX_ITERS: usize = 50;

/// Fellner-Schall convergence threshold on max |log λ_j_new − log λ_j_old|.
///
/// The F-S update is first-order convergent, so stopping at 1e-3 (≈ 0.1% relative
/// change in λ) leaves more drift than stopping at 1e-4 (≈ 0.01%).  A tighter
/// threshold costs at most a few extra iterations — F-S is cheap — but ensures λ
/// is genuinely stationary before the outer RS loop declares convergence.
const FS_TOL: f64 = 1e-4;

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
    /// Per-coefficient EDF contributions: `diag(V·X'WX)`. Summing a contiguous
    /// block attributes effective degrees of freedom to an individual term.
    edf_per_coeff: Array1<f64>,
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

        let (beta, _, edf, _) = fit_pwls(
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
    /// See docs/math/mathematics.md for the full derivation of dRSS/dlambda and dEDF/dlambda.
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
        let eig = penalty_eigen(self.penalty_matrices, &lambdas, REML_RANK_TOL_EPS)
            .map_err(Error::new)?;

        let lhs = &info.x_t_w_x + &s_lambda;
        let log_det_lhs = linalg::log_det_robust(&lhs).map_err(Error::new)?;

        let beta_s_beta = info.beta.0.dot(&s_lambda.dot(&info.beta.0));
        let m_p = eig.null_dim as f64;

        // V_r = ℓ − ½·βᵀS_λβ + ½·log|S_λ|_+ − ½·log|H+S_λ| + (M_p/2)·log(2π)
        // ℓ_partial = −½·RSS (constants in the working-likelihood independent of λ cancel).
        let v_r = -0.5 * info.rss - 0.5 * beta_s_beta + 0.5 * eig.log_pdet - 0.5 * log_det_lhs
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

        let eig = penalty_eigen(self.penalty_matrices, &lambdas, REML_RANK_TOL_EPS)
            .map_err(Error::new)?;

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

    let linesearch = MoreThuenteLineSearch::new();
    let solver = LBFGS::new(linesearch, 7);

    let res = Executor::new(cost_function, solver)
        .configure(|state| {
            // `target_cost` was previously set to MIN_DENOMINATOR, which would
            // prematurely stop L-BFGS whenever RSS is tiny (e.g. a log-scale sigma
            // parameter with a near-perfect working model) — the near-zero GCV cost
            // was declared "optimal" and λ was frozen for the remainder of fitting.
            // Removed here; L-BFGS converges in O(1) extra iterations from a good
            // warm start, so the savings were negligible and the bias was harmful.
            state.param(initial_log_lambdas).max_iters(50)
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
        Some(prev) if prev.len() == n_penalties => LogLambdas(prev.mapv(|l| {
            l.max(MIN_LAMBDA)
                .ln()
                .clamp(-LOG_LAMBDA_CLAMP, LOG_LAMBDA_CLAMP)
        })),
        _ => LogLambdas(initial_log_lambda(x_model, penalty_matrices)),
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
    let lbfgs_lambdas = clamped.mapv(f64::exp);

    // Deterministic Fellner-Schall polish. L-BFGS + MoreThuente can stall at a
    // warm-start-dependent, non-stationary point when the LAML surface has flat
    // ridges (e.g. several smooths collapsing to their null space with λ at the
    // clamp ceiling); the resulting per-cycle λ jitter keeps the outer RS loop
    // from ever seeing a stationary η. F-S iterates the same LAML target
    // monotonically and lands on the same fixed point from either side of a
    // ridge, making the per-cycle λ map deterministic. Keep whichever λ scores
    // better so the polish can never make the fit worse.
    // Best-effort: a linear-algebra failure inside the polish (e.g. an
    // eigensolver hiccup at a degenerate λ) falls back to the L-BFGS result
    // rather than failing the whole fit.
    let polished =
        match run_optimization_fellner_schall(x_model, z, w, penalty_matrices, Some(&lbfgs_lambdas))
        {
            Ok(p) => p,
            Err(_) => return Ok(lbfgs_lambdas),
        };
    let lbfgs_cost = lambda_cost(
        SmoothingCriterion::Reml,
        x_model,
        z,
        w,
        penalty_matrices,
        &lbfgs_lambdas,
    );
    let polished_cost = lambda_cost(
        SmoothingCriterion::Reml,
        x_model,
        z,
        w,
        penalty_matrices,
        &polished,
    );
    match (lbfgs_cost, polished_cost) {
        (Ok(lc), Ok(pc)) if pc <= lc => Ok(polished),
        _ => Ok(lbfgs_lambdas),
    }
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
        _ => initial_log_lambda(x_model, penalty_matrices).mapv(f64::exp),
    };

    let lambda_ceiling = LOG_LAMBDA_CLAMP.exp();

    for _iter in 0..FS_MAX_ITERS {
        let info = fit_pwls_with_grad_info(x_model, z, w, penalty_matrices, &lambdas)?;
        let eig = penalty_eigen(penalty_matrices, &lambdas, REML_RANK_TOL_EPS)?;

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
) -> Result<(Coefficients, CovarianceMatrix, f64, Array1<f64>), GamlssError> {
    let info = fit_pwls_with_grad_info(x_matrix, z, w_diag, penalty_matrices, lambdas)?;
    Ok((
        info.beta,
        CovarianceMatrix(info.v_matrix),
        info.edf,
        info.edf_per_coeff,
    ))
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
    // Keep the per-coefficient diagonal so callers can attribute EDF per term.
    let edf_per_coeff = v.dot(&x_t_w_x).diag().to_owned();
    let edf = edf_per_coeff.sum();

    let fitted = x.dot(&beta.0);
    let residuals = z - &fitted;
    let rss = (&residuals * &residuals * w_diag).sum();

    // X'Wr = (√W·X)' * (√W·r) - needed for gradient computation
    let x_t_w_r = x_weighted.t().dot(&(&residuals * &sqrt_w));

    Ok(PwlsGradientInfo {
        beta,
        v_matrix: v,
        edf,
        edf_per_coeff,
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

/// Compute REML eigendecomposition quantities, grouping penalties by their coefficient block.
///
/// The naive approach eigendecomposes S_λ = Σ_j λ_j S_j once with a single relative
/// threshold τ = eps · max(eigenvalue of S_λ).  When λ values span many orders of
/// magnitude (e.g. λ₁ ≈ 1e-8 and λ₄ ≈ 5e10 in a 5-smooth model), the threshold is
/// dominated by the large-λ term and misclassifies the non-null directions of small-λ
/// penalties as null space — producing the wrong REML gradient sign.
///
/// Fix: group penalty matrices by their non-zero block range and eigendecompose each
/// group independently.  Within a group the combined scaled block is formed first,
/// then eigendecomposed.  This gives two correctness guarantees simultaneously:
///
/// - **Per-group threshold**: τ_g = eps · max(eigenvalue of Σ_{j∈g} λ_j S_j_block)
///   is relative to the group's own eigenvalue scale, not polluted by other groups.
/// - **Overlapping supports**: penalties in the same coefficient block (e.g. the two
///   marginal penalties of a tensor-product smooth, which both act on the same k₁k₂
///   coefficients) are combined before eigendecomposition, so
///   `(λ₁S₁+λ₂S₂)⁺ ≠ (λ₁S₁)⁺ + (λ₂S₂)⁺` is never violated.
///
/// For the common all-disjoint case (each smooth has its own coefficient block), every
/// group is a single penalty and the formula reduces to the per-penalty decomposition.
///
/// ```text
/// log|S_λ|_+  = Σ_g Σ_{i: dᵢ > τ_g} ln(dᵢ)   where dᵢ = eigenvalue of Σ_{j∈g} λⱼSⱼ
/// S_λ⁺        = block-diag{ (Σ_{j∈g} λⱼSⱼ)⁺ }  (exact when groups have disjoint support)
/// M_p         = Σ_g null_dim(Σ_{j∈g} λⱼSⱼ_block)
/// ```
fn penalty_eigen(
    penalty_matrices: &[PenaltyMatrix],
    lambdas: &Array1<f64>,
    eps: f64,
) -> Result<PenaltyEigen, GamlssError> {
    debug_assert!(!penalty_matrices.is_empty());
    let p = penalty_matrices[0].0.nrows();

    let mut log_pdet = 0.0_f64;
    let mut null_dim = 0_usize;
    let mut pinv = Array2::<f64>::zeros((p, p));

    // Find the non-zero block range for each penalty.
    let ranges: Vec<(usize, usize)> = penalty_matrices
        .iter()
        .map(|s_j| penalty_nonzero_block_range(&s_j.0))
        .collect();

    // Process each unique block range exactly once, combining all penalties in that block.
    let mut processed = vec![false; penalty_matrices.len()];

    for first in 0..penalty_matrices.len() {
        if processed[first] {
            continue;
        }
        let (start, end) = ranges[first];
        let block_size = end - start + 1;

        // Collect indices of all penalties sharing this exact block range.
        let group: Vec<usize> = (first..penalty_matrices.len())
            .filter(|&k| ranges[k] == (start, end))
            .collect();
        for &k in &group {
            processed[k] = true;
        }

        // Build the combined scaled block: Σ_{j ∈ group} λ_j · S_j_block.
        // For a single-penalty group this is just λ_j · S_j_block; for a tensor-product
        // group this is the anisotropic penalty λ₁(S_x₁⊗I) + λ₂(I⊗S_x₂) restricted to
        // the shared k₁k₂ coefficient block.
        let mut combined = Array2::<f64>::zeros((block_size, block_size));
        for &k in &group {
            let slice = penalty_matrices[k].0.slice(s![start..=end, start..=end]);
            combined.scaled_add(lambdas[k], &slice);
        }

        let (eigvals, eigvecs) = linalg::symmetric_eigh(&combined)?;

        // Per-group relative threshold: relative to the combined block's spectral norm.
        let d_max = eigvals.iter().cloned().fold(0.0_f64, |a, x| a.max(x));
        let tau = (eps * d_max).max(1e-300);

        let mut d_inv = Array1::<f64>::zeros(block_size);
        for i in 0..block_size {
            let di = eigvals[i];
            if di > tau {
                log_pdet += di.ln();
                d_inv[i] = 1.0 / di;
            } else {
                null_dim += 1;
            }
        }

        // pinv[start..=end, start..=end] += Q · diag(d_inv) · Qᵀ
        let q_scaled = &eigvecs * &d_inv.view().insert_axis(Axis(0));
        let mut sub = pinv.slice_mut(s![start..=end, start..=end]);
        sub += &q_scaled.dot(&eigvecs.t());
    }

    Ok(PenaltyEigen {
        log_pdet,
        null_dim,
        pinv,
    })
}

/// Returns the `[start, end]` index range of the contiguous non-zero block in a symmetric
/// penalty matrix that was embedded into the full coefficient space by the assembler.
///
/// The embedder writes exact zeros outside the block, so any non-zero row marks the boundary.
/// Falls back to `(0, nrows-1)` if the matrix is entirely zero (degenerate).
fn penalty_nonzero_block_range(s: &Array2<f64>) -> (usize, usize) {
    let n = s.nrows();
    let start = (0..n)
        .find(|&i| s.row(i).iter().any(|&x| x != 0.0))
        .unwrap_or(0);
    let end = (0..n)
        .rev()
        .find(|&i| s.row(i).iter().any(|&x| x != 0.0))
        .unwrap_or(n - 1);
    (start, end)
}

/// Cold-start heuristic for log λ when no warm start is available.
///
/// Uses `tr(X'X) / tr(S_j)` (unweighted) rather than `tr(X'WX) / tr(S_j)`.
/// The unweighted form is scale-invariant: for a B-spline basis the column norms
/// depend only on the knot layout, not on the response scale, so the initial λ
/// stays in a numerically friendly range regardless of how large σ is.
/// `tr(X'WX) = tr(X'X) / σ²` for homoscedastic Gaussian, which makes the
/// weighted form tiny (≈ 10⁻²¹) for price-scale data (σ ≈ 45k) and forces
/// L-BFGS to start near the unpenalized OLS solution where the REML landscape is
/// badly conditioned — reliably causing the smooth to overshoot into full collapse.
pub(super) fn initial_log_lambda(
    x_matrix: &ModelMatrix,
    penalty_matrices: &[PenaltyMatrix],
) -> Array1<f64> {
    let x_t_x = x_matrix.0.t().dot(&x_matrix.0);
    let tr_xtx = x_t_x.diag().sum().max(MIN_LAMBDA);
    Array1::from_iter(penalty_matrices.iter().map(|s_j| {
        let tr_sj = s_j.0.diag().sum().max(MIN_LAMBDA);
        (tr_xtx / tr_sj)
            .ln()
            .clamp(-LOG_LAMBDA_CLAMP, LOG_LAMBDA_CLAMP)
    }))
}

/// Low-λ seed for the collapse-guarded restart (see [`RESTART_LOG_OFFSET`]).
///
/// Derived from the scale-aware cold-start heuristic so it adapts to the basis /
/// response scale, then shifted several decades down to sit below the high-λ
/// collapse shelf.
pub(super) fn restart_seed(
    x_matrix: &ModelMatrix,
    penalty_matrices: &[PenaltyMatrix],
) -> Array1<f64> {
    initial_log_lambda(x_matrix, penalty_matrices)
        .mapv(|log_lambda| (log_lambda - RESTART_LOG_OFFSET).exp().max(MIN_LAMBDA))
}

/// Value of the objective the given criterion minimizes, evaluated at a fixed λ.
///
/// Used by the collapse-guarded restart in [`super::scoring::step`] to compare a
/// restart's λ against the incumbent and keep the lower-objective fit — so the
/// guard can never make a fit worse, and genuinely null-space-optimal data (a
/// linear truth under an order-2 penalty) correctly *keeps* its collapsed fit
/// because that fit has the better marginal likelihood.
pub(super) fn lambda_cost(
    criterion: SmoothingCriterion,
    x_matrix: &ModelMatrix,
    z: &Array1<f64>,
    w: &Array1<f64>,
    penalty_matrices: &[PenaltyMatrix],
    lambdas: &Array1<f64>,
) -> Result<f64, GamlssError> {
    let log_lambdas = LogLambdas(lambdas.mapv(|l| l.max(MIN_LAMBDA).ln()));
    let map_err = |e: Error| GamlssError::Optimization(e.to_string());
    match criterion {
        SmoothingCriterion::Gcv => GamlssCost {
            x_matrix,
            z,
            w,
            penalty_matrices,
        }
        .cost(&log_lambdas)
        .map_err(map_err),
        // REML and Fellner-Schall minimize the same LAML target (−V_r).
        SmoothingCriterion::Reml | SmoothingCriterion::FellnerSchall => RemlCost {
            x_matrix,
            z,
            w,
            penalty_matrices,
        }
        .cost(&log_lambdas)
        .map_err(map_err),
    }
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
        let pm = PenaltyMatrix(p.clone());
        let lambdas = arr1(&[1.0_f64]);
        let eig = penalty_eigen(&[pm], &lambdas, REML_RANK_TOL_EPS).unwrap();
        assert_eq!(
            eig.null_dim, 2,
            "second-order P-spline penalty should have null dim 2"
        );
        assert!(eig.log_pdet.is_finite());

        // Symmetric pseudo-inverse.
        for i in 0..10 {
            for j in (i + 1)..10 {
                assert!(
                    (eig.pinv[[i, j]] - eig.pinv[[j, i]]).abs() < 1e-10,
                    "pinv should be symmetric at ({},{})",
                    i,
                    j
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
                    i,
                    j,
                    recon[[i, j]],
                    p[[i, j]]
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
        // 50×2 matrix of ones → X'X = [[50,50],[50,50]], tr(X'X) = 100.
        // tr(X'X) = 100, tr(S_j) = 4 → log(25) ≈ 3.22.
        use ndarray::Array2;
        let x = ModelMatrix(Array2::ones((50, 2)));
        let p = PenaltyMatrix(arr2(&[[2.0_f64, 0.0], [0.0, 2.0]]));
        let init = initial_log_lambda(&x, &[p]);
        assert_eq!(init.len(), 1);
        assert!(init[0].abs() <= LOG_LAMBDA_CLAMP);
        assert!((init[0] - 25.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn initial_log_lambda_clamps_extremes() {
        // tr(S_j) near-zero → λ would be huge → clamp at upper bound.
        use ndarray::Array2;
        let x = ModelMatrix(Array2::ones((50, 2)));
        let p = PenaltyMatrix(arr2(&[[1e-30, 0.0], [0.0, 1e-30]]));
        let init = initial_log_lambda(&x, &[p]);
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
        (ModelMatrix(basis), z, w, vec![PenaltyMatrix(penalty)])
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
            assert!(
                val.is_finite(),
                "REML cost not finite at log λ = {}: got {}",
                log_lambda,
                val
            );
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
        let mut lambdas = initial_log_lambda(&x, &ps).mapv(f64::exp);

        let lambda_ceiling = LOG_LAMBDA_CLAMP.exp();
        let mut scores: Vec<f64> = Vec::new();
        for _iter in 0..20 {
            let log_lambdas = LogLambdas(lambdas.mapv(f64::ln));
            let score = cost.cost(&log_lambdas).unwrap();
            scores.push(score);

            let info = fit_pwls_with_grad_info(&x, &z, &w, &ps, &lambdas).unwrap();
            let eig = penalty_eigen(&ps, &lambdas, REML_RANK_TOL_EPS).unwrap();
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
                prev,
                next
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
                log_lambda,
                analytic.0[0],
                fd,
                rel_err
            );
        }
    }

    #[test]
    fn gcv_gradient_matches_finite_diff() {
        // Analytic ∂GCV/∂ρ_j must match the central-difference approximation.
        // Mirrors `reml_gradient_matches_finite_diff` for the GCV path — the GCV
        // gradient (quotient-rule + dEDF/dλ) was previously untested.
        let (x, z, w, ps) = synthetic_pwls_problem();
        let cost = GamlssCost {
            x_matrix: &x,
            z: &z,
            w: &w,
            penalty_matrices: &ps,
        };

        let h = 1e-5;
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
                "GCV gradient mismatch at log λ = {}: analytic = {}, fd = {}, rel_err = {}",
                log_lambda,
                analytic.0[0],
                fd,
                rel_err
            );
        }
    }

    /// DIAGNOSTIC (Part 1, Q2 of the bistability investigation — `#[ignore]`d).
    ///
    /// Reconstructs the exact μ-subproblem of `mu_smooth_recovers_nonlinear_mean_control`
    /// and grids the REML/LAML objective `−V_r(λ)` over log λ. For a Gaussian with
    /// the identity link the IRLS working response is `z = y` exactly and the
    /// working weight is constant `w = 1/σ̂²` (σ̂ ≈ 0.2 here), so this *is* the
    /// landscape the outer loop's λ optimizer sees at convergence.
    ///
    /// Prints, for each grid point: λ, edf, and −V_r. The decision gate: is the
    /// collapsed region (edf → null-space ≈ 3, counting the unpenalized intercept)
    /// a *spurious local* optimum (interior λ has strictly lower −V_r) or the
    /// *global* optimum (REML genuinely prefers the line)?
    #[test]
    #[ignore = "diagnostic: prints the LAML-vs-logλ landscape for the bistable control case"]
    fn diagnostic_laml_landscape_control_case() {
        use crate::fitting::assembler::{assemble_model_matrices, resolve_terms};
        use crate::terms::{Smooth, Term};
        use crate::types::DataSet;
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};

        // --- Reproduce the control-case data verbatim (seed 7, n = 8000). ---
        let n = 8_000usize;
        let mut rng = StdRng::seed_from_u64(7);
        let true_curve = |x: f64| -0.7 + 0.8 * (2.0 * std::f64::consts::PI * x).sin();
        let x_vals: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
        let normal = Normal::new(0.0, 0.2).unwrap();
        let y_vals: Vec<f64> = x_vals
            .iter()
            .map(|&x| true_curve(x) + normal.sample(&mut rng))
            .collect();

        let mut data = DataSet::new();
        data.insert_column("x", Array1::from_vec(x_vals.clone()));

        // --- Assemble the real design + penalty (intercept + sum-to-zero P-spline). ---
        let terms = resolve_terms(
            &[
                Term::Intercept,
                Term::Smooth(Smooth::PSpline1D {
                    col_name: "x".to_string(),
                    n_splines: 15,
                    degree: 3,
                    penalty_order: 2,
                    range: None,
                }),
            ],
            &data,
        )
        .unwrap();
        let (x_model, penalties, _total, layouts) =
            assemble_model_matrices(&data, n, &terms).unwrap();

        // Gaussian identity link: z = y exactly; w = 1/σ̂² constant (σ̂ ≈ 0.2 → w ≈ 25).
        let z = Array1::from_vec(y_vals.clone());
        let w = Array1::from_elem(n, 1.0 / (0.2 * 0.2));

        let null_dim: usize = layouts
            .iter()
            .filter(|l| l.is_smooth)
            .map(|l| l.null_dim)
            .sum();
        eprintln!(
            "smooth null-space dim = {null_dim} (collapse ⇒ edf ≈ {} incl. intercept)",
            null_dim + 1
        );
        eprintln!("{:>10}  {:>10}  {:>16}", "log λ", "edf", "-V_r (LAML)");

        let cost = RemlCost {
            x_matrix: &x_model,
            z: &z,
            w: &w,
            penalty_matrices: &penalties,
        };

        let mut best = (f64::INFINITY, 0.0_f64, 0.0_f64); // (-V_r, log λ, edf)
        let mut rows: Vec<(f64, f64, f64)> = Vec::new();
        let mut log_lambda = -6.0_f64;
        while log_lambda <= 34.0 + 1e-9 {
            let lambdas = arr1(&[log_lambda.exp()]);
            let (_b, _v, edf, _e) = fit_pwls(&x_model, &z, &w, &penalties, &lambdas).unwrap();
            let neg_vr = cost.cost(&LogLambdas(arr1(&[log_lambda]))).unwrap();
            eprintln!("{log_lambda:>10.3}  {edf:>10.3}  {neg_vr:>16.4}");
            rows.push((log_lambda, edf, neg_vr));
            if neg_vr < best.0 {
                best = (neg_vr, log_lambda, edf);
            }
            log_lambda += 1.0;
        }

        eprintln!(
            "\nGLOBAL min of -V_r on grid: log λ = {:.3}  (λ = {:.4e}),  edf = {:.3},  -V_r = {:.4}",
            best.1,
            best.1.exp(),
            best.2,
            best.0
        );
        let collapsed = rows.last().unwrap();
        eprintln!(
            "COLLAPSE end (log λ = {:.1}): edf = {:.3}, -V_r = {:.4}  →  {} the global optimum",
            collapsed.0,
            collapsed.1,
            collapsed.2,
            if (collapsed.2 - best.0).abs() < 1e-6 {
                "IS"
            } else {
                "is NOT"
            }
        );
        eprintln!(
            "Interior vs collapse: global-min edf = {:.3} ⇒ {}",
            best.2,
            if best.2 > null_dim as f64 + 1.5 {
                "REML prefers an INTERIOR (curved) λ — collapse is a SPURIOUS LOCAL optimum (fixable)"
            } else {
                "REML's global optimum IS near-collapse — NOT an optimizer bug"
            }
        );
    }
}

#[cfg(test)]
mod diag_tensor_tests {
    use super::*;
    use crate::fitting::assembler::{assemble_model_matrices, resolve_terms};
    use crate::terms::{Smooth, Term};
    use crate::types::DataSet;

    #[test]
    #[ignore = "TEMP diagnostic"]
    fn diag_tensor13_lambda_surface() {
        let txt = std::fs::read_to_string("/tmp/claude-0/-home-user-glissando/f7210943-0cc0-539b-9bee-32d76d5afd3f/scratchpad/tensor13.csv").unwrap();
        let (mut yv, mut x1, mut x2) = (vec![], vec![], vec![]);
        for (i, line) in txt.lines().enumerate() {
            if i == 0 { continue; }
            let f: Vec<f64> = line.split(',').map(|v| v.parse().unwrap()).collect();
            yv.push(f[0]); x1.push(f[1]); x2.push(f[2]);
        }
        let n = yv.len();
        let mut data = DataSet::new();
        data.insert_column("x1", Array1::from_vec(x1));
        data.insert_column("x2", Array1::from_vec(x2));
        let terms = resolve_terms(&[Term::Smooth(Smooth::TensorProduct {
            col_name_1: "x1".into(), n_splines_1: 8, penalty_order_1: 2,
            col_name_2: "x2".into(), n_splines_2: 8, penalty_order_2: 2, degree: 3,
            range_1: None, range_2: None })], &data).unwrap();
        let (xm, ps, _, _) = assemble_model_matrices(&data, n, &terms).unwrap();
        let z = Array1::from_vec(yv);
        let sigma_hat = 0.2905_f64; // from the converged fit
        let w = Array1::from_elem(n, 1.0 / (sigma_hat * sigma_hat));

        let cost_at = |l1: f64, l2: f64| -> f64 {
            lambda_cost(SmoothingCriterion::Reml, &xm, &z, &w, &ps, &arr1(&[l1, l2])).unwrap()
        };
        eprintln!("corner (1e-10, 1.07e13): {:.4}", cost_at(1e-10, 1.068e13));
        for &l2 in &[1.0, 10.0, 100.0, 300.0, 1000.0] {
            eprintln!("(1e-10, {l2:>7.0}): {:.4}", cost_at(1e-10, l2));
        }
        // What do the optimizers do from cold start / restart seed?
        let cold = initial_log_lambda(&xm, &ps);
        eprintln!("cold-start log lambdas: {:?}", cold.to_vec());
        let from_cold = run_optimization_reml(&xm, &z, &w, &ps, None).unwrap();
        eprintln!("L-BFGS from cold: {:?} cost {:.4}", from_cold.to_vec(), cost_at(from_cold[0], from_cold[1]));
        let seed = restart_seed(&xm, &ps);
        eprintln!("restart seed: {:?}", seed.to_vec());
        let from_restart = run_optimization_reml(&xm, &z, &w, &ps, Some(&seed)).unwrap();
        eprintln!("L-BFGS from restart: {:?} cost {:.4}", from_restart.to_vec(), cost_at(from_restart[0], from_restart[1]));
        let fs = run_optimization_fellner_schall(&xm, &z, &w, &ps, Some(&seed)).unwrap();
        eprintln!("F-S from restart: {:?} cost {:.4}", fs.to_vec(), cost_at(fs[0], fs[1]));
    }
}
