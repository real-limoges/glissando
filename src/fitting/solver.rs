//! Penalized weighted least squares (PWLS) solver and GCV smoothing-parameter optimization.
//!
//! Two jobs live here. Cholesky decomposition solves the PWLS system itself, and a hand-rolled
//! L-BFGS with a strong-Wolfe line search sits on top, moving the smoothing parameters (lambda)
//! around to minimize the GCV/REML score. The rest of the file is mostly the bookkeeping that
//! keeps those two cheap.

use super::{
    Coefficients, CovarianceMatrix, GamlssError, LogLambdas, ModelMatrix, PenaltyMatrix,
    SmoothingCriterion,
};
use crate::linalg;
use argmin::core::{CostFunction, Error, Gradient};
use ndarray::prelude::*;
use std::collections::VecDeque;

/// Minimum denominator value to prevent division by zero in GCV computation
const MIN_DENOMINATOR: f64 = 1e-10;

/// Minimum lambda value for log-space conversion
const MIN_LAMBDA: f64 = 1e-10;

/// Relative tolerance for declaring a penalty eigenvalue numerically zero.
/// τ = REML_RANK_TOL_EPS · max(eigval); below this the direction is treated as
/// part of the penalty null space when computing the pseudo-determinant.
const REML_RANK_TOL_EPS: f64 = 1e-8;

/// Clamp on |log λ| applied to REML's optimized output before exponentiation.
/// Wide enough that no genuine solution ever hits it, narrow enough that L-BFGS
/// can't wander off into numerically pathological territory.
pub(super) const LOG_LAMBDA_CLAMP: f64 = 30.0;

/// Decades (in natural-log λ) below the cold-start heuristic used to seed the
/// collapse-guarded restart. The cold start sits *past* the interior LAML
/// optimum, out on the slope toward the flat high-λ shelf; subtract this offset
/// and the restart lands safely below both the optimum and the shelf, so a
/// gradient or fixed-point optimizer can descend into the (unimodal) interior
/// optimum instead of staying pinned to the penalty null space.
const RESTART_LOG_OFFSET: f64 = 8.0;

/// Max Fellner-Schall iterations before returning the current λ.
const FS_MAX_ITERS: usize = 50;

/// Fellner-Schall convergence threshold on max |log λ_j_new − log λ_j_old|.
///
/// The F-S update is only first-order convergent, so stopping at 1e-3 (≈ 0.1%
/// relative change in λ) leaves noticeably more drift than stopping at 1e-4
/// (≈ 0.01%). A tighter threshold costs at most a few extra iterations, and F-S
/// is cheap, so it is worth it to have λ genuinely stationary before the outer
/// RS loop calls the whole fit converged.
const FS_TOL: f64 = 1e-4;

/// Relative floor on the Fellner-Schall multiplicative-update numerator
/// `tr(S_λ⁺ S_j) − tr(V S_j)`. Wood-Fasiolo §3 discusses pathological signs
/// (the numerator should be non-negative under the smoothness prior); a small
/// relative floor keeps the update bounded when it dips.
const FS_NUMERATOR_FLOOR_REL: f64 = 1e-12;

/// Floor on the Fellner-Schall denominator `β̂ᵀ S_j β̂`, which is near-zero
/// when β̂ sits in the null space of S_j. Prevents division-by-zero.
const FS_DENOMINATOR_FLOOR: f64 = 1e-12;

/// Precomputed weighted normal-equation quantities for a fixed `(z, w)` pair.
///
/// `X`, `z`, and `w` never change for the whole duration of one smoothing-
/// parameter search; only `λ` moves, across the L-BFGS/Fellner-Schall iterations
/// and the collapse-guarded restart's basin probes (up to several hundred λ
/// evaluations per [`super::scoring::step`] call). So I build this once and reuse
/// it, which turns each λ evaluation's dominant cost from `O(n·p²)` (rebuilding
/// `X'WX`/`X'Wz` from the raw `n`-row design every time) into `O(p²)`–`O(p³)`
/// (factorizing the already-assembled `p×p` system). That is the gap that
/// actually matters whenever `n ≫ p`.
pub(crate) struct WeightedNormalEquations {
    x_t_w_x: Array2<f64>,
    x_t_w_z: Array1<f64>,
    /// `z'Wz`, used to compute RSS algebraically as
    /// `z'Wz − 2β'(X'Wz) + β'(X'WX)β` without ever re-touching the `n`-length
    /// residual vector once `X'WX`/`X'Wz` are formed.
    z_t_w_z: f64,
    n_obs: usize,
}

impl WeightedNormalEquations {
    pub(crate) fn new(x_matrix: &ModelMatrix, z: &Array1<f64>, w_diag: &Array1<f64>) -> Self {
        let x = &x_matrix.0;
        // Fold the weights in as √W rather than ever materializing the n×n
        // diagonal W. X'WX = (√W·X)'(√W·X) and X'Wz = (√W·X)'(√W·z), so the
        // memory drops from O(n²) to O(n·p). No reason to pay for the big matrix.
        let sqrt_w = w_diag.mapv(f64::sqrt);
        let x_weighted = x * &sqrt_w.view().insert_axis(Axis(1));
        let z_weighted = z * &sqrt_w;

        let x_t_w_x = x_weighted.t().dot(&x_weighted);
        let x_t_w_z = x_weighted.t().dot(&z_weighted);
        let z_t_w_z = z_weighted.dot(&z_weighted);

        Self {
            x_t_w_x,
            x_t_w_z,
            z_t_w_z,
            n_obs: z.len(),
        }
    }
}

/// Coefficient-block grouping of the penalty matrices: which contiguous
/// `[start, end]` range each penalty's non-zero entries occupy, with
/// penalties sharing an exact range (e.g. the two marginal penalties of a
/// tensor-product smooth) merged into one group.
///
/// This depends only on `penalty_matrices`, not λ, so (exactly like
/// [`WeightedNormalEquations`]) it is computed once per smoothing-parameter search
/// and reused across every [`penalty_eigen`] call that search makes, rather than
/// re-scanning every penalty's non-zero block on each one.
pub(crate) struct PenaltyGroups(Vec<(usize, usize, Vec<usize>)>);

pub(crate) fn group_penalties(penalty_matrices: &[PenaltyMatrix]) -> PenaltyGroups {
    let ranges: Vec<(usize, usize)> = penalty_matrices
        .iter()
        .map(|s_j| s_j.block_range())
        .collect();

    let mut processed = vec![false; penalty_matrices.len()];
    let mut groups = Vec::new();
    for first in 0..penalty_matrices.len() {
        if processed[first] {
            continue;
        }
        let (start, end) = ranges[first];
        let members: Vec<usize> = (first..penalty_matrices.len())
            .filter(|&k| ranges[k] == (start, end))
            .collect();
        for &k in &members {
            processed[k] = true;
        }
        groups.push((start, end, members));
    }
    PenaltyGroups(groups)
}

/// Result from PWLS fitting that includes gradient computation info.
struct PwlsGradientInfo {
    beta: Coefficients,
    v_matrix: Array2<f64>,
    edf: f64,
    /// Per-coefficient EDF contributions: `diag(V·X'WX)`. Summing a contiguous
    /// block attributes effective degrees of freedom to an individual term.
    edf_per_coeff: Array1<f64>,
    /// `S_λ = Σ_j λ_j·S_j`, already formed while assembling `lhs`; kept so
    /// callers needing it too (the REML objective's `β'S_λβ` term) don't
    /// rebuild it.
    s_lambda: Array2<f64>,
    x_t_w_r: Array1<f64>,
    rss: f64,
}

pub(crate) struct GamlssCost<'a> {
    pub(crate) nfo: &'a WeightedNormalEquations,
    pub(crate) penalty_matrices: &'a [PenaltyMatrix],
}

impl<'a> GamlssCost<'a> {
    /// Generalized Cross-Validation (GCV) score and its gradient wrt ρ = log λ,
    /// both from a single PWLS solve.
    ///
    /// GCV approximates leave-one-out CV without refitting n times:
    ///   GCV(λ) = n * RSS / (n - EDF)²
    /// where RSS is the weighted residual sum of squares and EDF the effective
    /// degrees of freedom. Minimizing it trades fit (low RSS) off against
    /// complexity (high EDF). The optimization runs in log-space (ρ = log λ),
    /// which buys numerical stability and an unconstrained problem for free.
    ///
    /// The gradient carries the full chain rule: β itself depends on λ through
    /// the penalized normal equations, so dRSS/dλ and dEDF/dλ each have more
    /// terms than they first look. See docs/math/mathematics.md for the
    /// derivation. Computing score and gradient together reuses one
    /// `fit_pwls_with_grad_info` solve rather than paying for two.
    fn objective_and_grad(&self, rho: &Array1<f64>) -> Result<(f64, Array1<f64>), GamlssError> {
        let lambdas = rho.mapv(f64::exp);
        let n_penalties = lambdas.len();

        let info = fit_pwls_with_grad_info(self.nfo, self.penalty_matrices, &lambdas)?;

        let n = self.nfo.n_obs as f64;
        let denom = n - info.edf;

        // Guard the divide-by-zero when EDF creeps up toward n (an overfit).
        if denom.powi(2).abs() < MIN_DENOMINATOR {
            return Ok((f64::MAX, Array1::zeros(n_penalties)));
        }
        let gcv_score = (n * info.rss) / denom.powi(2);

        let mut grad_vec = Array1::zeros(n_penalties);
        for j in 0..n_penalties {
            let s_j = &self.penalty_matrices[j];
            let (start, end) = s_j.block_range();
            let beta_block = info.beta.0.slice(s![start..=end]);

            // dRSS/dlambda_j = 2 * (X'Wr)' * V * Sj * beta. Sj*beta is zero
            // outside [start,end] and equals block.dot(beta_block) inside it,
            // so V*(Sj*beta) only needs V's [start,end] column slice.
            let sj_beta_block = s_j.block.dot(&beta_block);
            let v_sj_beta = info.v_matrix.slice(s![.., start..=end]).dot(&sj_beta_block);
            let d_rss = 2.0 * info.x_t_w_r.dot(&v_sj_beta);

            // dEDF/dlambda_j = -tr(V * Sj * V * X'WX). V·Sj is zero outside
            // columns [start,end] by the same reasoning, so the left multiply also
            // only needs V's [start,end] column slice. The final Hadamard-sum
            // against X'WX has to stay full-width (v_sj_v is genuinely dense), but
            // V·Sj·V is symmetric and so is X'WX, so that trace still collapses to
            // a Hadamard-product row sum rather than a third full matrix product.
            let v_sj_cols = info.v_matrix.slice(s![.., start..=end]).dot(&s_j.block);
            let v_sj_v = v_sj_cols.dot(&info.v_matrix.slice(s![start..=end, ..]));
            let d_edf = -(&v_sj_v * &self.nfo.x_t_w_x).sum();

            // Quotient rule: dGCV/dlambda_j = n * [dRSS*(n-EDF) + 2*RSS*dEDF] / (n-EDF)^3
            let d_gcv = n * (d_rss * denom + 2.0 * info.rss * d_edf) / denom.powi(3);

            // Chain rule for log-space: d/d(log lambda) = lambda * d/dlambda
            grad_vec[j] = lambdas[j] * d_gcv;
        }

        Ok((gcv_score, grad_vec))
    }

    /// GCV score alone (still one PWLS solve; the gradient loop it also runs is
    /// O(p²) per penalty, negligible against the solve).
    fn objective(&self, rho: &Array1<f64>) -> Result<f64, GamlssError> {
        Ok(self.objective_and_grad(rho)?.0)
    }

    /// GCV gradient wrt ρ = log λ.
    fn grad(&self, rho: &Array1<f64>) -> Result<Array1<f64>, GamlssError> {
        Ok(self.objective_and_grad(rho)?.1)
    }
}

// Thin argmin adapters over the inherent methods, which hold the real
// implementation. Retained only while the `argmin` dependency remains.
impl<'a> CostFunction for GamlssCost<'a> {
    type Param = LogLambdas;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        self.objective(&param.0).map_err(Error::new)
    }
}

impl<'a> Gradient for GamlssCost<'a> {
    type Param = LogLambdas;
    type Gradient = LogLambdas;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Param, Error> {
        self.grad(&param.0).map(LogLambdas).map_err(Error::new)
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
    pub(crate) nfo: &'a WeightedNormalEquations,
    pub(crate) penalty_matrices: &'a [PenaltyMatrix],
    pub(crate) groups: &'a PenaltyGroups,
}

impl<'a> RemlCost<'a> {
    /// LAML/REML objective `−V_r` and its analytic gradient wrt ρ = log λ, both
    /// from a single PWLS solve plus one penalty eigendecomposition.
    ///
    /// V_r = ℓ − ½·βᵀS_λβ + ½·log|S_λ|_+ − ½·log|H+S_λ| + (M_p/2)·log(2π)
    /// with ℓ = −½·RSS (constants in the working-likelihood independent of λ cancel).
    ///
    /// ∂V_r/∂ρ_j = −(λ_j/2)·β̂ᵀS_jβ̂ + (λ_j/2)·tr(S_λ⁺ S_j) − (λ_j/2)·tr(V S_j).
    /// V, S_λ⁺, and S_j are symmetric, so each trace reduces to a Hadamard-sum.
    fn objective_and_grad(&self, rho: &Array1<f64>) -> Result<(f64, Array1<f64>), GamlssError> {
        let lambdas = rho.mapv(f64::exp);
        let n_penalties = lambdas.len();

        let info = fit_pwls_with_grad_info(self.nfo, self.penalty_matrices, &lambdas)?;

        // No smoothing parameters: no penalty spectrum to form. `penalty_eigen`
        // debug-asserts a non-empty penalty set, so short-circuit here.
        if n_penalties == 0 {
            let lhs = &self.nfo.x_t_w_x + &info.s_lambda;
            let log_det_lhs = linalg::log_det_robust(&lhs)?;
            let v_r = -0.5 * info.rss - 0.5 * log_det_lhs;
            return Ok((-v_r, Array1::zeros(0)));
        }

        let eig = penalty_eigen(
            self.nfo.x_t_w_x.nrows(),
            self.penalty_matrices,
            &lambdas,
            REML_RANK_TOL_EPS,
            self.groups,
        )?;

        let lhs = &self.nfo.x_t_w_x + &info.s_lambda;
        let log_det_lhs = linalg::log_det_robust(&lhs)?;

        let beta_s_beta = info.beta.0.dot(&info.s_lambda.dot(&info.beta.0));
        let m_p = eig.null_dim as f64;

        let v_r = -0.5 * info.rss - 0.5 * beta_s_beta + 0.5 * eig.log_pdet - 0.5 * log_det_lhs
            + 0.5 * m_p * (2.0 * std::f64::consts::PI).ln();

        let mut grad = Array1::<f64>::zeros(n_penalties);
        for j in 0..n_penalties {
            let s_j = &self.penalty_matrices[j];
            let (start, end) = s_j.block_range();
            let beta_block = info.beta.0.slice(s![start..=end]);
            let bsb = s_j.block.dot(&beta_block).dot(&beta_block);
            let tr_v_s = (&info.v_matrix.slice(s![start..=end, start..=end]) * &s_j.block).sum();
            let tr_pinv_s = (&eig.pinv.slice(s![start..=end, start..=end]) * &s_j.block).sum();
            // ∂V_r/∂ρ_j; minimize −V_r so negate.
            let dvr = 0.5 * lambdas[j] * (-bsb + tr_pinv_s - tr_v_s);
            grad[j] = -dvr;
        }

        Ok((-v_r, grad))
    }

    /// LAML/REML objective `−V_r` alone.
    fn objective(&self, rho: &Array1<f64>) -> Result<f64, GamlssError> {
        Ok(self.objective_and_grad(rho)?.0)
    }

    /// Gradient of `−V_r` wrt ρ = log λ.
    fn grad(&self, rho: &Array1<f64>) -> Result<Array1<f64>, GamlssError> {
        Ok(self.objective_and_grad(rho)?.1)
    }
}

// Thin argmin adapters over the inherent methods, which hold the real
// implementation. Retained only while the `argmin` dependency remains.
impl<'a> CostFunction for RemlCost<'a> {
    type Param = LogLambdas;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        self.objective(&param.0).map_err(Error::new)
    }
}

impl<'a> Gradient for RemlCost<'a> {
    type Param = LogLambdas;
    type Gradient = LogLambdas;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Param, Error> {
        self.grad(&param.0).map(LogLambdas).map_err(Error::new)
    }
}

/// Tunables for the hand-rolled L-BFGS ([`lbfgs_minimize`]). One struct so every
/// magic number sits in one place, like the Fellner-Schall constants above.
struct LbfgsConfig {
    /// History size (number of `(s, y)` pairs). Matches the old `LBFGS::new(_, 7)`.
    m: usize,
    /// Iteration cap. Matches the old `.max_iters(50)`.
    max_iters: usize,
    /// Convergence on the gradient ∞-norm (in ρ = log λ units). At 1e-5 the inner
    /// λ-solve is 1-2 orders tighter than the outer RS tolerance (1e-3) and
    /// `FS_TOL` (1e-4), so it never limits outer accuracy.
    grad_tol: f64,
    /// Secondary convergence on relative objective change `|Δf| / max(|f|, 1)`,
    /// for flat-ridge cases where the gradient norm plateaus slowly but the
    /// objective is already stationary.
    rel_f_tol: f64,
    /// Armijo (sufficient-decrease) constant.
    c1: f64,
    /// Strong-curvature constant. 0.9 is the textbook quasi-Newton value.
    c2: f64,
    /// Line-search iteration cap (shared by bracketing and zoom).
    max_ls_iters: usize,
}

impl Default for LbfgsConfig {
    fn default() -> Self {
        Self {
            m: 7,
            max_iters: 50,
            grad_tol: 1e-5,
            rel_f_tol: 1e-9,
            c1: 1e-4,
            c2: 0.9,
            max_ls_iters: 20,
        }
    }
}

#[inline]
fn inf_norm(v: &Array1<f64>) -> f64 {
    v.iter().fold(0.0_f64, |m, &x| m.max(x.abs()))
}

/// Hand-rolled L-BFGS with a strong-Wolfe line search, minimizing a smooth
/// objective over ρ ∈ ℝⁿ. Replaces argmin's `LBFGS` + `MoreThuenteLineSearch`.
///
/// `eval` returns `(f, ∇f)` together, so one call is one PWLS solve (argmin
/// called cost and gradient separately, paying two). The returned point is the
/// lowest-objective point ever visited, so a failed line search or a numerical
/// excursion can never regress the answer below the start.
///
/// Deterministic given a deterministic `eval`: no RNG, and the only reductions
/// are dot products over the (single-threaded, per `.cargo/config.toml`) BLAS.
fn lbfgs_minimize<FG>(
    x0: Array1<f64>,
    mut eval: FG,
    cfg: &LbfgsConfig,
) -> Result<Array1<f64>, GamlssError>
where
    FG: FnMut(&Array1<f64>) -> Result<(f64, Array1<f64>), GamlssError>,
{
    let mut s_hist: VecDeque<Array1<f64>> = VecDeque::with_capacity(cfg.m);
    let mut y_hist: VecDeque<Array1<f64>> = VecDeque::with_capacity(cfg.m);
    let mut rho_hist: VecDeque<f64> = VecDeque::with_capacity(cfg.m);

    let mut x = x0;
    let (mut f, mut g) = eval(&x)?;

    if !f.is_finite() {
        return Err(GamlssError::Optimization(
            "L-BFGS: non-finite objective at the initial point".to_string(),
        ));
    }

    // Best-seen, seeded from the start so we can never return something worse.
    let mut best_x = x.clone();
    let mut best_f = f;

    if inf_norm(&g) <= cfg.grad_tol {
        return Ok(best_x);
    }

    for _iter in 0..cfg.max_iters {
        // Two-loop recursion: r ≈ H_k · g.
        let mut q = g.clone();
        let mut alpha = vec![0.0_f64; s_hist.len()];
        for i in (0..s_hist.len()).rev() {
            let a = rho_hist[i] * s_hist[i].dot(&q);
            alpha[i] = a;
            q.scaled_add(-a, &y_hist[i]);
        }
        // Initial-Hessian scaling γ = (sᵀy)/(yᵀy) from the newest pair; γ = 1
        // (H₀ = I) on the first iteration → scaled steepest descent.
        let gamma = match y_hist.back() {
            Some(y_last) => {
                let s_last = s_hist.back().unwrap();
                let yy = y_last.dot(y_last);
                if yy > 0.0 {
                    s_last.dot(y_last) / yy
                } else {
                    1.0
                }
            }
            None => 1.0,
        };
        let mut r = q.mapv(|v| v * gamma);
        for i in 0..s_hist.len() {
            let b = rho_hist[i] * y_hist[i].dot(&r);
            r.scaled_add(alpha[i] - b, &s_hist[i]);
        }

        // Search direction −r; fall back to steepest descent if it is not a
        // descent direction (numerical trouble).
        let mut d = r.mapv(|v| -v);
        let mut gd = g.dot(&d);
        if !gd.is_finite() || gd >= 0.0 {
            d = g.mapv(|v| -v);
            gd = g.dot(&d);
            if gd >= 0.0 {
                break; // g ≈ 0: stationary.
            }
        }

        let ls = strong_wolfe(&mut eval, &x, f, &g, &d, cfg)?;
        let (_alpha_k, x_new, f_new, g_new) = match ls {
            Some(t) => t,
            None => break, // line search gave up: keep best-so-far.
        };

        if f_new < best_f {
            best_f = f_new;
            best_x = x_new.clone();
        }

        // Curvature update: push (s, y) only when sᵀy is sufficiently positive
        // (keeps the implicit Hessian PD); otherwise skip the pair.
        let s = &x_new - &x;
        let y = &g_new - &g;
        let sy = s.dot(&y);
        let curv_ok = sy > 1e-10 * s.dot(&s).sqrt() * y.dot(&y).sqrt();
        if curv_ok {
            if s_hist.len() == cfg.m {
                s_hist.pop_front();
                y_hist.pop_front();
                rho_hist.pop_front();
            }
            rho_hist.push_back(1.0 / sy);
            s_hist.push_back(s);
            y_hist.push_back(y);
        }

        let f_prev = f;
        x = x_new;
        f = f_new;
        g = g_new;

        if inf_norm(&g) <= cfg.grad_tol {
            break;
        }
        if (f_prev - f).abs() / f_prev.abs().max(1.0) <= cfg.rel_f_tol {
            break;
        }
    }

    Ok(best_x)
}

/// `(alpha, x_new, f_new, g_new)` from a successful line search.
type LineSearchOut = (f64, Array1<f64>, f64, Array1<f64>);

/// Evaluate φ(α) = f(x₀ + α·d): returns `(f, ∇f, φ'(α) = ∇f·d)`. A non-finite `f`
/// yields a NaN slope so callers treat the step as "too long".
fn eval_phi<FG>(
    eval: &mut FG,
    x0: &Array1<f64>,
    d: &Array1<f64>,
    a: f64,
) -> Result<(f64, Array1<f64>, f64), GamlssError>
where
    FG: FnMut(&Array1<f64>) -> Result<(f64, Array1<f64>), GamlssError>,
{
    let x = x0 + &d.mapv(|v| v * a);
    let (f, g) = eval(&x)?;
    let dphi = if f.is_finite() { g.dot(d) } else { f64::NAN };
    Ok((f, g, dphi))
}

/// Strong-Wolfe line search (Nocedal & Wright Alg 3.5 bracketing + 3.6 zoom).
///
/// Enforces both Armijo `f(x+αd) ≤ f + c1·α·φ'(0)` and strong curvature
/// `|φ'(α)| ≤ c2·|φ'(0)|`, which pins a well-determined step (this is what makes
/// the landing point reproducible, unlike a bare-Armijo backtracker). Returns
/// `None` when Wolfe cannot be met within the budget; the caller then keeps its
/// best-so-far point. Non-finite objectives (including the GCV `f64::MAX`
/// divide-by-zero sentinel) are treated as "step too long".
fn strong_wolfe<FG>(
    eval: &mut FG,
    x0: &Array1<f64>,
    f0: f64,
    g0: &Array1<f64>,
    d: &Array1<f64>,
    cfg: &LbfgsConfig,
) -> Result<Option<LineSearchOut>, GamlssError>
where
    FG: FnMut(&Array1<f64>) -> Result<(f64, Array1<f64>), GamlssError>,
{
    let dphi0 = g0.dot(d); // < 0, guaranteed by the caller.
    let a_max = 1e10;

    let mut a_prev = 0.0;
    let mut f_prev = f0;
    let mut dphi_prev = dphi0;
    let mut a_i = 1.0; // L-BFGS unit step.

    for i in 0..cfg.max_ls_iters {
        let (f_i, g_i, dphi_i) = eval_phi(eval, x0, d, a_i)?;

        // Armijo violated, non-finite, or non-decreasing vs the previous trial.
        if !f_i.is_finite() || f_i > f0 + cfg.c1 * a_i * dphi0 || (i > 0 && f_i >= f_prev) {
            return zoom(
                eval, x0, f0, dphi0, d, a_prev, f_prev, dphi_prev, a_i, f_i, dphi_i, cfg,
            );
        }
        // Strong curvature satisfied: accept.
        if dphi_i.abs() <= -cfg.c2 * dphi0 {
            return Ok(Some((a_i, x0 + &d.mapv(|v| v * a_i), f_i, g_i)));
        }
        // Overshot the minimizer (slope turned non-negative): bracket is [i, prev].
        if dphi_i >= 0.0 {
            return zoom(
                eval, x0, f0, dphi0, d, a_i, f_i, dphi_i, a_prev, f_prev, dphi_prev, cfg,
            );
        }

        a_prev = a_i;
        f_prev = f_i;
        dphi_prev = dphi_i;
        if a_i >= a_max {
            return Ok(None);
        }
        a_i = (a_i * 2.0).min(a_max);
    }

    Ok(None)
}

/// The `zoom` half of the strong-Wolfe search: shrink `[a_lo, a_hi]` (either
/// orientation) until a point satisfies both Wolfe conditions. `a_lo` always
/// holds the lower objective among trials satisfying Armijo.
#[allow(clippy::too_many_arguments)]
fn zoom<FG>(
    eval: &mut FG,
    x0: &Array1<f64>,
    f0: f64,
    dphi0: f64,
    d: &Array1<f64>,
    mut a_lo: f64,
    mut f_lo: f64,
    mut dphi_lo: f64,
    mut a_hi: f64,
    mut f_hi: f64,
    mut dphi_hi: f64,
    cfg: &LbfgsConfig,
) -> Result<Option<LineSearchOut>, GamlssError>
where
    FG: FnMut(&Array1<f64>) -> Result<(f64, Array1<f64>), GamlssError>,
{
    for _ in 0..cfg.max_ls_iters {
        let a_j = interp_safeguarded(a_lo, f_lo, dphi_lo, a_hi, f_hi, dphi_hi);
        let (f_j, g_j, dphi_j) = eval_phi(eval, x0, d, a_j)?;

        if !f_j.is_finite() || f_j > f0 + cfg.c1 * a_j * dphi0 || f_j >= f_lo {
            a_hi = a_j;
            f_hi = f_j;
            dphi_hi = dphi_j;
        } else {
            if dphi_j.abs() <= -cfg.c2 * dphi0 {
                return Ok(Some((a_j, x0 + &d.mapv(|v| v * a_j), f_j, g_j)));
            }
            if dphi_j * (a_hi - a_lo) >= 0.0 {
                a_hi = a_lo;
                f_hi = f_lo;
                dphi_hi = dphi_lo;
            }
            a_lo = a_j;
            f_lo = f_j;
            dphi_lo = dphi_j;
        }
    }
    Ok(None)
}

/// Cubic minimizer of the Hermite interpolant through `(a, fa, dfa)` and
/// `(b, fb, dfb)` (Nocedal & Wright eq. 3.59). `None` when the cubic is
/// degenerate or non-finite.
fn cubic_min(a: f64, fa: f64, dfa: f64, b: f64, fb: f64, dfb: f64) -> Option<f64> {
    let d1 = dfa + dfb - 3.0 * (fa - fb) / (a - b);
    let disc = d1 * d1 - dfa * dfb;
    if disc < 0.0 || !disc.is_finite() {
        return None;
    }
    let d2 = (b - a).signum() * disc.sqrt();
    let denom = dfb - dfa + 2.0 * d2;
    if denom == 0.0 || !denom.is_finite() {
        return None;
    }
    let a_new = b - (b - a) * (dfb + d2 - d1) / denom;
    a_new.is_finite().then_some(a_new)
}

/// A trial α strictly inside the bracket: safeguarded cubic interpolation,
/// falling back to bisection when the cubic minimizer lands outside the inner
/// 80% of `[a_lo, a_hi]` (or is degenerate, e.g. a non-finite high endpoint).
fn interp_safeguarded(a_lo: f64, f_lo: f64, dphi_lo: f64, a_hi: f64, f_hi: f64, dphi_hi: f64) -> f64 {
    let lo = a_lo.min(a_hi);
    let hi = a_lo.max(a_hi);
    let width = hi - lo;
    let safe_lo = lo + 0.1 * width;
    let safe_hi = hi - 0.1 * width;
    match cubic_min(a_lo, f_lo, dphi_lo, a_hi, f_hi, dphi_hi) {
        Some(a) if a >= safe_lo && a <= safe_hi => a,
        _ => 0.5 * (lo + hi),
    }
}

/// Runs L-BFGS to find the optimal smoothing parameters (lambdas).
///
/// Warm-starts from the previous lambdas when they are available, which is most of
/// the time inside the RS loop and converges noticeably faster for it. Skips the
/// whole optimization when there are no penalty matrices to optimize over.
pub(crate) fn run_optimization(
    nfo: &WeightedNormalEquations,
    penalty_matrices: &[PenaltyMatrix],
    initial_lambdas: Option<&Array1<f64>>,
) -> Result<Array1<f64>, GamlssError> {
    let n_penalties = penalty_matrices.len();

    // Fast path: no penalties, nothing to optimize.
    if n_penalties == 0 {
        return Ok(Array1::zeros(0));
    }

    let cost = GamlssCost {
        nfo,
        penalty_matrices,
    };

    // Warm-start from the previous lambdas (in log-space) when we have them.
    let initial_log_lambdas = match initial_lambdas {
        Some(prev) if prev.len() == n_penalties => prev.mapv(|l| l.max(MIN_LAMBDA).ln()),
        _ => Array1::<f64>::zeros(n_penalties),
    };

    let cfg = LbfgsConfig::default();
    let best_log_lambdas =
        lbfgs_minimize(initial_log_lambdas, |rho| cost.objective_and_grad(rho), &cfg)?;

    Ok(best_log_lambdas.mapv(f64::exp))
}

/// REML/LAML analogue of `run_optimization`.
///
/// What differs from the GCV path:
/// - the cold start uses `initial_log_lambda` (the mgcv-style heuristic) instead of zeros;
/// - no `target_cost` early exit, because `−V_r` is not bounded below by zero;
/// - the returned log λ is clamped to `[-LOG_LAMBDA_CLAMP, LOG_LAMBDA_CLAMP]`
///   before exponentiation, so a runaway L-BFGS step can't hand back a non-finite λ.
///
/// `polish` decides whether to chase the L-BFGS solve with the deterministic
/// Fellner-Schall pass described below. The caller in `scoring::step` leaves it
/// `true` on the per-cycle adopted-λ path (where it is the documented fix for
/// L-BFGS stalls) and flips it `false` for the cheap comparison-only probes in the
/// basin-restart search. There it screens up to ~9 candidates a cycle, and paying
/// the extra polish plus 2 `lambda_cost` evaluations on every one would blow up
/// the single most expensive step in the fitting loop; the eventual winner gets
/// re-polished on the very next RS cycle anyway.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_optimization_reml(
    x_model: &ModelMatrix,
    nfo: &WeightedNormalEquations,
    penalty_matrices: &[PenaltyMatrix],
    groups: &PenaltyGroups,
    initial_lambdas: Option<&Array1<f64>>,
    polish: bool,
) -> Result<Array1<f64>, GamlssError> {
    let n_penalties = penalty_matrices.len();
    if n_penalties == 0 {
        return Ok(Array1::zeros(0));
    }

    let cost = RemlCost {
        nfo,
        penalty_matrices,
        groups,
    };

    let initial_log_lambdas = match initial_lambdas {
        Some(prev) if prev.len() == n_penalties => prev.mapv(|l| {
            l.max(MIN_LAMBDA)
                .ln()
                .clamp(-LOG_LAMBDA_CLAMP, LOG_LAMBDA_CLAMP)
        }),
        _ => initial_log_lambda(x_model, penalty_matrices),
    };

    let cfg = LbfgsConfig::default();
    let best_log_lambdas =
        lbfgs_minimize(initial_log_lambdas, |rho| cost.objective_and_grad(rho), &cfg)?;

    let clamped = best_log_lambdas.mapv(|l| l.clamp(-LOG_LAMBDA_CLAMP, LOG_LAMBDA_CLAMP));
    let lbfgs_lambdas = clamped.mapv(f64::exp);

    if !polish {
        return Ok(lbfgs_lambdas);
    }

    // Fellner-Schall polish. L-BFGS can stall at a warm-start-dependent
    // non-stationary point whenever the LAML surface has flat ridges (smooths
    // collapsing to their null space with λ at the clamp ceiling). F-S iterates the
    // same LAML target monotonically to a deterministic fixed point, ironing out
    // that per-cycle λ jitter. Every candidate below is scored by REML cost and the
    // lowest wins, with L-BFGS always in the running, so the polish can never make
    // the fit worse than L-BFGS left it. Each F-S run is best-effort: a linear-
    // algebra failure inside one (an eigensolver hiccup at a degenerate λ, say)
    // just drops that candidate rather than taking the whole fit down.
    let reml_cost =
        |lams: &Array1<f64>| lambda_cost(SmoothingCriterion::Reml, nfo, penalty_matrices, groups, lams);

    let mut best = lbfgs_lambdas;
    let mut best_cost = reml_cost(&best).unwrap_or(f64::INFINITY);

    let mut candidates: Vec<Array1<f64>> = Vec::new();
    // Warm-started from the L-BFGS point.
    if let Ok(p) = run_optimization_fellner_schall(x_model, nfo, penalty_matrices, groups, Some(&best)) {
        candidates.push(p);
    }
    // Cold-started from the heuristic, multi-penalty only. Anisotropic tensor
    // smooths grow corner basins where one margin's λ is driven very large but
    // short of the clamp bound, so the collapse-guarded restart in `scoring::step`
    // (which triggers only on a collapsed term or a bound-pinned λ) never fires,
    // and both L-BFGS and a warm F-S stall there. A cold F-S ignores the corner
    // warm start and descends into the interior optimum. Single-penalty problems
    // are unimodal, so this is skipped and their result is untouched.
    if n_penalties > 1 {
        if let Ok(c) = run_optimization_fellner_schall(x_model, nfo, penalty_matrices, groups, None) {
            candidates.push(c);
        }
    }

    for cand in candidates {
        if let Ok(c) = reml_cost(&cand) {
            if c <= best_cost {
                best = cand;
                best_cost = c;
            }
        }
    }
    Ok(best)
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
/// where `V = (X'WX + S_λ)⁻¹` and `β̂` come from a PIRLS solve at the current `λ`.
/// The nice part is that Wood & Fasiolo prove monotone improvement of the LAML
/// score under mild regularity, for any quadratically penalized smooth
/// log-likelihood, so this update can only ever help.
///
/// How it stacks up against `run_optimization_reml`:
/// - no outer L-BFGS and no line search → deterministic across linalg backends;
/// - first-order convergence (slower asymptotically) but no Hessian and no
///   step-size tuning to get wrong;
/// - shares every helper with the REML path (`fit_pwls_with_grad_info`,
///   `penalty_eigen`).
pub(crate) fn run_optimization_fellner_schall(
    x_model: &ModelMatrix,
    nfo: &WeightedNormalEquations,
    penalty_matrices: &[PenaltyMatrix],
    groups: &PenaltyGroups,
    initial_lambdas: Option<&Array1<f64>>,
) -> Result<Array1<f64>, GamlssError> {
    let n_penalties = penalty_matrices.len();
    if n_penalties == 0 {
        return Ok(Array1::zeros(0));
    }

    // Initialize λ: warm-start from the previous value, else the mgcv-style heuristic.
    let mut lambdas: Array1<f64> = match initial_lambdas {
        Some(prev) if prev.len() == n_penalties => prev.mapv(|l| l.max(MIN_LAMBDA)),
        _ => initial_log_lambda(x_model, penalty_matrices).mapv(f64::exp),
    };

    let lambda_ceiling = LOG_LAMBDA_CLAMP.exp();

    for _iter in 0..FS_MAX_ITERS {
        let info = fit_pwls_with_grad_info(nfo, penalty_matrices, &lambdas)?;
        let eig = penalty_eigen(
            nfo.x_t_w_x.nrows(),
            penalty_matrices,
            &lambdas,
            REML_RANK_TOL_EPS,
            groups,
        )?;

        // Every coordinate's update reads only `info` / `eig` (both computed above
        // at the *pre-loop* λ) plus its own `lambdas[j]`, so updating in place is
        // exactly equivalent to clone-and-swap, and matches the simultaneous
        // Fellner-Schall update. No coordinate sees another's fresh value mid-sweep.
        let mut max_log_change: f64 = 0.0;

        for j in 0..n_penalties {
            let s_j = &penalty_matrices[j];
            let (start, end) = s_j.block_range();
            let beta_block = info.beta.0.slice(s![start..=end]);
            let tr_pinv_s = (&eig.pinv.slice(s![start..=end, start..=end]) * &s_j.block).sum();
            let tr_v_s = (&info.v_matrix.slice(s![start..=end, start..=end]) * &s_j.block).sum();
            let bsb = s_j.block.dot(&beta_block).dot(&beta_block);

            let numerator = (tr_pinv_s - tr_v_s).max(FS_NUMERATOR_FLOOR_REL * lambdas[j]);
            let denominator = bsb.max(FS_DENOMINATOR_FLOOR);
            let lambda_new = (lambdas[j] * (numerator / denominator))
                .max(MIN_LAMBDA)
                .min(lambda_ceiling);

            let log_change = (lambda_new.ln() - lambdas[j].ln()).abs();
            if log_change > max_log_change {
                max_log_change = log_change;
            }
            lambdas[j] = lambda_new;
        }

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
/// Hands back the coefficients beta, the covariance matrix
/// V = (X'WX + sum lambda*S)^-1, and the effective degrees of freedom
/// EDF = tr(V * X'WX).
pub(crate) fn fit_pwls(
    nfo: &WeightedNormalEquations,
    penalty_matrices: &[PenaltyMatrix],
    lambdas: &Array1<f64>,
) -> Result<(Coefficients, CovarianceMatrix, f64, Array1<f64>), GamlssError> {
    let info = fit_pwls_with_grad_info(nfo, penalty_matrices, lambdas)?;
    Ok((
        info.beta,
        CovarianceMatrix(info.v_matrix),
        info.edf,
        info.edf_per_coeff,
    ))
}

fn fit_pwls_with_grad_info(
    nfo: &WeightedNormalEquations,
    penalty_matrices: &[PenaltyMatrix],
    lambdas: &Array1<f64>,
) -> Result<PwlsGradientInfo, GamlssError> {
    let s_lambda = weighted_penalty_sum(nfo.x_t_w_x.nrows(), lambdas, penalty_matrices);
    let lhs = &nfo.x_t_w_x + &s_lambda;

    // `solve_robust`/`inv_robust` drop to an eigendecomposition when `lhs` goes
    // near-singular in floating point (a smooth term collapsing toward its penalty
    // null space, say). The plain LU path has no fallback for exactly that, and it
    // bit us as a CI-only `dgesv`/`dpotrf` failure that wouldn't reproduce locally:
    // BLAS-build-dependent rounding tips a near-zero pivot over the edge. The fast
    // path matches the old direct solve/inv exactly, so nothing changes when `lhs`
    // is healthy.
    let beta_arr = linalg::solve_robust(&lhs, &nfo.x_t_w_z)?;
    let beta = Coefficients(beta_arr);

    let v = linalg::inv_robust(&lhs)?;

    // EDF (effective degrees of freedom) is the model-complexity measure.
    // EDF = tr(H) where H = X(X'WX + sum lambda*S)^-1 X'W is the hat matrix.
    // Equivalently, EDF = tr(V * X'WX), running from 0 (lambda->inf) to p (lambda->0).
    // Keep the per-coefficient diagonal around so callers can attribute EDF per term.
    //
    // Only the diagonal matters, and X'WX is a Gram matrix (symmetric), so
    // diag(V·X'WX)_i = Σ_k V[i,k]·X'WX[i,k]: an elementwise product with a row sum,
    // O(p²), instead of the full O(p³) matrix product. No reason to form the product.
    let edf_per_coeff = (&v * &nfo.x_t_w_x).sum_axis(Axis(1));
    let edf = edf_per_coeff.sum();

    // RSS and X'Wr from the precomputed normal equations rather than a fresh
    // n-length residual pass: r = z − Xβ, so
    //   RSS  = r'Wr = z'Wz − 2β'(X'Wz) + β'(X'WX)β
    //   X'Wr = X'Wz − (X'WX)β
    // Floating-point round-off can nudge the algebraic RSS very slightly negative
    // when λ drives β close to the unpenalized LS fit, so clamp it at 0.
    let xtwx_beta = nfo.x_t_w_x.dot(&beta.0);
    let rss = (nfo.z_t_w_z - 2.0 * beta.0.dot(&nfo.x_t_w_z) + beta.0.dot(&xtwx_beta)).max(0.0);
    let x_t_w_r = &nfo.x_t_w_z - &xtwx_beta;

    Ok(PwlsGradientInfo {
        beta,
        v_matrix: v,
        edf,
        edf_per_coeff,
        s_lambda,
        x_t_w_r,
        rss,
    })
}

/// Forms `S_λ = Σ_j λ_j · S_j` as an `n_coeffs × n_coeffs` matrix (zero when
/// `penalty_matrices` is empty, i.e. a purely parametric term with no
/// smoothing parameters). `n_coeffs` is taken from the caller rather than
/// `penalty_matrices[0]` so this works for that empty case too.
fn weighted_penalty_sum(
    n_coeffs: usize,
    lambdas: &Array1<f64>,
    penalty_matrices: &[PenaltyMatrix],
) -> Array2<f64> {
    let mut s = Array2::<f64>::zeros((n_coeffs, n_coeffs));
    for (i, s_j) in penalty_matrices.iter().enumerate() {
        let (start, end) = s_j.block_range();
        s.slice_mut(s![start..=end, start..=end])
            .scaled_add(lambdas[i], &s_j.block);
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
/// threshold τ = eps · max(eigenvalue of S_λ). This falls apart when the λ values
/// span many orders of magnitude (say λ₁ ≈ 1e-8 and λ₄ ≈ 5e10 in a 5-smooth model):
/// the threshold gets dominated by the large-λ term and then misreads the non-null
/// directions of the small-λ penalties as null space, which flips the sign of the
/// REML gradient. Silent and wrong, the worst combination.
///
/// The fix is to group penalty matrices by their non-zero block range (via
/// [`group_penalties`], computed once per λ search and passed in as `groups`) and
/// eigendecompose each group on its own. Within a group the combined scaled block is
/// formed first, then eigendecomposed. That buys two correctness guarantees at once:
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
    n_coeffs: usize,
    penalty_matrices: &[PenaltyMatrix],
    lambdas: &Array1<f64>,
    eps: f64,
    groups: &PenaltyGroups,
) -> Result<PenaltyEigen, GamlssError> {
    debug_assert!(!penalty_matrices.is_empty());
    let p = n_coeffs;

    let mut log_pdet = 0.0_f64;
    let mut null_dim = 0_usize;
    let mut pinv = Array2::<f64>::zeros((p, p));

    for (start, end, members) in &groups.0 {
        let (start, end) = (*start, *end);
        let block_size = end - start + 1;

        // Build the combined scaled block: Σ_{j ∈ group} λ_j · S_j_block.
        // For a single-penalty group this is just λ_j · S_j_block; for a tensor-product
        // group this is the anisotropic penalty λ₁(S_x₁⊗I) + λ₂(I⊗S_x₂) restricted to
        // the shared k₁k₂ coefficient block.
        let mut combined = Array2::<f64>::zeros((block_size, block_size));
        for &k in members {
            combined.scaled_add(lambdas[k], &penalty_matrices[k].block);
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

/// Cold-start heuristic for log λ when there is no warm start to lean on.
///
/// I use `tr(X'X) / tr(S_j)` (unweighted) rather than `tr(X'WX) / tr(S_j)`, and
/// the reason is scale-invariance. For a B-spline basis the column norms depend
/// only on the knot layout, not on the response scale, so the unweighted form
/// keeps the initial λ in a numerically friendly range no matter how large σ is.
/// The weighted form does not: `tr(X'WX) = tr(X'X) / σ²` for homoscedastic
/// Gaussian, which goes tiny (≈ 10⁻²¹) for price-scale data (σ ≈ 45k) and drops
/// L-BFGS right next to the unpenalized OLS solution, where the REML landscape is
/// badly conditioned. From there the smooth reliably overshoots into full collapse.
pub(super) fn initial_log_lambda(
    x_matrix: &ModelMatrix,
    penalty_matrices: &[PenaltyMatrix],
) -> Array1<f64> {
    // tr(X'X) = sum_ij X[i,j]^2: the diagonal entry (X'X)_jj is sum_i X[i,j]^2,
    // so summing the diagonal is just summing every squared entry of X. Lets us
    // skip materializing the full p×p X'X (O(n·p) instead of O(n·p²)).
    let tr_xtx = x_matrix
        .0
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .max(MIN_LAMBDA);
    Array1::from_iter(penalty_matrices.iter().map(|s_j| {
        let tr_sj = s_j.block.diag().sum().max(MIN_LAMBDA);
        (tr_xtx / tr_sj)
            .ln()
            .clamp(-LOG_LAMBDA_CLAMP, LOG_LAMBDA_CLAMP)
    }))
}

/// Low-λ seed for the collapse-guarded restart (see [`RESTART_LOG_OFFSET`]).
///
/// Derived from the scale-aware cold-start heuristic (an already-computed
/// [`initial_log_lambda`], so a caller needing both the raw heuristic and the
/// restart seed pays for one call, not two) so it adapts to the basis /
/// response scale, then shifted several decades down to sit below the high-λ
/// collapse shelf.
pub(super) fn restart_seed_from_heuristic(heur: &Array1<f64>) -> Array1<f64> {
    heur.mapv(|log_lambda| (log_lambda - RESTART_LOG_OFFSET).exp().max(MIN_LAMBDA))
}

/// Value of the objective the given criterion minimizes, evaluated at a fixed λ.
///
/// The collapse-guarded restart in [`super::scoring::step`] uses this to weigh a
/// restart's λ against the incumbent and keep whichever scores lower. Two things
/// fall out of that: the guard can never make a fit worse, and genuinely
/// null-space-optimal data (a linear truth under an order-2 penalty) correctly
/// *keeps* its collapsed fit, because that fit really does have the better
/// marginal likelihood.
pub(super) fn lambda_cost(
    criterion: SmoothingCriterion,
    nfo: &WeightedNormalEquations,
    penalty_matrices: &[PenaltyMatrix],
    groups: &PenaltyGroups,
    lambdas: &Array1<f64>,
) -> Result<f64, GamlssError> {
    let rho = lambdas.mapv(|l| l.max(MIN_LAMBDA).ln());
    match criterion {
        SmoothingCriterion::Gcv => GamlssCost {
            nfo,
            penalty_matrices,
        }
        .objective(&rho),
        // REML and Fellner-Schall minimize the same LAML target (−V_r).
        SmoothingCriterion::Reml | SmoothingCriterion::FellnerSchall => RemlCost {
            nfo,
            penalty_matrices,
            groups,
        }
        .objective(&rho),
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
        let pm = PenaltyMatrix {
            offset: 0,
            block: p.clone(),
        };
        let lambdas = arr1(&[1.0_f64]);
        let groups = group_penalties(std::slice::from_ref(&pm));
        let eig = penalty_eigen(
            p.nrows(),
            std::slice::from_ref(&pm),
            &lambdas,
            REML_RANK_TOL_EPS,
            &groups,
        )
        .unwrap();
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
        let p1 = PenaltyMatrix {
            offset: 0,
            block: arr2(&[[1.0_f64, 0.0], [0.0, 1.0]]),
        };
        let p2 = PenaltyMatrix {
            offset: 0,
            block: arr2(&[[2.0_f64, 1.0], [1.0, 2.0]]),
        };
        let lambdas = arr1(&[0.5_f64, 3.0]);
        let s = weighted_penalty_sum(2, &lambdas, &[p1, p2]);
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
        let p = PenaltyMatrix {
            offset: 0,
            block: arr2(&[[2.0_f64, 0.0], [0.0, 2.0]]),
        };
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
        let p = PenaltyMatrix {
            offset: 0,
            block: arr2(&[[1e-30, 0.0], [0.0, 1e-30]]),
        };
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
        (
            ModelMatrix(basis),
            z,
            w,
            vec![PenaltyMatrix {
                offset: 0,
                block: penalty,
            }],
        )
    }

    #[test]
    fn reml_cost_finite_on_simple_problem() {
        let (x, z, w, ps) = synthetic_pwls_problem();
        let nfo = WeightedNormalEquations::new(&x, &z, &w);
        let groups = group_penalties(&ps);
        let cost = RemlCost {
            nfo: &nfo,
            penalty_matrices: &ps,
            groups: &groups,
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

    /// The critical correctness gate for Fellner-Schall: capture λ at each F-S
    /// iteration and check that the LAML score (−V_r, as `RemlCost::cost` evaluates
    /// it) never rises along the trajectory. Wood & Fasiolo 2017 prove this holds,
    /// so if our implementation breaks it, the bug is in the update formula or one
    /// of the helpers it leans on. Nowhere else.
    #[test]
    fn fellner_schall_monotone_improves_laml() {
        let (x, z, w, ps) = synthetic_pwls_problem();
        let nfo = WeightedNormalEquations::new(&x, &z, &w);
        let groups = group_penalties(&ps);
        let cost = RemlCost {
            nfo: &nfo,
            penalty_matrices: &ps,
            groups: &groups,
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

            let info = fit_pwls_with_grad_info(&nfo, &ps, &lambdas).unwrap();
            let eig = penalty_eigen(
                nfo.x_t_w_x.nrows(),
                &ps,
                &lambdas,
                REML_RANK_TOL_EPS,
                &groups,
            )
            .unwrap();
            let mut new_lambdas = lambdas.clone();
            for j in 0..n_penalties {
                let s_j = &ps[j];
                let (start, end) = s_j.block_range();
                let beta_block = info.beta.0.slice(s![start..=end]);
                let tr_pinv_s = (&eig.pinv.slice(s![start..=end, start..=end]) * &s_j.block).sum();
                let tr_v_s =
                    (&info.v_matrix.slice(s![start..=end, start..=end]) * &s_j.block).sum();
                let bsb = s_j.block.dot(&beta_block).dot(&beta_block);
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
        let nfo = WeightedNormalEquations::new(&x, &z, &w);
        let groups = group_penalties(&ps);
        let cost = RemlCost {
            nfo: &nfo,
            penalty_matrices: &ps,
            groups: &groups,
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
        // Mirrors `reml_gradient_matches_finite_diff` for the GCV path; the GCV
        // gradient (quotient-rule + dEDF/dλ) had gone untested before this.
        let (x, z, w, ps) = synthetic_pwls_problem();
        let nfo = WeightedNormalEquations::new(&x, &z, &w);
        let cost = GamlssCost {
            nfo: &nfo,
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

    /// The hand-rolled L-BFGS must reach the same REML optimum the deterministic
    /// Fellner-Schall optimizer settles on (same LAML target), with the analytic
    /// gradient of −V_r effectively vanishing there. Run L-BFGS with the F-S
    /// polish OFF so this exercises the L-BFGS landing point, not the polish.
    #[test]
    fn lbfgs_reaches_reml_optimum() {
        let (x, z, w, ps) = synthetic_pwls_problem();
        let nfo = WeightedNormalEquations::new(&x, &z, &w);
        let groups = group_penalties(&ps);

        let lbfgs = run_optimization_reml(&x, &nfo, &ps, &groups, None, false).unwrap();
        let fs = run_optimization_fellner_schall(&x, &nfo, &ps, &groups, None).unwrap();

        for (a, b) in lbfgs.iter().zip(fs.iter()) {
            assert!(
                (a.ln() - b.ln()).abs() < 0.1,
                "L-BFGS λ = {:e} disagrees with Fellner-Schall λ = {:e}",
                a,
                b
            );
        }

        let cost = RemlCost {
            nfo: &nfo,
            penalty_matrices: &ps,
            groups: &groups,
        };
        let rho = lbfgs.mapv(|l| l.max(MIN_LAMBDA).ln());
        let g = cost.grad(&rho).unwrap();
        assert!(
            inf_norm(&g) < 1e-2,
            "gradient ∞-norm {} too large at the L-BFGS optimum",
            inf_norm(&g)
        );
    }

    /// Determinism is a hard requirement: the same inputs must give a
    /// bit-identical λ on every run (no RNG, no reduction-order dependence).
    #[test]
    fn lbfgs_is_deterministic() {
        let (x, z, w, ps) = synthetic_pwls_problem();
        let nfo = WeightedNormalEquations::new(&x, &z, &w);
        let groups = group_penalties(&ps);
        let a = run_optimization_reml(&x, &nfo, &ps, &groups, None, false).unwrap();
        let b = run_optimization_reml(&x, &nfo, &ps, &groups, None, false).unwrap();
        assert_eq!(a, b, "L-BFGS is not deterministic across runs");
    }

    /// The strong-Wolfe line search must return a step satisfying both the Armijo
    /// (sufficient-decrease) and strong-curvature conditions. Checked on a
    /// 1-D quadratic whose exact line minimizer is the unit step.
    #[test]
    fn strong_wolfe_satisfies_wolfe_conditions() {
        let cfg = LbfgsConfig::default();
        // f(t) = ½·(t − 3)², minimizer at t = 3.
        let mut eval = |x: &Array1<f64>| -> Result<(f64, Array1<f64>), GamlssError> {
            let t = x[0];
            Ok((0.5 * (t - 3.0).powi(2), arr1(&[t - 3.0])))
        };
        let x0 = arr1(&[0.0]);
        let (f0, g0) = eval(&x0).unwrap();
        let d = g0.mapv(|v| -v); // steepest descent
        let dphi0 = g0.dot(&d);

        let (alpha, x_new, f_new, g_new) = strong_wolfe(&mut eval, &x0, f0, &g0, &d, &cfg)
            .unwrap()
            .expect("line search should succeed on a convex quadratic");

        assert!(
            f_new <= f0 + cfg.c1 * alpha * dphi0 + 1e-12,
            "Armijo condition violated"
        );
        assert!(
            g_new.dot(&d).abs() <= -cfg.c2 * dphi0 + 1e-12,
            "strong-curvature condition violated"
        );
        assert!(
            (x_new[0] - alpha * d[0]).abs() < 1e-12,
            "returned point is not x0 + alpha*d"
        );
    }

    /// DIAGNOSTIC (Part 1, Q2 of the bistability investigation; `#[ignore]`d).
    ///
    /// Rebuilds the exact μ-subproblem of `mu_smooth_recovers_nonlinear_mean_control`
    /// and grids the REML/LAML objective `−V_r(λ)` over log λ. For a Gaussian with
    /// the identity link the IRLS working response is `z = y` exactly and the
    /// working weight is constant `w = 1/σ̂²` (σ̂ ≈ 0.2 here), so what this grids
    /// really *is* the landscape the outer loop's λ optimizer sees at convergence.
    ///
    /// For each grid point it prints λ, edf, and −V_r. The whole question it settles:
    /// is the collapsed region (edf → null-space ≈ 3, counting the unpenalized
    /// intercept) a *spurious local* optimum (some interior λ scores strictly lower
    /// −V_r) or the *global* one (REML honestly prefers the straight line)?
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
        let design = assemble_model_matrices(&data, n, &terms).unwrap();
        let (x_model, penalties, layouts) = (design.x, design.penalties, design.layouts);

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

        let nfo = WeightedNormalEquations::new(&x_model, &z, &w);
        let groups = group_penalties(&penalties);
        let cost = RemlCost {
            nfo: &nfo,
            penalty_matrices: &penalties,
            groups: &groups,
        };

        let mut best = (f64::INFINITY, 0.0_f64, 0.0_f64); // (-V_r, log λ, edf)
        let mut rows: Vec<(f64, f64, f64)> = Vec::new();
        let mut log_lambda = -6.0_f64;
        while log_lambda <= 34.0 + 1e-9 {
            let lambdas = arr1(&[log_lambda.exp()]);
            let (_b, _v, edf, _e) = fit_pwls(&nfo, &penalties, &lambdas).unwrap();
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
