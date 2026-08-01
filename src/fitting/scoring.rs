//! One Fisher-scoring step on a single distribution parameter.
//!
//! The Rigby–Stasinopoulos outer loop cycles through parameters; for each one,
//! [`step`] performs the five operations that constitute a P-IRLS update:
//!
//! 1. Snapshot every parameter on the response scale.
//! 2. Ask the family for score `u` and Fisher information `w` on the η-scale.
//! 3. Form the working response `z = η + u/w` (with weight floor and step clamp).
//! 4. Optimize smoothing parameters λ via the configured criterion (warm-started from the previous step).
//! 5. Solve penalized weighted least squares for `(β, V, EDF)`.
//!
//! `step` returns an [`Update`] describing the new state plus convergence and
//! diagnostic deltas. The caller applies the update; `step` itself is pure with
//! respect to the input `models` map, which keeps it unit-testable in isolation.

use super::solver::{
    fit_pwls, group_penalties, initial_log_lambda, lambda_cost, restart_seed_from_heuristic,
    run_optimization, run_optimization_fellner_schall, run_optimization_reml,
    WeightedNormalEquations,
};
use super::{
    global_deviance, global_deviance_with, max_abs_diff, FittingParameter, SmoothingCriterion,
};
use crate::distributions::{Distribution, MIN_WEIGHT};
use crate::error::GamlssError;
use crate::types::{Coefficients, CovarianceMatrix};
use indexmap::IndexMap;
use ndarray::{s, Array1, Zip};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::collections::HashMap;

/// Cap on the per-element Fisher-scoring step `u/w` (in η units): a pure
/// anti-overflow guard for degenerate score/information combinations, NOT a
/// robustness device. Overshoot control is the job of the deviance-guarded
/// step-halving (FIT-1), and neither mgcv nor gamlss clips the working
/// response at all.
///
/// The value must sit between two failure modes. Too tight inverts step
/// directions: when many rows clip, the update direction is decided by the
/// *count* of positive vs negative rows instead of the score-weighted
/// aggregate, which can point the Fisher step uphill (caps ≤ 1e4 break
/// Student-t ν recovery; legitimate transient steps exceed them). Too loose
/// lets a degenerate row (weight at the MIN_WEIGHT floor with an O(1) score,
/// e.g. a quasi-separated binomial observation) inject a large pseudo-residual
/// that distorts λ selection: an accepted exposure at 1e6 whose principled
/// fix is a REML criterion on the true likelihood rather than the working
/// (z, w) model. Applies only when step-halving is enabled; see
/// `MAX_STEP_NO_HALVING` for the disabled case.
const MAX_STEP: f64 = 1e6;

/// Fallback safety cap on the per-element accepted η-change when
/// `FitConfig::step_halving` is `false`. With step-halving on, `MAX_STEP`
/// above can safely be this loose because the deviance-guarded line search
/// owns overshoot control; with it off, nothing else bounds the step, so the
/// caller (`fitting::mod`) scales the raw Fisher step back to this bound
/// instead. This restores the pre-widening `MAX_STEP` value (20) as the sole
/// safety net for that path.
pub(super) const MAX_STEP_NO_HALVING: f64 = 20.0;

/// Backtracking floor for step-halving (FIT-1): `2^-10`. Below this the damped
/// step is accepted regardless so the loop always makes progress.
pub(super) const MIN_STEP_ALPHA: f64 = 1.0 / 1024.0;

/// New state plus per-step diagnostics produced by [`step`].
#[derive(Debug)]
pub(super) struct Update {
    pub beta: Coefficients,
    pub eta: Array1<f64>,
    /// Smoothing parameters on the response scale (not log-scale).
    pub lambdas: Array1<f64>,
    pub covariance: CovarianceMatrix,
    /// Per-term EDF.
    pub term_edf: Vec<f64>,
    /// Max |Δβ|; reported in diagnostics (`final_change`).
    pub max_diff: f64,
    /// Max |Δη| of the full-step proposal; drives the outer-loop convergence
    /// check. Measured in fit space rather than coefficient space so that
    /// movement along fit-irrelevant coefficient ridges (e.g. a flat REML
    /// valley where λ jitters but X·β is unchanged) cannot block convergence.
    pub eta_max_change: f64,
    pub eta_change: f64,
    pub lambda_change: f64,
    /// Observations whose working weight `w` was clamped at `MIN_WEIGHT`.
    pub weight_floor_hits: usize,
    /// Observations whose `u/w` step was clipped at `±MAX_STEP`.
    pub step_cap_hits: usize,
}

impl Update {
    /// Total EDF, derived from the per-term breakdown rather than tracked
    /// separately so the two can never drift apart.
    pub(super) fn edf(&self) -> f64 {
        self.term_edf.iter().sum()
    }
}

/// A (possibly damped) block update accepted by [`step_halving`].
#[derive(Debug)]
pub(super) struct Halved {
    pub beta: Coefficients,
    pub eta: Array1<f64>,
    pub mu: Array1<f64>,
    /// Number of halvings applied (`0` = the full Fisher step was accepted).
    pub hits: usize,
    /// True when the backtracking floor was reached with the penalized deviance
    /// still increasing and the whole block update was rejected (β unchanged).
    /// The caller must then also keep the previous λ/covariance/EDF: the
    /// full-step proposal's values describe a state that was never entered.
    pub rejected: bool,
    /// Max |Δη| actually applied: zero when `rejected`, otherwise the change
    /// between `eta` here and the pre-step η. Unlike `Update::eta_max_change`
    /// (the *proposed* full step), this reflects what was accepted, so the
    /// outer-loop convergence check never mistakes a rejected, re-proposed
    /// step for ongoing movement.
    pub eta_max_change: f64,
}

/// Backtrack a proposed block update on the PENALIZED global deviance, holding
/// the other parameters fixed (FIT-1).
///
/// The Fisher-scoring direction `d_k = β_new − β_old` is an ascent direction for
/// the **penalized** log-likelihood, i.e. a descent direction for
/// `GD_pen(β) = GD(μ(β)) + Σ_j λ_j·βᵀS_jβ`, NOT for the raw deviance. When the
/// current β is wigglier than the penalized optimum β*(λ) (e.g. λ grew across
/// cycles), every move toward β*(λ) legitimately *raises* the raw deviance;
/// backtracking on raw GD then rejects the step at every α and the fit freezes
/// at a non-stationary point (observed as a permanent poisson-smooth
/// non-convergence). Backtracking on `GD_pen`, evaluated at the *proposed*
/// step's λ so the objective is consistent along the α-path, restores the
/// guaranteed-descent property that makes halving terminate.
///
/// Returns the accepted step; `hits` is the number of halvings (`0` = full step).
pub(super) fn step_halving<D: Distribution + ?Sized>(
    family: &D,
    y: &Array1<f64>,
    prior_weights: Option<&Array1<f64>>,
    models: &IndexMap<String, FittingParameter>,
    param: &str,
    proposed: &Update,
    min_alpha: f64,
) -> Result<Halved, GamlssError> {
    let model = &models[param];
    let dir = &proposed.beta.0 - &model.beta.0; // d_k
    let (x, link) = (&model.x_matrix.0, &model.link);

    // Penalty CHANGE along the path, in cancellation-free form:
    //   Δpen(α) = (β₀+αd)ᵀS_λ(β₀+αd) − β₀ᵀS_λβ₀ = 2α·dᵀS_λβ₀ + α²·dᵀS_λd.
    // Evaluating the two quadratic forms separately and subtracting loses all
    // significance when λ is huge (e.g. a correctly collapsed smooth with λ at
    // the clamp ceiling ~1e13): the round-off noise of βᵀS_λβ dwarfs the true
    // difference and every step gets spuriously rejected.
    let (pen_cross, pen_dir) = {
        let mut cross = 0.0_f64; // dᵀ·S_λ·β₀
        let mut quad = 0.0_f64; // dᵀ·S_λ·d
        for (s_j, &lam) in model.penalty_matrices.iter().zip(proposed.lambdas.iter()) {
            let (start, end) = s_j.block_range();
            let dir_block = dir.slice(s![start..=end]);
            let beta_block = model.beta.0.slice(s![start..=end]);
            let s_d_block = s_j.block.dot(&dir_block);
            cross += lam * s_d_block.dot(&beta_block);
            quad += lam * s_d_block.dot(&dir_block);
        }
        (cross, quad)
    };

    let gd0 = global_deviance(family, y, prior_weights, models)?;
    // Round-off slack proportional to the deviance scale.
    let slack = 1e-8 * (1.0 + gd0.abs());

    let (mut alpha, mut hits) = (1.0_f64, 0usize);
    loop {
        let beta_a = &model.beta.0 + &(alpha * &dir);
        let eta_a = x.dot(&beta_a) + &model.offset; // η = X·β + offset (DATA-3)
        let mu_a = eta_a.mapv(|e| link.inv_link(e));
        let gd_a = global_deviance_with(family, y, prior_weights, models, param, &mu_a)?;
        let delta_pen = 2.0 * alpha * pen_cross + alpha * alpha * pen_dir;

        // Accept on no-increase of the penalized deviance.
        if gd_a - gd0 + delta_pen <= slack {
            let eta_max_change = max_abs_diff(&eta_a, &model.eta);
            return Ok(Halved {
                beta: Coefficients(beta_a),
                eta: eta_a,
                mu: mu_a,
                hits,
                rejected: false,
                eta_max_change,
            });
        }
        // At the backtracking floor the direction is uphill at every step size:
        // *reject* the block update (α = 0). Forcing a micro-step instead would
        // creep the parameter uphill every cycle: an unbounded slow divergence.
        // Rejection preserves the monotone-descent guarantee exactly; a block at
        // its optimum has a tiny full step and never reaches this branch.
        if alpha <= min_alpha {
            return Ok(Halved {
                beta: model.beta.clone(),
                eta: model.eta.clone(),
                mu: model.mu.clone(),
                hits,
                rejected: true,
                eta_max_change: 0.0,
            });
        }
        alpha *= 0.5;
        hits += 1;
    }
}

/// Run one Fisher-scoring step on `target_param`, given the current state of every
/// parameter in `models`.
///
/// `prior_weights` is an optional per-row scale applied to each observation's
/// likelihood contribution.  When `Some(w)`, the solve weight becomes
/// `w_solve_i = prior_i · safe_w_i`; the working response `z` always uses only
/// `safe_w_i` in its denominator so the IRLS step size is not distorted.
pub(super) fn step<D: Distribution + ?Sized>(
    family: &D,
    y: &Array1<f64>,
    prior_weights: Option<&Array1<f64>>,
    models: &IndexMap<String, FittingParameter>,
    target_param: &str,
    criterion: SmoothingCriterion,
) -> Result<Update, GamlssError> {
    // 1. Reference every parameter's cached μ; derivatives() expects all of them.
    //    The cache is maintained by the outer loop so we don't re-run inv_link here.
    let params_ref: HashMap<&str, &Array1<f64>> = family
        .parameters()
        .iter()
        .map(|name| (*name, &models[*name].mu))
        .collect();

    // 2. Score and Fisher info for the target parameter.
    let all_derivs = family.derivatives(y, &params_ref)?;
    let (deriv_u, deriv_w) = all_derivs
        .get(target_param)
        .ok_or_else(|| GamlssError::Input(format!("No derivation for {} found", target_param)))?;

    let target = models.get(target_param).ok_or_else(|| {
        GamlssError::Internal(format!("Model for parameter '{}' not found", target_param))
    })?;

    // 3. Working response z = η + u/w.  Floor weights and clamp the step in η units
    //    so degenerate Fisher information can't blow up the IRLS update. Count how
    //    often each guard fires so the caller can flag a degenerate fit.
    //
    //    A single `Zip::for_each` pass builds both `z` and the floored weights `w`
    //    and tallies the clamp counters — two output allocations instead of the
    //    four that mapv-chained intermediate arrays produced.
    let n = target.eta.len();
    let mut z = Array1::<f64>::zeros(n);
    let mut w = Array1::<f64>::zeros(n);
    let mut weight_floor_hits = 0usize;
    let mut step_cap_hits = 0usize;
    // When prior weights are absent, treat every row as weight 1 by using a temporary
    // ones array.  This lets the Zip stay uniform without a per-element branch.
    let ones;
    let pw: &Array1<f64> = match prior_weights {
        Some(pw) => pw,
        None => {
            ones = Array1::ones(n);
            &ones
        }
    };
    Zip::from(&target.eta)
        .and(deriv_u)
        .and(deriv_w)
        .and(pw)
        .and(&mut z)
        .and(&mut w)
        .for_each(|&eta_i, &u_i, &w_i, &prior_i, z_out, w_out| {
            let safe_w = if w_i < MIN_WEIGHT {
                weight_floor_hits += 1;
                MIN_WEIGHT
            } else {
                w_i
            };
            let step = u_i / safe_w;
            let clipped = if step > MAX_STEP {
                step_cap_hits += 1;
                MAX_STEP
            } else if step < -MAX_STEP {
                step_cap_hits += 1;
                -MAX_STEP
            } else {
                step
            };
            // z uses safe_w (not safe_w * prior_i): the IRLS step size is a property
            // of log-likelihood curvature and must not be scaled by the observation
            // weight.  Only w_solve carries the prior factor into the normal equations.
            *z_out = eta_i + clipped;
            *w_out = prior_i * safe_w;
        });

    // The solver fits X·β to z, so the fixed offset (which enters η but not β) is
    // subtracted out: the adjusted working response becomes (η + u/w) − offset =
    // X·β_old + u/w. η is reconstructed as X·β_new + offset below (DATA-3). This is
    // a no-op when there is no offset (the vector is all zeros).
    z -= &target.offset;

    // 4–5. Optimize λ (warm-started from the previous step), solve PWLS, and
    //      attribute per-term EDF. Purely parametric models (no penalties) take
    //      the zero-λ fast path inside `run_opt`.
    let penalties = &target.penalty_matrices;

    // `X`, `z`, and `w` are fixed for the rest of this function: every λ
    // evaluation below (L-BFGS/Fellner-Schall iterations, the collapse-guarded
    // restart's basin probes, the coarse grid search) re-solves the same
    // weighted normal equations at a different λ. Building them once here
    // turns each evaluation's dominant cost from O(n·p²) into O(p²)–O(p³);
    // `group_penalties` similarly caches the (λ-independent) coefficient-block
    // structure instead of re-scanning every penalty matrix on every call.
    let nfo = WeightedNormalEquations::new(&target.x_matrix, &z, &w);
    let groups = group_penalties(penalties);

    // Run the configured criterion's λ optimizer from `init`. `polish` only
    // affects the Reml branch: `true` on the adopted-λ path, `false` for cheap
    // basin-probe screening (see `run_optimization_reml`'s doc comment).
    let run_opt = |init: Option<&Array1<f64>>, polish: bool| -> Result<Array1<f64>, GamlssError> {
        if penalties.is_empty() {
            return Ok(Array1::zeros(0));
        }
        match criterion {
            SmoothingCriterion::Gcv => run_optimization(&nfo, penalties, init),
            SmoothingCriterion::Reml => {
                run_optimization_reml(&target.x_matrix, &nfo, penalties, &groups, init, polish)
            }
            SmoothingCriterion::FellnerSchall => {
                run_optimization_fellner_schall(&target.x_matrix, &nfo, penalties, &groups, init)
            }
        }
    };

    // Solve PWLS at `lambdas` and attribute EDF to each term by summing its
    // contiguous block of the per-coefficient EDF diagonal (column order matches
    // `term_layouts`).
    let fit_and_terms =
        |lambdas: &Array1<f64>| -> Result<(Coefficients, CovarianceMatrix, Vec<f64>), GamlssError> {
            let (beta, cov, _edf, edf_per_coeff) = fit_pwls(&nfo, penalties, lambdas)?;
            let mut term_edf = Vec::with_capacity(target.term_layouts.len());
            let mut offset = 0usize;
            for layout in &target.term_layouts {
                let end = offset + layout.n_coeffs;
                term_edf.push((offset..end).map(|k| edf_per_coeff[k]).sum());
                offset = end;
            }
            Ok((beta, cov, term_edf))
        };

    // A smooth term whose EDF has decayed to its penalty null-space dimension has
    // collapsed onto the unpenalized polynomial remainder (e.g. a straight line).
    let is_collapsed = |term_edf: &[f64]| -> bool {
        target
            .term_layouts
            .iter()
            .zip(term_edf)
            .any(|(l, &e)| l.is_smooth && e <= l.null_dim as f64 + super::EDF_COLLAPSE_SLACK)
    };

    // A λ pinned at (or beyond) the log-clamp bounds is the multi-penalty
    // analogue of a collapsed term: for a tensor smooth, one margin can be
    // driven to the ceiling while the term's TOTAL EDF still sits far above its
    // null-space dimension, so the EDF test alone cannot see it. A pinned λ is
    // never a genuine interior optimum: it deserves the same restart probe.
    let lambda_bounds = || -> (f64, f64) {
        (
            (super::solver::LOG_LAMBDA_CLAMP - 1e-6).exp(),
            (-super::solver::LOG_LAMBDA_CLAMP + 1e-6).exp(),
        )
    };
    let lambda_at_bound = |lambdas: &Array1<f64>| -> bool {
        let (hi, lo) = lambda_bounds();
        lambdas.iter().any(|&l| l >= hi || l <= lo)
    };

    let best_lambdas = run_opt(Some(&target.lambdas), true)?;
    let (best_lambdas, new_beta, cov_matrix, term_edf) = {
        let (beta, cov, term_edf) = fit_and_terms(&best_lambdas)?;

        // Collapse-guarded restart. The LAML/GCV objective is unimodal in λ but
        // carries a flat high-λ shelf where a smooth sits in its penalty null
        // space; BLAS reduction-order noise can occasionally tip the optimizer
        // onto that shelf (rare, nondeterministic). If the incumbent collapsed,
        // re-optimize from a low-λ seed (below the shelf) and keep whichever λ
        // has the better objective. Comparing objectives means a genuinely
        // null-space-optimal fit (a linear truth under an order-2 penalty) is
        // preserved — its collapse has the better marginal likelihood — while a
        // spuriously collapsed signal-bearing fit is repaired.
        // The trigger is the CHEAP suspicion set only: a collapsed term, or a λ
        // pinned at the log-clamp ceiling / MIN_LAMBDA floor. It re-runs every
        // cycle the state stays suspicious, deliberately: λ can reach a bound on
        // cycle 1–2 while the working (z, w) still reflect a poor scale estimate,
        // so an early probe finds nothing and a skip-once rule would never
        // re-probe at the converged state where the rescue is actually decidable.
        //
        // A prior revision ALSO probed every multi-penalty (anisotropic-tensor)
        // cycle unconditionally, to catch the one spurious shape the cheap set
        // misses: a margin's λ "merely very large" but not at a bound while the
        // term EDF sits above its null dim. That was ruinously expensive: each
        // firing runs a 7^k derivative-free grid plus several full
        // L-BFGS/Fellner-Schall optimizations, every one an eigendecomposition on
        // the term's k₁k₂ coefficient block, so a single default-10×10 tensor fit
        // ran ~30 s (OpenBLAS) to >2 min (pure-rust/nalgebra) in a debug build,
        // hanging the pre-push/CI suites, which build unoptimized and run both
        // backends. Nothing runnable guarded the payoff: the merely-large-λ
        // rescue is validated only by the `#[ignore]`d, data-gated
        // `benchmark/run_comparison.sh` mgcv sweep, so the cost was paid on every
        // tensor fit to protect a case no CI/pre-push test checks. Reverted to the
        // cheap trigger here. The ceiling/floor spurious basins (incl. the seed-9
        // "corner") are still caught by `lambda_at_bound`; only the
        // merely-large-interior sub-case is dropped, and it sits within the
        // sweep's 20–25 % EDF tolerance. Re-run the mgcv comparison before relying
        // on tensor EDF parity for a new seed.
        if !penalties.is_empty() && (is_collapsed(&term_edf) || lambda_at_bound(&best_lambdas)) {
            let cost_of = |lams: &Array1<f64>| -> Result<f64, GamlssError> {
                lambda_cost(criterion, &nfo, penalties, &groups, lams)
            };
            let mut winner = best_lambdas.clone();
            let mut winner_cost = cost_of(&winner)?;

            // Probe alternative basins and keep the best-scoring λ:
            //  1. the low-λ restart seed (below the high-λ collapse shelf);
            //  2. a fresh cold start (`initial_lambdas = None`), which explores
            //     the surface unanchored: warm-start history can pin λ in a
            //     corner basin that a cold start correctly avoids;
            //  3. for multi-penalty terms, PER-COORDINATE variants of the
            //     incumbent with each bound-pinned λ_j individually dropped to
            //     the restart level. Anisotropic tensor smooths develop corner
            //     traps where one margin's λ sits at the ceiling while the true
            //     LAML optimum has that margin interior, a geometry the
            //     all-coordinates seeds (1) and (2) can both miss because they
            //     descend into a different stationary point.
            let heur = initial_log_lambda(&target.x_matrix, penalties);
            let restart = restart_seed_from_heuristic(&heur);
            let mut seeds: Vec<Option<Array1<f64>>> = vec![Some(restart.clone()), None];
            if best_lambdas.len() > 1 {
                let (hi, lo) = lambda_bounds();
                for j in 0..best_lambdas.len() {
                    if best_lambdas[j] >= hi || best_lambdas[j] <= lo {
                        let mut s = best_lambdas.clone();
                        s[j] = restart[j];
                        seeds.push(Some(s));
                    }
                }
            }
            // For one- and two-penalty terms, additionally seed from the best
            // cell of a coarse log-λ grid around the cold-start heuristic.
            // Gradient descent from ANY single seed can fall into a spurious
            // stationary point of the multimodal anisotropic-tensor LAML
            // surface (observed: a corner with one margin at the ceiling
            // scoring 5 LAML units worse than the interior optimum, missed by
            // every gradient-started probe on some datasets); a grid evaluation
            // is derivative-free and cannot be captured by a basin boundary.
            if best_lambdas.len() <= 2 {
                let offsets: [f64; 7] = [-16.0, -12.0, -8.0, -4.0, 0.0, 4.0, 8.0];
                let n_dims = best_lambdas.len();
                let n_cells = offsets.len().pow(n_dims as u32);
                let cell_at = |idx: usize| -> Array1<f64> {
                    let mut cell = Array1::zeros(n_dims);
                    let mut rem = idx;
                    for j in 0..n_dims {
                        cell[j] = (heur[j] + offsets[rem % offsets.len()]).exp();
                        rem /= offsets.len();
                    }
                    cell
                };
                // Every grid cell's cost is independent of the others (at most
                // 7² = 49 cells), so evaluate them in parallel; picking the
                // best-scoring cell stays a serial reduction over the results.
                let cell_costs: Vec<Option<(f64, Array1<f64>)>> = {
                    #[cfg(feature = "parallel")]
                    {
                        (0..n_cells)
                            .into_par_iter()
                            .map(|idx| {
                                let cell = cell_at(idx);
                                cost_of(&cell).ok().map(|c| (c, cell))
                            })
                            .collect()
                    }
                    #[cfg(not(feature = "parallel"))]
                    {
                        (0..n_cells)
                            .map(|idx| {
                                let cell = cell_at(idx);
                                cost_of(&cell).ok().map(|c| (c, cell))
                            })
                            .collect()
                    }
                };
                let mut best_cell: Option<(f64, Array1<f64>)> = None;
                for result in cell_costs.into_iter().flatten() {
                    if best_cell.as_ref().is_none_or(|(bc, _)| result.0 < *bc) {
                        best_cell = Some(result);
                    }
                }
                if let Some((_, cell)) = best_cell {
                    seeds.push(Some(cell));
                }
            }
            // Screen candidates cheaply (no polish) and keep the best-scoring
            // one; a seed that combines an extreme pinned λ on one margin with
            // a tiny restart value on another can make the speculative solve
            // ill-conditioned, so a failed candidate is skipped rather than
            // aborting the whole fit, mirroring the grid-cell loop above. Each
            // seed's optimization is independent, so it runs in parallel too;
            // the winner-tracking reduction stays serial so a tie keeps the
            // earliest-listed seed, matching the sequential behavior.
            let seed_results: Vec<Option<(f64, Array1<f64>)>> = {
                #[cfg(feature = "parallel")]
                {
                    seeds
                        .par_iter()
                        .map(|seed| {
                            let candidate = run_opt(seed.as_ref(), false).ok()?;
                            let cost = cost_of(&candidate).ok()?;
                            Some((cost, candidate))
                        })
                        .collect()
                }
                #[cfg(not(feature = "parallel"))]
                {
                    seeds
                        .iter()
                        .map(|seed| {
                            let candidate = run_opt(seed.as_ref(), false).ok()?;
                            let cost = cost_of(&candidate).ok()?;
                            Some((cost, candidate))
                        })
                        .collect()
                }
            };
            for result in seed_results.into_iter().flatten() {
                if result.0 < winner_cost {
                    winner_cost = result.0;
                    winner = result.1;
                }
            }

            if winner.iter().zip(best_lambdas.iter()).any(|(a, b)| a != b) {
                let (rb, rc, rte) = fit_and_terms(&winner)?;
                (winner, rb, rc, rte)
            } else {
                (best_lambdas, beta, cov, term_edf)
            }
        } else {
            (best_lambdas, beta, cov, term_edf)
        }
    };

    let new_eta = target.x_matrix.dot(&new_beta.0) + &target.offset; // η = X·β + offset
    let max_diff = max_abs_diff(&new_beta.0, &target.beta.0);
    let eta_abs_diff = (&new_eta - &target.eta).mapv(f64::abs);
    let eta_max_change = eta_abs_diff.iter().copied().fold(0.0_f64, f64::max);
    let eta_change = eta_abs_diff.sum();
    let lambda_change = (&best_lambdas - &target.lambdas).mapv(f64::abs).sum();

    Ok(Update {
        beta: new_beta,
        eta: new_eta,
        lambdas: best_lambdas,
        covariance: cov_matrix,
        term_edf,
        max_diff,
        eta_max_change,
        eta_change,
        lambda_change,
        weight_floor_hits,
        step_cap_hits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::{Gaussian, IdentityLink, Link, LogLink};
    use crate::fitting::assembler::TermLayout;
    use crate::fitting::FittingParameter;
    use crate::terms::Term;
    use crate::types::ModelMatrix;
    use ndarray::{array, Array2};

    /// Layout for a single unpenalized term occupying `n_coeffs` columns.
    fn parametric_layout(n_coeffs: usize) -> Vec<TermLayout> {
        vec![TermLayout {
            n_coeffs,
            null_dim: 0,
            is_smooth: false,
        }]
    }

    fn intercept_only(eta_init: f64, n: usize) -> FittingParameter {
        let x = ModelMatrix(Array2::ones((n, 1)));
        let link = IdentityLink;
        let eta = Array1::from_elem(n, eta_init);
        let mu = eta.mapv(|e| link.inv_link(e));
        FittingParameter {
            terms: vec![Term::Intercept],
            term_layouts: parametric_layout(1),
            link: Box::new(link),
            x_matrix: x,
            penalty_matrices: vec![],
            beta: Coefficients(array![eta_init]),
            eta,
            mu,
            offset: Array1::zeros(n),
            lambdas: Array1::<f64>::zeros(0),
            covariance: None,
            term_edf: vec![0.0],
        }
    }

    fn intercept_only_log(eta_init: f64, n: usize) -> FittingParameter {
        let x = ModelMatrix(Array2::ones((n, 1)));
        let link = LogLink;
        let eta = Array1::from_elem(n, eta_init);
        let mu = eta.mapv(|e| link.inv_link(e));
        FittingParameter {
            terms: vec![Term::Intercept],
            term_layouts: parametric_layout(1),
            link: Box::new(link),
            x_matrix: x,
            penalty_matrices: vec![],
            beta: Coefficients(array![eta_init]),
            eta,
            mu,
            offset: Array1::zeros(n),
            lambdas: Array1::<f64>::zeros(0),
            covariance: None,
            term_edf: vec![0.0],
        }
    }

    #[test]
    fn gaussian_intercept_only_step_moves_mean_toward_y_bar() {
        // For a Gaussian intercept-only model with identity link on μ, one Fisher-scoring
        // step from μ=0 should move β toward ȳ.
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0]; // ȳ = 3
        let n = y.len();
        let mut models = IndexMap::new();
        models.insert("mu".to_string(), intercept_only(0.0, n));
        models.insert("sigma".to_string(), intercept_only_log(0.0, n)); // σ = 1

        let update = step(&Gaussian, &y, None, &models, "mu", SmoothingCriterion::Gcv).unwrap();
        assert!(
            (update.beta.0[0] - 3.0).abs() < 1e-6,
            "expected β ≈ 3.0 (ȳ), got {}",
            update.beta.0[0]
        );
        assert_eq!(update.lambdas.len(), 0);
    }

    #[test]
    fn step_returns_finite_diagnostics() {
        let y = array![1.0, 2.0, 3.0];
        let n = y.len();
        let mut models = IndexMap::new();
        models.insert("mu".to_string(), intercept_only(0.0, n));
        models.insert("sigma".to_string(), intercept_only_log(0.0, n));

        let update = step(&Gaussian, &y, None, &models, "mu", SmoothingCriterion::Gcv).unwrap();
        assert!(update.max_diff.is_finite() && update.max_diff > 0.0);
        assert!(update.eta_change.is_finite() && update.eta_change > 0.0);
        assert!(update.lambda_change.is_finite());
        assert!(update.edf().is_finite());
    }

    #[test]
    fn step_does_not_mutate_models_map() {
        // step() takes &HashMap; ensure caller's state is unchanged after the call.
        let y = array![1.0, 2.0, 3.0];
        let n = y.len();
        let mut models = IndexMap::new();
        models.insert("mu".to_string(), intercept_only(0.0, n));
        models.insert("sigma".to_string(), intercept_only_log(0.0, n));

        let beta_before = models["mu"].beta.0.clone();
        let _ = step(&Gaussian, &y, None, &models, "mu", SmoothingCriterion::Gcv).unwrap();
        let beta_after = &models["mu"].beta.0;
        assert_eq!(beta_before, *beta_after);
    }

    #[test]
    fn step_with_pspline_runs_gcv_optimizer() {
        // P-spline term carries one penalty matrix → step must take the run_optimization
        // branch and return a single positive λ.
        use crate::splines::{create_basis_matrix, create_penalty_matrix};
        use crate::types::PenaltyMatrix;

        let n = 30;
        let x = Array1::from_iter((0..n).map(|i| i as f64 / (n - 1) as f64));
        let y = x.mapv(|v| v.sin() * 2.0 + 1.0);

        let n_splines = 8;
        let basis = create_basis_matrix(&x, n_splines, 3);
        let penalty = create_penalty_matrix(n_splines, 2);

        let mu = FittingParameter {
            terms: vec![Term::Intercept],
            // The design matrix here is a bare spline basis (n_splines columns);
            // describe it as one smooth block so per-term EDF slicing matches.
            term_layouts: vec![TermLayout {
                n_coeffs: n_splines,
                null_dim: 2,
                is_smooth: true,
            }],
            link: Box::new(IdentityLink),
            x_matrix: ModelMatrix(basis),
            penalty_matrices: vec![PenaltyMatrix {
                offset: 0,
                block: penalty,
            }],
            beta: Coefficients(Array1::zeros(n_splines)),
            eta: Array1::from_elem(n, 0.0),
            // μ = inv_link(η) = η for IdentityLink.
            mu: Array1::from_elem(n, 0.0),
            offset: Array1::zeros(n),
            lambdas: Array1::ones(1),
            covariance: None,
            term_edf: vec![0.0],
        };
        let sigma = intercept_only_log(0.0, n);

        let mut models = IndexMap::new();
        models.insert("mu".to_string(), mu);
        models.insert("sigma".to_string(), sigma);

        let update = step(&Gaussian, &y, None, &models, "mu", SmoothingCriterion::Gcv).unwrap();
        assert_eq!(update.lambdas.len(), 1);
        assert!(update.lambdas[0].is_finite() && update.lambdas[0] > 0.0);
        assert!(update.edf() > 0.0 && update.edf() <= n_splines as f64);
        assert!(update.beta.0.iter().all(|b| b.is_finite()));
    }

    #[test]
    fn collapse_guarded_restart_recovers_from_shelf_warm_start() {
        // A signal-bearing sine on a P-spline, but warm-started with a huge λ that
        // sits on the high-λ "collapse shelf" (smooth pinned to its null space).
        // The collapse-guarded restart must detect the collapse, re-optimize from
        // a low-λ seed, and recover a genuinely curved fit (edf well above the
        // null-space dimension) with a far smaller λ.
        use crate::splines::{create_basis_matrix, create_penalty_matrix};
        use crate::types::PenaltyMatrix;

        let n = 400;
        let x = Array1::from_iter((0..n).map(|i| i as f64 / (n - 1) as f64));
        let y = x.mapv(|v| (2.0 * std::f64::consts::PI * v).sin());

        let n_splines = 12;
        let basis = create_basis_matrix(&x, n_splines, 3);
        let penalty = create_penalty_matrix(n_splines, 2);

        let mu = FittingParameter {
            terms: vec![Term::Intercept],
            term_layouts: vec![TermLayout {
                n_coeffs: n_splines,
                null_dim: 2,
                is_smooth: true,
            }],
            link: Box::new(IdentityLink),
            x_matrix: ModelMatrix(basis),
            penalty_matrices: vec![PenaltyMatrix {
                offset: 0,
                block: penalty,
            }],
            beta: Coefficients(Array1::zeros(n_splines)),
            eta: Array1::from_elem(n, 0.0),
            mu: Array1::from_elem(n, 0.0),
            offset: Array1::zeros(n),
            // Collapsed warm start: λ on the shelf.
            lambdas: Array1::from_elem(1, 1e12),
            covariance: None,
            term_edf: vec![0.0],
        };
        let sigma = intercept_only_log(0.0, n);

        let mut models = IndexMap::new();
        models.insert("mu".to_string(), mu);
        models.insert("sigma".to_string(), sigma);

        let update = step(&Gaussian, &y, None, &models, "mu", SmoothingCriterion::Reml).unwrap();

        assert!(
            update.term_edf[0] > 3.0,
            "guard should have recovered a curved fit, got smooth edf = {}",
            update.term_edf[0]
        );
        assert!(
            update.lambdas[0] < 1e6,
            "guard should have moved λ off the shelf, got λ = {}",
            update.lambdas[0]
        );
    }

    #[test]
    fn step_errors_when_target_not_in_family_parameters() {
        let y = array![1.0, 2.0];
        let n = y.len();
        let mut models = IndexMap::new();
        models.insert("mu".to_string(), intercept_only(0.0, n));
        models.insert("sigma".to_string(), intercept_only_log(0.0, n));

        let err = step(
            &Gaussian,
            &y,
            None,
            &models,
            "zeta",
            SmoothingCriterion::Gcv,
        )
        .unwrap_err();
        // family.derivatives() never produces a "zeta" entry, so we hit the missing-derivative arm.
        assert!(format!("{}", err).contains("zeta"));
    }

    /// Minimal `Update` carrying an arbitrary proposed β; `step_halving` only reads
    /// `proposed.beta`, recomputing η/μ from the trial α itself.
    fn proposed_update(beta_val: f64) -> Update {
        Update {
            beta: Coefficients(array![beta_val]),
            eta: array![beta_val],
            lambdas: Array1::<f64>::zeros(0),
            covariance: CovarianceMatrix(Array2::zeros((1, 1))),
            term_edf: vec![0.0],
            max_diff: 0.0,
            eta_max_change: 0.0,
            eta_change: 0.0,
            lambda_change: 0.0,
            weight_floor_hits: 0,
            step_cap_hits: 0,
        }
    }

    #[test]
    fn global_deviance_matches_minus_two_loglik() {
        // Gaussian intercept-only, μ = 0, σ = 1 ⇒ GD = Σ (y² + ln(2π)).
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let n = y.len();
        let mut models = IndexMap::new();
        models.insert("mu".to_string(), intercept_only(0.0, n));
        models.insert("sigma".to_string(), intercept_only_log(0.0, n));

        let gd = global_deviance(&Gaussian, &y, None, &models).unwrap();
        let expected: f64 = y
            .iter()
            .map(|&yi| yi * yi + (2.0 * std::f64::consts::PI).ln())
            .sum();
        assert!((gd - expected).abs() < 1e-9, "gd {gd} vs {expected}");
    }

    #[test]
    fn global_deviance_with_overrides_one_block() {
        // Swapping in the committed μ reproduces the plain global deviance.
        let y = array![1.0, 2.0, 3.0];
        let n = y.len();
        let mut models = IndexMap::new();
        models.insert("mu".to_string(), intercept_only(0.0, n));
        models.insert("sigma".to_string(), intercept_only_log(0.0, n));

        let plain = global_deviance(&Gaussian, &y, None, &models).unwrap();
        let same =
            global_deviance_with(&Gaussian, &y, None, &models, "mu", &models["mu"].mu).unwrap();
        assert!((plain - same).abs() < 1e-12);
    }

    #[test]
    fn step_halving_accepts_full_step_when_deviance_decreases() {
        // Moving μ from 0 toward ȳ = 3 strictly lowers the deviance, so the full
        // step (α = 1) is accepted with no halvings.
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let n = y.len();
        let mut models = IndexMap::new();
        models.insert("mu".to_string(), intercept_only(0.0, n));
        models.insert("sigma".to_string(), intercept_only_log(0.0, n));

        let proposed = proposed_update(3.0);
        let halved = step_halving(
            &Gaussian,
            &y,
            None,
            &models,
            "mu",
            &proposed,
            MIN_STEP_ALPHA,
        )
        .unwrap();
        assert_eq!(halved.hits, 0, "well-behaved step should not halve");
        assert!((halved.beta.0[0] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn step_halving_damps_an_overshooting_step() {
        // ȳ = 3; a proposed μ = 10 overshoots so far the deviance *increases*
        // (Σ(y−10)² = 255 > Σy² = 55). Halving once lands on μ = 5
        // (Σ(y−5)² = 30 < 55), a strict decrease — so hits ≥ 1 and the accepted
        // deviance is below the starting deviance.
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let n = y.len();
        let mut models = IndexMap::new();
        models.insert("mu".to_string(), intercept_only(0.0, n));
        models.insert("sigma".to_string(), intercept_only_log(0.0, n));

        let gd0 = global_deviance(&Gaussian, &y, None, &models).unwrap();
        let proposed = proposed_update(10.0);
        let halved = step_halving(
            &Gaussian,
            &y,
            None,
            &models,
            "mu",
            &proposed,
            MIN_STEP_ALPHA,
        )
        .unwrap();

        assert!(halved.hits >= 1, "overshooting step must be halved");
        assert!(
            halved.beta.0[0] < 10.0 && halved.beta.0[0] > 0.0,
            "accepted β should be a damped fraction of the full step, got {}",
            halved.beta.0[0]
        );
        let gd_accepted =
            global_deviance_with(&Gaussian, &y, None, &models, "mu", &halved.mu).unwrap();
        assert!(
            gd_accepted <= gd0 + 1e-8,
            "accepted deviance {gd_accepted} should not exceed start {gd0}"
        );
    }
}
