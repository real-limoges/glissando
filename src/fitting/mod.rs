//! GAMLSS fitting implementation via the Rigby-Stasinopoulos (RS) algorithm.
//!
//! The RS algorithm iteratively cycles through distribution parameters, fitting each as a
//! penalized additive model while holding others fixed. For each parameter:
//!
//! 1. Compute score (u) and Fisher information (w) from the distribution
//! 2. Form working response: z = η + u/w
//! 3. Optimize smoothing parameters (λ) via the configured criterion (REML default)
//! 4. Solve penalized weighted least squares: (X'WX + Σλ·S)·β = X'W·z
//! 5. Update linear predictor: η = X·β
//!
//! The module also handles posterior inference (sampling from the approximate posterior of coefficients).

pub(crate) mod assembler;
pub mod diagnostics;
pub mod mixture;
mod scoring;
pub mod selection;
mod solver;

use self::assembler::{assemble_model_matrices, resolve_terms, AssembledDesign};

use super::distributions::{link_from_name, Distribution, Link};
use super::error::GamlssError;
use super::terms::{Smooth, Term};
use super::types::*;
use crate::linalg;
use indexmap::IndexMap;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{rng, Rng, SeedableRng};
use rand_distr::{Distribution as _, StandardNormal};
use std::collections::HashMap;

const DEFAULT_MAX_ITER: usize = 200;
const DEFAULT_TOLERANCE: f64 = 1e-3;
/// Default absolute tolerance on the global-deviance change (FIT-2).
const DEFAULT_GD_TOLERANCE: f64 = 1e-3;

/// How close a smooth term's EDF must sit to its penalty null-space dimension
/// before the fit flags it as collapsed. Half an effective degree of freedom of
/// slack keeps a genuinely (near-)linear fit from tripping the warning while
/// still catching a smooth that has been fully penalized away.
const EDF_COLLAPSE_SLACK: f64 = 0.5;

/// Smoothing-parameter selection criterion.
///
/// `Reml` (the default) minimizes the Laplace-approximate marginal likelihood
/// (Wood 2011) via L-BFGS, applied per distributional parameter to its converged
/// PWLS subproblem. `Gcv` uses Generalized Cross-Validation (Craven & Wahba 1979).
/// `FellnerSchall` optimizes the same target as `Reml` via the multiplicative
/// fixed-point update of Wood & Fasiolo (2017) — deterministic, no line search.
/// REML tends to be less prone to local minima and undersmoothing than GCV at
/// moderate sample sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum SmoothingCriterion {
    Gcv,
    #[default]
    Reml,
    /// Fellner-Schall fixed-point optimizer for the LAML target
    /// (Wood & Fasiolo 2017). Same objective as `Reml`, deterministic update;
    /// no outer L-BFGS, no line search.
    FellnerSchall,
}

/// How the fitter treats rows carrying a missing (non-finite) value in the
/// response or any formula-referenced column (DATA-4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum NaAction {
    /// Drop any row with a missing value in `y` or a referenced column before
    /// fitting — R's `na.omit`, and the default. Weights and every model column
    /// are masked together so the design, working response, and weights stay
    /// aligned.
    #[default]
    DropRows,
    /// Reject the fit with [`GamlssError`](crate::GamlssError) if any model
    /// variable is non-finite (the historical behaviour). Use when a missing
    /// value should be a hard error rather than silently dropped.
    Fail,
}

/// Configuration options for the GAMLSS fitting algorithm.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FitConfig {
    /// Maximum number of RS algorithm iterations (default: 200).
    #[cfg_attr(feature = "serde", serde(default = "default_max_iter"))]
    pub max_iterations: usize,
    /// Convergence tolerance for relative coefficient changes (default: 1e-3).
    #[cfg_attr(feature = "serde", serde(default = "default_tolerance"))]
    pub tolerance: f64,
    /// Smoothing-parameter selection criterion (default: REML).
    #[cfg_attr(feature = "serde", serde(default))]
    pub criterion: SmoothingCriterion,
    /// Whether to step-halve (line-search on the global deviance) each accepted
    /// Fisher-scoring update so every cycle is a monotone descent (FIT-1).
    /// On by default; matches R `gamlss`'s RS loop. Disabling it recovers the
    /// raw (unguarded) full-step behaviour.
    #[cfg_attr(feature = "serde", serde(default = "default_true"))]
    pub step_halving: bool,
    /// Absolute tolerance on the global-deviance change between cycles (FIT-2),
    /// in deviance units: the same convention as R gamlss's `c.crit`.
    /// Convergence requires *both* this and the Δβ `tolerance` test to pass.
    /// Default: 1e-3.
    #[cfg_attr(feature = "serde", serde(default = "default_gd_tolerance"))]
    pub gd_tolerance: f64,
    /// Per-parameter link overrides, keyed by distribution-parameter name
    /// (e.g. `"mu" → "probit"`). Empty (the default) uses each family's
    /// [`default_link`](crate::distributions::Distribution::default_link). Names are
    /// validated against [`link_from_name`](crate::distributions::link_from_name) at
    /// fit time, so an unknown link yields [`GamlssError::Input`]. Build ergonomically
    /// with [`with_link`](FitConfig::with_link) / [`with_links`](FitConfig::with_links).
    #[cfg_attr(feature = "serde", serde(default))]
    pub links: IndexMap<String, String>,
    /// How to treat rows with a missing (non-finite) value in `y` or a referenced
    /// column. Default: [`NaAction::DropRows`] (R's `na.omit`). Set to
    /// [`NaAction::Fail`] to reject such inputs instead.
    #[cfg_attr(feature = "serde", serde(default))]
    pub na_action: NaAction,
}

impl FitConfig {
    /// Choose how missing values are handled, returning `self` for chaining:
    /// `FitConfig::default().with_na_action(NaAction::Fail)`.
    #[must_use]
    pub fn with_na_action(mut self, na_action: NaAction) -> Self {
        self.na_action = na_action;
        self
    }

    /// Override the link for a single distribution parameter, returning `self` for
    /// chaining: `FitConfig::default().with_link("mu", "probit")`.
    #[must_use]
    pub fn with_link(mut self, param: impl Into<String>, link: impl Into<String>) -> Self {
        self.links.insert(param.into(), link.into());
        self
    }

    /// Override the links for several parameters at once, returning `self` for chaining.
    #[must_use]
    pub fn with_links<K: Into<String>, V: Into<String>>(
        mut self,
        links: impl IntoIterator<Item = (K, V)>,
    ) -> Self {
        self.links
            .extend(links.into_iter().map(|(k, v)| (k.into(), v.into())));
        self
    }
}

#[cfg(feature = "serde")]
fn default_max_iter() -> usize {
    DEFAULT_MAX_ITER
}
#[cfg(feature = "serde")]
fn default_tolerance() -> f64 {
    DEFAULT_TOLERANCE
}
#[cfg(feature = "serde")]
fn default_true() -> bool {
    true
}
#[cfg(feature = "serde")]
fn default_gd_tolerance() -> f64 {
    DEFAULT_GD_TOLERANCE
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            max_iterations: DEFAULT_MAX_ITER,
            tolerance: DEFAULT_TOLERANCE,
            criterion: SmoothingCriterion::default(),
            step_halving: true,
            gd_tolerance: DEFAULT_GD_TOLERANCE,
            links: IndexMap::new(),
            na_action: NaAction::default(),
        }
    }
}

/// Diagnostic information from the model fitting process.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FitDiagnostics {
    /// Whether the algorithm converged within the maximum iterations.
    pub converged: bool,
    /// Number of RS algorithm iterations performed.
    pub iterations: usize,
    /// Maximum coefficient change in the final iteration.
    pub final_change: f64,
    /// Maximum gradient at convergence (if computed).
    pub max_gradient: Option<f64>,
    /// Per-parameter diagnostic information, ordered by `family.parameters()`.
    pub param_diagnostics: IndexMap<String, ParamDiagnostic>,
    /// Non-fatal fit-quality warnings (e.g. a smooth term that collapsed onto
    /// its penalty null space). Empty on a clean fit. Serialized with the model
    /// so JSON/FFI consumers can surface fit health.
    #[cfg_attr(feature = "serde", serde(default))]
    pub warnings: Vec<String>,
    /// Global deviance `−2·ℓ̂` at the final cycle (FIT-2). `None` only if no
    /// cycle ran (e.g. `max_iterations == 0`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub final_deviance: Option<f64>,
    /// Absolute global-deviance change at the final cycle,
    /// `|GD_{c−1} − GD_c|` (FIT-2; same units as gamlss's `c.crit`). `None` on
    /// the first cycle (no previous deviance) or if no cycle ran.
    #[cfg_attr(feature = "serde", serde(default))]
    pub final_deviance_change: Option<f64>,
}

/// Diagnostic information for a single distribution parameter.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ParamDiagnostic {
    pub final_eta_change: f64,
    pub final_lambda_change: f64,
    pub edf: f64,
    /// Non-zero suggests degenerate Fisher info (extreme score, vanishing curvature).
    pub weight_floor_hits: usize,
    /// Non-zero means IRLS steps were damped; persistent at convergence indicates trouble.
    pub step_cap_hits: usize,
    /// Number of global-deviance step-halvings applied to this parameter on the
    /// final cycle (FIT-1). Zero when the full Fisher step was accepted as-is or
    /// when `step_halving` is disabled.
    #[cfg_attr(feature = "serde", serde(default))]
    pub step_halving_hits: usize,
}

/// Fitted results for a single distribution parameter (e.g., mu, sigma).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FittedParameter {
    pub coefficients: Coefficients,
    pub covariance: CovarianceMatrix,
    pub terms: Vec<Term>,
    pub lambdas: Array1<f64>,
    pub eta: Array1<f64>,
    pub fitted_values: Array1<f64>,
    pub edf: f64,
    /// Per-term EDF aligned with `terms`, summing to `edf`. An entry at the
    /// term's penalty null-space dimension means the smooth collapsed to its
    /// unpenalized polynomial remainder — see `FitDiagnostics::warnings`.
    pub term_edf: Vec<f64>,
    /// `(term_name, first_col, last_col_exclusive)` per term, aligned with
    /// `terms`. Column order matches `coefficients` and the design matrix.
    #[cfg_attr(feature = "serde", serde(default))]
    pub term_blocks: Vec<(String, usize, usize)>,
    /// Canonical name of an *overridden* link (see
    /// [`FitConfig::links`]), or `None` when the family default was used. `None`
    /// re-derives the link via `default_link()` at predict time, so models fitted
    /// before this field existed deserialize unchanged.
    #[cfg_attr(feature = "serde", serde(default))]
    pub link: Option<String>,
}

pub(super) struct FittingParameter {
    pub(super) terms: Vec<Term>,
    /// Per-term layout (coefficient counts + penalty null-space dims), aligned
    /// with `terms`. Used to attribute EDF per term and detect null-space collapse.
    pub(super) term_layouts: Vec<assembler::TermLayout>,
    pub(super) link: Box<dyn Link>,
    pub(super) x_matrix: ModelMatrix,
    pub(super) penalty_matrices: Vec<PenaltyMatrix>,
    pub(super) beta: Coefficients,
    pub(super) eta: Array1<f64>,
    /// Cached link⁻¹(η), kept in lockstep with `eta` to avoid K length-n
    /// `inv_link` passes per Fisher-scoring step.
    pub(super) mu: Array1<f64>,
    /// Fixed per-row offset entering the linear predictor as `η = X·β + offset`
    /// (DATA-3). All-zeros unless the parameter's formula carries a
    /// [`Term::Offset`](crate::Term::Offset). The PWLS solver stays offset-unaware:
    /// the working response it sees is `z − offset`, and `η` is reconstructed as
    /// `X·β + offset` afterwards.
    pub(super) offset: Array1<f64>,
    pub(super) lambdas: Array1<f64>,
    pub(super) covariance: Option<CovarianceMatrix>,
    pub(super) edf: f64,
    /// Latest per-term EDF (aligned with `terms`); updated each RS cycle.
    pub(super) term_edf: Vec<f64>,
}

/// (Prior-weighted) global deviance of the current fit:
/// `GD(θ) = −2·Σᵢ wᵢ·log f(yᵢ | θᵢ)`.
///
/// This is the objective the Rigby–Stasinopoulos loop implicitly minimizes. It
/// is the shared substrate for step-halving (FIT-1) and the global-deviance
/// convergence test (FIT-2): each `FittingParameter.mu` already holds the
/// response-scale parameter, so the helper just assembles the params view and
/// calls the family's pointwise log-density.
pub(super) fn global_deviance<D: Distribution + ?Sized>(
    family: &D,
    y: &Array1<f64>,
    prior_weights: Option<&Array1<f64>>,
    models: &IndexMap<String, FittingParameter>,
) -> Result<f64, GamlssError> {
    deviance(family, y, prior_weights, models, None)
}

/// Same as [`global_deviance`], but evaluating one parameter block at a *proposed*
/// `mu_override` not yet committed to `models` — used by step-halving to score a
/// trial move before accepting it.
pub(super) fn global_deviance_with<D: Distribution + ?Sized>(
    family: &D,
    y: &Array1<f64>,
    prior_weights: Option<&Array1<f64>>,
    models: &IndexMap<String, FittingParameter>,
    param: &str,
    mu_override: &Array1<f64>,
) -> Result<f64, GamlssError> {
    deviance(family, y, prior_weights, models, Some((param, mu_override)))
}

/// Largest element-wise absolute difference `max|aᵢ − bᵢ|`, the convergence
/// yardstick used for both η and β moves. Returns `0.0` for empty inputs.
pub(super) fn max_abs_diff(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    (a - b)
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max)
}

/// Shared body of [`global_deviance`] / [`global_deviance_with`]: assemble the
/// params view from `models`, optionally swapping in a trial block, and scale the
/// (prior-weighted) log-likelihood by `−2`.
fn deviance<'a, D: Distribution + ?Sized>(
    family: &D,
    y: &Array1<f64>,
    prior_weights: Option<&Array1<f64>>,
    models: &'a IndexMap<String, FittingParameter>,
    override_: Option<(&'a str, &'a Array1<f64>)>,
) -> Result<f64, GamlssError> {
    let mut params: HashMap<&str, &Array1<f64>> =
        models.iter().map(|(k, m)| (k.as_str(), &m.mu)).collect();
    if let Some((param, mu_override)) = override_ {
        params.insert(param, mu_override);
    }
    let ll_pt = family.loglik_pointwise(y, &params)?;
    let ll = match prior_weights {
        Some(w) => (&ll_pt * w).sum(),
        None => ll_pt.sum(),
    };
    Ok(-2.0 * ll)
}

pub(crate) fn fit_gamlss<D: Distribution + ?Sized>(
    data: &DataSet,
    y: &Array1<f64>,
    prior_weights: Option<&Array1<f64>>,
    formula: &Formula,
    family: &D,
    config: &FitConfig,
) -> Result<(IndexMap<String, FittedParameter>, FitDiagnostics), GamlssError> {
    let n_obs = y.len();
    // `IndexMap` keeps insertion order = `family.parameters()` order so the
    // resulting `GamlssModel.models` iterates deterministically downstream.
    let mut models: IndexMap<String, FittingParameter> = IndexMap::new();

    for param_name in family.parameters() {
        let param_name_str = param_name.to_string();
        let formula_terms = formula.get(&param_name_str).ok_or_else(|| {
            GamlssError::Input(format!("Formula missing for parameter {}", param_name))
        })?;
        // Resolve CrSpline1D knots once from training data so they are stored in
        // FittedParameter::terms and replayed verbatim at predict time.
        let terms = resolve_terms(formula_terms, data)?;
        // Honor a per-parameter link override from the config; otherwise the
        // family's canonical default. The chosen link name is persisted into the
        // FittedParameter so predict reconstructs the *same* link.
        let link = match config.links.get(&param_name_str) {
            Some(name) => link_from_name(name)?,
            None => family.default_link(param_name)?,
        };

        let AssembledDesign {
            x: x_model,
            penalties: penalty_matrices,
            n_coeffs: total_coeffs,
            layouts: term_layouts,
            offset,
        } = assemble_model_matrices(data, n_obs, &terms)?;

        let response_scale_start = family.initial_value(param_name, y);
        let eta_start = link.link(response_scale_start);

        // Seed the intercept coefficient so the first IRLS step starts near
        // η = link(initial μ). `beta[0]` is only the intercept when the leading
        // term is `Term::Intercept`; for a smooth-only or leading-Linear formula
        // we leave β = 0 and η = X·β, which IRLS will move from there. The fixed
        // `offset` is always added: η = X·β + offset (DATA-3).
        let mut beta = Coefficients(Array1::zeros(total_coeffs));
        let intercept_leads = matches!(terms.first(), Some(Term::Intercept));
        let eta = if intercept_leads && total_coeffs > 0 {
            beta.0[0] = eta_start;
            Array1::from_elem(n_obs, eta_start) + &offset
        } else {
            offset.clone()
        };
        let mu = eta.mapv(|e| link.inv_link(e));
        let lambdas = if penalty_matrices.is_empty() {
            Array1::zeros(0)
        } else {
            // Seed from the trace-ratio heuristic so the first REML/F-S step
            // starts with a well-conditioned X'WX + S_lambda. lambda=1 can be
            // too small for high-cardinality bases (e.g. k=20) or models with
            // prior weights, leaving the system near-singular on the first call.
            solver::initial_log_lambda(&x_model, &penalty_matrices).mapv(f64::exp)
        };

        let n_terms = term_layouts.len();
        models.insert(
            param_name_str,
            FittingParameter {
                terms, // already owned (resolved from training data above)
                term_layouts,
                link,
                x_matrix: x_model,
                penalty_matrices,
                beta,
                eta,
                mu,
                offset,
                lambdas,
                covariance: None,
                edf: 0.0,
                term_edf: vec![0.0; n_terms],
            },
        );
    }

    let mut converged = false;
    let mut final_iteration = 0;
    let mut final_change = f64::MAX;
    let mut param_diagnostics: IndexMap<String, ParamDiagnostic> = IndexMap::new();

    // FIT-2: track the global deviance across cycles so convergence can be judged
    // on objective improvement, not just coefficient movement.
    let mut gd_prev = f64::INFINITY;
    let mut final_deviance: Option<f64> = None;
    let mut final_deviance_change: Option<f64> = None;

    for cycle in 0..config.max_iterations {
        param_diagnostics.clear();
        let mut max_diff = 0.0_f64; // kept for FitDiagnostics.final_change
        let mut all_converged = true;

        for param_name in family.parameters() {
            let update = scoring::step(
                family,
                y,
                prior_weights,
                &models,
                param_name,
                config.criterion,
            )?;
            if update.max_diff > max_diff {
                max_diff = update.max_diff;
            }

            // FIT-1: backtrack the proposed block update on the global deviance so
            // the accepted step never increases it (monotone descent). The full
            // step is the α = 1 case, so a well-behaved fit pays no halvings.
            let accepted = if config.step_halving {
                scoring::step_halving(
                    family,
                    y,
                    prior_weights,
                    &models,
                    param_name,
                    &update,
                    scoring::MIN_STEP_ALPHA,
                )?
            } else {
                // No step-halving: nothing else bounds the accepted step, so
                // scale the raw Fisher step back to at most `MAX_STEP_NO_HALVING`
                // per element in η (see that constant's doc comment) instead of
                // relying on scoring::step's internal MAX_STEP clamp, which is
                // now far too loose (1e6) to serve alone. Scaling β (not η
                // directly) keeps η = X·β + offset exact.
                let pre_model = &models[*param_name];
                let raw_max_change = update.eta_max_change;
                // scale == 1.0 reproduces the full-step proposal exactly, so a
                // single unconditional construction covers both the clamped
                // and unclamped cases.
                let scale = if raw_max_change > scoring::MAX_STEP_NO_HALVING {
                    scoring::MAX_STEP_NO_HALVING / raw_max_change
                } else {
                    1.0
                };
                let dir = &update.beta.0 - &pre_model.beta.0;
                let beta = &pre_model.beta.0 + &(scale * &dir);
                let eta = pre_model.x_matrix.0.dot(&beta) + &pre_model.offset;
                let mu = eta.mapv(|e| pre_model.link.inv_link(e));
                let eta_max_change = max_abs_diff(&eta, &pre_model.eta);
                scoring::Halved {
                    beta: Coefficients(beta),
                    eta,
                    mu,
                    hits: 0,
                    rejected: false,
                    eta_max_change,
                }
            };

            // Per-parameter relative convergence check, in FIT SPACE (η = X·β).
            //
            // A coefficient-space (Δβ) test is vulnerable to false negatives on
            // penalized models: when the smoothing objective has a flat valley,
            // the per-cycle λ re-optimization can jitter between fit-equivalent
            // (λ, β) pairs whose linear predictors are identical: β moves along
            // a fit-irrelevant ridge forever and Δβ never passes, even though
            // the model (η, μ, deviance) is fully stationary. Measuring the
            // change of η instead is invariant to such ridges; combined with the
            // global-deviance test below this is strictly stronger than gamlss's
            // deviance-only criterion.
            //
            // Each parameter is checked against its own |η| scale, with a floor
            // of 1.0 so the test is equivalent to an absolute threshold when the
            // linear predictor is O(1). The test uses `accepted.eta_max_change`
            // (what step-halving actually applied), not the full-step proposal:
            // as the fit approaches the optimum the score → 0, so the full step
            // → 0 and step-halving accepts α = 1, at which point the two
            // coincide. They diverge only when the whole block update was
            // rejected (α = 0, model frozen); using the proposal there would
            // keep re-flagging "still moving" for a state that hasn't changed
            // since the first rejection, burning the iteration budget on an
            // identical re-derived-and-rejected step every cycle. Far from the
            // optimum a large accepted step keeps this test conservative, which
            // is exactly when the GD test below is the one that should decide.
            let param_eta_scale = update
                .eta
                .iter()
                .copied()
                .map(f64::abs)
                .fold(1.0_f64, f64::max);
            if accepted.eta_max_change / param_eta_scale >= config.tolerance {
                all_converged = false;
            }

            param_diagnostics.insert(
                param_name.to_string(),
                ParamDiagnostic {
                    final_eta_change: update.eta_change,
                    final_lambda_change: update.lambda_change,
                    edf: update.edf,
                    weight_floor_hits: update.weight_floor_hits,
                    step_cap_hits: update.step_cap_hits,
                    step_halving_hits: accepted.hits,
                },
            );

            // Infallible: `models` was populated from this same `family.parameters()`
            // list above and nothing removes entries (see `&models[*param_name]`).
            let model = &mut models[*param_name];
            // β/η/μ come from the accepted (possibly damped) step; covariance /
            // EDF / λ keep the values `scoring::step` computed at the full step.
            // Near convergence α → 1 so those are evaluated at the right point —
            // matching how gamlss reports them at the converged step.
            //
            // When the whole block update was REJECTED (uphill at every α), the
            // previous λ/covariance/EDF are kept too: the proposal's values
            // describe a state that was never entered, and installing them
            // would pair the old β with a covariance/EDF evaluated elsewhere,
            // corrupting SEs, GAIC, and the collapse warnings. The only
            // exception is the very first cycle, where there is no previous
            // covariance yet; the proposal's is then the best available.
            model.beta = accepted.beta;
            model.eta = accepted.eta;
            model.mu = accepted.mu;
            if !accepted.rejected || model.covariance.is_none() {
                model.lambdas = update.lambdas;
                model.covariance = Some(update.covariance);
                model.edf = update.edf;
                model.term_edf = update.term_edf;
            }
        }

        // FIT-2: global-deviance change after the full sweep. Require *both* the
        // Δβ test and the GD test to agree before declaring convergence.
        //
        // The change is measured in ABSOLUTE deviance units, matching R gamlss's
        // `c.crit` (default 0.001). A relative test (|ΔGD|/|GD|) was used
        // previously, but its slack scales with the deviance magnitude: at
        // GD ≈ 4000 it declared convergence while the fit was still improving
        // by ~4 deviance units per cycle, far short of the optimum that gamlss
        // (absolute criterion) reaches on the same data.
        let gd = global_deviance(family, y, prior_weights, &models)?;
        let gd_abs_change = (gd_prev - gd).abs();
        let gd_converged = cycle > 0 && gd_abs_change < config.gd_tolerance;

        final_deviance = Some(gd);
        final_deviance_change = if cycle > 0 { Some(gd_abs_change) } else { None };
        gd_prev = gd;

        final_iteration = cycle + 1;
        final_change = max_diff;

        if all_converged && gd_converged {
            converged = true;
            break;
        }
    }

    let mut final_results: IndexMap<String, FittedParameter> = IndexMap::new();
    let mut warnings: Vec<String> = Vec::new();

    for (name, model) in models {
        let covariance = model.covariance.ok_or_else(|| {
            GamlssError::Internal(format!(
                "Covariance matrix not computed for parameter '{}'",
                name
            ))
        })?;

        // Flag any smooth term whose EDF has decayed to its penalty null-space
        // dimension: the penalty has driven the smooth down to its unpenalized
        // polynomial remainder (e.g. a straight line for an order-2 P-spline),
        // so the curve it was meant to capture is effectively gone.
        for (layout, &t_edf) in model.term_layouts.iter().zip(model.term_edf.iter()) {
            if layout.is_smooth && t_edf <= layout.null_dim as f64 + EDF_COLLAPSE_SLACK {
                warnings.push(format!(
                    "parameter '{name}': a smooth term collapsed onto its penalty null space \
                     (edf {t_edf:.2} ≈ null-space dimension {nd}); it is effectively the \
                     unpenalized polynomial remainder, not a curve. The smoothing parameter may \
                     be over-selected, or the data carries little signal for this term.",
                    nd = layout.null_dim,
                ));
            }
        }

        // Build the term→column-block map from term_layouts (same offset walk as
        // the EDF attribution in scoring.rs). Must be done before `model.terms`
        // is moved into the FittedParameter literal below.
        let mut offset = 0usize;
        let term_blocks: Vec<(String, usize, usize)> = model
            .terms
            .iter()
            .zip(model.term_layouts.iter())
            .map(|(term, layout)| {
                let block = (term.term_name(), offset, offset + layout.n_coeffs);
                offset += layout.n_coeffs;
                block
            })
            .collect();

        // Persist the link name only when the user overrode it; a default-link
        // parameter stays `None` so predict re-derives via `default_link`.
        let link = config.links.get(&name).cloned();

        let fitted_param = FittedParameter {
            coefficients: model.beta,
            covariance,
            terms: model.terms,
            lambdas: model.lambdas,
            eta: model.eta,
            // `mu` is kept in sync with `eta` throughout fitting (see C.5 cache).
            fitted_values: model.mu,
            edf: model.edf,
            term_edf: model.term_edf,
            term_blocks,
            link,
        };
        final_results.insert(name, fitted_param);
    }

    let diagnostics = FitDiagnostics {
        converged,
        iterations: final_iteration,
        final_change,
        max_gradient: None,
        param_diagnostics,
        warnings,
        final_deviance,
        final_deviance_change,
    };

    Ok((final_results, diagnostics))
}

// ============================================================================
// Posterior sampling
// ============================================================================

/// Draws samples from N(beta_hat, V_beta) via Cholesky. Advanced consumers with
/// a pre-fitted `(β̂, V_β)` pair can call this directly; most callers should use
/// [`crate::GamlssModel::posterior_samples`].
///
/// # Errors
///
/// Returns [`GamlssError::PosteriorNotPositiveDefinite`] if Cholesky fails.
pub fn sample_posterior(
    beta_hat: &Coefficients,
    v_beta: &CovarianceMatrix,
    n_samples: usize,
) -> Result<Vec<Array1<f64>>, GamlssError> {
    sample_posterior_seeded(beta_hat, v_beta, n_samples, None)
}

/// Like [`sample_posterior`] with an optional seed; `None` uses the unseeded RNG.
///
/// # Errors
///
/// Returns [`GamlssError::PosteriorNotPositiveDefinite`] if Cholesky fails.
pub fn sample_posterior_seeded(
    beta_hat: &Coefficients,
    v_beta: &CovarianceMatrix,
    n_samples: usize,
    seed: Option<u64>,
) -> Result<Vec<Array1<f64>>, GamlssError> {
    let l_factor =
        linalg::cholesky_lower(&v_beta.0).map_err(|_| GamlssError::PosteriorNotPositiveDefinite)?;
    Ok(match seed {
        Some(s) => sample_from_cholesky(
            &beta_hat.0,
            &l_factor,
            n_samples,
            &mut StdRng::seed_from_u64(s),
        ),
        None => {
            let mut rng_rs = rng();
            sample_from_cholesky(&beta_hat.0, &l_factor, n_samples, &mut rng_rs)
        }
    })
}

pub(crate) fn sample_from_cholesky(
    mean: &Array1<f64>,
    l_factor: &Array2<f64>,
    n_samples: usize,
    rng: &mut impl Rng,
) -> Vec<Array1<f64>> {
    let dim = mean.len();

    // Reuse a single length-`dim` buffer for the standard-normal draws; the
    // matrix-vector product allocates the per-sample output.
    let mut z = Array1::<f64>::zeros(dim);

    (0..n_samples)
        .map(|_| {
            for v in z.iter_mut() {
                *v = StandardNormal.sample(rng);
            }
            mean + l_factor.dot(&z)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{sample_from_cholesky, FitConfig, SmoothingCriterion};
    use ndarray::{array, Array2};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn fit_config_default_matches_documented_values() {
        let c = FitConfig::default();
        assert_eq!(c.max_iterations, 200);
        assert_eq!(c.tolerance, 1e-3);
        assert_eq!(c.criterion, SmoothingCriterion::Reml);
        assert_eq!(SmoothingCriterion::default(), SmoothingCriterion::Reml);
        // FIT-1/FIT-2 defaults: step-halving on, GD tolerance 1e-3.
        assert!(c.step_halving);
        assert_eq!(c.gd_tolerance, 1e-3);
    }

    #[test]
    fn sample_from_cholesky_zero_factor_returns_mean() {
        // L = 0 ⇒ mean + 0·z = mean exactly, whatever the draws are.
        let mean = array![1.0, -2.0, 3.5];
        let l = Array2::<f64>::zeros((3, 3));
        let mut rng = StdRng::seed_from_u64(7);
        let samples = sample_from_cholesky(&mean, &l, 4, &mut rng);
        assert_eq!(samples.len(), 4);
        for s in &samples {
            assert_eq!(s.len(), 3);
            assert_eq!(*s, mean);
        }
    }

    #[test]
    fn sample_from_cholesky_is_reproducible_with_seed() {
        let mean = array![0.0, 0.0];
        let l = Array2::<f64>::eye(2);
        let a = sample_from_cholesky(&mean, &l, 5, &mut StdRng::seed_from_u64(42));
        let b = sample_from_cholesky(&mean, &l, 5, &mut StdRng::seed_from_u64(42));
        assert_eq!(a, b);
        assert_eq!(a.len(), 5);
        assert!(a.iter().all(|s| s.len() == 2));
    }

    #[test]
    fn sample_from_cholesky_shifts_by_mean() {
        // Identity factor ⇒ sample = mean + z. Same seed ⇒ same z, so the
        // difference between a shifted-mean run and a zero-mean run is the shift.
        let l = Array2::<f64>::eye(2);
        let base = sample_from_cholesky(&array![0.0, 0.0], &l, 3, &mut StdRng::seed_from_u64(99));
        let moved =
            sample_from_cholesky(&array![10.0, -5.0], &l, 3, &mut StdRng::seed_from_u64(99));
        for (z, m) in base.iter().zip(moved.iter()) {
            let diff = m - z;
            assert!((diff[0] - 10.0).abs() < 1e-9);
            assert!((diff[1] + 5.0).abs() < 1e-9);
        }
    }
}
