//! GAMLSS fitting via the Rigby-Stasinopoulos (RS) algorithm.
//!
//! The idea I keep coming back to: RS cycles through the distribution parameters one at a
//! time, fitting each as a penalized additive model while the others sit frozen. So no single
//! step is doing anything exotic; it is a plain penalized GLM update, just wrapped in an outer
//! loop that rotates which parameter is "live". For each parameter:
//!
//! 1. Compute score (u) and Fisher information (w) from the distribution
//! 2. Form working response: z = η + u/w
//! 3. Optimize smoothing parameters (λ) via the configured criterion (REML default)
//! 4. Solve penalized weighted least squares: (X'WX + Σλ·S)·β = X'W·z
//! 5. Update linear predictor: η = X·β
//!
//! Posterior inference lives here too: sampling from the approximate posterior of the
//! coefficients.

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

/// How close a smooth term's EDF has to sit to its penalty null-space dimension
/// before I call it collapsed. Half an effective degree of freedom of slack is
/// the compromise: enough that a genuinely (near-)linear fit doesn't trip the
/// warning, tight enough that a smooth penalized all the way down still gets caught.
const EDF_COLLAPSE_SLACK: f64 = 0.5;

/// Smoothing-parameter selection criterion.
///
/// `Reml` (the default) minimizes the Laplace-approximate marginal likelihood
/// (Wood 2011) via L-BFGS, applied per distributional parameter to its converged
/// PWLS subproblem. `Gcv` uses Generalized Cross-Validation (Craven & Wahba 1979).
/// `FellnerSchall` chases the same target as `Reml` but through the multiplicative
/// fixed-point update of Wood & Fasiolo (2017): deterministic, no line search.
/// I default to REML because at moderate sample sizes it is the one least likely to
/// wander into a local minimum or undersmooth on me; GCV is more willing to do both.
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

impl SmoothingCriterion {
    /// Parse a `SmoothingCriterion` from its wire name (`"gcv"`, `"reml"`,
    /// `"fellner_schall"`, case-insensitive). `json.rs` already gets this for free
    /// from serde's `rename_all = "snake_case"`; this exists so `python.rs` shares
    /// the exact same mapping instead of hand-rolling its own copy that drifts.
    pub fn from_name(name: &str) -> Result<SmoothingCriterion, GamlssError> {
        match name.to_ascii_lowercase().as_str() {
            "gcv" => Ok(SmoothingCriterion::Gcv),
            "reml" => Ok(SmoothingCriterion::Reml),
            "fellner_schall" => Ok(SmoothingCriterion::FellnerSchall),
            other => Err(GamlssError::Input(format!(
                "Unknown criterion '{}', expected 'gcv', 'reml', or 'fellner_schall'",
                other
            ))),
        }
    }
}

/// What the fitter does with a row that carries a missing (non-finite) value in
/// the response or any formula-referenced column (DATA-4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum NaAction {
    /// Drop any row with a missing value in `y` or a referenced column before
    /// fitting (R's `na.omit`), and the default. Weights and every model column
    /// get masked together, so the design, working response, and weights all stay
    /// aligned; nothing goes out of step underneath you.
    #[default]
    DropRows,
    /// Reject the fit with [`GamlssError`] the moment any model
    /// variable is non-finite (the historical behavior). Reach for this when a
    /// missing value ought to be a hard error, not something quietly dropped.
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
    /// On by default, matching R `gamlss`'s RS loop. Turn it off and you get the
    /// raw, unguarded full step back, which is faster right up until it isn't.
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
    /// [`default_link`](crate::distributions::Distribution::default_link). Build
    /// ergonomically with [`with_link`](FitConfig::with_link) /
    /// [`with_links`](FitConfig::with_links).
    ///
    /// Three things are checked at fit time, each yielding [`GamlssError::Input`]:
    /// the key must name one of the family's
    /// [`parameters`](crate::distributions::Distribution::parameters), the family
    /// must accept an override there (see
    /// [`allows_link_override`](crate::distributions::Distribution::allows_link_override)),
    /// and the value must be a link
    /// [`link_from_name`] recognizes.
    ///
    /// **No domain checking is done.** A link whose range does not contain the
    /// parameter's support is accepted and quietly produces nonsense instead of an
    /// error: a logit link on a Poisson μ pins the mean into `(0, 1)`, and a
    /// `sqrt` or `inverse_square` link on a Gaussian μ cannot represent a negative
    /// mean at all. Picking a link that actually suits the parameter is on you.
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
    /// Total EDF, always populated as `term_edf.iter().sum()` at construction
    /// time. Kept as its own stored field, rather than a computed method like
    /// `FittingParameter::edf`, because this type round-trips through
    /// `to_json`/`from_json` and other FFI surfaces: removing the field would
    /// break wire compatibility for existing consumers.
    pub edf: f64,
    /// Per-term EDF aligned with `terms`, summing to `edf`. An entry at the
    /// term's penalty null-space dimension means the smooth collapsed to its
    /// unpenalized polynomial remainder; see `FitDiagnostics::warnings`.
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
    /// Latest per-term EDF (aligned with `terms`); updated each RS cycle.
    pub(super) term_edf: Vec<f64>,
}

/// (Prior-weighted) global deviance of the current fit:
/// `GD(θ) = −2·Σᵢ wᵢ·log f(yᵢ | θᵢ)`.
///
/// This is the objective the Rigby–Stasinopoulos loop is quietly minimizing the
/// whole time, and it does double duty: step-halving (FIT-1) and the
/// global-deviance convergence test (FIT-2) both read it. Each
/// `FittingParameter.mu` already carries the response-scale parameter, so there
/// is nothing clever to do here; assemble the params view and hand it to the
/// family's pointwise log-density.
pub(super) fn global_deviance<D: Distribution + ?Sized>(
    family: &D,
    y: &Array1<f64>,
    prior_weights: Option<&Array1<f64>>,
    models: &IndexMap<String, FittingParameter>,
) -> Result<f64, GamlssError> {
    deviance(family, y, prior_weights, models, None)
}

/// Same as [`global_deviance`], but evaluating one parameter block at a *proposed*
/// `mu_override` not yet committed to `models`; used by step-halving to score a
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

/// Check every `config.links` key against the family before any fitting starts.
///
/// Both of these failures used to be silent, which is exactly why they earned a
/// guard. An unknown key just never matched inside the per-parameter loop below,
/// so `with_link("sigma", "log")` on `Beta` (whose second parameter is `phi`) did
/// precisely nothing. And a parameter whose family hardcodes its own link in
/// `eta_derivatives` would accept the override for `η → μ` while still computing
/// the score and weight against the original link: the very bug class the
/// generic-chain-rule work exists to kill.
///
/// Runs before `assemble_model_matrices`, so a typo costs you nothing.
fn validate_link_overrides<D: Distribution + ?Sized>(
    family: &D,
    config: &FitConfig,
) -> Result<(), GamlssError> {
    for key in config.links.keys() {
        // Membership first. On a family that refuses every parameter, a
        // misspelled key should complain about the misspelling, not the refusal.
        if !family.parameters().contains(&key.as_str()) {
            return Err(GamlssError::Input(format!(
                "link override for unknown parameter '{key}': {} has parameters [{}]",
                family.name(),
                family.parameters().join(", "),
            )));
        }
        if !family.allows_link_override(key) {
            return Err(GamlssError::Input(format!(
                "{} does not support a link override for '{key}': its score and \
                 weight are derived against that parameter's default link and \
                 cannot be re-chained generically. Remove the override, or model \
                 the parameter with a family that supports one.",
                family.name(),
            )));
        }
    }
    Ok(())
}

pub(crate) fn fit_gamlss<D: Distribution + ?Sized>(
    data: &DataSet,
    y: &Array1<f64>,
    prior_weights: Option<&Array1<f64>>,
    formula: &Formula,
    family: &D,
    config: &FitConfig,
) -> Result<(IndexMap<String, FittedParameter>, FitDiagnostics), GamlssError> {
    validate_link_overrides(family, config)?;

    let n_obs = y.len();
    // `IndexMap` keeps insertion order = `family.parameters()` order, so
    // `GamlssModel.models` iterates deterministically everywhere downstream.
    let mut models: IndexMap<String, FittingParameter> = IndexMap::new();

    for param_name in family.parameters() {
        let param_name_str = param_name.to_string();
        let formula_terms = formula.get(&param_name_str).ok_or_else(|| {
            GamlssError::Input(format!("Formula missing for parameter {}", param_name))
        })?;
        // Resolve CrSpline1D knots once from the training data. They get stored in
        // FittedParameter::terms and replayed verbatim at predict time; fit and
        // predict have to see the identical basis or the whole thing is a lie.
        let terms = resolve_terms(formula_terms, data)?;
        // Honor a per-parameter link override from the config, else fall back to
        // the family's canonical default. Whichever wins, its name is persisted
        // into the FittedParameter so predict rebuilds the *same* link.
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
        // η = link(initial μ). `beta[0]` is the intercept only when the leading
        // term is `Term::Intercept`; for a smooth-only or leading-Linear formula
        // we just leave β = 0 and η = X·β and let IRLS walk it from there. The
        // fixed `offset` is always added on top: η = X·β + offset (DATA-3).
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
            // opens with a well-conditioned X'WX + S_lambda. lambda=1 sounds
            // innocent but is often too small for a high-cardinality basis (say
            // k=20) or a model with prior weights, and leaves the system
            // near-singular on the very first call.
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
                term_edf: vec![0.0; n_terms],
            },
        );
    }

    let mut converged = false;
    let mut final_iteration = 0;
    let mut final_change = f64::MAX;
    let mut param_diagnostics: IndexMap<String, ParamDiagnostic> = IndexMap::new();

    // FIT-2: track the global deviance across cycles so I can judge convergence on
    // actual objective improvement, not just coefficients wiggling around.
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
            // the accepted step can never raise it (monotone descent). The full
            // step is just the α = 1 case, so a well-behaved fit pays no halvings
            // at all; you only pay when the step was going to overshoot.
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
                // No step-halving means nothing else bounds the accepted step, so
                // scale the raw Fisher step back to at most `MAX_STEP_NO_HALVING`
                // per element in η (see that constant's doc comment). scoring::step
                // has its own internal MAX_STEP clamp, but it is now far too loose
                // (1e6) to be the only guard. Scale β, not η directly, so that
                // η = X·β + offset stays exact.
                let pre_model = &models[*param_name];
                let raw_max_change = update.eta_max_change;
                // scale == 1.0 reproduces the full-step proposal exactly, so one
                // unconditional construction handles both the clamped and the
                // unclamped case without a branch.
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

            // Per-parameter relative convergence check, done in FIT SPACE (η = X·β).
            //
            // A coefficient-space (Δβ) test gives false negatives on penalized
            // models, and here is why. When the smoothing objective has a flat
            // valley, the per-cycle λ re-optimization jitters between
            // fit-equivalent (λ, β) pairs whose linear predictors are identical:
            // β wanders along a fit-irrelevant ridge forever and Δβ never passes,
            // even though the model (η, μ, deviance) has stopped moving. Watching
            // Δη instead is blind to those ridges, and paired with the
            // global-deviance test below it is strictly stronger than gamlss's
            // deviance-only criterion.
            //
            // Each parameter is checked against its own |η| scale, floored at 1.0
            // so the test degrades to a plain absolute threshold when the linear
            // predictor is O(1). It reads `accepted.eta_max_change` (what
            // step-halving actually applied), not the full-step proposal, and the
            // two agree where it matters: near the optimum the score → 0, so the
            // full step → 0, step-halving takes α = 1, and they coincide. They part
            // ways only when the whole block update was rejected (α = 0, model
            // frozen). Use the proposal there and you keep flagging "still moving"
            // for a state that has not budged since the first rejection, burning
            // the iteration budget re-deriving and re-rejecting the same step every
            // cycle. Far from the optimum a large accepted step keeps this test
            // conservative, which is precisely when the GD test below should be the
            // one calling it.
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
                    edf: update.edf(),
                    weight_floor_hits: update.weight_floor_hits,
                    step_cap_hits: update.step_cap_hits,
                    step_halving_hits: accepted.hits,
                },
            );

            // Infallible: `models` was populated from this same `family.parameters()`
            // list above and nothing ever removes entries (see `&models[*param_name]`).
            let model = &mut models[*param_name];
            // β/η/μ come from the accepted (possibly damped) step; covariance /
            // EDF / λ keep the values `scoring::step` computed at the full step.
            // Near convergence α → 1, so those are evaluated at the right point
            // anyway, matching what gamlss reports at the converged step.
            //
            // When the whole block update got REJECTED (uphill at every α), keep
            // the previous λ/covariance/EDF too. The proposal's values describe a
            // state we never actually entered; install them and you pair the old β
            // with a covariance/EDF measured somewhere else entirely, which quietly
            // corrupts SEs, GAIC, and the collapse warnings. One exception: the very
            // first cycle, where there is no previous covariance yet, so the
            // proposal's is simply the best we have.
            model.beta = accepted.beta;
            model.eta = accepted.eta;
            model.mu = accepted.mu;
            if !accepted.rejected || model.covariance.is_none() {
                model.lambdas = update.lambdas;
                model.covariance = Some(update.covariance);
                model.term_edf = update.term_edf;
            }
        }

        // FIT-2: global-deviance change after the full sweep. I want *both* the Δβ
        // test and the GD test to agree before I call it converged.
        //
        // Measured in ABSOLUTE deviance units, matching R gamlss's `c.crit`
        // (default 0.001). A relative test (|ΔGD|/|GD|) was here before, and the
        // trouble with it is that its slack scales with the deviance magnitude: at
        // GD ≈ 4000 it happily declared convergence while the fit was still
        // improving ~4 deviance units per cycle, well short of the optimum gamlss
        // reaches on the same data with its absolute criterion.
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

        // Flag any smooth term whose EDF has decayed down to its penalty
        // null-space dimension. The penalty has ground the smooth down to its
        // unpenalized polynomial remainder (a straight line, for an order-2
        // P-spline), so the curve it was supposed to capture is gone.
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

        // Build the term→column-block map from term_layouts (the same offset walk
        // the EDF attribution in scoring.rs does). Has to happen before
        // `model.terms` is moved into the FittedParameter literal below.
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

        // Persist the link name only when the user overrode it. A default-link
        // parameter stays `None` and predict re-derives it via `default_link`.
        let link = config.links.get(&name).cloned();

        let fitted_param = FittedParameter {
            coefficients: model.beta,
            covariance,
            terms: model.terms,
            lambdas: model.lambdas,
            eta: model.eta,
            // `mu` is kept in sync with `eta` throughout fitting (see C.5 cache).
            fitted_values: model.mu,
            // Summed from `term_edf` (`FittingParameter` no longer stores a
            // separate total) so the two can never drift apart.
            edf: model.term_edf.iter().sum(),
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
