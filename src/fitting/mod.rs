//! GAMLSS fitting implementation via the Rigby-Stasinopoulos (RS) algorithm.
//!
//! The RS algorithm iteratively cycles through distribution parameters, fitting each as a
//! penalized additive model while holding others fixed. For each parameter:
//!
//! 1. Compute score (u) and Fisher information (w) from the distribution
//! 2. Form working response: z = η + u/w
//! 3. Optimize smoothing parameters (λ) via GCV using L-BFGS
//! 4. Solve penalized weighted least squares: (X'WX + Σλ·S)·β = X'W·z
//! 5. Update linear predictor: η = X·β
//!
//! The module also handles posterior inference (sampling from the approximate posterior of coefficients).

pub(crate) mod assembler;
pub mod diagnostics;
mod scoring;
mod solver;

use self::assembler::{assemble_model_matrices, resolve_terms};

use super::distributions::{Distribution, Link};
use super::error::GamlssError;
use super::terms::{Smooth, Term};
use super::types::*;
use crate::linalg;
use indexmap::IndexMap;
use ndarray::{Array1, Array2};
use rand::{rng, Rng};
use rand_distr::{Distribution as _, StandardNormal};

const DEFAULT_MAX_ITER: usize = 200;
const DEFAULT_TOLERANCE: f64 = 1e-3;

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
}

fn default_max_iter() -> usize { DEFAULT_MAX_ITER }
fn default_tolerance() -> f64  { DEFAULT_TOLERANCE }

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            max_iterations: DEFAULT_MAX_ITER,
            tolerance: DEFAULT_TOLERANCE,
            criterion: SmoothingCriterion::default(),
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
}

/// Diagnostic information for a single distribution parameter.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ParamDiagnostic {
    /// Sum of absolute changes in linear predictor (eta) in final iteration.
    pub final_eta_change: f64,
    /// Sum of absolute changes in smoothing parameters (lambda) in final iteration.
    pub final_lambda_change: f64,
    /// Effective degrees of freedom for this parameter's model.
    pub edf: f64,
    /// Number of observations whose IRLS working weight hit the lower floor
    /// in the final iteration. Non-zero values suggest degenerate Fisher info
    /// (extreme score with vanishing curvature) — the fit may be unreliable.
    pub weight_floor_hits: usize,
    /// Number of observations whose Fisher-scoring step `u/w` was clipped to
    /// `±MAX_STEP` in the final iteration. Non-zero values suggest the IRLS
    /// step is being damped to keep η updates finite — typically transient
    /// during early iterations, persistent at convergence indicates trouble.
    pub step_cap_hits: usize,
}

/// Fitted results for a single distribution parameter (e.g., mu, sigma).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FittedParameter {
    /// Estimated regression coefficients (beta).
    pub coefficients: Coefficients,
    /// Covariance matrix of the coefficient estimates.
    pub covariance: CovarianceMatrix,
    /// Terms included in this parameter's model formula.
    pub terms: Vec<Term>,
    /// Optimized smoothing parameters for each penalty matrix.
    pub lambdas: Array1<f64>,
    /// Linear predictor values (X * beta).
    pub eta: Array1<f64>,
    /// Fitted values on the response scale (link^-1(eta)).
    pub fitted_values: Array1<f64>,
    /// Effective degrees of freedom.
    pub edf: f64,
    /// Effective degrees of freedom attributed to each term, aligned with
    /// `terms`. Sums to `edf`. A smooth term whose entry sits at its penalty
    /// null-space dimension has been driven (near-)linear by its penalty —
    /// see `FitDiagnostics::warnings`.
    pub term_edf: Vec<f64>,
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
    /// Cached response-scale predictor `μ = link⁻¹(η)`. Kept in lockstep with
    /// `eta` so the IRLS step can hand out `&μ` references to every parameter
    /// instead of re-running `inv_link` on the full vector at each call —
    /// eliminates K length-n allocations per Fisher-scoring step.
    pub(super) mu: Array1<f64>,
    pub(super) lambdas: Array1<f64>,
    pub(super) covariance: Option<CovarianceMatrix>,
    pub(super) edf: f64,
    /// Latest per-term EDF (aligned with `terms`); updated each RS cycle.
    pub(super) term_edf: Vec<f64>,
}

/// Fits a GAMLSS model using the RS (Rigby-Stasinopoulos) algorithm.
///
/// The RS algorithm cycles through distribution parameters (μ, σ, ν, ...) fitting each
/// as a penalized additive model while holding others fixed. Each parameter update uses
/// penalized iteratively reweighted least squares (P-IRLS) with a working response z
/// and working weights w derived from the distribution's score and Fisher information.
pub(crate) fn fit_gamlss<D: Distribution + ?Sized>(
    data: &DataSet,
    y: &Array1<f64>,
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
        let link = family.default_link(param_name)?;

        let (x_model, penalty_matrices, total_coeffs, term_layouts) =
            assemble_model_matrices(data, n_obs, &terms)?;

        // Initialize on response scale using distribution-specific logic
        let response_scale_start = family.initial_value(param_name, y);
        let eta_start = link.link(response_scale_start);

        // Seed the intercept coefficient so the first IRLS step starts near
        // η = link(initial μ). `beta[0]` is only the intercept when the leading
        // term is `Term::Intercept`; for a smooth-only or leading-Linear formula
        // we leave β = 0 and η = 0, which is consistent with X·β and IRLS will
        // move us from there.
        let mut beta = Coefficients(Array1::zeros(total_coeffs));
        let intercept_leads = matches!(terms.first(), Some(Term::Intercept));
        let eta = if intercept_leads && total_coeffs > 0 {
            beta.0[0] = eta_start;
            Array1::from_elem(n_obs, eta_start)
        } else {
            Array1::zeros(n_obs)
        };
        let mu = eta.mapv(|e| link.inv_link(e));
        let lambdas = if penalty_matrices.is_empty() {
            Array1::zeros(0)
        } else {
            Array1::ones(penalty_matrices.len())
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

    for cycle in 0..config.max_iterations {
        param_diagnostics.clear();
        let mut max_diff = 0.0;

        for param_name in family.parameters() {
            let update = scoring::step(family, y, &models, param_name, config.criterion)?;
            if update.max_diff > max_diff {
                max_diff = update.max_diff;
            }
            param_diagnostics.insert(
                param_name.to_string(),
                ParamDiagnostic {
                    final_eta_change: update.eta_change,
                    final_lambda_change: update.lambda_change,
                    edf: update.edf,
                    weight_floor_hits: update.weight_floor_hits,
                    step_cap_hits: update.step_cap_hits,
                },
            );

            let model = models.get_mut(*param_name).ok_or_else(|| {
                GamlssError::Internal(format!("Model for parameter '{}' not found", param_name))
            })?;
            model.beta = update.beta;
            model.eta = update.eta;
            model.mu = update.mu;
            model.lambdas = update.lambdas;
            model.covariance = Some(update.covariance);
            model.edf = update.edf;
            model.term_edf = update.term_edf;
        }

        final_iteration = cycle + 1;
        final_change = max_diff;

        // Relative convergence: scale by the largest |β| across all parameters so the
        // threshold is unit-agnostic. A floor of 1.0 keeps the check identical to the
        // old absolute threshold when coefficients are O(1) (e.g. normalized data).
        let beta_scale = models
            .values()
            .flat_map(|m| m.beta.0.iter().copied().map(f64::abs))
            .fold(1.0_f64, f64::max);
        if max_diff / beta_scale < config.tolerance {
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
    };

    Ok((final_results, diagnostics))
}

// ============================================================================
// Posterior sampling
// ============================================================================

/// Draws samples from the approximate posterior N(beta_hat, V_beta) via Cholesky decomposition.
///
/// Most callers should reach for [`crate::GamlssModel::posterior_samples`] or
/// [`crate::GamlssModel::predict_samples`]; this is exposed for advanced consumers that
/// already have a fitted `(β̂, V_β)` pair and want to drive sampling directly.
///
/// # Errors
///
/// Returns [`GamlssError::PosteriorNotPositiveDefinite`] if the Cholesky factorization
/// of `v_beta` fails. A non-PD covariance signals a degenerate fit; callers should
/// surface this rather than silently dropping the samples.
pub fn sample_posterior(
    beta_hat: &Coefficients,
    v_beta: &CovarianceMatrix,
    n_samples: usize,
) -> Result<Vec<Array1<f64>>, GamlssError> {
    let l_factor =
        linalg::cholesky_lower(&v_beta.0).map_err(|_| GamlssError::PosteriorNotPositiveDefinite)?;

    let mut rng_rs = rng();
    Ok(sample_from_cholesky(
        &beta_hat.0,
        &l_factor,
        n_samples,
        &mut rng_rs,
    ))
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
