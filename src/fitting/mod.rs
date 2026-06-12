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
use rand::rngs::StdRng;
use rand::{rng, Rng, SeedableRng};
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

#[cfg(feature = "serde")]
fn default_max_iter() -> usize {
    DEFAULT_MAX_ITER
}
#[cfg(feature = "serde")]
fn default_tolerance() -> f64 {
    DEFAULT_TOLERANCE
}

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
    pub final_eta_change: f64,
    pub final_lambda_change: f64,
    pub edf: f64,
    /// Non-zero suggests degenerate Fisher info (extreme score, vanishing curvature).
    pub weight_floor_hits: usize,
    /// Non-zero means IRLS steps were damped; persistent at convergence indicates trouble.
    pub step_cap_hits: usize,
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
    pub(super) lambdas: Array1<f64>,
    pub(super) covariance: Option<CovarianceMatrix>,
    pub(super) edf: f64,
    /// Latest per-term EDF (aligned with `terms`); updated each RS cycle.
    pub(super) term_edf: Vec<f64>,
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
        let link = family.default_link(param_name)?;

        let (x_model, penalty_matrices, total_coeffs, term_layouts) =
            assemble_model_matrices(data, n_obs, &terms)?;

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

            // Per-parameter relative convergence check.
            //
            // The previous (global) test divided the max β-change across *all*
            // parameters by the max |β| across *all* parameters.  When one parameter
            // (e.g. mu) has large coefficients while another (e.g. log-scale sigma)
            // has small ones, the shared denominator is dominated by mu's scale and
            // the loop could declare convergence while sigma is still drifting.
            //
            // Fix: each parameter is checked against its own |β| scale.  The floor of
            // 1.0 keeps the test equivalent to an absolute threshold when all
            // coefficients are O(1) (normalised data).
            let param_beta_scale = update
                .beta
                .0
                .iter()
                .copied()
                .map(f64::abs)
                .fold(1.0_f64, f64::max);
            if update.max_diff / param_beta_scale >= config.tolerance {
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

        if all_converged {
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
