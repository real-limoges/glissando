//! Model selection and comparison over fitted [`GamlssModel`]s.
//!
//! Three facilities sharing one mechanism — compare models by a penalized
//! log-likelihood (an information criterion) or a deviance difference — over one
//! substrate, the global deviance `−2·ℓ̂` and total effective degrees of freedom
//! already produced by [`fitting::diagnostics`](crate::diagnostics):
//!
//! - [`ic_table`] ranks any set of models (nested or not) by EDF / global
//!   deviance / GAIC — a comparison, no test.
//! - [`lr_test`] runs a likelihood-ratio χ² test for a nested pair.
//! - [`step_gaic`] greedily adds/drops one term at a time to minimize GAIC(k).
//!
//! Every score flows through [`compute_gaic`],
//! so these comparisons stay consistent with `diagnostics(..).aic`/`.bic`.

use super::diagnostics::{compute_gaic, total_edf};
use crate::distributions::Distribution;
use crate::terms::Term;
use crate::types::{DataSet, Formula};
use crate::FitConfig;
use crate::GamlssError;
use crate::GamlssModel;
use ndarray::Array1;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use statrs::function::gamma::gamma_ur;
use std::collections::HashMap;

/// Snapshot of a fitted model's parameters in the shape the [`Distribution`]
/// trait expects. Thin wrapper over the shared [`super::diagnostics::fitted_params_view`]
/// so the `&GamlssModel` call sites read cleanly.
fn params_view(model: &GamlssModel) -> HashMap<&str, &Array1<f64>> {
    super::diagnostics::fitted_params_view(&model.models)
}

/// One row of an information-criterion comparison table.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IcRow {
    /// Caller-supplied label identifying the model.
    pub label: String,
    /// Total effective degrees of freedom (summed smoother traces).
    pub edf: f64,
    /// Global deviance `−2·ℓ̂` (gamlss convention; no saturated-model subtraction).
    pub global_deviance: f64,
    /// Generalized AIC at the requested penalty `k`.
    pub gaic: f64,
}

/// Result of a likelihood-ratio test of `small` nested in `big`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LrTest {
    /// `LR = 2·(ℓ̂_big − ℓ̂_small) = GD_small − GD_big`.
    pub lr_stat: f64,
    /// Degrees of freedom `ν = edf_big − edf_small` (generally fractional).
    pub df: f64,
    /// `P(χ²_ν > LR)` — the asymptotic null tail probability.
    pub p_value: f64,
}

/// Tabulate a set of models by EDF, global deviance, and GAIC(`k`).
///
/// A ranking, not a test: it works for any models fit to the same response `y`,
/// nested or not — this is how you compare different families or non-nested term
/// sets. The returned rows are in the order the models were passed; sort by
/// `gaic` (or `global_deviance`) to rank.
///
/// # Errors
///
/// Propagates [`GamlssError`] from a model's log-likelihood evaluation (e.g.
/// [`GamlssError::UnknownParameter`] if `family` needs a parameter a model lacks).
pub fn ic_table<D: Distribution + ?Sized>(
    models: &[(&str, &GamlssModel)],
    family: &D,
    y: &Array1<f64>,
    k: f64,
) -> Result<Vec<IcRow>, GamlssError> {
    models
        .iter()
        .map(|(label, model)| {
            let ll = family.loglik(y, &params_view(model))?;
            let edf = total_edf(&model.models);
            Ok(IcRow {
                label: (*label).to_string(),
                edf,
                global_deviance: -2.0 * ll,
                gaic: compute_gaic(ll, edf, k),
            })
        })
        .collect()
}

/// Likelihood-ratio test of `small` (the null) nested in `big` (the alternative).
///
/// Wilks' theorem gives the asymptotic null distribution of the deviance
/// difference: `LR = 2·(ℓ̂_big − ℓ̂_small) →ᵈ χ²_ν` with `ν = edf_big − edf_small`.
/// The χ² survival function is evaluated through the regularized upper incomplete
/// gamma `Q(ν/2, LR/2)` ([`gamma_ur`]), which handles the fractional `ν` that
/// penalized smooths produce.
///
/// # Caveat
///
/// Penalized smooths have non-integer effective df, so `ν` is generally
/// fractional and the χ² reference is **approximate** — exactly the caveat
/// `anova.gam`/`summary.gam` carry in mgcv. For unpenalized (integer-df) nested
/// linear models it is exact up to the asymptotics.
///
/// # Errors
///
/// Returns [`GamlssError::Input`] when the pair is mis-ordered or non-nested
/// (`edf_big ≤ edf_small`, or `big` fits worse than `small` beyond rounding).
/// Propagates log-likelihood evaluation errors otherwise.
pub fn lr_test<D: Distribution + ?Sized>(
    small: &GamlssModel,
    big: &GamlssModel,
    family: &D,
    y: &Array1<f64>,
) -> Result<LrTest, GamlssError> {
    let ll0 = family.loglik(y, &params_view(small))?;
    let ll1 = family.loglik(y, &params_view(big))?;
    let edf0 = total_edf(&small.models);
    let edf1 = total_edf(&big.models);

    let lr = 2.0 * (ll1 - ll0);
    let df = edf1 - edf0;

    // Nesting guard: `big` must be the larger, at-least-as-good-fitting model.
    if df <= 0.0 || lr < -1e-6 {
        return Err(GamlssError::Input(
            "lr_test expects `small` nested in `big` (edf_big > edf_small, loglik_big ≥ loglik_small)"
                .into(),
        ));
    }

    // P(χ²_df > LR) = Q(df/2, LR/2). Clamp a tiny negative LR (rounding) to 0.
    let p_value = gamma_ur(df / 2.0, lr.max(0.0) / 2.0);
    Ok(LrTest {
        lr_stat: lr,
        df,
        p_value,
    })
}

// ----------------------------------------------------------------------------
// INFER-4 — stepwise term selection (stepGAIC analog)
// ----------------------------------------------------------------------------

/// The moves the stepwise search may make for one distribution parameter.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StepScope {
    /// Distribution parameter the candidates apply to (`"mu"`, `"sigma"`, …).
    pub param: String,
    /// Terms eligible to add (if absent) or drop (if present) on `param`.
    pub candidates: Vec<Term>,
}

/// Which direction the greedy search may move at each step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum Direction {
    /// Only add absent candidate terms.
    Forward,
    /// Only drop present candidate terms.
    Backward,
    /// Add or drop, whichever lowers GAIC most.
    Both,
}

impl Direction {
    /// Parse a `Direction` from its wire name (`"forward"`, `"backward"`, `"both"`,
    /// case-insensitive). Shared by the JSON and Python FFI layers so the mapping
    /// and error text only live once.
    pub fn from_name(name: &str) -> Result<Direction, GamlssError> {
        match name.to_ascii_lowercase().as_str() {
            "forward" => Ok(Direction::Forward),
            "backward" => Ok(Direction::Backward),
            "both" => Ok(Direction::Both),
            other => Err(GamlssError::Input(format!(
                "Unknown direction '{}', expected 'forward', 'backward', or 'both'",
                other
            ))),
        }
    }
}

/// One accepted step in a [`step_gaic`] run.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StepRecord {
    /// Human-readable description of the accepted move, e.g. `"+ x1 on mu"`.
    pub move_: String,
    /// GAIC of the model after this move.
    pub gaic: f64,
    /// Total EDF of the model after this move.
    pub edf: f64,
}

/// Outcome of a [`step_gaic`] search: the selected model, its formula, and the
/// ordered trace of accepted moves.
pub struct StepResult {
    /// The fitted model at the selected (locally optimal) formula.
    pub model: GamlssModel,
    /// The selected formula.
    pub formula: Formula,
    /// Accepted moves in the order they were taken; each strictly lowered GAIC.
    pub trace: Vec<StepRecord>,
}

/// True if `param`'s term list in `f` already contains a term with `t`'s name.
fn has_term(f: &Formula, param: &str, t: &Term) -> bool {
    f.get(param)
        .is_some_and(|ts| ts.iter().any(|x| x.term_name() == t.term_name()))
}

/// Copy of `f` with `t` appended to `param`'s term list.
fn with_added(f: &Formula, param: &str, t: &Term) -> Formula {
    let mut terms = f.get(param).cloned().unwrap_or_default();
    terms.push(t.clone());
    f.clone().with_terms(param.to_string(), terms)
}

/// Copy of `f` with every term named like `t` removed from `param`'s term list.
fn with_dropped(f: &Formula, param: &str, t: &Term) -> Formula {
    let terms: Vec<Term> = f
        .get(param)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter(|x| x.term_name() != t.term_name())
        .collect();
    f.clone().with_terms(param.to_string(), terms)
}

/// Greedy stepwise term selection by GAIC(`k`) — the `stepGAIC` analog.
///
/// At each step the search evaluates every single-term add/drop allowed by
/// `scope` and `direction`, refits, and accepts the move that lowers GAIC the
/// most, stopping when no move improves it by more than a small tolerance. The
/// penalty `k` is the selection knob: `k = 2` (AIC) is permissive, `k = log n`
/// (BIC) is parsimonious.
///
/// The search is **greedy and single-term** (no look-ahead, no interaction
/// synthesis), matching gamlss's `stepGAIC`, so the result is a **local**
/// optimum. Candidate moves are enumerated in a deterministic order (scope order,
/// then candidate order), so identical inputs yield an identical [`StepResult::trace`].
/// A trial fit that errors (non-convergence, singular system) is skipped rather
/// than aborting the search.
///
/// # Errors
///
/// Returns [`GamlssError`] if the initial fit at `start` fails, or if scoring a
/// fitted model's log-likelihood fails. Trial fits that error are skipped.
// The full fitting context (data, response, family, starting formula, scope,
// penalty, direction, config) is irreducible here — each is an independent input
// to the search, so the argument count is intentional.
#[allow(clippy::too_many_arguments)]
pub fn step_gaic<D: Distribution + ?Sized>(
    data: &DataSet,
    y: &Array1<f64>,
    family: &D,
    start: Formula,
    scope: &[StepScope],
    k: f64,
    direction: Direction,
    config: FitConfig,
) -> Result<StepResult, GamlssError> {
    // Guard against churn when a move barely changes the score (e.g. ties).
    const EPS: f64 = 1e-6;

    let mut current = start;
    let mut model = GamlssModel::fit_with_config(data, y, None, &current, family, config.clone())?;
    let mut best_gaic = model.gaic(family, y, k)?;
    let mut trace = Vec::new();

    // One enumerated candidate move: which term, on which param, add or drop,
    // and the trial formula it produces. Enumeration itself is cheap and reads
    // `current`, so it stays sequential; only the fit + score per candidate
    // (below) is parallelized.
    struct Candidate {
        is_add: bool,
        term_name: String,
        param: String,
        trial: Formula,
    }
    type CandidateOutcome = Result<Option<(f64, f64, String, Formula, GamlssModel)>, GamlssError>;

    loop {
        // Enumerate every candidate move in scope order, then candidate order,
        // so the flattened list preserves the deterministic trace order even
        // though the fits below run in parallel.
        let mut candidates: Vec<Candidate> = Vec::new();
        for s in scope {
            for t in &s.candidates {
                let present = has_term(&current, &s.param, t);
                let is_add = match direction {
                    Direction::Forward if present => continue,
                    Direction::Forward => true,
                    Direction::Backward if !present => continue,
                    Direction::Backward => false,
                    Direction::Both => !present,
                };
                let trial = if is_add {
                    with_added(&current, &s.param, t)
                } else {
                    with_dropped(&current, &s.param, t)
                };
                candidates.push(Candidate {
                    is_add,
                    term_name: t.term_name().to_string(),
                    param: s.param.clone(),
                    trial,
                });
            }
        }

        // The candidate fits are independent (each refits `data`/`y` at its own
        // trial formula), so run them in parallel; picking the best-improving
        // move stays a serial reduction below so ties still resolve to the
        // first candidate in scope/candidate order, matching the sequential
        // behavior.
        let evaluate = |c: &Candidate| -> CandidateOutcome {
            // Skip a move that fails to fit (non-convergence, singular, …).
            let fit =
                match GamlssModel::fit_with_config(data, y, None, &c.trial, family, config.clone())
                {
                    Ok(m) => m,
                    Err(_) => return Ok(None),
                };
            let g = fit.gaic(family, y, k)?;
            let e = total_edf(&fit.models);
            let label = format!(
                "{} {} on {}",
                if c.is_add { "+" } else { "-" },
                c.term_name,
                c.param
            );
            Ok(Some((g, e, label, c.trial.clone(), fit)))
        };
        let outcomes: Vec<CandidateOutcome> = {
            #[cfg(feature = "parallel")]
            {
                candidates.par_iter().map(evaluate).collect()
            }
            #[cfg(not(feature = "parallel"))]
            {
                candidates.iter().map(evaluate).collect()
            }
        };

        let mut best: Option<(f64, f64, String, Formula, GamlssModel)> = None;
        for outcome in outcomes {
            if let Some(entry) = outcome? {
                if best.as_ref().is_none_or(|(bg, ..)| entry.0 < *bg) {
                    best = Some(entry);
                }
            }
        }

        match best {
            Some((g, e, label, trial, fit)) if g < best_gaic - EPS => {
                best_gaic = g;
                current = trial;
                model = fit;
                trace.push(StepRecord {
                    move_: label,
                    gaic: g,
                    edf: e,
                });
            }
            // No improving move: stop at the local optimum.
            _ => break,
        }
    }

    Ok(StepResult {
        model,
        formula: current,
        trace,
    })
}
