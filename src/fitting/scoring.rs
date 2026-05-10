//! One Fisher-scoring step on a single distribution parameter.
//!
//! The Rigby–Stasinopoulos outer loop cycles through parameters; for each one,
//! [`step`] performs the five operations that constitute a P-IRLS update:
//!
//! 1. Snapshot every parameter on the response scale.
//! 2. Ask the family for score `u` and Fisher information `w` on the η-scale.
//! 3. Form the working response `z = η + u/w` (with weight floor and step clamp).
//! 4. Optimize smoothing parameters λ via GCV (warm-started from the previous step).
//! 5. Solve penalized weighted least squares for `(β, V, EDF)`.
//!
//! `step` returns an [`Update`] describing the new state plus convergence and
//! diagnostic deltas. The caller applies the update; `step` itself is pure with
//! respect to the input `models` map, which keeps it unit-testable in isolation.

use super::solver::{fit_pwls, run_optimization};
use super::FittingParameter;
use crate::distributions::Distribution;
use crate::error::GamlssError;
use crate::types::{Coefficients, CovarianceMatrix};
use ndarray::Array1;
use std::collections::HashMap;

/// Lower bound for IRLS working weights, preventing division by near-zero.
const MIN_WEIGHT: f64 = 1e-6;

/// Cap on the per-element Fisher-scoring step `u/w` (in η units), guarding against
/// huge updates from extreme score / small Fisher info combinations.
const MAX_STEP: f64 = 20.0;

/// New state plus per-step diagnostics produced by [`step`].
#[derive(Debug)]
pub(super) struct Update {
    pub beta: Coefficients,
    /// Linear predictor η = X·β.
    pub eta: Array1<f64>,
    /// Smoothing parameters on the response scale (not log).
    pub lambdas: Array1<f64>,
    pub covariance: CovarianceMatrix,
    pub edf: f64,
    /// Max absolute change in β; drives the outer-loop convergence check.
    pub max_diff: f64,
    /// Sum of absolute changes in η, recorded as a per-parameter diagnostic.
    pub eta_change: f64,
    /// Sum of absolute changes in λ, recorded as a per-parameter diagnostic.
    pub lambda_change: f64,
}

/// Run one Fisher-scoring step on `target_param`, given the current state of every
/// parameter in `models`.
pub(super) fn step<D: Distribution + ?Sized>(
    family: &D,
    y: &Array1<f64>,
    models: &HashMap<String, FittingParameter>,
    target_param: &str,
) -> Result<Update, GamlssError> {
    // 1. Snapshot every parameter on the response scale; derivatives() expects all of them.
    let current_params: HashMap<&str, Array1<f64>> = family
        .parameters()
        .iter()
        .map(|name| {
            let model = &models[*name];
            (*name, model.eta.mapv(|e| model.link.inv_link(e)))
        })
        .collect();
    let params_ref: HashMap<&str, &Array1<f64>> =
        current_params.iter().map(|(k, v)| (*k, v)).collect();

    // 2. Score and Fisher info for the target parameter.
    let all_derivs = family.derivatives(y, &params_ref)?;
    let (deriv_u, deriv_w) = all_derivs.get(target_param).ok_or_else(|| {
        GamlssError::Input(format!("No derivation for {} found", target_param))
    })?;

    let target = models.get(target_param).ok_or_else(|| {
        GamlssError::Internal(format!("Model for parameter '{}' not found", target_param))
    })?;

    // 3. Working response z = η + u/w.  Floor weights and clamp the step in η units
    //    so degenerate Fisher information can't blow up the IRLS update.
    let safe_w = deriv_w.mapv(|w: f64| w.max(MIN_WEIGHT));
    let adjustment = (deriv_u / &safe_w).mapv(|v| v.clamp(-MAX_STEP, MAX_STEP));
    let z = &target.eta + &adjustment;
    let w = safe_w;

    // 4. Optimize λ via GCV.  Warm-start from previous values; fast-path purely
    //    parametric models with no penalties.
    let best_lambdas = if target.penalty_matrices.is_empty() {
        Array1::zeros(0)
    } else {
        run_optimization(
            &target.x_matrix,
            &z,
            &w,
            &target.penalty_matrices,
            Some(&target.lambdas),
        )?
    };

    // 5. Penalized weighted least squares: (X'WX + Σλ·S)·β = X'W·z.
    let (new_beta, cov_matrix, edf) =
        fit_pwls(&target.x_matrix, &z, &w, &target.penalty_matrices, &best_lambdas)?;

    let new_eta = target.x_matrix.dot(&new_beta.0);
    let max_diff = (&new_beta.0 - &target.beta.0)
        .iter()
        .map(|x| x.abs())
        .fold(0.0_f64, |a, b| a.max(b));
    let eta_change = (&new_eta - &target.eta).mapv(f64::abs).sum();
    let lambda_change = (&best_lambdas - &target.lambdas).mapv(f64::abs).sum();

    Ok(Update {
        beta: new_beta,
        eta: new_eta,
        lambdas: best_lambdas,
        covariance: cov_matrix,
        edf,
        max_diff,
        eta_change,
        lambda_change,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::{Gaussian, IdentityLink, LogLink};
    use crate::fitting::FittingParameter;
    use crate::terms::Term;
    use crate::types::ModelMatrix;
    use ndarray::{array, Array2};

    fn intercept_only(eta_init: f64, n: usize) -> FittingParameter {
        let x = ModelMatrix(Array2::ones((n, 1)));
        FittingParameter {
            terms: vec![Term::Intercept],
            link: Box::new(IdentityLink),
            x_matrix: x,
            penalty_matrices: vec![],
            beta: Coefficients(array![eta_init]),
            eta: Array1::from_elem(n, eta_init),
            lambdas: Array1::<f64>::zeros(0),
            covariance: None,
            edf: 0.0,
        }
    }

    fn intercept_only_log(eta_init: f64, n: usize) -> FittingParameter {
        let x = ModelMatrix(Array2::ones((n, 1)));
        FittingParameter {
            terms: vec![Term::Intercept],
            link: Box::new(LogLink),
            x_matrix: x,
            penalty_matrices: vec![],
            beta: Coefficients(array![eta_init]),
            eta: Array1::from_elem(n, eta_init),
            lambdas: Array1::<f64>::zeros(0),
            covariance: None,
            edf: 0.0,
        }
    }

    #[test]
    fn gaussian_intercept_only_step_moves_mean_toward_y_bar() {
        // For a Gaussian intercept-only model with identity link on μ, one Fisher-scoring
        // step from μ=0 should move β toward ȳ.
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0]; // ȳ = 3
        let n = y.len();
        let mut models = HashMap::new();
        models.insert("mu".to_string(), intercept_only(0.0, n));
        models.insert("sigma".to_string(), intercept_only_log(0.0, n)); // σ = 1

        let update = step(&Gaussian, &y, &models, "mu").unwrap();
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
        let mut models = HashMap::new();
        models.insert("mu".to_string(), intercept_only(0.0, n));
        models.insert("sigma".to_string(), intercept_only_log(0.0, n));

        let update = step(&Gaussian, &y, &models, "mu").unwrap();
        assert!(update.max_diff.is_finite() && update.max_diff > 0.0);
        assert!(update.eta_change.is_finite() && update.eta_change > 0.0);
        assert!(update.lambda_change.is_finite());
        assert!(update.edf.is_finite());
    }

    #[test]
    fn step_does_not_mutate_models_map() {
        // step() takes &HashMap; ensure caller's state is unchanged after the call.
        let y = array![1.0, 2.0, 3.0];
        let n = y.len();
        let mut models = HashMap::new();
        models.insert("mu".to_string(), intercept_only(0.0, n));
        models.insert("sigma".to_string(), intercept_only_log(0.0, n));

        let beta_before = models["mu"].beta.0.clone();
        let _ = step(&Gaussian, &y, &models, "mu").unwrap();
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
            link: Box::new(IdentityLink),
            x_matrix: ModelMatrix(basis),
            penalty_matrices: vec![PenaltyMatrix(penalty)],
            beta: Coefficients(Array1::zeros(n_splines)),
            eta: Array1::from_elem(n, 0.0),
            lambdas: Array1::ones(1),
            covariance: None,
            edf: 0.0,
        };
        let sigma = intercept_only_log(0.0, n);

        let mut models = HashMap::new();
        models.insert("mu".to_string(), mu);
        models.insert("sigma".to_string(), sigma);

        let update = step(&Gaussian, &y, &models, "mu").unwrap();
        assert_eq!(update.lambdas.len(), 1);
        assert!(update.lambdas[0].is_finite() && update.lambdas[0] > 0.0);
        assert!(update.edf > 0.0 && update.edf <= n_splines as f64);
        assert!(update.beta.0.iter().all(|b| b.is_finite()));
    }

    #[test]
    fn step_errors_when_target_not_in_family_parameters() {
        let y = array![1.0, 2.0];
        let n = y.len();
        let mut models = HashMap::new();
        models.insert("mu".to_string(), intercept_only(0.0, n));
        models.insert("sigma".to_string(), intercept_only_log(0.0, n));

        let err = step(&Gaussian, &y, &models, "zeta").unwrap_err();
        // family.derivatives() never produces a "zeta" entry, so we hit the missing-derivative arm.
        assert!(format!("{}", err).contains("zeta"));
    }
}
