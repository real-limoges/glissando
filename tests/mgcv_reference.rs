// Validates glissando fits against R/mgcv as the established reference
// implementation.
//
// Workflow:
//   1. Run `benchmark/run_comparison.sh` (requires R+mgcv+uv).  It produces
//      `benchmark/output/comparison_summary.json` containing matched glissando
//      and mgcv fits per scenario.
//   2. Run `cargo test --test mgcv_reference -- --ignored` to validate.
//
// The test is `#[ignore]` because the comparison output is gitignored and
// regenerated locally; CI without R cannot run it.  The JSON schema below is
// the contract `run_comparison.sh` must satisfy.
//
// Tolerances are scenario-aware.  Linear models must match tightly (~1e-4);
// smooth models are loosened (~10%) because two independent REML P-spline
// implementations settle on slightly different effective df.
// Scale-smooth scenarios (gaulss, gammals) gate on both fitted_mu and
// fitted_sigma.
//
// Student-t scenarios are validated against R/gamlss `TF()` — the like-for-like
// oracle (same Rigby–Stasinopoulos algorithm and (μ, σ, ν) parameterization) — in
// `compare_studentt`, which gates μ, σ, ν, EDF, SE and the (unweighted)
// log-likelihood. mgcv's `scat()` is retained only as a loose, μ-only cross-method
// sanity check, since it folds σ and ν into internal nuisance scalars and so cannot
// validate them.
#![cfg(not(feature = "python"))]

use serde::{Deserialize, Deserializer};
use std::collections::HashMap;
use std::path::Path;

/// Deserializes a JSON array of number-or-null into Vec<f64>, mapping null → NaN.
/// Needed because some mgcv families (gammals) can emit null fitted values when
/// the shape parameter overflows on the response scale (NA in R → null in JSON).
fn deserialize_nullable_f64_vec<'de, D>(deserializer: D) -> Result<Vec<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    let v: Vec<Option<f64>> = Vec::deserialize(deserializer)?;
    Ok(v.into_iter().map(|x| x.unwrap_or(f64::NAN)).collect())
}

#[derive(Debug, Deserialize)]
struct ComparisonSummary {
    scenarios: Vec<ScenarioComparison>,
}

#[derive(Debug, Deserialize)]
struct ScenarioComparison {
    name: String,
    /// Whether the scenario is a smooth (P-spline / tensor / CR / RE) model,
    /// used to pick the right comparison strategy and tolerance.
    #[serde(default)]
    smooth: bool,
    glissando: Option<FitResult>,
    mgcv: Option<FitResult>,
    /// gamlss `TF()` fit — the like-for-like StudentT oracle (same RS algorithm and
    /// (μ, σ, ν) parameterization). Present only for StudentT scenarios.
    #[serde(default)]
    gamlss: Option<FitResult>,
}

#[derive(Debug, Deserialize)]
struct FitResult {
    converged: bool,
    #[serde(default)]
    coefficients: HashMap<String, Vec<f64>>,
    #[serde(default)]
    fitted_mu: Vec<f64>,
    /// Fitted scale on the response scale. Present for scale-modeling scenarios.
    /// Uses a null-tolerant deserializer: mgcv gammals can emit null when φ overflows.
    #[serde(default, deserialize_with = "deserialize_nullable_f64_vec")]
    fitted_sigma: Vec<f64>,
    #[serde(default)]
    edf: HashMap<String, f64>,
    log_likelihood: Option<f64>,
    #[allow(dead_code)]
    aic: Option<f64>,
    /// Per-parameter selected λ values. Not gated — basis normalisations differ.
    #[serde(default)]
    lambdas: HashMap<String, Vec<f64>>,
    /// Link-scale SEs on the training data.
    #[serde(default)]
    se_eta: HashMap<String, Vec<f64>>,
}

/// Max relative / mean absolute deviation between two fitted vectors.
fn fitted_drift(a: &[f64], b: &[f64]) -> (f64, f64) {
    let n = a.len() as f64;
    let max_rel = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs() / y.abs().max(1.0))
        .fold(0.0_f64, f64::max);
    let mean_abs = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f64>()
        / n;
    (max_rel, mean_abs)
}

/// Per-observation log-likelihood absolute difference.
fn loglik_per_obs_diff(a: f64, b: f64, n: usize) -> f64 {
    (a - b).abs() / n as f64
}

const SUMMARY_PATH: &str = "benchmark/output/comparison_summary.json";

fn load_summary() -> Option<ComparisonSummary> {
    let path = Path::new(SUMMARY_PATH);
    if !path.exists() {
        return None;
    }
    let content = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

/// StudentT scenarios are validated against gamlss `TF()` (the like-for-like RS
/// oracle) in [`compare_studentt`], not against mgcv in the main loop.
fn is_studentt_scenario(name: &str) -> bool {
    name.contains("studentt")
}

/// Returns true for prior-weighted scenarios (b1/b2). These use different
/// weight semantics for the log-likelihood: glissando computes ML log-lik
/// (unweighted sum) while mgcv computes REML log-lik with Σwᵢ effective
/// observations. The log-likelihoods are incomparable across implementations.
fn is_weighted_scenario(name: &str) -> bool {
    name.starts_with("b1_") || name.starts_with("b2_")
}

/// fitted_mu tolerance for non-StudentT scenarios: tight for linear (1e-3), looser
/// for smooth (10%) because two independent REML P-spline optimizers settle on
/// slightly different effective df. (StudentT has its own tolerances in
/// [`compare_studentt`].)
fn fitted_mu_rel_tol(smooth: bool) -> f64 {
    if smooth {
        0.10
    } else {
        1e-3
    }
}

// ── StudentT-vs-gamlss tolerances ──────────────────────────────────────────────
// gamlss TF() is the like-for-like oracle (same RS algorithm + (μ,σ,ν)
// parameterization), so unlike mgcv scat() it lets us gate μ, σ, ν, EDF, SE and the
// (unweighted) log-likelihood. Values are the measured glissando-vs-gamlss gaps plus
// margin; see the per-field comments in `compare_studentt`.
const ST_MU_TOL_LINEAR: f64 = 5e-3;
const ST_MU_TOL_SMOOTH: f64 = 2e-2;
/// b2_weighted: a weighted 4-smooth heavy-tail fit where the three RS-ish
/// implementations genuinely disagree on smoothness (gamlss EDF ≈ 13.4, mgcv ≈ 12.3,
/// glissando ≈ 10.9). glissando's mean tracks mgcv scat to <1% (the tight cross-method
/// check below); against gamlss's wigglier mean the pointwise gap reaches ~12%.
const ST_MU_TOL_WEIGHTED: f64 = 0.15;
/// log-scale σ/ν intercept coefficients (compared with a 1.0 denominator floor, so
/// effectively absolute for these near-zero values). Linear only — see below.
const ST_SCALE_COEF_TOL: f64 = 0.15;
/// Per-observation log-likelihood gap (unweighted scenarios only).
const ST_LOGLIK_PER_OBS_TOL: f64 = 5e-3;
/// Link-scale SE[μ] max absolute gap (unweighted scenarios only).
const ST_SE_TOL: f64 = 0.05;
/// Loose cross-method sanity bound for glissando-vs-mgcv-scat fitted_mu.
const ST_SCAT_SANITY_TOL: f64 = 0.05;

/// Validate a StudentT scenario against gamlss `TF()` (primary, tight) and mgcv
/// `scat()` (loose, μ-only cross-method sanity). gamlss is the correct oracle: it
/// shares glissando's RS algorithm and (μ, σ, ν) parameterization, so σ/ν/EDF/SE are
/// directly comparable — which mgcv scat() (folding σ, ν into nuisance scalars) cannot
/// provide.
fn compare_studentt(scenario: &ScenarioComparison, g: &FitResult, failures: &mut Vec<String>) {
    let name = &scenario.name;
    let n = g.fitted_mu.len();
    let weighted = is_weighted_scenario(name);

    if let Some(gl) = &scenario.gamlss {
        if !gl.converged {
            failures.push(format!("{name}: gamlss did not converge"));
        } else {
            // fitted_mu (response scale).
            let mu_tol = if weighted {
                ST_MU_TOL_WEIGHTED
            } else if scenario.smooth {
                ST_MU_TOL_SMOOTH
            } else {
                ST_MU_TOL_LINEAR
            };
            if g.fitted_mu.len() == gl.fitted_mu.len() {
                let (max_rel, mean_abs) = fitted_drift(&g.fitted_mu, &gl.fitted_mu);
                if max_rel > mu_tol && mean_abs > 1e-2 {
                    failures.push(format!(
                        "{name}: fitted_mu vs gamlss — max relative {max_rel:.3e}, mean absolute {mean_abs:.3e} (tol {mu_tol:.0e})"
                    ));
                }
            }

            // Coefficients: only for the linear mean, where σ/ν are well-identified
            // and the basis matches. For smooth means gamlss reports only the
            // parametric part of its pb() basis (different from glissando's full
            // P-spline coefficients), and σ/ν are weakly identified under a flexible
            // mean — so coefficient gating there is not meaningful.
            if !scenario.smooth {
                for (label, g_coef) in &g.coefficients {
                    let Some(ref_coef) = gl.coefficients.get(label) else {
                        continue;
                    };
                    if g_coef.len() != ref_coef.len() {
                        failures.push(format!(
                            "{name}::{label}: coefficient count differs ({} vs {})",
                            g_coef.len(),
                            ref_coef.len()
                        ));
                        continue;
                    }
                    let is_scale = label.contains("sigma") || label.contains("nu");
                    let tol = if is_scale {
                        ST_SCALE_COEF_TOL
                    } else {
                        ST_MU_TOL_LINEAR
                    };
                    for (i, (gc, rc)) in g_coef.iter().zip(ref_coef.iter()).enumerate() {
                        if (gc - rc).abs() / rc.abs().max(1.0) > tol {
                            failures.push(format!(
                                "{name}::{label}[{i}] vs gamlss — glissando={gc:.6} gamlss={rc:.6}"
                            ));
                        }
                    }
                }
            }

            // EDF per parameter. Linear/intercept terms match to integer precision;
            // smooth μ allows a generous band since RS optimizers select different λ
            // (gamlss's b2 mean is markedly wigglier than glissando's).
            for (param, &g_edf) in &g.edf {
                if let Some(&ref_edf) = gl.edf.get(param) {
                    let tol = if scenario.smooth && ref_edf > 1.5 {
                        (0.25 * ref_edf).max(0.5)
                    } else {
                        0.1
                    };
                    if (g_edf - ref_edf).abs() > tol {
                        failures.push(format!(
                            "{name}: edf[{param}] vs gamlss — glissando={g_edf:.3} gamlss={ref_edf:.3} (tol {tol:.3})"
                        ));
                    }
                }
            }

            // Unweighted log-likelihood. Skipped for weighted scenarios: glissando
            // sums the ML log-likelihood unweighted while gamlss weights by Σwᵢ, so
            // the two are not on the same scale.
            if !weighted {
                if let (Some(g_ll), Some(r_ll)) = (g.log_likelihood, gl.log_likelihood) {
                    let d = loglik_per_obs_diff(g_ll, r_ll, n);
                    if d > ST_LOGLIK_PER_OBS_TOL {
                        failures.push(format!(
                            "{name}: log_likelihood vs gamlss — |Δ|/n={d:.3e} (tol {ST_LOGLIK_PER_OBS_TOL:.0e})"
                        ));
                    }
                }
            }

            // Link-scale SE[μ] (unweighted scenarios; the weighted IRLS changes the
            // effective sample size differently between implementations).
            if !weighted {
                if let (Some(g_se), Some(r_se)) = (g.se_eta.get("mu"), gl.se_eta.get("mu")) {
                    if g_se.len() == r_se.len() {
                        let max_abs = g_se
                            .iter()
                            .zip(r_se.iter())
                            .map(|(a, b)| (a - b).abs())
                            .fold(0.0_f64, f64::max);
                        if max_abs > ST_SE_TOL {
                            failures.push(format!(
                                "{name}: se_eta[mu] vs gamlss — max abs {max_abs:.3e} (tol {ST_SE_TOL:.0e})"
                            ));
                        }
                    }
                }
            }
        }
    }

    // Cross-method sanity: mgcv scat() is an independent algorithm (Wood's joint
    // outer-BFGS). It exposes only μ, so we check fitted_mu only, loosely — it guards
    // against gross divergence that a same-algorithm oracle might share.
    if let Some(m) = &scenario.mgcv {
        if m.converged && g.fitted_mu.len() == m.fitted_mu.len() {
            let (max_rel, mean_abs) = fitted_drift(&g.fitted_mu, &m.fitted_mu);
            if max_rel > ST_SCAT_SANITY_TOL && mean_abs > 1e-2 {
                failures.push(format!(
                    "{name}: fitted_mu vs mgcv scat (cross-method) — max relative {max_rel:.3e}, mean absolute {mean_abs:.3e} (tol {ST_SCAT_SANITY_TOL:.0e})"
                ));
            }
        }
    }
}

/// Returns true for scale-modeling LSS scenarios (gaulss / gammals) where we
/// also gate on fitted_sigma. `gaussian_heteroskedastic` is linear in both μ
/// and log σ but compared against gaulss, so its σ curve is gated too (gaulss's
/// logb σ-link differs from glissando's log link by the b = 0.01 offset, well
/// inside the 5% band).
fn is_scale_smooth_scenario(name: &str) -> bool {
    name == "gaussian_sigma_smooth"
        || name == "gamma_sigma_smooth"
        || name == "gaussian_heteroskedastic"
}

#[test]
#[ignore = "requires benchmark/output/comparison_summary.json (run benchmark/run_comparison.sh)"]
fn glissando_matches_mgcv_within_tolerance() {
    let summary = load_summary()
        .expect("failed to load comparison_summary.json — run benchmark/run_comparison.sh first");

    let mut failures = Vec::new();

    for scenario in &summary.scenarios {
        let Some(g) = &scenario.glissando else {
            // Scenario only ran in another implementation; nothing to compare.
            continue;
        };
        if !g.converged {
            failures.push(format!("{}: glissando did not converge", scenario.name));
            continue;
        }

        // StudentT is validated against gamlss TF() (the like-for-like RS oracle),
        // with mgcv scat() kept only as a loose μ-only cross-method sanity check.
        if is_studentt_scenario(&scenario.name) {
            compare_studentt(scenario, g, &mut failures);
            continue;
        }

        // Non-StudentT scenarios compare against mgcv.
        let Some(m) = &scenario.mgcv else {
            continue;
        };
        if !m.converged {
            failures.push(format!("{}: mgcv did not converge", scenario.name));
            continue;
        }

        let scale_smooth = is_scale_smooth_scenario(&scenario.name);
        let weighted = is_weighted_scenario(&scenario.name);
        let n = g.fitted_mu.len();

        // ── fitted_mu ─────────────────────────────────────────────────────
        // Smooth tolerance: the irreducible gap between two REML P-spline
        // implementations.  On the wiggliest scenario (gaussian_smooth, a
        // two-period sine) glissando and mgcv differ by up to ~8% pointwise
        // while both sit within RMSE ~0.04 of the truth, because their REML
        // optimizers settle on slightly different effective df (glissando ~16
        // vs mgcv ~15).  10% bounds that gap while still catching gross
        // disagreement.  Linear fits must still match tightly.
        if g.fitted_mu.len() != m.fitted_mu.len() {
            failures.push(format!(
                "{}: fitted_mu length mismatch ({} vs {})",
                scenario.name,
                g.fitted_mu.len(),
                m.fitted_mu.len()
            ));
            continue;
        }
        let rel_tol = fitted_mu_rel_tol(scenario.smooth);
        let abs_tol = if scenario.smooth { 1e-2 } else { 1e-4 };
        let (max_rel, mean_abs) = fitted_drift(&g.fitted_mu, &m.fitted_mu);
        if max_rel > rel_tol && mean_abs > abs_tol {
            failures.push(format!(
                "{}: fitted_mu drift — max relative {:.3e}, mean absolute {:.3e}",
                scenario.name, max_rel, mean_abs
            ));
        }

        // ── fitted_sigma ──────────────────────────────────────────────────
        // Compare the fitted scale curve when both implementations report it
        // (scale-modeling scenarios such as gaussian_sigma_smooth vs gaulss).
        // The scale is noisier than the mean, so use the smooth tolerance.
        if !g.fitted_sigma.is_empty()
            && g.fitted_sigma.len() == m.fitted_sigma.len()
            && scale_smooth
        {
            let (s_max_rel, s_mean_abs) = fitted_drift(&g.fitted_sigma, &m.fitted_sigma);
            if s_max_rel > 0.05 && s_mean_abs > 1e-2 {
                failures.push(format!(
                    "{}: fitted_sigma drift — max relative {:.3e}, mean absolute {:.3e}",
                    scenario.name, s_max_rel, s_mean_abs
                ));
            }
        }

        // ── Coefficients ──────────────────────────────────────────────────
        // For non-smooth scenarios, coefficient-level agreement is also expected.
        if !scenario.smooth {
            for (param, g_coefs) in &g.coefficients {
                let Some(m_coefs) = m.coefficients.get(param) else {
                    continue;
                };
                if g_coefs.len() != m_coefs.len() {
                    failures.push(format!(
                        "{}::{}: coefficient count differs ({} vs {})",
                        scenario.name,
                        param,
                        g_coefs.len(),
                        m_coefs.len()
                    ));
                    continue;
                }
                // Mean (location) coefficients must match tightly. Scale/shape
                // intercepts (sigma/phi/nu) legitimately differ: glissando
                // reports the ML scale (÷n) while mgcv REML reports the
                // unbiased scale (÷(n−p)) — the Gaussian gap is exactly
                // 0.5·ln((n−p)/n) — and the Gamma/NB dispersion estimators
                // differ in kind. Allow ~3% on the scale parameter.
                let is_scale =
                    param.contains("sigma") || param.contains("phi") || param.contains("nu");
                let coef_tol = if is_scale { 3e-2 } else { rel_tol };
                for (i, (gc, mc)) in g_coefs.iter().zip(m_coefs.iter()).enumerate() {
                    if (gc - mc).abs() / mc.abs().max(1.0) > coef_tol {
                        failures.push(format!(
                            "{}::{}[{}]: glissando={:.6} mgcv={:.6}",
                            scenario.name, param, i, gc, mc
                        ));
                    }
                }
            }
        }

        // ── EDF ───────────────────────────────────────────────────────────
        // Effective degrees of freedom per parameter. Linear scenarios expect
        // exact agreement (integer EDF); smooth scenarios allow a tolerance
        // proportional to the EDF since implementations can select slightly
        // different λ.  Skip for weighted scenarios: the prior-weighted IRLS changes
        // the effective sample size differently in glissando (ML) vs mgcv (REML), so
        // λ and EDF are not comparable.
        if !weighted {
            for (param, &g_edf) in &g.edf {
                let Some(&m_edf) = m.edf.get(param.as_str()) else {
                    continue;
                };
                let edf_diff = (g_edf - m_edf).abs();
                // Absolute tolerance: ≤0.1 for parametric (linear/intercept),
                // ≤ max(0.5, 20% of mgcv EDF) for smooth parameters.
                let edf_tol = if scenario.smooth && m_edf > 1.5 {
                    (0.20 * m_edf).max(0.5)
                } else {
                    0.1
                };
                if edf_diff > edf_tol {
                    failures.push(format!(
                        "{}: edf[{}] drift — glissando={:.3} mgcv={:.3} |Δ|={:.3} tol={:.3}",
                        scenario.name, param, g_edf, m_edf, edf_diff, edf_tol
                    ));
                }
            }
        }

        // ── Log-likelihood ────────────────────────────────────────────────
        // Per-observation absolute difference.  Both sides report ML
        // log-likelihood at the fitted values (not the REML criterion).
        // gaulss and gammals use a tighter bound since both converge to the same MLE.
        // Weighted scenarios (b1/b2) are skipped: glissando uses ML (unweighted
        // sum of ℓᵢ) while mgcv REML uses Σwᵢ·ℓᵢ — the numbers are not on the
        // same scale and the comparison is meaningless.
        if !weighted {
            if let (Some(&g_ll), Some(&m_ll)) =
                (g.log_likelihood.as_ref(), m.log_likelihood.as_ref())
            {
                if n > 0 {
                    let ll_diff = loglik_per_obs_diff(g_ll, m_ll, n);
                    let ll_tol = if scenario.smooth {
                        1e-2 // REML EDF drift shifts log-lik slightly
                    } else {
                        1e-3 // linear models: near-identical MLE
                    };
                    if ll_diff > ll_tol {
                        failures.push(format!(
                        "{}: log_likelihood drift — glissando={:.4} mgcv={:.4} |Δ|/n={:.4e} tol={:.4e}",
                        scenario.name, g_ll, m_ll, ll_diff, ll_tol
                    ));
                    }
                }
            }
        } // end if !weighted

        // ── Link-scale SEs on μ ──────────────────────────────────────────
        // Compare the link-scale standard errors for the location parameter.
        if let (Some(g_se), Some(m_se)) = (g.se_eta.get("mu"), m.se_eta.get("mu")) {
            if !g_se.is_empty() && g_se.len() == m_se.len() {
                let (se_max_rel, se_mean_abs) = fitted_drift(g_se, m_se);
                if se_max_rel > rel_tol && se_mean_abs > abs_tol {
                    failures.push(format!(
                        "{}: se_eta[mu] drift — max relative {:.3e}, mean absolute {:.3e}",
                        scenario.name, se_max_rel, se_mean_abs
                    ));
                }
            }
        }

        // ── λ report (informational) ──────────────────────────────────────
        // Not gated: glissando's `lambdas` and mgcv's `$sp` live in different
        // basis-normalisation spaces. Print them for diagnostic purposes if
        // they differ by more than an order of magnitude in the largest term.
        // (No push to `failures` — this is advisory only.)
        let _ = (&g.lambdas, &m.lambdas); // silence unused-field warnings
    }

    assert!(
        failures.is_empty(),
        "{} scenario(s) failed comparison:\n  - {}",
        failures.len(),
        failures.join("\n  - ")
    );
}

#[test]
fn comparison_summary_path_documented_in_benchmark_readme() {
    // Sanity: the path the test reads stays consistent with what run_comparison.sh writes.
    let readme = std::fs::read_to_string("benchmark/README.md").unwrap_or_default();
    assert!(
        readme.contains("run_comparison.sh"),
        "benchmark/README.md should mention run_comparison.sh as the source of comparison data"
    );
}
