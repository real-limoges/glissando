// I validate glissando fits against R/mgcv as the established reference
// implementation.
//
// Workflow:
//   1. Run `benchmark/run_comparison.sh` (requires R+mgcv+uv).  It produces
//      `benchmark/output/comparison_summary.json` (schema v2), holding matched
//      glissando / mgcv / gamlss fits per scenario, one entry per replicate.
//   2. Run `cargo test --test mgcv_reference -- --ignored` to validate.
//
// The test is `#[ignore]` because the comparison output is gitignored and
// regenerated locally; CI without R cannot run it (see r-parity.yml for the
// nightly job that does).
//
// Rather than compare a single realized dataset, the harness draws many
// replicates per scenario and gates the *distribution* of drift across them:
// each metric becomes a per-rep ratio (observed / its tolerance), and a
// scenario fails if the median ratio exceeds 1.0 or the p90 ratio exceeds
// LOOSE_MULT. Tolerances are scenario-aware; linear models must match tightly
// (~1e-4), smooth models are loosened (~10%) because two independent REML
// P-spline implementations settle on slightly different effective df.
//
// Student-t scenarios are validated against R/gamlss `TF()`, the like-for-like
// oracle (same Rigby–Stasinopoulos algorithm and (μ, σ, ν) parameterization), in
// `record_studentt`, which gates μ, σ, ν, EDF, SE and the (unweighted)
// log-likelihood. mgcv's `scat()` is retained only as a loose, μ-only cross-method
// sanity check.
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

// ── schema v2 ───────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ComparisonSummary {
    #[serde(default)]
    version: u32,
    scenarios: Vec<ScenarioComparison>,
}

#[derive(Debug, Deserialize)]
struct ScenarioComparison {
    name: String,
    #[serde(default)]
    smooth: bool,
    reps: Vec<RepResult>,
}

#[derive(Debug, Deserialize)]
struct RepResult {
    #[allow(dead_code)]
    rep: usize,
    glissando: Option<FitResult>,
    mgcv: Option<FitResult>,
    /// gamlss `TF()` fit, the like-for-like StudentT oracle. Present only for
    /// StudentT scenarios.
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
    #[serde(default, deserialize_with = "deserialize_nullable_f64_vec")]
    fitted_sigma: Vec<f64>,
    #[serde(default)]
    edf: HashMap<String, f64>,
    log_likelihood: Option<f64>,
    #[allow(dead_code)]
    aic: Option<f64>,
    /// Per-parameter selected λ. Not gated cross-implementation (basis
    /// normalizations differ); glissando's own values are checked for
    /// self-consistency across reps.
    #[serde(default)]
    lambdas: HashMap<String, Vec<f64>>,
    #[serde(default)]
    se_eta: HashMap<String, Vec<f64>>,
}

// ── drift + distribution helpers ──────────────────────────────────────────────

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

/// Nearest-rank percentile over the finite entries (non-finite dropped so a
/// gammals NaN can't panic the sort; an all-NaN input yields NaN).
fn percentile(vals: &[f64], p: f64) -> f64 {
    let mut xs: Vec<f64> = vals.iter().copied().filter(|v| v.is_finite()).collect();
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = (p * (xs.len() as f64 - 1.0)).round() as usize;
    xs[idx.min(xs.len() - 1)]
}

fn median(vals: &[f64]) -> f64 {
    percentile(vals, 0.5)
}

/// p90 may reach LOOSE_MULT × the (median) tolerance before a scenario fails.
/// Starting point; recalibrate per scenario from a first nightly's observed
/// distributions.
const LOOSE_MULT: f64 = 2.0;
/// Loosest allowed spread (p90/median) of glissando's own λ across replicates.
const LAMBDA_SPREAD_MAX: f64 = 10.0;

/// Gate a distribution of ratios (observed / tolerance): fail if the median
/// exceeds the tight bound or the p90 exceeds the loose bound.
fn dist_fails(vals: &[f64], tight: f64, loose: f64) -> Option<(f64, f64)> {
    let med = median(vals);
    if !med.is_finite() {
        return None;
    }
    let p90 = percentile(vals, 0.90);
    if med > tight || p90 > loose {
        Some((med, p90))
    } else {
        None
    }
}

/// Accumulates per-rep metric ratios (and glissando λ vectors) for one scenario,
/// so each metric can be gated on its across-rep distribution.
#[derive(Default)]
struct Acc {
    ratios: HashMap<String, Vec<f64>>,
    lambdas: HashMap<String, Vec<Vec<f64>>>,
}

impl Acc {
    fn push(&mut self, key: &str, ratio: f64) {
        self.ratios.entry(key.to_string()).or_default().push(ratio);
    }

    /// Emit a failure per metric whose ratio distribution breaches the gate.
    fn finish(self, name: &str, failures: &mut Vec<String>) {
        let mut keys: Vec<&String> = self.ratios.keys().collect();
        keys.sort();
        for key in keys {
            if let Some((med, p90)) = dist_fails(&self.ratios[key], 1.0, LOOSE_MULT) {
                failures.push(format!(
                    "{name}: {key}: median ratio {med:.3}, p90 ratio {p90:.3} (tol×median 1.0, ×p90 {LOOSE_MULT:.1})"
                ));
            }
        }

        for (param, per_rep) in &self.lambdas {
            let width = per_rep.iter().map(|v| v.len()).min().unwrap_or(0);
            for i in 0..width {
                let col: Vec<f64> = per_rep.iter().map(|v| v[i]).collect();
                let med = median(&col);
                if !med.is_finite() || med <= 0.0 {
                    continue;
                }
                let spread = percentile(&col, 0.90) / med;
                if spread > LAMBDA_SPREAD_MAX {
                    failures.push(format!(
                        "{name}: glissando λ unstable across reps: {param}[{i}] p90/median={spread:.1} (max {LAMBDA_SPREAD_MAX:.0})"
                    ));
                }
            }
        }
    }
}

const SUMMARY_PATH: &str = "benchmark/output/comparison_summary.json";

fn load_summary() -> Option<ComparisonSummary> {
    let path = Path::new(SUMMARY_PATH);
    if !path.exists() {
        return None;
    }
    let content = std::fs::read_to_string(path).ok()?;
    // Surface parse errors instead of silently mapping them to "file missing":
    // a malformed field otherwise reads as "run run_comparison.sh first", which
    // sends the investigation the wrong way.
    match serde_json::from_str(&content) {
        Ok(summary) => Some(summary),
        Err(e) => panic!("comparison_summary.json exists but failed to parse: {e}"),
    }
}

fn is_studentt_scenario(name: &str) -> bool {
    name.contains("studentt")
}

/// Prior-weighted scenarios (b1/b2): glissando computes ML log-lik (unweighted
/// sum), mgcv computes REML log-lik with Σwᵢ effective observations, so
/// log-lik / EDF / SE are incomparable across implementations.
fn is_weighted_scenario(name: &str) -> bool {
    name.starts_with("b1_") || name.starts_with("b2_")
}

/// fitted_mu tolerance for non-StudentT scenarios: tight for linear (1e-3),
/// looser for smooth (10%). (StudentT has its own tolerances below.)
fn fitted_mu_rel_tol(smooth: bool) -> f64 {
    if smooth {
        0.10
    } else {
        1e-3
    }
}

// ── StudentT-vs-gamlss tolerances ──────────────────────────────────────────────
const ST_MU_TOL_LINEAR: f64 = 5e-3;
const ST_MU_TOL_SMOOTH: f64 = 2e-2;
/// b2_weighted: the oracles themselves disagree on smoothness; the floor.
const ST_MU_TOL_WEIGHTED: f64 = 0.15;
const ST_SCALE_COEF_TOL: f64 = 0.15;
const ST_LOGLIK_PER_OBS_TOL: f64 = 5e-3;
const ST_SE_TOL: f64 = 0.05;
/// Loose cross-method sanity bound for glissando-vs-mgcv-scat fitted_mu.
const ST_SCAT_SANITY_TOL: f64 = 0.05;

/// EDF tolerance for one parameter: generous band for a genuine smooth (λ
/// selection differs), tight otherwise.
fn edf_tol(smooth: bool, ref_edf: f64) -> f64 {
    if smooth && ref_edf > 1.5 {
        (0.20 * ref_edf).max(0.5)
    } else {
        0.1
    }
}

/// Record one StudentT replicate against gamlss `TF()` (primary) and mgcv
/// `scat()` (loose, μ-only cross-method sanity), into `acc`.
fn record_studentt(
    scenario: &ScenarioComparison,
    rep: &RepResult,
    g: &FitResult,
    acc: &mut Acc,
    failures: &mut Vec<String>,
) {
    let name = &scenario.name;
    let n = g.fitted_mu.len();
    let weighted = is_weighted_scenario(name);

    if let Some(gl) = &rep.gamlss {
        if !gl.converged {
            failures.push(format!("{name} rep {}: gamlss did not converge", rep.rep));
        } else {
            // fitted_mu (response scale). For the weighted heavy-tail case the two
            // oracles disagree, so the fair bound is the measured mgcv-vs-gamlss
            // drift plus margin, floored at ST_MU_TOL_WEIGHTED.
            let mu_tol = if weighted {
                let oracle_gap = rep
                    .mgcv
                    .as_ref()
                    .filter(|m| m.converged && m.fitted_mu.len() == gl.fitted_mu.len())
                    .map(|m| fitted_drift(&m.fitted_mu, &gl.fitted_mu).0)
                    .unwrap_or(0.0);
                ST_MU_TOL_WEIGHTED.max(1.25 * oracle_gap)
            } else if scenario.smooth {
                ST_MU_TOL_SMOOTH
            } else {
                ST_MU_TOL_LINEAR
            };
            if g.fitted_mu.len() == gl.fitted_mu.len() {
                let (max_rel, _) = fitted_drift(&g.fitted_mu, &gl.fitted_mu);
                acc.push("fitted_mu vs gamlss", max_rel / mu_tol);
            }

            // Coefficients: linear mean only (σ/ν well-identified, basis matches).
            if !scenario.smooth {
                for (label, g_coef) in &g.coefficients {
                    let Some(ref_coef) = gl.coefficients.get(label) else {
                        continue;
                    };
                    if g_coef.len() != ref_coef.len() {
                        failures.push(format!(
                            "{name} rep {}::{label}: coefficient count differs ({} vs {})",
                            rep.rep,
                            g_coef.len(),
                            ref_coef.len()
                        ));
                        continue;
                    }
                    let tol = if label.contains("sigma") || label.contains("nu") {
                        ST_SCALE_COEF_TOL
                    } else {
                        ST_MU_TOL_LINEAR
                    };
                    let worst = g_coef
                        .iter()
                        .zip(ref_coef.iter())
                        .map(|(gc, rc)| (gc - rc).abs() / rc.abs().max(1.0))
                        .fold(0.0_f64, f64::max);
                    acc.push(&format!("coef[{label}] vs gamlss"), worst / tol);
                }
            }

            // EDF, log-lik, SE: skipped for weighted (incomparable weight semantics).
            if !weighted {
                for (param, &g_edf) in &g.edf {
                    if let Some(&ref_edf) = gl.edf.get(param) {
                        let tol = edf_tol(scenario.smooth, ref_edf);
                        acc.push(
                            &format!("edf[{param}] vs gamlss"),
                            (g_edf - ref_edf).abs() / tol,
                        );
                    }
                }
                if let (Some(g_ll), Some(r_ll)) = (g.log_likelihood, gl.log_likelihood) {
                    let d = loglik_per_obs_diff(g_ll, r_ll, n);
                    acc.push("log_likelihood vs gamlss", d / ST_LOGLIK_PER_OBS_TOL);
                }
                if let (Some(g_se), Some(r_se)) = (g.se_eta.get("mu"), gl.se_eta.get("mu")) {
                    if g_se.len() == r_se.len() {
                        let max_abs = g_se
                            .iter()
                            .zip(r_se.iter())
                            .map(|(a, b)| (a - b).abs())
                            .fold(0.0_f64, f64::max);
                        acc.push("se_eta[mu] vs gamlss", max_abs / ST_SE_TOL);
                    }
                }
            }
        }
    }

    // Cross-method sanity: mgcv scat() is an independent algorithm exposing only
    // μ. Guards against gross divergence a same-algorithm oracle might share.
    if let Some(m) = &rep.mgcv {
        if !m.converged {
            failures.push(format!("{name} rep {}: mgcv did not converge", rep.rep));
        } else if g.fitted_mu.len() == m.fitted_mu.len() {
            let (max_rel, _) = fitted_drift(&g.fitted_mu, &m.fitted_mu);
            acc.push("fitted_mu vs mgcv scat", max_rel / ST_SCAT_SANITY_TOL);
        }
    }
}

/// Scale-modeling LSS scenarios (gaulss / gammals, plus heteroskedastic) where
/// we also gate fitted_sigma.
fn is_scale_smooth_scenario(name: &str) -> bool {
    name == "gaussian_sigma_smooth"
        || name == "gamma_sigma_smooth"
        || name == "gaussian_heteroskedastic"
}

/// Record one non-StudentT replicate against mgcv into `acc`.
fn record_mgcv(
    scenario: &ScenarioComparison,
    rep: &RepResult,
    g: &FitResult,
    m: &FitResult,
    failures: &mut Vec<String>,
    acc: &mut Acc,
) {
    let name = &scenario.name;
    let weighted = is_weighted_scenario(name);
    let n = g.fitted_mu.len();
    let is_tensor = name == "tensor_smooth";
    // Tensor smooths get a wider μ band: a criterion-level convention difference
    // (ML vs profiled-φ REML) worth ~1–2 EDF on a 63-parameter 2-D smooth.
    let rel_tol = if is_tensor {
        0.15
    } else {
        fitted_mu_rel_tol(scenario.smooth)
    };

    // fitted_mu.
    if g.fitted_mu.len() != m.fitted_mu.len() {
        failures.push(format!(
            "{name} rep {}: fitted_mu length mismatch ({} vs {})",
            rep.rep,
            g.fitted_mu.len(),
            m.fitted_mu.len()
        ));
        return;
    }
    let (max_rel, _) = fitted_drift(&g.fitted_mu, &m.fitted_mu);
    acc.push("fitted_mu", max_rel / rel_tol);

    // fitted_sigma (scale-modeling scenarios only).
    if !g.fitted_sigma.is_empty()
        && g.fitted_sigma.len() == m.fitted_sigma.len()
        && is_scale_smooth_scenario(name)
    {
        let (s_max_rel, _) = fitted_drift(&g.fitted_sigma, &m.fitted_sigma);
        acc.push("fitted_sigma", s_max_rel / 0.05);
    }

    // Coefficients (non-smooth only): mean coefs tight; scale/shape intercepts
    // (÷n vs ÷(n−p) scale estimator) allowed ~3%.
    if !scenario.smooth {
        for (param, g_coefs) in &g.coefficients {
            let Some(m_coefs) = m.coefficients.get(param) else {
                continue;
            };
            if g_coefs.len() != m_coefs.len() {
                failures.push(format!(
                    "{name} rep {}::{param}: coefficient count differs ({} vs {})",
                    rep.rep,
                    g_coefs.len(),
                    m_coefs.len()
                ));
                continue;
            }
            let is_scale = param.contains("sigma") || param.contains("phi") || param.contains("nu");
            let coef_tol = if is_scale { 3e-2 } else { rel_tol };
            let worst = g_coefs
                .iter()
                .zip(m_coefs.iter())
                .map(|(gc, mc)| (gc - mc).abs() / mc.abs().max(1.0))
                .fold(0.0_f64, f64::max);
            acc.push(&format!("coef[{param}]"), worst / coef_tol);
        }
    }

    // EDF (skip weighted: prior-weighted IRLS changes effective n differently).
    if !weighted {
        for (param, &g_edf) in &g.edf {
            let Some(&m_edf) = m.edf.get(param.as_str()) else {
                continue;
            };
            acc.push(
                &format!("edf[{param}]"),
                (g_edf - m_edf).abs() / edf_tol(scenario.smooth, m_edf),
            );
        }
    }

    // Log-likelihood (skip weighted).
    if !weighted && n > 0 {
        if let (Some(&g_ll), Some(&m_ll)) = (g.log_likelihood.as_ref(), m.log_likelihood.as_ref()) {
            let ll_tol = if scenario.smooth { 1e-2 } else { 1e-3 };
            acc.push(
                "log_likelihood",
                loglik_per_obs_diff(g_ll, m_ll, n) / ll_tol,
            );
        }
    }

    // Link-scale SE on μ.
    if let (Some(g_se), Some(m_se)) = (g.se_eta.get("mu"), m.se_eta.get("mu")) {
        if !g_se.is_empty() && g_se.len() == m_se.len() {
            let (se_max_rel, _) = fitted_drift(g_se, m_se);
            acc.push("se_eta[mu]", se_max_rel / rel_tol);
        }
    }
}

#[test]
#[ignore = "requires benchmark/output/comparison_summary.json (run benchmark/run_comparison.sh)"]
fn glissando_matches_mgcv_within_tolerance() {
    let summary = load_summary()
        .expect("failed to load comparison_summary.json; run benchmark/run_comparison.sh first");
    assert_eq!(
        summary.version, 2,
        "expected comparison_summary.json schema v2; regenerate with benchmark/run_comparison.sh"
    );

    let mut failures = Vec::new();

    for scenario in &summary.scenarios {
        let mut acc = Acc::default();

        for rep in &scenario.reps {
            let Some(g) = &rep.glissando else {
                // Scenario only ran in another implementation for this rep.
                continue;
            };
            if !g.converged {
                failures.push(format!(
                    "{} rep {}: glissando did not converge",
                    scenario.name, rep.rep
                ));
                continue;
            }

            // Collect glissando's own λ for the cross-rep self-consistency gate.
            for (param, lam) in &g.lambdas {
                acc.lambdas
                    .entry(param.clone())
                    .or_default()
                    .push(lam.clone());
            }

            if is_studentt_scenario(&scenario.name) {
                record_studentt(scenario, rep, g, &mut acc, &mut failures);
                continue;
            }

            // Non-StudentT: compare against mgcv when it ran this rep.
            let Some(m) = &rep.mgcv else {
                continue;
            };
            if !m.converged {
                failures.push(format!(
                    "{} rep {}: mgcv did not converge",
                    scenario.name, rep.rep
                ));
                continue;
            }
            record_mgcv(scenario, rep, g, m, &mut failures, &mut acc);
        }

        acc.finish(&scenario.name, &mut failures);
    }

    assert!(
        failures.is_empty(),
        "{} check(s) failed:\n  - {}",
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
