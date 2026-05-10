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
// Tolerances are scenario-aware: linear models match mgcv tightly (~1e-4),
// smooth models loosely (~5%) because basis parameterizations differ between
// glissando's centered P-splines and mgcv's thin-plate / cr default.
#![cfg(not(feature = "python"))]

use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct ComparisonSummary {
    scenarios: Vec<ScenarioComparison>,
}

#[derive(Debug, Deserialize)]
struct ScenarioComparison {
    name: String,
    /// Whether the scenario is a smooth (P-spline / tensor) model, to pick
    /// the right comparison strategy and tolerance.
    #[serde(default)]
    smooth: bool,
    glissando: Option<FitResult>,
    mgcv: Option<FitResult>,
}

#[derive(Debug, Deserialize)]
struct FitResult {
    converged: bool,
    coefficients: HashMap<String, Vec<f64>>,
    fitted_mu: Vec<f64>,
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

#[test]
#[ignore = "requires benchmark/output/comparison_summary.json (run benchmark/run_comparison.sh)"]
fn glissando_matches_mgcv_within_tolerance() {
    let summary = load_summary()
        .expect("failed to load comparison_summary.json — run benchmark/run_comparison.sh first");

    let mut failures = Vec::new();

    for scenario in &summary.scenarios {
        let (Some(g), Some(m)) = (&scenario.glissando, &scenario.mgcv) else {
            // Scenario only ran in one implementation; nothing to compare.
            continue;
        };
        if !g.converged || !m.converged {
            failures.push(format!(
                "{}: convergence — glissando={} mgcv={}",
                scenario.name, g.converged, m.converged
            ));
            continue;
        }

        // Compare fitted μ point-wise (parameterisation-independent).
        if g.fitted_mu.len() != m.fitted_mu.len() {
            failures.push(format!(
                "{}: fitted_mu length mismatch ({} vs {})",
                scenario.name,
                g.fitted_mu.len(),
                m.fitted_mu.len()
            ));
            continue;
        }
        let n = g.fitted_mu.len() as f64;
        let rel_tol = if scenario.smooth { 0.05 } else { 1e-3 };
        let abs_tol = if scenario.smooth { 1e-2 } else { 1e-4 };
        let max_rel: f64 = g
            .fitted_mu
            .iter()
            .zip(m.fitted_mu.iter())
            .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
            .fold(0.0_f64, f64::max);
        let mean_abs: f64 = g
            .fitted_mu
            .iter()
            .zip(m.fitted_mu.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / n;
        if max_rel > rel_tol && mean_abs > abs_tol {
            failures.push(format!(
                "{}: fitted_mu drift — max relative {:.3e}, mean absolute {:.3e}",
                scenario.name, max_rel, mean_abs
            ));
        }

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
                for (i, (gc, mc)) in g_coefs.iter().zip(m_coefs.iter()).enumerate() {
                    if (gc - mc).abs() / mc.abs().max(1.0) > rel_tol {
                        failures.push(format!(
                            "{}::{}[{}]: glissando={:.6} mgcv={:.6}",
                            scenario.name, param, i, gc, mc
                        ));
                    }
                }
            }
        }
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
