//! GAMLSS Comparison Framework: Rust Fitting Binary
//!
//! Reads parquet data, fits models using glissando,
//! and outputs standardized JSON results for comparison with R/mgcv.
//!
//! Usage:
//!   cargo run -p glissando_benchmark --bin compare_fit -- \
//!     --data path/to/data.parquet --scenario gaussian_linear --output result.json

use glissando::distributions::{
    Beta, Binomial, Distribution, Gamma, Gaussian, NegativeBinomial, Poisson, StudentT,
};
use glissando::{DataSet, FitConfig, Formula, GamlssModel, Smooth, Term};
use ndarray::Array1;
use polars::prelude::*;
use serde::Serialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Serialize)]
struct FitResult {
    converged: bool,
    iterations: usize,
    fit_time_ms: f64,
    coefficients: HashMap<String, Vec<f64>>,
    fitted_mu: Vec<f64>,
    /// Fitted scale on the response scale. Present for scale-modeling scenarios
    /// (e.g. `gaussian_sigma_smooth` vs mgcv `gaulss`); empty otherwise.
    #[serde(default)]
    fitted_sigma: Vec<f64>,
    edf: HashMap<String, f64>,
    log_likelihood: Option<f64>,
    aic: Option<f64>,
    /// Selected smoothing parameters per distribution parameter, one value per
    /// penalty matrix. Informational only — not gated against mgcv `$sp`
    /// (different basis normalisations make the raw values incommensurable).
    lambdas: HashMap<String, Vec<f64>>,
    /// Link-scale standard errors per distribution parameter on the training
    /// data, matching mgcv `predict(type="link", se.fit=TRUE)`.
    se_eta: HashMap<String, Vec<f64>>,
    error: Option<String>,
}

fn extract_column(df: &DataFrame, name: &str) -> Array1<f64> {
    let col = df
        .column(name)
        .unwrap_or_else(|_| panic!("column '{}' not found", name));
    let cast = col
        .cast(&DataType::Float64)
        .unwrap_or_else(|_| panic!("column '{}' cannot be cast to f64", name));
    let ca = cast
        .as_materialized_series()
        .f64()
        .unwrap_or_else(|_| panic!("column '{}' cannot be read as f64", name));
    Array1::from_vec(ca.into_no_null_iter().collect())
}

fn error_result(start: Instant, e: glissando::GamlssError) -> FitResult {
    FitResult {
        converged: false,
        iterations: 0,
        fit_time_ms: start.elapsed().as_secs_f64() * 1000.0,
        coefficients: HashMap::new(),
        fitted_mu: vec![],
        fitted_sigma: vec![],
        edf: HashMap::new(),
        log_likelihood: None,
        aic: None,
        lambdas: HashMap::new(),
        se_eta: HashMap::new(),
        error: Some(e.to_string()),
    }
}

/// Uniform result builder: populates all comparison fields from a fitted model.
///
/// `coef_names` maps `(param_key, output_label)` for the `coefficients` JSON
/// field.  `sigma_param` names the parameter whose `fitted_values` populate
/// `fitted_sigma`; pass `None` for single-parameter families (Poisson, Binomial).
fn build_result<D: Distribution + ?Sized>(
    start: Instant,
    model: &GamlssModel,
    family: &D,
    y: &Array1<f64>,
    data: &DataSet,
    coef_names: &[(&str, &str)],
    sigma_param: Option<&str>,
) -> FitResult {
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    // Coefficients by requested output labels.
    let mut coefficients = HashMap::new();
    for (param_key, output_label) in coef_names {
        if let Some(fp) = model.models.get(*param_key) {
            coefficients.insert((*output_label).to_string(), fp.coefficients.0.to_vec());
        }
    }

    // Response-scale fitted values.
    let fitted_mu = model
        .models
        .get("mu")
        .map(|fp| fp.fitted_values.to_vec())
        .unwrap_or_default();
    let fitted_sigma = sigma_param
        .and_then(|p| model.models.get(p))
        .map(|fp| fp.fitted_values.to_vec())
        .unwrap_or_default();

    // EDF and λ per distribution parameter.
    let mut edf = HashMap::new();
    let mut lambdas = HashMap::new();
    for (param_name, fp) in &model.models {
        edf.insert(param_name.clone(), fp.edf);
        lambdas.insert(param_name.clone(), fp.lambdas.to_vec());
    }

    // Log-likelihood and AIC (require `family` + `y`).
    let (log_likelihood, aic) = match model.diagnostics(family, y) {
        Ok(diag) => (Some(diag.log_likelihood), Some(diag.aic)),
        Err(_) => (None, None),
    };

    // Link-scale SEs on training data.
    let se_eta = match model.predict_with_se(data, family) {
        Ok(results) => results
            .into_iter()
            .map(|(k, v)| (k, v.se_eta.to_vec()))
            .collect(),
        Err(_) => HashMap::new(),
    };

    FitResult {
        converged: model.converged(),
        iterations: model.diagnostics.iterations,
        fit_time_ms: elapsed,
        coefficients,
        fitted_mu,
        fitted_sigma,
        edf,
        log_likelihood,
        aic,
        lambdas,
        se_eta,
        error: None,
    }
}

// ─── Gaussian ────────────────────────────────────────────────────────────────

fn fit_gaussian_linear(df: &DataFrame) -> FitResult {
    let start = Instant::now();
    let y = extract_column(df, "y");
    let mut data = DataSet::new();
    data.insert_column("x", extract_column(df, "x"));

    let formula = Formula::new()
        .with_terms(
            "mu",
            vec![
                Term::Intercept,
                Term::Linear {
                    col_name: "x".to_string(),
                },
            ],
        )
        .with_terms("sigma", vec![Term::Intercept]);

    let family = Gaussian::new();
    match GamlssModel::fit(&data, &y, &formula, &family) {
        Ok(model) => build_result(
            start,
            &model,
            &family,
            &y,
            &data,
            &[("mu", "mu"), ("sigma", "log_sigma")],
            Some("sigma"),
        ),
        Err(e) => error_result(start, e),
    }
}

fn fit_gaussian_heteroskedastic(df: &DataFrame) -> FitResult {
    let start = Instant::now();
    let y = extract_column(df, "y");
    let mut data = DataSet::new();
    data.insert_column("x", extract_column(df, "x"));

    let formula = Formula::new()
        .with_terms(
            "mu",
            vec![
                Term::Intercept,
                Term::Linear {
                    col_name: "x".to_string(),
                },
            ],
        )
        .with_terms(
            "sigma",
            vec![
                Term::Intercept,
                Term::Linear {
                    col_name: "x".to_string(),
                },
            ],
        );

    let family = Gaussian::new();
    match GamlssModel::fit(&data, &y, &formula, &family) {
        Ok(model) => build_result(
            start,
            &model,
            &family,
            &y,
            &data,
            &[("mu", "mu"), ("sigma", "log_sigma")],
            Some("sigma"),
        ),
        Err(e) => error_result(start, e),
    }
}

fn fit_gaussian_smooth(df: &DataFrame) -> FitResult {
    let start = Instant::now();
    let y = extract_column(df, "y");
    let mut data = DataSet::new();
    data.insert_column("x", extract_column(df, "x"));

    let formula = Formula::new()
        .with_terms(
            "mu",
            vec![Term::Smooth(Smooth::PSpline1D {
                col_name: "x".to_string(),
                n_splines: 20,
                degree: 3,
                penalty_order: 2, range: None,
            })],
        )
        .with_terms("sigma", vec![Term::Intercept]);

    let family = Gaussian::new();
    match GamlssModel::fit(&data, &y, &formula, &family) {
        Ok(model) => build_result(
            start,
            &model,
            &family,
            &y,
            &data,
            &[("mu", "mu_smooth"), ("sigma", "log_sigma")],
            Some("sigma"),
        ),
        Err(e) => error_result(start, e),
    }
}

fn fit_gaussian_multiple(df: &DataFrame) -> FitResult {
    let start = Instant::now();
    let y = extract_column(df, "y");
    let mut data = DataSet::new();
    data.insert_column("x1", extract_column(df, "x1"));
    data.insert_column("x2", extract_column(df, "x2"));
    data.insert_column("x3", extract_column(df, "x3"));

    let formula = Formula::new()
        .with_terms(
            "mu",
            vec![
                Term::Intercept,
                Term::Linear {
                    col_name: "x1".to_string(),
                },
                Term::Linear {
                    col_name: "x2".to_string(),
                },
                Term::Linear {
                    col_name: "x3".to_string(),
                },
            ],
        )
        .with_terms("sigma", vec![Term::Intercept]);

    let family = Gaussian::new();
    match GamlssModel::fit(&data, &y, &formula, &family) {
        Ok(model) => build_result(
            start,
            &model,
            &family,
            &y,
            &data,
            &[("mu", "mu"), ("sigma", "log_sigma")],
            Some("sigma"),
        ),
        Err(e) => error_result(start, e),
    }
}

fn fit_gaussian_large(df: &DataFrame) -> FitResult {
    fit_gaussian_linear(df)
}

fn fit_gaussian_quadratic(df: &DataFrame) -> FitResult {
    fit_gaussian_smooth(df)
}

/// Smooth on the *scale* parameter: μ constant, log σ a P-spline of x. The
/// scale-smooth analogue of `gaussian_smooth`; compared against mgcv `gaulss`.
fn fit_gaussian_sigma_smooth(df: &DataFrame) -> FitResult {
    let start = Instant::now();
    let y = extract_column(df, "y");
    let mut data = DataSet::new();
    data.insert_column("x", extract_column(df, "x"));

    let formula = Formula::new()
        .with_terms("mu", vec![Term::Intercept])
        .with_terms(
            "sigma",
            vec![
                Term::Intercept,
                Term::Smooth(Smooth::PSpline1D {
                    col_name: "x".to_string(),
                    n_splines: 20,
                    degree: 3,
                    penalty_order: 2, range: None,
                }),
            ],
        );

    let family = Gaussian::new();
    match GamlssModel::fit(&data, &y, &formula, &family) {
        Ok(model) => build_result(
            start,
            &model,
            &family,
            &y,
            &data,
            &[("mu", "mu"), ("sigma", "log_sigma_smooth")],
            Some("sigma"),
        ),
        Err(e) => error_result(start, e),
    }
}

/// CR spline on μ; compared against mgcv `s(x, bs="cr", k=10)`.
fn fit_gaussian_cr_smooth(df: &DataFrame) -> FitResult {
    let start = Instant::now();
    let y = extract_column(df, "y");
    let mut data = DataSet::new();
    data.insert_column("x", extract_column(df, "x"));

    let formula = Formula::new()
        .with_terms(
            "mu",
            vec![Term::Smooth(Smooth::CrSpline1D {
                col_name: "x".to_string(),
                k: 10,
                pc: None,
                knots: vec![], // resolved at fit time
            })],
        )
        .with_terms("sigma", vec![Term::Intercept]);

    let family = Gaussian::new();
    match GamlssModel::fit(&data, &y, &formula, &family) {
        Ok(model) => build_result(
            start,
            &model,
            &family,
            &y,
            &data,
            &[("mu", "mu_cr_smooth"), ("sigma", "log_sigma")],
            Some("sigma"),
        ),
        Err(e) => error_result(start, e),
    }
}

/// Tensor-product smooth on μ; compared against mgcv `te(x1, x2, bs="ps", k=c(8,8))`.
fn fit_tensor_smooth(df: &DataFrame) -> FitResult {
    let start = Instant::now();
    let y = extract_column(df, "y");
    let mut data = DataSet::new();
    data.insert_column("x1", extract_column(df, "x1"));
    data.insert_column("x2", extract_column(df, "x2"));

    let formula = Formula::new()
        .with_terms(
            "mu",
            vec![Term::Smooth(Smooth::TensorProduct {
                col_name_1: "x1".to_string(),
                n_splines_1: 8,
                penalty_order_1: 2,
                col_name_2: "x2".to_string(),
                n_splines_2: 8,
                penalty_order_2: 2,
                degree: 3, range_1: None, range_2: None,
            })],
        )
        .with_terms("sigma", vec![Term::Intercept]);

    let family = Gaussian::new();
    match GamlssModel::fit(&data, &y, &formula, &family) {
        Ok(model) => build_result(
            start,
            &model,
            &family,
            &y,
            &data,
            &[("mu", "mu_tensor"), ("sigma", "log_sigma")],
            Some("sigma"),
        ),
        Err(e) => error_result(start, e),
    }
}

/// Linear effect + random intercept; compared against mgcv `x + s(g, bs="re")`.
fn fit_random_effect(df: &DataFrame) -> FitResult {
    let start = Instant::now();
    let y = extract_column(df, "y");
    let mut data = DataSet::new();
    data.insert_column("x", extract_column(df, "x"));
    data.insert_column("g", extract_column(df, "g"));

    let formula = Formula::new()
        .with_terms(
            "mu",
            vec![
                Term::Intercept,
                Term::Linear {
                    col_name: "x".to_string(),
                },
                Term::Smooth(Smooth::RandomEffect {
                    col_name: "g".to_string(), levels: vec![],
                }),
            ],
        )
        .with_terms("sigma", vec![Term::Intercept]);

    let family = Gaussian::new();
    match GamlssModel::fit(&data, &y, &formula, &family) {
        Ok(model) => build_result(
            start,
            &model,
            &family,
            &y,
            &data,
            &[("mu", "mu"), ("sigma", "log_sigma")],
            Some("sigma"),
        ),
        Err(e) => error_result(start, e),
    }
}

// ─── Poisson ─────────────────────────────────────────────────────────────────

fn fit_poisson_linear(df: &DataFrame) -> FitResult {
    let start = Instant::now();
    let y = extract_column(df, "y");
    let mut data = DataSet::new();
    data.insert_column("x", extract_column(df, "x"));

    let formula = Formula::new().with_terms(
        "mu",
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );

    let family = Poisson::new();
    match GamlssModel::fit(&data, &y, &formula, &family) {
        Ok(model) => build_result(start, &model, &family, &y, &data, &[("mu", "log_mu")], None),
        Err(e) => error_result(start, e),
    }
}

fn fit_poisson_smooth(df: &DataFrame) -> FitResult {
    let start = Instant::now();
    let y = extract_column(df, "y");
    let mut data = DataSet::new();
    data.insert_column("x", extract_column(df, "x"));

    let formula = Formula::new().with_terms(
        "mu",
        vec![Term::Smooth(Smooth::PSpline1D {
            col_name: "x".to_string(),
            n_splines: 20,
            degree: 3,
            penalty_order: 2, range: None,
        })],
    );

    let family = Poisson::new();
    match GamlssModel::fit(&data, &y, &formula, &family) {
        Ok(model) => build_result(
            start,
            &model,
            &family,
            &y,
            &data,
            &[("mu", "log_mu_smooth")],
            None,
        ),
        Err(e) => error_result(start, e),
    }
}

// ─── Binomial ────────────────────────────────────────────────────────────────

/// Bernoulli / logistic regression (n_trials=1); compared against mgcv
/// `binomial(link="logit")`.
fn fit_binomial_linear(df: &DataFrame) -> FitResult {
    let start = Instant::now();
    let y = extract_column(df, "y");
    let mut data = DataSet::new();
    data.insert_column("x", extract_column(df, "x"));

    let formula = Formula::new().with_terms(
        "mu",
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );

    let family = Binomial::new(1);
    match GamlssModel::fit(&data, &y, &formula, &family) {
        Ok(model) => build_result(
            start,
            &model,
            &family,
            &y,
            &data,
            &[("mu", "logit_mu")],
            None,
        ),
        Err(e) => error_result(start, e),
    }
}

/// Logistic smooth; compared against mgcv `s(x, bs="ps")` with `binomial()`.
fn fit_binomial_smooth(df: &DataFrame) -> FitResult {
    let start = Instant::now();
    let y = extract_column(df, "y");
    let mut data = DataSet::new();
    data.insert_column("x", extract_column(df, "x"));

    let formula = Formula::new().with_terms(
        "mu",
        vec![Term::Smooth(Smooth::PSpline1D {
            col_name: "x".to_string(),
            n_splines: 20,
            degree: 3,
            penalty_order: 2, range: None,
        })],
    );

    let family = Binomial::new(1);
    match GamlssModel::fit(&data, &y, &formula, &family) {
        Ok(model) => build_result(
            start,
            &model,
            &family,
            &y,
            &data,
            &[("mu", "logit_mu_smooth")],
            None,
        ),
        Err(e) => error_result(start, e),
    }
}

// ─── Gamma ───────────────────────────────────────────────────────────────────

fn fit_gamma_linear(df: &DataFrame) -> FitResult {
    let start = Instant::now();
    let y = extract_column(df, "y");
    let mut data = DataSet::new();
    data.insert_column("x", extract_column(df, "x"));

    let formula = Formula::new()
        .with_terms(
            "mu",
            vec![
                Term::Intercept,
                Term::Linear {
                    col_name: "x".to_string(),
                },
            ],
        )
        .with_terms("sigma", vec![Term::Intercept]);

    let family = Gamma::new();
    match GamlssModel::fit(&data, &y, &formula, &family) {
        Ok(model) => build_result(
            start,
            &model,
            &family,
            &y,
            &data,
            &[("mu", "log_mu"), ("sigma", "log_sigma")],
            Some("sigma"),
        ),
        Err(e) => error_result(start, e),
    }
}

fn fit_gamma_smooth(df: &DataFrame) -> FitResult {
    let start = Instant::now();
    let y = extract_column(df, "y");
    let mut data = DataSet::new();
    data.insert_column("x", extract_column(df, "x"));

    let formula = Formula::new()
        .with_terms(
            "mu",
            vec![Term::Smooth(Smooth::PSpline1D {
                col_name: "x".to_string(),
                n_splines: 20,
                degree: 3,
                penalty_order: 2, range: None,
            })],
        )
        .with_terms("sigma", vec![Term::Intercept]);

    let family = Gamma::new();
    match GamlssModel::fit(&data, &y, &formula, &family) {
        Ok(model) => build_result(
            start,
            &model,
            &family,
            &y,
            &data,
            &[("mu", "log_mu_smooth"), ("sigma", "log_sigma")],
            Some("sigma"),
        ),
        Err(e) => error_result(start, e),
    }
}

/// Constant mean, smooth CV σ; compared against mgcv `gammals()`.
/// glissando: σ is coefficient of variation (log link); mgcv: shape φ = 1/σ².
fn fit_gamma_sigma_smooth(df: &DataFrame) -> FitResult {
    let start = Instant::now();
    let y = extract_column(df, "y");
    let mut data = DataSet::new();
    data.insert_column("x", extract_column(df, "x"));

    let formula = Formula::new()
        .with_terms("mu", vec![Term::Intercept])
        .with_terms(
            "sigma",
            vec![
                Term::Intercept,
                Term::Smooth(Smooth::PSpline1D {
                    col_name: "x".to_string(),
                    n_splines: 20,
                    degree: 3,
                    penalty_order: 2, range: None,
                }),
            ],
        );

    let family = Gamma::new();
    match GamlssModel::fit(&data, &y, &formula, &family) {
        Ok(model) => build_result(
            start,
            &model,
            &family,
            &y,
            &data,
            &[("mu", "log_mu"), ("sigma", "log_sigma_smooth")],
            Some("sigma"),
        ),
        Err(e) => error_result(start, e),
    }
}

// ─── Student-t ───────────────────────────────────────────────────────────────

fn fit_studentt_linear(df: &DataFrame) -> FitResult {
    let start = Instant::now();
    let y = extract_column(df, "y");
    let mut data = DataSet::new();
    data.insert_column("x", extract_column(df, "x"));

    let formula = Formula::new()
        .with_terms(
            "mu",
            vec![
                Term::Intercept,
                Term::Linear {
                    col_name: "x".to_string(),
                },
            ],
        )
        .with_terms("sigma", vec![Term::Intercept])
        .with_terms("nu", vec![Term::Intercept]);

    let family = StudentT::new();
    match GamlssModel::fit(&data, &y, &formula, &family) {
        Ok(model) => build_result(
            start,
            &model,
            &family,
            &y,
            &data,
            &[("mu", "mu"), ("sigma", "log_sigma"), ("nu", "log_nu")],
            Some("sigma"),
        ),
        Err(e) => error_result(start, e),
    }
}

fn fit_studentt_smooth(df: &DataFrame) -> FitResult {
    let start = Instant::now();
    let y = extract_column(df, "y");
    let mut data = DataSet::new();
    data.insert_column("x", extract_column(df, "x"));

    let formula = Formula::new()
        .with_terms(
            "mu",
            vec![Term::Smooth(Smooth::PSpline1D {
                col_name: "x".to_string(),
                n_splines: 20,
                degree: 3,
                penalty_order: 2, range: None,
            })],
        )
        .with_terms("sigma", vec![Term::Intercept])
        .with_terms("nu", vec![Term::Intercept]);

    let family = StudentT::new();
    match GamlssModel::fit(&data, &y, &formula, &family) {
        Ok(model) => build_result(
            start,
            &model,
            &family,
            &y,
            &data,
            &[
                ("mu", "mu_smooth"),
                ("sigma", "log_sigma"),
                ("nu", "log_nu"),
            ],
            Some("sigma"),
        ),
        Err(e) => error_result(start, e),
    }
}

// ─── Negative Binomial ───────────────────────────────────────────────────────

fn fit_negative_binomial_linear(df: &DataFrame) -> FitResult {
    let start = Instant::now();
    let y = extract_column(df, "y");
    let mut data = DataSet::new();
    data.insert_column("x", extract_column(df, "x"));

    let formula = Formula::new()
        .with_terms(
            "mu",
            vec![
                Term::Intercept,
                Term::Linear {
                    col_name: "x".to_string(),
                },
            ],
        )
        .with_terms("sigma", vec![Term::Intercept]);

    let family = NegativeBinomial::new();
    match GamlssModel::fit(&data, &y, &formula, &family) {
        Ok(model) => build_result(
            start,
            &model,
            &family,
            &y,
            &data,
            &[("mu", "log_mu"), ("sigma", "log_sigma")],
            Some("sigma"),
        ),
        Err(e) => error_result(start, e),
    }
}

fn fit_negative_binomial_smooth(df: &DataFrame) -> FitResult {
    let start = Instant::now();
    let y = extract_column(df, "y");
    let mut data = DataSet::new();
    data.insert_column("x", extract_column(df, "x"));

    let formula = Formula::new()
        .with_terms(
            "mu",
            vec![Term::Smooth(Smooth::PSpline1D {
                col_name: "x".to_string(),
                n_splines: 20,
                degree: 3,
                penalty_order: 2, range: None,
            })],
        )
        .with_terms("sigma", vec![Term::Intercept]);

    let family = NegativeBinomial::new();
    match GamlssModel::fit(&data, &y, &formula, &family) {
        Ok(model) => build_result(
            start,
            &model,
            &family,
            &y,
            &data,
            &[("mu", "log_mu_smooth"), ("sigma", "log_sigma")],
            Some("sigma"),
        ),
        Err(e) => error_result(start, e),
    }
}

// ─── Beta ────────────────────────────────────────────────────────────────────

fn fit_beta_linear(df: &DataFrame) -> FitResult {
    let start = Instant::now();
    let y = extract_column(df, "y");
    let mut data = DataSet::new();
    data.insert_column("x", extract_column(df, "x"));

    let formula = Formula::new()
        .with_terms(
            "mu",
            vec![
                Term::Intercept,
                Term::Linear {
                    col_name: "x".to_string(),
                },
            ],
        )
        .with_terms("phi", vec![Term::Intercept]);

    let family = Beta::new();
    match GamlssModel::fit(&data, &y, &formula, &family) {
        Ok(model) => build_result(
            start,
            &model,
            &family,
            &y,
            &data,
            &[("mu", "logit_mu"), ("phi", "log_phi")],
            Some("phi"),
        ),
        Err(e) => error_result(start, e),
    }
}

/// Beta smooth on μ; compared against mgcv `s(x, bs="ps")` with `betar()`.
fn fit_beta_smooth(df: &DataFrame) -> FitResult {
    let start = Instant::now();
    let y = extract_column(df, "y");
    let mut data = DataSet::new();
    data.insert_column("x", extract_column(df, "x"));

    let formula = Formula::new()
        .with_terms(
            "mu",
            vec![Term::Smooth(Smooth::PSpline1D {
                col_name: "x".to_string(),
                n_splines: 20,
                degree: 3,
                penalty_order: 2, range: None,
            })],
        )
        .with_terms("phi", vec![Term::Intercept]);

    let family = Beta::new();
    match GamlssModel::fit(&data, &y, &formula, &family) {
        Ok(model) => build_result(
            start,
            &model,
            &family,
            &y,
            &data,
            &[("mu", "logit_mu_smooth"), ("phi", "log_phi")],
            Some("phi"),
        ),
        Err(e) => error_result(start, e),
    }
}

// ─── Prior-weighted models ────────────────────────────────────────────────────

fn fit_b1_weighted_gaussian(df: &DataFrame) -> FitResult {
    // B1: Gaussian, 5 P-spline smooths + binary linear term, listing-level weights.
    let start = Instant::now();
    let y = extract_column(df, "y");
    let weights = extract_column(df, "weights");
    let mut data = DataSet::new();
    for col in ["x1", "x2", "x3", "x4", "x5", "d1"] {
        data.insert_column(col, extract_column(df, col));
    }

    let mk_smooth = |col: &str| {
        Term::Smooth(Smooth::PSpline1D {
            col_name: col.to_string(),
            n_splines: 20,
            degree: 3,
            penalty_order: 2, range: None,
        })
    };

    let formula = Formula::new()
        .with_terms(
            "mu",
            vec![
                Term::Intercept,
                mk_smooth("x1"),
                mk_smooth("x2"),
                mk_smooth("x3"),
                mk_smooth("x4"),
                mk_smooth("x5"),
                Term::Linear {
                    col_name: "d1".to_string(),
                },
            ],
        )
        .with_terms("sigma", vec![Term::Intercept]);

    let family = Gaussian::new();
    // REML, matching mgcv's method="REML" on the same model. (This scenario
    // previously used GCV as a workaround for L-BFGS stalling on the flat LAML
    // ridges of the weak smooths; the Fellner-Schall polish fixed that, and
    // GCV's different criterion diverged visibly from mgcv at some seeds.)
    let config = FitConfig::default();
    match GamlssModel::fit_with_config(&data, &y, Some(&weights), &formula, &family, config) {
        Ok(model) => build_result(
            start,
            &model,
            &family,
            &y,
            &data,
            &[("mu", "mu"), ("sigma", "log_sigma")],
            Some("sigma"),
        ),
        Err(e) => error_result(start, e),
    }
}

fn fit_b2_weighted_studentt(df: &DataFrame) -> FitResult {
    // B2: StudentT, 4 P-spline smooths, listing-level weights.
    let start = Instant::now();
    let y = extract_column(df, "y");
    let weights = extract_column(df, "weights");
    let mut data = DataSet::new();
    for col in ["x1", "x2", "x3", "x4"] {
        data.insert_column(col, extract_column(df, col));
    }

    let mk_smooth = |col: &str| {
        Term::Smooth(Smooth::PSpline1D {
            col_name: col.to_string(),
            n_splines: 20,
            degree: 3,
            penalty_order: 2, range: None,
        })
    };

    let formula = Formula::new()
        .with_terms(
            "mu",
            vec![
                Term::Intercept,
                mk_smooth("x1"),
                mk_smooth("x2"),
                mk_smooth("x3"),
                mk_smooth("x4"),
            ],
        )
        .with_terms("sigma", vec![Term::Intercept])
        .with_terms("nu", vec![Term::Intercept]);

    let family = StudentT::new();
    match GamlssModel::fit_weighted(&data, &y, &weights, &formula, &family) {
        Ok(model) => build_result(
            start,
            &model,
            &family,
            &y,
            &data,
            &[
                ("mu", "mu_smooth"),
                ("sigma", "log_sigma"),
                ("nu", "log_nu"),
            ],
            Some("sigma"),
        ),
        Err(e) => error_result(start, e),
    }
}

// ─── main ─────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut data_path: Option<PathBuf> = None;
    let mut scenario: Option<String> = None;
    let mut output_path: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data" => {
                i += 1;
                data_path = Some(PathBuf::from(&args[i]));
            }
            "--scenario" => {
                i += 1;
                scenario = Some(args[i].clone());
            }
            "--output" => {
                i += 1;
                output_path = Some(PathBuf::from(&args[i]));
            }
            _ => {}
        }
        i += 1;
    }

    let data_path = data_path.expect("Must provide --data argument");
    let scenario = scenario.expect("Must provide --scenario argument");
    let output_path = output_path.expect("Must provide --output argument");

    let df = LazyFrame::scan_parquet(&data_path, Default::default())
        .expect("Failed to open parquet")
        .collect()
        .expect("Failed to read parquet");

    let result = match scenario.as_str() {
        "gaussian_linear" => fit_gaussian_linear(&df),
        "gaussian_heteroskedastic" => fit_gaussian_heteroskedastic(&df),
        "gaussian_smooth" => fit_gaussian_smooth(&df),
        "gaussian_multiple" => fit_gaussian_multiple(&df),
        "gaussian_large" => fit_gaussian_large(&df),
        "gaussian_quadratic" => fit_gaussian_quadratic(&df),
        "gaussian_sigma_smooth" => fit_gaussian_sigma_smooth(&df),
        "gaussian_cr_smooth" => fit_gaussian_cr_smooth(&df),
        "tensor_smooth" => fit_tensor_smooth(&df),
        "random_effect" => fit_random_effect(&df),
        "poisson_linear" => fit_poisson_linear(&df),
        "poisson_smooth" => fit_poisson_smooth(&df),
        "binomial_linear" => fit_binomial_linear(&df),
        "binomial_smooth" => fit_binomial_smooth(&df),
        "gamma_linear" => fit_gamma_linear(&df),
        "gamma_smooth" => fit_gamma_smooth(&df),
        "gamma_sigma_smooth" => fit_gamma_sigma_smooth(&df),
        "studentt_linear" => fit_studentt_linear(&df),
        "studentt_smooth" => fit_studentt_smooth(&df),
        "negative_binomial_linear" => fit_negative_binomial_linear(&df),
        "negative_binomial_smooth" => fit_negative_binomial_smooth(&df),
        "beta_linear" => fit_beta_linear(&df),
        "beta_smooth" => fit_beta_smooth(&df),
        "b1_weighted_gaussian" => fit_b1_weighted_gaussian(&df),
        "b2_weighted_studentt" => fit_b2_weighted_studentt(&df),
        other => FitResult {
            converged: false,
            iterations: 0,
            fit_time_ms: 0.0,
            coefficients: HashMap::new(),
            fitted_mu: vec![],
            fitted_sigma: vec![],
            edf: HashMap::new(),
            log_likelihood: None,
            aic: None,
            lambdas: HashMap::new(),
            se_eta: HashMap::new(),
            error: Some(format!("Unknown scenario: {}", other)),
        },
    };

    let file = File::create(&output_path).expect("Failed to create output file");
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &result).expect("Failed to write JSON");

    eprintln!("Rust fitting complete: {}", scenario);
}
