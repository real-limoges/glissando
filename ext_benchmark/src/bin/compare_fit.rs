//! GAMLSS Comparison Framework: Rust Fitting Binary
//!
//! Reads parquet data, fits models using gamlss_rs,
//! and outputs standardized JSON results for comparison with R/mgcv.
//!
//! Usage:
//!   cargo run --bin compare_fit -- --data path/to/data.parquet --scenario gaussian_linear --output result.json

use gamlss_rs::distributions::{Beta, Gamma, Gaussian, NegativeBinomial, Poisson, StudentT};
use gamlss_rs::{GamlssModel, Smooth, Term};
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
    fitted_sigma: Vec<f64>,
    edf: HashMap<String, f64>,
    log_likelihood: Option<f64>,
    aic: Option<f64>,
    error: Option<String>,
}

fn fit_gaussian_linear(df: &DataFrame) -> FitResult {
    let start = Instant::now();

    let mut formula = HashMap::new();
    formula.insert(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );
    formula.insert("sigma".to_string(), vec![Term::Intercept]);

    match GamlssModel::fit(df, "y", &formula, &Gaussian::new()) {
        Ok(model) => {
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;

            let mut coefficients = HashMap::new();
            coefficients.insert("mu".to_string(), model.models["mu"].coefficients.0.to_vec());
            coefficients.insert(
                "log_sigma".to_string(),
                model.models["sigma"].coefficients.0.to_vec(),
            );

            let mut edf = HashMap::new();
            edf.insert("mu".to_string(), model.models["mu"].edf);
            edf.insert("sigma".to_string(), model.models["sigma"].edf);

            FitResult {
                converged: model.converged(),
                iterations: model.diagnostics.iterations,
                fit_time_ms: elapsed,
                coefficients,
                fitted_mu: model.models["mu"].fitted_values.to_vec(),
                fitted_sigma: model.models["sigma"].fitted_values.to_vec(),
                edf,
                log_likelihood: None, // TODO: implement in gamlss_rs
                aic: None,
                error: None,
            }
        }
        Err(e) => FitResult {
            converged: false,
            iterations: 0,
            fit_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            coefficients: HashMap::new(),
            fitted_mu: vec![],
            fitted_sigma: vec![],
            edf: HashMap::new(),
            log_likelihood: None,
            aic: None,
            error: Some(e.to_string()),
        },
    }
}

fn fit_gaussian_heteroskedastic(df: &DataFrame) -> FitResult {
    let start = Instant::now();

    let mut formula = HashMap::new();
    formula.insert(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );
    formula.insert(
        "sigma".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );

    match GamlssModel::fit(df, "y", &formula, &Gaussian::new()) {
        Ok(model) => {
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;

            let mut coefficients = HashMap::new();
            coefficients.insert("mu".to_string(), model.models["mu"].coefficients.0.to_vec());
            coefficients.insert(
                "log_sigma".to_string(),
                model.models["sigma"].coefficients.0.to_vec(),
            );

            let mut edf = HashMap::new();
            edf.insert("mu".to_string(), model.models["mu"].edf);
            edf.insert("sigma".to_string(), model.models["sigma"].edf);

            FitResult {
                converged: model.converged(),
                iterations: model.diagnostics.iterations,
                fit_time_ms: elapsed,
                coefficients,
                fitted_mu: model.models["mu"].fitted_values.to_vec(),
                fitted_sigma: model.models["sigma"].fitted_values.to_vec(),
                edf,
                log_likelihood: None,
                aic: None,
                error: None,
            }
        }
        Err(e) => FitResult {
            converged: false,
            iterations: 0,
            fit_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            coefficients: HashMap::new(),
            fitted_mu: vec![],
            fitted_sigma: vec![],
            edf: HashMap::new(),
            log_likelihood: None,
            aic: None,
            error: Some(e.to_string()),
        },
    }
}

fn fit_gaussian_smooth(df: &DataFrame) -> FitResult {
    let start = Instant::now();

    let mut formula = HashMap::new();
    formula.insert(
        "mu".to_string(),
        vec![Term::Smooth(Smooth::PSpline1D {
            col_name: "x".to_string(),
            n_splines: 20,
            degree: 3,
            penalty_order: 2,
        })],
    );
    formula.insert("sigma".to_string(), vec![Term::Intercept]);

    match GamlssModel::fit(df, "y", &formula, &Gaussian::new()) {
        Ok(model) => {
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;

            let mut coefficients = HashMap::new();
            coefficients.insert(
                "mu_smooth".to_string(),
                model.models["mu"].coefficients.0.to_vec(),
            );
            coefficients.insert(
                "log_sigma".to_string(),
                model.models["sigma"].coefficients.0.to_vec(),
            );

            let mut edf = HashMap::new();
            edf.insert("mu".to_string(), model.models["mu"].edf);
            edf.insert("sigma".to_string(), model.models["sigma"].edf);

            FitResult {
                converged: model.converged(),
                iterations: model.diagnostics.iterations,
                fit_time_ms: elapsed,
                coefficients,
                fitted_mu: model.models["mu"].fitted_values.to_vec(),
                fitted_sigma: model.models["sigma"].fitted_values.to_vec(),
                edf,
                log_likelihood: None,
                aic: None,
                error: None,
            }
        }
        Err(e) => FitResult {
            converged: false,
            iterations: 0,
            fit_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            coefficients: HashMap::new(),
            fitted_mu: vec![],
            fitted_sigma: vec![],
            edf: HashMap::new(),
            log_likelihood: None,
            aic: None,
            error: Some(e.to_string()),
        },
    }
}

fn fit_poisson_linear(df: &DataFrame) -> FitResult {
    let start = Instant::now();

    let mut formula = HashMap::new();
    formula.insert(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );

    match GamlssModel::fit(df, "y", &formula, &Poisson::new()) {
        Ok(model) => {
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;

            let mut coefficients = HashMap::new();
            coefficients.insert(
                "log_mu".to_string(),
                model.models["mu"].coefficients.0.to_vec(),
            );

            let mut edf = HashMap::new();
            edf.insert("mu".to_string(), model.models["mu"].edf);

            FitResult {
                converged: model.converged(),
                iterations: model.diagnostics.iterations,
                fit_time_ms: elapsed,
                coefficients,
                fitted_mu: model.models["mu"].fitted_values.to_vec(),
                fitted_sigma: vec![],
                edf,
                log_likelihood: None,
                aic: None,
                error: None,
            }
        }
        Err(e) => FitResult {
            converged: false,
            iterations: 0,
            fit_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            coefficients: HashMap::new(),
            fitted_mu: vec![],
            fitted_sigma: vec![],
            edf: HashMap::new(),
            log_likelihood: None,
            aic: None,
            error: Some(e.to_string()),
        },
    }
}

fn fit_gamma_linear(df: &DataFrame) -> FitResult {
    let start = Instant::now();

    let mut formula = HashMap::new();
    formula.insert(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );
    formula.insert("sigma".to_string(), vec![Term::Intercept]);

    match GamlssModel::fit(df, "y", &formula, &Gamma::new()) {
        Ok(model) => {
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;

            let mut coefficients = HashMap::new();
            coefficients.insert(
                "log_mu".to_string(),
                model.models["mu"].coefficients.0.to_vec(),
            );
            coefficients.insert(
                "log_sigma".to_string(),
                model.models["sigma"].coefficients.0.to_vec(),
            );

            let mut edf = HashMap::new();
            edf.insert("mu".to_string(), model.models["mu"].edf);
            edf.insert("sigma".to_string(), model.models["sigma"].edf);

            FitResult {
                converged: model.converged(),
                iterations: model.diagnostics.iterations,
                fit_time_ms: elapsed,
                coefficients,
                fitted_mu: model.models["mu"].fitted_values.to_vec(),
                fitted_sigma: model.models["sigma"].fitted_values.to_vec(),
                edf,
                log_likelihood: None,
                aic: None,
                error: None,
            }
        }
        Err(e) => FitResult {
            converged: false,
            iterations: 0,
            fit_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            coefficients: HashMap::new(),
            fitted_mu: vec![],
            fitted_sigma: vec![],
            edf: HashMap::new(),
            log_likelihood: None,
            aic: None,
            error: Some(e.to_string()),
        },
    }
}

fn fit_studentt_linear(df: &DataFrame) -> FitResult {
    let start = Instant::now();

    let mut formula = HashMap::new();
    formula.insert(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );
    formula.insert("sigma".to_string(), vec![Term::Intercept]);
    formula.insert("nu".to_string(), vec![Term::Intercept]);

    match GamlssModel::fit(df, "y", &formula, &StudentT::new()) {
        Ok(model) => {
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;

            let mut coefficients = HashMap::new();
            coefficients.insert("mu".to_string(), model.models["mu"].coefficients.0.to_vec());
            coefficients.insert(
                "log_sigma".to_string(),
                model.models["sigma"].coefficients.0.to_vec(),
            );
            coefficients.insert(
                "log_nu".to_string(),
                model.models["nu"].coefficients.0.to_vec(),
            );

            let mut edf = HashMap::new();
            edf.insert("mu".to_string(), model.models["mu"].edf);
            edf.insert("sigma".to_string(), model.models["sigma"].edf);
            edf.insert("nu".to_string(), model.models["nu"].edf);

            FitResult {
                converged: model.converged(),
                iterations: model.diagnostics.iterations,
                fit_time_ms: elapsed,
                coefficients,
                fitted_mu: model.models["mu"].fitted_values.to_vec(),
                fitted_sigma: model.models["sigma"].fitted_values.to_vec(),
                edf,
                log_likelihood: None,
                aic: None,
                error: None,
            }
        }
        Err(e) => FitResult {
            converged: false,
            iterations: 0,
            fit_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            coefficients: HashMap::new(),
            fitted_mu: vec![],
            fitted_sigma: vec![],
            edf: HashMap::new(),
            log_likelihood: None,
            aic: None,
            error: Some(e.to_string()),
        },
    }
}

fn fit_negative_binomial_linear(df: &DataFrame) -> FitResult {
    let start = Instant::now();

    let mut formula = HashMap::new();
    formula.insert(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );
    formula.insert("sigma".to_string(), vec![Term::Intercept]);

    match GamlssModel::fit(df, "y", &formula, &NegativeBinomial::new()) {
        Ok(model) => {
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;

            let mut coefficients = HashMap::new();
            coefficients.insert(
                "log_mu".to_string(),
                model.models["mu"].coefficients.0.to_vec(),
            );
            coefficients.insert(
                "log_sigma".to_string(),
                model.models["sigma"].coefficients.0.to_vec(),
            );

            let mut edf = HashMap::new();
            edf.insert("mu".to_string(), model.models["mu"].edf);
            edf.insert("sigma".to_string(), model.models["sigma"].edf);

            FitResult {
                converged: model.converged(),
                iterations: model.diagnostics.iterations,
                fit_time_ms: elapsed,
                coefficients,
                fitted_mu: model.models["mu"].fitted_values.to_vec(),
                fitted_sigma: model.models["sigma"].fitted_values.to_vec(),
                edf,
                log_likelihood: None,
                aic: None,
                error: None,
            }
        }
        Err(e) => FitResult {
            converged: false,
            iterations: 0,
            fit_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            coefficients: HashMap::new(),
            fitted_mu: vec![],
            fitted_sigma: vec![],
            edf: HashMap::new(),
            log_likelihood: None,
            aic: None,
            error: Some(e.to_string()),
        },
    }
}

fn fit_beta_linear(df: &DataFrame) -> FitResult {
    let start = Instant::now();

    let mut formula = HashMap::new();
    formula.insert(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );
    formula.insert("phi".to_string(), vec![Term::Intercept]);

    match GamlssModel::fit(df, "y", &formula, &Beta::new()) {
        Ok(model) => {
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;

            let mut coefficients = HashMap::new();
            coefficients.insert(
                "logit_mu".to_string(),
                model.models["mu"].coefficients.0.to_vec(),
            );
            coefficients.insert(
                "log_phi".to_string(),
                model.models["phi"].coefficients.0.to_vec(),
            );

            let mut edf = HashMap::new();
            edf.insert("mu".to_string(), model.models["mu"].edf);
            edf.insert("phi".to_string(), model.models["phi"].edf);

            FitResult {
                converged: model.converged(),
                iterations: model.diagnostics.iterations,
                fit_time_ms: elapsed,
                coefficients,
                fitted_mu: model.models["mu"].fitted_values.to_vec(),
                fitted_sigma: model.models["phi"].fitted_values.to_vec(),
                edf,
                log_likelihood: None,
                aic: None,
                error: None,
            }
        }
        Err(e) => FitResult {
            converged: false,
            iterations: 0,
            fit_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            coefficients: HashMap::new(),
            fitted_mu: vec![],
            fitted_sigma: vec![],
            edf: HashMap::new(),
            log_likelihood: None,
            aic: None,
            error: Some(e.to_string()),
        },
    }
}

fn fit_poisson_smooth(df: &DataFrame) -> FitResult {
    let start = Instant::now();

    let mut formula = HashMap::new();
    formula.insert(
        "mu".to_string(),
        vec![Term::Smooth(Smooth::PSpline1D {
            col_name: "x".to_string(),
            n_splines: 20,
            degree: 3,
            penalty_order: 2,
        })],
    );

    match GamlssModel::fit(df, "y", &formula, &Poisson::new()) {
        Ok(model) => {
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;

            let mut coefficients = HashMap::new();
            coefficients.insert(
                "log_mu_smooth".to_string(),
                model.models["mu"].coefficients.0.to_vec(),
            );

            let mut edf = HashMap::new();
            edf.insert("mu".to_string(), model.models["mu"].edf);

            FitResult {
                converged: model.converged(),
                iterations: model.diagnostics.iterations,
                fit_time_ms: elapsed,
                coefficients,
                fitted_mu: model.models["mu"].fitted_values.to_vec(),
                fitted_sigma: vec![],
                edf,
                log_likelihood: None,
                aic: None,
                error: None,
            }
        }
        Err(e) => FitResult {
            converged: false,
            iterations: 0,
            fit_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            coefficients: HashMap::new(),
            fitted_mu: vec![],
            fitted_sigma: vec![],
            edf: HashMap::new(),
            log_likelihood: None,
            aic: None,
            error: Some(e.to_string()),
        },
    }
}

fn fit_gamma_smooth(df: &DataFrame) -> FitResult {
    let start = Instant::now();

    let mut formula = HashMap::new();
    formula.insert(
        "mu".to_string(),
        vec![Term::Smooth(Smooth::PSpline1D {
            col_name: "x".to_string(),
            n_splines: 20,
            degree: 3,
            penalty_order: 2,
        })],
    );
    formula.insert("sigma".to_string(), vec![Term::Intercept]);

    match GamlssModel::fit(df, "y", &formula, &Gamma::new()) {
        Ok(model) => {
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;

            let mut coefficients = HashMap::new();
            coefficients.insert(
                "log_mu_smooth".to_string(),
                model.models["mu"].coefficients.0.to_vec(),
            );
            coefficients.insert(
                "log_sigma".to_string(),
                model.models["sigma"].coefficients.0.to_vec(),
            );

            let mut edf = HashMap::new();
            edf.insert("mu".to_string(), model.models["mu"].edf);
            edf.insert("sigma".to_string(), model.models["sigma"].edf);

            FitResult {
                converged: model.converged(),
                iterations: model.diagnostics.iterations,
                fit_time_ms: elapsed,
                coefficients,
                fitted_mu: model.models["mu"].fitted_values.to_vec(),
                fitted_sigma: model.models["sigma"].fitted_values.to_vec(),
                edf,
                log_likelihood: None,
                aic: None,
                error: None,
            }
        }
        Err(e) => FitResult {
            converged: false,
            iterations: 0,
            fit_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            coefficients: HashMap::new(),
            fitted_mu: vec![],
            fitted_sigma: vec![],
            edf: HashMap::new(),
            log_likelihood: None,
            aic: None,
            error: Some(e.to_string()),
        },
    }
}

fn fit_gaussian_multiple(df: &DataFrame) -> FitResult {
    let start = Instant::now();

    let mut formula = HashMap::new();
    formula.insert(
        "mu".to_string(),
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
    );
    formula.insert("sigma".to_string(), vec![Term::Intercept]);

    match GamlssModel::fit(df, "y", &formula, &Gaussian::new()) {
        Ok(model) => {
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;

            let mut coefficients = HashMap::new();
            coefficients.insert("mu".to_string(), model.models["mu"].coefficients.0.to_vec());
            coefficients.insert(
                "log_sigma".to_string(),
                model.models["sigma"].coefficients.0.to_vec(),
            );

            let mut edf = HashMap::new();
            edf.insert("mu".to_string(), model.models["mu"].edf);
            edf.insert("sigma".to_string(), model.models["sigma"].edf);

            FitResult {
                converged: model.converged(),
                iterations: model.diagnostics.iterations,
                fit_time_ms: elapsed,
                coefficients,
                fitted_mu: model.models["mu"].fitted_values.to_vec(),
                fitted_sigma: model.models["sigma"].fitted_values.to_vec(),
                edf,
                log_likelihood: None,
                aic: None,
                error: None,
            }
        }
        Err(e) => FitResult {
            converged: false,
            iterations: 0,
            fit_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            coefficients: HashMap::new(),
            fitted_mu: vec![],
            fitted_sigma: vec![],
            edf: HashMap::new(),
            log_likelihood: None,
            aic: None,
            error: Some(e.to_string()),
        },
    }
}

fn fit_gaussian_large(df: &DataFrame) -> FitResult {
    // Same as gaussian_linear, just with larger data
    fit_gaussian_linear(df)
}

fn fit_studentt_smooth(df: &DataFrame) -> FitResult {
    let start = Instant::now();

    let mut formula = HashMap::new();
    formula.insert(
        "mu".to_string(),
        vec![Term::Smooth(Smooth::PSpline1D {
            col_name: "x".to_string(),
            n_splines: 20,
            degree: 3,
            penalty_order: 2,
        })],
    );
    formula.insert("sigma".to_string(), vec![Term::Intercept]);
    formula.insert("nu".to_string(), vec![Term::Intercept]);

    match GamlssModel::fit(df, "y", &formula, &StudentT::new()) {
        Ok(model) => {
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;

            let mut coefficients = HashMap::new();
            coefficients.insert(
                "mu_smooth".to_string(),
                model.models["mu"].coefficients.0.to_vec(),
            );
            coefficients.insert(
                "log_sigma".to_string(),
                model.models["sigma"].coefficients.0.to_vec(),
            );
            coefficients.insert(
                "log_nu".to_string(),
                model.models["nu"].coefficients.0.to_vec(),
            );

            let mut edf = HashMap::new();
            edf.insert("mu".to_string(), model.models["mu"].edf);
            edf.insert("sigma".to_string(), model.models["sigma"].edf);
            edf.insert("nu".to_string(), model.models["nu"].edf);

            FitResult {
                converged: model.converged(),
                iterations: model.diagnostics.iterations,
                fit_time_ms: elapsed,
                coefficients,
                fitted_mu: model.models["mu"].fitted_values.to_vec(),
                fitted_sigma: model.models["sigma"].fitted_values.to_vec(),
                edf,
                log_likelihood: None,
                aic: None,
                error: None,
            }
        }
        Err(e) => FitResult {
            converged: false,
            iterations: 0,
            fit_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            coefficients: HashMap::new(),
            fitted_mu: vec![],
            fitted_sigma: vec![],
            edf: HashMap::new(),
            log_likelihood: None,
            aic: None,
            error: Some(e.to_string()),
        },
    }
}

fn fit_negative_binomial_smooth(df: &DataFrame) -> FitResult {
    let start = Instant::now();

    let mut formula = HashMap::new();
    formula.insert(
        "mu".to_string(),
        vec![Term::Smooth(Smooth::PSpline1D {
            col_name: "x".to_string(),
            n_splines: 20,
            degree: 3,
            penalty_order: 2,
        })],
    );
    formula.insert("sigma".to_string(), vec![Term::Intercept]);

    match GamlssModel::fit(df, "y", &formula, &NegativeBinomial::new()) {
        Ok(model) => {
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;

            let mut coefficients = HashMap::new();
            coefficients.insert(
                "log_mu_smooth".to_string(),
                model.models["mu"].coefficients.0.to_vec(),
            );
            coefficients.insert(
                "log_sigma".to_string(),
                model.models["sigma"].coefficients.0.to_vec(),
            );

            let mut edf = HashMap::new();
            edf.insert("mu".to_string(), model.models["mu"].edf);
            edf.insert("sigma".to_string(), model.models["sigma"].edf);

            FitResult {
                converged: model.converged(),
                iterations: model.diagnostics.iterations,
                fit_time_ms: elapsed,
                coefficients,
                fitted_mu: model.models["mu"].fitted_values.to_vec(),
                fitted_sigma: model.models["sigma"].fitted_values.to_vec(),
                edf,
                log_likelihood: None,
                aic: None,
                error: None,
            }
        }
        Err(e) => FitResult {
            converged: false,
            iterations: 0,
            fit_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            coefficients: HashMap::new(),
            fitted_mu: vec![],
            fitted_sigma: vec![],
            edf: HashMap::new(),
            log_likelihood: None,
            aic: None,
            error: Some(e.to_string()),
        },
    }
}

fn fit_gaussian_quadratic(df: &DataFrame) -> FitResult {
    // Same as gaussian_smooth - testing smooth on quadratic data
    fit_gaussian_smooth(df)
}

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

    // Read parquet
    let df = LazyFrame::scan_parquet(&data_path, Default::default())
        .expect("Failed to open parquet")
        .collect()
        .expect("Failed to read parquet");

    // Dispatch to appropriate fitter
    let result = match scenario.as_str() {
        "gaussian_linear" => fit_gaussian_linear(&df),
        "gaussian_heteroskedastic" => fit_gaussian_heteroskedastic(&df),
        "gaussian_smooth" => fit_gaussian_smooth(&df),
        "gaussian_multiple" => fit_gaussian_multiple(&df),
        "gaussian_large" => fit_gaussian_large(&df),
        "gaussian_quadratic" => fit_gaussian_quadratic(&df),
        "poisson_linear" => fit_poisson_linear(&df),
        "poisson_smooth" => fit_poisson_smooth(&df),
        "gamma_linear" => fit_gamma_linear(&df),
        "gamma_smooth" => fit_gamma_smooth(&df),
        "studentt_linear" => fit_studentt_linear(&df),
        "studentt_smooth" => fit_studentt_smooth(&df),
        "negative_binomial_linear" => fit_negative_binomial_linear(&df),
        "negative_binomial_smooth" => fit_negative_binomial_smooth(&df),
        "beta_linear" => fit_beta_linear(&df),
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
            error: Some(format!("Unknown scenario: {}", other)),
        },
    };

    // Write JSON output
    let file = File::create(&output_path).expect("Failed to create output file");
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &result).expect("Failed to write JSON");

    eprintln!("Rust fitting complete: {}", scenario);
}
