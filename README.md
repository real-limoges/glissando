# glissando

A Rust implementation of Generalized Additive Models for Location, Scale, and Shape (GAMLSS).

GAMLSS extends traditional regression by modeling not just the mean, but also variance and other distribution parameters as functions of predictors. This enables flexible modeling of heteroskedastic data, heavy-tailed distributions, and other complex data structures.

## Features

- **Multiple distribution parameters**: Model mean, variance, and shape parameters simultaneously
- **Flexible terms**: Intercept, linear effects, P-splines, tensor products, and random effects
- **Automatic smoothing**: Smoothing parameters selected via REML (default), GCV, or Fellner-Schall
- **Dual backends**: OpenBLAS (default, max performance) or pure Rust via nalgebra (no system deps)
- **WASM support**: Fit models and predict directly in the browser via wasm-bindgen
- **Type-safe API**: `DataSet`, `Formula`, and newtype wrappers prevent misuse

## Installation

The friendliest first build uses the **pure-Rust** backend — no system libraries, works on a clean machine and in WASM:

```toml
[dependencies]
glissando = { git = "https://github.com/real-limoges/glissando", default-features = false, features = ["pure-rust", "serialization"] }
```

For maximum performance, opt into the OpenBLAS backend instead (this is the default feature set, but it links a system OpenBLAS — see [Requirements](#requirements)):

```toml
[dependencies]
glissando = { git = "https://github.com/real-limoges/glissando" }  # default = openblas + parallel
```

`openblas` and `pure-rust` are mutually exclusive — pick one backend.

### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `openblas` | OpenBLAS backend (ndarray-linalg) — max performance | yes |
| `pure-rust` | nalgebra backend — no system dependencies, WASM-compatible | no |
| `serialization` | Serde support for model serialization | no |
| `wasm` | WASM fitting + prediction API (implies `pure-rust` + `serialization`, no parallelism) | no |
| `python` | PyO3 bindings for Python integration (implies `openblas` + `parallel` + `serialization`) | no |
| `parallel` | Rayon parallelism for large datasets (incompatible with WASM) | yes |

**Note**: `openblas` and `pure-rust` are mutually exclusive (select your linear algebra backend). The `wasm` feature automatically disables parallelism.

### Requirements

- Rust 2021 edition
- OpenBLAS (only with default `openblas` feature)

On macOS:
```bash
brew install openblas
```

On Ubuntu/Debian:
```bash
sudo apt-get install libopenblas-dev
```

For pure Rust or WASM builds, no system dependencies are needed.

## Quick Start

```rust
use glissando::{GamlssModel, DataSet, Formula, Term};
use glissando::distributions::Gaussian;
use ndarray::Array1;

let y = Array1::from_vec(vec![2.1, 4.0, 5.9, 8.1, 10.0]);

let mut data = DataSet::new();
data.insert_column("x", Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]));

let formula = Formula::new()
    .with_terms("mu", vec![
        Term::Intercept,
        Term::Linear { col_name: "x".to_string() },
    ])
    .with_terms("sigma", vec![Term::Intercept]);

let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

println!("Converged: {}", model.converged());
let mu_coeffs = &model.models["mu"].coefficients;
println!("Intercept: {}, Slope: {}", mu_coeffs[0], mu_coeffs[1]);
```

> **ndarray version.** The public API hands back `ndarray` types (`Array1<f64>`,
> `Array2<f64>`), so you must build against the same `ndarray` major (currently
> **0.17**). To avoid guessing, use the re-export: `glissando::ndarray::Array1`
> resolves to exactly the version this crate is built against.

## Distributions

| Distribution | Parameters | Default Links | Use Case |
|--------------|------------|---------------|----------|
| `Poisson` | mu | log | Count data |
| `Binomial` | mu | logit | Binary/count with known trials |
| `Gaussian` | mu, sigma | identity, log | Continuous data |
| `StudentT` | mu, sigma, nu | identity, log, log | Heavy-tailed continuous |
| `Gamma` | mu, sigma | log, log | Positive continuous |
| `NegativeBinomial` | mu, sigma | log, log | Overdispersed counts |
| `Beta` | mu, phi | logit, log | Proportions (0, 1) |

### Usage

```rust
use glissando::distributions::{Poisson, Binomial, Gaussian, StudentT, Gamma, NegativeBinomial, Beta};

let poisson = Poisson::new();             // Count data
let binomial = Binomial::new(10);         // Binary/count with 10 trials
let gaussian = Gaussian::new();           // Continuous data
let student_t = StudentT::new();          // Heavy-tailed continuous data
let gamma = Gamma::new();                 // Positive continuous (e.g., durations)
let neg_bin = NegativeBinomial::new();    // Overdispersed counts
let beta = Beta::new();                   // Proportions/rates in (0, 1)
```

## Term Types

### Intercept

A constant term (bias).

```rust
Term::Intercept
```

### Linear

A linear effect for a single predictor.

```rust
Term::Linear { col_name: "x".to_string() }
```

### P-Spline (1D Smooth)

A penalized B-spline smooth for nonlinear effects.

```rust
Term::Smooth(Smooth::PSpline1D {
    col_name: "x".to_string(),
    n_splines: 10,      // Number of basis functions
    degree: 3,          // Spline degree (3 = cubic)
    penalty_order: 2,   // Penalty on 2nd derivatives
})
```

### Tensor Product (2D Smooth)

Interaction smooth for two predictors.

```rust
Term::Smooth(Smooth::TensorProduct {
    col_name_1: "x1".to_string(),
    n_splines_1: 5,
    penalty_order_1: 2,
    col_name_2: "x2".to_string(),
    n_splines_2: 5,
    penalty_order_2: 2,
    degree: 3,
})
```

### Random Effect

Group-level random intercepts.

```rust
Term::Smooth(Smooth::RandomEffect {
    col_name: "group".to_string(),
})
```

## Configuration

```rust
use glissando::{FitConfig, SmoothingCriterion};

let config = FitConfig {
    max_iterations: 200,
    tolerance: 1e-3,
    criterion: SmoothingCriterion::Reml,  // also: Gcv, FellnerSchall
};

let model = GamlssModel::fit_with_config(
    &data, &y, &formula, &Gaussian::new(), config
)?;
```

`SmoothingCriterion` selects the smoothing-parameter optimizer:

- `Reml` (default) — Laplace-approximate marginal likelihood (Wood 2011), optimized via L-BFGS
- `Gcv` — Generalized Cross-Validation (Craven & Wahba 1979), optimized via L-BFGS
- `FellnerSchall` — multiplicative fixed-point update for the LAML target (Wood & Fasiolo 2017); deterministic, no line search

## Accessing Results

```rust
let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new())?;

// Convergence diagnostics
println!("Converged: {}", model.diagnostics.converged);
println!("Iterations: {}", model.diagnostics.iterations);

// Per-parameter results
let fitted_mu = &model.models["mu"];
fitted_mu.coefficients     // Coefficients newtype (Deref to Array1<f64>)
fitted_mu.covariance       // CovarianceMatrix newtype (Deref to Array2<f64>)
fitted_mu.fitted_values    // Fitted values on response scale
fitted_mu.eta              // Linear predictor (X * beta)
fitted_mu.edf              // Effective degrees of freedom
fitted_mu.lambdas          // Smoothing parameters
fitted_mu.terms            // Formula terms
```

## Prediction

```rust
let mut new_data = DataSet::new();
new_data.insert_column("x", Array1::from_vec(vec![1.5, 2.5, 3.5]));

let family = Gaussian::new();

// Point predictions (fitted values on response scale)
let predictions = model.predict(&new_data, &family)?;
let mu_pred = &predictions["mu"];

// Predictions with standard errors
let results = model.predict_with_se(&new_data, &family)?;
let mu_result = &results["mu"];
println!("Fitted values (response scale): {:?}", mu_result.fitted);
println!("Linear predictor (eta): {:?}", mu_result.eta);
println!("Standard errors on eta scale: {:?}", mu_result.se_eta);

// Posterior samples for uncertainty quantification
let samples = model.predict_samples(&new_data, &family, 1000)?;
let mu_samples = &samples["mu"];  // Vec<Array1<f64>> with 1000 samples
```

## Model Diagnostics

```rust
use glissando::diagnostics::{
    pearson_residuals_gaussian, response_residuals,
    loglik_gaussian, compute_aic, compute_bic, total_edf,
};

let mu = &model.models["mu"].fitted_values;
let sigma = &model.models["sigma"].fitted_values;

let pearson_resid = pearson_residuals_gaussian(&y, mu, sigma);
let ll = loglik_gaussian(&y, mu, sigma);
let edf = total_edf(&model.models);
let aic = compute_aic(ll, edf);
let bic = compute_bic(ll, edf, y.len());
```

Distribution-specific residual and log-likelihood functions are available for all supported distributions (e.g., `pearson_residuals_poisson`, `loglik_gamma`, etc.).

## Examples

### Heteroskedastic Regression

Model where both mean and variance depend on x:

```rust
let formula = Formula::new()
    .with_terms("mu", vec![
        Term::Intercept,
        Term::Linear { col_name: "x".to_string() },
    ])
    .with_terms("sigma", vec![
        Term::Intercept,
        Term::Linear { col_name: "x".to_string() },
    ]);

let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new())?;
```

### Nonlinear Smooth

```rust
let formula = Formula::new()
    .with_terms("mu", vec![
        Term::Smooth(Smooth::PSpline1D {
            col_name: "x".to_string(),
            n_splines: 15,
            degree: 3,
            penalty_order: 2,
        }),
    ])
    .with_terms("sigma", vec![Term::Intercept]);

let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new())?;
```

### Count Data with Poisson

```rust
use glissando::distributions::Poisson;

let formula = Formula::new()
    .with_terms("mu", vec![
        Term::Intercept,
        Term::Linear { col_name: "predictor".to_string() },
    ]);

let model = GamlssModel::fit(&data, &counts, &formula, &Poisson::new())?;
```

### Binary/Binomial Data

```rust
use glissando::distributions::Binomial;

let formula = Formula::new()
    .with_terms("mu", vec![
        Term::Intercept,
        Term::Linear { col_name: "x".to_string() },
    ]);

// Fixed number of trials
let model = GamlssModel::fit(&data, &successes, &formula, &Binomial::new(20))?;

// Or varying trials per observation
let trials = Array1::from_vec(vec![10.0, 15.0, 20.0, 25.0]);
let model = GamlssModel::fit(&data, &successes, &formula, &Binomial::with_trials(trials))?;
```

### Heavy-Tailed Data with Student-t

```rust
use glissando::distributions::StudentT;

let formula = Formula::new()
    .with_terms("mu", vec![
        Term::Intercept,
        Term::Linear { col_name: "x".to_string() },
    ])
    .with_terms("sigma", vec![Term::Intercept])
    .with_terms("nu", vec![Term::Intercept]);

let model = GamlssModel::fit(&data, &y, &formula, &StudentT::new())?;
```

### Mixed Effects Model

```rust
let formula = Formula::new()
    .with_terms("mu", vec![
        Term::Intercept,
        Term::Linear { col_name: "x".to_string() },
        Term::Smooth(Smooth::RandomEffect {
            col_name: "subject_id".to_string(),
        }),
    ])
    .with_terms("sigma", vec![Term::Intercept]);

let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new())?;
```

### Overdispersed Count Data

```rust
use glissando::distributions::NegativeBinomial;

let formula = Formula::new()
    .with_terms("mu", vec![
        Term::Intercept,
        Term::Linear { col_name: "x".to_string() },
    ])
    .with_terms("sigma", vec![Term::Intercept]);

let model = GamlssModel::fit(&data, &counts, &formula, &NegativeBinomial::new())?;
```

### Proportion/Rate Data

```rust
use glissando::distributions::Beta;

let formula = Formula::new()
    .with_terms("mu", vec![
        Term::Intercept,
        Term::Linear { col_name: "x".to_string() },
    ])
    .with_terms("phi", vec![Term::Intercept]);

let model = GamlssModel::fit(&data, &proportions, &formula, &Beta::new())?;
```

### Duration/Positive Continuous Data

```rust
use glissando::distributions::Gamma;

let formula = Formula::new()
    .with_terms("mu", vec![
        Term::Intercept,
        Term::Linear { col_name: "age".to_string() },
    ])
    .with_terms("sigma", vec![Term::Intercept]);

let model = GamlssModel::fit(&data, &durations, &formula, &Gamma::new())?;
```

## Error Handling

The library uses `GamlssError` for error handling:

```rust
use glissando::GamlssError;

match GamlssModel::fit(&data, &y, &formula, &Gaussian::new()) {
    Ok(model) => {
        // Use the fitted model
    }
    Err(GamlssError::Input(msg)) => {
        eprintln!("Input error: {}", msg);
    }
    Err(GamlssError::MissingVariable { name }) => {
        eprintln!("Variable '{}' not found in data", name);
    }
    Err(GamlssError::NonFiniteValues { name, count }) => {
        eprintln!("Variable '{}' has {} non-finite values", name, count);
    }
    Err(GamlssError::Convergence(iters)) => {
        eprintln!("Failed to converge after {} iterations", iters);
    }
    Err(e) => {
        eprintln!("Error: {}", e);
    }
}
```

### Error Types

| Error | Description |
|-------|-------------|
| `Input` | Invalid input data or formula |
| `MissingVariable` | Required variable not found in data |
| `MissingFormula` | Formula missing terms for a distribution parameter |
| `NonFiniteValues` | Variable contains NaN or Inf values (includes count of non-finite values) |
| `EmptyData` | No observations provided |
| `Convergence` | Algorithm failed to converge after N iterations |
| `Optimization` | Smoothing parameter optimization (L-BFGS) failed |
| `Linalg` | Linear algebra computation failed (Cholesky, matrix solve, etc.) |
| `PosteriorNotPositiveDefinite` | Posterior covariance is not positive definite — `predict_samples` / `posterior_samples` failed Cholesky |
| `UnknownParameter` | Unknown parameter for the given distribution |
| `Shape` | Array shape mismatch |
| `Internal` | Internal logic error (indicates a bug) |

## Embedding glissando behind your own FFI

glissando ships three faces — the typed Rust API, the WASM bindings, and the
Python extension. If you are embedding the crate behind a *different* boundary
(a [Rustler](https://github.com/rusterlium/rustler) NIF, a C ABI, a JSON
service), you do not need to re-implement the wire format: the `glissando::json`
module (enabled by the `serialization` feature) is the same tested JSON
marshalling the WASM bindings use, exposed for any embedder.

```rust
use glissando::json;

// Strings in, model + boxed distribution out — keep the model in memory and
// predict interactively (no per-call re-fit).
let y       = "[1.2, 2.1, 2.9, 4.2, 4.8]";
let data    = r#"{"x": [1.0, 2.0, 3.0, 4.0, 5.0]}"#;
let formula = r#"{
    "mu":    [{"Intercept": null}, {"Linear": {"col_name": "x"}}],
    "sigma": [{"Intercept": null}]
}"#;

let (model, family) = json::fit(y, data, formula, "Gaussian", None)?;

// Strings out: predictions, SEs, posterior samples, and fit diagnostics.
let preds       = json::predict(&model, family.as_ref(), r#"{"x": [6.0, 7.0]}"#)?;
let with_se     = json::predict_with_se(&model, family.as_ref(), r#"{"x": [6.0]}"#)?;
let samples     = json::predict_samples(&model, family.as_ref(), r#"{"x": [6.0]}"#, 500)?;
let diagnostics = json::diagnostics(&model)?;   // converged, per-param + per-term EDF, warnings

// Persist and reload (round-trips the distribution name too).
let blob = model.to_json(family.as_ref())?;
let (restored, family) = json::load(&blob)?;
# Ok::<(), glissando::GamlssError>(())
```

If you need typed dispatch instead of the string facade,
`glissando::distributions::from_name("Gaussian") -> Box<dyn Distribution>`
resolves any stateless family by name (every distribution except `Binomial`,
which needs `n_trials` state — construct that one through the typed API). The
`json` parsing/serialization helpers (`parse_data`, `parse_formula`,
`serialize_predictions`, …) are also public if you want to mix glissando's wire
format with your own fitting flow.

## Serialization & WASM

Models can be serialized to JSON for transfer to browsers or other systems. Enable with the `serialization` feature:

```toml
[dependencies]
glissando = { git = "...", features = ["serialization"] }
```

```rust
// Serialize a fitted model (native side)
let json = model.to_json(&Gaussian::new())?;

// Deserialize (returns model + distribution name)
let (model, dist_name) = GamlssModel::from_json(&json)?;
```

For browser-based fitting and prediction, build with the `wasm` feature:

```bash
wasm-pack build --no-default-features --features wasm
```

Note: Do not use the `--target web` flag, as it can cause issues with wasm-pack 0.14.0.

### Fitting in the Browser

```js
import { WasmGamlssModel } from './pkg/glissando.js';

const y = JSON.stringify([2.1, 4.0, 5.9, 8.1, 10.0]);
const data = JSON.stringify({ x: [1.0, 2.0, 3.0, 4.0, 5.0] });
const formula = JSON.stringify({
  mu: [{ Intercept: null }, { Linear: { col_name: "x" } }],
  sigma: [{ Intercept: null }],
});

const model = WasmGamlssModel.fit(y, data, formula, "Gaussian");
console.log("Converged:", model.converged());

// With custom configuration
// criterion: "reml" (default), "gcv", or "fellner_schall"
const config = JSON.stringify({ max_iterations: 200, tolerance: 0.001, criterion: "reml" });
const model2 = WasmGamlssModel.fitWithConfig(y, data, formula, "Gaussian", config);
```

Supported distributions: `Gaussian`, `Poisson`, `StudentT`, `Gamma`, `NegativeBinomial`, `Beta`. Note: `Binomial` is not supported in WASM as it requires state (number of trials) that cannot be recovered from the distribution name alone.

### Loading Pre-fitted Models

```js
const model = WasmGamlssModel.fromJson(modelJson);
```

### Prediction

```js
const predictions = JSON.parse(model.predict('{"x": [1, 2, 3]}'));

// With standard errors
const results = JSON.parse(model.predictWithSe('{"x": [1, 2, 3]}'));

// Access fitted values and coefficients directly
const mu_fitted = model.fittedValues("mu");
const mu_coeffs = model.coefficients("mu");

// Diagnostics
const diagnostics = JSON.parse(model.diagnosticsJson());
```

## Python bindings

Build the wheel with [maturin](https://github.com/PyO3/maturin):

```bash
maturin develop --release      # install into current venv
maturin build --release        # produce wheel under target/wheels/
```

```python
import numpy as np
from glissando import GamlssModel, Gaussian, Poisson

y = np.array([2.1, 4.0, 5.9, 8.1, 10.0])
data = {"x": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
formula = {
    "mu":    [("intercept",), ("linear", "x")],
    "sigma": [("intercept",)],
}

# Fit (with optional config dict)
model = GamlssModel.fit(data, y, formula, Gaussian())
model_cfg = GamlssModel.fit_with_config(
    data, y, formula, Gaussian(),
    {"max_iterations": 300, "tolerance": 1e-4, "criterion": "reml"},  # also: "gcv", "fellner_schall"
)

# Point predictions (response scale) for new data
preds = model.predict({"x": np.array([6.0, 7.0])})  # {"mu": np.ndarray, "sigma": …}

# Predictions with standard errors on the linear-predictor scale
se = model.predict_with_se({"x": np.array([6.0, 7.0])})
mu_block = se["mu"]   # {"fitted": …, "eta": …, "se_eta": …}

# Posterior samples (one fitted-value array per posterior draw)
samples = model.predict_samples({"x": np.array([6.0, 7.0])}, n_samples=500)
mu_samples = samples["mu"]   # list of np.ndarray of length n_obs

# Per-parameter accessors
mu_coefs = model.coefficients("mu")        # np.ndarray
mu_fits  = model.fitted_values("mu")       # np.ndarray
```

Supported distribution classes mirror the WASM surface — `Gaussian`, `Poisson`, `StudentT`, `Gamma`, `NegativeBinomial`, `Beta`, and `Binomial(n_trials)` (Binomial is Python-only because it carries `n_trials` state that can't be reconstructed from a name alone).

## Dependencies

**Core dependencies**:
- [ndarray](https://crates.io/crates/ndarray) - N-dimensional arrays (v0.17)
- [argmin](https://crates.io/crates/argmin) - L-BFGS optimization (v0.11)
- [statrs](https://crates.io/crates/statrs) - Statistical functions (v0.18)
- [rand](https://crates.io/crates/rand) - Random number generation (v0.10)

**Linear algebra backends** (select one):
- [ndarray-linalg](https://crates.io/crates/ndarray-linalg) - OpenBLAS backend (v0.18, `openblas` feature)
- [nalgebra](https://crates.io/crates/nalgebra) - Pure Rust backend (v0.33, `pure-rust` feature)

**Optional dependencies**:
- [indexmap](https://crates.io/crates/indexmap) - Insertion-ordered map for deterministic parameter iteration (v2)
- [rayon](https://crates.io/crates/rayon) - Parallel computation (v1.11, `parallel` feature)
- [serde](https://crates.io/crates/serde) / [serde_json](https://crates.io/crates/serde_json) - Serialization (`serialization` feature)
- [wasm-bindgen](https://crates.io/crates/wasm-bindgen) - JavaScript interop (v0.2, `wasm` feature)
- [pyo3](https://crates.io/crates/pyo3) / [numpy](https://crates.io/crates/numpy) - Python bindings (v0.28, `python` feature)

## Project Structure

This repository is a Cargo workspace:

- **`glissando`** (root) — the core library
- **`benchmark/`** (`glissando_benchmark`) — comparison framework against R/mgcv

## Algorithm

GAMLSS fitting uses a penalized quasi-likelihood approach (Rigby-Stasinopoulos algorithm):

1. **Initialization**: Set starting values for all distribution parameters
2. **Outer loop**: Cycle through distribution parameters
3. **Inner loop**: For each parameter, compute working response and weights from derivatives, then fit a penalized weighted least squares model
4. **Smoothing selection**: Optimize smoothing parameters via REML (default), GCV, or Fellner-Schall — selectable through `FitConfig::criterion`
5. **Convergence**: Check if coefficient changes are below tolerance

## Performance

The library includes several optimizations for large datasets:

- **Batched derivatives**: Distribution derivatives are computed for all observations at once, enabling SIMD vectorization
- **Parallel computation**: Special functions (digamma, trigamma) use Rayon parallel iterators for n >= 10,000
- **Warm-starting**: L-BFGS optimization reuses previous smoothing parameters for faster convergence
- **Efficient matrix operations**: Uses sqrt-weighted approach to avoid O(n²) memory allocation

## Benchmark (Comparison with R)

The `benchmark/` directory contains a comparison framework that validates glissando against R's mgcv and gamlss packages across 16 scenarios (linear, smooth, heteroskedastic, and a scale smooth via mgcv `gaulss`) and all supported distributions.

### Quick Start

```bash
# Build the Rust comparison binary
cargo build -p glissando_benchmark --release

# Run a single scenario
cargo run -p glissando_benchmark --release --bin compare_fit -- \
    --data benchmark/output/data_gaussian_linear.parquet \
    --scenario gaussian_linear \
    --output result.json

# Run the full comparison suite (requires Python with numpy/polars and R with mgcv)
./benchmark/run_comparison.sh
```

The orchestrator (`benchmark/orchestrate.py`) generates synthetic data with known parameters, fits models in both Rust and R, and produces a comparison summary with fitted value correlations, coefficient recovery, and timing.

## License

MIT
