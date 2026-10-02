# glissando

A Rust implementation of Generalized Additive Models for Location, Scale, and Shape (GAMLSS).

Ordinary regression models only the mean of the response.
GAMLSS gives each parameter of the response distribution (location, scale, and shape) its own regression on the predictors.
This makes it suited to heteroskedastic, skewed, and heavy-tailed data.

## Features

- **Multiple distribution parameters**: model mean, variance, and shape parameters simultaneously, across 12 families.
- **Flexible terms**: intercept, linear effects, factors, interactions, offsets, P-splines, cubic regression splines, tensor products, and random effects.
- **R-style formula strings**: `"y ~ s(x) + factor(g) + a:b + offset(log_e)"` parses into the same terms the builder API produces.
- **Automatic smoothing**: smoothing parameters selected via REML (default), GCV, or Fellner-Schall.
- **Link overrides**: any of 9 named links per parameter (`FitConfig::with_link`).
- **Distributional outputs**: `cdf` / `pdf` / `quantile` per family, randomized quantile residuals, and centile / quantile prediction.
- **Structural likelihoods** *(Rust and Python)*: censored, truncated, and hurdle responses via `Censored` / `Truncated` / `Hurdle` wrappers over any base family (survival-style data, detection limits, two-part zero models).
- **Finite mixtures** *(Rust API)*: `K`-component mixtures fit by EM (`fit_mixture` returns a `MixtureModel`).
- **Model selection**: GAIC at any penalty `k`, stepwise term selection (`step_gaic`), and ANOVA / likelihood-ratio comparison (`ic_table`, `lr_test`).
- **Fitting controls**: step-halving line search, global-deviance convergence in the RS loop, prior weights, and missing-value handling (`NaAction`).
- **Two backends**: OpenBLAS (default, faster) or pure Rust via nalgebra (no system deps).
- **WASM support**: fit models and predict directly in the browser via wasm-bindgen.
- **Python bindings**: a PyO3 extension built with maturin.
- **Type-safe API**: `DataSet`, `Formula`, and newtype wrappers prevent misuse.

## Installation

The pure-Rust backend needs no system libraries, so it builds on a clean machine and for WASM.

```toml
[dependencies]
glissando = { git = "https://github.com/real-limoges/glissando", default-features = false, features = ["pure-rust", "serialization"] }
```

The OpenBLAS backend is faster.
It is the default feature set and links against a system OpenBLAS (see [Requirements](#requirements)):

```toml
[dependencies]
glissando = { git = "https://github.com/real-limoges/glissando" }  # default = openblas + parallel
```

`openblas` and `pure-rust` are mutually exclusive; pick one backend.

### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `openblas` | OpenBLAS backend (ndarray-linalg), the faster of the two | yes |
| `pure-rust` | nalgebra backend, no system dependencies, WASM-compatible | no |
| `serialization` | Serde support for model serialization and the `glissando::json` facade | no |
| `wasm` | WASM fitting + prediction API (implies `pure-rust` + `serialization`, no parallelism) | no |
| `python` | PyO3 bindings for Python integration (implies `openblas` + `parallel` + `serialization`) | no |
| `parallel` | Rayon parallelism for large datasets (incompatible with WASM) | yes |

Because `openblas` is a default feature, anything that enables `pure-rust` or `wasm` must also pass `--no-default-features`; otherwise both backends activate and the build stops with a `compile_error!`.
The `wasm` feature automatically disables parallelism.

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
use glissando::ndarray::Array1;

let y = Array1::from_vec(vec![2.1, 4.0, 5.9, 8.1, 10.0]);

let mut data = DataSet::new();
data.insert_column("x", Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]));

let formula = Formula::new()
    .with_terms("mu", vec![
        Term::Intercept,
        Term::linear("x"),
    ])
    .with_terms("sigma", vec![Term::Intercept]);

let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

println!("Converged: {}", model.converged());
let mu_coeffs = &model.models["mu"].coefficients;
println!("Intercept: {}, Slope: {}", mu_coeffs[0], mu_coeffs[1]);
```

The same formula, written as R-style strings:

```rust
let formula = Formula::from_strings([("mu", "y ~ x"), ("sigma", "~ 1")])?;
```

> **ndarray version.**
> The public API hands back `ndarray` types (`Array1<f64>`, `Array2<f64>`), so you must build against the same `ndarray` major (currently **0.17**).
> The re-export `glissando::ndarray::Array1` resolves to the version this crate is built against.

## Distributions

| Distribution | Parameters | Default Links | Use Case |
|--------------|------------|---------------|----------|
| `Poisson` | mu | log | Count data |
| `Binomial` | mu | logit | Binary/count with known trials |
| `Gaussian` | mu, sigma | identity, log | Continuous data |
| `StudentT` | mu, sigma, nu | identity, log, floored-log (ν≥2) | Heavy-tailed continuous |
| `Gamma` | mu, sigma | log, log | Positive continuous |
| `Weibull` | mu, sigma | log, log | Positive continuous (survival, durations) |
| `NegativeBinomial` | mu, sigma | log, log | Overdispersed counts |
| `Beta` | mu, phi | logit, log | Proportions (0, 1) |
| `BCCG` | mu, sigma, nu | log, log, identity | Positive, skewed (Box-Cox Cole-Green; LMS centiles) |
| `BCT` | mu, sigma, nu, tau | log, log, identity, log | Positive, skewed and heavy-tailed (Box-Cox t) |
| `BCPE` | mu, sigma, nu, tau | log, log, identity, log | Positive, skewed, any kurtosis (Box-Cox power exponential) |
| `Ocat` | mu, delta_1 … delta_{R-1} | identity, identity, log … | Ordered categories, `R` = 2 to 5 levels (proportional odds) |

On top of any base family, three structural wrappers change the likelihood: `Censored`, `Truncated`, and `Hurdle` (see [Structural Likelihoods](#structural-likelihoods)).

### Usage

```rust
use glissando::distributions::{
    Poisson, Binomial, Gaussian, StudentT, Gamma, Weibull, NegativeBinomial, Beta,
    BCCG, BCT, BCPE, Ocat,
};

let poisson = Poisson::new();             // Count data
let binomial = Binomial::new(10);         // Binary/count with 10 trials
let gaussian = Gaussian::new();           // Continuous data
let student_t = StudentT::new();          // Heavy-tailed continuous data
let gamma = Gamma::new();                 // Positive continuous (e.g., durations)
let weibull = Weibull::new();             // Positive continuous (survival, time-to-event)
let neg_bin = NegativeBinomial::new();    // Overdispersed counts
let beta = Beta::new();                   // Proportions/rates in (0, 1)
let bccg = BCCG::new();                   // Skewed positive data (growth centiles)
let bct = BCT::new();                     // Skewed, heavy-tailed positive data
let bcpe = BCPE::new();                   // Skewed positive data, flexible kurtosis
let ocat = Ocat::new(4);                  // Ordered response coded 1..=4
```

### Ordered categories

`Ocat` takes a response coded `1..=R` and a formula block for `mu` plus each `delta_k`.
`predict_class_probabilities` returns an `(n_obs, R)` matrix whose rows sum to 1:

```rust
use glissando::distributions::Ocat;

let family = Ocat::new(3);
let formula = Formula::from_strings([
    ("mu", "y ~ x"),
    ("delta_1", "~ 1"),
    ("delta_2", "~ 1"),
])?;
let model = GamlssModel::fit(&data, &y, &formula, &family)?;
let probs = model.predict_class_probabilities(&new_data, &family)?;
```

## Structural Likelihoods

*(Rust and Python; not yet constructible on the WASM surface.)*

Censoring, truncation, and hurdle structure each transform a base family's likelihood using extra per-observation information.
Each is a wrapper that holds a boxed base `Distribution` and fits the base family's parameters through the standard RS loop.

**Censoring.**
Each row is either observed exactly (`Event`) or known only to lie below (`Left`), above (`Right`, the survival case), or within an interval (`Interval`):

```rust
use glissando::distributions::{Gaussian, Censored, CensorStatus};
use glissando::ndarray::Array1;

// y carries the observed time; status the censoring code per row.
let status = Array1::from_vec(vec![
    CensorStatus::Event, CensorStatus::Right, CensorStatus::Event, CensorStatus::Right,
]);
let family = Censored::new(Box::new(Gaussian::new()), status);
let model = GamlssModel::fit(&data, &y, &formula, &family)?;
// Interval censoring: Censored::with_upper(base, status, upper_bounds)
```

**Truncation.**
The response is only observed within `(lo, hi)`; out-of-range values are absent from the data rather than censored.
Use `±∞` for an open side:

```rust
use glissando::distributions::{Gaussian, Truncated};
use glissando::ndarray::Array1;

let lower = Array1::from_elem(y.len(), 0.0);            // left-truncated at 0
let upper = Array1::from_elem(y.len(), f64::INFINITY);
let family = Truncated::new(Box::new(Gaussian::new()), lower, upper);
let model = GamlssModel::fit(&data, &y, &formula, &family)?;
```

**Hurdle.**
A point mass at zero, plus a zero-truncated base family for the positive part.
It adds a logit-linked parameter `xi = P(Y = 0)`, so the formula needs an `"xi"` block:

```rust
use glissando::distributions::{Gamma, Hurdle};

let formula = Formula::new()
    .with_terms("mu", vec![Term::Intercept])
    .with_terms("sigma", vec![Term::Intercept])
    .with_terms("xi", vec![Term::Intercept]);     // the zero-atom probability
let family = Hurdle::new(Box::new(Gamma::new()));
let model = GamlssModel::fit(&data, &y, &formula, &family)?;
```

The censoring and truncation derivatives are analytic for the location and scale parameters of Gaussian and Student-t, and for the Gamma mean (through the `cdf_theta_derivatives` trait hook).
Every other parameter falls back to a central difference.

In Python, the wrappers take a base family instance: `Censored(Gaussian(), ["event", "right", ...])`, `Truncated(Gaussian(), lower, upper)`, `Hurdle(Gamma())`.

## Finite Mixtures

*(Rust API.)*
Fit a `K`-component mixture `f(y) = Σ_k w_k g_k(y)` by EM: an outer loop wrapped around the existing prior-weighted RS fit.

```rust
use glissando::{fit_mixture, FitConfig};
use glissando::distributions::Gaussian;

// `seed` makes the randomized EM initialization reproducible.
let mix = fit_mixture(&data, &y, &formula, &Gaussian::new(), 2, &FitConfig::default(), Some(42))?;

println!("converged: {}, weights: {:?}", mix.converged, mix.weights);
println!("mixture log-likelihood: {}, AIC: {}", mix.log_likelihood, mix.aic());

// Each component is a full GamlssModel.
for comp in &mix.components {
    println!("component mu intercept: {}", comp.models["mu"].coefficients[0]);
}

// Mixture mean on new data: Σ_k w_k · E_k[Y].
let mean = mix.predict_expected_value(&new_data, &Gaussian::new())?;

// MixtureModel round-trips through JSON too.
let blob = mix.to_json()?;
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
Term::linear("x")   // shorthand for Term::Linear { col_name: "x".into() }
```

### Factor

A categorical predictor (numeric level codes) expanded into dummy columns.
Treatment coding is the default; sum-to-zero is available.
Levels resolve from the training column at fit time and replay at predict time.

```rust
Term::factor("region")                               // treatment contrasts
Term::factor_with("region", Contrast::SumToZero)     // sum-to-zero contrasts
```

### Interaction

The row-wise product of two terms' design columns (`x:z` in a formula string).

```rust
Term::interaction(Term::linear("age"), Term::factor("sex"))
```

### Offset

A column added to the linear predictor with a fixed coefficient of 1, such as `log(exposure)` in a rate model.

```rust
Term::offset("log_exposure")
```

### P-Spline (1D Smooth)

A penalized B-spline smooth for nonlinear effects.
`Smooth::ps` has defaults (`n_splines = 10`, `degree = 3`, `penalty_order = 2`); chain builders to override.

```rust
Term::smooth(Smooth::ps("x"))                       // all defaults
Term::smooth(Smooth::ps("x").n_splines(20))         // override one default
```

### CR-Spline (1D Smooth)

A natural cubic regression spline (mgcv `bs = "cr"`); knots resolve from the data at fit time.

```rust
Term::smooth(Smooth::cr("x"))            // default k = 6
Term::smooth(Smooth::cr("x").k(10))      // more knots
Term::smooth(Smooth::cr("x").pc(0.0))    // pin f(0) = 0
```

### Tensor Product (2D Smooth)

Interaction smooth for two predictors.

```rust
Term::smooth(Smooth::tensor("x1", "x2"))
```

### Random Effect

Group-level random intercepts.

```rust
Term::smooth(Smooth::re("group"))
```

### Formula strings

`Formula::from_strings` (and `parse_formula_string` for a single string) accept an R/mgcv-style grammar.
Each `+`-separated piece is one term:

| Piece | Term |
|-------|------|
| `1` / `0` / `-1` | explicit intercept / suppress the (otherwise implicit) intercept |
| `x` | `Term::linear("x")` |
| `s(x)`, `s(x, k=10)` | P-spline |
| `s(x, bs="cr")` | cubic regression spline |
| `s(g, bs="re")` | random effect |
| `te(x, z)` | tensor product |
| `factor(g)`, `factor(g, sum)` | factor, treatment or sum-to-zero contrasts |
| `offset(e)` | offset |
| `a:b`, `a*b` | interaction; `a*b` expands to `a + b + a:b` |

The response name left of `~` is ignored (glissando takes `y` separately), so `"y ~ s(x)"` and `"~ s(x)"` mean the same thing.

## Configuration

```rust
use glissando::{FitConfig, GamlssModel, NaAction, SmoothingCriterion};

let config = FitConfig {
    max_iterations: 200,
    tolerance: 1e-3,
    criterion: SmoothingCriterion::Reml,  // also: Gcv, FellnerSchall
    ..FitConfig::default()                // step_halving, gd_tolerance, links, na_action
};

// The third argument is optional per-observation prior weights.
let model = GamlssModel::fit_with_config(
    &data, &y, None, &formula, &Gaussian::new(), config,
)?;

// Builders cover the less common knobs.
let config = FitConfig::default()
    .with_link("mu", "probit")            // override a parameter's link
    .with_na_action(NaAction::Fail);      // reject missing values instead of dropping rows
```

`SmoothingCriterion` selects the smoothing-parameter optimizer:

- `Reml` (default): Laplace-approximate marginal likelihood (Wood 2011), optimized via L-BFGS.
- `Gcv`: Generalized Cross-Validation (Craven & Wahba 1979), optimized via L-BFGS.
- `FellnerSchall`: multiplicative fixed-point update for the LAML target (Wood & Fasiolo 2017); deterministic, no line search.

The other fields:

- `step_halving` (default `true`): line-search each update on the global deviance, so every cycle is a monotone descent.
- `gd_tolerance` (default `1e-3`): absolute global-deviance change, the gamlss `c.crit` convention; convergence needs both this and `tolerance`.
- `links` (default empty): per-parameter link overrides.
- `na_action` (default `NaAction::DropRows`): drop any row with a non-finite value in `y` or a referenced column (R's `na.omit`), or `NaAction::Fail` to reject it.

### Link overrides

Every family has default links (see the table above), and any parameter can take one of 9 named links instead: `identity`, `log`, `logit`, `probit`, `cloglog`, `inverse`, `inverse_square`, `sqrt`, `cauchit`.
Two exceptions refuse an override: every `Ocat` parameter, and `StudentT`'s `nu`.
The fit checks that the parameter exists and that the link name is known, but it does **not** check that the link's range suits the parameter; a logit link on a Poisson mean pins it into `(0, 1)` and the fit still succeeds.

The same override is a `links` key in the JSON / WASM config (`{"links": {"mu": "probit"}}`) and in the Python config dict.

### Prior weights

`GamlssModel::fit_weighted(&data, &y, &weights, &formula, &family)` scales each observation's likelihood contribution, matching `mgcv::bam(..., weights = w)`.
Weights must be finite, non-negative, and the same length as `y`; a weight of zero excludes the row.

## Accessing Results

```rust
let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new())?;

// Convergence diagnostics
println!("Converged: {}", model.diagnostics.converged);
println!("Iterations: {}", model.diagnostics.iterations);
println!("Warnings: {:?}", model.diagnostics.warnings);

// Per-parameter results
let fitted_mu = &model.models["mu"];
fitted_mu.coefficients     // Coefficients newtype (Deref to Array1<f64>)
fitted_mu.covariance       // CovarianceMatrix newtype (Deref to Array2<f64>)
fitted_mu.fitted_values    // Fitted values on response scale
fitted_mu.eta              // Linear predictor (X * beta)
fitted_mu.edf              // Effective degrees of freedom
fitted_mu.term_edf         // Per-term EDF, summing to edf
fitted_mu.lambdas          // Smoothing parameters
fitted_mu.terms            // Formula terms

// mgcv-style accessors
let x = model.design_matrix(&new_data, "mu")?;      // predict(type = "lpmatrix")
let vcov = model.covariance_matrix("mu")?;
let index = model.term_index_map("mu")?;            // (term name, first col, end col exclusive)
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

// Posterior samples for uncertainty quantification; Some(seed) makes them reproducible
let samples = model.predict_samples(&new_data, &family, 1000, Some(42))?;
let mu_samples = &samples["mu"];  // Vec<Array1<f64>> with 1000 samples

// Raw coefficient draws from the posterior
let beta_draws = model.posterior_samples("mu", 1000, Some(42))?;

// Centile curves (response scale); percentiles in percent
let centiles = model.centiles(&new_data, &family, &[2.0, 10.0, 50.0, 90.0, 98.0])?;
let median = &centiles["C50"];  // one column per requested centile, keyed "C<pct>"

// Per-observation quantile prediction (row-varying level), for prediction intervals
let p = Array1::from_elem(median.len(), 0.95);
let upper_95 = model.quantile_prediction(&new_data, &family, &p)?;
```

Predicting with a different family than the one the model was fit with returns `GamlssError::FamilyMismatch`.

## Model Diagnostics

`GamlssModel::diagnostics` computes the aggregate diagnostics in one call:

```rust
let family = Gaussian::new();
let d = model.diagnostics(&family, &y)?;   // ModelDiagnostics

println!("log-likelihood {}, EDF {}", d.log_likelihood, d.total_edf);
println!("AIC {}, BIC {}", d.aic, d.bic);
let pearson = &d.pearson_residuals;          // (y - E[Y]) / sqrt(Var(Y))
let response = &d.response_residuals;        // y - E[Y]
```

The building blocks are public in `glissando::diagnostics` and work for every family: `pearson_residuals(family, y, params)`, `response_residuals`, `compute_gaic`, `compute_aic`, `compute_bic`, and `total_edf`.

**Randomized quantile residuals** (gamlss's default residual) map any family to N(0, 1) when the model is correct, so one Q-Q / worm plot works across distributions.
The discrete-family construction is randomized; pass a seed for reproducibility:

```rust
let family = Gaussian::new();
// seed = Some(s) makes discrete-family residuals reproducible; ignored for continuous families
let resid = model.quantile_residuals(&family, &y, Some(42))?;
```

## Model Selection

Compare and select models by an information criterion or a deviance test:

```rust
use glissando::selection::{ic_table, lr_test, step_gaic, Direction, StepScope};

let family = Gaussian::new();

// GAIC at any penalty: k = 2 is AIC, k = ln(n) is BIC
let aic = model.gaic(&family, &y, 2.0)?;

// Information-criterion table ranking several fitted models (nested or not)
let rows = ic_table(&[("null", &m0), ("with_x", &m1)], &family, &y, 2.0)?;

// Likelihood-ratio test of a nested pair (small ⊂ big)
let lrt = lr_test(&m0, &m1, &family, &y)?;       // { lr_stat, df, p_value }

// Greedy stepwise term selection by GAIC(k)
let scope = vec![StepScope { param: "mu".into(), candidates: vec![/* Linear / Smooth terms */] }];
let result = step_gaic(&data, &y, &family, start_formula, &scope,
                       (y.len() as f64).ln(), Direction::Both, FitConfig::default())?;
let selected = result.model;   // result.trace records the accepted moves
```

## Examples

Runnable programs live in `examples/` (`cargo run --example quickstart`, `families`, `selection`), with Python and JavaScript versions under `examples/python/` and `examples/wasm/`.

### Heteroskedastic Regression

Model where both mean and variance depend on x:

```rust
let formula = Formula::new()
    .with_terms("mu", vec![
        Term::Intercept,
        Term::linear("x"),
    ])
    .with_terms("sigma", vec![
        Term::Intercept,
        Term::linear("x"),
    ]);

let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new())?;
```

### Nonlinear Smooth

```rust
let formula = Formula::new()
    .with_terms("mu", vec![
        Term::smooth(Smooth::ps("x").n_splines(15)),
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
        Term::linear("predictor"),
    ]);

let model = GamlssModel::fit(&data, &counts, &formula, &Poisson::new())?;
```

### Binary/Binomial Data

```rust
use glissando::distributions::Binomial;

let formula = Formula::new()
    .with_terms("mu", vec![
        Term::Intercept,
        Term::linear("x"),
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
        Term::linear("x"),
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
        Term::linear("x"),
        Term::smooth(Smooth::re("subject_id")),
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
        Term::linear("x"),
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
        Term::linear("x"),
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
        Term::linear("age"),
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
| `Input` | Invalid input data, formula, or configuration |
| `MissingVariable` | Required variable not found in data |
| `MissingFormula` | Formula missing terms for a distribution parameter |
| `NonFiniteValues` | Variable contains NaN or Inf values (includes count of non-finite values); at fit time the default `NaAction::DropRows` drops such rows instead |
| `EmptyData` | No observations provided |
| `Convergence` | Algorithm failed to converge after N iterations |
| `Optimization` | Smoothing parameter optimization (L-BFGS) failed |
| `Linalg` | Linear algebra computation failed (Cholesky, matrix solve, etc.) |
| `PosteriorNotPositiveDefinite` | Posterior covariance is not positive definite, so `predict_samples` / `posterior_samples` failed Cholesky |
| `UnknownParameter` | Unknown parameter for the given distribution |
| `FamilyMismatch` | A model was used with a different family than the one it was fit with |
| `Shape` | Array shape mismatch |
| `Internal` | Internal logic error (indicates a bug) |

## Embedding glissando behind your own FFI

glissando has three interfaces: the typed Rust API, the WASM bindings, and the Python extension.
To embed the crate behind a different boundary (a [Rustler](https://github.com/rusterlium/rustler) NIF, a C ABI, a JSON service), you do not need to re-implement the wire format.
The `glissando::json` module (enabled by the `serialization` feature) exposes the JSON marshaling that the WASM bindings use.

```rust
use glissando::json;

// Strings in, model + boxed distribution out: keep the model in memory and
// predict interactively (no per-call re-fit).
let y       = "[1.2, 2.1, 2.9, 4.2, 4.8]";
let data    = r#"{"x": [1.0, 2.0, 3.0, 4.0, 5.0]}"#;
let formula = r#"{"mu": "y ~ x", "sigma": "~ 1"}"#;

// Trailing arguments: optional config JSON, optional prior-weights JSON.
let (model, family) = json::fit(y, data, formula, "Gaussian", None, None)?;

// Strings out: predictions, SEs, posterior samples, and fit diagnostics.
let preds       = json::predict(&model, family.as_ref(), r#"{"x": [6.0, 7.0]}"#)?;
let with_se     = json::predict_with_se(&model, family.as_ref(), r#"{"x": [6.0]}"#)?;
let samples     = json::predict_samples(&model, family.as_ref(), r#"{"x": [6.0]}"#, 500, Some(42))?;
let diagnostics = json::diagnostics(&model)?;   // converged, per-param + per-term EDF, warnings

// Persist and reload (round-trips the family descriptor, so wrappers/Binomial
// rebuild correctly; `json::load` returns the model + boxed distribution).
let blob = model.to_json(family.as_ref())?;
let (restored, family) = json::load(&blob)?;
# Ok::<(), glissando::GamlssError>(())
```

A formula can also be the structured form, a list of serialized `Term`s per parameter (`{"mu": [{"Intercept": null}, {"Linear": {"col_name": "x"}}]}`).

For typed dispatch instead of the string facade, `glissando::distributions::from_name("Gaussian") -> Box<dyn Distribution>` resolves any stateless family by name: `Gaussian`, `Poisson`, `StudentT`, `Gamma`, `NegativeBinomial`, `Beta`, `Weibull`, `BCCG`, `BCT`, `BCPE`.
`Binomial` (which carries `n_trials`), `Ocat` (which carries its category count), and the structural wrappers (which carry per-row state) are not name-resolvable; build them through the typed API.
The `json` parsing and serialization helpers (`parse_data`, `parse_formula`, `serialize_predictions`, and others) are also public, for use in a custom fitting flow.

## Serialization & WASM

Models can be serialized to JSON for transfer to browsers or other systems.
Enable with the `serialization` feature:

```toml
[dependencies]
glissando = { git = "...", features = ["serialization"] }
```

```rust
// Serialize a fitted model (native side). The JSON bundles a `FamilyDescriptor`,
// so stateful families (`Binomial`, `Ocat`) and the structural wrappers round-trip
// too, not just a bare distribution name.
let json = model.to_json(&Gaussian::new())?;

// Deserialize: returns the model and a `FamilyDescriptor`; call `.build()` to
// reconstruct the boxed distribution.
let (model, descriptor) = GamlssModel::from_json(&json)?;
let family = descriptor.build()?;   // Box<dyn Distribution>
```

For browser-based fitting and prediction, build with the `wasm` feature (and `--no-default-features`, to drop the default `openblas` backend):

```bash
wasm-pack build --no-default-features --features wasm
```

`--target web` had known issues with wasm-pack 0.14; it builds and loads cleanly on 0.15.

### Fitting in the Browser

```js
import { WasmGamlssModel } from './pkg/glissando.js';

const y = JSON.stringify([2.1, 4.0, 5.9, 8.1, 10.0]);
const data = JSON.stringify({ x: [1.0, 2.0, 3.0, 4.0, 5.0] });
const formula = JSON.stringify({ mu: "y ~ x", sigma: "~ 1" });

// Note the argument order: y before data.
const model = WasmGamlssModel.fit(y, data, formula, "Gaussian");
console.log("Converged:", model.converged());

// With custom configuration
// criterion: "reml" (default), "gcv", or "fellner_schall"
const config = JSON.stringify({
  max_iterations: 200, tolerance: 0.001, criterion: "reml",
  links: { mu: "identity" },
});
const model2 = WasmGamlssModel.fitWithConfig(y, data, formula, "Gaussian", config);
```

Both `fit` and `fitWithConfig` take an optional trailing prior-weights JSON array.

Supported distributions are the name-resolvable ones: `Gaussian`, `Poisson`, `StudentT`, `Gamma`, `NegativeBinomial`, `Beta`, `Weibull`, and the Box-Cox family `BCCG`, `BCT`, `BCPE`.
`Binomial`, `Ocat`, and the structural wrappers are not available in WASM, because they carry state that cannot be recovered from a distribution name alone.

### Loading Pre-fitted Models

```js
const model = WasmGamlssModel.fromJson(modelJson);
const blob = model.toJson();
```

### Prediction

```js
const predictions = JSON.parse(model.predict('{"x": [1, 2, 3]}'));

// With standard errors
const results = JSON.parse(model.predictWithSe('{"x": [1, 2, 3]}'));

// Posterior samples; the seed is a BigInt (or undefined for an unseeded RNG)
const samples = JSON.parse(model.predictSamples('{"x": [1, 2, 3]}', 500, 42n));

// Access fitted values and coefficients directly
const mu_fitted = model.fittedValues("mu");
const mu_coeffs = model.coefficients("mu");

// Diagnostics
const diagnostics = JSON.parse(model.diagnosticsJson());

// Distributional outputs and selection (all JSON strings)
const curves = JSON.parse(model.centiles('{"x": [1, 2, 3]}', "[10, 50, 90]"));
const { gaic } = JSON.parse(model.gaic(y, 2.0));
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

# A formula string per parameter...
formula = {"mu": "y ~ x", "sigma": "~ 1"}
# ...or a list of term tuples: ("intercept",), ("linear", "x"),
# ("smooth", "x", {"n_splines": 10}), ("random", "group")
formula_tuples = {
    "mu":    [("intercept",), ("linear", "x")],
    "sigma": [("intercept",)],
}

# Fit (optional prior weights; optional config dict)
model = GamlssModel.fit(data, y, formula, Gaussian())
model_cfg = GamlssModel.fit_with_config(
    data, y, formula, Gaussian(),
    {
        "max_iterations": 300,
        "tolerance": 1e-4,
        "criterion": "reml",             # also: "gcv", "fellner_schall"
        "links": {"mu": "identity"},     # per-parameter link overrides
        "na_action": "drop_rows",        # or "fail"
    },
)

# Point predictions (response scale) for new data
preds = model.predict({"x": np.array([6.0, 7.0])})  # {"mu": np.ndarray, "sigma": ...}

# Predictions with standard errors on the linear-predictor scale
se = model.predict_with_se({"x": np.array([6.0, 7.0])})
mu_block = se["mu"]   # {"fitted": ..., "eta": ..., "se_eta": ...}

# Posterior samples (one fitted-value array per posterior draw)
samples = model.predict_samples({"x": np.array([6.0, 7.0])}, n_samples=500, seed=42)
mu_samples = samples["mu"]   # list of np.ndarray of length n_obs

# Per-parameter accessors
mu_coefs = model.coefficients("mu")        # np.ndarray
mu_fits  = model.fitted_values("mu")       # np.ndarray

# Distributional outputs
resid    = model.quantile_residuals(y, seed=42)        # randomized quantile residuals
curves   = model.centiles(data, [10.0, 50.0, 90.0])    # {"C10": ..., "C50": ..., "C90": ...}
q95      = model.quantile_prediction(data, np.full(len(y), 0.95))

# Model selection
aic   = model.gaic(y, 2.0)                              # k = 2 is AIC, ln(n) is BIC
table = GamlssModel.ic_table([("null", m0), ("x", m1)], y, 2.0)   # list of dicts
lrt   = m0.lr_test(m1, y)                               # {"lr_stat", "df", "p_value"}
out   = GamlssModel.step_gaic(data, y, Gaussian(), start, scope, np.log(len(y)),
                              "forward")                 # {"model": GamlssModel, "trace": [...]}
```

Python exposes every family the Rust API has: `Gaussian`, `Poisson`, `StudentT`, `Gamma`, `NegativeBinomial`, `Beta`, `Weibull`, the Box-Cox family `BCCG`, `BCT`, `BCPE`, plus the stateful ones WASM cannot build:

- `Binomial(n_trials)`, where `n_trials` is a list of per-row trial counts.
- `Ocat(n_categories)`, with `model.predict_class_probabilities(new_data)`.
- The structural wrappers `Censored(base, status, upper=None)`, `Truncated(base, lower, upper)`, and `Hurdle(base)`.

Finite mixtures are Rust-only for now.

## Dependencies

**Core dependencies**:
- [ndarray](https://crates.io/crates/ndarray) - N-dimensional arrays (v0.17)
- [statrs](https://crates.io/crates/statrs) - Statistical functions (v0.18)
- [rand](https://crates.io/crates/rand) - Random number generation (v0.10)
- [indexmap](https://crates.io/crates/indexmap) - Insertion-ordered map for deterministic parameter iteration (v2)

**Linear algebra backends** (select one):
- [ndarray-linalg](https://crates.io/crates/ndarray-linalg) - OpenBLAS backend (v0.18, `openblas` feature)
- [nalgebra](https://crates.io/crates/nalgebra) - Pure Rust backend (v0.33, `pure-rust` feature)

**Optional dependencies**:
- [rayon](https://crates.io/crates/rayon) - Parallel computation (v1.11, `parallel` feature)
- [serde](https://crates.io/crates/serde) / [serde_json](https://crates.io/crates/serde_json) - Serialization (`serialization` feature)
- [wasm-bindgen](https://crates.io/crates/wasm-bindgen) - JavaScript interop (v0.2, `wasm` feature)
- [pyo3](https://crates.io/crates/pyo3) / [numpy](https://crates.io/crates/numpy) - Python bindings (v0.28, `python` feature)

## Project Structure

This repository is a Cargo workspace:

- **`glissando`** (root): the core library
- **`benchmark/`** (`glissando_benchmark`): comparison framework against R/mgcv
- **`fuzz/`**: cargo-fuzz targets (its own workspace, nightly only)
- **`pkg/`**: the generated WASM/npm package
- **`docs/reference/`**: the cookbook and the per-family mathematics (`mathematics.md`)

## Algorithm

GAMLSS fitting uses a penalized quasi-likelihood approach (Rigby-Stasinopoulos algorithm):

1. **Initialization**: set starting values for all distribution parameters.
2. **Outer loop**: cycle through distribution parameters.
3. **Inner loop**: for each parameter, compute working response and weights from derivatives, then fit a penalized weighted least squares model.
4. **Smoothing selection**: optimize smoothing parameters via REML (default), GCV, or Fellner-Schall, selectable through `FitConfig::criterion`.
5. **Step control**: backtrack each block update on the penalized deviance (monotone descent).
6. **Convergence**: both the per-parameter linear-predictor change and the absolute global-deviance change (gamlss `c.crit` convention) must fall below tolerance.

## Performance

The library includes several optimizations for large datasets:

- **Batched derivatives**: distribution derivatives are computed for all observations at once, enabling SIMD vectorization.
- **Parallel computation**: special functions (digamma, trigamma) use Rayon parallel iterators for n >= 10,000.
- **Warm-starting**: L-BFGS optimization reuses previous smoothing parameters for faster convergence.
- **Square-root weighting**: a sqrt-weighted approach avoids O(n²) memory allocation.

## Benchmark (Comparison with R)

The `benchmark/` directory contains a comparison framework that validates glissando against R's mgcv and gamlss packages across 25 scenarios (linear, smooth, heteroskedastic, tensor, random-effect, weighted, and scale smooths via mgcv `gaulss`) and all supported distributions.

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
See `benchmark/README.md` for the full scenario list.

## License

MIT
