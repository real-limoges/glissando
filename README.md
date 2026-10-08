# glissando

A Rust implementation of Generalized Additive Models for Location, Scale, and Shape (GAMLSS).

Ordinary regression models only the mean of the response.
GAMLSS gives each parameter of the response distribution (location, scale, and shape) its own regression on the predictors, which suits heteroskedastic, skewed, and heavy-tailed data.
glissando fits them with the Rigby-Stasinopoulos algorithm and ships as a Rust crate, a WASM/npm package, and a Python extension.

## Features

- **12 families**: Gaussian, Poisson, Binomial, Student-t, Gamma, Weibull, negative binomial, Beta, BCCG, BCT, BCPE, and ordered categorical, plus censored, truncated, and hurdle wrappers over any of them and finite mixtures fit by EM.
- **Flexible terms**: linear effects, factors, interactions, offsets, P-splines, cubic regression splines, tensor products, and random effects, from a builder API or R-style strings (`"y ~ s(x) + factor(g)"`).
- **Automatic smoothing**: REML (default), GCV, or Fellner-Schall.
- **Distributional outputs**: centiles, quantile prediction, posterior samples, and randomized quantile residuals.
- **Model selection**: GAIC, stepwise term selection, and likelihood-ratio tests.
- **Two backends**: OpenBLAS (default, faster) or pure Rust via nalgebra (no system dependencies, WASM-compatible).

## Installation

```toml
[dependencies]
glissando = { git = "https://github.com/real-limoges/glissando" }  # openblas + parallel
```

The default backend links a system OpenBLAS (`brew install openblas`, or `apt-get install libopenblas-dev`).
For no system dependencies, use the pure-Rust backend:

```toml
glissando = { git = "https://github.com/real-limoges/glissando", default-features = false, features = ["pure-rust"] }
```

| Feature | Description | Default |
|---------|-------------|---------|
| `openblas` | OpenBLAS backend, the faster of the two | Yes |
| `pure-rust` | nalgebra backend, no system dependencies | No |
| `parallel` | Rayon parallelism (incompatible with WASM) | Yes |
| `serialization` | Serde support and the `glissando::json` facade | No |
| `wasm` | WASM bindings (implies `pure-rust` and `serialization`) | No |
| `python` | PyO3 bindings (implies `openblas`, `parallel`, and `serialization`) | No |

The two backends are mutually exclusive, so enabling `pure-rust` or `wasm` also needs `default-features = false` (or `--no-default-features`).

## Quick Start

```rust
use glissando::{GamlssModel, DataSet, Formula};
use glissando::distributions::Gaussian;
use glissando::ndarray::Array1;

let y = Array1::from_vec(vec![2.1, 4.0, 5.9, 8.1, 10.0]);
let mut data = DataSet::new();
data.insert_column("x", Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]));

// One predictor per distribution parameter.
let formula = Formula::from_strings([("mu", "y ~ x"), ("sigma", "~ 1")])?;
let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new())?;

let preds = model.predict(&data, &Gaussian::new())?;
```

Build arrays through the re-exported `glissando::ndarray`; a different `ndarray` major (the crate uses 0.17) produces confusing type-mismatch errors.

## Documentation

- [Quickstart](docs/reference/cookbook-quickstart.md): one model end to end in Rust, Python, and JavaScript.
- [Distribution gallery](docs/reference/cookbook-families.md): every family, the structural wrappers, and mixtures.
- [Rust API](docs/reference/rust-api.md): terms, configuration, prediction, diagnostics, selection, and errors.
- [Serialization and bindings](docs/reference/bindings.md): saving models, the JSON facade for FFI, and Python.
- [WASM/npm package](pkg/README.md): fitting and predicting in the browser.
- [Migrating from R](docs/reference/cookbook-mgcv-migration.md): `mgcv` and `gamlss` equivalents and current gaps.
- [Mathematics](docs/reference/mathematics.md): per-family likelihoods, scores, and the fitting algorithm.

Runnable examples are under `examples/` (`cargo run --example quickstart`).
[`benchmark/`](benchmark/README.md) validates fits against R's `mgcv` and `gamlss` across 25 scenarios.

## License

MIT
