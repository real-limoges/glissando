# Glissando Benchmark Suite

This is how I keep the Rust `glissando` implementation honest: I fit the same models in R's established GAMLSS tools (`mgcv` and `gamlss`) and check that the numbers agree.

## Overview

The benchmark holds glissando (Rust) up against two R oracles:
- **R/mgcv**: for Gaussian, Poisson, Gamma, Negative Binomial, and Beta.
- **R/gamlss**: the like-for-like oracle for Student-t (`TF()`). It is the oracle because it
  implements the same Rigby–Stasinopoulos algorithm and the same (μ, σ, ν)
  location-scale-df parameterization glissando uses, so a disagreement is a real bug rather
  than a difference of convention. mgcv's `scat()` is *also* run for Student-t, but only as a
  loose, μ-only cross-method sanity check: it cannot validate σ/ν/EDF/SE, because it folds σ
  and ν into internal nuisance scalars instead of exposing them as modeled predictors.

The suite draws synthetic data with known parameters, fits it in both implementations, and writes out detailed comparison reports.

## Quick Start

### Prerequisites

**Rust** (with OpenBLAS):
```bash
brew install openblas  # macOS
# or
sudo apt-get install libopenblas-dev  # Ubuntu/Debian
```

**Python** with dependencies:
```bash
cd benchmark
pip install -e .
```

**R with packages** (optional, for full comparison):
```r
install.packages(c("arrow", "mgcv", "jsonlite", "optparse", "gamlss"))
```

### Run Full Comparison

```bash
cd benchmark
./run_comparison.sh
```

Once the data is regenerated, `tests/mgcv_reference.rs` checks the glissando results
against mgcv coefficient-by-coefficient and pointwise on fitted μ:

```bash
cargo test --test mgcv_reference -- --ignored
```

The test reads `benchmark/output/comparison_summary.json` (gitignored) and asserts
agreement within scenario-aware tolerances (~1e-3 relative for linear models, ~5% for
smooths). It is `#[ignore]`-gated on purpose: the comparison output has to exist locally
first, and CI without R has no way to produce it.

## Commands

### Build

```bash
cargo build -p glissando_benchmark --release
```

### Run Individual Scenario

```bash
# Generate data
python3 orchestrate.py --generate-only --output-dir ./test_data --scenarios gaussian_linear

# Run Rust
./target/release/compare_fit \
  --data ./test_data/data_gaussian_linear.parquet \
  --scenario gaussian_linear \
  --output result.json

# Run R
Rscript fit_mgcv.R \
  --data ./test_data/data_gaussian_linear.parquet \
  --scenario gaussian_linear \
  --output result_r.json
```

## Scenarios

All scenarios use 500 observations by default (configurable via `N_OBS`).

| Scenario | Distribution | Formula | Use Case |
|----------|--------------|---------|----------|
| `gaussian_linear` | Gaussian | mu ~ x; sigma constant | Linear regression |
| `gaussian_heteroskedastic` | Gaussian | mu ~ x; log(sigma) ~ x | Heteroskedastic regression |
| `gaussian_smooth` | Gaussian | mu ~ smooth(x) | Nonlinear mean |
| `poisson_linear` | Poisson | log(mu) ~ x | Count regression |
| `poisson_smooth` | Poisson | log(mu) ~ smooth(x) | Nonlinear counts |
| `gamma_linear` | Gamma | log(mu) ~ x | Positive continuous |
| `gamma_smooth` | Gamma | log(mu) ~ smooth(x) | Nonlinear gamma |
| `studentt_linear` | Student-t | mu ~ x | Heavy-tailed data |
| `studentt_smooth` | Student-t | mu ~ smooth(x) | Heavy-tailed smooth |
| `negative_binomial_linear` | Negative Binomial | log(mu) ~ x | Overdispersed counts |
| `beta_linear` | Beta | logit(mu) ~ x | Proportions in (0,1) |

## Output Files

- **`output/comparison_summary.json`** - Aggregate metrics
- **`output/data_*.parquet`** - Generated test data
- **`output/rust_result_*.json`** - Rust fitting results
- **`output/r_result_*.json`** - R fitting results

## Interpretation

### Convergence
Both implementations should converge on every scenario.
If one of them does not, that is the finding.

### Performance
Speedup = R time / Rust time (typically 2-10x on large data).

### Accuracy
- **Correlation**: > 0.99 for linear, > 0.95 for smooth.
- **RMSE**: smaller is better.
- **Coefficient differences**: within ~1e-6 to 1e-3.

## Dependencies

### Python
- numpy, polars, pyarrow

### Rust
- glissando (path dependency)
- ndarray, polars, serde/serde_json

### System
- OpenBLAS
- R with arrow, mgcv, gamlss (optional)

## License

See main glissando LICENSE file.
