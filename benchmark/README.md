# Glissando Benchmark Suite

Comparison framework for validating the Rust `glissando` implementation against R's established GAMLSS tools (`mgcv` and `gamlss` packages).

## Overview

This benchmark compares glissando (Rust) against:
- **R/mgcv**: For Gaussian, Poisson, Gamma, Negative Binomial, and Beta distributions
- **R/gamlss**: For Student-t distribution and advanced GAMLSS features

The suite generates synthetic data with known parameters, fits models in both implementations, and produces detailed comparison reports.

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
Both implementations should converge for all scenarios.

### Performance
Speedup = R time / Rust time (typical range: 2-10x for large data)

### Accuracy
- **Correlation**: Should be > 0.99 for linear, > 0.95 for smooth
- **RMSE**: Smaller is better
- **Coefficient differences**: Should be within ~1e-6 to 1e-3

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
