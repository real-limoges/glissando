# Glissando Benchmark Suite

This is how I keep the Rust `glissando` implementation honest: I fit the same models in R's established GAMLSS tools (`mgcv` and `gamlss`) and check that the numbers agree.

## Overview

The benchmark holds glissando (Rust) up against two R oracles:
- **R/mgcv**: for Gaussian, Poisson, Binomial, Gamma, Negative Binomial, and Beta, including the tensor-product, random-effect and cubic-regression-spline scenarios.
- **R/gamlss**: the like-for-like oracle for Student-t (`TF()`).
  It is the oracle because it implements the same Rigby–Stasinopoulos algorithm and the same (μ, σ, ν) location-scale-df parameterization glissando uses, so a disagreement is a real bug rather than a difference of convention.
  mgcv's `scat()` is *also* run for Student-t, but only as a loose, μ-only cross-method sanity check: it cannot validate σ/ν/EDF/SE, because it folds σ and ν into internal nuisance scalars instead of exposing them as modeled predictors.

The suite draws synthetic data with known parameters, fits it in both implementations, and writes out detailed comparison reports.
Each scenario is replicated (25 replicates by default), and the reference test gates the distribution of drift across replicates rather than a single sample.

## Quick Start

### Prerequisites

**Rust** (with OpenBLAS):
```bash
brew install openblas  # macOS
# or
sudo apt-get install libopenblas-dev  # Ubuntu/Debian
```

**Python**: the environment is managed by [uv](https://docs.astral.sh/uv/) from `pyproject.toml` and `uv.lock`.
`run_comparison.sh` calls `uv run`, which creates the environment on first use; to create it ahead of time:
```bash
cd benchmark
uv sync
```

**R with packages** (optional, for the full comparison):
```r
install.packages(c("mgcv", "gamlss", "jsonlite", "optparse"))
install.packages("arrow")  # or "nanoparquet"; either one reads the parquet inputs
```
`common.R` uses `arrow` when it is installed and falls back to the dependency-free `nanoparquet` otherwise.
Without `gamlss`, Student-t parity falls back to the loose mgcv `scat()` check.

### Run Full Comparison

```bash
cd benchmark
./run_comparison.sh                # 1000 observations, 25 replicates per scenario
REPS=5 ./run_comparison.sh         # fewer replicates for a quicker local pass
N_OBS=500 SEED=7 ./run_comparison.sh
```

Once the data is regenerated, `tests/mgcv_reference.rs` checks the glissando results against mgcv coefficient-by-coefficient and pointwise on fitted μ.
There are two entry points into the same comparison logic (`assert_parity`):

```bash
# Regenerate-and-check: reads the freshly-produced summary, requires it to exist.
cargo test --test mgcv_reference -- --ignored

# Per-commit gate: checks the committed fixture if present, skips cleanly if not.
cargo test --test mgcv_reference glissando_matches_committed_mgcv_fixture
```

Both assert agreement within scenario-aware tolerances (~1e-3 relative for linear models, ~5% for smooths).
The `--ignored` test is gated because it needs R to have just produced the data; the per-commit gate is not `#[ignore]`d and runs in the normal suite.

### Fixture cadence

`benchmark/output/` is gitignored except for one allowlisted file, `comparison_summary.json`, which is the fixture the per-commit gate reads.
The nightly `r-parity` workflow is the source of truth: it installs R + mgcv + gamlss, regenerates the summary, asserts parity, and uploads the regenerated `comparison_summary.json` as a build artifact.
When families or scenarios change, download that artifact and commit it to refresh the guardrail.
Machines and CI jobs without the fixture simply skip the per-commit gate, so a fresh clone stays green.

## Commands

### Build

```bash
cargo build -p glissando_benchmark --release
```

The benchmark is a member of the root Cargo workspace, so the binaries land in the workspace `target/` directory (`../target/release/` from inside `benchmark/`), not in `benchmark/target/`.

### Run Individual Scenario

```bash
cd benchmark

# Generate data (one replicate)
uv run python3 orchestrate.py --generate-only --output-dir ./test_data --reps 1 --scenarios gaussian_linear

# Run Rust
../target/release/compare_fit \
  --data ./test_data/data_gaussian_linear_rep0.parquet \
  --scenario gaussian_linear \
  --output result.json

# Run R
Rscript fit_mgcv.R \
  --data ./test_data/data_gaussian_linear_rep0.parquet \
  --scenario gaussian_linear \
  --output result_r.json
```

## Scenarios

`orchestrate.py` defines 25 scenarios in its `SCENARIOS` registry.
Every scenario uses 1000 observations by default (`N_OBS`, or `--n-obs`), except `gaussian_large`, which uses 10,000 observations and is capped at 5 replicates.
"Smooth" means a P-spline `s(x, bs="ps")` unless noted.

| Scenario | Distribution | Fitted model | Oracle |
|----------|--------------|--------------|--------|
| `gaussian_linear` | Gaussian | mu ~ x | mgcv |
| `gaussian_heteroskedastic` | Gaussian | mu ~ x; log(sigma) ~ x | mgcv `gaulss` |
| `gaussian_smooth` | Gaussian | mu ~ s(x) | mgcv |
| `gaussian_multiple` | Gaussian | mu ~ x1 + x2 + x3 | mgcv |
| `gaussian_large` | Gaussian | mu ~ x, n = 10,000 | mgcv |
| `gaussian_quadratic` | Gaussian | mu ~ s(x), quadratic truth | mgcv |
| `gaussian_sigma_smooth` | Gaussian | mu ~ 1; log(sigma) ~ s(x) | mgcv `gaulss` |
| `gaussian_cr_smooth` | Gaussian | mu ~ s(x, bs="cr") | mgcv |
| `tensor_smooth` | Gaussian | mu ~ te(x1, x2) | mgcv |
| `random_effect` | Gaussian | mu ~ x + s(g, bs="re"), 10 groups | mgcv |
| `b1_weighted_gaussian` | Gaussian | five smooths + a binary dummy, prior weights | mgcv |
| `poisson_linear` | Poisson | log(mu) ~ x | mgcv |
| `poisson_smooth` | Poisson | log(mu) ~ s(x) | mgcv |
| `binomial_linear` | Binomial (Bernoulli) | logit(mu) ~ x | mgcv |
| `binomial_smooth` | Binomial (Bernoulli) | logit(mu) ~ s(x) | mgcv |
| `gamma_linear` | Gamma | log(mu) ~ x | mgcv |
| `gamma_smooth` | Gamma | log(mu) ~ s(x) | mgcv |
| `gamma_sigma_smooth` | Gamma | mu ~ 1; log(sigma) ~ s(x) | mgcv `gammals` |
| `studentt_linear` | Student-t | mu ~ x | gamlss `TF()` (mgcv `scat()` for μ) |
| `studentt_smooth` | Student-t | mu ~ s(x) | gamlss `TF()` (mgcv `scat()` for μ) |
| `b2_weighted_studentt` | Student-t | four smooths, prior weights | gamlss `TF()` (mgcv `scat()` for μ) |
| `negative_binomial_linear` | Negative Binomial | log(mu) ~ x | mgcv |
| `negative_binomial_smooth` | Negative Binomial | log(mu) ~ s(x) | mgcv |
| `beta_linear` | Beta | logit(mu) ~ x | mgcv |
| `beta_smooth` | Beta | logit(mu) ~ s(x) | mgcv |

Adding or renaming a scenario touches three places in lockstep: the `SCENARIOS` list in `orchestrate.py`, a match arm in `src/bin/compare_fit.rs`, and a fitter in `fit_mgcv.R`.

## Ordered-categorical (Ocat) benchmark

`Ocat` has its own harness outside the scenario suite, because it compares category-probability matrices on held-out data rather than fitted μ.
It fits `Ocat(R=4)` against mgcv `ocat(R=4)`: an intercept-only log-likelihood cross-check, and a probability-matrix comparison under a two-smooth model.

```bash
cd benchmark
uv run python ocat_benchmark.py --output-dir /tmp/ocat_bench
```

- `gen_ocat_spike.py` generates the train/test parquets.
- `src/bin/fit_ocat.rs` (`fit_ocat`) is the glissando side; `--intercept-only` selects the log-likelihood cross-check mode.
- `fit_ocat_mgcv.R` is the mgcv side.
- `src/bin/spike_ocat.rs` (`spike_ocat`) and `spike_ocat_report.py` are the original Phase 0 spike: three independent Binomial cumulative-logit fits, kept as a baseline that shows why a joint Ocat model is needed (the independent fits can violate monotonicity).

## Output Files

All paths are under `output/`, and every per-fit file is suffixed with its replicate index.

- **`comparison_summary.json`**: aggregate metrics (schema v2), the file `tests/mgcv_reference.rs` reads.
- **`data_<scenario>_rep<k>.parquet`**: generated test data.
- **`glissando_<scenario>_rep<k>.json`**: Rust fitting results.
- **`mgcv_<scenario>_rep<k>.json`** and **`gamlss_<scenario>_rep<k>.json`**: R fitting results.

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
- numpy, polars, pyarrow (managed by uv)

### Rust
- glissando (path dependency, built with its default `openblas` + `parallel` features)
- ndarray, polars, serde/serde_json

### System
- OpenBLAS
- R with mgcv, gamlss, jsonlite, optparse, and arrow or nanoparquet (optional)

## License

See main glissando LICENSE file.
