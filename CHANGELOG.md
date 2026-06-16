# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `impl Display for GamlssModel` produces an R-style summary block (convergence
  status, iteration count, per-parameter EDF + λ values, truncated coefficients
  head).
- `Clone` derive on `fitting::FittedParameter` so callers can duplicate fitted
  parameters without a serde round-trip.
- `ParamDiagnostic` now carries `weight_floor_hits: usize` and
  `step_cap_hits: usize` — counts of observations whose IRLS weight or step
  was clamped in the final iteration. Non-zero values signal a degenerate fit.
- Python bindings reach parity with WASM: new methods `fit_with_config`,
  `predict_with_se`, `predict_samples`, `fitted_values(param)`,
  `coefficients(param)` on `glissando.GamlssModel`.
- New `GamlssError::PosteriorNotPositiveDefinite` variant returned when the
  posterior covariance fails Cholesky factorization.
- `scripts/audit-duplicates.sh` (+ CI job + pre-push hook) enforces an
  allowlist of duplicate transitive dependencies, failing if a new duplicate
  appears.
- New test suites: `tests/correctness.rs` (B.1/B.2/B.3 + D.4 coverage),
  `tests/analytic.rs` (closed-form OLS anchor), `tests/terms.rs`
  (`Term::Smooth(RandomEffect)` end-to-end).
- A `proptest` in `src/preprocessing` exercises NaN/∞ rejection in
  `validate_inputs`.
- New Poisson cases for `predict_with_se` and `predict_samples` integration
  tests covering the non-identity-link path.
- `FlooredLogLink` — a log link with a lower bound on the response-scale value
  (`μ = max(exp(η), floor)`). Used for StudentT's ν with `floor = 2` so the
  variance `σ²ν/(ν−2)` stays finite as the optimizer explores the heavy-tail
  region.
- Student-t parity is now validated against R **gamlss `TF()`** — the
  like-for-like Rigby–Stasinopoulos oracle (same μ/σ/ν parameterization) — via
  the new `benchmark/fit_gamlss.R`, wired into `orchestrate.py` /
  `run_comparison.sh`. This gates μ, σ, ν, EDF, SE and the (unweighted)
  log-likelihood; mgcv `scat()` is retained only as a loose μ-only cross-method
  sanity check.

### Changed

- StudentT initialization is now robust to heavy tails: `μ` seeds from the
  sample median, `σ` from `1.4826·MAD(y)` (instead of mean / sample SD, which the
  tails bias), and `ν` from a fixed seed of 5 (a sample-kurtosis seed is biased
  for regression data and could tip the multi-smooth weighted fit into a
  degenerate over-smoothed basin).
- StudentT's `ν` now uses a floored log link (`ν ≥ 2`) instead of a plain log
  link, guaranteeing finite variance; the floor does not bind when the true `ν`
  is well above 2. The reported `nu` default link changes from `log` to
  `floored-log`.
- `tests/mgcv_reference.rs` validates StudentT against gamlss `TF()` with tight,
  measured tolerances, replacing the previous mgcv `scat()` comparison whose
  parameterization mismatch had forced a 40% `fitted_mu` tolerance and skipped
  σ/ν/EDF/SE entirely.

- **Breaking.** `GamlssModel.models` is now `IndexMap<String, FittedParameter>`
  instead of `HashMap<String, FittedParameter>`. Iteration over `models`,
  `predict_samples` output, and `posterior_samples` is now deterministic
  (`family.parameters()` order). Downstream code that uses `model.models["mu"]`
  works unchanged.
- **Breaking.** `fitting::sample_posterior` returns
  `Result<Vec<Array1<f64>>, GamlssError>` (was `Vec<Array1<f64>>`). Failure
  is now surfaced as `PosteriorNotPositiveDefinite` instead of a silent empty
  vector.
- **Breaking.** `GamlssModel::posterior_samples` returns
  `Result<Vec<Coefficients>, GamlssError>` (was `Vec<Coefficients>`), and
  also surfaces `UnknownParameter` if the parameter name is not in the model.
- **Breaking.** `Formula`'s inner `HashMap` field is now `pub(crate)` (was
  `pub`). External callers should use `Formula::new()`, `with_terms`,
  `add_terms`, or `Deref` for read access.
- **Breaking.** `FitDiagnostics::param_diagnostics` is now
  `IndexMap<String, ParamDiagnostic>` (was `HashMap`). Same iteration-order
  story as `GamlssModel.models`.
- `GamlssError::Linalg` payload is now uniformly `String` across both
  `openblas` and `pure-rust` backends (previously was typed
  `ndarray_linalg::error::LinalgError` under openblas). Downstream pattern
  matching is now backend-independent.
- `src/linalg.rs` emits a `compile_error!` with an actionable message when
  both `openblas` and `pure-rust` features activate (e.g. through Cargo
  feature unification in workspace builds) instead of a cryptic
  `E0428: name backend defined multiple times`.

### Performance

- IRLS step caches `μ = link⁻¹(η)` on `FittingParameter` so each Fisher-scoring
  step no longer re-runs `inv_link` over every parameter's full η vector.
  Saves K length-n allocations per step (K = number of distribution parameters).
- IRLS step builds `z` (working response) and `w` (floored weights) in a
  single `Zip::for_each` pass instead of four chained `mapv` allocations.
- Pure-rust `linalg::from_dmatrix` uses a single `from_shape_vec` on the
  transposed nalgebra slice instead of a per-element nested loop.
- `sample_from_cholesky` reuses one length-p Gaussian-draw buffer across all
  `n_samples` iterations.
- `math::par_zip3_map` uses an iterator chain (`par_iter().zip().zip()`)
  instead of indexed access, dropping per-element bounds checks on the
  parallel path.

### Internal

- `src/types.rs` (614 lines) split into `src/types/{newtypes,dataset,formula,mod}.rs`.
- `src/distributions/mod.rs` extracted `links.rs` (Link trait + 3 link types
  + their tests).
- `src/splines.rs` items (`sum_to_zero_basis`, `kronecker_product`,
  `row_kronecker_into`, `create_basis_matrix`, `create_penalty_matrix`)
  narrowed from `pub` to `pub(crate)` — they were already in a private module
  but the misleading `pub` keyword is now gone.
- `argmin-math 0.5.1` dependency duplication situation
  (`ndarray 0.16` + `ndarray-linalg 0.17` alongside our `0.17`/`0.18`)
  documented in `Cargo.toml`.
