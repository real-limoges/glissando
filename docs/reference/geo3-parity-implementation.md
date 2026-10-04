# GEO-3 Implementation Guide: mgcv Parity for the Spatial Smooths

This is a step-by-step walkthrough of GEO-3 from decision `0009`: three new scenarios in the R comparison harness, so the MRF (GEO-1) and thin-plate (GEO-2) smooths are checked against mgcv every night, the same way every other smooth is.
The unit and integration tests in the GEO-1 and GEO-2 guides establish that each smooth is internally correct.
They cannot establish that glissando and mgcv fit the same model to the same data.
That is what this guide adds.

| Scenario | glissando | mgcv reference |
|---|---|---|
| `spatial_tp_gaussian` | Gaussian, `mu ~ tp(x1, x2)` | `gam(y ~ s(x1, x2, bs = "tp", k = 30))` |
| `spatial_tp_sigma` | Gaussian, `mu ~ tp(x1, x2)`, `sigma ~ tp(x1, x2)` | `gam(list(y ~ s(x1, x2, k = 30), ~ s(x1, x2, k = 20)), family = gaulss())` |
| `spatial_mrf_poisson` | Poisson, `mu ~ mrf(region)` on a 10 x 10 lattice with five empty regions | `gam(y ~ s(region, bs = "mrf", xt = list(nb = nb)), family = poisson)` |

**Done criterion**: `./benchmark/run_comparison.sh` produces all three scenarios in `benchmark/output/comparison_summary.json`.
`cargo test --test mgcv_reference -- --ignored` gates their fitted values, sigma (for `spatial_tp_sigma`), EDF, log-likelihood, link-scale standard errors, and the MRF's predictions for the empty regions.
The nightly `r-parity.yml` run passes with them in it.

**Scope**: benchmark code and one test file.
No library code changes.

    benchmark/orchestrate.py           <- three generators, the lattice graph writer, three SCENARIOS entries (appended)
    benchmark/src/bin/compare_fit.rs   <- three fitters, three match arms, an extra_predictions field
    benchmark/fit_mgcv.R               <- three fitters, three dispatch entries, extra_predictions in emit()
    tests/mgcv_reference.rs            <- extra_predictions field and gate; spatial_tp_sigma in the scale whitelist

Every code block below is a **planned target shape**, labeled with the file it will live in.
Line numbers cited for existing code were verified on 2026-09-30.

The MRF scenario depends only on GEO-1 and can land with it.
The two thin-plate scenarios depend on GEO-2.
Each scenario is independent in the harness, so they can land in either order.

------------------------------------------------------------------------

## 1. Layout

A scenario lives in three places that must stay in lockstep (`benchmark/CLAUDE.md`): a data generator in Python, a fitter in Rust, a fitter in R.
The orchestrator runs them, merges the per-rep JSON, and the Rust test gates the merged file.

    orchestrate.py --generate-only
       gen_spatial_*(rng, n)  ->  data_<scenario>_rep<k>.parquet         (Float64 columns)
       write_mrf_graph()      ->  graph_spatial_mrf_poisson.json          (one fixed graph, read by both fitters)
       |
       v  orchestrate.py --rust-binary ... --r-script ...
    compare_fit  --data ... --scenario ... --output glissando_<scenario>_rep<k>.json
    fit_mgcv.R   --data ... --scenario ... --output mgcv_<scenario>_rep<k>.json
       |
       v  merged, unchanged, per rep
    comparison_summary.json   { version: 2, scenarios: [ { name, smooth, reps: [ { rep, glissando, mgcv, gamlss } ] } ] }
       |
       v
    tests/mgcv_reference.rs   per-rep ratios (observed / tolerance), gated on median <= 1 and p90 <= 2

The orchestrator embeds each engine's JSON dict as-is (`orchestrate.py:452-462`), so a new field emitted by both fitters reaches the test with no Python change.

------------------------------------------------------------------------

## 2. What is compared, and why not coefficients

The harness already compares fitted values, EDF, log-likelihood and `se_eta[mu]` for every smooth scenario, and coefficients only for non-smooth ones (`tests/mgcv_reference.rs:391-491`).
That policy is exactly right here, for three reasons specific to these smooths.

- **Identifiability constraints differ.**
  glissando centers the MRF with a coefficient sum-to-zero transform and the thin-plate spline by dropping its constant column; mgcv absorbs a column-mean constraint by QR.
  Same function space, different coordinates.
- **Eigenvectors are defined only up to sign**, and up to rotation within repeated eigenvalues, so two correct thin-plate bases can differ column by column.
- **gaulss models precision, not sigma.**
  Its second linear predictor goes through the `logb` link (`tau = 1/sigma = 0.01 + exp(eta_2)`), so even its sigma coefficients live on a different scale.
  The existing `fit_gaussian_heteroskedastic` comment (`fit_mgcv.R:343-349`) explains the same point.

What is compared beyond the existing metrics is the MRF's prediction for regions with no data, because that is the property that distinguishes an MRF from a random effect.
The `FitResult` schema has no slot for it, so this guide adds one (section 5).

------------------------------------------------------------------------

## 3. Data: `benchmark/orchestrate.py`

### 3a. Generators

Generators take `(rng, n)` and return a dict of float arrays (`orchestrate.py:33-45` for the `Scenario` dataclass).
Add these after `gen_random_effect` (line 228).

**`benchmark/orchestrate.py`**

```python
def _bump(x1, x2):
    return np.exp(-((x1 - 0.5) ** 2 + (x2 - 0.5) ** 2) / 0.05)


def gen_spatial_tp_gaussian(rng, n):
    # Isotropic surface on the unit square: a bump on a gentle tilt.
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    mu = _bump(x1, x2) + 0.5 * x1
    y = rng.normal(mu, scale=0.2)
    return {"y": y, "x1": x1, "x2": x2}


def gen_spatial_tp_sigma(rng, n):
    # Smooth mean, and a scale that rises inside the bump.
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    mu = 1.0 + 0.5 * np.sin(2 * np.pi * x1)
    sigma = np.exp(-1.2 + 1.0 * _bump(x1, x2))
    y = rng.normal(mu, scale=sigma)
    return {"y": y, "x1": x1, "x2": x2}


# The MRF scenario's lattice. Fixed, not drawn: the graph and the empty regions
# are part of the scenario's definition, shared by every rep and both engines.
MRF_ROWS, MRF_COLS = 10, 10
MRF_EMPTY = [12, 37, 55, 81, 99]


def mrf_lattice():
    """Rook adjacency on the MRF_ROWS x MRF_COLS lattice, keyed by region code."""
    def code(r, c):
        return r * MRF_COLS + c
    graph = {}
    for r in range(MRF_ROWS):
        for c in range(MRF_COLS):
            around = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            graph[str(code(r, c))] = [
                str(code(rr, cc)) for rr, cc in around
                if 0 <= rr < MRF_ROWS and 0 <= cc < MRF_COLS
            ]
    return graph


def gen_spatial_mrf_poisson(rng, n):
    # Poisson counts over the lattice; the empty regions get no rows.
    observed = np.array([c for c in range(MRF_ROWS * MRF_COLS) if c not in MRF_EMPTY])
    region = rng.choice(observed, size=n)
    row, col = region // MRF_COLS, region % MRF_COLS
    field = 0.8 * np.sin(row / 3.0) + 0.5 * np.cos(col / 4.0)
    y = rng.poisson(np.exp(1.0 + field))
    return {"y": y.astype(float), "region": region.astype(float)}
```

Region codes are floats in the parquet, like `gen_random_effect`'s `g` (line 224), because `write_parquet` writes every column as Float64 (line 331).

### 3b. The graph file

Both fitters need the same graph and the same list of empty regions.
Writing them once, from the Python that defines the scenario, keeps a single source of truth; rebuilding the lattice in Rust and R as well would mean three copies of the adjacency rule.

**`benchmark/orchestrate.py`** (new function, next to `write_parquet` at line 331)

```python
def write_mrf_graph(output_dir: Path) -> None:
    """The MRF scenario's neighbor graph and empty regions, read by both fitters
    from the directory that holds the parquet files."""
    graph = {"neighbors": mrf_lattice(), "empty": [str(c) for c in MRF_EMPTY]}
    (output_dir / "graph_spatial_mrf_poisson.json").write_text(json.dumps(graph, indent=2))
```

Call it in `main()` right after `args.output_dir.mkdir(...)` (around line 397), unconditionally; it is a few kilobytes and costs nothing.
Keys are `str(int)`, which is also what glissando's `f64::to_string` and R's `as.character(as.integer(...))` produce for whole numbers, so all three sides agree on `"12"`.

### 3c. Registration

**Append** to `SCENARIOS` after `gaussian_cr_smooth` (line 325).
Never insert in the middle: per-scenario seeds are spawned in iteration order, so an insert reshuffles every later scenario's data and invalidates stored results (the comment at lines 283-285).

**`benchmark/orchestrate.py`**

```python
    # GEO-3: spatial smooths (0009). n stays below the 2000-knot cap so both
    # engines build the thin-plate basis from the same knots.
    Scenario("spatial_tp_gaussian",      True,  True,  None,   gen_spatial_tp_gaussian),
    Scenario("spatial_tp_sigma",         True,  True,  None,   gen_spatial_tp_sigma),
    Scenario("spatial_mrf_poisson",      True,  True,  None,   gen_spatial_mrf_poisson),
```

The default `n` is 1000 (`--n-obs`), below the knot cap.
If a future run raises `N_OBS` above 2000, the thin-plate scenarios need `n_obs_override=1000`, because above the cap the two engines subsample different knots and stop being comparable.
Set the override now if you want that guarantee to be structural rather than a comment.

------------------------------------------------------------------------

## 4. Rust fitters: `benchmark/src/bin/compare_fit.rs`

### 4a. The `FitResult` field

**`benchmark/src/bin/compare_fit.rs`** (`struct FitResult`, line 24)

```rust
    /// Extra predictions on data other than the training rows, keyed by name
    /// (e.g. "mu_empty_regions" for the MRF scenario). Empty for most scenarios.
    extra_predictions: HashMap<String, Vec<f64>>,
```

`FitResult` is built as a struct literal in `build_result`, `error_result`, and the unknown-scenario arm (around line 1090).
The compiler lists every site; each gets `extra_predictions: HashMap::new()`.

### 4b. The thin-plate fitters

These follow `fit_tensor_smooth` (line 386) and `fit_gaussian_sigma_smooth` (line 314).

**`benchmark/src/bin/compare_fit.rs`**

```rust
/// Thin-plate smooth on mu; compared against mgcv `s(x1, x2, bs="tp", k=30)`.
fn fit_spatial_tp_gaussian(df: &DataFrame) -> FitResult {
    let start = Instant::now();
    let y = extract_column(df, "y");
    let mut data = DataSet::new();
    data.insert_column("x1", extract_column(df, "x1"));
    data.insert_column("x2", extract_column(df, "x2"));

    let formula = Formula::new()
        .with_terms("mu", vec![Term::Intercept, Term::Smooth(Smooth::tp("x1", "x2").k(30))])
        .with_terms("sigma", vec![Term::Intercept]);

    let family = Gaussian::new();
    match GamlssModel::fit(&data, &y, &formula, &family) {
        Ok(model) => build_result(
            start,
            &model,
            &family,
            &y,
            &data,
            &[("mu", "mu_tp"), ("sigma", "log_sigma")],
            Some("sigma"),
        ),
        Err(e) => error_result(start, e),
    }
}

/// Thin-plate smooths on mu and sigma; compared against mgcv gaulss with
/// s(x1, x2, k=30) and s(x1, x2, k=20).
fn fit_spatial_tp_sigma(df: &DataFrame) -> FitResult {
    let start = Instant::now();
    let y = extract_column(df, "y");
    let mut data = DataSet::new();
    data.insert_column("x1", extract_column(df, "x1"));
    data.insert_column("x2", extract_column(df, "x2"));

    let formula = Formula::new()
        .with_terms("mu", vec![Term::Intercept, Term::Smooth(Smooth::tp("x1", "x2").k(30))])
        .with_terms("sigma", vec![Term::Intercept, Term::Smooth(Smooth::tp("x1", "x2").k(20))]);

    let family = Gaussian::new();
    match GamlssModel::fit(&data, &y, &formula, &family) {
        Ok(model) => build_result(
            start,
            &model,
            &family,
            &y,
            &data,
            &[("mu", "mu_tp"), ("sigma", "log_sigma_tp")],
            Some("sigma"),
        ),
        Err(e) => error_result(start, e),
    }
}
```

Both include an explicit `Term::Intercept`, unlike `fit_tensor_smooth`, whose mu formula is the tensor alone.
mgcv's `s()` always comes with an intercept and a centered smooth; matching that structure keeps the EDF comparison like for like.

### 4c. The MRF fitter

The MRF fitter needs the graph file, which sits next to the parquet file.
The match in `main` passes only `&df` today (line 1062), so this one arm also receives the data path.

**`benchmark/src/bin/compare_fit.rs`**

```rust
/// The MRF scenario's graph, written by orchestrate.py next to the parquet files.
#[derive(serde::Deserialize)]
struct MrfGraph {
    neighbors: HashMap<String, Vec<String>>,
    empty: Vec<String>,
}

fn load_mrf_graph(data_path: &Path) -> MrfGraph {
    let path = data_path
        .parent()
        .expect("data path has a parent directory")
        .join("graph_spatial_mrf_poisson.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    serde_json::from_str(&text).expect("graph file is valid JSON")
}

/// MRF on mu over a 10 x 10 lattice; compared against mgcv
/// `s(region, bs="mrf", xt=list(nb=nb))`, including predictions for the regions
/// with no data.
fn fit_spatial_mrf_poisson(df: &DataFrame, data_path: &Path) -> FitResult {
    let start = Instant::now();
    let graph = load_mrf_graph(data_path);
    let y = extract_column(df, "y");
    let mut data = DataSet::new();
    data.insert_column("region", extract_column(df, "region"));

    let formula = Formula::new().with_terms(
        "mu",
        vec![Term::Intercept, Term::Smooth(Smooth::mrf("region", graph.neighbors))],
    );

    let family = Poisson::new();
    match GamlssModel::fit(&data, &y, &formula, &family) {
        Ok(model) => {
            let mut result = build_result(
                start,
                &model,
                &family,
                &y,
                &data,
                &[("mu", "log_mu_mrf")],
                None,
            );
            let codes: Vec<f64> = graph.empty.iter().map(|c| c.parse().expect("numeric code")).collect();
            let mut empty = DataSet::new();
            empty.insert_column("region", Array1::from_vec(codes));
            match model.predict(&empty, &family) {
                Ok(pred) => {
                    result.extra_predictions.insert("mu_empty_regions".into(), pred["mu"].to_vec());
                }
                Err(e) => result.error = Some(format!("empty-region prediction failed: {e}")),
            }
            result
        }
        Err(e) => error_result(start, e),
    }
}
```

Add `use std::path::Path;`; `serde_json` is already a dependency of the benchmark crate.

**`benchmark/src/bin/compare_fit.rs`** (the match in `main`, line 1062; append before `other =>`)

```rust
        "spatial_tp_gaussian" => fit_spatial_tp_gaussian(&df),
        "spatial_tp_sigma" => fit_spatial_tp_sigma(&df),
        "spatial_mrf_poisson" => fit_spatial_mrf_poisson(&df, &data_path),
```

------------------------------------------------------------------------

## 5. R fitters: `benchmark/fit_mgcv.R`

### 5a. `emit()` gains `extra_predictions`

**`benchmark/fit_mgcv.R`** (`emit`, line 68)

```r
emit <- function(path, m, coefficients, edf, fit_time_ms,
                 fitted_sigma = list(), extra_predictions = NULL, error_msg = NA) {
  # ... unchanged ...
  result <- list(
    # ... unchanged fields ...
    extra_predictions = extra_predictions,
    error          = error_msg
  )
  write_json(result, path, auto_unbox = TRUE, pretty = TRUE, na = "null")
}
```

The default is `NULL`, not `list()`.
`jsonlite` writes a `NULL` list element as `{}` and an empty `list()` as `[]`, and the Rust side deserializes this field into a `HashMap`, which accepts `{}` but rejects `[]`.
Verify that against the installed `jsonlite` the first time the scenario runs; if it writes `[]`, the gate in section 6 will report a deserialize error rather than silently passing.

### 5b. The thin-plate fitters

**`benchmark/fit_mgcv.R`** (after `fit_random_effect`)

```r
# ─── Spatial (GEO-3, decision 0009) ──────────────────────────────────────────

fit_spatial_tp_gaussian <- function(df, output) {
  start <- Sys.time()
  m <- gam(y ~ s(x1, x2, bs = "tp", k = 30), data = df,
           family = gaussian(), method = "REML")
  emit(
    output, m,
    coefficients = list(
      mu_tp     = unname(coef(m)),
      log_sigma = list(0.5 * log(m$sig2))
    ),
    edf = list(mu = sum(m$edf), sigma = 1.0),
    fit_time_ms = elapsed_ms(start)
  )
}

# gaulss: column 2 of fitted() is precision 1/sigma (see fit_gaussian_heteroskedastic).
fit_spatial_tp_sigma <- function(df, output) {
  start <- Sys.time()
  m <- gam(list(y ~ s(x1, x2, bs = "tp", k = 30), ~ s(x1, x2, bs = "tp", k = 20)),
           data = df, family = gaulss(), method = "REML")
  fv <- fitted(m)

  # Split EDF by linear predictor. lpi lists each predictor's coefficient
  # columns, so the mu intercept stays out of sigma's EDF.
  lpi <- attr(predict(m, type = "lpmatrix"), "lpi")

  se_pred <- tryCatch(predict(m, type = "link", se.fit = TRUE), error = function(e) NULL)
  se_eta_out <- if (!is.null(se_pred) && is.matrix(se_pred$se.fit)) {
    list(mu = as.list(unname(se_pred$se.fit[, 1])))
  } else {
    list()
  }

  result <- list(
    converged      = gam_converged(m),
    iterations     = if (!is.null(m$outer.info$iter)) as.integer(m$outer.info$iter) else 0L,
    fit_time_ms    = elapsed_ms(start),
    coefficients   = list(mu_tp = unname(coef(m))[lpi[[1]]]),
    fitted_mu      = as.list(unname(fv[, 1])),
    fitted_sigma   = as.list(unname(1.0 / fv[, 2])),
    edf            = list(mu = sum(m$edf[lpi[[1]]]), sigma = sum(m$edf[lpi[[2]]])),
    log_likelihood = as.numeric(stats::logLik(m)),
    aic            = AIC(m),
    sp             = sp_list(m),
    se_eta         = se_eta_out,
    extra_predictions = NULL,
    error          = NA
  )
  write_json(result, output, auto_unbox = TRUE, pretty = TRUE, na = "null")
}
```

`se_eta` carries mu only.
glissando's sigma link is log; gaulss's is `logb` on precision, so link-scale sigma standard errors are not comparable, and `tests/mgcv_reference.rs` gates only `se_eta[mu]` anyway.

### 5c. The MRF fitter

**`benchmark/fit_mgcv.R`** (continued)

```r
fit_spatial_mrf_poisson <- function(df, output) {
  graph <- jsonlite::fromJSON(
    file.path(dirname(opts$data), "graph_spatial_mrf_poisson.json"),
    simplifyVector = FALSE
  )
  # Levels in glissando's order (numeric), and EVERY region, including the
  # empty ones; mgcv gives a coefficient to each level of the factor.
  levels_all <- as.character(sort(as.integer(names(graph$neighbors))))
  nb <- lapply(levels_all, function(r) match(unlist(graph$neighbors[[r]]), levels_all))
  names(nb) <- levels_all
  df$region <- factor(as.character(as.integer(df$region)), levels = levels_all)

  start <- Sys.time()
  m <- gam(y ~ s(region, bs = "mrf", xt = list(nb = nb)), data = df,
           family = poisson(link = "log"), method = "REML")

  empty <- data.frame(region = factor(unlist(graph$empty), levels = levels_all))
  emit(
    output, m,
    coefficients = list(log_mu_mrf = unname(coef(m))),
    edf = list(mu = sum(m$edf)),
    fit_time_ms = elapsed_ms(start),
    extra_predictions = list(
      mu_empty_regions = as.list(unname(predict(m, empty, type = "response")))
    )
  )
}
```

`opts` is the script-level options object (`fit_mgcv.R:37-42`), so the fitter can read `opts$data` without a signature change.
`nb` uses integer indices into `levels_all`, which is the form mgcv's `mrf` documentation describes; names alone are also accepted by recent mgcv, but indices are unambiguous.

**`benchmark/fit_mgcv.R`** (`dispatch`, line 550; append)

```r
  spatial_tp_gaussian       = fit_spatial_tp_gaussian,
  spatial_tp_sigma          = fit_spatial_tp_sigma,
  spatial_mrf_poisson       = fit_spatial_mrf_poisson
```

(Add the comma after the current last entry, `b2_weighted_studentt`.)

------------------------------------------------------------------------

## 6. The gate: `tests/mgcv_reference.rs`

### 6a. Deserialize the new field

**`tests/mgcv_reference.rs`** (`struct FitResult`, line 74)

```rust
    /// Predictions on data other than the training rows (e.g. the MRF's empty
    /// regions), keyed by name. Absent from most scenarios.
    #[serde(default)]
    extra_predictions: HashMap<String, Vec<f64>>,
```

`serde(default)` keeps every existing summary file loadable.

### 6b. Gate the extra predictions

**`tests/mgcv_reference.rs`** (in `record_mgcv`, after the `fitted_sigma` block around line 430)

```rust
    // Extra predictions (e.g. MRF empty regions): same relative tolerance as
    // fitted_mu, one metric per key so a failure names what drifted.
    for (key, g_pred) in &g.extra_predictions {
        let Some(m_pred) = m.extra_predictions.get(key) else {
            failures.push(format!("{name} rep {}: mgcv has no extra prediction '{key}'", rep.rep));
            continue;
        };
        if g_pred.len() != m_pred.len() {
            failures.push(format!("{name} rep {}: '{key}' length mismatch", rep.rep));
            continue;
        }
        let (max_rel, _) = fitted_drift(g_pred, m_pred);
        acc.push(key, max_rel / rel_tol);
    }
```

A key glissando emits that mgcv lacks is a failure, not a skip, so a broken R fitter cannot pass by omission.

### 6c. Gate sigma for `spatial_tp_sigma`

`is_scale_smooth_scenario` (line 384) is a hard-coded whitelist; without an entry, `fitted_sigma` is never compared.

**`tests/mgcv_reference.rs`**

```rust
fn is_scale_smooth_scenario(name: &str) -> bool {
    name == "gaussian_sigma_smooth"
        || name == "gamma_sigma_smooth"
        || name == "gaussian_heteroskedastic"
        || name == "spatial_tp_sigma"
}
```

### 6d. Tolerances (Q-GEO-4)

Settled the way `0009` leaned: tolerances are set from the first benchmark run, not pinned in advance.
Start the three scenarios on the existing smooth defaults:

| Metric | Tolerance | Where |
|---|---|---|
| `fitted_mu` and `mu_empty_regions` | 0.10 relative | `fitted_mu_rel_tol(true)`, line 245 |
| `fitted_sigma` | 0.05 relative | line 430 |
| EDF | `max(0.2 * ref, 0.5)` | `edf_tol`, line 266 |
| log-likelihood | 1e-2 per observation | |
| `se_eta[mu]` | 0.10 relative | |

Do not add the spatial scenarios to the `is_tensor` 0.15 band (line 402) preemptively.
That band exists because of a measured ML-versus-REML convention gap on a 63-coefficient tensor; it is not a general allowance for 2D smooths.
If the first nightly shows a systematic gap, record the observed median and p90 ratios and the reason in `0009`, then widen only the metric that needs it.
Revisable: if the first run is clean, these defaults are the answer to Q-GEO-4.

------------------------------------------------------------------------

## 7. Running it

```bash
# Build glissando's comparison binary and run only the new scenarios.
cargo build -p glissando_benchmark --release
python benchmark/orchestrate.py --output-dir benchmark/output --generate-only \
    --scenarios spatial_tp_gaussian spatial_tp_sigma spatial_mrf_poisson
python benchmark/orchestrate.py --output-dir benchmark/output \
    --rust-binary target/release/compare_fit --r-script benchmark/fit_mgcv.R \
    --scenarios spatial_tp_gaussian spatial_tp_sigma spatial_mrf_poisson --reps 5

# Gate.
cargo test --test mgcv_reference -- --ignored --nocapture
```

A single fit, for debugging one engine against one file:

```bash
cargo run -p glissando_benchmark --release --bin compare_fit -- \
    --data benchmark/output/data_spatial_mrf_poisson_rep0.parquet \
    --scenario spatial_mrf_poisson --output /tmp/g.json
Rscript benchmark/fit_mgcv.R \
    --data benchmark/output/data_spatial_mrf_poisson_rep0.parquet \
    --scenario spatial_mrf_poisson --output /tmp/m.json
```

`--scenarios` limits a run to the named scenarios (`orchestrate.py:392`), which keeps iteration on three scenarios to minutes rather than the full suite.
The full suite is `./benchmark/run_comparison.sh`, which needs Python (numpy, polars) and R (mgcv, arrow, jsonlite).

------------------------------------------------------------------------

## 8. glissando vs. mgcv: what should and should not match

- **Fitted values, EDF, log-likelihood**: should match within the smooth tolerances.
  Both engines select one smoothing parameter per penalty by REML, over the same function space and the same penalty up to a constant factor.
- **The MRF's empty-region predictions**: should match.
  Both penalties are the graph Laplacian, so both put each empty region at the mean of its neighbors on the link scale.
- **Coefficients**: not compared, for the reasons in section 2.
- **Smoothing parameters**: recorded, never gated (`LAMBDA_SPREAD_NOTE` at `tests/mgcv_reference.rs:149`).
  glissando rescales the thin-plate penalty the way mgcv does, but the MRF penalty is not rescaled, so raw lambda values are not commensurable.
- **The knot subsample**: never exercised, because `n` stays below the cap.

------------------------------------------------------------------------

## 9. Decisions and gotchas

### Three places, in lockstep, appended

Every scenario is a Python generator, a Rust fitter, and an R fitter.
Missing the Rust arm gives `Unknown scenario`; missing the R entry gives `scenario '...' not supported by this script`.
Missing the append-only rule gives no error at all, just reshuffled data for every scenario after the insert.

### One graph file, written by the scenario's owner

The lattice and the empty regions are defined once, in `orchestrate.py`, and read by both fitters.
If the scenario ever needs a second graph, write a second file; do not teach either fitter to rebuild one.

### The R factor must carry the empty regions

`factor(df$region)` builds levels from the data, which drops the five empty regions, and mgcv then has nothing to predict them with.
`levels = levels_all` is what makes the comparison possible.

### Do not copy the sigma-EDF sum from `gaussian_sigma_smooth`

That fitter emits `sigma = sum(m$edf)` (`fit_mgcv.R:415`), which includes the mu intercept, so its sigma EDF runs about one higher than glissando's.
`fit_spatial_tp_sigma` splits by `lpi` instead.
The existing fitter is not changed here; fixing it is a separate change that moves that scenario's EDF ratios, and it deserves its own look at the stored results.

### The summary file is not committed

`.gitignore` re-includes `benchmark/output/comparison_summary.json`, but no copy is tracked, so the per-commit `glissando_matches_committed_mgcv_fixture` test always skips.
In practice parity is enforced by the nightly `r-parity.yml` alone.
Run the commands in section 7 locally before calling a GEO scenario done.

### Thin-plate fits are slower

Each thin-plate fit decomposes a 1000 x 1000 kernel matrix at `n = 1000`, in both engines.
At 25 reps across two scenarios this is noticeable in the nightly but not prohibitive.
If it becomes a problem, `reps_override` on the `Scenario` (as `gaussian_large` uses) caps the cost without touching the seeds.

------------------------------------------------------------------------

## 10. Exit checklist

- [ ] `orchestrate.py`: three generators, `mrf_lattice`, `write_mrf_graph` called from `main`, three `SCENARIOS` entries **appended** after `gaussian_cr_smooth`.
- [ ] `compare_fit.rs`: `extra_predictions` on `FitResult` at every construction site; three fitters; three match arms; the MRF arm receives the data path.
- [ ] `fit_mgcv.R`: `emit()` takes `extra_predictions = NULL`; three fitters; three `dispatch` entries; the MRF factor carries every region as a level.
- [ ] `tests/mgcv_reference.rs`: `extra_predictions` deserializes with `serde(default)`; the gate fails on a missing mgcv key; `spatial_tp_sigma` is in `is_scale_smooth_scenario`.
- [ ] A local five-rep run of the three scenarios passes `cargo test --test mgcv_reference -- --ignored`.
- [ ] The observed median and p90 ratios for each metric are recorded in `0009` as the answer to Q-GEO-4, with any widened tolerance and its reason.
- [ ] The nightly `r-parity.yml` passes with the scenarios included.

When these hold, GEO-3 is done, and with GEO-1 and GEO-2 it closes decision `0009`: move its status to `accepted`, record the answers to Q-GEO-1 through Q-GEO-4, and update the roadmap in `0000`.
