# glissando Documentation

Reference material for glissando, a Rust GAMLSS engine (Rigby-Stasinopoulos backfitting) shipped as a crate, a WASM/npm package, and a Python extension.

Start with the cookbook for copy-pasteable examples; use `rust-api.md` to look up a specific call, and `mathematics.md` for the exact likelihood, score, or expected information behind a family.
Pages written for this project live under `internal/`; `external/` is for cited third-party sources and has none yet.

| File | What It Is |
|---|---|
| [`cookbook-overview.md`](internal/cookbook-overview.md) | Landing page: scope, the three surfaces (Rust / WASM-JS / Python), and the core API. |
| [`cookbook-quickstart.md`](internal/cookbook-quickstart.md) | One model end to end (fit, predict, diagnose) on all three surfaces. |
| [`cookbook-families.md`](internal/cookbook-families.md) | Distribution gallery: how to construct and fit each of the 12 base families, the structural wrappers, and mixtures. |
| [`cookbook-mgcv-migration.md`](internal/cookbook-mgcv-migration.md) | R `mgcv` / `gamlss` to glissando migration: family and basis mapping, common workflows, and the current gaps. |
| [`rust-api.md`](internal/rust-api.md) | Rust API reference: terms and formula strings, `FitConfig`, results, prediction, diagnostics, model selection, and errors. |
| [`bindings.md`](internal/bindings.md) | Model serialization, the `glissando::json` facade for embedding behind another FFI, and the Python API. |
| [`mathematics.md`](internal/mathematics.md) | Algorithm derivations: per-family log-likelihoods, score functions, expected information, CDF/quantile, and the backfitting and penalty machinery. Built to `mathematics.pdf` by `build-math-pdf.sh`. |
