# Serialization and Bindings

glissando ships three surfaces (the typed Rust API, a WASM/npm package, and a Python extension) that share one JSON wire format.
This page covers persisting models, the JSON facade for embedding the crate behind another boundary, and the Python API.
The WASM package is documented in [`pkg/README.md`](../../pkg/README.md).

## Saving and Loading Models

With the `serialization` feature, a fitted model round-trips through JSON:

```rust
let json = model.to_json(&Gaussian::new())?;
let (model, descriptor) = GamlssModel::from_json(&json)?;
let family = descriptor.build()?;   // Box<dyn Distribution>
```

The JSON carries a `FamilyDescriptor`, so stateful families (`Binomial`, `Ocat`) and the structural wrappers rebuild correctly, not just families named by a string.
`MixtureModel::to_json` works the same way.

## Embedding Behind Your Own FFI

To put glissando behind a different boundary (a [Rustler](https://github.com/rusterlium/rustler) NIF, a C ABI, a JSON service), use `glissando::json`, the same string-in, string-out layer the WASM bindings use:

```rust
use glissando::json;

let y       = "[1.2, 2.1, 2.9, 4.2, 4.8]";
let data    = r#"{"x": [1.0, 2.0, 3.0, 4.0, 5.0]}"#;
let formula = r#"{"mu": "y ~ x", "sigma": "~ 1"}"#;

// Trailing arguments: optional config JSON, optional prior-weights JSON.
let (model, family) = json::fit(y, data, formula, "Gaussian", None, None)?;

let preds       = json::predict(&model, family.as_ref(), r#"{"x": [6.0, 7.0]}"#)?;
let with_se     = json::predict_with_se(&model, family.as_ref(), r#"{"x": [6.0]}"#)?;
let samples     = json::predict_samples(&model, family.as_ref(), r#"{"x": [6.0]}"#, 500, Some(42))?;
let diagnostics = json::diagnostics(&model)?;

let blob = model.to_json(family.as_ref())?;
let (restored, family) = json::load(&blob)?;
```

A formula can also be the structured form, a list of serialized `Term`s per parameter (`{"mu": [{"Intercept": null}, {"Linear": {"col_name": "x"}}]}`).
The config JSON takes the `FitConfig` fields, with links as `{"links": {"mu": "probit"}}`.
The parsing and serialization helpers (`parse_data`, `parse_formula`, `serialize_predictions`, and others) are public for a custom flow.

`glissando::distributions::from_name` resolves the stateless families by name: `Gaussian`, `Poisson`, `StudentT`, `Gamma`, `NegativeBinomial`, `Beta`, `Weibull`, `BCCG`, `BCT`, `BCPE`.
`Binomial`, `Ocat`, and the structural wrappers carry state a name cannot express, so the string facade and WASM cannot build them; use Rust or Python.

## Python

Build with [maturin](https://github.com/PyO3/maturin): `maturin develop --release` installs into the current virtual environment, and `maturin build --release` writes a wheel under `target/wheels/`.
Runnable scripts are in [`examples/python/`](../../examples/python/).

```python
import numpy as np
from glissando import GamlssModel, Gaussian

y = np.array([2.1, 4.0, 5.9, 8.1, 10.0])
data = {"x": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
new = {"x": np.array([6.0, 7.0])}

# A formula string per parameter, or term tuples:
# {"mu": [("intercept",), ("linear", "x"), ("smooth", "x", {"n_splines": 10}), ("random", "g")]}
formula = {"mu": "y ~ x", "sigma": "~ 1"}

model = GamlssModel.fit(data, y, formula, Gaussian())
model = GamlssModel.fit_with_config(data, y, formula, Gaussian(), {
    "max_iterations": 300,
    "criterion": "reml",            # or "gcv", "fellner_schall"
    "links": {"mu": "identity"},
    "na_action": "drop_rows",       # or "fail"
})

model.predict(new)                                  # {"mu": ndarray, "sigma": ndarray}
model.predict_with_se(new)["mu"]                    # {"fitted", "eta", "se_eta"}
model.predict_samples(new, n_samples=500, seed=42)  # {"mu": [ndarray, ...], ...}
model.coefficients("mu"); model.fitted_values("mu")

model.quantile_residuals(y, seed=42)
model.centiles(data, [10.0, 50.0, 90.0])            # {"C10": ..., "C50": ..., "C90": ...}
model.quantile_prediction(data, np.full(len(y), 0.95))

model.gaic(y, 2.0)
GamlssModel.ic_table([("null", m0), ("x", m1)], y, 2.0)
m0.lr_test(m1, y)                                   # {"lr_stat", "df", "p_value"}
GamlssModel.step_gaic(data, y, Gaussian(), start, scope, np.log(len(y)), "forward")
```

Python has every family the Rust API has, including the stateful ones:

- `Binomial(n_trials)`, where `n_trials` is a list of per-row trial counts.
- `Ocat(n_categories)`, with `model.predict_class_probabilities(new_data)`.
- `Censored(base, status, upper=None)` with status strings such as `"event"` and `"right"`, `Truncated(base, lower, upper)`, and `Hurdle(base)`.

Finite mixtures are Rust-only.
