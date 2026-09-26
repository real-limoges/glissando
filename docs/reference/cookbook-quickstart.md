# Quickstart: fit, predict, diagnose

**Goal:** fit one Gaussian location-scale model end to end, then predict and check residuals, in Rust, Python, and JavaScript.
**Model:** `mu ~ 1 + s(x)`, `sigma ~ 1 + x`, so both the mean and the spread vary with `x`; this is the thing a plain GAM cannot do and GAMLSS can.
**Runnable:** the Rust version is `examples/quickstart.rs` (`cargo run --example quickstart`); the Python version is `examples/python/quickstart.py`; the JavaScript version is `examples/wasm/quickstart.mjs`.

See `cookbook-families.md` to swap the family, and `cookbook-mgcv-migration.md` if you are coming from R.

## Rust

```rust
use glissando::distributions::Gaussian;
use glissando::{DataSet, Formula, GamlssModel, Smooth, Term};
use glissando::ndarray::Array1;

// 1. response and predictors (length n, all f64)
let x: Vec<f64> = (0..120).map(|i| i as f64 * 0.1).collect();
let y = Array1::from_vec(x.iter().map(|&x| x.sin() + 0.1 * x).collect());
let mut data = DataSet::new();
data.insert_column("x", Array1::from_vec(x));

// 2. one additive predictor per parameter
let formula = Formula::new()
    .with_terms("mu", vec![Term::Intercept, Term::smooth(Smooth::ps("x").n_splines(20))])
    .with_terms("sigma", vec![Term::Intercept, Term::linear("x")]);

// 3. fit
let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();
assert!(model.converged());

// 4. predict on the response scale (keyed by parameter name)
let preds = model.predict(&data, &Gaussian::new()).unwrap();
let mu_hat = &preds["mu"];

// 5. diagnose: randomized quantile residuals are the GAMLSS default
let resid = model.quantile_residuals(&Gaussian::new(), &y, Some(42)).unwrap();

// 6. model quality
let aic = model.gaic(&Gaussian::new(), &y, 2.0).unwrap();          // k = 2 is AIC
let bic = model.gaic(&Gaussian::new(), &y, (y.len() as f64).ln()).unwrap();
```

The formula can also be written as a string, which is terser and the only comfortable way to spell smooths in the bindings:

```rust
use glissando::Formula;
let formula = Formula::from_strings([
    ("mu", "y ~ s(x)"),
    ("sigma", "~ x"),
]).unwrap();
```

`Formula::parse("mu", "y ~ s(x)")` builds a single parameter; `from_strings` builds several at once.
The response on the left of `~` is parsed and discarded, because `y` is passed to `fit` separately.

## Python

Install into the current environment with `maturin develop --release`, then:

```python
import numpy as np
import glissando

x = np.arange(120) * 0.1
y = np.sin(x) + 0.1 * x
data = {"x": x}

# string formulas are the ergonomic path, especially for smooths
formula = {"mu": "y ~ s(x)", "sigma": "~ x"}

model = glissando.GamlssModel.fit(data, y, formula, glissando.Gaussian())
assert model.converged()

preds = model.predict(data)                       # {"mu": np.ndarray, "sigma": np.ndarray}
resid = model.quantile_residuals(y, seed=42)
aic = model.gaic(y, 2.0)
bic = model.gaic(y, np.log(len(y)))
```

The `data` argument is a dict of column name to 1-D numpy float array, and `formula` is a dict of parameter name to a formula string (or a list of term tuples).
Predictions come back as a dict of numpy arrays.

## JavaScript / WASM

The published npm package is `glissando` (the committed `pkg/`).
Every argument and return value crosses the boundary as a JSON string.

```js
import { WasmGamlssModel } from "glissando";

const x = Array.from({ length: 120 }, (_, i) => i * 0.1);
const y = x.map((v) => Math.sin(v) + 0.1 * v);

const formula = { mu: "y ~ s(x)", sigma: "~ x" };

// NOTE: fit takes y FIRST, then the data (the reverse of Rust/Python).
const model = WasmGamlssModel.fit(
  JSON.stringify(y),
  JSON.stringify({ x }),
  JSON.stringify(formula),
  "Gaussian",
);

const preds = JSON.parse(model.predict(JSON.stringify({ x })));  // { mu: [...], sigma: [...] }
const resid = JSON.parse(model.quantileResiduals(JSON.stringify(y), 42n));
const aic = model.gaic(JSON.stringify(y), 2.0);
```

Two things to keep straight on this surface: the static `fit` takes `y` before `data` (opposite of the other two surfaces), and the committed `pkg/` is a default (bundler-target) build, so raw Node needs `wasm-pack build --no-default-features --features wasm --target nodejs` first.
See `examples/wasm/README.md` for the runnable setup.
