# Python examples

Runnable Python versions of the cookbook (`docs/cookbook/`).

## Setup

The `glissando` module is a compiled extension built with [maturin](https://github.com/PyO3/maturin).
Build and install it into the current virtual environment:

```bash
maturin develop --release
```

`numpy` is the only runtime dependency of these scripts.

## Run

```bash
python examples/python/quickstart.py    # fit, predict, diagnose one model
python examples/python/families.py      # a fit from each distribution group
```

## The shape of the API

- `data` and `new_data` are a dict of column name to a 1-D numpy float array.
- `y` is a 1-D numpy float array, passed separately from `data`.
- `formula` is a dict of parameter name to a formula string (`{"mu": "y ~ s(x)", "sigma": "~ 1"}`) or a list of term tuples (`{"mu": [("intercept",), ("linear", "x")]}`).
- `predict` returns a dict of parameter name to numpy array.

One trap worth knowing: `glissando.Binomial(...)` takes a *list* of per-row trial counts, not a scalar integer.
