"""Quickstart: fit, predict, and diagnose a Gaussian location-scale model.

Both the mean (mu) and the spread (sigma) vary with x.

Install the extension into the current environment first:
    maturin develop --release
Then run:
    python examples/python/quickstart.py
"""

import numpy as np
import glissando

n = 120
x = np.arange(n) * 0.1
y = np.sin(x) + 0.1 * x
data = {"x": x}

# One additive predictor per parameter. String formulas are the ergonomic path,
# and the only comfortable way to spell smooths.
formula = {"mu": "y ~ s(x)", "sigma": "~ x"}

model = glissando.GamlssModel.fit(data, y, formula, glissando.Gaussian())
print("converged:", model.converged())

# predict returns a dict of parameter name -> numpy array (response scale).
preds = model.predict(data)
print("mu[:3]    =", preds["mu"][:3])
print("sigma[:3] =", preds["sigma"][:3])

# Randomized quantile residuals are the GAMLSS default residual.
resid = model.quantile_residuals(y, seed=42)
print("mean quantile residual (~0): %.4f" % resid.mean())

# k = 2 is AIC, k = ln(n) is BIC.
print("AIC = %.2f" % model.gaic(y, 2.0))
print("BIC = %.2f" % model.gaic(y, np.log(len(y))))
