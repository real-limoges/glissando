"""A representative fit from each distribution group, mirroring examples/families.rs.

    maturin develop --release
    python examples/python/families.py
"""

import numpy as np
import glissando

rng = np.random.default_rng(42)
n = 200
x = np.linspace(0.0, 6.0, n)
data = {"x": x}

# Gaussian: symmetric continuous.
y = np.sin(x) + 0.3 * (rng.random(n) - 0.5)
m = glissando.GamlssModel.fit(data, y, {"mu": "y ~ s(x)", "sigma": "~ 1"}, glissando.Gaussian())
print("Gaussian:    converged=%s  AIC=%.1f" % (m.converged(), m.gaic(y, 2.0)))

# Negative Binomial: overdispersed counts (non-negative integers).
counts = np.round(np.exp(0.4 + 0.35 * x) * (0.6 + 0.8 * rng.random(n)))
m = glissando.GamlssModel.fit(data, counts, {"mu": "y ~ x", "sigma": "~ 1"}, glissando.NegativeBinomial())
print("NegBinomial: converged=%s" % m.converged())

# Weibull: strictly positive continuous.
pos = (0.3 + 0.5 * x) * (0.5 + rng.random(n)) + 0.05
m = glissando.GamlssModel.fit(data, pos, {"mu": "y ~ x", "sigma": "~ 1"}, glissando.Weibull())
print("Weibull:     converged=%s" % m.converged())

# BCCG (Cole-Green): the LMS centile family; three parameters, so name nu.
formula = {"mu": "y ~ s(x)", "sigma": "~ 1", "nu": "~ 1"}
m = glissando.GamlssModel.fit(data, pos, formula, glissando.BCCG())
curves = m.centiles(data, [3.0, 50.0, 97.0])
print("BCCG:        converged=%s  %d centile curves" % (m.converged(), len(curves)))
