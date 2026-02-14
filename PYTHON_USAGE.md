# Python Bindings for gamlss_rs

Minimal PyO3 bindings for the gamlss_rs library, exposing core functionality to Python.

## Building

### Requirements
- Python 3.7+
- Rust toolchain
- OpenBLAS (or use `pure-rust` feature)

### Using maturin (recommended)

Install maturin:
```bash
pip install maturin
```

Development build:
```bash
maturin develop --features python
```

Release build:
```bash
maturin build --release --features python
```

This creates a wheel in `target/wheels/` that you can install with pip.

### Alternative: Using setuptools-rust

Create a `setup.py` file and build with:
```bash
python setup.py develop
```

## Python API

### Distributions

All 7 distributions are exposed:

```python
from gamlss_rs import (
    Gaussian,
    Poisson,
    Binomial,
    Gamma,
    NegativeBinomial,
    Beta,
    StudentT
)

# Zero-sized distributions (no parameters)
gaussian = Gaussian()
poisson = Poisson()
gamma = Gamma()
neg_binom = NegativeBinomial()
beta = Beta()
student_t = StudentT()

# Binomial requires trials
binomial = Binomial(n_trials=[10.0, 10.0, 10.0])  # List of trials per observation
```

### Model Fitting

```python
from gamlss_rs import GamlssModel
import numpy as np

# Prepare data
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([2.1, 4.0, 5.9, 8.1, 10.0])

# Fit model
model = GamlssModel.fit(
    data={'x': x},                    # Dict of column_name -> array
    y=y,                               # Response variable
    formula={                          # Dict of parameter -> terms
        'mu': [                        # Model for mean parameter
            ('intercept',),            # Intercept term
            ('linear', 'x')            # Linear effect of x
        ],
        'sigma': [('intercept',)]      # Model for variance parameter
    },
    family=Gaussian()                  # Distribution family
)

# Check convergence
print(model.converged())  # True or False

# Predict on new data
new_data = {'x': np.array([6.0, 7.0, 8.0])}
predictions = model.predict(new_data=new_data)
# Returns: {'mu': array([...]), 'sigma': array([...])}
```

### Formula Specification

Each term is a tuple with the following formats:

#### Intercept
```python
('intercept',)
```

#### Linear effect
```python
('linear', 'column_name')
```

#### Smooth (nonlinear) effect
```python
# With defaults (n_splines=10, degree=3, penalty_order=2)
('smooth', 'column_name')

# With custom parameters
('smooth', 'column_name', {
    'n_splines': 15,
    'degree': 3,
    'penalty_order': 2
})
```

#### Random effect
```python
('random', 'group_column')
```

Note: Random effect groups should be encoded as numeric values (0.0, 1.0, 2.0, etc.)

### Complete Examples

See `python_example.py` for working examples including:
- Linear regression
- Smooth effects
- Poisson regression for count data
- Binomial regression
- Random effects models

## API Reference

### `GamlssModel`

#### `GamlssModel.fit(data, y, formula, family)`

Static method to fit a GAMLSS model.

**Parameters:**
- `data` (dict): Dictionary mapping column names to 1D numpy arrays
- `y` (array): Response variable (1D numpy array)
- `formula` (dict): Dictionary mapping parameter names to lists of term tuples
- `family` (Distribution): Distribution object (Gaussian(), Poisson(), etc.)

**Returns:** Fitted `GamlssModel` instance

**Raises:** RuntimeError if fitting fails

#### `model.predict(new_data)`

Predict fitted values for new data.

**Parameters:**
- `new_data` (dict): Dictionary mapping column names to 1D numpy arrays

**Returns:** Dictionary mapping parameter names to predicted values (numpy arrays)

**Raises:** RuntimeError if prediction fails

#### `model.converged()`

Check if the model converged.

**Returns:** `bool` - True if converged, False otherwise

## Implementation Details

### Minimal Exposure
Only the core functionality is exposed:
- Model fitting with default configuration (no FitConfig customization)
- Prediction
- Convergence check

### Data Conversion
- Python dicts → Rust `DataSet` (HashMap of column names to Array1)
- Formula dicts → Rust `Formula` (HashMap of parameter names to Vec<Term>)
- Numpy arrays → ndarray `Array1<f64>`
- Results automatically converted back to Python/numpy

### Distribution Dispatch
Uses an internal `FamilyType` enum to dispatch to the appropriate Rust distribution implementation at runtime.

### Memory Management
All data is copied during conversion. The Rust model owns its data, and Python owns the returned predictions.

## Limitations

- No FitConfig customization (uses defaults: max_iter=100, tol=1e-6)
- No access to:
  - Effective degrees of freedom
  - Coefficient covariance matrices
  - Detailed diagnostics
  - Model serialization
  - Posterior sampling

These features can be added if needed, but the goal is minimal exposure.

## Error Handling

- Input validation errors → `ValueError`
- Fitting/prediction failures → `RuntimeError`

Error messages include details from the Rust error.

## Performance Notes

- Uses default feature flags (openblas + parallel)
- For WASM compatibility, use `pure-rust` feature instead
- No Python GIL release during long computations (single-threaded from Python's perspective)
- For large-scale parallel processing, consider using the Rust API directly

## Future Extensions

Possible additions if needed:
- FitConfig exposure (max_iter, tolerance)
- Model diagnostics (AIC, BIC, residuals)
- Coefficient extraction
- Standard error prediction
- Model serialization
- More flexible formula parsing (string formulas)
- Tensor product smooths
