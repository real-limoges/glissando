//! Residual computations: Pearson and response residuals.

use crate::distributions::{Distribution, MIN_POSITIVE};
use crate::GamlssError;
use ndarray::Array1;
use std::collections::HashMap;

/// Computes Pearson residuals via the family's marginal moments:
/// `r_i = (y_i − E[Y_i]) / √Var(Y_i)`.
///
/// Variance is floored at `MIN_POSITIVE = 1e-10` before the square-root to
/// keep residuals finite when the fitted variance is degenerate.
pub fn pearson_residuals<D: Distribution + ?Sized>(
    family: &D,
    y: &Array1<f64>,
    params: &HashMap<&str, &Array1<f64>>,
) -> Result<Array1<f64>, GamlssError> {
    let e = family.expected_value(params)?;
    let v = family.variance(params)?;
    let sd = v.mapv(|vi| vi.max(MIN_POSITIVE).sqrt());
    Ok((y - &e) / &sd)
}

pub fn response_residuals(y: &Array1<f64>, expected: &Array1<f64>) -> Array1<f64> {
    y - expected
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::Gaussian;
    use ndarray::array;

    // --- response_residuals ---

    #[test]
    fn response_residuals_subtracts() {
        let y = array![3.0, 5.0, 7.0];
        let e = array![1.0, 2.0, 4.0];
        let r = response_residuals(&y, &e);
        assert_eq!(r, array![2.0, 3.0, 3.0]);
    }

    // --- pearson_residuals composes E[Y] and Var(Y) ---

    #[test]
    fn pearson_residuals_gaussian_via_trait() {
        // For Gaussian, E[Y]=mu, Var(Y)=sigma^2, so Pearson = (y-mu)/sigma.
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.5, 2.0, 2.5];
        let sigma = array![0.5, 0.5, 0.5];
        let params: HashMap<&str, &Array1<f64>> = HashMap::from([("mu", &mu), ("sigma", &sigma)]);
        let r = pearson_residuals(&Gaussian, &y, &params).unwrap();
        assert!((r[0] - (-1.0)).abs() < 1e-10);
        assert!((r[1] - 0.0).abs() < 1e-10);
        assert!((r[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn pearson_residuals_handles_zero_variance_gracefully() {
        // sigma -> 0 would divide by zero; MIN_POSITIVE clamp must keep residuals finite.
        let y = array![0.0];
        let mu = array![0.0];
        let sigma = array![0.0];
        let params: HashMap<&str, &Array1<f64>> = HashMap::from([("mu", &mu), ("sigma", &sigma)]);
        let r = pearson_residuals(&Gaussian, &y, &params).unwrap();
        assert!(r.iter().all(|v| v.is_finite()));
    }
}
