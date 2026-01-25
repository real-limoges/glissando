use ndarray::Array1;
use rayon::prelude::*;
use statrs::function::gamma::digamma as statrs_digamma;

/// Threshold for using parallel computation (below this, sequential is faster)
const PARALLEL_THRESHOLD: usize = 10_000;

/// Scalar digamma function (re-exported from statrs for accuracy).
#[inline]
#[cfg_attr(not(test), allow(dead_code))]
pub fn digamma(x: f64) -> f64 {
    statrs_digamma(x)
}

/// Trigamma function: psi'(x) = d^2/dx^2 log(Gamma(x))
/// Uses recurrence relation for x < 10, asymptotic expansion for x >= 10.
/// See docs/mathematics.md for derivation and test values.
#[inline]
pub fn trigamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }

    let mut x_shifted = x;
    let mut result = 0.0;

    // Recurrence: psi'(x) = psi'(x+1) + 1/x^2
    while x_shifted < 10.0 {
        result += 1.0 / (x_shifted * x_shifted);
        x_shifted += 1.0;
    }

    // Asymptotic expansion from Abramowitz & Stegun 6.4.11
    let inv_x = 1.0 / x_shifted;
    let inv_x2 = inv_x * inv_x;
    let inv_x3 = inv_x2 * inv_x;
    let inv_x5 = inv_x3 * inv_x2;
    let inv_x7 = inv_x5 * inv_x2;

    let expansion = inv_x + inv_x2 / 2.0 + inv_x3 / 6.0 - inv_x5 / 30.0 + inv_x7 / 42.0;

    expansion + result
}

/// Batch digamma function optimized for vectorization.
///
/// Computes digamma for all elements in the input array.
/// Uses parallel computation for large arrays (n >= 10,000).
#[inline]
pub fn digamma_batch(x: &Array1<f64>) -> Array1<f64> {
    let n = x.len();
    if n < PARALLEL_THRESHOLD {
        x.mapv(statrs_digamma)
    } else {
        let result: Vec<f64> = x
            .as_slice()
            .expect("input array not contiguous")
            .par_iter()
            .map(|&v| statrs_digamma(v))
            .collect();
        Array1::from_vec(result)
    }
}

/// Batch trigamma function optimized for vectorization.
///
/// Computes trigamma for all elements in the input array.
/// Uses parallel computation for large arrays (n >= 10,000).
#[inline]
pub fn trigamma_batch(x: &Array1<f64>) -> Array1<f64> {
    let n = x.len();
    if n < PARALLEL_THRESHOLD {
        x.mapv(trigamma)
    } else {
        let result: Vec<f64> = x
            .as_slice()
            .expect("input array not contiguous")
            .par_iter()
            .map(|&v| trigamma(v))
            .collect();
        Array1::from_vec(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_digamma() {
        // Known values from Mathematica/WolframAlpha
        assert!((digamma(1.0) - (-0.5772156649015329)).abs() < 1e-10);
        assert!((digamma(2.0) - 0.4227843350984671).abs() < 1e-10);
        assert!((digamma(10.0) - 2.2517525890667214).abs() < 1e-10);
    }

    #[test]
    fn test_trigamma() {
        assert!((trigamma(1.0) - 1.6449340668482264).abs() < 1e-10);
        assert!((trigamma(2.0) - 0.6449340668482264).abs() < 1e-10);
        assert!((trigamma(10.0) - 0.10516633568168575).abs() < 1e-10);
    }

    #[test]
    fn test_digamma_batch() {
        let x = Array1::from_vec(vec![1.0, 2.0, 5.0, 10.0, 0.5]);
        let result = digamma_batch(&x);

        for i in 0..x.len() {
            let expected = digamma(x[i]);
            assert!(
                (result[i] - expected).abs() < 1e-10,
                "digamma_batch mismatch at {}: got {}, expected {}",
                x[i],
                result[i],
                expected
            );
        }
    }

    #[test]
    fn test_trigamma_batch() {
        let x = Array1::from_vec(vec![1.0, 2.0, 5.0, 10.0, 0.5]);
        let result = trigamma_batch(&x);

        for i in 0..x.len() {
            let expected = trigamma(x[i]);
            assert!(
                (result[i] - expected).abs() < 1e-10,
                "trigamma_batch mismatch at {}: got {}, expected {}",
                x[i],
                result[i],
                expected
            );
        }
    }
}
