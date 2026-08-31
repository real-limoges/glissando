//! Batch digamma and trigamma over `Array1<f64>`.
//!
//! These are the special functions the score and Fisher-information terms lean
//! on, so they run over whole arrays at a time, with optional Rayon parallelism
//! once an array gets big (n >= 10,000). Small arguments go through recurrence
//! relations; large ones through asymptotic expansions. Same math either way,
//! just the numerically well-behaved path for each regime.

use ndarray::{Array1, Zip};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use statrs::function::gamma::digamma as statrs_digamma;

/// Under this many elements, spinning up Rayon costs more than it saves, so we don't bother.
#[cfg(feature = "parallel")]
pub(crate) const PARALLEL_THRESHOLD: usize = 10_000;

/// Element-wise map over two `Array1<f64>`s, parallelized for large inputs.
///
/// Below [`PARALLEL_THRESHOLD`] elements (or with the `parallel` feature off) this
/// just runs sequentially through `ndarray::Zip`. Above it, Rayon takes the
/// underlying slices, and if an input isn't contiguous it quietly drops back to
/// sequential iteration. Either way the caller never has to think about layout.
#[inline]
pub(crate) fn par_zip_map<F>(a: &Array1<f64>, b: &Array1<f64>, f: F) -> Array1<f64>
where
    F: Fn(f64, f64) -> f64 + Send + Sync,
{
    #[cfg(feature = "parallel")]
    {
        if a.len() >= PARALLEL_THRESHOLD {
            if let (Some(av), Some(bv)) = (a.as_slice(), b.as_slice()) {
                let result: Vec<f64> = av
                    .par_iter()
                    .zip(bv.par_iter())
                    .map(|(&x, &y)| f(x, y))
                    .collect();
                return Array1::from_vec(result);
            }
        }
    }
    Zip::from(a).and(b).map_collect(|&x, &y| f(x, y))
}

/// Element-wise map over a single `Array1<f64>`, parallelized for large inputs.
#[inline]
pub(crate) fn par_map<F>(a: &Array1<f64>, f: F) -> Array1<f64>
where
    F: Fn(f64) -> f64 + Send + Sync,
{
    #[cfg(feature = "parallel")]
    {
        if a.len() >= PARALLEL_THRESHOLD {
            if let Some(av) = a.as_slice() {
                let result: Vec<f64> = av.par_iter().map(|&v| f(v)).collect();
                return Array1::from_vec(result);
            }
        }
    }
    a.mapv(f)
}

/// Three-input variant of [`par_zip_map`].
#[inline]
pub(crate) fn par_zip3_map<F>(
    a: &Array1<f64>,
    b: &Array1<f64>,
    c: &Array1<f64>,
    f: F,
) -> Array1<f64>
where
    F: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    #[cfg(feature = "parallel")]
    {
        if a.len() >= PARALLEL_THRESHOLD {
            if let (Some(av), Some(bv), Some(cv)) = (a.as_slice(), b.as_slice(), c.as_slice()) {
                // Iterator chain, not indexing. The indexed `(0..len).map(|i| av[i]…)`
                // form pays a bounds check on every element; the zip chain pays none.
                let result: Vec<f64> = av
                    .par_iter()
                    .zip(bv.par_iter())
                    .zip(cv.par_iter())
                    .map(|((&x, &y), &z)| f(x, y, z))
                    .collect();
                return Array1::from_vec(result);
            }
        }
    }
    Zip::from(a)
        .and(b)
        .and(c)
        .map_collect(|&x, &y, &z| f(x, y, z))
}

/// Standard-normal quantile `Φ⁻¹(p)` via the inverse error function.
///
/// `p` is clamped to `[1e-12, 1−1e-12]` so the tails stay finite instead of
/// blowing up to `±∞`. Both [`Gaussian::quantile`](crate::distributions::Gaussian)
/// and the randomized quantile residuals (INFER-1) call this, so there is exactly
/// one definition to keep honest.
#[inline]
pub(crate) fn std_normal_quantile(p: f64) -> f64 {
    use statrs::function::erf::erf_inv;
    std::f64::consts::SQRT_2 * erf_inv(2.0 * p.clamp(1e-12, 1.0 - 1e-12) - 1.0)
}

/// Standard-normal CDF `Φ(x)` via the error function. Shared by `ProbitLink`,
/// `Gaussian::cdf`, and the Box-Cox families (`BCCG`), so all use one definition.
#[inline]
pub(crate) fn std_normal_cdf(x: f64) -> f64 {
    use statrs::function::erf::erf;
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Standard-normal PDF `φ(x) = exp(−x²/2)/√(2π)`. The inverse-link derivative of
/// `ProbitLink`; also the density core of the Box-Cox normal families.
#[inline]
pub(crate) fn std_normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Median of `y`, finite entries only. Returns 0.0 on an empty slice; that case
/// shouldn't reach here, since `validate_inputs` already rejects an empty `y` on
/// the public path, so the return value is just a defensive floor rather than
/// anything a caller relies on. The robust `initial_value` seeds for `StudentT`
/// and the Box-Cox families (`BCCG` / `BCT` / `BCPE`) all start from this.
pub(crate) fn median(y: &Array1<f64>) -> f64 {
    let mut v: Vec<f64> = y.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = v.len() / 2;
    if v.len().is_multiple_of(2) {
        0.5 * (v[m - 1] + v[m])
    } else {
        v[m]
    }
}

/// Median absolute deviation about the median: `median(|yᵢ − median(y)|)`.
pub(crate) fn median_abs_deviation(y: &Array1<f64>) -> f64 {
    let med = median(y);
    let dev = y.mapv(|x| (x - med).abs());
    median(&dev)
}

/// Digamma function: psi(x) = d/dx log(Gamma(x)).
/// Just a thin pass-through to statrs. Got an array? Reach for [`digamma_batch`] instead.
#[inline]
pub fn digamma(x: f64) -> f64 {
    statrs_digamma(x)
}

/// Trigamma function: psi'(x) = d²/dx² log(Gamma(x)).
///
/// Uses recurrence relation for x < 10, then asymptotic expansion (A&S 6.4.11).
#[inline]
pub fn trigamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }

    let mut x_shifted = x;
    let mut result = 0.0;

    // Recurrence: psi'(x) = psi'(x+1) + 1/x^2
    // Walk x up until x_shifted >= 10; the asymptotic expansion is only accurate out there.
    while x_shifted < 10.0 {
        result += 1.0 / (x_shifted * x_shifted);
        x_shifted += 1.0;
    }

    // Asymptotic expansion from Abramowitz & Stegun 6.4.11.
    // Now that x is large, this nails it to full precision.
    let inv_x = 1.0 / x_shifted;
    let inv_x2 = inv_x * inv_x;
    let inv_x3 = inv_x2 * inv_x;
    let inv_x5 = inv_x3 * inv_x2;
    let inv_x7 = inv_x5 * inv_x2;

    let expansion = inv_x + inv_x2 / 2.0 + inv_x3 / 6.0 - inv_x5 / 30.0 + inv_x7 / 42.0;

    expansion + result
}

/// Vectorized digamma over an array. Parallelizes via Rayon when n >= [`PARALLEL_THRESHOLD`].
#[inline]
pub fn digamma_batch(x: &Array1<f64>) -> Array1<f64> {
    par_map(x, statrs_digamma)
}

/// Vectorized trigamma over an array. Parallelizes via Rayon when n >= [`PARALLEL_THRESHOLD`].
#[inline]
pub fn trigamma_batch(x: &Array1<f64>) -> Array1<f64> {
    par_map(x, trigamma)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_digamma() {
        // Ground-truth values straight from Mathematica/WolframAlpha
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

    #[test]
    fn trigamma_returns_nan_for_non_positive() {
        assert!(trigamma(0.0).is_nan());
        assert!(trigamma(-1.0).is_nan());
    }

    #[test]
    fn par_zip_map_sequential_for_small() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![10.0, 20.0, 30.0]);
        let r = par_zip_map(&a, &b, |x, y| x + y);
        assert_eq!(r.to_vec(), vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn par_map_sequential_for_small() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let r = par_map(&a, |x| x * 2.0);
        assert_eq!(r.to_vec(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn par_zip3_map_sequential_for_small() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![10.0, 20.0, 30.0]);
        let c = Array1::from_vec(vec![100.0, 200.0, 300.0]);
        let r = par_zip3_map(&a, &b, &c, |x, y, z| x + y + z);
        assert_eq!(r.to_vec(), vec![111.0, 222.0, 333.0]);
    }

    /// Push an array past PARALLEL_THRESHOLD so the parallel branch actually runs.
    #[cfg(feature = "parallel")]
    #[test]
    fn par_zip_map_parallel_branch_matches_sequential() {
        let n = PARALLEL_THRESHOLD + 100;
        let a = Array1::from_iter((0..n).map(|i| i as f64));
        let b = Array1::from_iter((0..n).map(|i| (n - i) as f64));
        let r = par_zip_map(&a, &b, |x, y| x + y);
        for i in 0..n {
            assert_eq!(r[i], n as f64);
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn par_map_parallel_branch_matches_sequential() {
        let n = PARALLEL_THRESHOLD + 50;
        let a = Array1::from_iter((0..n).map(|i| i as f64));
        let r = par_map(&a, |x| x * 2.0);
        for i in 0..n {
            assert_eq!(r[i], 2.0 * i as f64);
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn par_zip3_map_parallel_branch_matches_sequential() {
        let n = PARALLEL_THRESHOLD + 25;
        let a = Array1::from_elem(n, 1.0);
        let b = Array1::from_elem(n, 2.0);
        let c = Array1::from_elem(n, 3.0);
        let r = par_zip3_map(&a, &b, &c, |x, y, z| x + y + z);
        for v in r.iter() {
            assert_eq!(*v, 6.0);
        }
    }
}
