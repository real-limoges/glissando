//! B-spline basis construction and penalty matrices for P-spline smoothing,
//! plus natural cubic regression spline (mgcv `bs="cr"`) basis and penalty.
//!
//! Provides routines for building B-spline / natural-cubic-spline basis matrices,
//! penalty matrices, and Kronecker product utilities used by smooth terms.

mod cr;
mod penalty;
mod pspline;
mod reparam;
mod tensor;

pub(crate) use cr::*;
pub(crate) use penalty::*;
pub(crate) use pspline::*;
pub(crate) use reparam::*;
pub(crate) use tensor::*;

#[cfg(test)]
pub(crate) mod test_support {
    use ndarray::Array2;

    pub(crate) fn is_symmetric(m: &Array2<f64>, eps: f64) -> bool {
        let (r, c) = m.dim();
        if r != c {
            return false;
        }
        for i in 0..r {
            for j in 0..r {
                if (m[[i, j]] - m[[j, i]]).abs() > eps {
                    return false;
                }
            }
        }
        true
    }

    pub(crate) fn is_psd(m: &Array2<f64>) -> bool {
        // A matrix M = D'D is automatically PSD; verify by checking x'Mx >= 0 for random x.
        // Cheap stand-in for an eigenvalue computation that doesn't need a linalg backend.
        use ndarray::{Array, Array1};
        let n = m.dim().0;
        let mut rng = StdRngStub::new(42);
        for _ in 0..20 {
            let v: Array1<f64> = Array::from_shape_fn(n, |_| rng.next() - 0.5);
            let q = v.dot(&m.dot(&v));
            if q < -1e-9 {
                return false;
            }
        }
        true
    }

    /// Tiny LCG for test-only deterministic numbers (no rand dep needed in the test scope).
    pub(crate) struct StdRngStub {
        state: u64,
    }
    impl StdRngStub {
        pub(crate) fn new(seed: u64) -> Self {
            Self { state: seed.max(1) }
        }
        pub(crate) fn next(&mut self) -> f64 {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.state >> 33) as f64) / (1u64 << 31) as f64
        }
    }
}
