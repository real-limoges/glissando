//! Kronecker product utilities for tensor product smooth terms.

use ndarray::{s, Array2, ArrayView1, ArrayViewMut1, Zip};

/// Compute the Kronecker product of two matrices: C = A ⊗ B.
///
/// If A is (m × n) and B is (p × q), the result is (mp × nq). I use it to build
/// tensor product basis matrices.
pub(crate) fn kronecker_product(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (m, n) = a.dim();
    let (p, q) = b.dim();

    let mut c = Array2::<f64>::zeros((m * p, n * q));

    for i in 0..m {
        for j in 0..n {
            let a_scalar = a[[i, j]];
            let block = c.slice_mut(s![i * p..(i + 1) * p, j * q..(j + 1) * q]);
            Zip::from(block).and(b).for_each(|c_val, &b_val| {
                *c_val = a_scalar * b_val;
            });
        }
    }
    c
}

/// Compute row-wise Kronecker product into a pre-allocated output buffer.
///
/// Given vectors a (length m) and b (length n), computes their Kronecker product
/// (length m*n) in place. This is the hot path for tensor product basis evaluation.
#[inline]
pub(crate) fn row_kronecker_into(
    a: ArrayView1<f64>,
    b: ArrayView1<f64>,
    mut out: ArrayViewMut1<f64>,
) {
    let len_b = b.len();
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            out[i * len_b + j] = ai * bj;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    #[cfg(not(target_arch = "wasm32"))]
    use proptest::prelude::*;

    // --- kronecker_product ---

    #[test]
    fn kronecker_product_dimensions() {
        let a = Array2::<f64>::zeros((2, 3));
        let b = Array2::<f64>::zeros((4, 5));
        let c = kronecker_product(&a, &b);
        assert_eq!(c.dim(), (8, 15));
    }

    #[test]
    fn kronecker_product_with_identities_is_identity() {
        let i2 = Array2::<f64>::eye(2);
        let i3 = Array2::<f64>::eye(3);
        let k = kronecker_product(&i2, &i3);
        let i6 = Array2::<f64>::eye(6);
        assert_eq!(k, i6);
    }

    #[test]
    fn kronecker_product_block_structure() {
        let a = ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let b = ndarray::arr2(&[[0.0, 5.0], [6.0, 7.0]]);
        let c = kronecker_product(&a, &b);
        // Top-left 2x2 block = a[0,0] * b
        assert_eq!(c[[0, 0]], 0.0);
        assert_eq!(c[[0, 1]], 5.0);
        assert_eq!(c[[1, 0]], 6.0);
        assert_eq!(c[[1, 1]], 7.0);
        // Top-right 2x2 block = a[0,1] * b
        assert_eq!(c[[0, 2]], 0.0);
        assert_eq!(c[[0, 3]], 10.0);
        // Bottom-right 2x2 block = a[1,1] * b
        assert_eq!(c[[2, 2]], 0.0);
        assert_eq!(c[[3, 3]], 28.0);
    }

    // --- row_kronecker_into ---

    #[test]
    fn row_kronecker_into_matches_naive() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![10.0, 20.0]);
        let mut out = Array1::<f64>::zeros(6);
        row_kronecker_into(a.view(), b.view(), out.view_mut());
        assert_eq!(out.to_vec(), vec![10.0, 20.0, 20.0, 40.0, 30.0, 60.0]);
    }

    #[cfg(not(target_arch = "wasm32"))]
    proptest! {
        #[test]
        fn row_kronecker_into_matches_outer_product_flatten(
            a_vec in proptest::collection::vec(-10.0f64..10.0, 1..6),
            b_vec in proptest::collection::vec(-10.0f64..10.0, 1..6),
        ) {
            let a = Array1::from_vec(a_vec.clone());
            let b = Array1::from_vec(b_vec.clone());
            let mut out = Array1::<f64>::zeros(a.len() * b.len());
            row_kronecker_into(a.view(), b.view(), out.view_mut());
            for i in 0..a.len() {
                for j in 0..b.len() {
                    let expected = a_vec[i] * b_vec[j];
                    let actual = out[i * b.len() + j];
                    prop_assert!((expected - actual).abs() < 1e-12);
                }
            }
        }
    }
}
