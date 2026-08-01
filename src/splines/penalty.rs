//! B-spline difference penalty matrices for P-spline smoothing.

use ndarray::Array2;

/// P-spline penalty S = D'D.  Order 2 approximates the integral of the squared
/// second derivative (exact for the natural cubic spline, approximate for B-splines).
pub(crate) fn create_penalty_matrix(n_splines: usize, order: usize) -> Array2<f64> {
    let n_rows_d = n_splines.saturating_sub(order);
    if n_rows_d == 0 {
        return Array2::<f64>::zeros((n_splines, n_splines));
    }

    // General order-d difference coefficients: convolve [1, -1] with itself d
    // times, giving the alternating binomial row (1, -d, ..., ±1): identical to
    // R's diff(diag(k), differences = d). The previous code special-cased orders
    // 1 and 2 and silently reused the order-2 row (with the wrong number of
    // rows) for anything higher.
    let mut coef = vec![1.0_f64];
    for _ in 0..order {
        let mut next = vec![0.0; coef.len() + 1];
        for (k, &c) in coef.iter().enumerate() {
            next[k] += c;
            next[k + 1] -= c;
        }
        coef = next;
    }

    let mut d_matrix = Array2::<f64>::zeros((n_rows_d, n_splines));
    for i in 0..n_rows_d {
        for (j, &c) in coef.iter().enumerate() {
            d_matrix[[i, i + j]] = c;
        }
    }
    d_matrix.t().dot(&d_matrix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::splines::test_support::{is_psd, is_symmetric};
    use ndarray::Array1;
    #[cfg(not(target_arch = "wasm32"))]
    use proptest::prelude::*;

    // --- create_penalty_matrix ---

    #[test]
    fn penalty_matrix_order2_shape_and_symmetry() {
        let p = create_penalty_matrix(10, 2);
        assert_eq!(p.dim(), (10, 10));
        assert!(is_symmetric(&p, 1e-12));
    }

    #[test]
    fn penalty_matrix_order1_shape_and_symmetry() {
        let p = create_penalty_matrix(8, 1);
        assert_eq!(p.dim(), (8, 8));
        assert!(is_symmetric(&p, 1e-12));
    }

    #[test]
    fn penalty_matrix_psd_order2() {
        let p = create_penalty_matrix(10, 2);
        assert!(is_psd(&p));
    }

    #[test]
    fn penalty_matrix_psd_order1() {
        let p = create_penalty_matrix(10, 1);
        assert!(is_psd(&p));
    }

    #[test]
    fn penalty_matrix_order1_constant_in_null_space() {
        // First-order difference penalty has constant vectors in its null space.
        let p = create_penalty_matrix(8, 1);
        let ones = Array1::from_elem(8, 1.0);
        let q = ones.dot(&p.dot(&ones));
        assert!(
            q.abs() < 1e-10,
            "constant should lie in null space, got {}",
            q
        );
    }

    #[test]
    fn penalty_matrix_order2_linear_in_null_space() {
        // Second-order difference penalty has linear vectors in its null space.
        let p = create_penalty_matrix(8, 2);
        let lin: Array1<f64> = Array1::from_iter((0..8).map(|i| i as f64));
        let q = lin.dot(&p.dot(&lin));
        assert!(
            q.abs() < 1e-10,
            "linear should lie in null space, got {}",
            q
        );
    }

    #[test]
    fn penalty_matrix_degenerate_when_order_ge_n_splines() {
        let p = create_penalty_matrix(2, 3);
        assert_eq!(p.dim(), (2, 2));
        assert!(p.iter().all(|&v| v == 0.0));
    }

    #[cfg(not(target_arch = "wasm32"))]
    proptest! {
        #[test]
        fn penalty_matrix_always_symmetric(
            n_splines in 4usize..20,
            order in 1usize..3,
        ) {
            let p = create_penalty_matrix(n_splines, order);
            prop_assert_eq!(p.dim(), (n_splines, n_splines));
            for i in 0..n_splines {
                for j in 0..n_splines {
                    prop_assert!((p[[i, j]] - p[[j, i]]).abs() < 1e-12);
                }
            }
        }
    }
}
