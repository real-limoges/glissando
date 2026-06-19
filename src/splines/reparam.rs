//! Sum-to-zero reparameterization for smooth bases.

use ndarray::{Array1, Array2};

/// Householder-built orthonormal basis (k × (k−1)) of the null-space of `1_k`.
///
/// Used as a sum-to-zero reparameterization for smooth bases that exhibit the
/// partition-of-unity property (P-spline: `B · 1_k = 1_n`). When an [`Intercept`]
/// term sits alongside such a smooth on the same parameter, the design matrix
/// `[1 | B]` is rank-deficient because `1_n ∈ col(B)`; replacing `B` with `B · Z`
/// and the penalty `S` with `Z' · S · Z` removes the constant direction from the
/// smooth and restores identifiability.
///
/// `Z` depends only on `k`, not on training data, so prediction reapplies the
/// same constraint deterministically.
///
/// [`Intercept`]: crate::Term::Intercept
pub(crate) fn sum_to_zero_basis(k: usize) -> Array2<f64> {
    assert!(k >= 2, "sum-to-zero basis requires k >= 2 (got {})", k);

    // Householder reflector that maps 1_k onto −√k · e_1:
    //   v = 1_k + √k · e_1   (sign chosen so v_0 = 1 + √k never vanishes),
    //   H = I − 2·v·v'/(v'·v).
    // The remaining columns H[:, 1..] span the orthogonal complement of 1_k and
    // are orthonormal because H is orthogonal.
    let sqrt_k = (k as f64).sqrt();
    let mut v = Array1::ones(k);
    v[0] += sqrt_k;
    let v_norm_sq: f64 = v.iter().map(|x: &f64| x * x).sum();
    let factor = 2.0 / v_norm_sq;

    let mut z = Array2::<f64>::zeros((k, k - 1));
    for j in 1..k {
        for i in 0..k {
            let kron = if i == j { 1.0 } else { 0.0 };
            z[[i, j - 1]] = kron - factor * v[i] * v[j];
        }
    }
    z
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- sum_to_zero_basis ---

    #[test]
    fn sum_to_zero_basis_has_correct_shape() {
        let z = sum_to_zero_basis(5);
        assert_eq!(z.dim(), (5, 4));
    }

    #[test]
    fn sum_to_zero_basis_is_orthogonal_to_ones() {
        // Z' · 1_k = 0 — every column sums to zero.
        for k in [2usize, 3, 5, 8, 12] {
            let z = sum_to_zero_basis(k);
            for j in 0..z.ncols() {
                let s: f64 = z.column(j).sum();
                assert!(s.abs() < 1e-12, "k={} col {} sum = {}", k, j, s);
            }
        }
    }

    #[test]
    fn sum_to_zero_basis_columns_are_orthonormal() {
        // Z' · Z = I_{k-1}.
        for k in [2usize, 3, 5, 8] {
            let z = sum_to_zero_basis(k);
            let zt_z = z.t().dot(&z);
            let identity = Array2::<f64>::eye(k - 1);
            for i in 0..(k - 1) {
                for j in 0..(k - 1) {
                    assert!(
                        (zt_z[[i, j]] - identity[[i, j]]).abs() < 1e-10,
                        "k={} Z'Z[{},{}]={} (expected {})",
                        k,
                        i,
                        j,
                        zt_z[[i, j]],
                        identity[[i, j]]
                    );
                }
            }
        }
    }

    #[test]
    fn sum_to_zero_constraint_removes_constant_direction() {
        // After centering, B·Z·γ should not be a non-zero constant for any γ ≠ 0.
        // Equivalent statement: 1_k'·Z·γ = 0 always (Z'·1_k = 0), so β = Z·γ is
        // never parallel to 1_k for non-zero γ.
        let k = 6;
        let z = sum_to_zero_basis(k);
        let gamma = Array1::from_vec(vec![0.3, -1.2, 2.1, 0.7, -0.5]);
        let beta = z.dot(&gamma);
        let dot_with_ones: f64 = beta.iter().sum();
        assert!(dot_with_ones.abs() < 1e-12);
    }
}
