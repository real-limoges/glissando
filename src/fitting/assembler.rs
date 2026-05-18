//! Design matrix and penalty matrix assembly from formula terms.
//!
//! Converts a [`Formula`] into numeric [`ModelMatrix`] and [`PenaltyMatrix`] structures
//! that feed into the penalized weighted least squares solver.

use super::{GamlssError, PenaltyMatrix, Smooth, Term};
use crate::splines::{
    create_basis_matrix, create_penalty_matrix, kronecker_product, row_kronecker_into,
    sum_to_zero_basis,
};
use crate::types::{DataSet, ModelMatrix};
use ndarray::concatenate;
use ndarray::{s, Array1, Array2, Axis};
use std::collections::HashMap;

fn get_col<'a>(data: &'a DataSet, name: &str) -> Result<&'a Array1<f64>, GamlssError> {
    data.get(name).ok_or_else(|| GamlssError::MissingVariable {
        name: name.to_string(),
    })
}

fn assemble_smooth(
    data: &DataSet,
    n_obs: usize,
    smooth: &Smooth,
    apply_constraint: bool,
) -> Result<(Array2<f64>, Vec<PenaltyMatrix>), GamlssError> {
    match smooth {
        Smooth::PSpline1D {
            col_name,
            n_splines,
            degree,
            penalty_order,
        } => {
            let x_col = get_col(data, col_name)?;
            let basis = create_basis_matrix(x_col, *n_splines, *degree);
            let penalty = create_penalty_matrix(*n_splines, *penalty_order);

            if apply_constraint && *n_splines >= 2 {
                let z = sum_to_zero_basis(*n_splines);
                let basis_c = basis.dot(&z);
                let penalty_c = z.t().dot(&penalty).dot(&z);
                Ok((basis_c, vec![PenaltyMatrix(penalty_c)]))
            } else {
                Ok((basis, vec![PenaltyMatrix(penalty)]))
            }
        }

        Smooth::TensorProduct {
            col_name_1,
            n_splines_1,
            penalty_order_1,
            col_name_2,
            n_splines_2,
            penalty_order_2,
            degree,
        } => {
            let x1 = get_col(data, col_name_1)?;
            let b1_raw = create_basis_matrix(x1, *n_splines_1, *degree);
            let s1_raw = create_penalty_matrix(*n_splines_1, *penalty_order_1);

            let x2 = get_col(data, col_name_2)?;
            let b2_raw = create_basis_matrix(x2, *n_splines_2, *degree);
            let s2_raw = create_penalty_matrix(*n_splines_2, *penalty_order_2);

            // Apply sum-to-zero to each marginal independently when an Intercept is
            // also on the parameter — the row-Kronecker of two partition-of-unity
            // bases is itself partition-of-unity, so without this the tensor
            // smooth makes [1 | B] rank-deficient too.
            let (b1, s1, k1) = if apply_constraint && *n_splines_1 >= 2 {
                let z1 = sum_to_zero_basis(*n_splines_1);
                let b1 = b1_raw.dot(&z1);
                let s1 = z1.t().dot(&s1_raw).dot(&z1);
                let k1 = *n_splines_1 - 1;
                (b1, s1, k1)
            } else {
                (b1_raw, s1_raw, *n_splines_1)
            };
            let (b2, s2, k2) = if apply_constraint && *n_splines_2 >= 2 {
                let z2 = sum_to_zero_basis(*n_splines_2);
                let b2 = b2_raw.dot(&z2);
                let s2 = z2.t().dot(&s2_raw).dot(&z2);
                let k2 = *n_splines_2 - 1;
                (b2, s2, k2)
            } else {
                (b2_raw, s2_raw, *n_splines_2)
            };

            let n_coeffs_total = k1 * k2;
            let mut basis = Array2::<f64>::zeros((n_obs, n_coeffs_total));
            for i in 0..n_obs {
                row_kronecker_into(b1.row(i), b2.row(i), basis.row_mut(i));
            }

            // Anisotropic penalties: S1⊗I2 for x1 direction, I1⊗S2 for x2 direction.
            let i_k1 = Array2::<f64>::eye(k1);
            let i_k2 = Array2::<f64>::eye(k2);

            let penalty_1 = kronecker_product(&s1, &i_k2);
            let penalty_2 = kronecker_product(&i_k1, &s2);
            Ok((
                basis,
                vec![PenaltyMatrix(penalty_1), PenaltyMatrix(penalty_2)],
            ))
        }

        Smooth::RandomEffect { col_name } => {
            // Ridge-penalized indicators: equivalent to alpha ~ N(0, 1/lambda).
            let group_var = get_col(data, col_name)?;

            let mut group_to_id: HashMap<String, usize> = HashMap::new();
            for val in group_var.iter() {
                let key: String = val.to_string();
                let next_id = group_to_id.len();
                group_to_id.entry(key).or_insert(next_id);
            }

            let n_groups = group_to_id.len();
            let mut basis = Array2::<f64>::zeros((n_obs, n_groups));

            for (i, val) in group_var.iter().enumerate() {
                let key: String = val.to_string();
                if let Some(&group_id) = group_to_id.get(&key) {
                    basis[[i, group_id]] = 1.0;
                }
            }

            // Indicator basis is partition-of-unity (each row sums to 1), same
            // rank-deficiency story as P-splines.
            if apply_constraint && n_groups >= 2 {
                let z = sum_to_zero_basis(n_groups);
                let basis_c = basis.dot(&z);
                // Z'·I·Z = Z'·Z = I_{k-1} since Z has orthonormal columns.
                let penalty_c = Array2::<f64>::eye(n_groups - 1);
                Ok((basis_c, vec![PenaltyMatrix(penalty_c)]))
            } else {
                let penalty = Array2::<f64>::eye(n_groups);
                Ok((basis, vec![PenaltyMatrix(penalty)]))
            }
        }
    }
}

/// Assemble the design matrix X and penalty matrices S_j from formula terms.
///
/// Horizontally concatenates basis matrices for each term (intercept, linear, smooth)
/// and embeds penalty blocks at the correct offsets in the full coefficient space.
pub(crate) fn assemble_model_matrices(
    data: &DataSet,
    n_obs: usize,
    terms: &[Term],
) -> Result<(ModelMatrix, Vec<PenaltyMatrix>, usize), GamlssError> {
    // Smooth bases on this codebase (P-spline, tensor-product, random-effect indicator)
    // are all partition-of-unity, so `1_n ∈ col(B)`. When an `Intercept` term is also
    // present the design matrix is rank-deficient. Apply a sum-to-zero
    // reparameterization to the smooths in that case to restore identifiability.
    let has_intercept = terms.iter().any(|t| matches!(t, Term::Intercept));

    let mut model_matrix_parts = Vec::with_capacity(terms.len());
    let mut penalty_blocks: Vec<(usize, PenaltyMatrix)> = Vec::new();
    let mut total_coeffs = 0;

    for term in terms {
        match term {
            Term::Intercept => {
                let part = Array1::ones(n_obs).insert_axis(Axis(1));
                model_matrix_parts.push(part);
                total_coeffs += 1;
            }
            Term::Linear { col_name } => {
                let x_col_vec = get_col(data, col_name)?;
                let part: Array2<f64> = x_col_vec.to_owned().insert_axis(Axis(1));
                model_matrix_parts.push(part);
                total_coeffs += 1;
            }
            Term::Smooth(smooth) => {
                let (basis, penalties) = assemble_smooth(data, n_obs, smooth, has_intercept)?;
                let n_coeffs = basis.ncols();
                model_matrix_parts.push(basis);

                for penalty_block in penalties {
                    penalty_blocks.push((total_coeffs, penalty_block));
                }
                total_coeffs += n_coeffs;
            }
        }
    }

    let x_model = ModelMatrix(concatenate(
        Axis(1),
        &model_matrix_parts
            .iter()
            .map(|m| m.view())
            .collect::<Vec<_>>(),
    )?);

    let penalty_matrices = penalty_blocks
        .into_iter()
        .map(|(start_index, block)| {
            let mut s_j = PenaltyMatrix(Array2::<f64>::zeros((total_coeffs, total_coeffs)));
            let n = block.ncols();
            s_j.slice_mut(s![
                start_index..start_index + n,
                start_index..start_index + n
            ])
            .assign(&block);
            s_j
        })
        .collect::<Vec<_>>();

    Ok((x_model, penalty_matrices, total_coeffs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terms::Smooth;

    #[test]
    fn pspline_basis_is_partition_of_unity() {
        // P-spline basis rows should sum to 1 — the property that makes them
        // compatible with an intercept column (and triggers sum-to-zero
        // reparameterization in `assemble_smooth`).
        let mut data = DataSet::new();
        let n_obs = 100;
        data.insert_column("x", Array1::linspace(0.0, 1.0, n_obs));

        let term = Term::Smooth(Smooth::PSpline1D {
            col_name: "x".into(),
            n_splines: 10,
            degree: 3,
            penalty_order: 2,
        });

        let (mm, _, _) = assemble_model_matrices(&data, n_obs, &[term]).unwrap();
        for row in mm.0.rows() {
            let row_sum: f64 = row.sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-10,
                "spline basis row sums to {} (expected 1.0)",
                row_sum
            );
        }
    }
}
