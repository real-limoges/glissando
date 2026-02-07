use super::{GamlssError, PenaltyMatrix, Smooth, Term};
use crate::splines::{
    create_basis_matrix, create_penalty_matrix, kronecker_product, row_kronecker_into,
};
use crate::types::DataSet;
use crate::ModelMatrix;
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

            Ok((basis, vec![PenaltyMatrix(penalty)]))
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
            let b1 = create_basis_matrix(x1, *n_splines_1, *degree);
            let s1 = create_penalty_matrix(*n_splines_1, *penalty_order_1);

            let x2 = get_col(data, col_name_2)?;
            let b2 = create_basis_matrix(x2, *n_splines_2, *degree);
            let s2 = create_penalty_matrix(*n_splines_2, *penalty_order_2);

            let n_coeffs_total = *n_splines_1 * *n_splines_2;

            let mut basis = Array2::<f64>::zeros((n_obs, n_coeffs_total));

            for i in 0..n_obs {
                row_kronecker_into(b1.row(i), b2.row(i), basis.row_mut(i));
            }

            // Anisotropic penalties: S1⊗I2 for x1 direction, I1⊗S2 for x2 direction
            let i_k1 = Array2::<f64>::eye(*n_splines_1);
            let i_k2 = Array2::<f64>::eye(*n_splines_2);

            let penalty_1 = kronecker_product(&s1, &i_k2);
            let penalty_2 = kronecker_product(&i_k1, &s2);
            Ok((
                basis,
                vec![PenaltyMatrix(penalty_1), PenaltyMatrix(penalty_2)],
            ))
        }

        Smooth::RandomEffect { col_name } => {
            // Ridge-penalized indicators: equivalent to alpha ~ N(0, 1/lambda)
            let group_var = get_col(data, col_name)?;

            // Find unique groups and create mapping
            let mut unique_groups = Vec::new();
            let mut group_to_id: HashMap<String, usize> = HashMap::new();

            for val in group_var.iter() {
                let key: String = val.to_string();
                if let std::collections::hash_map::Entry::Vacant(e) = group_to_id.entry(key.clone())
                {
                    let id = unique_groups.len();
                    unique_groups.push(key);
                    e.insert(id);
                }
            }

            let n_groups = unique_groups.len();
            let mut basis = Array2::<f64>::zeros((n_obs, n_groups));

            for (i, val) in group_var.iter().enumerate() {
                let key: String = val.to_string();
                if let Some(&group_id) = group_to_id.get(&key) {
                    basis[[i, group_id]] = 1.0;
                }
            }

            let penalty = Array2::<f64>::eye(n_groups);

            Ok((basis, vec![PenaltyMatrix(penalty)]))
        }
    }
}

/// Assemble the design matrix and penalty matrices from formula terms.
///
/// Constructs the combined model matrix X by concatenating columns for each term
/// (intercept, linear, smooth), and builds the corresponding penalty matrices
/// for smooth terms.
///
/// # Returns
/// * `ModelMatrix` - Combined design matrix (n_obs × total_coeffs)
/// * `Vec<PenaltyMatrix>` - Penalty matrices for each smooth term
/// * `usize` - Total number of coefficients
pub fn assemble_model_matrices(
    data: &DataSet,
    n_obs: usize,
    terms: &[Term],
) -> Result<(ModelMatrix, Vec<PenaltyMatrix>, usize), GamlssError> {
    let mut model_matrix_parts = Vec::new();
    let mut penalty_blocks = Vec::new();
    let mut total_coeffs = 0;

    for term in terms.iter() {
        match term {
            Term::Intercept => {
                let part = Array1::ones(n_obs)
                    .into_shape_with_order((n_obs, 1))
                    .map_err(|err| GamlssError::ComputationError(err.to_string()))?;
                model_matrix_parts.push(part);
                total_coeffs += 1;
            }
            Term::Linear { col_name } => {
                let x_col_vec = get_col(data, col_name)?;
                let part: Array2<f64> = x_col_vec
                    .to_owned()
                    .into_shape_with_order((n_obs, 1))
                    .map_err(|err: ndarray::ShapeError| {
                        GamlssError::ComputationError(err.to_string())
                    })?;
                model_matrix_parts.push(part);
                total_coeffs += 1;
            }
            Term::Smooth(smooth) => {
                let (basis, penalties) = assemble_smooth(data, n_obs, smooth)?;
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
