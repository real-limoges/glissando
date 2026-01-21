use super::{GamlssError, PenaltyMatrix, Smooth, Term};
use crate::ModelMatrix;
use crate::splines::{create_basis_matrix, create_penalty_matrix, kronecker_product};
use ndarray::concatenate;
use ndarray::{Array1, Array2, Axis, s, stack};
use polars::chunked_array::ChunkedArray;
use polars::prelude::ChunkUnique;
use polars::prelude::{DataFrame, DataType, IntoSeries, NamedFrom, PolarsError, Series};

fn get_col_as_f64(data: &DataFrame, name: &str, n_obs: usize) -> Result<Array1<f64>, GamlssError> {
    let series = data
        .column(name)
        .map_err(|e| GamlssError::Input(format!("Column '{}' not found: {}", name, e)))?;

    let f64_series = if series.dtype() != &DataType::Float64 {
        series.cast(&DataType::Float64)?
    } else {
        series.clone()
    };

    let f64_chunked_array = f64_series.f64()?;

    let ndarray_data = f64_chunked_array.to_ndarray()?;
    let arr = ndarray_data
        .to_shape(n_obs)
        .map_err(|e| GamlssError::Shape(e.to_string()));

    Ok(Array1::<f64>::from(arr?.to_vec()).to_owned())
}

fn assemble_smooth(
    data: &DataFrame,
    n_obs: usize,
    smooth: &Smooth,
) -> Result<(Array2<f64>, Vec<PenaltyMatrix>), GamlssError> {
    // each smooth has its own arm of the match

    match smooth {
        Smooth::PSpline1D {
            col_name,
            n_splines,
            degree,
            penalty_order,
        } => {
            // super straight forward flow
            let x_col = get_col_as_f64(data, col_name, n_obs)?;
            let basis = create_basis_matrix(&x_col, *n_splines, *degree);
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
            // Tensor product smooths model 2D surfaces f(x1, x2).
            // The basis is the row-wise Kronecker product of the marginal bases.
            // See docs/mathematics.md for the full formulation.
            let x1 = get_col_as_f64(data, col_name_1, n_obs)?;
            let b1 = create_basis_matrix(&x1, *n_splines_1, *degree);
            let s1 = create_penalty_matrix(*n_splines_1, *penalty_order_1);

            let x2 = get_col_as_f64(data, col_name_2, n_obs)?;
            let b2 = create_basis_matrix(&x2, *n_splines_2, *degree);
            let s2 = create_penalty_matrix(*n_splines_2, *penalty_order_2);

            let n_coeffs_total = *n_splines_1 * *n_splines_2;

            let mut basis = Array2::<f64>::zeros((n_obs, n_coeffs_total));

            // Row-wise Kronecker: each row of basis = b1[i,:] kron b2[i,:]
            for i in 0..n_obs {
                let row1 = b1.row(i);
                let row2 = b2.row(i);
                let row_out = kronecker_product(
                    &row1.insert_axis(Axis(0)).to_owned(),
                    &row2.insert_axis(Axis(0)).to_owned(),
                );
                basis.row_mut(i).assign(&row_out.row(0));
            }

            // Tensor product penalties: S1 kron I2 penalizes roughness in x1 direction,
            // I1 kron S2 penalizes roughness in x2 direction. Each gets its own lambda,
            // allowing anisotropic smoothing (different smoothness in each direction).
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
            // Random effects are implemented as ridge-penalized indicator variables.
            // The basis matrix Z is n x G with Z[i,g] = 1 if observation i is in group g.
            // The identity penalty S = I_G gives lambda * sum(alpha_g^2), shrinking
            // group effects toward zero. This is equivalent to assuming alpha ~ N(0, 1/lambda).
            let series = data.column(col_name)?;
            let cat_series = series.categorical()?;
            let id_codes = cat_series.physical();

            let n_groups = id_codes.n_unique()?;
            let mut basis = Array2::<f64>::zeros((n_obs, n_groups));

            let id_col_ndarray = id_codes
                .to_ndarray()?
                .into_shape_with_order(n_obs)
                .map_err(|err| GamlssError::ComputationError(err.to_string()))?;

            for i in 0..n_obs {
                let group_id = id_col_ndarray[i] as usize;
                if group_id < n_groups {
                    basis[[i, group_id]] = 1.0;
                }
            }
            let penalty = Array2::<f64>::eye(n_groups);

            Ok((basis, vec![PenaltyMatrix(penalty)]))
        }
    }
}

pub fn assemble_model_matrices(
    data: &DataFrame,
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
                let x_col_vec = get_col_as_f64(data, col_name, n_obs)?;
                let part = x_col_vec
                    .into_shape_with_order((n_obs, 1))
                    .map_err(|err| GamlssError::ComputationError(err.to_string()))?;
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
