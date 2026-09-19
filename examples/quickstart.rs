//! Quickstart: fit, predict, and diagnose a Gaussian location-scale model.
//!
//! Both the mean (`mu`) and the spread (`sigma`) vary with `x`, which is the
//! thing a plain GAM cannot do and GAMLSS can.
//!
//! Run with: `cargo run --example quickstart`

use glissando::distributions::Gaussian;
use glissando::ndarray::Array1;
use glissando::{DataSet, Formula, GamlssError, GamlssModel, Smooth, Term};

fn main() -> Result<(), GamlssError> {
    let n = 120;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
    let y = Array1::from_vec(x.iter().map(|&v| v.sin() + 0.1 * v).collect());

    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_vec(x));

    // one additive predictor per parameter
    let formula = Formula::new()
        .with_terms(
            "mu",
            vec![Term::Intercept, Term::smooth(Smooth::ps("x").n_splines(20))],
        )
        .with_terms("sigma", vec![Term::Intercept, Term::linear("x")]);

    let family = Gaussian::new();
    let model = GamlssModel::fit(&data, &y, &formula, &family)?;
    println!("converged: {}", model.converged());

    // predict on the response scale, keyed by parameter name
    let preds = model.predict(&data, &family)?;
    println!("mu[0..3]    = {:?}", &preds["mu"].as_slice().unwrap()[..3]);
    println!(
        "sigma[0..3] = {:?}",
        &preds["sigma"].as_slice().unwrap()[..3]
    );

    // randomized quantile residuals are the GAMLSS default residual
    let resid = model.quantile_residuals(&family, &y, Some(42))?;
    println!("mean quantile residual (~0): {:.4}", resid.mean().unwrap());

    // model quality: k = 2 is AIC, k = ln(n) is BIC
    let aic = model.gaic(&family, &y, 2.0)?;
    let bic = model.gaic(&family, &y, (y.len() as f64).ln())?;
    println!("AIC = {aic:.2}   BIC = {bic:.2}");

    Ok(())
}
