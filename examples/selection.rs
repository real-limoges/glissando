//! Model selection: GAIC (AIC/BIC), a likelihood-ratio test between nested
//! models, and forward stepwise term selection.
//!
//! Run with: `cargo run --example selection`

use glissando::distributions::Gaussian;
use glissando::ndarray::Array1;
use glissando::selection::{lr_test, step_gaic, Direction, StepScope};
use glissando::{DataSet, FitConfig, Formula, GamlssError, GamlssModel, Term};

fn main() -> Result<(), GamlssError> {
    let n = 150;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * 4.0 - 2.0).collect();
    let z: Vec<f64> = x.iter().map(|&v| (v * 1.3).cos()).collect();
    // y depends on x but not z, so selection should keep x and drop z.
    // A small deterministic wobble keeps the fit realistic (not a perfect line).
    let y = Array1::from_vec(
        x.iter()
            .enumerate()
            .map(|(i, &v)| 1.0 + 0.8 * v + 0.15 * ((i as f64) * 1.7).sin())
            .collect(),
    );

    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_vec(x));
    data.insert_column("z", Array1::from_vec(z));

    let family = Gaussian::new();

    // Two nested models: intercept-only mu, versus mu ~ x.
    let f_null = Formula::from_strings([("mu", "y ~ 1"), ("sigma", "~ 1")])?;
    let f_x = Formula::from_strings([("mu", "y ~ x"), ("sigma", "~ 1")])?;
    let m_null = GamlssModel::fit(&data, &y, &f_null, &family)?;
    let m_x = GamlssModel::fit(&data, &y, &f_x, &family)?;

    println!(
        "AIC: null={:.1}   +x={:.1}",
        m_null.gaic(&family, &y, 2.0)?,
        m_x.gaic(&family, &y, 2.0)?
    );

    // Likelihood-ratio test (small model first, then the larger nested model).
    let lr = lr_test(&m_null, &m_x, &family, &y)?;
    println!("LR test: stat={:.2}  df={:.0}  p={:.4}", lr.lr_stat, lr.df, lr.p_value);

    // Forward stepwise over candidate linear terms {x, z} on mu.
    let scope = vec![StepScope {
        param: "mu".to_string(),
        candidates: vec![Term::linear("x"), Term::linear("z")],
    }];
    let result = step_gaic(
        &data,
        &y,
        &family,
        f_null,
        &scope,
        2.0,
        Direction::Forward,
        FitConfig::default(),
    )?;
    println!("stepwise: {} steps in trace", result.trace.len());

    Ok(())
}
