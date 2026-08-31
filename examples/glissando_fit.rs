use glissando::distributions::StudentT;
use glissando::{DataSet, Formula, GamlssError, GamlssModel, Smooth, Term};
use ndarray::Array1;
use rand::RngExt;

fn main() -> Result<(), GamlssError> {
    // Synthetic data.

    let mut rng = rand::rng();
    let n = 200;

    let x_vals: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1).collect();

    let y_vals: Vec<f64> = x_vals
        .iter()
        .map(|&x| {
            let mu = x.sin();
            let sigma = 0.5 + 0.1 * x;

            let noise: f64 = rng.random_range(-1.0..1.0);
            mu + sigma * noise
        })
        .collect();

    let y = Array1::from_vec(y_vals);
    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_vec(x_vals));

    // The formula, using the terse constructors. `Smooth::ps` already carries sane
    // defaults (degree 3, 2nd-order penalty); builders like `.n_splines(20)` override them.
    //   mu    ~ intercept + P-spline(x)   (smooth mean)
    //   sigma ~ intercept + x             (linear heteroskedasticity)
    //   nu    ~ intercept                 (constant tail weight)
    let formulas = Formula::new()
        .with_terms(
            "mu",
            vec![Term::Intercept, Term::smooth(Smooth::ps("x").n_splines(20))],
        )
        .with_terms("sigma", vec![Term::Intercept, Term::linear("x")])
        .with_terms("nu", vec![Term::Intercept]);

    // Fit it.
    println!("Fitting GAMLSS model...");
    let model = GamlssModel::fit(&data, &y, &formulas, &StudentT::new())?;
    println!("Successfully Trained GAMLSS Model!");

    // Poke at the results.
    let mu_model = &model.models["mu"];
    let sigma_model = &model.models["sigma"];
    let nu_model = &model.models["nu"];

    println!("--- Results ---");
    println!("Mu coefficients count: {}", mu_model.coefficients.len());
    println!("Sigma coefficients: {:?}", sigma_model.coefficients);
    println!("Nu coefficients: {:?}", nu_model.coefficients);

    Ok(())
}
