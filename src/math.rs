/// Trigamma function: psi'(x) = d^2/dx^2 log(Gamma(x))
/// Uses recurrence relation for x < 5, asymptotic expansion for x >= 5.
/// See docs/mathematics.md for derivation and test values.
pub fn trigamma(x: f64) -> f64 {
    let mut x = x;
    let mut result = 0.0;

    // Recurrence: psi'(x) = psi'(x+1) + 1/x^2. Shift x up until x >= 5.
    while x < 5.0 {
        result += 1.0 / (x * x);
        x += 1.0;
    }

    // Asymptotic expansion from Abramowitz & Stegun 6.4.11
    let inv_x = 1.0 / x;
    let inv_x2 = inv_x * inv_x;

    let expansion = inv_x + 0.5 * inv_x2 + (1.0 / 6.0) * inv_x2.powi(1) * inv_x
        - (1.0 / 30.0) * inv_x2.powi(2) * inv_x
        + (1.0 / 42.0) * inv_x2.powi(3) * inv_x
        - (1.0 / 30.0) * inv_x2.powi(4) * inv_x;

    expansion + result
}
