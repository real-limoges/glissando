/// Trigamma function: psi'(x) = d^2/dx^2 log(Gamma(x))
/// Uses recurrence relation for x < 5, asymptotic expansion for x >= 5.
/// See docs/mathematics.md for derivation and test values.
pub fn trigamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }

    let mut x_shifted = x;
    let mut result = 0.0;

    // Recurrence: psi'(x) = psi'(x+1) + 1/x^2. Shift x up until x >= 5.
    while x_shifted < 10.0 {
        result += 1.0 / (x_shifted * x_shifted);
        x_shifted += 1.0;
    }

    // Asymptotic expansion from Abramowitz & Stegun 6.4.11
    let inv_x = 1.0 / x_shifted;
    let inv_x2 = inv_x * inv_x;
    let inv_x3 = inv_x2 * inv_x;
    let inv_x5 = inv_x3 * inv_x2;
    let inv_x7 = inv_x5 * inv_x2;

    let expansion = inv_x + inv_x2 / 2.0 + inv_x3 / 6.0 - inv_x5 / 30.0 + inv_x7 / 42.0;

    expansion + result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trigamma() {
        assert!((trigamma(1.0) - 1.6449340668482264).abs() < 1e-10);

        assert!((trigamma(2.0) - 0.6449340668482264).abs() < 1e-10);
    }
}
