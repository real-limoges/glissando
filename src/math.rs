pub fn trigamma(x: f64) -> f64 {
    let mut x = x;
    let mut result = 0.0;

    // shift it over where expansion is accurate
    while x < 5.0 {
        result += 1.0 / (x * x);
        x += 1.0;
    }

    let inv_x = 1.0 / x;
    let inv_x2 = inv_x * inv_x;

    let expansion = inv_x + 0.5 * inv_x2 + (1.0 / 6.0) * inv_x2.powi(1) * inv_x
        - (1.0 / 30.0) * inv_x2.powi(2) * inv_x
        + (1.0 / 42.0) * inv_x2.powi(3) * inv_x
        - (1.0 / 30.0) * inv_x2.powi(4) * inv_x;

    let return_val = expansion + result;
    return_val
}
