#![no_main]

//! Fuzz the DATA-5 string formula parser. The parser must be total: any `&str`
//! yields `Ok`/`Err`, never a panic, out-of-bounds index, or `remove(0)` on an
//! empty fold. `tests/formula_parse.rs` proves the same property with proptest;
//! this drives it with coverage-guided input for the long tail.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &str| {
    let _ = glissando::parse_formula_string(data);
});
