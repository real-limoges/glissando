#![no_main]

//! Fuzz the `glissando::json` embedding facade's string-in parse seams. Each entry
//! point must be total: arbitrary bytes yield `Ok`/`Err`, never a panic. `load`
//! also exercises the deserialize + `FamilyDescriptor::build` path. The proptest
//! mirror lives in `tests/prop_json.rs`.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &str| {
    let _ = glissando::json::parse_response(data);
    let _ = glissando::json::parse_data(data);
    let _ = glissando::json::parse_formula(data);
    let _ = glissando::json::parse_config(data);
    let _ = glissando::json::load(data);
});
