.PHONY: fmt clippy test build ci check-wasm build-wasm

fmt:
	cargo fmt

clippy:
	cargo clippy -- -D warnings

test:
	cargo test

build:
	cargo build

build-native:
	RUSTFLAGS="-C target-cpu=native" cargo build --release

check-wasm:
	cargo check --target wasm32-unknown-unknown --no-default-features --features wasm

build-wasm:
	wasm-pack build --no-default-features --features wasm
	@printf '# Legacy artifacts from old crate name\ngamlss_rs*\n' > pkg/.gitignore

ci: fmt clippy test build-native
