.PHONY: fmt clippy test build ci

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

ci: fmt clippy test build-native
