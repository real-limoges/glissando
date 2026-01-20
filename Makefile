.PHONY: fmt clippy test build ci

fmt:
	cargo fmt

clippy:
	cargo clippy -- -D warnings

test:
	cargo test

build:
	cargo build

ci: fmt clippy test build
