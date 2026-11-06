.PHONY: fmt clippy test build build-native ci check-wasm build-wasm hooks

hooks:
	ln -sf ../../scripts/pre-push.sh .git/hooks/pre-push
	@echo "pre-push hook activated (scripts/pre-push.sh)"

fmt:
	cargo fmt

clippy:
	cargo clippy --workspace --all-targets -- -D warnings

test:
	cargo test

build:
	cargo build

build-native:
	RUSTFLAGS="-C target-cpu=native" cargo build --release

check-wasm:
	cargo check --target wasm32-unknown-unknown --no-default-features --features wasm

# wasm-pack overwrites the hand-maintained pkg/README.md and pkg/.gitignore;
# restore both from git after every build.
build-wasm:
	wasm-pack build --no-default-features --features wasm
	git checkout -- pkg/README.md pkg/.gitignore

ci: fmt clippy test build-native
