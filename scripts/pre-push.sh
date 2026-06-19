#!/usr/bin/env bash
# Mirror of .github/workflows/rust.yml blocking jobs. Runs locally before push.
#
# Install once per clone:
#   ln -sf ../../scripts/pre-push.sh .git/hooks/pre-push
#
# Skip individual stages with env vars:
#   SKIP_FMT=1 SKIP_CLIPPY=1 SKIP_AUDIT_DUPLICATES=1 SKIP_TEST_OPENBLAS=1 \
#   SKIP_TEST_PURE_RUST=1 SKIP_CHECK_WASM=1 SKIP_TEST_WASM=1 \
#   SKIP_TEST_PYTHON=1 SKIP_DOCTEST=1 SKIP_COVERAGE=1 git push
#
# Bypass entirely:
#   git push --no-verify

set -euo pipefail

# Match CI's coverage floor.
COVERAGE_THRESHOLD="${COVERAGE_THRESHOLD:-92}"

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

red()    { printf '\033[31m%s\033[0m\n' "$*"; }
green()  { printf '\033[32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }
bold()   { printf '\033[1m%s\033[0m\n' "$*"; }

stage() {
  local name="$1"; shift
  local upper; upper="$(printf '%s' "${name//-/_}" | tr '[:lower:]' '[:upper:]')"
  local skip_var="SKIP_${upper}"
  if [[ -n "${!skip_var:-}" ]]; then
    yellow "▸ skip   $name (${skip_var}=${!skip_var})"
    return 0
  fi
  bold   "▸ run    $name"
  local start; start=$(date +%s)
  if "$@"; then
    local end; end=$(date +%s)
    green  "  ok     $name (${name} took $((end - start))s)"
  else
    red    "  FAIL   $name — fix or rerun with ${skip_var}=1 to skip; --no-verify to bypass all"
    exit 1
  fi
}

require() {
  if ! command -v "$1" >/dev/null 2>&1; then
    red "missing required tool: $1"
    red "install or skip the relevant stage with its SKIP_* env var"
    exit 1
  fi
}

bold "pre-push: mirroring CI (rust.yml). Set SKIP_<JOB>=1 to skip a stage."

# --- fmt ---------------------------------------------------------------------
stage fmt cargo fmt --check

# --- clippy (workspace, deny warnings) ---------------------------------------
stage clippy cargo clippy --workspace -- -D warnings

# --- audit-duplicates (fail on new transitive duplicates) --------------------
stage audit-duplicates ./scripts/audit-duplicates.sh

# --- test-openblas (default features) ----------------------------------------
stage test-openblas cargo test --verbose

# --- test-pure-rust ----------------------------------------------------------
stage test-pure-rust cargo test --no-default-features --features pure-rust --verbose

# --- check-wasm --------------------------------------------------------------
check_wasm() {
  require cargo
  if ! rustup target list --installed | grep -q '^wasm32-unknown-unknown$'; then
    red "wasm32-unknown-unknown target missing. Install: rustup target add wasm32-unknown-unknown"
    return 1
  fi
  cargo check --target wasm32-unknown-unknown --no-default-features --features wasm
}
stage check-wasm check_wasm

# --- test-wasm (wasm-pack test --node) ---------------------------------------
test_wasm() {
  require wasm-pack
  require node
  wasm-pack test --node --no-default-features --features wasm --test wasm
}
stage test-wasm test_wasm

# --- test-python (maturin develop + pytest, isolated uv venv) --------------
# Uses `maturin develop` instead of `build`+`install` so we skip wheel-repair
# (which fails on macOS for the openblas → libquadmath chain). CI on Linux
# uses `maturin build --out dist` because audited wheels are needed there.
test_python() {
  require maturin
  require uv
  local venv="${repo_root}/.venv-prepush"
  if [[ ! -x "${venv}/bin/python" ]]; then
    uv venv "$venv" --quiet
  fi
  uv pip install --quiet --python "${venv}/bin/python" pytest numpy
  VIRTUAL_ENV="$venv" maturin develop --release --features python
  "${venv}/bin/pytest" python/tests -q
}
stage test-python test_python

# --- doctest -----------------------------------------------------------------
stage doctest cargo test --doc --features openblas,parallel,serialization

# --- coverage (line-coverage floor) ------------------------------------------
coverage() {
  require cargo-llvm-cov
  cargo llvm-cov --workspace \
    --features openblas,parallel,serialization \
    --ignore-filename-regex '(wasm|python)\.rs$|benchmark/' \
    --fail-under-lines "$COVERAGE_THRESHOLD"
}
stage coverage coverage

green "all CI-blocking stages passed"
