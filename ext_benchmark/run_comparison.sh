#!/bin/bash
#
# GAMLSS Comparison Runner
#
# Builds Rust, generates data, runs both implementations, and produces comparison.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output"
N_OBS=${N_OBS:-1000}
SEED=${SEED:-42}

echo "========================================"
echo "GAMLSS Comparison Framework"
echo "========================================"
echo "Output directory: ${OUTPUT_DIR}"
echo "Observations: ${N_OBS}"
echo "Seed: ${SEED}"
echo ""

# Step 1: Build Rust binary
echo "[1/4] Building Rust binary..."
cd "${SCRIPT_DIR}"
if cargo build --release --bin compare_fit 2>/dev/null; then
    RUST_BINARY="${SCRIPT_DIR}/target/release/compare_fit"
    echo "      Built: ${RUST_BINARY}"
else
    echo "      Warning: Rust build failed. Will skip Rust fitting."
    RUST_BINARY=""
fi
cd "${SCRIPT_DIR}"

# Step 2: Check R dependencies
echo "[2/4] Checking R dependencies..."
if Rscript -e 'library(arrow); library(mgcv); library(jsonlite)' 2>/dev/null; then
    R_SCRIPT="${SCRIPT_DIR}/fit_mgcv.R"
    echo "      R ready: ${R_SCRIPT}"
else
    echo "      Warning: R dependencies missing. Will skip R fitting."
    R_SCRIPT=""
fi

# Step 3: Generate test data
echo "[3/4] Generating test data..."
mkdir -p "${OUTPUT_DIR}"
python3 "${SCRIPT_DIR}/orchestrate.py" \
    --generate-only \
    --output-dir "${OUTPUT_DIR}" \
    --n-obs "${N_OBS}" \
    --seed "${SEED}"

# Step 4: Run comparison
echo "[4/4] Running comparison..."

ARGS="--output-dir ${OUTPUT_DIR} --n-obs ${N_OBS} --seed ${SEED}"

if [ -n "${R_SCRIPT}" ]; then
    ARGS="${ARGS} --r-script ${R_SCRIPT}"
fi

if [ -n "${RUST_BINARY}" ]; then
    ARGS="${ARGS} --rust-binary ${RUST_BINARY}"
fi

python3 "${SCRIPT_DIR}/orchestrate.py" ${ARGS}

echo ""
echo "========================================"
echo "Comparison complete!"
echo "Results: ${OUTPUT_DIR}/comparison_summary.json"
echo "========================================"
