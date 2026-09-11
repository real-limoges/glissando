#!/usr/bin/env bash
# Audit Cargo dependency duplication.
#
# `cargo tree --duplicates` lists every crate compiled at more than one version.
# The remaining duplication is the rand/getrandom chain driven by upstream version
# pins: statrs 0.18 -> rand 0.8 -> getrandom 0.2, our direct rand 0.10 ->
# getrandom 0.4, and the proptest dev-dependency -> rand 0.9 -> getrandom 0.3.
# This script enforces that only the documented allowlist appears, so new
# duplicates fail the audit and get investigated before merge.
#
# Run locally:
#   ./scripts/audit-duplicates.sh
#
# Exit 0 ⇒ no new duplicates. Exit 1 ⇒ unexpected duplicates; check the output.

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

# Crates known to coexist at multiple versions. See the Cargo.toml comment for the
# `rand`/`getrandom` split; the transitive duplicates (rand_core, rand_chacha,
# rand_distr) follow mechanically from those root duplications.
#
# Platform-conditional entries only appear on Linux CI runners (not on the macOS
# dev tree), pulled in via dev/test infrastructure crates that aren't under our
# control:
#   - `rustix` — used by `tempfile`, `is-terminal`, `cargo-llvm-cov`, and other
#     coverage/test scaffolding. Long-standing dual-version coexistence in the
#     Rust ecosystem; benign for our purposes.
ALLOWLIST=(
  getrandom
  rand
  rand_chacha
  rand_core
  rand_distr
  rustix
)

# Extract the set of duplicated crate names from `cargo tree --duplicates`.
# Top-level entries look like `name vX.Y.Z`; transitive lines start with tree
# characters (└──, ├──, etc.) which begin with whitespace or punctuation.
duplicates=$(
  cargo tree --duplicates 2>/dev/null \
    | grep -E '^[a-zA-Z]' \
    | awk '{print $1}' \
    | sort -u
)

if [[ -z "$duplicates" ]]; then
  echo "audit-duplicates: no duplicate dependencies"
  exit 0
fi

unexpected=()
while IFS= read -r crate; do
  if ! printf '%s\n' "${ALLOWLIST[@]}" | grep -qx "$crate"; then
    unexpected+=("$crate")
  fi
done <<< "$duplicates"

if (( ${#unexpected[@]} == 0 )); then
  echo "audit-duplicates: all duplicates are allowlisted"
  printf '  - %s\n' "${ALLOWLIST[@]}" | sort
  exit 0
fi

echo "audit-duplicates: FAIL — new duplicate dependencies detected:"
printf '  - %s\n' "${unexpected[@]}"
echo
echo "If the new duplicate is unavoidable (upstream pin), document it in"
echo "Cargo.toml and add it to ALLOWLIST in scripts/audit-duplicates.sh."
echo "Otherwise, consolidate the dep so cargo tree --duplicates is clean."
exit 1
