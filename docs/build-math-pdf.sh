#!/usr/bin/env bash
# Build docs/mathematics.pdf from docs/mathematics.md via pandoc + xelatex.
# xelatex is used (not pdflatex) so UTF-8 math glyphs (∈, ⊗, ψ, etc.) render
# without explicit symbol macros.
set -euo pipefail
cd "$(dirname "$0")"

pandoc mathematics.md \
  --from=markdown+tex_math_dollars+pipe_tables+raw_tex \
  --pdf-engine=xelatex \
  --toc --toc-depth=2 \
  --number-sections \
  -V geometry:margin=1in \
  -V documentclass=article \
  -V colorlinks=true \
  -V linkcolor=blue \
  -V urlcolor=blue \
  -o mathematics.pdf

echo "Wrote $(pwd)/mathematics.pdf"
