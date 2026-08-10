#!/usr/bin/env bash
set -euo pipefail

export SOURCE_DATE_EPOCH=1786320000

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$ROOT"
PYTHONPATH=src "$PYTHON_BIN" paper/prepare_empirical_results.py
PYTHONPATH=src "$PYTHON_BIN" paper/generate_supplement.py
PYTHONPATH=src "$PYTHON_BIN" paper/generate_figures.py \
  --grayscale --output-dir paper/figures

(
  cd paper
  pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
  bibtex manuscript
  pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
  pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
  pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
  pdflatex -interaction=nonstopmode -halt-on-error online_supplement.tex
  pdflatex -interaction=nonstopmode -halt-on-error online_supplement.tex
)

"$PYTHON_BIN" paper/combine_preprint.py

PAGES="$(pdfinfo output/pdf/irt_viz_eval_preprint.pdf | awk '/^Pages:/ {print $2}')"
if [ "$PAGES" != "48" ]; then
  echo "Preprint page check failed: ${PAGES:-unknown} pages" >&2
  exit 1
fi

echo "Built output/pdf/irt_viz_eval_preprint.pdf ($PAGES pages)"
