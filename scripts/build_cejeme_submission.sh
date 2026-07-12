#!/usr/bin/env bash
set -euo pipefail

export SOURCE_DATE_EPOCH=1783728000

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/output/submission"
SOURCE_DIR="$OUT/CEJEME_LaTeX_Source"
SUPPLEMENT_DIR="$OUT/CEJEME_Anonymous_Code_Supplement"

cd "$ROOT"
python3 paper/prepare_empirical_results.py
python3 paper/generate_figures.py
latexmk -pdf -cd paper/manuscript.tex

rm -rf "$SOURCE_DIR"
mkdir -p "$SOURCE_DIR/figures"
cp paper/manuscript.tex "$SOURCE_DIR/"
cp paper/references.bib "$SOURCE_DIR/"
cp paper/figures/*.pdf "$SOURCE_DIR/figures/"
cp paper/manuscript.pdf "$OUT/CEJEME_Main_Document.pdf"

rm -f "$OUT/CEJEME_LaTeX_Source.zip"
(
  cd "$OUT"
  zip -qr CEJEME_LaTeX_Source.zip CEJEME_LaTeX_Source
)

rm -rf "$SUPPLEMENT_DIR"
mkdir -p "$SUPPLEMENT_DIR/paper" "$SUPPLEMENT_DIR/scripts" "$SUPPLEMENT_DIR/output"
cp -R src data schemas prompts tests "$SUPPLEMENT_DIR/"
cp -R output/analysis "$SUPPLEMENT_DIR/output/"
cp -R output/empirical_analysis "$SUPPLEMENT_DIR/output/"
cp paper/prepare_empirical_results.py "$SUPPLEMENT_DIR/paper/"
cp paper/generate_figures.py "$SUPPLEMENT_DIR/paper/"
cp scripts/run_local_benchmark.sh "$SUPPLEMENT_DIR/scripts/"
cp pyproject.toml requirements.txt requirements-openai.txt requirements-huggingface.txt "$SUPPLEMENT_DIR/"
cp submission/ANONYMOUS_SUPPLEMENT_README.md "$SUPPLEMENT_DIR/README.md"
find "$SUPPLEMENT_DIR" -type d -name __pycache__ -prune -exec rm -rf {} +
find "$SUPPLEMENT_DIR" -type f -name '*.pyc' -delete

rm -f "$OUT/CEJEME_Anonymous_Code_Supplement.zip"
(
  cd "$OUT"
  zip -qr CEJEME_Anonymous_Code_Supplement.zip CEJEME_Anonymous_Code_Supplement
)

PAGES="$(pdfinfo "$OUT/CEJEME_Main_Document.pdf" | awk '/^Pages:/ {print $2}')"
if [ -z "$PAGES" ] || [ "$PAGES" -gt 40 ]; then
  echo "CEJEME page-limit check failed: ${PAGES:-unknown} pages" >&2
  exit 1
fi

cp "$OUT/CEJEME_Main_Document.pdf" output/pdf/irt_viz_eval_manuscript.pdf
echo "Built CEJEME manuscript, LaTeX source, and anonymous code supplement ($PAGES pages) in $OUT"
