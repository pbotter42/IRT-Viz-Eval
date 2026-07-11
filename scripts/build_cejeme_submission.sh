#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/output/submission"
SOURCE_DIR="$OUT/CEJEME_LaTeX_Source"

cd "$ROOT"
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

PAGES="$(pdfinfo "$OUT/CEJEME_Main_Document.pdf" | awk '/^Pages:/ {print $2}')"
if [ -z "$PAGES" ] || [ "$PAGES" -gt 40 ]; then
  echo "CEJEME page-limit check failed: ${PAGES:-unknown} pages" >&2
  exit 1
fi

cp "$OUT/CEJEME_Main_Document.pdf" output/pdf/irt_viz_eval_manuscript.pdf
echo "Built CEJEME submission package ($PAGES pages) in $OUT"
