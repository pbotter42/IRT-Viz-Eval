"""Append the online supplement to the compiled manuscript."""

from __future__ import annotations

import subprocess
from pathlib import Path

from pypdf import PdfReader, PdfWriter


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUTPUT = ROOT / "output" / "pdf" / "irt_viz_eval_preprint.pdf"
TEMP = ROOT / "output" / "pdf" / ".preprint_without_metadata.pdf"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["pdfunite", str(PAPER / "manuscript.pdf"), str(PAPER / "online_supplement.pdf"), str(TEMP)],
        check=True,
    )

    writer = PdfWriter()
    writer.clone_document_from_reader(PdfReader(TEMP))
    writer.add_metadata(
        {
            "/Title": "How Large Language Model Benchmarks Are Built: A Measurement-Oriented Tutorial and Empirical Demonstration with IRT-Viz-Eval",
            "/Author": "Preston Botter",
            "/Subject": "LLM benchmark development for social-science and measurement audiences",
            "/Keywords": "large language models, benchmark design, validity, psychological measurement, item response theory",
        }
    )
    with OUTPUT.open("wb") as stream:
        writer.write(stream)
    TEMP.unlink()

    page_count = len(PdfReader(OUTPUT).pages)
    if page_count != 48:
        raise RuntimeError(f"Expected 48 pages, found {page_count}")
    print(f"Wrote {OUTPUT} ({page_count} pages)")


if __name__ == "__main__":
    main()
