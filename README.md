# IRT Visual Interpretation Benchmark

This repository implements a reproducible benchmark for evaluating how well
multimodal AI systems interpret Item Response Theory (IRT) and psychometric
figures. It was built from the project idea in `idea.pdf`.

The benchmark generates simulated ground truth, renders figures under multiple
visual styles, asks vision models psychometric interpretation questions, scores
responses against known answers, and aggregates model performance.

## What Is Included

- Dichotomous IRT curves: 1PL, 2PL, 3PL, and 4PL ICCs.
- Information curves: item information functions and test information functions.
- Test-level curves: test characteristic curves.
- Polytomous models: PCM, GPCM, and GRM category response curves.
- Non-monotonic curves: transparent ideal-point response curves for unfolding-style stress tests.
- Adversarial visual styles: matplotlib, seaborn, ggplot-like, grayscale/no-grid, dark/high-contrast, and lattice-like renderings.
- Metadata schemas for stimuli, tasks, target responses, and judgments.
- Offline baselines and deterministic scoring so the full pipeline runs without API keys.
- A resumable OpenAI Responses API adapter for real vision-model administrations.
- Prompt templates for real target VLM runs and MLLM-as-a-judge scoring.
- A CEJEME-formatted methodological tutorial for social scientists, using IRT-Viz-Eval
  as a worked example of LLM benchmark construction.
- Twelve reproducible process diagrams and result figures for the manuscript.

## Quickstart

```bash
python3 -m pip install -r requirements.txt
PYTHONPATH=src python3 -m irt_viz_eval.cli all
```

Or run:

```bash
bash scripts/run_local_benchmark.sh
```

Outputs:

- `data/benchmark/manifest.json`: benchmark run metadata.
- `data/benchmark/stimuli.jsonl`: rendered figures and ground truth.
- `data/benchmark/tasks.jsonl`: prompts and expected answers.
- `data/benchmark/responses.jsonl`: local baseline responses.
- `data/benchmark/judgments.jsonl`: rubric scores.
- `output/analysis/`: summary tables and figures.

## Running Real Vision Models

Real model runs use API model identifiers, not consumer labels such as
"ChatGPT." Keep diagnostic and real responses in separate files. Install the
optional adapter and set the API key in your shell:

```bash
python3 -m pip install -e '.[openai]'
export OPENAI_API_KEY='your-key-here'
```

Start with a small paid pilot using model identifiers available to your API
account:

```bash
PYTHONPATH=src python3 -m irt_viz_eval.cli run-openai \
  --models MODEL_ID_1,MODEL_ID_2 \
  --limit 10 \
  --repetitions 1
```

The default output is `data/benchmark/openai_responses.jsonl`. The runner stores
requested and returned model identifiers, timestamps, image detail, latency,
token usage, repetitions, raw output, and failures. It resumes completed calls
by default. Check cost and output quality before removing `--limit`; then use at
least three repetitions if the study will make claims about response stability.

Score and analyze the real responses separately:

```bash
PYTHONPATH=src python3 -m irt_viz_eval.cli judge \
  --responses data/benchmark/openai_responses.jsonl \
  --output data/benchmark/openai_judgments.jsonl
PYTHONPATH=src python3 -m irt_viz_eval.cli analyze \
  --judgments data/benchmark/openai_judgments.jsonl \
  --output output/openai_analysis
```

The included deterministic judge scores parseable JSON outputs. Failed calls
remain in the denominator as zero-scored responses unless the analysis protocol
prespecifies another rule. For free-form outputs or a publication-grade automated review, use
`prompts/judge_system_prompt.md` with a frontier multimodal judge and store the
judge result using `schemas/judgment.schema.json`.

The same `tasks.jsonl` and response schema can support additional provider
adapters. Do not commit API keys or unreviewed paid-run outputs.

## Dataset Design

Each simulated mathematical stimulus is rendered under several style profiles.
This separates psychometric understanding from superficial chart-template
recognition. Metadata records include the IRT model, parameters, monotonicity,
curve family, plotting engine/style, axes, file paths, and exact values at
selected theta points.

## Paper

The manuscript introduces how LLM benchmarks are designed, how they differ
from educational and psychological assessments, and how validity, prompting,
scoring, contamination, uncertainty, and maintenance should be handled. It then
uses IRT-Viz-Eval as a complete worked example. The source is in
`paper/manuscript.tex`; references are in `paper/references.bib`. Primary-source
PDFs consulted during development remain local and are not redistributed.

The submitted main document is anonymized for CEJEME double-blind review. The
downloaded literature directory and submission correspondence are intentionally
excluded from GitHub.

Regenerate all manuscript figures and compile from the repository root with:

```bash
python3 paper/generate_figures.py
latexmk -pdf -cd paper/manuscript.tex
```

The final delivery PDF is written to
`output/pdf/irt_viz_eval_manuscript.pdf`.

## Testing

```bash
python3 -m pip install -e '.[test]'
pytest
```

Tests use a fake API client and do not spend API credits.

## Citation and License

Citation metadata are provided in `CITATION.cff`. Source code is released under
the MIT License. The manuscript is not covered by the software license and
remains subject to the target journal's publication agreement.
