# IRT-Viz-Eval

IRT-Viz-Eval tests how vision-language models read common Item Response Theory
(IRT) plots. The benchmark uses generated figures with known parameters, so
model answers can be scored against stored ground truth rather than subjective
labels.

The current release contains 14 mathematical base stimuli, six rendering
profiles, 84 images, and 198 scored tasks. Five complete model administrations
are included.

## Included model runs

| Model route | Provider | Tasks | Mean score |
|---|---|---:|---:|
| `Qwen/Qwen2.5-VL-72B-Instruct:ovhcloud` | OVHcloud | 198 | .719 |
| `Qwen/Qwen3-VL-30B-A3B-Instruct:novita` | Novita | 198 | .704 |
| `google/gemma-3-4b-it:deepinfra` | DeepInfra | 198 | .629 |
| `zai-org/GLM-4.5V:novita` | Novita | 198 | .628 |
| `google/gemma-3-12b-it:deepinfra` | DeepInfra | 198 | .600 |

These are task-normalized scores for the recorded model, provider, prompt, and
scoring configuration. They are not estimates of a general or human-like
ability.

## Quick start

IRT-Viz-Eval requires Python 3.10 or newer.

```bash
python3 -m pip install -e '.[test]'
PYTHONPATH=src python3 -m irt_viz_eval.cli all
```

The command generates the benchmark, runs three local diagnostic baselines,
scores them, and writes summaries to `output/analysis/`. No API key is needed.

## Reproduce the reported analysis

The five deduplicated response files are in `data/empirical/`. Regenerate the
judgments, bootstrap intervals, tables, and figures with:

```bash
PYTHONPATH=src python3 paper/prepare_empirical_results.py
PYTHONPATH=src python3 paper/generate_supplement.py
PYTHONPATH=src python3 paper/generate_figures.py \
  --grayscale --output-dir paper/figures
```

The reported study uses the deterministic reference rubric. It does not use an
LLM as a judge. Empty, malformed, or missing requested fields remain in the
denominator and receive no credit for the affected fields.

## Run another Hugging Face model

Authenticate locally and install the provider adapter:

```bash
python3 -m pip install -e '.[huggingface]'
hf auth login
```

Start with ten tasks and a new output file:

```bash
PYTHONPATH=src python3 -m irt_viz_eval.cli run-huggingface \
  --models MODEL_ID:PROVIDER \
  --limit 10 \
  --request-timeout 180 \
  --max-retries 3 \
  --output data/benchmark/model_pilot.jsonl
```

If the pilot is sound, rerun without `--limit` and keep the same output path.
Completed response identifiers are skipped, so an interrupted run resumes
instead of starting over.

Score and summarize the completed file with:

```bash
PYTHONPATH=src python3 -m irt_viz_eval.cli judge \
  --responses data/benchmark/model_pilot.jsonl \
  --output data/benchmark/model_judgments.jsonl
PYTHONPATH=src python3 -m irt_viz_eval.cli analyze \
  --judgments data/benchmark/model_judgments.jsonl \
  --output output/model_analysis
```

An OpenAI Responses API adapter is also available through the `openai` optional
dependency. Consumer ChatGPT or Gemini subscriptions are not API credentials
and do not define a reproducible administration condition.

## Benchmark contents

The task set covers:

- 1PL, 2PL, 3PL, and 4PL item characteristic curves;
- item and test information functions;
- test characteristic curves;
- PCM, GPCM, and GRM category response curves; and
- a transparent ideal-point curve used as a non-monotonic stress test.

Each mathematical object is rendered in six styles. The response and judgment
schemas retain model identifiers, provider routes, timestamps, latency, token
usage, raw output, parser results, field-level scores, and scoring provenance.

## Repository map

- `src/irt_viz_eval/`: generation, rendering, provider adapters, scoring, and analysis
- `data/benchmark/`: frozen stimuli, images, tasks, and diagnostic baselines
- `data/empirical/`: five complete model response files
- `output/empirical_analysis/`: published judgments and result tables
- `schemas/`: JSON schemas for benchmark records
- `prompts/`: target-response and optional judge prompt documentation
- `paper/`: preprint source, supplement, tables, and figure scripts
- `tests/`: provider-adapter tests using fake clients

## Build the paper

Install the paper dependency and run the build script:

```bash
python3 -m pip install -r requirements-paper.txt
bash scripts/build_preprint.sh
```

The script creates the grayscale 49-page manuscript and appended supplement at
`output/pdf/irt_viz_eval_preprint.pdf`.

## Scope

The benchmark samples clean synthetic plots. It does not establish that a model
can independently review operational psychometric analyses, and it has not been
validated as a latent-trait scale for AI systems. The raw records are included
so task profiles, parsing failures, and aggregation choices can be examined
instead of relying only on the overall ranking.

## Citation and license

Citation metadata are in `CITATION.cff`. Source code is released under the MIT
License. The manuscript and its figures are not covered by the software license.
