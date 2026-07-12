# Empirical Model Responses

This directory contains the deduplicated response records used in the CEJEME
manuscript. Each file contains one response for every one of the 198 frozen
IRT-Viz-Eval tasks.

- `glm45v_responses.jsonl`: `zai-org/GLM-4.5V:novita`, routed through Hugging
  Face Inference Providers and served by Novita.
- `gemma3_4b_responses.jsonl`: `google/gemma-3-4b-it:deepinfra`, routed through
  Hugging Face Inference Providers and served by DeepInfra.

The records contain model identifiers, task identifiers, timestamps, latency,
provider-reported token usage, and raw model output. They contain no API keys or
authentication tokens. Generate judgments, tables, confidence intervals, and
manuscript figures with:

```bash
PYTHONPATH=src python3 paper/prepare_empirical_results.py
python3 paper/generate_figures.py
```

The model outputs are provided as research artifacts. Model-specific terms may
also apply to downstream reuse.
