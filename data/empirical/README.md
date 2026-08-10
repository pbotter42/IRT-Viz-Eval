# Empirical model responses

Each JSONL file contains one deduplicated response for every one of the 198
frozen IRT-Viz-Eval tasks.

| File | Requested route |
|---|---|
| `glm45v_responses.jsonl` | `zai-org/GLM-4.5V:novita` |
| `gemma3_4b_responses.jsonl` | `google/gemma-3-4b-it:deepinfra` |
| `gemma3_12b_responses.jsonl` | `google/gemma-3-12b-it:deepinfra` |
| `qwen3_vl_30b_responses.jsonl` | `Qwen/Qwen3-VL-30B-A3B-Instruct:novita` |
| `qwen2_5_vl_72b_responses.jsonl` | `Qwen/Qwen2.5-VL-72B-Instruct:ovhcloud` |

The records include model and task identifiers, provider routes, timestamps,
latency, provider-reported token usage, raw model output, and parser status. No
API keys or authentication tokens are present.

Regenerate the published analysis from the repository root:

```bash
PYTHONPATH=src python3 paper/prepare_empirical_results.py
PYTHONPATH=src python3 paper/generate_supplement.py
PYTHONPATH=src python3 paper/generate_figures.py \
  --grayscale --output-dir paper/figures
```

Model outputs are provided as research artifacts. Model- and provider-specific
terms may also apply to downstream reuse.
