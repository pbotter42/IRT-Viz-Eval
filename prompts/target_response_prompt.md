# Target Model Prompt Template

Use each row in `data/benchmark/tasks.jsonl`.

Inputs:

- `image_path`: local PNG stimulus.
- `prompt`: the task-specific question.
- `prompt_template_id`: stable identifier for the prompt wording.

Target model instruction:

```text
You are interpreting a psychometric figure. Answer only the requested task.
Use the chart visually; do not assume the values are provided elsewhere.
Return compact JSON using exactly the keys requested in the prompt. Include a
short `rationale` field naming the visual evidence you used.
```

Record each response as one JSONL object matching `schemas/response.schema.json`.
