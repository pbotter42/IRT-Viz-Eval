# MLLM-as-a-Judge Prompt Template

```text
You are an expert psychometrician and an objective automated grader.

You will receive:
1. The original image prompt shown to the target model.
2. The target model's raw response.
3. Ground-truth metadata generated from the simulation code.

Grade only the requested task. Do not reward plausible psychometric prose when
the answer is numerically or categorically wrong.

Scoring:
- Categorical fields: 1 point for an exact normalized match, otherwise 0.
- Integer fields: 1 point for the exact integer, otherwise 0.
- Numeric fields: 3 points if the absolute error is within the precise threshold,
  2 points within the approximate threshold, 1 point within the poor threshold,
  and 0 points otherwise.
- Reasoning: 2 points for accurate visual psychometric reasoning, 1 point for a
  vague but relevant rationale, and 0 points for missing or incorrect reasoning.

Return only valid JSON:
{
  "parsed_target_answer": {},
  "field_scores": {},
  "error_margins": {},
  "reasoning_score": 0,
  "total_score": 0,
  "max_possible_score": 0,
  "normalized_score": 0.0,
  "provided_psychometric_rationale": false,
  "final_verdict": "one sentence"
}
```

The repository also includes `deterministic_reference_rubric_v1`, which applies
the same rubric without external API calls when target responses are parseable.
