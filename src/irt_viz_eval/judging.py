"""Deterministic and prompt-based judging utilities."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from .io import read_jsonl, write_jsonl


REASONING_TERMS = {
    "asymptote",
    "slope",
    "inflection",
    "threshold",
    "category",
    "intersection",
    "peak",
    "information",
    "expected",
    "theta",
    "probability",
    "non-monotonic",
    "non_monotonic",
}


def normalize_label(value: Any) -> str:
    text = str(value).lower().strip()
    text = text.replace("-", "_").replace(" ", "_")
    aliases = {
        "rasch": "1pl",
        "partial_credit_model": "pcm",
        "generalized_partial_credit_model": "gpcm",
        "graded_response_model": "grm",
        "idealpoint": "idealpoint",
        "ideal_point": "idealpoint",
        "mixeddichotomousform": "mixeddichotomousform",
        "mixed_dichotomous_form": "mixeddichotomousform",
    }
    return aliases.get(text, text)


def extract_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    return {}


def score_numeric(observed: Any, expected: float, thresholds: list[float]) -> tuple[int, float | None]:
    try:
        value = float(observed)
    except (TypeError, ValueError):
        return 0, None
    error = abs(value - expected)
    if error <= thresholds[0]:
        return 3, error
    if error <= thresholds[1]:
        return 2, error
    if error <= thresholds[2]:
        return 1, error
    return 0, error


def score_reasoning(parsed: Mapping[str, Any], raw_text: str) -> int:
    rationale = str(parsed.get("rationale", raw_text)).lower()
    hits = sum(1 for term in REASONING_TERMS if term in rationale)
    if hits >= 2:
        return 2
    if hits == 1 or len(rationale.split()) >= 8:
        return 1
    return 0


def judge_one(task: Mapping[str, Any], response: Mapping[str, Any]) -> dict[str, Any]:
    parsed = extract_json_object(response["target_raw_output"])
    field_scores: dict[str, Any] = {}
    errors: dict[str, Any] = {}
    total = 0
    max_possible = 0

    for key, spec in task["expected_answers"].items():
        observed = parsed.get(key)
        if spec["type"] == "numeric":
            score, error = score_numeric(observed, float(spec["value"]), list(spec["thresholds"]))
            field_scores[key] = score
            errors[key] = None if error is None else round(float(error), 4)
            total += score
            max_possible += 3
        elif spec["type"] == "integer":
            score = int(observed == spec["value"])
            field_scores[key] = score
            errors[key] = None if observed is None else int(observed) - int(spec["value"]) if str(observed).lstrip("-").isdigit() else None
            total += score
            max_possible += 1
        elif spec["type"] == "categorical":
            score = int(normalize_label(observed) == normalize_label(spec["value"]))
            field_scores[key] = score
            errors[key] = {"observed": observed, "expected": spec["value"]}
            total += score
            max_possible += 1

    reasoning_score = score_reasoning(parsed, response["target_raw_output"])
    total += reasoning_score
    max_possible += 2

    return {
        "judgment_id": f"{response['response_id']}__deterministic_judge",
        "response_id": response["response_id"],
        "task_id": task["task_id"],
        "stimulus_id": task["stimulus_id"],
        "target_model_name": response["target_model_name"],
        "judge_model_name": "deterministic_reference_rubric_v1",
        "scores": {
            "field_scores": field_scores,
            "reasoning_score": reasoning_score,
        },
        "error_margins": errors,
        "total_score": total,
        "max_possible_score": max_possible,
        "normalized_score": round(total / max_possible, 4) if max_possible else None,
        "provided_psychometric_rationale": reasoning_score > 0,
        "judge_reasoning": "Parsed target JSON, compared fields to stored ground truth, and applied fixed rubric thresholds.",
    }


def judge_responses(tasks_path: str, responses_path: str, output_path: str) -> int:
    tasks = {task["task_id"]: task for task in read_jsonl(tasks_path)}
    rows = []
    for response in read_jsonl(responses_path):
        task = tasks[response["task_id"]]
        rows.append(judge_one(task, response))
    write_jsonl(output_path, rows)
    return len(rows)
