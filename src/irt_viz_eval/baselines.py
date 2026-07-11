"""Offline target-model baselines for pipeline testing."""

from __future__ import annotations

import json
import time
from typing import Any, Mapping

import numpy as np

from .io import read_jsonl, write_jsonl


def _answer_from_expected(task: Mapping[str, Any], rng: np.random.Generator, mode: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, spec in task["expected_answers"].items():
        if spec["type"] == "numeric":
            value = float(spec["value"])
            if mode == "oracle":
                payload[key] = round(value, 4)
            elif mode == "noisy":
                scale = float(spec["thresholds"][1])
                payload[key] = round(value + rng.normal(0.0, scale * 0.55), 4)
            else:
                payload[key] = round(value + rng.normal(0.0, max(abs(value), 1.0) * 0.6), 4)
        elif spec["type"] == "integer":
            value = int(spec["value"])
            if mode in {"oracle", "noisy"}:
                payload[key] = value
            else:
                payload[key] = max(0, value + int(rng.choice([-2, -1, 1, 2])))
        elif spec["type"] == "categorical":
            value = str(spec["value"])
            if mode in {"oracle", "noisy"}:
                payload[key] = value
            else:
                payload[key] = "2PL" if value != "2PL" else "3PL"
    if mode == "oracle":
        payload["rationale"] = "Used the visible slope, asymptotes, category crossings, and peak locations to read the chart."
    elif mode == "noisy":
        payload["rationale"] = "The visual pattern suggests this answer, though the estimate is approximate from the axes."
    else:
        payload["rationale"] = "The chart appears to be a generic line plot with an increasing trend."
    return payload


def build_baseline_responses(
    tasks_path: str,
    output_path: str,
    model_names: list[str] | None = None,
    seed: int = 20260711,
) -> int:
    model_names = model_names or ["oracle", "noisy", "blind"]
    rng = np.random.default_rng(seed)
    rows = []
    for task in read_jsonl(tasks_path):
        for model_name in model_names:
            started = time.perf_counter()
            payload = _answer_from_expected(task, rng, model_name)
            latency_ms = int((time.perf_counter() - started) * 1000)
            rows.append(
                {
                    "response_id": f"{task['task_id']}__{model_name}",
                    "task_id": task["task_id"],
                    "stimulus_id": task["stimulus_id"],
                    "target_model_name": f"local_{model_name}_baseline",
                    "target_model_family": "local_baseline",
                    "prompt_template_id": task["prompt_template_id"],
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "inference_latency_ms": latency_ms,
                    "target_raw_output": json.dumps(payload, sort_keys=True),
                }
            )
    write_jsonl(output_path, rows)
    return len(rows)
