"""Prepare the complete real-model results used in the preprint."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from irt_viz_eval.analysis import analyze
from irt_viz_eval.io import read_jsonl, write_jsonl
from irt_viz_eval.judging import extract_json_object, judge_one


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "benchmark"
EMPIRICAL_DATA = ROOT / "data" / "empirical"
OUT = ROOT / "output" / "empirical_analysis"
TASKS_PATH = DATA / "tasks.jsonl"
STIMULI_PATH = DATA / "stimuli.jsonl"

MODEL_RUNS = {
    "GLM-4.5V": (
        DATA / "hf_glm45v_full_responses.jsonl",
        EMPIRICAL_DATA / "glm45v_responses.jsonl",
    ),
    "Gemma 3 4B": (
        DATA / "hf_gemma34b_full_responses.jsonl",
        EMPIRICAL_DATA / "gemma3_4b_responses.jsonl",
    ),
    "Gemma 3 12B": (
        DATA / "hf_gemma312b_full_responses.jsonl",
        EMPIRICAL_DATA / "gemma3_12b_responses.jsonl",
    ),
    "Qwen3-VL 30B": (
        DATA / "hf_qwen3vl30b_full_responses.jsonl",
        EMPIRICAL_DATA / "qwen3_vl_30b_responses.jsonl",
    ),
    "Qwen2.5-VL 72B": (
        DATA / "hf_qwen25vl72b_full_responses.jsonl",
        EMPIRICAL_DATA / "qwen2_5_vl_72b_responses.jsonl",
    ),
}


def deduplicate(rows: list[dict]) -> tuple[list[dict], int]:
    """Keep one record per response id, preferring a completed record."""
    selected: dict[str, dict] = {}
    duplicates = 0
    for row in rows:
        key = str(row["response_id"])
        if key in selected:
            duplicates += 1
            if selected[key].get("status") != "completed" and row.get("status") == "completed":
                selected[key] = row
        else:
            selected[key] = row
    return list(selected.values()), duplicates


def cluster_bootstrap_ci(frame: pd.DataFrame, *, seed: int, draws: int = 10_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    clusters = sorted(frame["base_stimulus_id"].unique())
    cluster_values = {key: frame.loc[frame.base_stimulus_id == key, "normalized_score"].to_numpy() for key in clusters}
    estimates = np.empty(draws)
    for index in range(draws):
        sample = rng.choice(clusters, size=len(clusters), replace=True)
        estimates[index] = np.concatenate([cluster_values[key] for key in sample]).mean()
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def paired_cluster_bootstrap(frame: pd.DataFrame, model_a: str, model_b: str, *, seed: int, draws: int = 10_000) -> dict:
    wide = frame.pivot(index=["task_id", "base_stimulus_id"], columns="target_model_name", values="normalized_score").reset_index()
    wide["difference"] = wide[model_a] - wide[model_b]
    rng = np.random.default_rng(seed)
    clusters = sorted(wide["base_stimulus_id"].unique())
    values = {key: wide.loc[wide.base_stimulus_id == key, "difference"].to_numpy() for key in clusters}
    estimates = np.empty(draws)
    for index in range(draws):
        sample = rng.choice(clusters, size=len(clusters), replace=True)
        estimates[index] = np.concatenate([values[key] for key in sample]).mean()
    return {
        "model_a": model_a,
        "model_b": model_b,
        "mean_difference": wide["difference"].mean(),
        "ci_low": np.quantile(estimates, 0.025),
        "ci_high": np.quantile(estimates, 0.975),
    }


def usage_value(row: dict, key: str) -> int:
    usage = row.get("usage") or {}
    value = usage.get(key)
    return int(value) if value is not None else 0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    EMPIRICAL_DATA.mkdir(parents=True, exist_ok=True)
    tasks = {row["task_id"]: row for row in read_jsonl(TASKS_PATH)}
    task_order = {task_id: index for index, task_id in enumerate(tasks)}
    stimuli = {row["stimulus_id"]: row for row in read_jsonl(STIMULI_PATH)}
    expected_task_ids = set(tasks)
    all_judgments: list[dict] = []
    administration_rows: list[dict] = []

    for model_name, (raw_path, published_path) in MODEL_RUNS.items():
        response_path = raw_path if raw_path.exists() else published_path
        if not response_path.exists():
            raise FileNotFoundError(f"Missing response file for {model_name}: {raw_path} or {published_path}")
        raw_rows = list(read_jsonl(response_path))
        rows, duplicate_count = deduplicate(raw_rows)
        coverage = Counter(row["task_id"] for row in rows)
        missing = expected_task_ids - set(coverage)
        repeated_tasks = {task_id: count for task_id, count in coverage.items() if count != 1}
        if missing or repeated_tasks or len(rows) != len(tasks):
            raise RuntimeError(
                f"{model_name} is incomplete: {len(rows)}/{len(tasks)} unique responses, "
                f"{len(missing)} missing tasks, repeated={repeated_tasks}"
            )

        rows.sort(key=lambda row: task_order[row["task_id"]])
        for row in rows:
            row["target_model_name"] = model_name
        write_jsonl(published_path, rows)

        judgments = [judge_one(tasks[row["task_id"]], row) for row in rows]
        write_jsonl(OUT / f"{model_name.lower().replace(' ', '_').replace('.', '')}_judgments.jsonl", judgments)
        all_judgments.extend(judgments)

        completed = [row for row in rows if row.get("status") == "completed"]
        nonempty = [row for row in completed if str(row.get("target_raw_output", "")).strip()]
        parseable = [row for row in nonempty if extract_json_object(str(row.get("target_raw_output", "")))]
        latencies = np.array([row.get("inference_latency_ms") for row in completed if row.get("inference_latency_ms") is not None])
        administration_rows.append(
            {
                "target_model_name": model_name,
                "requested_model": rows[0].get("requested_model"),
                "returned_model": rows[0].get("returned_model", rows[0].get("target_model_name")),
                "n_raw_rows": len(raw_rows),
                "n_unique_tasks": len(rows),
                "duplicates_removed": duplicate_count,
                "n_completed": len(completed),
                "n_api_errors": len(rows) - len(completed),
                "n_nonempty_outputs": len(nonempty),
                "n_parseable_outputs": len(parseable),
                "empty_output_rate": 1 - len(nonempty) / len(rows),
                "parseable_output_rate": len(parseable) / len(rows),
                "median_latency_ms": float(np.median(latencies)),
                "latency_q1_ms": float(np.quantile(latencies, 0.25)),
                "latency_q3_ms": float(np.quantile(latencies, 0.75)),
                "total_prompt_tokens": sum(usage_value(row, "prompt_tokens") for row in completed),
                "total_completion_tokens": sum(usage_value(row, "completion_tokens") for row in completed),
                "total_tokens": sum(usage_value(row, "total_tokens") for row in completed),
            }
        )

    combined_judgments = OUT / "combined_judgments.jsonl"
    write_jsonl(combined_judgments, all_judgments)
    analyze(str(DATA), str(combined_judgments), str(OUT))
    pd.DataFrame(administration_rows).to_csv(OUT / "administration_summary.csv", index=False)

    judgment_frame = pd.DataFrame(all_judgments)
    task_frame = pd.DataFrame(tasks.values())[["task_id", "task_type", "curve_family", "stimulus_id"]]
    stimulus_frame = pd.DataFrame(stimuli.values())[["stimulus_id", "base_stimulus_id"]]
    merged = judgment_frame.merge(task_frame, on=["task_id", "stimulus_id"]).merge(stimulus_frame, on="stimulus_id")

    overall_rows = []
    for index, (model_name, group) in enumerate(merged.groupby("target_model_name")):
        ci_low, ci_high = cluster_bootstrap_ci(group, seed=20260712 + index)
        overall_rows.append(
            {
                "target_model_name": model_name,
                "n": len(group),
                "mean_normalized_score": group["normalized_score"].mean(),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "total_points": group["total_score"].sum(),
                "maximum_points": group["max_possible_score"].sum(),
            }
        )
    pd.DataFrame(overall_rows).to_csv(OUT / "overall_results.csv", index=False)

    for filename, grouping in {
        "task_type_results.csv": "task_type",
        "curve_family_results.csv": "curve_family",
        "style_results.csv": "style_profile_id",
    }.items():
        if grouping == "style_profile_id":
            style_map = pd.DataFrame(stimuli.values())[["stimulus_id", "visual"]]
            style_map["style_profile_id"] = style_map["visual"].map(lambda value: value["style_profile_id"])
            grouped_frame = merged.merge(style_map[["stimulus_id", "style_profile_id"]], on="stimulus_id")
        else:
            grouped_frame = merged
        summary = (
            grouped_frame.groupby(["target_model_name", grouping])["normalized_score"]
            .agg([("n", "size"), ("mean_normalized_score", "mean")])
            .reset_index()
        )
        summary.to_csv(OUT / filename, index=False)

    model_names = list(MODEL_RUNS)
    pairwise_rows = []
    pair_index = 0
    for index, model_a in enumerate(model_names):
        for model_b in model_names[index + 1 :]:
            pairwise_rows.append(
                paired_cluster_bootstrap(
                    merged,
                    model_a,
                    model_b,
                    seed=20260714 + pair_index,
                )
            )
            pair_index += 1
    pd.DataFrame(pairwise_rows).to_csv(OUT / "paired_model_difference.csv", index=False)
    print(f"Prepared complete empirical results for {', '.join(MODEL_RUNS)} in {OUT}")


if __name__ == "__main__":
    main()
