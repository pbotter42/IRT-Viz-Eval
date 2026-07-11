"""Generate simulated IRT benchmark stimuli and metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io import ensure_dir, write_json, write_jsonl
from .psychometrics import (
    curve_peak,
    dichotomous_information,
    dichotomous_probability,
    expected_score,
    grm_category_probabilities,
    ideal_point_information,
    ideal_point_probability,
    nearest_value,
    pcm_category_probabilities,
    theta_grid,
)
from .rendering import get_style_profiles, render_stimulus
from .tasks import build_tasks_for_stimulus


BINARY_MODELS = ["1PL", "2PL", "3PL", "4PL"]
POLYTOMOUS_MODELS = ["PCM", "GPCM", "GRM"]


def _round_float(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _sample_binary_params(rng: np.random.Generator, model: str) -> dict[str, float]:
    b = float(np.clip(rng.normal(0.0, 1.25), -3.0, 3.0))
    if model == "1PL":
        return {"a": 1.0, "b": b, "c": 0.0, "d": 1.0}
    a = float(np.clip(rng.lognormal(mean=0.0, sigma=0.28), 0.45, 2.5))
    if model == "2PL":
        return {"a": a, "b": b, "c": 0.0, "d": 1.0}
    c = float(rng.uniform(0.05, 0.25))
    if model == "3PL":
        return {"a": a, "b": b, "c": c, "d": 1.0}
    d = float(rng.uniform(0.82, 0.98))
    return {"a": a, "b": b, "c": c, "d": d}


def _sample_thresholds(rng: np.random.Generator, n_steps: int = 3) -> list[float]:
    raw = np.sort(rng.normal(0.0, 1.0, n_steps))
    raw = np.clip(raw, -2.8, 2.8)
    return [_round_float(x) for x in raw]


def _series(values: np.ndarray, label: str) -> dict[str, Any]:
    return {"label": label, "values": [_round_float(x) for x in values]}


def _base_stimulus(
    stimulus_id: str,
    curve_family: str,
    irt_family: str,
    irt_model: str,
    monotonicity: str,
    assessment_level: str,
    display_title: str,
    y_axis_label: str,
    parameters: dict[str, Any],
    coordinates: dict[str, Any],
    ground_truth: dict[str, Any],
    constituent_item_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "stimulus_id": stimulus_id,
        "assessment_level": assessment_level,
        "curve_family": curve_family,
        "irt_family": irt_family,
        "irt_model": irt_model,
        "monotonicity": monotonicity,
        "display_title": display_title,
        "y_axis_label": y_axis_label,
        "parameters": parameters,
        "coordinates": coordinates,
        "ground_truth": ground_truth,
        "constituent_item_ids": constituent_item_ids or [],
    }


def binary_stimuli(rng: np.random.Generator, n_per_model: int, theta: np.ndarray) -> list[dict[str, Any]]:
    stimuli: list[dict[str, Any]] = []
    for model in BINARY_MODELS:
        for idx in range(n_per_model):
            params = _sample_binary_params(rng, model)
            p = dichotomous_probability(theta, **params)
            info = dichotomous_information(theta, **params)
            item_id = f"{model.lower()}_{idx + 1:03d}"
            prob_points = {str(t): _round_float(nearest_value(theta, p, t)) for t in [-2.0, -1.0, 0.0, 1.0, 2.0]}
            info_peak_theta, info_peak = curve_peak(theta, info)
            common_truth = {
                "parameters": {k: _round_float(v) for k, v in params.items()},
                "probability_at_theta": prob_points,
                "information_peak_theta": _round_float(info_peak_theta),
                "information_peak": _round_float(info_peak),
            }
            stimuli.append(
                _base_stimulus(
                    stimulus_id=f"{item_id}_icc",
                    curve_family="dichotomous_icc",
                    irt_family="dichotomous",
                    irt_model=model,
                    monotonicity="monotonic",
                    assessment_level="item",
                    display_title="Item characteristic curve",
                    y_axis_label="Probability of keyed response",
                    parameters={k: _round_float(v) for k, v in params.items()},
                    coordinates={"theta": [_round_float(x) for x in theta], "series": [_series(p, "Item response")]},
                    ground_truth=common_truth,
                )
            )
            stimuli.append(
                _base_stimulus(
                    stimulus_id=f"{item_id}_iif",
                    curve_family="item_information",
                    irt_family="dichotomous",
                    irt_model=model,
                    monotonicity="monotonic",
                    assessment_level="item",
                    display_title="Item information function",
                    y_axis_label="Information",
                    parameters={k: _round_float(v) for k, v in params.items()},
                    coordinates={"theta": [_round_float(x) for x in theta], "series": [_series(info, "Item information")]},
                    ground_truth=common_truth,
                )
            )
    return stimuli


def polytomous_stimuli(rng: np.random.Generator, n_per_model: int, theta: np.ndarray) -> list[dict[str, Any]]:
    stimuli: list[dict[str, Any]] = []
    for model in POLYTOMOUS_MODELS:
        for idx in range(n_per_model):
            thresholds = _sample_thresholds(rng, n_steps=3)
            a = 1.0 if model == "PCM" else float(np.clip(rng.lognormal(mean=0.0, sigma=0.24), 0.5, 2.2))
            if model == "GRM":
                probs = grm_category_probabilities(theta, thresholds, a=a)
            else:
                probs = pcm_category_probabilities(theta, thresholds, a=a)
            exp_score = expected_score(probs)
            peak_categories = {f"category_{k}": _round_float(theta[int(np.argmax(probs[:, k]))]) for k in range(probs.shape[1])}
            stimulus_id = f"{model.lower()}_{idx + 1:03d}_crc"
            params = {"a": _round_float(a), "step_thresholds": thresholds, "num_categories": int(probs.shape[1])}
            series = [_series(probs[:, k], f"Category {k}") for k in range(probs.shape[1])]
            truth = {
                "parameters": params,
                "expected_score_at_theta": {str(t): _round_float(nearest_value(theta, exp_score, t)) for t in [-1.0, 0.0, 1.0]},
                "category_peak_theta": peak_categories,
                "most_likely_category_at_theta_0": int(np.argmax(probs[int(np.argmin(np.abs(theta))), :])),
            }
            stimuli.append(
                _base_stimulus(
                    stimulus_id=stimulus_id,
                    curve_family="polytomous_crc",
                    irt_family="polytomous",
                    irt_model=model,
                    monotonicity="monotonic",
                    assessment_level="item",
                    display_title="Category response curves",
                    y_axis_label="Category probability",
                    parameters=params,
                    coordinates={"theta": [_round_float(x) for x in theta], "series": series},
                    ground_truth=truth,
                )
            )
    return stimuli


def unfolding_stimuli(rng: np.random.Generator, n: int, theta: np.ndarray) -> list[dict[str, Any]]:
    stimuli: list[dict[str, Any]] = []
    for idx in range(n):
        location = float(np.clip(rng.normal(0.0, 1.2), -2.5, 2.5))
        width = float(rng.uniform(0.65, 1.45))
        lower = float(rng.uniform(0.0, 0.08))
        upper = float(rng.uniform(0.82, 0.98))
        p = ideal_point_probability(theta, location=location, width=width, lower=lower, upper=upper)
        info = ideal_point_information(theta, location=location, width=width, lower=lower, upper=upper)
        peak_theta, peak_prob = curve_peak(theta, p)
        info_peak_theta, info_peak = curve_peak(theta, info)
        params = {
            "location": _round_float(location),
            "width": _round_float(width),
            "lower": _round_float(lower),
            "upper": _round_float(upper),
        }
        truth = {
            "parameters": params,
            "peak_theta": _round_float(peak_theta),
            "peak_probability": _round_float(peak_prob),
            "information_peak_theta": _round_float(info_peak_theta),
            "information_peak": _round_float(info_peak),
            "probability_at_theta": {str(t): _round_float(nearest_value(theta, p, t)) for t in [-2.0, 0.0, 2.0]},
        }
        stimuli.append(
            _base_stimulus(
                stimulus_id=f"unfolding_{idx + 1:03d}_icc",
                curve_family="unfolding",
                irt_family="ideal_point",
                irt_model="IdealPoint",
                monotonicity="non_monotonic",
                assessment_level="item",
                display_title="Non-monotonic item response curve",
                y_axis_label="Probability of endorsement",
                parameters=params,
                coordinates={"theta": [_round_float(x) for x in theta], "series": [_series(p, "Endorsement probability")]},
                ground_truth=truth,
            )
        )
    return stimuli


def test_level_stimuli(rng: np.random.Generator, n_forms: int, theta: np.ndarray, items_per_form: int = 8) -> list[dict[str, Any]]:
    stimuli: list[dict[str, Any]] = []
    for form_idx in range(n_forms):
        item_params = [_sample_binary_params(rng, rng.choice(BINARY_MODELS)) for _ in range(items_per_form)]
        probs = np.vstack([dichotomous_probability(theta, **params) for params in item_params])
        infos = np.vstack([dichotomous_information(theta, **params) for params in item_params])
        tcc = probs.sum(axis=0)
        tif = infos.sum(axis=0)
        tcc_peak_theta, tcc_max = curve_peak(theta, tcc)
        tif_peak_theta, tif_max = curve_peak(theta, tif)
        item_ids = [f"form_{form_idx + 1:03d}_item_{i + 1:02d}" for i in range(items_per_form)]
        common = {
            "num_items_aggregated": items_per_form,
            "max_possible_score": items_per_form,
            "constituent_item_ids": item_ids,
            "constituent_item_parameters": [
                {k: _round_float(v) for k, v in params.items()} for params in item_params
            ],
            "expected_score_at_theta": {str(t): _round_float(nearest_value(theta, tcc, t)) for t in [-2.0, -1.0, 0.0, 1.0, 2.0]},
            "test_information_at_theta": {str(t): _round_float(nearest_value(theta, tif, t)) for t in [-2.0, 0.0, 2.0]},
            "tcc_max_theta": _round_float(tcc_peak_theta),
            "tcc_max_expected_score": _round_float(tcc_max),
            "tif_peak_theta": _round_float(tif_peak_theta),
            "tif_peak_information": _round_float(tif_max),
        }
        stimuli.append(
            _base_stimulus(
                stimulus_id=f"form_{form_idx + 1:03d}_tcc",
                curve_family="test_characteristic",
                irt_family="mixed_dichotomous",
                irt_model="MixedDichotomousForm",
                monotonicity="monotonic",
                assessment_level="test",
                display_title="Test characteristic curve",
                y_axis_label="Expected raw score",
                parameters={"num_items": items_per_form},
                coordinates={"theta": [_round_float(x) for x in theta], "series": [_series(tcc, "Expected score")]},
                ground_truth=common,
                constituent_item_ids=item_ids,
            )
        )
        stimuli.append(
            _base_stimulus(
                stimulus_id=f"form_{form_idx + 1:03d}_tif",
                curve_family="test_information",
                irt_family="mixed_dichotomous",
                irt_model="MixedDichotomousForm",
                monotonicity="monotonic",
                assessment_level="test",
                display_title="Test information function",
                y_axis_label="Information",
                parameters={"num_items": items_per_form},
                coordinates={"theta": [_round_float(x) for x in theta], "series": [_series(tif, "Test information")]},
                ground_truth=common,
                constituent_item_ids=item_ids,
            )
        )
    return stimuli


def write_coordinate_csv(stimulus: dict[str, Any], output_dir: Path) -> str:
    coord_dir = ensure_dir(output_dir / "coordinates")
    rows: dict[str, Any] = {"theta": stimulus["coordinates"]["theta"]}
    for idx, series in enumerate(stimulus["coordinates"]["series"]):
        label = series["label"].lower().replace(" ", "_")
        rows[f"series_{idx}_{label}"] = series["values"]
    path = coord_dir / f"{stimulus['stimulus_id']}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def generate_benchmark(
    output_dir: str | Path,
    n_per_model: int = 1,
    styles: list[str] | None = None,
    seed: int = 20260711,
) -> dict[str, Any]:
    output_path = ensure_dir(output_dir)
    rng = np.random.default_rng(seed)
    theta = theta_grid()
    style_profiles = get_style_profiles(styles)

    base_stimuli = []
    base_stimuli.extend(binary_stimuli(rng, n_per_model=n_per_model, theta=theta))
    base_stimuli.extend(polytomous_stimuli(rng, n_per_model=n_per_model, theta=theta))
    base_stimuli.extend(unfolding_stimuli(rng, n=n_per_model, theta=theta))
    base_stimuli.extend(test_level_stimuli(rng, n_forms=n_per_model, theta=theta))

    rendered_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for base in base_stimuli:
        coordinates_path = write_coordinate_csv(base, output_path)
        for profile in style_profiles:
            row = {k: v for k, v in base.items() if k != "coordinates"}
            row["coordinates_path"] = coordinates_path
            row["visual"] = render_stimulus(base, profile, output_path)
            row["stimulus_id"] = f"{base['stimulus_id']}__{profile['style_profile_id']}"
            row["base_stimulus_id"] = base["stimulus_id"]
            rendered_rows.append(row)
            task_rows.extend(build_tasks_for_stimulus(row))

    manifest = {
        "benchmark_id": f"irt_viz_eval_seed_{seed}",
        "version": "0.1.0",
        "seed": seed,
        "theta_grid": {"min": float(theta.min()), "max": float(theta.max()), "step": 0.05},
        "n_per_model": n_per_model,
        "style_profiles": style_profiles,
        "counts": {
            "base_stimuli": len(base_stimuli),
            "rendered_stimuli": len(rendered_rows),
            "tasks": len(task_rows),
        },
        "files": {
            "stimuli": str(output_path / "stimuli.jsonl"),
            "tasks": str(output_path / "tasks.jsonl"),
        },
    }

    write_json(output_path / "manifest.json", manifest)
    write_jsonl(output_path / "stimuli.jsonl", rendered_rows)
    write_jsonl(output_path / "tasks.jsonl", task_rows)
    return manifest
