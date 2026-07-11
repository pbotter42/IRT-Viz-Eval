"""Benchmark task prompt generation."""

from __future__ import annotations

from typing import Any, Mapping


DEFAULT_PARAM_THRESHOLDS = [0.15, 0.30, 0.50]
DEFAULT_PROB_THRESHOLDS = [0.05, 0.10, 0.20]
DEFAULT_SCORE_THRESHOLDS = [0.25, 0.50, 1.00]


def _numeric(value: float, thresholds: list[float] | None = None) -> dict[str, Any]:
    return {"type": "numeric", "value": float(value), "thresholds": thresholds or DEFAULT_PARAM_THRESHOLDS}


def _integer(value: int) -> dict[str, Any]:
    return {"type": "integer", "value": int(value)}


def _categorical(value: str) -> dict[str, Any]:
    return {"type": "categorical", "value": str(value)}


def _common_task(stimulus: Mapping[str, Any], task_type: str, prompt: str, expected: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": f"{stimulus['stimulus_id']}__{task_type}",
        "stimulus_id": stimulus["stimulus_id"],
        "base_stimulus_id": stimulus["base_stimulus_id"],
        "task_type": task_type,
        "assessment_level": stimulus["assessment_level"],
        "curve_family": stimulus["curve_family"],
        "irt_family": stimulus["irt_family"],
        "irt_model": stimulus["irt_model"],
        "style_profile_id": stimulus["visual"]["style_profile_id"],
        "image_path": stimulus["visual"]["image_path"],
        "prompt_template_id": f"zero_shot_{task_type}_v1",
        "prompt": prompt,
        "expected_answers": expected,
        "rubric": {
            "categorical": "1 point for an exact normalized match; otherwise 0.",
            "integer": "1 point for the exact integer; otherwise 0.",
            "numeric": "3/2/1/0 points for error within the provided precise/approximate/poor thresholds.",
            "reasoning": "0 to 2 points for psychometrically valid visual reasoning.",
        },
    }


def _json_instruction(keys: list[str]) -> str:
    joined = ", ".join(f'"{key}"' for key in keys)
    return f"Return a compact JSON object with these keys only: {joined}. Include a short rationale field."


def _model_identification(stimulus: Mapping[str, Any]) -> dict[str, Any]:
    prompt = (
        "Analyze the psychometric figure. Identify the most likely IRT model or curve family "
        "that generated it, using visual evidence such as asymptotes, slope, category curves, "
        "information peaks, or non-monotonicity. "
        + _json_instruction(["irt_model", "rationale"])
    )
    return _common_task(stimulus, "model_identification", prompt, {"irt_model": _categorical(stimulus["irt_model"])})


def _dichotomous_parameter_task(stimulus: Mapping[str, Any]) -> dict[str, Any]:
    params = stimulus["ground_truth"]["parameters"]
    expected = {
        "irt_model": _categorical(stimulus["irt_model"]),
        "a": _numeric(params["a"]),
        "b": _numeric(params["b"]),
        "c": _numeric(params["c"]),
        "d": _numeric(params["d"]),
    }
    prompt = (
        "Estimate the item parameters from this item characteristic curve: discrimination a, "
        "difficulty b, lower asymptote c, and upper asymptote d. "
        + _json_instruction(["irt_model", "a", "b", "c", "d", "rationale"])
    )
    return _common_task(stimulus, "parameter_estimation", prompt, expected)


def _probability_reading_task(stimulus: Mapping[str, Any], target_theta: str = "0.0") -> dict[str, Any]:
    value = stimulus["ground_truth"]["probability_at_theta"][target_theta]
    expected = {"probability": _numeric(value, DEFAULT_PROB_THRESHOLDS)}
    prompt = (
        f"Read the curve visually. At theta = {target_theta}, what is the approximate probability "
        "of the keyed or endorsed response? "
        + _json_instruction(["probability", "rationale"])
    )
    return _common_task(stimulus, f"probability_at_theta_{target_theta}", prompt, expected)


def _information_peak_task(stimulus: Mapping[str, Any]) -> dict[str, Any]:
    truth = stimulus["ground_truth"]
    if stimulus["curve_family"] == "test_information":
        expected = {
            "peak_theta": _numeric(truth["tif_peak_theta"], [0.20, 0.40, 0.75]),
            "peak_information": _numeric(truth["tif_peak_information"], [0.30, 0.75, 1.50]),
        }
    else:
        expected = {
            "peak_theta": _numeric(truth["information_peak_theta"], [0.20, 0.40, 0.75]),
            "peak_information": _numeric(truth["information_peak"], [0.15, 0.35, 0.75]),
        }
    prompt = (
        "Locate the peak of the information curve. Estimate the theta value at which information "
        "is largest and estimate the maximum information. "
        + _json_instruction(["peak_theta", "peak_information", "rationale"])
    )
    return _common_task(stimulus, "information_peak", prompt, expected)


def _polytomous_task(stimulus: Mapping[str, Any]) -> dict[str, Any]:
    truth = stimulus["ground_truth"]
    expected = {
        "irt_model": _categorical(stimulus["irt_model"]),
        "num_categories": _integer(truth["parameters"]["num_categories"]),
        "most_likely_category_at_theta_0": _integer(truth["most_likely_category_at_theta_0"]),
        "expected_score_at_theta_0": _numeric(truth["expected_score_at_theta"]["0.0"], DEFAULT_SCORE_THRESHOLDS),
    }
    prompt = (
        "Interpret the category response curves. Identify the polytomous IRT model, count the "
        "response categories, name the most likely category at theta = 0, and estimate the "
        "expected item score at theta = 0. "
        + _json_instruction(
            ["irt_model", "num_categories", "most_likely_category_at_theta_0", "expected_score_at_theta_0", "rationale"]
        )
    )
    return _common_task(stimulus, "category_curve_reasoning", prompt, expected)


def _tcc_task(stimulus: Mapping[str, Any]) -> dict[str, Any]:
    truth = stimulus["ground_truth"]
    expected = {
        "num_items": _integer(truth["num_items_aggregated"]),
        "expected_score_at_theta_0": _numeric(truth["expected_score_at_theta"]["0.0"], DEFAULT_SCORE_THRESHOLDS),
    }
    prompt = (
        "Interpret the test characteristic curve. Estimate the number of scored items and the "
        "expected raw score at theta = 0. "
        + _json_instruction(["num_items", "expected_score_at_theta_0", "rationale"])
    )
    return _common_task(stimulus, "tcc_expected_score", prompt, expected)


def _unfolding_task(stimulus: Mapping[str, Any]) -> dict[str, Any]:
    truth = stimulus["ground_truth"]
    expected = {
        "monotonicity": _categorical("non_monotonic"),
        "peak_theta": _numeric(truth["peak_theta"], [0.20, 0.40, 0.75]),
        "peak_probability": _numeric(truth["peak_probability"], DEFAULT_PROB_THRESHOLDS),
    }
    prompt = (
        "Interpret the non-monotonic item response curve. State whether the curve is monotonic "
        "or non_monotonic, then estimate the theta location and probability at the peak. "
        + _json_instruction(["monotonicity", "peak_theta", "peak_probability", "rationale"])
    )
    return _common_task(stimulus, "non_monotonic_peak", prompt, expected)


def build_tasks_for_stimulus(stimulus: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks = [_model_identification(stimulus)]
    family = stimulus["curve_family"]
    if family == "dichotomous_icc":
        tasks.append(_dichotomous_parameter_task(stimulus))
        tasks.append(_probability_reading_task(stimulus, "0.0"))
    elif family == "item_information":
        tasks.append(_information_peak_task(stimulus))
    elif family == "polytomous_crc":
        tasks.append(_polytomous_task(stimulus))
    elif family == "test_characteristic":
        tasks.append(_tcc_task(stimulus))
    elif family == "test_information":
        tasks.append(_information_peak_task(stimulus))
    elif family == "unfolding":
        tasks.append(_unfolding_task(stimulus))
        tasks.append(_probability_reading_task(stimulus, "0.0"))
    return tasks
