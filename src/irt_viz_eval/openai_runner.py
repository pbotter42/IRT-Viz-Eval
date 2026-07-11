"""Run IRT-Viz-Eval tasks against OpenAI vision-capable API models."""

from __future__ import annotations

import base64
import mimetypes
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .io import append_jsonl, read_jsonl


def image_data_url(path: str | Path) -> str:
    image_path = Path(path)
    media_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def _resolve_image_path(image_path: str, tasks_path: Path, repository_root: Path | None) -> Path:
    candidate = Path(image_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    roots = [repository_root, Path.cwd(), tasks_path.parent]
    for root in roots:
        if root is not None:
            resolved = root / candidate
            if resolved.exists():
                return resolved
    raise FileNotFoundError(f"Could not resolve benchmark image: {image_path}")


def _usage_dict(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, Mapping):
        return dict(usage)
    return {key: getattr(usage, key) for key in ("input_tokens", "output_tokens", "total_tokens") if hasattr(usage, key)}


def _existing_completed_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    return {
        str(row["response_id"])
        for row in read_jsonl(output_path)
        if row.get("status", "completed") == "completed"
    }


def run_openai_models(
    tasks_path: str | Path,
    output_path: str | Path,
    models: Iterable[str],
    *,
    repetitions: int = 1,
    detail: str = "high",
    max_output_tokens: int = 600,
    temperature: float | None = None,
    limit: int | None = None,
    resume: bool = True,
    fail_fast: bool = False,
    repository_root: str | Path | None = None,
    client: Any | None = None,
) -> dict[str, int]:
    """Administer benchmark tasks and append auditable response records.

    ``client`` is injectable so the request/response contract can be tested without
    network access. Normal use constructs ``openai.OpenAI`` from OPENAI_API_KEY.
    """
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    if detail not in {"low", "high", "original", "auto"}:
        raise ValueError("detail must be low, high, original, or auto")

    model_list = [model.strip() for model in models if model.strip()]
    if not model_list:
        raise ValueError("At least one model identifier is required")

    if client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the API extra with: pip install -e '.[openai]'") from exc
        client = OpenAI()

    tasks_file = Path(tasks_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    root = Path(repository_root) if repository_root is not None else None
    tasks = list(read_jsonl(tasks_file))
    if limit is not None:
        tasks = tasks[: max(limit, 0)]

    completed_ids = _existing_completed_ids(target) if resume else set()
    counts = {"completed": 0, "failed": 0, "skipped": 0}

    for task in tasks:
        image_path = _resolve_image_path(str(task["image_path"]), tasks_file, root)
        data_url = image_data_url(image_path)
        for model in model_list:
            for repetition in range(1, repetitions + 1):
                response_id = f"{task['task_id']}__openai__{_slug(model)}__r{repetition:03d}"
                if response_id in completed_ids:
                    counts["skipped"] += 1
                    continue

                request: dict[str, Any] = {
                    "model": model,
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": task["prompt"]},
                                {"type": "input_image", "image_url": data_url, "detail": detail},
                            ],
                        }
                    ],
                    "max_output_tokens": max_output_tokens,
                }
                if temperature is not None:
                    request["temperature"] = temperature

                started_at = datetime.now(timezone.utc).isoformat()
                started = time.perf_counter()
                try:
                    response = client.responses.create(**request)
                    latency_ms = int((time.perf_counter() - started) * 1000)
                    raw_output = str(getattr(response, "output_text", ""))
                    row = {
                        "response_id": response_id,
                        "task_id": task["task_id"],
                        "stimulus_id": task["stimulus_id"],
                        "target_model_name": str(getattr(response, "model", model)),
                        "target_model_family": "openai_api",
                        "provider": "OpenAI",
                        "requested_model": model,
                        "returned_model": str(getattr(response, "model", model)),
                        "api_response_id": str(getattr(response, "id", "")),
                        "prompt_template_id": task["prompt_template_id"],
                        "temperature": temperature,
                        "image_detail": detail,
                        "max_output_tokens": max_output_tokens,
                        "repetition": repetition,
                        "started_at_utc": started_at,
                        "inference_latency_ms": latency_ms,
                        "usage": _usage_dict(response),
                        "status": "completed",
                        "target_raw_output": raw_output,
                    }
                    append_jsonl(target, row)
                    completed_ids.add(response_id)
                    counts["completed"] += 1
                except Exception as exc:
                    latency_ms = int((time.perf_counter() - started) * 1000)
                    append_jsonl(
                        target,
                        {
                            "response_id": response_id,
                            "task_id": task["task_id"],
                            "stimulus_id": task["stimulus_id"],
                            "target_model_name": model,
                            "target_model_family": "openai_api",
                            "provider": "OpenAI",
                            "requested_model": model,
                            "prompt_template_id": task["prompt_template_id"],
                            "temperature": temperature,
                            "image_detail": detail,
                            "max_output_tokens": max_output_tokens,
                            "repetition": repetition,
                            "started_at_utc": started_at,
                            "inference_latency_ms": latency_ms,
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "target_raw_output": "",
                        },
                    )
                    counts["failed"] += 1
                    if fail_fast:
                        raise
    return counts
