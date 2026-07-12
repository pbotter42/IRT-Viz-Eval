"""Run IRT-Viz-Eval tasks through Hugging Face Inference Providers."""

from __future__ import annotations

import base64
import mimetypes
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .io import append_jsonl, read_jsonl


def _image_data_url(path: Path) -> str:
    media_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def _resolve_image(image_path: str, tasks_path: Path, repository_root: Path | None) -> Path:
    candidate = Path(image_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    for root in (repository_root, Path.cwd(), tasks_path.parent):
        if root is not None and (root / candidate).exists():
            return root / candidate
    raise FileNotFoundError(f"Could not resolve benchmark image: {image_path}")


def _usage_dict(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, Mapping):
        return dict(usage)
    return {key: getattr(usage, key) for key in ("prompt_tokens", "completion_tokens", "total_tokens") if hasattr(usage, key)}


def _completed_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    return {str(row["response_id"]) for row in read_jsonl(output_path) if row.get("status", "completed") == "completed"}


def _is_retryable(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    if any(marker in name for marker in ("timeout", "connection", "network", "transport")):
        return True
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    return status_code in {408, 409, 425, 429} or (isinstance(status_code, int) and status_code >= 500)


def run_huggingface_models(
    tasks_path: str | Path,
    output_path: str | Path,
    models: Iterable[str],
    *,
    repetitions: int = 1,
    max_tokens: int = 600,
    request_timeout: float = 180.0,
    max_retries: int = 2,
    retry_backoff_seconds: float = 2.0,
    limit: int | None = None,
    resume: bool = True,
    fail_fast: bool = False,
    repository_root: str | Path | None = None,
    client_factory: Any | None = None,
) -> dict[str, int]:
    """Run tasks with routed HF providers; ``:cheapest`` is accepted in model ids."""
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    if request_timeout <= 0:
        raise ValueError("request_timeout must be greater than zero")
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")
    model_list = [model.strip() for model in models if model.strip()]
    if not model_list:
        raise ValueError("At least one model identifier is required")

    try:
        from huggingface_hub import InferenceClient
    except ImportError as exc:
        raise RuntimeError("Install the Hugging Face extra with: pip install -e '.[huggingface]'") from exc

    factory = client_factory or (lambda model: InferenceClient(model=model, timeout=request_timeout))
    tasks_file = Path(tasks_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    root = Path(repository_root) if repository_root is not None else None
    tasks = list(read_jsonl(tasks_file))
    if limit is not None:
        tasks = tasks[: max(limit, 0)]
    completed = _completed_ids(target) if resume else set()
    counts = {"completed": 0, "failed": 0, "skipped": 0}
    total_requests = len(tasks) * len(model_list) * repetitions
    request_number = 0

    for task in tasks:
        image = _resolve_image(str(task["image_path"]), tasks_file, root)
        image_url = _image_data_url(image)
        for model in model_list:
            for repetition in range(1, repetitions + 1):
                request_number += 1
                response_id = f"{task['task_id']}__huggingface__{_slug(model)}__r{repetition:03d}"
                if response_id in completed:
                    counts["skipped"] += 1
                    continue
                print(
                    f"[{request_number}/{total_requests}] Requesting {model} for {task['task_id']} "
                    f"(repetition {repetition})",
                    flush=True,
                )
                started_at = datetime.now(timezone.utc).isoformat()
                started = time.perf_counter()
                try:
                    client = factory(model)
                    for attempt in range(max_retries + 1):
                        try:
                            response = client.chat.completions.create(
                                model=model,
                                messages=[
                                    {
                                        "role": "user",
                                        "content": [
                                            {"type": "text", "text": task["prompt"]},
                                            {"type": "image_url", "image_url": {"url": image_url}},
                                        ],
                                    }
                                ],
                                max_tokens=max_tokens,
                            )
                            break
                        except Exception as exc:
                            if attempt >= max_retries or not _is_retryable(exc):
                                raise
                            delay = retry_backoff_seconds * (2**attempt)
                            print(
                                f"  Transient {type(exc).__name__}; retrying in {delay:g}s "
                                f"({attempt + 1}/{max_retries})",
                                flush=True,
                            )
                            time.sleep(delay)
                    message = response.choices[0].message
                    raw_output = str(getattr(message, "content", ""))
                    returned_model = str(getattr(response, "model", model))
                    append_jsonl(
                        target,
                        {
                            "response_id": response_id,
                            "task_id": task["task_id"],
                            "stimulus_id": task["stimulus_id"],
                            "target_model_name": returned_model,
                            "target_model_family": "huggingface_inference_provider",
                            "provider": "Hugging Face Inference Providers",
                            "requested_model": model,
                            "returned_model": returned_model,
                            "api_response_id": str(getattr(response, "id", "")),
                            "prompt_template_id": task["prompt_template_id"],
                            "max_output_tokens": max_tokens,
                            "repetition": repetition,
                            "started_at_utc": started_at,
                            "inference_latency_ms": int((time.perf_counter() - started) * 1000),
                            "usage": _usage_dict(getattr(response, "usage", None)),
                            "status": "completed",
                            "target_raw_output": raw_output,
                        },
                    )
                    completed.add(response_id)
                    counts["completed"] += 1
                    print(
                        f"  Completed in {int((time.perf_counter() - started) * 1000)} ms",
                        flush=True,
                    )
                except Exception as exc:
                    append_jsonl(
                        target,
                        {
                            "response_id": response_id,
                            "task_id": task["task_id"],
                            "stimulus_id": task["stimulus_id"],
                            "target_model_name": model,
                            "target_model_family": "huggingface_inference_provider",
                            "provider": "Hugging Face Inference Providers",
                            "requested_model": model,
                            "prompt_template_id": task["prompt_template_id"],
                            "max_output_tokens": max_tokens,
                            "repetition": repetition,
                            "started_at_utc": started_at,
                            "inference_latency_ms": int((time.perf_counter() - started) * 1000),
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "target_raw_output": "",
                        },
                    )
                    counts["failed"] += 1
                    print(f"  Failed: {type(exc).__name__}: {exc}", flush=True)
                    if fail_fast:
                        raise
    return counts
