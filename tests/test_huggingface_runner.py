from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from irt_viz_eval.huggingface_runner import run_huggingface_models
from irt_viz_eval.io import read_jsonl


class FakeCompletions:
    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.failures:
            self.failures -= 1
            raise TimeoutError("provider stalled")
        return SimpleNamespace(
            id="response_test",
            model="vision-model-snapshot",
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"irt_model":"1PL"}'))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        )


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    image_path = tmp_path / "curve.png"
    Image.new("RGB", (8, 8), "white").save(image_path)
    task = {
        "task_id": "task_1",
        "stimulus_id": "stimulus_1",
        "image_path": str(image_path),
        "prompt": "Identify the curve.",
        "prompt_template_id": "template_v1",
    }
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(json.dumps(task) + "\n", encoding="utf-8")
    return tasks_path, tmp_path / "responses.jsonl"


def test_runner_retries_transient_timeout_and_resumes(tmp_path: Path) -> None:
    tasks_path, output_path = _fixture(tmp_path)
    completions = FakeCompletions(failures=1)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    counts = run_huggingface_models(
        tasks_path,
        output_path,
        ["vision-model:provider"],
        client_factory=lambda model: client,
        retry_backoff_seconds=0,
    )

    assert counts == {"completed": 1, "failed": 0, "skipped": 0}
    assert len(completions.requests) == 2
    rows = list(read_jsonl(output_path))
    assert rows[0]["target_raw_output"] == '{"irt_model":"1PL"}'
    assert rows[0]["usage"]["total_tokens"] == 120

    resumed = run_huggingface_models(
        tasks_path,
        output_path,
        ["vision-model:provider"],
        client_factory=lambda model: client,
        retry_backoff_seconds=0,
    )
    assert resumed == {"completed": 0, "failed": 0, "skipped": 1}


def test_runner_records_final_timeout_as_failure(tmp_path: Path) -> None:
    tasks_path, output_path = _fixture(tmp_path)
    completions = FakeCompletions(failures=2)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    counts = run_huggingface_models(
        tasks_path,
        output_path,
        ["vision-model:provider"],
        max_retries=1,
        retry_backoff_seconds=0,
        client_factory=lambda model: client,
    )

    assert counts == {"completed": 0, "failed": 1, "skipped": 0}
    row = next(iter(read_jsonl(output_path)))
    assert row["status"] == "error"
    assert row["error_type"] == "TimeoutError"
