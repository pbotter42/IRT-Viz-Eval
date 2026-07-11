from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from irt_viz_eval.io import read_jsonl
from irt_viz_eval.openai_runner import image_data_url, run_openai_models


class FakeResponses:
    def __init__(self) -> None:
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(
            id="resp_test",
            model="vision-model-snapshot",
            output_text='{"irt_model":"1PL","rationale":"lower asymptote and slope"}',
            usage=SimpleNamespace(input_tokens=100, output_tokens=20, total_tokens=120),
        )


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    image_path = tmp_path / "curve.png"
    Image.new("RGB", (32, 32), "white").save(image_path)
    task = {
        "task_id": "task_1",
        "stimulus_id": "stimulus_1",
        "image_path": str(image_path),
        "prompt": "Identify the curve. Return JSON.",
        "prompt_template_id": "template_v1",
    }
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(json.dumps(task) + "\n", encoding="utf-8")
    return tasks_path, tmp_path / "responses.jsonl"


def test_image_data_url(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "white").save(image)
    assert image_data_url(image).startswith("data:image/png;base64,")


def test_runner_records_provenance_and_resumes(tmp_path: Path) -> None:
    tasks_path, output_path = _write_fixture(tmp_path)
    responses = FakeResponses()
    client = SimpleNamespace(responses=responses)

    counts = run_openai_models(
        tasks_path,
        output_path,
        ["vision-model"],
        repetitions=2,
        detail="high",
        temperature=0.0,
        client=client,
    )
    assert counts == {"completed": 2, "failed": 0, "skipped": 0}
    rows = list(read_jsonl(output_path))
    assert len(rows) == 2
    assert rows[0]["requested_model"] == "vision-model"
    assert rows[0]["returned_model"] == "vision-model-snapshot"
    assert rows[0]["usage"]["total_tokens"] == 120
    assert responses.requests[0]["input"][0]["content"][1]["detail"] == "high"

    resumed = run_openai_models(tasks_path, output_path, ["vision-model"], repetitions=2, client=client)
    assert resumed == {"completed": 0, "failed": 0, "skipped": 2}
    assert len(responses.requests) == 2
