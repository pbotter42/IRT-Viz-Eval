"""Command-line interface for the IRT visual interpretation benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import analyze
from .baselines import build_baseline_responses
from .judging import judge_responses
from .simulation import generate_benchmark


def parse_style_list(value: str | None) -> list[str] | None:
    if not value or value == "all":
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def cmd_generate(args: argparse.Namespace) -> None:
    manifest = generate_benchmark(
        output_dir=args.output,
        n_per_model=args.n_per_model,
        styles=parse_style_list(args.styles),
        seed=args.seed,
    )
    print(f"Generated {manifest['counts']['rendered_stimuli']} stimuli and {manifest['counts']['tasks']} tasks in {args.output}")


def cmd_run_baseline(args: argparse.Namespace) -> None:
    tasks_path = Path(args.data) / "tasks.jsonl"
    output_path = args.output or str(Path(args.data) / "responses.jsonl")
    count = build_baseline_responses(
        tasks_path=str(tasks_path),
        output_path=output_path,
        model_names=[part.strip() for part in args.models.split(",") if part.strip()],
        seed=args.seed,
    )
    print(f"Wrote {count} baseline responses to {output_path}")


def cmd_judge(args: argparse.Namespace) -> None:
    tasks_path = Path(args.data) / "tasks.jsonl"
    responses_path = args.responses or str(Path(args.data) / "responses.jsonl")
    output_path = args.output or str(Path(args.data) / "judgments.jsonl")
    count = judge_responses(str(tasks_path), responses_path, output_path)
    print(f"Wrote {count} judgments to {output_path}")


def cmd_analyze(args: argparse.Namespace) -> None:
    judgments_path = args.judgments or str(Path(args.data) / "judgments.jsonl")
    outputs = analyze(args.data, judgments_path, args.output)
    print("Analysis outputs:")
    for key, value in outputs.items():
        print(f"  {key}: {value}")


def cmd_run_openai(args: argparse.Namespace) -> None:
    from .openai_runner import run_openai_models

    tasks_path = Path(args.data) / "tasks.jsonl"
    output_path = args.output or str(Path(args.data) / "openai_responses.jsonl")
    counts = run_openai_models(
        tasks_path=tasks_path,
        output_path=output_path,
        models=[part.strip() for part in args.models.split(",") if part.strip()],
        repetitions=args.repetitions,
        detail=args.detail,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        limit=args.limit,
        resume=not args.no_resume,
        fail_fast=args.fail_fast,
    )
    print(
        f"OpenAI run: {counts['completed']} completed, "
        f"{counts['failed']} failed, {counts['skipped']} skipped; output={output_path}"
    )


def cmd_all(args: argparse.Namespace) -> None:
    manifest = generate_benchmark(
        output_dir=args.data,
        n_per_model=args.n_per_model,
        styles=parse_style_list(args.styles),
        seed=args.seed,
    )
    print(f"Generated {manifest['counts']['rendered_stimuli']} stimuli and {manifest['counts']['tasks']} tasks in {args.data}")
    responses_path = str(Path(args.data) / "responses.jsonl")
    response_count = build_baseline_responses(
        tasks_path=str(Path(args.data) / "tasks.jsonl"),
        output_path=responses_path,
        model_names=[part.strip() for part in args.models.split(",") if part.strip()],
        seed=args.seed,
    )
    print(f"Wrote {response_count} baseline responses to {responses_path}")
    judgments_path = str(Path(args.data) / "judgments.jsonl")
    judgment_count = judge_responses(str(Path(args.data) / "tasks.jsonl"), responses_path, judgments_path)
    print(f"Wrote {judgment_count} judgments to {judgments_path}")
    outputs = analyze(args.data, judgments_path, args.analysis_output)
    print("Analysis outputs:")
    for key, value in outputs.items():
        print(f"  {key}: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IRT visual interpretation benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate benchmark stimuli, metadata, images, and tasks")
    generate.add_argument("--output", default="data/benchmark")
    generate.add_argument("--n-per-model", type=int, default=1)
    generate.add_argument("--styles", default="all", help="Comma-separated style profile ids, or all")
    generate.add_argument("--seed", type=int, default=20260711)
    generate.set_defaults(func=cmd_generate)

    baseline = subparsers.add_parser("run-baseline", help="Write offline baseline target-model responses")
    baseline.add_argument("--data", default="data/benchmark")
    baseline.add_argument("--output", default=None)
    baseline.add_argument("--models", default="oracle,noisy,blind")
    baseline.add_argument("--seed", type=int, default=20260711)
    baseline.set_defaults(func=cmd_run_baseline)

    openai_run = subparsers.add_parser("run-openai", help="Run tasks with OpenAI vision-capable API models")
    openai_run.add_argument("--data", default="data/benchmark")
    openai_run.add_argument("--output", default=None)
    openai_run.add_argument("--models", required=True, help="Comma-separated API model identifiers")
    openai_run.add_argument("--repetitions", type=int, default=1)
    openai_run.add_argument("--detail", choices=["low", "high", "original", "auto"], default="high")
    openai_run.add_argument("--max-output-tokens", type=int, default=600)
    openai_run.add_argument("--temperature", type=float, default=None)
    openai_run.add_argument("--limit", type=int, default=None, help="Run only the first N tasks for a pilot")
    openai_run.add_argument("--no-resume", action="store_true", help="Do not skip completed response ids")
    openai_run.add_argument("--fail-fast", action="store_true", help="Stop on the first API error")
    openai_run.set_defaults(func=cmd_run_openai)

    judge = subparsers.add_parser("judge", help="Score target-model responses with the deterministic rubric")
    judge.add_argument("--data", default="data/benchmark")
    judge.add_argument("--responses", default=None)
    judge.add_argument("--output", default=None)
    judge.set_defaults(func=cmd_judge)

    analysis = subparsers.add_parser("analyze", help="Aggregate scores and write analysis figures")
    analysis.add_argument("--data", default="data/benchmark")
    analysis.add_argument("--judgments", default=None)
    analysis.add_argument("--output", default="output/analysis")
    analysis.set_defaults(func=cmd_analyze)

    all_cmd = subparsers.add_parser("all", help="Generate data, run baselines, judge, and analyze")
    all_cmd.add_argument("--data", default="data/benchmark")
    all_cmd.add_argument("--analysis-output", default="output/analysis")
    all_cmd.add_argument("--n-per-model", type=int, default=1)
    all_cmd.add_argument("--styles", default="all")
    all_cmd.add_argument("--models", default="oracle,noisy,blind")
    all_cmd.add_argument("--seed", type=int, default=20260711)
    all_cmd.set_defaults(func=cmd_all)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
