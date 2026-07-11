"""Aggregate benchmark judgments into tables and figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .io import ensure_dir, read_jsonl


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    rows = [[str(value) for value in row] for row in df.to_numpy()]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows)) if rows else len(headers[idx])
        for idx in range(len(headers))
    ]
    header_line = "| " + " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
    divider = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, divider, *body])


def analyze(data_dir: str, judgments_path: str, output_dir: str) -> dict[str, str]:
    out = ensure_dir(output_dir)
    stimuli = pd.DataFrame(read_jsonl(Path(data_dir) / "stimuli.jsonl"))
    tasks = pd.DataFrame(read_jsonl(Path(data_dir) / "tasks.jsonl"))
    judgments = pd.DataFrame(read_jsonl(judgments_path))

    merged = judgments.merge(tasks[["task_id", "task_type", "curve_family", "irt_model", "style_profile_id"]], on="task_id")
    merged = merged.merge(stimuli[["stimulus_id", "visual"]], on="stimulus_id", how="left")
    merged["render_engine"] = merged["visual"].map(lambda row: row["render_engine"] if isinstance(row, dict) else None)
    merged["theme"] = merged["visual"].map(lambda row: row["theme"] if isinstance(row, dict) else None)

    outputs: dict[str, str] = {}
    detail_path = out / "judgment_detail.csv"
    merged.to_csv(detail_path, index=False)
    outputs["detail"] = str(detail_path)

    groupings = {
        "summary_by_model.csv": ["target_model_name"],
        "summary_by_model_and_family.csv": ["target_model_name", "curve_family"],
        "summary_by_style.csv": ["target_model_name", "style_profile_id"],
        "summary_by_task_type.csv": ["target_model_name", "task_type"],
    }
    for filename, cols in groupings.items():
        summary = (
            merged.groupby(cols, dropna=False)
            .agg(
                n=("normalized_score", "size"),
                mean_normalized_score=("normalized_score", "mean"),
                mean_total_score=("total_score", "mean"),
                mean_max_possible_score=("max_possible_score", "mean"),
            )
            .reset_index()
            .sort_values(cols)
        )
        path = out / filename
        summary.to_csv(path, index=False)
        outputs[filename] = str(path)

    sns.set_theme(style="whitegrid")
    family_summary = (
        merged.groupby(["target_model_name", "curve_family"], dropna=False)["normalized_score"].mean().reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    sns.barplot(data=family_summary, x="curve_family", y="normalized_score", hue="target_model_name", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Curve family")
    ax.set_ylabel("Mean normalized score")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(loc="lower left", bbox_to_anchor=(1.02, 0.0), frameon=True)
    fig.tight_layout()
    family_plot = out / "score_by_family.png"
    fig.savefig(family_plot, bbox_inches="tight")
    plt.close(fig)
    outputs["score_by_family"] = str(family_plot)

    style_summary = (
        merged.groupby(["target_model_name", "style_profile_id"], dropna=False)["normalized_score"].mean().reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    sns.barplot(data=style_summary, x="style_profile_id", y="normalized_score", hue="target_model_name", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Style profile")
    ax.set_ylabel("Mean normalized score")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(loc="lower left", bbox_to_anchor=(1.02, 0.0), frameon=True)
    fig.tight_layout()
    style_plot = out / "score_by_style.png"
    fig.savefig(style_plot, bbox_inches="tight")
    plt.close(fig)
    outputs["score_by_style"] = str(style_plot)

    readme = out / "analysis_summary.md"
    model_summary = pd.read_csv(out / "summary_by_model.csv")
    with readme.open("w", encoding="utf-8") as handle:
        handle.write("# Analysis Summary\n\n")
        handle.write("This summary was generated from the current benchmark judgments.\n\n")
        handle.write(dataframe_to_markdown(model_summary))
        handle.write("\n")
    outputs["markdown_summary"] = str(readme)
    return outputs
