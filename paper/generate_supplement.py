"""Generate current empirical tables used by the online supplement."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "output" / "empirical_analysis"
OUT = ROOT / "paper" / "supplement_tables.tex"


def tex(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def score(value: object) -> str:
    return f"{float(value):.3f}".lstrip("0")


def table(title: str, label: str, columns: list[str], rows: list[list[object]], widths: str | None = None) -> str:
    spec = widths or ("l" + "c" * (len(columns) - 1))
    lines = [
        rf"\begin{{table}}[H]",
        r"\centering",
        r"\scriptsize",
        rf"\caption{{{title}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        " & ".join(tex(c) for c in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(tex(c) for c in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def main() -> None:
    overall = pd.read_csv(ANALYSIS / "overall_results.csv")
    admin = pd.read_csv(ANALYSIS / "administration_summary.csv")
    tasks = pd.read_csv(ANALYSIS / "task_type_results.csv")
    families = pd.read_csv(ANALYSIS / "curve_family_results.csv")
    styles = pd.read_csv(ANALYSIS / "style_results.csv")
    pairs = pd.read_csv(ANALYSIS / "paired_model_difference.csv")

    model_order = ["Qwen2.5-VL 72B", "Qwen3-VL 30B", "GLM-4.5V", "Gemma 3 4B", "Gemma 3 12B"]
    task_order = [
        "model_identification",
        "parameter_estimation",
        "probability_at_theta_0.0",
        "information_peak",
        "category_curve_reasoning",
        "tcc_expected_score",
        "non_monotonic_peak",
    ]
    task_labels = {
        "model_identification": "Model identification",
        "parameter_estimation": "Parameter estimation",
        "probability_at_theta_0.0": "Probability at theta=0",
        "information_peak": "Information peak",
        "category_curve_reasoning": "Category-curve reasoning",
        "tcc_expected_score": "Test expected score",
        "non_monotonic_peak": "Non-monotonic peak",
    }
    family_order = [
        "dichotomous_icc",
        "item_information",
        "polytomous_crc",
        "test_characteristic",
        "test_information",
        "unfolding",
    ]
    family_labels = {
        "dichotomous_icc": "Dichotomous ICC",
        "item_information": "Item information",
        "polytomous_crc": "Polytomous curves",
        "test_characteristic": "Test characteristic",
        "test_information": "Test information",
        "unfolding": "Unfolding",
    }
    style_order = [
        "matplotlib_light",
        "seaborn_whitegrid",
        "ggplot_gray",
        "grayscale_no_grid",
        "dark_high_contrast",
        "lattice_emulation",
    ]
    style_labels = {
        "matplotlib_light": "Matplotlib light",
        "seaborn_whitegrid": "Seaborn white-grid",
        "ggplot_gray": "ggplot gray",
        "grayscale_no_grid": "Grayscale",
        "dark_high_contrast": "Dark high-contrast",
        "lattice_emulation": "Lattice-like",
    }

    chunks: list[str] = []
    chunks.append(
        table(
            "Overall results for all complete administrations.",
            "tab:supp-overall",
            ["Model", "$n$", "Mean", "95% CI", "Points", "Empty", "Median latency"],
            [
                [
                    model,
                    int(row.n),
                    score(row.mean_normalized_score),
                    f"[{score(row.ci_low)}, {score(row.ci_high)}]",
                    f"{int(row.total_points)}/{int(row.maximum_points)}",
                    f"{100 * float(admin.loc[admin.target_model_name == model, 'empty_output_rate'].iloc[0]):.1f}%",
                    f"{float(admin.loc[admin.target_model_name == model, 'median_latency_ms'].iloc[0]) / 1000:.2f} s",
                ]
                for model, row in overall.set_index("target_model_name").loc[model_order].iterrows()
            ],
            widths="p{0.22\\linewidth}cccccc",
        )
    )
    chunks.append(
        table(
            "Administration and output-format diagnostics.",
            "tab:supp-admin",
            ["Model", "Requested route", "Returned identifier", "Nonempty", "Parseable", "Tokens"],
            [
                [
                    model,
                    row.requested_model,
                    row.returned_model,
                    f"{int(row.n_nonempty_outputs)}/{int(row.n_unique_tasks)}",
                    f"{int(row.n_parseable_outputs)}/{int(row.n_unique_tasks)}",
                    f"{int(row.total_tokens):,}",
                ]
                for model, row in admin.set_index("target_model_name").loc[model_order].iterrows()
            ],
            widths="p{0.17\\linewidth}p{0.27\\linewidth}p{0.24\\linewidth}ccc",
        )
    )
    chunks.append(
        table(
            "Task-type results for every complete model run.",
            "tab:supp-task",
            ["Task type", "$n$"] + model_order,
            [
                [task_labels[t], int(tasks.loc[tasks.task_type == t, "n"].iloc[0])]
                + [score(tasks.loc[(tasks.task_type == t) & (tasks.target_model_name == m), "mean_normalized_score"].iloc[0]) for m in model_order]
                for t in task_order
            ],
            widths="p{0.24\\linewidth}c" + "c" * len(model_order),
        )
    )
    chunks.append(
        table(
            "Curve-family results for every complete model run.",
            "tab:supp-family",
            ["Curve family", "$n$"] + model_order,
            [
                [family_labels[f], int(families.loc[families.curve_family == f, "n"].iloc[0])]
                + [score(families.loc[(families.curve_family == f) & (families.target_model_name == m), "mean_normalized_score"].iloc[0]) for m in model_order]
                for f in family_order
            ],
            widths="p{0.24\\linewidth}c" + "c" * len(model_order),
        )
    )
    chunks.append(
        table(
            "Rendering-profile results for every complete model run.",
            "tab:supp-style",
            ["Rendering profile"] + model_order,
            [
                [style_labels[s]]
                + [score(styles.loc[(styles.style_profile_id == s) & (styles.target_model_name == m), "mean_normalized_score"].iloc[0]) for m in model_order]
                for s in style_order
            ],
            widths="p{0.28\\linewidth}" + "c" * len(model_order),
        )
    )
    chunks.append(
        table(
            "All paired cluster-bootstrap differences (row model minus column model).",
            "tab:supp-pairs",
            ["Model A", "Model B", "Difference", "95% CI"],
            [
                [row.model_a, row.model_b, score(row.mean_difference), f"[{score(row.ci_low)}, {score(row.ci_high)}]"]
                for row in pairs.itertuples()
            ],
            widths="p{0.24\\linewidth}p{0.24\\linewidth}cc",
        )
    )
    OUT.write_text("\n".join(chunks), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
