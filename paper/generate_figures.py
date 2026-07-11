"""Generate publication figures for the benchmark-design tutorial paper."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
IMAGES = ROOT / "data" / "benchmark" / "images"

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
PINK = "#CC79A7"
RED = "#D55E00"
SKY = "#56B4E9"
INK = "#27323A"
MID = "#65727C"
LIGHT = "#E8EDF0"
PALE_BLUE = "#E7F2F8"
PALE_ORANGE = "#FBF0D5"
PALE_GREEN = "#E4F3ED"
PALE_PINK = "#F5E9F2"


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def rounded_box(ax, xy, width, height, text, face, edge, fontsize=8.5, weight="normal"):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=1.2,
        facecolor=face,
        edgecolor=edge,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        weight=weight,
    )
    return patch


def arrow(ax, start, end, color=MID, connectionstyle="arc3", lw=1.4):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=lw,
            color=color,
            connectionstyle=connectionstyle,
        )
    )


def assessment_vs_benchmark() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.245, 0.96, "Educational or psychological assessment", ha="center", weight="bold", color=BLUE)
    ax.text(0.755, 0.96, "LLM benchmark", ha="center", weight="bold", color=ORANGE)

    left = [
        ("Object", "Person or group"),
        ("Elicitation", "Items under standardized conditions"),
        ("Observation", "Responses and process data"),
        ("Score", "Estimated standing on a construct"),
        ("Use", "Interpretation or decision about people"),
    ]
    right = [
        ("Object", "Model + version + interface"),
        ("Elicitation", "Prompts, inputs, tools, decoding"),
        ("Observation", "Generated outputs, latency, refusals"),
        ("Score", "Performance on a sampled task distribution"),
        ("Use", "Model comparison, diagnosis, or selection"),
    ]

    ys = np.linspace(0.78, 0.16, 5)
    for i, (y, l, r) in enumerate(zip(ys, left, right)):
        rounded_box(ax, (0.04, y - 0.06), 0.41, 0.115, f"{l[0]}\n{l[1]}", PALE_BLUE, BLUE)
        rounded_box(ax, (0.55, y - 0.06), 0.41, 0.115, f"{r[0]}\n{r[1]}", PALE_ORANGE, ORANGE)
        if i < 4:
            arrow(ax, (0.245, y - 0.065), (0.245, ys[i + 1] + 0.065), color=BLUE)
            arrow(ax, (0.755, y - 0.065), (0.755, ys[i + 1] + 0.065), color=ORANGE)

    ax.plot([0.5, 0.5], [0.11, 0.90], color=LIGHT, linewidth=2)
    ax.text(
        0.5,
        0.04,
        "Shared principle: observed performance supports an inference only through an explicit validity argument.",
        ha="center",
        va="center",
        color=INK,
        fontsize=8.5,
    )
    save(fig, "assessment_vs_benchmark")


def claim_evidence_chain() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.65))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    labels = [
        ("Intended use", "What decision will the score inform?"),
        ("Capability claim", "What must the model be able to do?"),
        ("Observable evidence", "What behavior would support the claim?"),
        ("Task and stimulus", "What input should elicit that behavior?"),
        ("Scored response", "How is evidence extracted and aggregated?"),
    ]
    colors = [(PALE_BLUE, BLUE), (PALE_GREEN, GREEN), (PALE_PINK, PINK), (PALE_ORANGE, ORANGE), ("#F2ECE8", RED)]
    xs = np.linspace(0.025, 0.815, 5)
    for i, ((head, body), (face, edge), x) in enumerate(zip(labels, colors, xs)):
        rounded_box(ax, (x, 0.36), 0.16, 0.34, "", face, edge)
        ax.text(x + 0.08, 0.62, fill(head, 15), ha="center", va="center", fontsize=8.0, weight="bold", color=INK)
        ax.text(x + 0.08, 0.48, fill(body, 16), ha="center", va="center", fontsize=7.8, color=INK)
        if i < 4:
            arrow(ax, (x + 0.162, 0.53), (xs[i + 1] - 0.004, 0.53), color=MID)
    ax.text(0.5, 0.86, "Design moves from claims to observations", ha="center", weight="bold", color=INK)
    ax.text(0.5, 0.16, "Validation asks whether each link is defensible", ha="center", weight="bold", color=INK)
    arrow(ax, (0.89, 0.30), (0.11, 0.30), color=PINK, connectionstyle="arc3,rad=-0.16")
    save(fig, "claim_evidence_chain")


def benchmark_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = [
        ("1", "Scope", "Intended use\nand object"),
        ("2", "Define", "Capability\nand evidence"),
        ("3", "Build", "Stimuli, tasks,\nand ground truth"),
        ("4", "Adapt", "Prompt, tools,\nand decoding"),
        ("5", "Assemble", "Sampling and\ntest forms"),
        ("6", "Score", "Parsing, rubric,\nand aggregation"),
        ("7", "Validate", "Robustness, human\nreview, uncertainty"),
        ("8", "Maintain", "Version, monitor,\nrefresh, retire"),
    ]
    positions = [(0.04 + i * 0.24, 0.61) for i in range(4)] + [(0.76 - i * 0.24, 0.19) for i in range(4)]
    edge_colors = [BLUE, GREEN, PINK, ORANGE, RED, SKY, PINK, MID]
    faces = [PALE_BLUE, PALE_GREEN, PALE_PINK, PALE_ORANGE, "#F2ECE8", "#E9F4F8", PALE_PINK, "#EDF0F2"]
    for i, ((n, head, body), (x, y), edge, face) in enumerate(zip(stages, positions, edge_colors, faces)):
        rounded_box(ax, (x, y), 0.19, 0.22, f"{head}\n{body}", face, edge, fontsize=8.2)
        ax.add_patch(Circle((x + 0.018, y + 0.202), 0.025, facecolor=edge, edgecolor="white", linewidth=0.8))
        ax.text(x + 0.018, y + 0.202, n, ha="center", va="center", color="white", fontsize=8, weight="bold")
        if i < len(stages) - 1:
            x2, y2 = positions[i + 1]
            if i == 3:
                arrow(ax, (x + 0.095, y - 0.015), (x2 + 0.095, y2 + 0.235), color=MID)
            elif i < 3:
                arrow(ax, (x + 0.193, y + 0.11), (x2 - 0.003, y2 + 0.11), color=MID)
            else:
                arrow(ax, (x - 0.003, y + 0.11), (x2 + 0.193, y2 + 0.11), color=MID)
    ax.text(0.5, 0.93, "An LLM benchmark is a maintained measurement system, not just a question file", ha="center", weight="bold", color=INK)
    save(fig, "benchmark_pipeline")


def design_matrix() -> None:
    capabilities = ["Identify model", "Estimate parameters", "Read probability", "Locate information", "Reason across categories", "Detect non-monotonicity"]
    families = ["Dichotomous ICC", "Item information", "Polytomous curves", "Test characteristic", "Test information", "Ideal-point curve"]
    matrix = np.array(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 0, 0, 1, 0, 0],
            [1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 0, 0, 1, 0, 0],
            [1, 0, 1, 0, 0, 1],
        ]
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.05))
    cmap = plt.matplotlib.colors.ListedColormap(["#F1F3F4", GREEN])
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(capabilities)), labels=[fill(x, 14) for x in capabilities])
    ax.set_yticks(np.arange(len(families)), labels=families)
    ax.tick_params(axis="x", rotation=0, pad=7)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, "included" if matrix[i, j] else "", ha="center", va="center", color="white", fontsize=7.4, weight="bold")
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title("The content specification crosses visual objects with observable capabilities", pad=12, weight="bold", color=INK)
    ax.set_xlabel("Capability sampled by the task", labelpad=10)
    ax.set_ylabel("Psychometric visual object")
    fig.tight_layout()
    save(fig, "design_matrix")


def factorial_expansion() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = [
        (0.03, "14", "base mathematical\nstimuli", BLUE, PALE_BLUE),
        (0.30, "x 6", "rendering\nprofiles", ORANGE, PALE_ORANGE),
        (0.57, "84", "image-level\nstimuli", GREEN, PALE_GREEN),
        (0.80, "198", "scored\ntasks", PINK, PALE_PINK),
    ]
    widths = [0.17, 0.17, 0.17, 0.17]
    for i, ((x, value, label, edge, face), w) in enumerate(zip(stages, widths)):
        rounded_box(ax, (x, 0.29), w, 0.30, "", face, edge)
        ax.text(x + w / 2, 0.48, value, ha="center", va="center", fontsize=17, weight="bold", color=edge)
        ax.text(x + w / 2, 0.36, label, ha="center", va="center", fontsize=8.4, color=INK)
        if i < 3:
            arrow(ax, (x + w + 0.01, 0.44), (stages[i + 1][0] - 0.01, 0.44), color=MID)
    ax.text(0.5, 0.84, "Programmatic generation separates mathematical content from visual presentation", ha="center", weight="bold", color=INK)
    ax.text(0.5, 0.10, "A full local pipeline run then created 594 responses and judgments across three diagnostic baselines.", ha="center", color=MID)
    save(fig, "factorial_expansion")


def instance_anatomy() -> None:
    img = plt.imread(IMAGES / "3pl_001_icc__matplotlib_light.png")
    fig = plt.figure(figsize=(7.2, 5.0))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1, 1], wspace=0.14, hspace=0.18)
    ax_img = fig.add_subplot(grid[:, 0])
    ax_img.imshow(img)
    ax_img.axis("off")
    ax_img.set_title("1. Stimulus shown to the model", loc="left", weight="bold", color=BLUE)
    ax_prompt = fig.add_subplot(grid[0, 1])
    ax_score = fig.add_subplot(grid[1, 1])
    for ax in [ax_prompt, ax_score]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    rounded_box(
        ax_prompt,
        (0.02, 0.08),
        0.96,
        0.80,
        "Estimate a, b, c, and d from the item\ncharacteristic curve.\n\nReturn JSON with the requested fields\nand a short rationale.",
        PALE_ORANGE,
        ORANGE,
        fontsize=8.0,
    )
    ax_prompt.set_title("2. Standardized elicitation", loc="left", weight="bold", color=ORANGE)
    rounded_box(
        ax_score,
        (0.02, 0.51),
        0.96,
        0.40,
        '{"a": 1.42,  "b": -0.36,\n "c": 0.18,  "d": 1.00}',
        PALE_GREEN,
        GREEN,
        fontsize=7.8,
    )
    rounded_box(
        ax_score,
        (0.02, 0.08),
        0.96,
        0.33,
        "Compare each field with stored ground truth.\nRetain parsing failures, rationale score,\nmodel version, prompt, and latency.",
        PALE_PINK,
        PINK,
        fontsize=7.4,
    )
    ax_score.set_title("3. Structured observation and scoring", loc="left", weight="bold", color=GREEN)
    fig.suptitle("An individual benchmark instance is a complete evidence record", y=0.99, weight="bold", color=INK)
    save(fig, "instance_anatomy")


def style_invariance() -> None:
    styles = [
        ("matplotlib_light", "Matplotlib"),
        ("seaborn_whitegrid", "Seaborn"),
        ("ggplot_gray", "ggplot-like"),
        ("grayscale_no_grid", "Grayscale"),
        ("dark_high_contrast", "Dark"),
        ("lattice_emulation", "Lattice-like"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8))
    for ax, (style, label) in zip(axes.flat, styles):
        img = plt.imread(IMAGES / f"2pl_001_icc__{style}.png")
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(label, fontsize=9, weight="bold", color=INK, pad=3)
    fig.suptitle("The mathematics is fixed while the rendering changes", weight="bold", color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=0.7, w_pad=0.5)
    save(fig, "style_invariance")


def scoring_logic() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    err = np.linspace(0, 0.65, 260)
    score = np.piecewise(err, [err <= 0.15, (err > 0.15) & (err <= 0.30), (err > 0.30) & (err <= 0.50), err > 0.50], [3, 2, 1, 0])
    ax.step(err, score, where="post", color=BLUE, linewidth=2.4)
    ax.fill_between(err, score, step="post", alpha=0.12, color=BLUE)
    for x, label in [(0.15, "precise"), (0.30, "approx."), (0.50, "poor")]:
        ax.axvline(x, color=MID, linewidth=0.8, linestyle="--")
        ax.text(x - 0.006, 3.2, label, rotation=90, va="bottom", ha="right", fontsize=7.5, color=MID)
    ax.set_xlim(0, 0.65)
    ax.set_ylim(-0.15, 3.65)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xlabel("Absolute numeric error")
    ax.set_ylabel("Field score")
    ax.set_title("Numeric fields use tolerance bands", weight="bold", color=INK)
    ax.spines[["top", "right"]].set_visible(False)

    ax2 = axes[1]
    components = ["Answer\nfields", "Visual\nrationale", "Normalized\ntotal"]
    heights = [3.0, 2.0, 5.0]
    colors = [GREEN, ORANGE, PINK]
    ax2.bar(components, heights, color=colors, width=0.62)
    for i, h in enumerate(heights):
        ax2.text(i, h + 0.12, f"up to {h:.0f}", ha="center", fontsize=8, color=INK)
    ax2.set_ylim(0, 5.8)
    ax2.set_ylabel("Illustrative points")
    ax2.set_title("Evidence is retained by component", weight="bold", color=INK)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=2.2)
    save(fig, "scoring_logic")


def validation_profiles() -> None:
    by_model = pd.read_csv(ROOT / "output" / "analysis" / "summary_by_model.csv")
    by_task = pd.read_csv(ROOT / "output" / "analysis" / "summary_by_task_type.csv")
    rename_models = {
        "local_oracle_baseline": "Oracle",
        "local_noisy_baseline": "Noisy",
        "local_blind_baseline": "Blind",
    }
    order = ["Oracle", "Noisy", "Blind"]
    palette = {"Oracle": GREEN, "Noisy": ORANGE, "Blind": MID}
    by_model["model"] = by_model["target_model_name"].map(rename_models)
    by_task["model"] = by_task["target_model_name"].map(rename_models)
    task_labels = {
        "category_curve_reasoning": "Categories",
        "information_peak": "Information",
        "model_identification": "Identify",
        "non_monotonic_peak": "Non-monotonic",
        "parameter_estimation": "Parameters",
        "probability_at_theta_0.0": "Probability",
        "tcc_expected_score": "Test score",
    }
    by_task["task"] = by_task["task_type"].map(task_labels)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.75), gridspec_kw={"width_ratios": [0.8, 1.8]})
    ax = axes[0]
    vals = [float(by_model.loc[by_model.model == m, "mean_normalized_score"].iloc[0]) for m in order]
    ax.bar(order, vals, color=[palette[m] for m in order], width=0.68)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.025, f"{v:.2f}", ha="center", fontsize=8.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Mean normalized score")
    ax.set_title("Pipeline separation", weight="bold", color=INK)
    ax.spines[["top", "right"]].set_visible(False)

    ax2 = axes[1]
    task_order = ["Identify", "Parameters", "Probability", "Information", "Categories", "Test score", "Non-monotonic"]
    x = np.arange(len(task_order))
    width = 0.24
    for offset, model in zip([-width, 0, width], order):
        subset = by_task[by_task.model == model].set_index("task")
        vals2 = [float(subset.loc[t, "mean_normalized_score"]) for t in task_order]
        ax2.bar(x + offset, vals2, width=width, color=palette[model], label=model)
    ax2.set_xticks(x, labels=task_order, rotation=32, ha="right")
    ax2.set_ylim(0, 1.12)
    ax2.set_ylabel("Mean normalized score")
    ax2.set_title("Separation across task families", weight="bold", color=INK)
    ax2.legend(frameon=False, ncol=3, loc="upper center", fontsize=7.5)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=2.2)
    save(fig, "validation_profiles")


def threats_to_inference() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    nodes = [
        (0.04, "Stimulus"),
        (0.24, "Prompting"),
        (0.44, "Model output"),
        (0.64, "Scoring"),
        (0.84, "Claim"),
    ]
    for i, (x, label) in enumerate(nodes):
        rounded_box(ax, (x, 0.43), 0.12, 0.15, label, "#F5F6F7", MID, fontsize=8.4, weight="bold")
        if i < len(nodes) - 1:
            arrow(ax, (x + 0.122, 0.505), (nodes[i + 1][0] - 0.003, 0.505), color=MID)

    threats = [
        (0.10, 0.79, "Contamination", "Prior exposure can mimic capability", RED),
        (0.30, 0.17, "Prompt sensitivity", "Wording or option order changes behavior", ORANGE),
        (0.50, 0.79, "Stochasticity", "Repeated calls need not agree", BLUE),
        (0.70, 0.17, "Parser or judge bias", "The scorer can create apparent errors", PINK),
        (0.90, 0.79, "Overgeneralization", "A score travels beyond its sampled domain", GREEN),
    ]
    for x, y, head, body, color in threats:
        ax.add_patch(Circle((x, y), 0.035, facecolor=color, edgecolor="white"))
        ax.text(x, y, "!", ha="center", va="center", color="white", weight="bold", fontsize=11)
        align_y = y - 0.065 if y > 0.5 else y + 0.065
        va = "top" if y > 0.5 else "bottom"
        ax.text(x, align_y, f"{head}\n{fill(body, 27)}", ha="center", va=va, fontsize=7.8, color=INK, weight="bold")
        target_y = 0.585 if y > 0.5 else 0.425
        arrow(ax, (x, y - 0.04 if y > 0.5 else y + 0.04), (x, target_y), color=color, lw=1.0)
    ax.text(0.5, 0.95, "Threats can enter at every link between an input and an inference", ha="center", weight="bold", color=INK)
    save(fig, "threats_to_inference")


def real_model_administration() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.85))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    stages = [
        (0.02, "Frozen form", "Image + prompt\n+ task identifier", BLUE, PALE_BLUE),
        (0.215, "Adapter", "Same input\ncontract for\neach provider", ORANGE, PALE_ORANGE),
        (0.41, "Versioned model", "Requested + returned\nmodel identifiers", GREEN, PALE_GREEN),
        (0.605, "Evidence record", "Raw output, settings,\nlatency, tokens, errors", PINK, PALE_PINK),
        (0.80, "Scoring", "Parser + fixed rubric\n+ uncertainty", RED, "#F2ECE8"),
    ]
    for i, (x, head, body, edge, face) in enumerate(stages):
        rounded_box(ax, (x, 0.45), 0.16, 0.24, "", face, edge)
        ax.text(x + 0.08, 0.62, head, ha="center", va="center", fontsize=7.8, weight="bold", color=INK)
        ax.text(x + 0.08, 0.52, body, ha="center", va="center", fontsize=6.8, color=INK)
        if i < len(stages) - 1:
            arrow(ax, (x + 0.162, 0.57), (stages[i + 1][0] - 0.003, 0.57), color=MID)

    ax.text(0.50, 0.90, "Administering a benchmark to actual models is a controlled study, not a manual chat", ha="center", weight="bold", color=INK)
    ax.text(
        0.50,
        0.28,
        "Repeat across models and administrations",
        ha="center",
        weight="bold",
        color=BLUE,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.5},
    )
    arrow(ax, (0.88, 0.40), (0.12, 0.40), color=BLUE, connectionstyle="arc3,rad=-0.15")
    ax.text(
        0.50,
        0.10,
        "Use diagnostic baselines to test mechanics; use real model calls only when making claims about model performance.",
        ha="center",
        color=MID,
        fontsize=8.2,
    )
    save(fig, "real_model_administration")


def maintenance_cycle() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = [
        (0.50, 0.82, "Release", "Version data, code, prompts"),
        (0.80, 0.61, "Monitor", "Track exposure, drift, saturation"),
        (0.69, 0.25, "Refresh", "Add forms and targeted failures"),
        (0.31, 0.25, "Revalidate", "Recheck scoring and interpretations"),
        (0.20, 0.61, "Retire", "Stop unsupported comparisons"),
    ]
    colors = [BLUE, ORANGE, GREEN, PINK, RED]
    faces = [PALE_BLUE, PALE_ORANGE, PALE_GREEN, PALE_PINK, "#F2ECE8"]
    for i, ((x, y, head, body), color, face) in enumerate(zip(stages, colors, faces)):
        rounded_box(ax, (x - 0.10, y - 0.07), 0.20, 0.14, f"{head}\n{body}", face, color, fontsize=8)
        x2, y2, _, _ = stages[(i + 1) % len(stages)]
        arrow(ax, (x + 0.09 * np.sign(x2 - x), y - 0.04), (x2 - 0.09 * np.sign(x2 - x), y2 + 0.04), color=MID, connectionstyle="arc3,rad=0.08")
    ax.add_patch(Circle((0.50, 0.50), 0.12, facecolor="#F5F6F7", edgecolor=MID, linewidth=1.2))
    ax.text(0.50, 0.50, "Benchmark\nvalidity window", ha="center", va="center", weight="bold", color=INK)
    ax.text(0.5, 0.96, "Publication begins the benchmark lifecycle; it does not end it", ha="center", weight="bold", color=INK)
    save(fig, "maintenance_cycle")


def main() -> None:
    setup()
    assessment_vs_benchmark()
    claim_evidence_chain()
    benchmark_pipeline()
    design_matrix()
    factorial_expansion()
    instance_anatomy()
    style_invariance()
    scoring_logic()
    validation_profiles()
    real_model_administration()
    threats_to_inference()
    maintenance_cycle()
    print(f"Wrote 12 figures to {OUT}")


if __name__ == "__main__":
    main()
