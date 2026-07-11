"""Rendering utilities for adversarial psychometric figures."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Mapping, Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .io import ensure_dir


STYLE_PROFILES: list[dict[str, Any]] = [
    {
        "style_profile_id": "matplotlib_light",
        "render_engine": "matplotlib",
        "theme": "light",
        "color_palette": "tab10",
        "gridlines": "standard",
        "legend_position": "best",
        "tick_density": "standard",
        "line_width": 2.4,
        "style_context": "default",
    },
    {
        "style_profile_id": "seaborn_whitegrid",
        "render_engine": "seaborn",
        "theme": "light",
        "color_palette": "deep",
        "gridlines": "standard",
        "legend_position": "right",
        "tick_density": "standard",
        "line_width": 2.4,
        "style_context": "whitegrid",
    },
    {
        "style_profile_id": "ggplot_gray",
        "render_engine": "matplotlib_ggplot_style",
        "theme": "light",
        "color_palette": "muted",
        "gridlines": "heavy",
        "legend_position": "best",
        "tick_density": "dense",
        "line_width": 2.2,
        "style_context": "ggplot",
    },
    {
        "style_profile_id": "grayscale_no_grid",
        "render_engine": "matplotlib",
        "theme": "light",
        "color_palette": "grayscale",
        "gridlines": "none",
        "legend_position": "bottom",
        "tick_density": "sparse",
        "line_width": 2.6,
        "style_context": "default",
    },
    {
        "style_profile_id": "dark_high_contrast",
        "render_engine": "matplotlib",
        "theme": "dark",
        "color_palette": "bright",
        "gridlines": "faint",
        "legend_position": "best",
        "tick_density": "standard",
        "line_width": 2.6,
        "style_context": "dark_background",
    },
    {
        "style_profile_id": "lattice_emulation",
        "render_engine": "matplotlib_lattice_emulation",
        "theme": "light",
        "color_palette": "colorblind",
        "gridlines": "faint",
        "legend_position": "outside",
        "tick_density": "standard",
        "line_width": 2.0,
        "style_context": "classic",
    },
]


def get_style_profiles(style_ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
    if style_ids is None:
        return STYLE_PROFILES
    wanted = set(style_ids)
    profiles = [profile for profile in STYLE_PROFILES if profile["style_profile_id"] in wanted]
    missing = wanted - {profile["style_profile_id"] for profile in profiles}
    if missing:
        raise ValueError(f"Unknown style profile(s): {sorted(missing)}")
    return profiles


@contextmanager
def style_context(profile: Mapping[str, Any]):
    context = profile["style_context"]
    if profile["render_engine"] == "seaborn":
        with sns.axes_style(context):
            yield
    else:
        with plt.style.context(context):
            yield


def palette_for(profile: Mapping[str, Any], n: int) -> list[Any]:
    name = profile["color_palette"]
    if name == "grayscale":
        values = np.linspace(0.15, 0.7, max(n, 2))
        return [(v, v, v) for v in values[:n]]
    if name == "bright":
        return sns.color_palette("bright", n)
    if name == "colorblind":
        return sns.color_palette("colorblind", n)
    if name == "muted":
        return sns.color_palette("muted", n)
    if name == "deep":
        return sns.color_palette("deep", n)
    return sns.color_palette("tab10", n)


def configure_axes(ax: plt.Axes, profile: Mapping[str, Any], xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Ability (theta)")
    if profile["gridlines"] == "none":
        ax.grid(False)
    elif profile["gridlines"] == "heavy":
        ax.grid(True, linewidth=1.2, alpha=0.8)
    elif profile["gridlines"] == "faint":
        ax.grid(True, linewidth=0.5, alpha=0.25)
    else:
        ax.grid(True, linewidth=0.8, alpha=0.45)

    if profile["tick_density"] == "sparse":
        ax.set_xticks([-4, -2, 0, 2, 4])
        y_min, y_max = ylim
        ax.set_yticks(np.linspace(y_min, y_max, 4))
    elif profile["tick_density"] == "dense":
        ax.set_xticks(np.arange(-4, 4.1, 1.0))
        y_min, y_max = ylim
        ax.set_yticks(np.linspace(y_min, y_max, 9))

    if profile["theme"] == "dark":
        ax.set_facecolor("#111111")
        ax.figure.set_facecolor("#111111")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("white")


def place_legend(ax: plt.Axes, profile: Mapping[str, Any]) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    position = profile["legend_position"]
    if position == "outside":
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    elif position == "bottom":
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=min(len(handles), 4), frameon=True)
    elif position == "right":
        ax.legend(loc="center right", frameon=True)
    else:
        ax.legend(loc="best", frameon=True)


def render_stimulus(stimulus: Mapping[str, Any], profile: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    image_dir = ensure_dir(Path(output_dir) / "images")
    theta = np.asarray(stimulus["coordinates"]["theta"], dtype=float)
    series = stimulus["coordinates"]["series"]
    xlim = (float(theta.min()), float(theta.max()))
    y_max = max(max(row["values"]) for row in series)
    y_min = min(min(row["values"]) for row in series)
    padding = max((y_max - y_min) * 0.08, 0.05)
    if stimulus["curve_family"] in {"dichotomous_icc", "polytomous_crc", "unfolding"}:
        ylim = (0.0, 1.02)
    else:
        ylim = (max(0.0, y_min - padding), y_max + padding)

    style_id = profile["style_profile_id"]
    image_name = f"{stimulus['stimulus_id']}__{style_id}.png"
    image_path = image_dir / image_name

    with style_context(profile):
        fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=150)
        colors = palette_for(profile, len(series))
        line_styles = ["-", "--", "-.", ":"]
        for idx, row in enumerate(series):
            ax.plot(
                theta,
                row["values"],
                label=row["label"],
                color=colors[idx],
                linewidth=profile["line_width"],
                linestyle=line_styles[idx % len(line_styles)] if profile["color_palette"] == "grayscale" else "-",
            )

        ax.set_title(stimulus["display_title"])
        ax.set_ylabel(stimulus["y_axis_label"])
        configure_axes(ax, profile, xlim=xlim, ylim=ylim)
        place_legend(ax, profile)
        fig.tight_layout()
        fig.savefig(image_path, bbox_inches="tight")
        plt.close(fig)

    visual = dict(profile)
    visual.update(
        {
            "image_path": str(image_path),
            "image_format": "png",
            "width_px": 1125,
            "height_px": 750,
            "num_rendered_lines": len(series),
            "x_axis": {"label": "Ability (theta)", "min": xlim[0], "max": xlim[1]},
            "y_axis": {"label": stimulus["y_axis_label"], "min": ylim[0], "max": ylim[1]},
        }
    )
    return visual
