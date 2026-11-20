"""Plot mapping sensitivity using exported result tables only."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from common.config import get_feature_output_dir
from common.plotting import (
    COST_LOSS_YLABEL, DPI_SAVE, GRID_COLOR, GRID_LINEWIDTH,
    normalize_figure_typography, plot_style, style_axes,
)



PAPER_FIGURE_SIZE = (5.6, 3.6)
PAPER_ADJUST = dict(left=0.13, right=0.985, top=0.89, bottom=0.17,
                    wspace=0.40, hspace=0.72)
PAPER_CAPTION_OFFSET = -30
PAPER_LEGEND_SIZE = 6.5
PAPER_STYLES = {
    "Fixed": dict(facecolor="#DCECF7", edgecolor="#1879B8", hatch="///"),
    "Reoptimized": dict(facecolor="#FDE3E5", edgecolor="#F11C26", hatch="\\" * 3),
}
MAPPING_LABELS = {
    "baseline": r"$e$",
    "square": r"$e^2$",
    "sqrt": r"$\sqrt{e}$",
    "scale_0_8": r"$0.8e$",
}
FIGURE_NAME = "probability_mapping_robustness_paper"


def output_directory(input_excel=None, output_dir=None):
    """Resolve an explicit result directory or this feature's latest export."""
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    root = get_feature_output_dir("probability_mapping", input_excel)
    pointer = root / "latest.json"
    if not pointer.is_file():
        raise FileNotFoundError(f"No probability mapping results: {pointer}")
    with pointer.open(encoding="utf-8-sig") as stream:
        selected = Path(json.load(stream)["output_dir"])
    return (selected if selected.is_absolute() else root / selected).resolve()


def _require_columns(frame, columns, filename):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{filename} missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{filename} is empty")


def read_tables(directory):
    """Reject partial or mismatched comparisons before creating any image."""
    directory = Path(directory)
    summary = pd.read_csv(directory / "mapping_summary.csv", float_precision="round_trip")
    runs = pd.read_csv(directory / "mapping_runs.csv", float_precision="round_trip")
    summary_values = ("Fixed_J", "Fixed_Loss", "Reoptimized_J", "Reoptimized_Loss",
                      "Pool_Best_J", "Candidate_Gap", "Resource_Changes")
    run_values = ("Best_J", "Best_Loss", "Seed")
    _require_columns(summary, ("Mapping_ID", "Mapping_Order", *summary_values), "mapping_summary.csv")
    _require_columns(runs, ("Mapping_ID", "Status", *run_values), "mapping_runs.csv")
    if summary["Mapping_ID"].duplicated().any() or summary["Mapping_Order"].duplicated().any():
        raise ValueError("Duplicate mapping summary rows or order")
    if set(summary["Mapping_ID"]) != set(MAPPING_LABELS):
        raise ValueError("Mapping summary must contain all four configured mappings")
    if set(runs["Mapping_ID"]) != set(MAPPING_LABELS) or not runs["Status"].eq("ok").all():
        raise ValueError("Mapping searches are incomplete or failed")
    if runs.duplicated(["Mapping_ID", "Seed"]).any():
        raise ValueError("Duplicate mapping/seed runs")
    for frame, columns in ((summary, (*summary_values, "Mapping_Order")), (runs, run_values)):
        for column in columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
                raise ValueError(f"Nonfinite {column} in mapping results")
    if (summary[["Candidate_Gap", "Resource_Changes"]] < 0).any().any():
        raise ValueError("Candidate gap and resource changes must be nonnegative")
    if not np.allclose(summary["Candidate_Gap"], summary["Fixed_J"] - summary["Pool_Best_J"],
                       rtol=1e-9, atol=1e-8):
        raise ValueError("Candidate gap does not match same-mapping objective values")
    if runs.groupby("Mapping_ID")["Seed"].apply(lambda values: tuple(sorted(values))).nunique() != 1:
        raise ValueError("Mappings must use matching seed sets")
    for identity in ("Case_ID", "Run_ID", "Experiment_ID"):
        if identity in summary or identity in runs:
            if identity not in summary or identity not in runs:
                raise ValueError(f"Missing result identity {identity}")
            combined = pd.concat([summary[identity], runs[identity]])
            if combined.isna().any() or combined.astype(str).str.strip().eq("").any() or combined.nunique() != 1:
                raise ValueError(f"Mixed or missing result identity {identity}")
    return summary.sort_values("Mapping_Order"), runs.sort_values(["Mapping_ID", "Seed"])


def _format_axis(axis, summary, caption, ylabel):
    positions = np.arange(len(summary))
    axis.set_xticks(positions, [MAPPING_LABELS[value] for value in summary["Mapping_ID"]])
    axis.set_xlim(-0.6, len(summary) - 1 + 0.6)
    axis.set_xlabel("Probability mapping", labelpad=2)
    axis.set_ylabel(ylabel, labelpad=2)
    axis.tick_params(axis="both", length=2.5, pad=2)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
    axis.set_axisbelow(True)
    style_axes(axis)
    axis.grid(True, axis="both", linestyle="-", linewidth=GRID_LINEWIDTH, color=GRID_COLOR)
    axis.annotate(caption, xy=(0.5, 0), xycoords="axes fraction",
                  xytext=(0, PAPER_CAPTION_OFFSET), textcoords="offset points",
                  ha="center", va="top")


def _paired_panel(axis, summary, runs, metric):
    positions = np.arange(len(summary))
    for offset, role in ((-0.18, "Fixed"), (0.18, "Reoptimized")):
        axis.bar(positions + offset, summary[f"{role}_{metric}"], width=0.32,
                 linewidth=0.65, zorder=3, **PAPER_STYLES[role])
    for position, mapping_id in enumerate(summary["Mapping_ID"]):
        values = runs.loc[runs["Mapping_ID"].eq(mapping_id), f"Best_{metric}"].to_numpy()
        jitter = np.linspace(-0.09, 0.09, len(values)) if len(values) > 1 else np.zeros(1)
        axis.scatter(position + 0.18 + jitter, values, s=9, marker="o",
                     facecolors="white", edgecolors="#333333", linewidths=0.55, zorder=4)
    maximum = max(summary[[f"Fixed_{metric}", f"Reoptimized_{metric}"]].to_numpy().max(),
                  runs[f"Best_{metric}"].max())
    minimum = min(0.0, summary[[f"Fixed_{metric}", f"Reoptimized_{metric}"]].to_numpy().min(),
                  runs[f"Best_{metric}"].min())
    axis.set_ylim(minimum, maximum * 1.12 if maximum > 0 else 1.0)


def make_figure(summary, runs):
    """Use landscape panels at half the original area with shared typography."""
    figure, axes = plt.subplots(2, 2, figsize=PAPER_FIGURE_SIZE, squeeze=False)
    figure.subplots_adjust(**PAPER_ADJUST)
    _paired_panel(axes[0, 0], summary, runs, "J")
    _paired_panel(axes[0, 1], summary, runs, "Loss")
    for axis, field in ((axes[1, 0], "Candidate_Gap"), (axes[1, 1], "Resource_Changes")):
        axis.bar(np.arange(len(summary)), summary[field], width=0.5,
                 linewidth=0.65, zorder=3, **PAPER_STYLES["Reoptimized"])
        axis.set_ylim(0, max(float(summary[field].max()) * 1.12, 1.0))
    captions = ("(a) Total objective", "(b) Expected loss",
                "(c) Candidate gap", "(d) Resource changes")
    ylabels = ("Total objective\n" + r"$J(d_{0})$ (\$’000)",
               COST_LOSS_YLABEL.replace(" $L", "\n$L"),
               "Candidate gap\n" + r"(\$’000)", "Resource changes\n(count)")
    for axis, caption, ylabel in zip(axes.flat, captions, ylabels):
        _format_axis(axis, summary, caption, ylabel)
    axes[1, 1].yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    handles = [Patch(**PAPER_STYLES[role], linewidth=0.65, label=label)
               for role, label in (("Fixed", "Fixed decision"), ("Reoptimized", "Merged-front decision"))]
    handles.append(Line2D([], [], color="#333333", marker="o", markerfacecolor="white",
                          markersize=2.5, linewidth=0, label="Per-seed decision"))
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
                  ncol=3, fontsize=PAPER_LEGEND_SIZE, handlelength=1.8,
                  handleheight=1.0, borderpad=0.3, columnspacing=1.2, framealpha=1.0)
    normalize_figure_typography(figure, kind="multi")
    return figure


def _save(figure, directory):
    paths = []
    try:
        with plt.rc_context({"savefig.bbox": None}):
            for extension in ("png", "svg"):
                path = directory / f"{FIGURE_NAME}.{extension}"
                figure.savefig(path, format=extension, bbox_inches=None, pad_inches=0, dpi=DPI_SAVE)
                paths.append(path)
    finally:
        plt.close(figure)
    return paths


def plot(input_excel=None, output_dir=None):
    """Render manuscript SVG/PNG without reading Excel or running a search."""
    directory = output_directory(input_excel, output_dir)
    summary, runs = read_tables(directory)
    with plt.rc_context({**plot_style(kind="multi"), "hatch.linewidth": 0.6}):
        figure = make_figure(summary, runs)
        paths = _save(figure, directory)
    print(f"Probability mapping figures: {directory}")
    return paths
