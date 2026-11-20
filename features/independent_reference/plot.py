"""Plot independent-reference comparisons from exported CSV files only."""

from pathlib import Path
import json
import re

import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd

from common.config import get_feature_output_dir
from common.plotting import (
    AXES_LINEWIDTH, DPI_SAVE, GRID_COLOR, GRID_LINEWIDTH,
    PANEL_CAPTION_FONTSIZE, SPINE_COLOR, normalize_figure_typography, plot_style,
)



ALGORITHMS = ("NSGA2_MA", "NSGA2", "MOEA/D", "SPEA2")
DISPLAY_NAMES = ("NSGA-II+MA", "NSGA-II", "MOEA/D", "SPEA2")
COLORS = ("#0000ff", "#ff7f0e", "#b276cf", "#2ca02c")
LIGHT_COLORS = ("#b3bcf5", "#f8d9bf", "#ceb8d8", "#b9d4b9")
FIGURE_SIZE = (5.6, 9.2)
PARETO_ADJUST = dict(left=0.08, right=0.99, bottom=0.062, top=0.995,
                     wspace=0.26, hspace=0.34)
PARETO_CAPTION_OFFSET = 28
METRICS_ADJUST = dict(left=0.08, right=0.99, bottom=0.125, top=0.995,
                      wspace=0.34, hspace=0.44)
METRICS_CAPTION_OFFSET = 52
RUN_KEY = ("Experiment_ID", "Protocol", "Budget_ID", "Algorithm", "Run", "Seed")
COORDINATES = ("Total_Cost", "Total_Defender_Loss")
METRICS = ("HV", "IGD", "Run_Wall_Seconds")
METRIC_LABELS = ("Normalized HV", "Normalized IGD", "Wall-clock runtime (s)")


def _require_columns(frame, columns, filename):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{filename} missing columns: {sorted(missing)}")


def _true(series):
    return series.fillna(False).astype(str).str.lower().isin(("true", "1"))


def _valid_rows(frame):
    valid = frame["Status"].eq("ok") & _true(frame["Comparison_Valid"])
    if "Protocol_Completed" in frame:
        valid &= _true(frame["Protocol_Completed"])
    return frame.loc[valid].copy()


def _front_rows(frame, raw):
    _require_columns(frame, (*RUN_KEY, *COORDINATES, "Status", "Comparison_Valid",
                             "Stage", "Front_Kind"), "front CSV")
    selected = _valid_rows(frame)
    selected = selected.loc[selected["Stage"].eq("final") & selected["Front_Kind"].eq("strict")]
    return selected.merge(raw[list(RUN_KEY)], on=list(RUN_KEY), how="inner", validate="many_to_one")


def read_groups(output_dir):
    """Validate one experiment, two FE budgets, and a fixed representative seed."""
    directory = Path(output_dir)
    raw = pd.read_csv(directory / "performance_metrics_raw.csv")
    _require_columns(raw, (*RUN_KEY, "Status", "Comparison_Valid", "HV", "IGD", "Reference_ID"),
                     "performance_metrics_raw.csv")
    if "Run_Wall_Seconds" not in raw:
        _require_columns(raw, ("Runtime_Seconds",), "performance_metrics_raw.csv")
        raw["Run_Wall_Seconds"] = raw["Runtime_Seconds"]
    raw = _valid_rows(raw.loc[raw["Protocol"].eq("equal_fe")])
    if raw.empty or raw["Experiment_ID"].nunique() != 1:
        raise ValueError("Two FE budgets must belong to one valid experiment")
    if raw.duplicated(list(RUN_KEY)).any():
        raise ValueError("Duplicate comparison run identifiers")
    budgets = {}
    for identifier in raw["Budget_ID"].unique():
        match = re.fullmatch(r"fe_([1-9][0-9]*)", str(identifier))
        if match is None:
            raise ValueError(f"Invalid FE budget identifier: {identifier}")
        budgets[int(match.group(1))] = identifier
    if len(budgets) != 2:
        raise ValueError("The 4x2 and 3x2 figures require exactly two FE budgets")
    reference = pd.read_csv(directory / "reference_front.csv")
    _require_columns(reference, (*COORDINATES, "Reference_ID"), "reference_front.csv")
    identifiers = reference["Reference_ID"].dropna().unique()
    if (len(identifiers) != 1 or reference["Reference_ID"].isna().any()
            or raw["Reference_ID"].isna().any()
            or not raw["Reference_ID"].eq(identifiers[0]).all()):
        raise ValueError("Reference_ID differs between metrics and reference front")
    coordinates = reference[list(COORDINATES)].to_numpy(dtype=float)
    if not len(coordinates) or not np.isfinite(coordinates).all():
        raise ValueError("The reference front must contain finite objective values")
    reference = reference.sort_values(list(COORDINATES))
    points = _front_rows(pd.read_csv(directory / "all_pareto_points.csv"), raw)
    best = _front_rows(pd.read_csv(directory / "best_pareto_fronts.csv"), raw)
    seeds = best["Seed"].dropna().unique()
    declared = raw["Representative_Seed"].dropna().unique() if "Representative_Seed" in raw else seeds
    if len(seeds) != 1 or len(declared) != 1 or seeds[0] != declared[0]:
        raise ValueError("All algorithms and budgets must use one fixed representative seed")
    groups = {}
    for budget, identifier in sorted(budgets.items()):
        group = {"raw": raw.loc[raw["Budget_ID"].eq(identifier)].copy(),
                 "points": points.loc[points["Budget_ID"].eq(identifier)].copy(),
                 "representative": best.loc[best["Budget_ID"].eq(identifier)].copy()}
        if any(set(frame["Algorithm"]) != set(ALGORITHMS) for frame in group.values()):
            raise ValueError(f"FE={budget} is missing an algorithm or representative front")
        for field in ("points", "representative"):
            if not np.isfinite(group[field][list(COORDINATES)].to_numpy(dtype=float)).all():
                raise ValueError(f"FE={budget} contains nonfinite objectives")
        groups[budget] = group
    return groups, reference


def _style_axes(axis):
    axis.grid(True, color=GRID_COLOR, linewidth=GRID_LINEWIDTH)
    axis.set_axisbelow("line")
    for spine in axis.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(AXES_LINEWIDTH)


def _save(figure, directory, name):
    normalize_figure_typography(figure, kind="multi")
    try:
        with plt.rc_context({"savefig.bbox": None}):
            for extension in ("svg", "png"):
                figure.savefig(directory / f"{name}.{extension}", bbox_inches=None,
                               pad_inches=0, dpi=DPI_SAVE)
    finally:
        plt.close(figure)


def _pareto_figure(groups, reference, directory, token):
    figure, axes = plt.subplots(4, 2, figsize=FIGURE_SIZE)
    coordinates = np.vstack([reference[list(COORDINATES)].to_numpy(dtype=float),
                             *[group["points"][list(COORDINATES)].to_numpy(dtype=float)
                               for group in groups.values()]])
    lower, upper = coordinates.min(axis=0), coordinates.max(axis=0)
    padding = np.maximum((upper - lower) * 0.06, np.maximum(np.abs(upper), 1) * 0.02)
    for row, algorithm in enumerate(ALGORITHMS):
        for column, (budget, group) in enumerate(groups.items()):
            axis = axes[row, column]
            _style_axes(axis)
            axis.plot(reference[COORDINATES[0]], reference[COORDINATES[1]],
                      color="#aaaaaa", linewidth=0.9, zorder=1)
            for field, face, linewidth, alpha in (("points", "white", 1.0, 0.8),
                                                   ("representative", COLORS[row], 1.2, 1.0)):
                points = group[field].loc[group[field]["Algorithm"].eq(algorithm)]
                axis.scatter(points[COORDINATES[0]], points[COORDINATES[1]],
                             s=40, facecolors=face, edgecolors=LIGHT_COLORS[row],
                             linewidths=linewidth, alpha=alpha,
                             zorder=2 if field == "points" else 3)
            axis.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
            axis.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
            axis.set_xlabel(r"$C(d_0)$ (\$’000)")
            axis.set_ylabel(r"$L(d_0)$ (\$’000)")
            axis.xaxis.labelpad = 2
            axis.yaxis.labelpad = 2
            axis.tick_params(axis="both", which="major", pad=2)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
            axis.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{int(value)}"))
            axis.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{int(value)}"))
            axis.annotate(f"({chr(97 + row * 2 + column)}) {DISPLAY_NAMES[row]}, FE={budget}",
                          xy=(0.5, 0), xycoords="axes fraction",
                          xytext=(0, -PARETO_CAPTION_OFFSET),
                          textcoords="offset points", ha="center", va="top",
                          fontsize=PANEL_CAPTION_FONTSIZE, annotation_clip=False)
    figure.subplots_adjust(**PARETO_ADJUST)
    _save(figure, directory, f"equal_fe_{token}_pareto_fronts_4x2")


def _draw_metric(axis, raw, metric, label):
    _style_axes(axis)
    axis.xaxis.grid(False)
    axis.set_axisbelow(True)
    generator = np.random.default_rng(42)
    for position, algorithm in enumerate(ALGORITHMS, start=1):
        values = pd.to_numeric(raw.loc[raw["Algorithm"].eq(algorithm), metric],
                               errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        color = COLORS[position - 1]
        if not len(values):
            axis.text(position, 0.5, "No valid metric", rotation=90,
                      ha="center", va="center", transform=axis.get_xaxis_transform())
            continue
        if len(values) > 1 and np.ptp(values) > 0:
            violin = axis.violinplot([values], positions=[position], widths=0.72,
                                     showmeans=False, showmedians=False, showextrema=False,
                                     points=150, bw_method="scott")
            body = violin["bodies"][0]
            vertices = body.get_paths()[0].vertices
            vertices[:, 0] = np.minimum(vertices[:, 0], position)
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_linewidth(0.8)
            body.set_alpha(0.23)
            body.set_zorder(2)
        axis.boxplot([values], positions=[position], widths=0.13, patch_artist=True,
                     showfliers=False, showmeans=True, whis=1.5, manage_ticks=False,
                     boxprops={"facecolor": "white", "edgecolor": color, "linewidth": 1.0},
                     medianprops={"color": SPINE_COLOR, "linewidth": 1.2},
                     whiskerprops={"color": color, "linewidth": 0.9},
                     capprops={"color": color, "linewidth": 0.9},
                     meanprops={"marker": "D", "markerfacecolor": color,
                                "markeredgecolor": "white", "markeredgewidth": 0.35,
                                "markersize": 3.4}, zorder=3)
        axis.scatter(position + generator.uniform(0.12, 0.34, len(values)), values,
                     s=5, color=color, edgecolors="none", alpha=0.55, zorder=4)
    axis.set_xticks(range(1, 5), DISPLAY_NAMES, rotation=60, ha="right")
    axis.set_xlim(0.5, 4.5)
    axis.set_ylabel(label)
    axis.margins(y=0.08)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    if metric == "Run_Wall_Seconds":
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{int(round(value)):,}"))
    axis.yaxis.labelpad = 2
    axis.tick_params(axis="both", which="major", pad=2)
    axis.tick_params(axis="y", labelleft=True)


def _metrics_figure(groups, directory, token):
    figure, axes = plt.subplots(3, 2, figsize=FIGURE_SIZE, sharey="row")
    for row, (metric, label) in enumerate(zip(METRICS, METRIC_LABELS)):
        for column, (budget, group) in enumerate(groups.items()):
            axis = axes[row, column]
            _draw_metric(axis, group["raw"], metric, label)
            panel = f"({chr(97 + row * 2 + column)})"
            caption = (f"{panel} Wall-clock runtime\n(FE budget = {budget})"
                       if metric == "Run_Wall_Seconds" else f"{panel} FE={budget}: {metric}")
            axis.annotate(caption,
                          xy=(0.5, 0), xycoords="axes fraction",
                          xytext=(0, -METRICS_CAPTION_OFFSET),
                          textcoords="offset points", ha="center", va="top",
                          fontsize=PANEL_CAPTION_FONTSIZE, annotation_clip=False)
    figure.subplots_adjust(**METRICS_ADJUST)
    _save(figure, directory, f"equal_fe_{token}_metrics_violin_3x2")


def plot(input_excel=None, output_dir=None):
    """Read saved CSVs and preserve the measured runtime of the original searches."""
    directory = (get_feature_output_dir("independent_reference", input_excel)
                 if output_dir is None else Path(output_dir).expanduser().resolve())
    pointer = directory / "latest.json"
    if output_dir is None and pointer.is_file():
        directory = Path(json.loads(pointer.read_text(encoding="utf-8"))["output_dir"])
    groups, reference = read_groups(directory)
    token = "_".join(["fe", *(str(budget) for budget in groups)])
    with plt.rc_context(plot_style(kind="multi", legend_fontsize=8)):
        _pareto_figure(groups, reference, directory, token)
        _metrics_figure(groups, directory, token)
    return directory


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-excel", type=Path)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    plot(input_excel=arguments.input_excel, output_dir=arguments.output_dir)
