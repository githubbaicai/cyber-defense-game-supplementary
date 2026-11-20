"""Render ablation CSVs, matching native-front axes to the case's Pareto figure."""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from common.config import CSV_FILES
from common.plotting import (
    COST_LOSS_YLABEL, TITLE_FONTSIZE, normalize_figure_typography,
    cost_loss_axis_limits, format_cost_loss_axes,
    save_figure, set_plot_style, style_axes,
)
from .export_csv import TABLE_NAMES, output_directory, validate_tables



NPG_COLORS = ("#4DBBD5", "#E64B35", "#00A087", "#3C5488", "#F39B7F", "#8491B4", "#91D1C2")
COLORS = tuple(
    matplotlib.colors.to_hex(0.8 * np.asarray(matplotlib.colors.to_rgb(color)) + 0.2)
    for color in NPG_COLORS
)

VARIANT_COLORS = dict(zip(
    ("baseline", "no_history", "no_egt", "exposure_zero", "response_half"),
    COLORS,
))

FIXED_LOSS_COLORS = (
    "#6389AE", "#FEB032", "#ED706A", "#91C1CB", "#53AE8D",
    "#C5BD3E", "#BA7798", "#BEA096", "#ABB6B0", "#777777",
)

FIXED_LOSS_FACE_COLORS = ("none",) * len(FIXED_LOSS_COLORS)
FIXED_LOSS_EDGE_COLORS = (
    "#2F79B7", "#E9A216", "#E84B45", "#389EAF", "#249B72",
    "#A6A323", "#AC4E82", "#9A7564", "#6D8D7E", "#626262",
)
FIXED_LOSS_EDGE_WIDTH = 0.75
FIXED_LOSS_HATCH_WIDTH = 0.6
FIXED_LOSS_BAR_WIDTH_SCALE = 2 / 3

HATCHES = ("//", "\\" * 2, "///", "\\" * 3, "xx", "/", "x")

LOLLIPOP_MARKERS = ("o", "s", "^", "D", "v", "h", "X")
LOLLIPOP_GROUP_WIDTH = 0.64

LOLLIPOP_FIGSIZE_MM = (159.2, 88.6)

NATIVE_FRONTS_FIGSIZE_MM = (108.0152, 86.6097)
COMPACT_EXPORT_PAD_MM = 0.5
LOLLIPOP_YLABEL = COST_LOSS_YLABEL
FIGURE_NAMES = (
    "fixed_portfolio_loss", "native_and_common_fronts", "defense_domain_costs",
    "parameter_and_detection_effects", "fixed_portfolio_loss_lollipop", "native_fronts",
)


def _save_compact(figure, directory, name):
    """Crop outer whitespace without scaling any figure elements."""
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        figure.savefig(directory / f"{name}.{extension}", format=extension,
                       bbox_inches="tight", pad_inches=COMPACT_EXPORT_PAD_MM / 25.4)
    plt.close(figure)


def _finish(figure, axes, directory, name, *, compact=False):
    for axis in np.asarray(axes, dtype=object).reshape(-1):
        style_axes(axis)
        if axis.lines or axis.collections:
            axis.legend(frameon=False, fontsize=9)
    normalize_figure_typography(figure)
    figure.tight_layout()
    if compact:
        _save_compact(figure, directory, name)
    else:
        save_figure(figure, directory, name)


def _empty(axis, message):
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)


def _fixed_loss(frame, directory):
    with plt.rc_context({"hatch.linewidth": FIXED_LOSS_HATCH_WIDTH}):
        figure, axis = plt.subplots(figsize=(8, 4.5))
        frame = frame.loc[frame.get("status", "ok").astype(str).eq("ok")].copy()
        if frame.empty:
            _empty(axis, "No fixed-portfolio results")
        else:
            matrix = frame.pivot_table(index="portfolio_id", columns="variant", values="defender_loss", aggfunc="mean", sort=False)
            matrix.plot.bar(
                ax=axis, color=FIXED_LOSS_FACE_COLORS, width=0.8,
                linewidth=FIXED_LOSS_EDGE_WIDTH, zorder=3,
            )
            for index, container in enumerate(axis.containers):
                edge_color = FIXED_LOSS_EDGE_COLORS[index % len(FIXED_LOSS_EDGE_COLORS)]
                for bar in container:
                    width = bar.get_width()
                    bar.set_x(bar.get_x() + width * (1 - FIXED_LOSS_BAR_WIDTH_SCALE) / 2)
                    bar.set_width(width * FIXED_LOSS_BAR_WIDTH_SCALE)
                    bar.set_hatch(HATCHES[index % len(HATCHES)])
                    bar.set_edgecolor(edge_color)
            axis.set_ylim(0, max(float(matrix.max().max()), 1.0) * 1.12)
            axis.tick_params(axis="x", rotation=30)
            axis.legend(
                title="Model variant", loc="upper right", frameon=False,
                fontsize=7.2, title_fontsize=8, labelspacing=0.25,
                handlelength=1.8, handleheight=0.85, handletextpad=0.55, borderaxespad=0.6,
            )
        axis.set_xlabel("Fixed complete portfolio")
        axis.set_ylabel("Expected defender loss")
        _finish(figure, [axis], directory, FIGURE_NAMES[0])


def _fixed_loss_lollipop(frame, directory):
    """Plot recorded fixed portfolios in their original monetary units."""
    figure, axis = plt.subplots(figsize=tuple(value / 25.4 for value in LOLLIPOP_FIGSIZE_MM))
    frame = frame.loc[frame["status"].astype(str).eq("ok")].copy()
    if frame.empty:
        _empty(axis, "No fixed-portfolio results")
    else:
        matrix = frame.pivot_table(
            index="portfolio_id", columns="variant", values="defender_loss",
            aggfunc="mean", sort=False,
        )
        positions = np.arange(len(matrix.index))
        offsets = (np.linspace(-LOLLIPOP_GROUP_WIDTH / 2, LOLLIPOP_GROUP_WIDTH / 2, len(matrix.columns))
                   if len(matrix.columns) > 1 else np.zeros(1))
        for index, (variant, losses) in enumerate(matrix.items()):
            x = positions + offsets[index]
            color = VARIANT_COLORS[variant]
            valid = losses.notna().to_numpy()
            y = losses.to_numpy(dtype=float)[valid]
            axis.vlines(x[valid], 0, y, color=color, linewidth=0.85, alpha=0.65, zorder=2)
            axis.scatter(
                x[valid], y, color=color, alpha=0.8,
                marker=LOLLIPOP_MARKERS[index % len(LOLLIPOP_MARKERS)],
                s=28, edgecolors="white", linewidths=0.45, zorder=3,
                label=str(variant),
            )
        axis.set_xticks(positions)
        axis.set_xticklabels(matrix.index.astype(str))
        axis.tick_params(axis="x", length=0, pad=8)
        axis.set_xlim(-0.55, len(matrix.index) - 0.45)
        axis.set_ylim(0, max(float(matrix.max().max()), 1.0) * 1.12)
        axis.legend(
            title="Model variant", loc="upper right", frameon=False,
            fontsize=7.2, title_fontsize=8, labelspacing=0.3,
            handletextpad=0.55, borderaxespad=0.6,
        )
    axis.set_xlabel("Fixed complete portfolio", labelpad=9)
    axis.set_ylabel(LOLLIPOP_YLABEL)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    style_axes(axis)
    normalize_figure_typography(figure)
    figure.tight_layout()
    
    _save_compact(figure, directory, FIGURE_NAMES[4])


def _fronts(fronts, cross, directory):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, frame, label, title in (
        (axes[0], fronts, "variant", "Each model's native prediction"),
        (axes[1], cross, "source_variant", "Representatives under a common model"),
    ):
        if frame.empty:
            _empty(axis, "No optimized portfolios")
        elif label == "source_variant":
            evaluator = "baseline" if "baseline" in set(frame["evaluation_variant"]) else str(frame["evaluation_variant"].iloc[0])
            selected = frame.loc[frame["evaluation_variant"].eq(evaluator)]
            for variant, group in selected.groupby(label, sort=False):
                axis.scatter(group["purchase_cost"], group["defender_loss"], label=variant, color=VARIANT_COLORS[variant], s=18, alpha=0.8)
            title += f" ({evaluator})"
        else:
            for variant, group in frame.groupby(label, sort=False):
                axis.scatter(group["purchase_cost"], group["defender_loss"], label=variant, color=VARIANT_COLORS[variant], s=18, alpha=0.8)
        axis.set_title(title, fontsize=TITLE_FONTSIZE)
        axis.set_xlabel("Purchase cost")
        axis.set_ylabel("Expected defender loss")
    _finish(figure, axes, directory, FIGURE_NAMES[1])


def _native_front_axis_limits(fronts, directory):
    reference = directory.parent / "nsga2_ma"
    files = CSV_FILES["nsga2_ma"]
    required = ("pareto_front", "optimization_history", "run_metadata")
    if all((reference / files[name]).is_file() for name in required):
        pareto = pd.read_csv(reference / files["pareto_front"])
        history = pd.read_csv(reference / files["optimization_history"])
        metadata = pd.read_csv(reference / files["run_metadata"])
        budget = float(metadata.iloc[0]["Budget"])
        return cost_loss_axis_limits(
            [pareto["Total_Cost"], history["Total_Cost"]],
            [pareto["Total_Defender_Loss"], history["Total_Defender_Loss"]],
            budget,
        )
    budget = None
    manifest_path = directory / CSV_FILES["structural_ablation"]["experiment_manifest"]
    if manifest_path.is_file():
        manifest = pd.read_csv(manifest_path)
        if "resolved_config" in manifest and not manifest.empty:
            budget = json.loads(manifest.iloc[0]["resolved_config"]).get("budget")
    return cost_loss_axis_limits([fronts["purchase_cost"]], [fronts["defender_loss"]], budget)


def _native_fronts(fronts, directory):
    """Render the native-prediction panel as a standalone paper figure."""
    figure, axis = plt.subplots(figsize=tuple(value / 25.4 for value in NATIVE_FRONTS_FIGSIZE_MM))
    if fronts.empty:
        _empty(axis, "No optimized portfolios")
    else:
        for variant, group in fronts.groupby("variant", sort=False):
            axis.scatter(group["purchase_cost"], group["defender_loss"],
                         label=variant, color=VARIANT_COLORS[variant], s=18, alpha=0.8)
    format_cost_loss_axes(axis, _native_front_axis_limits(fronts, directory))
    _finish(figure, [axis], directory, FIGURE_NAMES[5], compact=True)


def _domain_costs(fronts, directory):
    figure, axis = plt.subplots(figsize=(8, 4))
    if fronts.empty:
        _empty(axis, "No optimized portfolios")
    else:
        selected = fronts.sort_values(["defender_loss", "purchase_cost"], kind="stable").groupby(["variant", "seed"], sort=False).head(1)
        grouped = selected.groupby("variant", sort=False)[["prevention_cost", "detection_cost", "response_cost"]].mean()
        grouped.columns = ["Prevention", "Detection", "Response"]
        grouped.plot.bar(stacked=True, ax=axis, color=COLORS[:3], width=0.7)
        axis.tick_params(axis="x", rotation=30)
        axis.legend(frameon=False)
    axis.set_xlabel("Variant (minimum-loss feasible portfolio per seed)")
    axis.set_ylabel("Mean purchase cost by domain")
    _finish(figure, [axis], directory, FIGURE_NAMES[2])


def _parameter_effects(fixed, paths, directory):
    groups = (
        ("exposure_penalty_ratio", ("baseline", "exposure_zero")),
        ("response_effectiveness_multiplier", ("baseline", "response_half")),
    )
    parameters = [(column, variants) for column, variants in groups if column in fixed and fixed[column].notna().any()]
    figure, axes = plt.subplots(1, max(2, len(parameters) + 1), figsize=(4 * max(2, len(parameters) + 1), 4), squeeze=False)
    axes = axes.ravel()
    if not parameters:
        if fixed.empty:
            _empty(axes[0], "No parameter scenarios")
        else:
            means = fixed.groupby("variant", sort=False)["defender_loss"].mean()
            axes[0].bar(means.index.astype(str), means.values, color=COLORS[0])
            axes[0].tick_params(axis="x", rotation=35)
        axes[0].set_xlabel("Variant / parameter scenario")
        axes[0].set_ylabel("Mean fixed-portfolio loss")
    for axis, (parameter, variants) in zip(axes, parameters):
        selected = fixed.loc[fixed[parameter].notna() & fixed["variant"].isin(variants)].copy()
        selected[parameter] = pd.to_numeric(selected[parameter], errors="raise")
        for portfolio, group in selected.groupby("portfolio_id", sort=False):
            curve = group.groupby(parameter)["defender_loss"].mean().sort_index()
            axis.plot(curve.index, curve.values, marker="o", label=str(portfolio))
        axis.set_xlabel(parameter.replace("_", " "))
        axis.set_ylabel("Expected defender loss")
    detection = axes[-1]
    time_column = next((column for column in ("total_attack_time", "current_time", "elapsed_time") if column in paths), None)
    if "network_detection" in paths and time_column is not None and not paths.empty:
        selected_paths = paths.loc[paths["variant"].isin(("baseline", "no_history"))].copy()
        if "source" in selected_paths:
            selected_paths = selected_paths.loc[selected_paths["source"].eq("fixed")]
        portfolio = next((name for name in ("PDR", "DR", "D") if name in set(selected_paths["portfolio_id"])), None)
        selected_paths = selected_paths.loc[selected_paths["portfolio_id"].eq(portfolio)]
        for variant, group in selected_paths.groupby("variant", sort=False):
            selected = group.dropna(subset=[time_column, "network_detection"]).sort_values(time_column)
            if not selected.empty:
                detection.scatter(selected[time_column], selected["network_detection"], s=14, alpha=0.65, label=str(variant))
        detection.set_title(f"Fixed portfolio: {portfolio}", fontsize=TITLE_FONTSIZE)
        detection.set_xlabel("Elapsed time including current attack (h)")
        detection.set_ylabel("Network detection probability")
    else:
        _empty(detection, "Detection diagnostics not exported")
        detection.set_ylabel("Network detection probability")
    _finish(figure, axes, directory, FIGURE_NAMES[3])


def plot(input_excel=None, output_dir=None):
    """Read validated CSVs and save six figures; never load Excel or cached objects."""
    directory = output_directory(input_excel, output_dir)
    tables = {}
    for name in TABLE_NAMES:
        path = directory / CSV_FILES["structural_ablation"][name]
        if not path.is_file():
            raise FileNotFoundError(f"Missing structural-ablation CSV: {path}")
        tables[name] = pd.read_csv(path, dtype={"case_id": str, "input_sha256": str}, low_memory=False)
    tables = validate_tables(tables, expected_input=input_excel)
    fixed_results = tables["fixed_portfolio_results"]
    fixed_ok = fixed_results.loc[fixed_results["status"].astype(str).eq("ok")].copy()
    with plt.rc_context({"hatch.linewidth": 0.28}):
        set_plot_style(legend_fontsize=9)
        _fixed_loss(fixed_ok, directory)
        _fixed_loss_lollipop(fixed_ok, directory)
        _fronts(tables["ablation_fronts"], tables["cross_evaluation"], directory)
        _native_fronts(tables["ablation_fronts"], directory)
        _domain_costs(tables["ablation_fronts"], directory)
        _parameter_effects(fixed_ok, tables["ablation_paths"], directory)
    return [directory / f"{name}.{extension}" for name in FIGURE_NAMES for extension in ("png", "svg")]


def main():
    parser = argparse.ArgumentParser(description="Plot exported structural-ablation CSVs")
    parser.add_argument("--input-excel", default=None)
    parser.add_argument("--output-dir", default=None)
    arguments = parser.parse_args()
    plot(input_excel=arguments.input_excel, output_dir=arguments.output_dir)


if __name__ == "__main__":
    main()
