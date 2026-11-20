

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.config import BUDGET, CSV_FILES, get_feature_output_dir
from features.nsga2_ma.annotations import (
    REFERENCE_POINTS_FILENAME,
    SELECTED_POINTS_FILENAME,
    select_cost_breakdown_points,
)


# =========================================================

# =========================================================

from common.plotting import (
    FONT_FAMILY, FONT_SERIF, BASE_FONTSIZE, AX_LABELSIZE, TICK_LABELSIZE,
    AXES_LINEWIDTH, GRID_COLOR, GRID_LINEWIDTH, SPINE_COLOR, DPI_FIG, DPI_SAVE,
    MULTI_AXIS_LABELSIZE, PANEL_CAPTION_FONTSIZE, TITLE_FONTSIZE,
    cost_loss_axis_limits, format_cost_loss_axes,
    normalize_figure_typography, set_plot_style as _set_shared_plot_style,
)

LEGEND_FONTSIZE = 9



FIG4_FIGSIZE = (4, 3.05)
FIG4_XLIM_START = -1
FIG4_YLIM_START = -1
FIG4_XLIM_END = None
FIG4_YLIM_END = None
FIG4_AUTO_MARGIN = 0.05
FIG4_MIN_XSPAN = 1.0
FIG4_MIN_YSPAN = 1.0
FIG4_ADJUST = dict(left=0.10, right=0.98, top=0.995, bottom=0.06)


FIG5_FIGSIZE = (5.6, 3.7)
FIG5_X_LIM = None
FIG5_YLOSS_START = 0
FIG5_YCOST_START = 0
FIG5_YLOSS_END = None
FIG5_YCOST_END = None
FIG5_AUTO_MARGIN = 0.05
FIG5_MIN_YSPAN_LOSS = 1.0
FIG5_MIN_YSPAN_COST = 1.0
FIG5_ADJUST = dict(left=0.10, right=0.98, top=0.995, bottom=0.06)


FIG5_C_PREV = "#F3D7A6"
FIG5_C_DET = "#F5B6B6"
FIG5_C_RESP = "#E97D7D"
FIG5_LEGEND_FONTSIZE = LEGEND_FONTSIZE * 2 / 3
FIG5_SELECTION_STYLES = {
    "knee_ideal_distance": ("#8E3A9D", "D", "Ideal distance"),
    "min_cost_plus_loss": ("#207348", "*", "Min. cost + loss"),
}
FIG5_REFERENCE_STYLES = {
    "all_prevention": ("#9B6300", "s", "All prevention"),
    "all_detection_max_response": ("#006A80", "p", "All detection + max. response"),
}


PARETO_S_ALL = 40
PARETO_S_BEST = 40
PARETO_FACE = "#0000FF"
PARETO_EDGE = "#B3BCF5"


def set_plot_style() -> None:
    
    _set_shared_plot_style(legend_fontsize=LEGEND_FONTSIZE)


def style_standard_axes(ax: plt.Axes) -> None:
    
    ax.grid(
        True,
        color=GRID_COLOR,
        linewidth=GRID_LINEWIDTH,
    )
    for spine in ["bottom", "left", "top", "right"]:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(AXES_LINEWIDTH)


def save_figure(fig, filename, output_dir=None):
    
    output_dir = get_feature_output_dir("nsga2_ma") if output_dir is None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    normalize_figure_typography(fig)
    for extension in ("png", "svg"):
        fig.savefig(
            output_dir / f"{filename}.{extension}",
            bbox_inches="tight",
        )
    plt.close(fig)


def _finite_max(series_list):
    
    maxima = []
    for value in series_list:
        array = np.asarray(value, dtype=float).reshape(-1)
        array = array[np.isfinite(array)]
        if array.size:
            maxima.append(float(array.max()))
    if not maxima:
        return None
    return max(maxima)


def _auto_upper(series_list, start, end, margin, min_span):
    
    if end is not None:
        return float(end)

    data_max = _finite_max(series_list)
    if data_max is None:
        return float(start) + max(min_span, 1.0)

    span = data_max - float(start)
    if span < min_span:
        return float(start) + min_span
    return float(start) + span * (1.0 + margin)


def _read_budget(output_dir=None):
    
    output_dir = get_feature_output_dir("nsga2_ma") if output_dir is None else output_dir
    metadata_path = output_dir / CSV_FILES["nsga2_ma"]["run_metadata"]
    if metadata_path.is_file():
        try:
            frame = pd.read_csv(metadata_path)
            if "Budget" in frame.columns and len(frame) > 0:
                value = float(frame.iloc[0]["Budget"])
                if np.isfinite(value) and value > 0:
                    return value
        except (OSError, ValueError, KeyError, IndexError):
            pass
    return float(BUDGET)


# =========================================================

# =========================================================
def plot_pareto_front(pareto, history, output_dir=None):
    
    if pareto.empty and history.empty:
        return

    fig, ax = plt.subplots(figsize=FIG4_FIGSIZE)
    style_standard_axes(ax)

    
    if not history.empty:
        ax.scatter(
            history["Total_Cost"],
            history["Total_Defender_Loss"],
            s=PARETO_S_ALL,
            facecolors="white",
            edgecolors=PARETO_EDGE,
            linewidths=1.0,
            alpha=0.85,
            label="_nolegend_",
            zorder=2,
        )

    
    if not pareto.empty:
        ax.scatter(
            pareto["Total_Cost"],
            pareto["Total_Defender_Loss"],
            s=PARETO_S_BEST,
            c=PARETO_FACE,
            edgecolors=PARETO_EDGE,
            linewidths=1.5,
            label="Pareto optimal solutions",
            zorder=3,
        )

    
    budget = _read_budget(output_dir)
    ax.axvline(
        x=budget,
        color="green",
        linewidth=1,
        linestyle=(0, (3, 2)),
        label="Budget constraint",
    )

    
    x_series = [pareto["Total_Cost"], history["Total_Cost"], [budget]]
    y_series = [
        pareto["Total_Defender_Loss"],
        history["Total_Defender_Loss"],
    ]
    limits = cost_loss_axis_limits(
        x_series, y_series,
        x_start=FIG4_XLIM_START, y_start=FIG4_YLIM_START,
        x_end=FIG4_XLIM_END, y_end=FIG4_YLIM_END,
        margin=FIG4_AUTO_MARGIN,
        min_xspan=FIG4_MIN_XSPAN, min_yspan=FIG4_MIN_YSPAN,
    )
    format_cost_loss_axes(ax, limits)

    ax.legend(frameon=True, framealpha=1.0, loc="upper right")

    fig.subplots_adjust(**FIG4_ADJUST)
    save_figure(fig, "pareto_front", output_dir)


# =========================================================

# =========================================================
def _cost_to_front_position(total_cost, cost_data):
    
    positions = pd.DataFrame({
        "TotalCost": cost_data["TotalCost"].to_numpy(dtype=float),
        "Position": np.arange(1, len(cost_data) + 1, dtype=float),
    }).groupby("TotalCost", sort=True)["Position"].mean()
    return float(np.interp(
        total_cost, positions.index.to_numpy(), positions.to_numpy(),
        left=0.0, right=float(len(cost_data) + 1),
    ))


def _read_breakdown_annotations(cost_data, output_dir):
    
    selected_path = output_dir / SELECTED_POINTS_FILENAME
    references_path = output_dir / REFERENCE_POINTS_FILENAME
    selected = (
        pd.read_csv(selected_path) if selected_path.is_file()
        else select_cost_breakdown_points(cost_data)
    )
    references = pd.read_csv(references_path) if references_path.is_file() else pd.DataFrame()
    return selected, references


def _draw_selected_breakdown(ax_cost, cost_data, selected):
    
    for _, point in selected.drop_duplicates("Solution_ID").iterrows():
        selection = point["Selection"]
        if selection not in FIG5_SELECTION_STYLES:
            raise ValueError(f"Unknown cost-breakdown point-selection rule: {selection}")
        matched = cost_data.index[cost_data["Solution_ID"] == point["Solution_ID"]]
        if len(matched) != 1:
            raise ValueError("The selection CSV and cost-breakdown CSV disagree; rerun export_csv")
        position = int(matched[0]) + 1
        actual = cost_data.iloc[position - 1]
        values = ["Prevention", "Detection", "Response", "TotalCost", "DefenderLoss"]
        if not np.allclose(actual[values].to_numpy(dtype=float), point[values].to_numpy(dtype=float)):
            raise ValueError("Selection CSV values are stale; rerun export_csv")
        color = "#6B3FA0" if selection == "knee_ideal_distance" else "#287A4B"
        bottom = 0.0
        for category in ("Prevention", "Detection", "Response"):
            amount = float(point[category])
            if amount > 0:
                ax_cost.text(
                    position, bottom + amount / 2, f"{amount:g}",
                    ha="center", va="center", rotation=90, fontsize=TICK_LABELSIZE,
                    color=color, fontweight="bold", zorder=8,
                )
            bottom += amount


def _draw_reference_points(ax_loss, cost_data, references):
    
    for _, point in references.iterrows():
        color, marker, _ = FIG5_REFERENCE_STYLES[point["Reference"]]
        position = _cost_to_front_position(point["TotalCost"], cost_data)
        ax_loss.scatter(
            [position], [point["DefenderLoss"]], s=65, marker=marker,
            color=color, edgecolors="white", linewidths=0.7, zorder=7,
        )


def plot_defense_cost_breakdown(cost_data, output_dir=None, *, selected=None, references=None):
    
    if cost_data.empty:
        return

    cost_data = (
        cost_data.sort_values(["TotalCost", "DefenderLoss", "Solution_ID"], kind="stable")
        .reset_index(drop=True)
    )
    if selected is None or references is None:
        directory = get_feature_output_dir("nsga2_ma") if output_dir is None else output_dir
        stored_selected, stored_references = _read_breakdown_annotations(cost_data, directory)
        selected = stored_selected if selected is None else selected
        references = stored_references if references is None else references

    n = len(cost_data)
    x_range = np.arange(1, n + 1)

    fig, ax_loss = plt.subplots(figsize=FIG5_FIGSIZE)
    ax_cost = ax_loss.twinx()

    style_standard_axes(ax_loss)
    style_standard_axes(ax_cost)

    
    ax_loss.grid(False)
    ax_cost.grid(False)

    
    ax_cost.set_zorder(1)
    ax_loss.set_zorder(2)
    ax_loss.patch.set_alpha(0.0)

    
    bar_w = 0.72
    prev = cost_data["Prevention"].to_numpy(dtype=float)
    det = cost_data["Detection"].to_numpy(dtype=float)
    resp = cost_data["Response"].to_numpy(dtype=float)

    ax_cost.bar(
        x_range, prev,
        width=bar_w,
        color=FIG5_C_PREV,
        edgecolor="black",
        linewidth=0.6,
        label="prevention",
    )
    ax_cost.bar(
        x_range, det,
        width=bar_w,
        bottom=prev,
        color=FIG5_C_DET,
        edgecolor="black",
        linewidth=0.6,
        label="detection",
    )
    ax_cost.bar(
        x_range, resp,
        width=bar_w,
        bottom=prev + det,
        color=FIG5_C_RESP,
        edgecolor="black",
        linewidth=0.6,
        label="response",
    )

    
    loss = cost_data["DefenderLoss"].to_numpy(dtype=float)

    ax_loss.plot(
        x_range, loss,
        color="#4061CD",
        linewidth=1.0,
    )
    ax_loss.scatter(
        x_range, loss,
        s=PARETO_S_BEST,
        c=PARETO_FACE,
        edgecolors=PARETO_EDGE,
        zorder=5,
    )

    _draw_selected_breakdown(ax_cost, cost_data, selected)
    _draw_reference_points(ax_loss, cost_data, references)

    ax_loss.set_xlabel("Pareto-optimal solutions")
    ax_loss.set_ylabel(r"$L(d_{0})$ (\$’000)", color=PARETO_FACE)
    ax_cost.set_ylabel(r"$C(d_{0})$ (\$’000)")

    
    ax_loss.tick_params(axis="y", colors=PARETO_FACE)
    ax_loss.spines["left"].set_color(PARETO_FACE)
    ax_loss.yaxis.label.set_color(PARETO_FACE)

    
    ax_loss.set_xlim(-0.5 if not references.empty else 0, n + 1.5 if not references.empty else n + 1)
    ax_loss.set_ylim(
        FIG5_YLOSS_START,
        _auto_upper(
            [loss, references.get("DefenderLoss", [])], FIG5_YLOSS_START,
            FIG5_YLOSS_END, FIG5_AUTO_MARGIN, FIG5_MIN_YSPAN_LOSS,
        ),
    )
    
    ax_cost.set_ylim(
        FIG5_YCOST_START,
        _auto_upper(
            [cost_data["TotalCost"]], FIG5_YCOST_START,
            FIG5_YCOST_END, FIG5_AUTO_MARGIN, FIG5_MIN_YSPAN_COST,
        ),
    )

    
    handles, labels = ax_cost.get_legend_handles_labels()
    ax_loss.legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=(0.025, 0.48),
        fontsize=FIG5_LEGEND_FONTSIZE,
        frameon=True,
        framealpha=1.0,
        labelspacing=0.2,
        borderpad=0.3,
        handletextpad=0.5,
        handlelength=1.4,
    )

    fig.subplots_adjust(**FIG5_ADJUST)
    save_figure(fig, "defense_cost_breakdown", output_dir)


# =========================================================

# =========================================================
def plot_optimization_history(history, output_dir=None):
    
    if history.empty:
        return

    best = (
        history.groupby(
            ["Generation", "Total_FE"],
            as_index=False,
        )
        .agg(
            Minimum_Cost=("Total_Cost", "min"),
            Minimum_Loss=("Total_Defender_Loss", "min"),
        )
        .sort_values("Total_FE")
    )

    fig, axes = plt.subplots(
        2, 1,
        figsize=(7.5, 6),
        sharex=True,
    )
    axes[0].plot(
        best["Total_FE"],
        best["Minimum_Cost"],
        color="#4DBBD5",
    )
    axes[0].set_ylabel("Minimum cost")
    axes[1].plot(
        best["Total_FE"],
        best["Minimum_Loss"],
        color="#E64B35",
    )
    axes[1].set_ylabel("Minimum loss")
    axes[1].set_xlabel("Function evaluations (FE)")
    for ax in axes:
        style_standard_axes(ax)
    save_figure(fig, "optimization_history_vs_fe", output_dir)


def plot(input_excel=None):
    set_plot_style()
    output_dir = get_feature_output_dir("nsga2_ma", input_excel)

    files = CSV_FILES["nsga2_ma"]
    pareto = pd.read_csv(
        output_dir / files["pareto_front"]
    )
    history = pd.read_csv(
        output_dir / files["optimization_history"]
    )
    costs = pd.read_csv(
        output_dir / files["cost_breakdown_data"]
    )

    plot_pareto_front(pareto, history, output_dir)
    plot_defense_cost_breakdown(costs, output_dir)
    plot_optimization_history(history, output_dir)

    print(f"✅ Single-algorithm plots saved to {output_dir}")


if __name__ == "__main__":
    plot()
