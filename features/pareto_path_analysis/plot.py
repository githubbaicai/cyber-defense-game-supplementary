

import argparse
import itertools
import re
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.case_config import get_input_excel
from common.config import CSV_FILES, get_feature_output_dir
from features.pareto_path_analysis.export_csv import SUMMARY_FILES


# =========================================================

# =========================================================

from common.plotting import (
    FONT_FAMILY, FONT_SERIF, BASE_FONTSIZE, AX_LABELSIZE, TICK_LABELSIZE,
    AXES_LINEWIDTH, GRID_COLOR, GRID_LINEWIDTH, SPINE_COLOR, DPI_FIG, DPI_SAVE,
    MULTI_AXIS_LABELSIZE, PANEL_CAPTION_FONTSIZE, TITLE_FONTSIZE,
    normalize_figure_typography, set_plot_style as _set_shared_plot_style,
)

LEGEND_FONTSIZE = 9


FIG7_MARKERS_LIST = ["o", "s", "^", "v", "D", "X", "P", "*"]
FIG7_XLIM = (-1, 502)
FIG7_YLIM = (-0.02, 1.05)
FIG7_MARKEVERY = 8
FIG7_FIGSIZE = (8, 6)
FIG7_COMBINED_FIGSIZE = (8, 9)
FIG7_RESPONSE_LINESTYLE = "--"
FIG7_AUTO_XLIM = True
FIG7_AUTO_MARKEVERY = True
FIG7_ANNOTATE_X = None
FIG7_LEGEND_OUTSIDE = False


SUMMARY_FIGURE_WIDTH = 6.4
SUMMARY_ROW_HEIGHT = 2.7
SUMMARY_CAPTION_Y = -0.225
SUMMARY_CAPTION_FONTSIZE = PANEL_CAPTION_FONTSIZE
SUMMARY_AXIS_LABELSIZE = MULTI_AXIS_LABELSIZE
SUMMARY_TICK_LABELSIZE = TICK_LABELSIZE
SUMMARY_LEGEND_FONTSIZE = 6.5
SUMMARY_ADJUST = dict(
    left=0.10,
    right=0.985,
    top=0.96,
    wspace=0.20,
    hspace=0.45,
)


SUMMARY_LEGEND_LOCATIONS = (
    "center right",
    "upper right",
    "upper left",
    "lower right",
    "lower left",
)


SUMMARY_LEGEND_CLEARANCE_PT = 2.0


COST_BAR_FACE = "#4DBBD5"
BAR_EDGE = "#333333"
LOSS_ORIGINAL = "#E64B35"
LOSS_RESIM = "#3C5488"


RISK_BAR_FACE = "#91D1C2"

FILES = CSV_FILES["pareto_path_analysis"]


def set_plot_style() -> None:
    
    _set_shared_plot_style(legend_fontsize=LEGEND_FONTSIZE)


def style_standard_axes(ax: plt.Axes) -> None:
    
    ax.grid(
        True,
        color=GRID_COLOR,
        linewidth=GRID_LINEWIDTH,
    )

    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(AXES_LINEWIDTH)


def save_figure(
    fig,
    name,
    output_dir=None,
    *,
    tight_layout=True,
):
    
    if output_dir is None:
        output_dir = get_feature_output_dir(
            "pareto_path_analysis"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    normalize_figure_typography(fig)

    if tight_layout:
        fig.tight_layout()

    for extension in ("png", "svg"):
        fig.savefig(
            output_dir / f"{name}.{extension}",
            bbox_inches="tight",
        )

    plt.close(fig)


def _safe_name(value):
    
    return "".join(
        char if char.isalnum() else "_"
        for char in str(value)
    )


def format_strategy_name(strategy: str):
    
    if not isinstance(strategy, str):
        return strategy

    if strategy.strip().lower() == "do nothing":
        return strategy

    return re.sub(
        r"Attack\s+v([A-Za-z]\d*,?\d*)",
        r"Attack v$_{\1}$",
        strategy,
    )


def annotate_final_value_at_x(
    ax: plt.Axes,
    generations: np.ndarray,
    probabilities: np.ndarray,
    x_place=None,
    fmt: str = "{:.2f}",
    eps: float = 0.0,
    text_color=None,
    fontsize: int = TICK_LABELSIZE,
    y_offset_frac: float = 0.03,
) -> None:
    
    generations = np.asarray(generations, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)

    if generations.size == 0 or probabilities.size == 0:
        return

    final_value = float(probabilities[-1])

    if not np.isfinite(final_value):
        return

    rounded_value = round(final_value, 2)

    if abs(rounded_value) <= eps:
        return

    if x_place is None:
        x_value = float(generations[-1])
        horizontal_alignment = "right"
    else:
        index = int(
            np.argmin(np.abs(generations - float(x_place)))
        )
        x_value = float(generations[index])
        horizontal_alignment = "center"

    y_min, y_max = ax.get_ylim()
    y_range = max(y_max - y_min, 1e-12)
    offset = y_offset_frac * y_range

    vertical_alignment = "bottom"
    y_text = final_value + offset

    if y_text > y_max - 0.1 * y_range:
        vertical_alignment = "top"
        y_text = final_value - 2 * offset

    ax.text(
        x_value,
        y_text,
        fmt.format(rounded_value),
        ha=horizontal_alignment,
        va=vertical_alignment,
        fontsize=fontsize,
        color=(
            text_color
            if text_color is not None
            else "black"
        ),
        zorder=10,
        clip_on=True,
        path_effects=[
            pe.withStroke(
                linewidth=3,
                foreground="white",
            )
        ],
    )


def _legend_overlap_count(ax, legend_box, renderer):
    
    overlap_count = 0

    clearance = renderer.points_to_pixels(
        SUMMARY_LEGEND_CLEARANCE_PT
    )

    for line in ax.lines:
        if not line.get_visible():
            continue

        line_padding = line.get_linewidth() / 2

        if line.get_marker() not in (
            None,
            "None",
            "",
            " ",
        ):
            line_padding += line.get_markersize() / 2

        padded_box = legend_box.padded(
            clearance + renderer.points_to_pixels(line_padding)
        )

        path = line.get_path().transformed(
            line.get_transform()
        )

        if path.intersects_bbox(
            padded_box,
            filled=False,
        ):
            overlap_count += 1

    for text in ax.texts:
        if not text.get_visible() or not text.get_text():
            continue

        text_box = text.get_window_extent(renderer)

        if (
            text_box.overlaps(ax.bbox)
            and text_box.overlaps(
                legend_box.padded(clearance)
            )
        ):
            overlap_count += 1

    return overlap_count


def _place_evolution_legends(fig, axes, fontsize):
    
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    legend_options = {
        "fontsize": fontsize,
        "frameon": True,
        "edgecolor": "black",
        "fancybox": True,
        "labelspacing": 0.1,
        "borderpad": 0.3,
    }

    for ax in np.asarray(axes, dtype=object).flat:
        handles, labels = ax.get_legend_handles_labels()

        if not handles:
            continue

        best_location = SUMMARY_LEGEND_LOCATIONS[0]
        best_overlap = float("inf")

        for location in SUMMARY_LEGEND_LOCATIONS:
            legend = ax.legend(
                loc=location,
                **legend_options,
            )

            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()

            overlap = _legend_overlap_count(
                ax,
                legend.get_window_extent(renderer),
                renderer,
            )

            if overlap < best_overlap:
                best_location = location
                best_overlap = overlap

            if overlap == 0:
                break

        ax.legend(
            loc=best_location,
            **legend_options,
        )


def _plot_evolution_panel(
    ax,
    group,
    title,
    linestyle="-",
):
    style_standard_axes(ax)

    first_generation = float(
        group["Generation"].min()
    )
    last_generation = float(
        group["Generation"].max()
    )

    generation_span = max(
        last_generation - first_generation,
        1.0,
    )

    if FIG7_AUTO_XLIM:
        ax.set_xlim(
            first_generation - 0.005 * generation_span,
            last_generation + 0.02 * generation_span,
        )
    else:
        ax.set_xlim(*FIG7_XLIM)

    ax.set_ylim(*FIG7_YLIM)

    marker_cycle = itertools.cycle(
        FIG7_MARKERS_LIST
    )

    for strategy, line in group.groupby(
        "Strategy",
        sort=False,
    ):
        line = line.sort_values("Generation")

        generations = line[
            "Generation"
        ].to_numpy(dtype=float)

        probabilities = line[
            "Probability"
        ].to_numpy(dtype=float)

        marker_interval = (
            max(
                FIG7_MARKEVERY,
                len(generations) // 25,
            )
            if FIG7_AUTO_MARKEVERY
            else FIG7_MARKEVERY
        )

        plotted, = ax.plot(
            generations,
            probabilities,
            linestyle=linestyle,
            linewidth=1.5,
            marker=next(marker_cycle),
            markersize=4,
            markevery=marker_interval,
            label=format_strategy_name(
                str(strategy)
            ),
        )

        annotate_final_value_at_x(
            ax,
            generations,
            probabilities,
            x_place=FIG7_ANNOTATE_X,
            fmt="{:.2f}",
            eps=0.0,
            text_color=plotted.get_color(),
            fontsize=TICK_LABELSIZE,
            y_offset_frac=0.03,
        )

    ax.set_title(
        title,
        fontsize=AX_LABELSIZE,
    )
    ax.set_ylabel("Probability")

    if FIG7_LEGEND_OUTSIDE:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=True,
            edgecolor="black",
            fancybox=True,
            fontsize=LEGEND_FONTSIZE,
            labelspacing=0.1,
        )
    else:
        ax.legend(
            loc="upper right",
            frameon=True,
            edgecolor="black",
            fancybox=True,
            fontsize=LEGEND_FONTSIZE,
            labelspacing=0.1,
        )


def _validate_evolution_history(history):
    if history.empty:
        return

    required = {
        "Selection_Type",
        "Solution_ID",
        "Node_Path",
        "Generation",
        "Strategy",
        "Probability",
        "Player",
        "Probability_Type",
    }

    if not required.issubset(history.columns):
        raise ValueError(
            "The legacy CSV does not contain three-state defense histories; first run "
            "python -m features.pareto_path_analysis.solve --plot"
        )

    for identity, group in history.groupby(
        [
            "Selection_Type",
            "Solution_ID",
            "Node_Path",
        ],
        sort=False,
    ):
        if set(group["Player"]) != {
            "Attacker",
            "Defender",
        }:
            raise ValueError(
                f"{identity} is missing attack or defense-state histories; "
                "rerun solve"
            )

        generation_sets = []

        for player, probability_type in (
            ("Attacker", "attacker_strategy"),
            ("Defender", "defender_state"),
        ):
            selected = group[
                group["Player"] == player
            ]

            if set(selected["Probability_Type"]) != {
                probability_type
            }:
                raise ValueError(
                    f"{identity} uses an incorrect probability convention; "
                    "rerun solve"
                )

            if (
                player == "Defender"
                and selected["Strategy"].nunique()
                not in (3, 4)
            ):
                raise ValueError(
                    f"{identity} must contain three or four defense states; "
                    "rerun solve"
                )

            if selected.duplicated(
                ["Generation", "Strategy"]
            ).any():
                raise ValueError(
                    f"{identity} contains duplicate evolution data"
                )

            distribution = selected.pivot(
                index="Generation",
                columns="Strategy",
                values="Probability",
            ).sort_index()

            probabilities = distribution.to_numpy(dtype=float)

            if (
                not np.isfinite(probabilities).all()
                or np.any(probabilities < 0)
                or np.any(probabilities > 1)
                or not np.allclose(
                    probabilities.sum(axis=1),
                    1,
                    rtol=0,
                    atol=1e-12,
                )
            ):
                raise ValueError(
                    f"{identity} has incomplete or unnormalized probability distributions in one or more generations"
                )

            generations = distribution.index.to_numpy()

            if not np.array_equal(
                generations,
                np.arange(len(distribution)),
            ):
                raise ValueError(
                    f"{identity} evolution data must be recorded continuously from generation 0"
                )

            generation_sets.append(generations)

        if not np.array_equal(
            generation_sets[0],
            generation_sets[1],
        ):
            raise ValueError(
                f"{identity} attack and defense histories have different generation counts"
            )


def _panel_letter(index):
    letters = ""

    while index >= 0:
        index, remainder = divmod(index, 26)
        letters = chr(97 + remainder) + letters
        index -= 1

    return letters


def _plot_summary_panel(
    ax,
    group,
    player,
    last_generation,
    caption,
):
    style_standard_axes(ax)

    generation_span = max(last_generation, 1)
    ax.set_xlim(
        -0.005 * generation_span,
        last_generation + 0.01 * generation_span,
    )
    ax.set_ylim(*FIG7_YLIM)

    ax.set_xticks(
        np.linspace(
            0,
            last_generation,
            5,
        )
    )

    ax.set_xlabel(
        "Generation",
        fontsize=SUMMARY_AXIS_LABELSIZE,
    )
    ax.set_ylabel(
        "Probability",
        fontsize=SUMMARY_AXIS_LABELSIZE,
    )

    ax.tick_params(
        labelsize=SUMMARY_TICK_LABELSIZE,
    )

    ax.set_title(
        f"{player} strategy",
        fontsize=SUMMARY_AXIS_LABELSIZE,
    )

    marker_cycle = itertools.cycle(
        FIG7_MARKERS_LIST
    )

    for strategy, line in group.groupby(
        "Strategy",
        sort=False,
    ):
        line = line.sort_values("Generation")

        generations = line["Generation"].to_numpy(dtype=float)

        probabilities = line[
            "Probability"
        ].to_numpy(dtype=float)

        plotted, = ax.plot(
            generations,
            probabilities,
            linestyle=(
                FIG7_RESPONSE_LINESTYLE
                if player == "Defender"
                else "-"
            ),
            linewidth=1.2,
            marker=next(marker_cycle),
            markersize=3,
            markevery=max(1, len(line) // 25),
            label=format_strategy_name(
                str(strategy)
            ),
        )

        annotate_final_value_at_x(
            ax,
            generations,
            probabilities,
            text_color=plotted.get_color(),
            fontsize=TICK_LABELSIZE,
        )

    ax.text(
        0.5,
        SUMMARY_CAPTION_Y,
        caption,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=SUMMARY_CAPTION_FONTSIZE,
    )


def _plot_summary_figures(output_dir):
    paths = {
        name: output_dir / filename
        for name, filename in SUMMARY_FILES.items()
    }

    if not all(
        path.is_file()
        for path in paths.values()
    ):
        warnings.warn(
            "The effective-step CSV is missing; rerun "
            "python -m features.pareto_path_analysis.solve --plot",
            UserWarning,
            stacklevel=2,
        )
        return

    steps = pd.read_csv(
        paths["summary_step_selection"]
    )
    history = pd.read_csv(
        paths["useful_game_strategy_history"]
    )

    included = steps.loc[
        steps["Included"].eq(True)
    ]

    _validate_evolution_history(history)

    keys = [
        "Selection_Type",
        "Solution_ID",
        "Node_Path",
    ]

    if included.duplicated(keys).any():
        raise ValueError(
            "The effective-step list contains duplicate nodes"
        )

    included_keys = set(
        map(tuple, included[keys].to_numpy())
    )

    history_keys = set(
        map(
            tuple,
            history[keys].drop_duplicates().to_numpy(),
        )
    )

    if included_keys != history_keys:
        raise ValueError(
            "The effective-step list and effective evolution histories disagree; "
            "rerun solve"
        )

    for (
        selection_type,
        solution_id,
    ), group in included.groupby(
        [
            "Selection_Type",
            "Solution_ID",
        ],
        sort=False,
    ):
        group = group.sort_values(
            "Step_Order"
        )
        row_count = len(group)

        fig, axes = plt.subplots(
            row_count,
            2,
            squeeze=False,
            figsize=(
                SUMMARY_FIGURE_WIDTH,
                SUMMARY_ROW_HEIGHT * row_count,
            ),
        )

        fig.subplots_adjust(
            **SUMMARY_ADJUST,
            bottom=0.28 / row_count,
        )

        for row_index, step in enumerate(
            group.itertuples(index=False)
        ):
            node_history = history.loc[
                history["Selection_Type"].eq(
                    selection_type
                )
                & history["Solution_ID"].eq(
                    solution_id
                )
                & history["Node_Path"].eq(
                    step.Node_Path
                )
            ]

            last_generation = float(
                node_history["Generation"].max()
            )

            node_label = "→".join(
                part.strip()
                for part in str(
                    step.Node_Path
                ).split("→")
            )

            for column_index, player in enumerate(
                ("Attacker", "Defender")
            ):
                letter = _panel_letter(
                    2 * row_index + column_index
                )

                caption_kind = (
                    "Attack"
                    if player == "Attacker"
                    else "Defense"
                )

                caption = (
                    f"({letter}) "
                    f"{caption_kind} step {node_label}"
                )

                _plot_summary_panel(
                    axes[row_index, column_index],
                    node_history.loc[
                        node_history["Player"].eq(player)
                    ],
                    player,
                    last_generation,
                    caption,
                )

        _place_evolution_legends(
            fig,
            axes,
            SUMMARY_LEGEND_FONTSIZE,
        )

        save_figure(
            fig,
            (
                f"evolution_{selection_type}_"
                f"{int(solution_id)}_summary"
            ),
            output_dir,
            tight_layout=False,
        )


def plot(input_excel=None):
    
    set_plot_style()

    input_excel = get_input_excel(input_excel)

    output_dir = get_feature_output_dir(
        "pareto_path_analysis",
        input_excel,
    )

    points = pd.read_csv(
        output_dir / FILES["selected_points"]
    )

    risk = pd.read_csv(
        output_dir / FILES["node_risk"]
    )

    history = pd.read_csv(
        output_dir / FILES["game_strategy_history"]
    )

    if points.empty:
        return

    _validate_evolution_history(history)

    # =====================================================
    
    # =====================================================

    labels = points["Selection_Type"].tolist()
    positions = np.arange(len(points))

    fig, left = plt.subplots(figsize=(9, 5))

    left.bar(
        positions - 0.18,
        points["Total_Cost"],
        width=0.36,
        label="Defense cost",
        color=COST_BAR_FACE,
        edgecolor=BAR_EDGE,
        linewidth=0.6,
        zorder=3,
    )

    left.set_ylabel("Defense cost")
    left.set_xticks(positions)
    left.set_xticklabels(labels)
    style_standard_axes(left)

    right = left.twinx()

    right.plot(
        positions,
        points["Original_Defender_Loss"],
        "o-",
        color=LOSS_ORIGINAL,
        label="Original loss",
    )

    right.plot(
        positions,
        points["Resimulated_Defender_Loss"],
        "s--",
        color=LOSS_RESIM,
        label="Resimulated loss",
    )

    right.set_ylabel("Defender loss")
    right.grid(False)

    left_handles, left_labels = (
        left.get_legend_handles_labels()
    )
    right_handles, right_labels = (
        right.get_legend_handles_labels()
    )

    left.legend(
        left_handles + right_handles,
        left_labels + right_labels,
        loc="best",
        frameon=True,
        framealpha=1.0,
    )

    save_figure(
        fig,
        "selected_points_comparison",
        output_dir,
    )

    # =====================================================
    
    # =====================================================

    if not risk.empty:
        valid_risk = risk.dropna(
            subset=["Compromise_Probability"]
        )

        for selection_type, group in valid_risk.groupby(
            "Selection_Type",
            sort=False,
        ):
            group = (
                group[group["Node_Path"] != "S"]
                .sort_values(
                    "Compromise_Probability",
                    ascending=False,
                )
                .head(15)
            )

            if group.empty:
                continue

            fig, ax = plt.subplots(figsize=(11, 5))

            ax.bar(
                group["Node_Path"].astype(str),
                group["Compromise_Probability"],
                color=RISK_BAR_FACE,
                edgecolor=BAR_EDGE,
                linewidth=0.5,
                zorder=3,
            )

            ax.set_ylabel("Compromise probability")
            ax.set_xlabel("Node / path")
            ax.tick_params(axis="x", rotation=45)
            style_standard_axes(ax)

            save_figure(
                fig,
                f"node_risk_{selection_type}",
                output_dir,
            )

    # =====================================================
    
    # =====================================================

    if history.empty:
        return

    for (
        selection_type,
        solution_id,
        node,
    ), group in history.groupby(
        [
            "Selection_Type",
            "Solution_ID",
            "Node_Path",
        ],
        sort=False,
    ):
        attacker = group[
            group["Player"] == "Attacker"
        ]

        defender = group[
            group["Player"] == "Defender"
        ]

        fig, axes = plt.subplots(
            2,
            1,
            figsize=FIG7_COMBINED_FIGSIZE,
            sharex=True,
        )

        _plot_evolution_panel(
            axes[0],
            attacker,
            "Attacker strategies",
        )

        _plot_evolution_panel(
            axes[1],
            defender,
            "Defender detection and response states",
            linestyle=FIG7_RESPONSE_LINESTYLE,
        )

        axes[1].set_xlabel("Generation")

        _place_evolution_legends(
            fig,
            axes,
            LEGEND_FONTSIZE,
        )

        save_figure(
            fig,
            (
                f"evolution_{selection_type}_"
                f"{int(solution_id)}_"
                f"{_safe_name(node)}"
            ),
            output_dir,
        )

    _plot_summary_figures(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument("--input-excel")
    arguments = parser.parse_args()

    plot(input_excel=arguments.input_excel)

    print("✅ Pareto-analysis plots saved.")