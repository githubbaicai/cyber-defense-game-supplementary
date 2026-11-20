

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from common.config import (
    CSV_FILES,
    get_feature_output_dir,
)


# =========================================================

# =========================================================

from common.plotting import (
    FONT_FAMILY, FONT_SERIF, BASE_FONTSIZE, AX_LABELSIZE, TICK_LABELSIZE,
    AXES_LINEWIDTH, GRID_COLOR, GRID_LINEWIDTH, SPINE_COLOR, DPI_FIG, DPI_SAVE,
    MULTI_AXIS_LABELSIZE, PANEL_CAPTION_FONTSIZE, TITLE_FONTSIZE,
    normalize_figure_typography, set_plot_style as _set_shared_plot_style,
)

LEGEND_FONTSIZE = 9


def set_plot_style() -> None:
    
    _set_shared_plot_style(legend_fontsize=LEGEND_FONTSIZE)


def style_standard_axes(ax: plt.Axes) -> None:
    
    ax.grid(
        True,
        axis="y",
        color=GRID_COLOR,
        linewidth=GRID_LINEWIDTH,
        linestyle="--",
        zorder=0
    )
    for spine in [
        "bottom",
        "left",
        "top",
        "right"
    ]:
        ax.spines[spine].set_color(
            SPINE_COLOR
        )
        ax.spines[spine].set_linewidth(
            AXES_LINEWIDTH
        )


def save_figure(fig, filename, output_dir=None):
    
    output_dir = get_feature_output_dir("initial_simulation") if output_dir is None else output_dir
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    normalize_figure_typography(fig)
    fig.savefig(
        output_dir / f"{filename}.png",
        format="png",
        bbox_inches="tight"
    )
    fig.savefig(
        output_dir / f"{filename}.svg",
        format="svg",
        bbox_inches="tight"
    )
    plt.close(fig)


def plot(input_excel=None):
    
    set_plot_style()
    output_dir = get_feature_output_dir("initial_simulation", input_excel)

    node_data = pd.read_csv(
        output_dir
        / CSV_FILES["initial_simulation"]["node_metrics"]
    )
    availability = pd.read_csv(
        output_dir
        / CSV_FILES["initial_simulation"]["availability"]
    )
    history = pd.read_csv(
        output_dir
        / CSV_FILES["initial_simulation"][
            "game_strategy_history"
        ]
    )

    merged = node_data.merge(
        availability,
        on="Node_Path",
        how="left"
    )
    merged = merged[
        merged["Node_Path"] != "S"
    ].copy()

    if not merged.empty:
        merged = merged.sort_values(
            "Probability",
            ascending=False
        )
        fig, ax = plt.subplots(
            figsize=(10, 5)
        )
        ax.bar(
            merged["Node_Path"],
            merged["Probability"],
            color="#4DBBD5",
            edgecolor="black",
            linewidth=0.5,
            zorder=3
        )
        ax.set_xlabel("Node / attack path")
        ax.set_ylabel(
            "Cumulative compromise probability"
        )
        ax.set_ylim(0, 1.05)
        ax.tick_params(
            axis="x",
            rotation=45
        )
        style_standard_axes(ax)
        fig.tight_layout()
        save_figure(
            fig,
            "initial_compromise_probability", output_dir
        )

    if not history.empty:
        for node, group in history.groupby(
            "Node_Path",
            sort=False
        ):
            fig, ax = plt.subplots(
                figsize=(10, 6)
            )
            for strategy, line in group.groupby(
                "Strategy",
                sort=False
            ):
                line = line.sort_values(
                    "Generation"
                )
                ax.plot(
                    line["Generation"],
                    line["Probability"],
                    label=strategy,
                    linewidth=1.8
                )

            ax.set_xlabel(
                "Evolutionary Generations (Iterations)"
            )
            ax.set_ylabel(
                "Probability of Selection"
            )
            ax.set_ylim(-0.05, 1.05)
            ax.legend(
                loc="best",
                fontsize=LEGEND_FONTSIZE
            )
            style_standard_axes(ax)
            fig.tight_layout()

            safe_node = "".join(
                char if char.isalnum() else "_"
                for char in str(node)
            )
            save_figure(
                fig,
                f"initial_game_evolution_{safe_node}", output_dir
            )


if __name__ == "__main__":
    plot()
    print(
        "✅ Initial simulation PNG/SVG figures exported."
    )
