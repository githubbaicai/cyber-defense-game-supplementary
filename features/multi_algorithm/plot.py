

from pathlib import Path
import re
from time import perf_counter

import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

from common.config import CSV_FILES, get_feature_output_dir


# =========================================================

# =========================================================

from common.plotting import (
    FONT_FAMILY, FONT_SERIF, BASE_FONTSIZE, AX_LABELSIZE, TICK_LABELSIZE,
    AXES_LINEWIDTH, GRID_COLOR, GRID_LINEWIDTH, SPINE_COLOR, DPI_FIG, DPI_SAVE,
    MULTI_AXIS_LABELSIZE, PANEL_CAPTION_FONTSIZE, TITLE_FONTSIZE,
    normalize_figure_typography, set_plot_style as _set_shared_plot_style,
)

LEGEND_FONTSIZE = 8


PARETO_S_ALL = 40
PARETO_S_BEST = 40


ALGO_ORDER = ["NSGA2_MA", "NSGA2", "MOEA/D", "SPEA2"]

ALGO_COLORS = {
    "NSGA2":    {"Best": "#ff7f0e"},   # orange
    "SPEA2":    {"Best": "#2ca02c"},   # green
    "MOEA/D":   {"Best": "#b276cf"},   # purple
    "NSGA2_MA": {"Best": "#0000FF"},   # blue (anchor)
}

EDGE_BEST = {
    "NSGA2":    {"Best": "#f8d9bf"},
    "SPEA2":    {"Best": "#b9d4b9"},
    "MOEA/D":   {"Best": "#ceb8d8"},
    "NSGA2_MA": {"Best": "#B3BCF5"},   # light blue edge (anchor)
}


ALGO_DISPLAY = {
    "NSGA2_MA": "NSGA-II+MA",
    "NSGA2":    "NSGA-II",
    "SPEA2":    "SPEA2",
    "MOEA/D":   "MOEA/D",
}


FIG8_XLIM = (-1, 65)
FIG8_YLIM = (-1, 150)
FIG8_FIGSIZE = (6.4, 5.4)
FIG8_ADJUST = dict(left=0.08, right=0.98, top=0.995, bottom=0.06,
                   wspace=0.22, hspace=0.35)

CAPTION_FONTSIZE = PANEL_CAPTION_FONTSIZE
CAPTION_Y_FIG8 = -0.22

COMBINED_BUDGETS = (500, 1000)
COMBINED_METRICS = ("HV", "IGD", "Run_Wall_Seconds")
COMBINED_METRIC_TITLES = {"HV": "HV", "IGD": "IGD", "Run_Wall_Seconds": "Wall-clock runtime"}
COMBINED_FIGURE_WIDTH = 5.6
COMBINED_PARETO_HEIGHT = 9.2
COMBINED_METRICS_HEIGHT = 9.2
COMBINED_PARETO_ADJUST = dict(left=0.08, right=0.99, bottom=0.062, top=0.995,
                              wspace=0.26, hspace=0.34)
COMBINED_PARETO_CAPTION_OFFSET = 28
COMBINED_METRICS_ADJUST = dict(left=0.08, right=0.99, bottom=0.125, top=0.995,
                               wspace=0.34, hspace=0.44)
COMBINED_METRICS_CAPTION_OFFSET = 52


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


def save_figure(fig, name, output_dir=None):
    
    output_dir = get_feature_output_dir("multi_algorithm") if output_dir is None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    normalize_figure_typography(fig)
    for extension in ("png", "svg"):
        fig.savefig(
            output_dir / f"{name}.{extension}",
            bbox_inches="tight",
        )
    plt.close(fig)


def save_paper_figure(figure, output_dir, name):
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalize_figure_typography(figure, kind="multi")
    try:
        with plt.rc_context({"savefig.bbox": None}):
            for extension in ("svg", "png"):
                figure.savefig(output_dir / f"{name}.{extension}", bbox_inches=None,
                               pad_inches=0, dpi=DPI_SAVE)
    finally:
        plt.close(figure)


def _int_formatter(x, pos) -> str:
    try:
        return f"{int(x)}"
    except Exception:
        return str(x)


def _comma_int_formatter(x, pos) -> str:
    try:
        return f"{int(round(x)):,}"
    except Exception:
        return str(x)


# =========================================================

# =========================================================
def _plot_one_algorithm_on_ax(ax, df_all, df_best, algorithm, axis_limits=None):
    
    data_all = df_all[df_all["Algorithm"] == algorithm]
    data_best = df_best[df_best["Algorithm"] == algorithm]

    face = ALGO_COLORS.get(algorithm, {}).get("Best", "#0000FF")
    edge = EDGE_BEST.get(algorithm, {}).get("Best", "#B3BCF5")

    style_standard_axes(ax)

    
    if not data_all.empty:
        ax.scatter(
            data_all["Total_Cost"],
            data_all["Total_Defender_Loss"],
            s=PARETO_S_ALL,
            facecolors="white",
            edgecolors=edge,
            linewidths=1.0,
            alpha=0.8,
            label="All valid runs (strict)",
            zorder=2,
        )

    
    if not data_best.empty:
        ax.scatter(
            data_best["Total_Cost"],
            data_best["Total_Defender_Loss"],
            s=PARETO_S_BEST,
            facecolors=face,
            edgecolors=edge,
            linewidths=1.2,
            label="Representative run (strict)",
            zorder=3,
        )
    if data_all.empty and data_best.empty:
        ax.text(0.5, 0.5, "No valid final front", transform=ax.transAxes,
                ha="center", va="center", fontsize=LEGEND_FONTSIZE)
    elif data_best.empty:
        ax.text(0.5, 0.95, "Representative seed unavailable", transform=ax.transAxes,
                ha="center", va="top", fontsize=LEGEND_FONTSIZE)
    if not data_all.empty or not data_best.empty:
        ax.legend(fontsize=LEGEND_FONTSIZE)

    ax.set_xlabel(r"$C(d_0)$ (\$’000)")
    ax.set_ylabel(r"$L(d_0)$ (\$’000)")

    xlim, ylim = (FIG8_XLIM, FIG8_YLIM) if axis_limits is None else axis_limits
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.xaxis.set_major_formatter(FuncFormatter(_int_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(_int_formatter))


def plot_pareto_fronts_4in1(all_points, best_fronts, output_dir=None,
                          name="best_pareto_fronts", title=None):
    
    fig, axes = plt.subplots(
        2, 2,
        figsize=FIG8_FIGSIZE,
        constrained_layout=False,
    )
    axes = axes.flatten()
    points = pd.concat([all_points, best_fronts], ignore_index=True)
    coordinates = points.reindex(columns=["Total_Cost", "Total_Defender_Loss"]).to_numpy(dtype=float)
    coordinates = coordinates[np.isfinite(coordinates).all(axis=1)]
    if len(coordinates):
        lower = coordinates.min(axis=0)
        upper = coordinates.max(axis=0)
        padding = np.maximum((upper - lower) * 0.06, np.maximum(np.abs(upper), 1.0) * 0.02)
        axis_limits = tuple((float(lower[index] - padding[index]),
                             float(upper[index] + padding[index])) for index in range(2))
    else:
        axis_limits = (FIG8_XLIM, FIG8_YLIM)

    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    for i, algo in enumerate(ALGO_ORDER):
        ax = axes[i]
        _plot_one_algorithm_on_ax(
            ax, all_points, best_fronts, algo, axis_limits
        )
        
        ax.text(
            0.5, CAPTION_Y_FIG8,
            f"{panel_labels[i]} {ALGO_DISPLAY.get(algo, algo)}",
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=CAPTION_FONTSIZE,
        )

    
    for j in range(len(ALGO_ORDER), 4):
        axes[j].axis("off")

    fig.subplots_adjust(**FIG8_ADJUST)
    if title:
        fig.suptitle(title, fontsize=BASE_FONTSIZE)
        fig.subplots_adjust(top=0.94)
    save_figure(fig, name, output_dir)


# =========================================================

# =========================================================
def plot_summary_bars(summary, output_dir=None, prefix=""):
    if summary.empty:
        return

    for metric, ylabel in (
        ("Pareto_Size", "Strict Pareto front size"),
        ("HV", "Normalized HV"),
        ("IGD", "Normalized IGD"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4))

        mean_col = f"{metric}_Mean"
        std_col = f"{metric}_Std"
        if mean_col not in summary:
            plt.close(fig)
            continue

        
        ordered = summary.set_index("Algorithm").reindex(
            [a for a in ALGO_ORDER if a in set(summary["Algorithm"])]
        ).reset_index()

        means = ordered[mean_col].to_numpy(dtype=float)
        deviations = (ordered[std_col].fillna(0.0).to_numpy(dtype=float)
                      if std_col in ordered else np.zeros(len(ordered)))

        face_colors = [
            EDGE_BEST.get(a, {}).get("Best", "#4DBBD5")
            for a in ordered["Algorithm"]
        ]
        edge_colors = [
            ALGO_COLORS.get(a, {}).get("Best", "#333333")
            for a in ordered["Algorithm"]
        ]

        ax.bar(
            np.arange(len(ordered)),
            means,
            yerr=deviations,
            capsize=3,
            color=face_colors,
            edgecolor=edge_colors,
            linewidth=1.0,
            zorder=3,
        )

        ax.set_xticks(np.arange(len(ordered)))
        ax.set_xticklabels(
            [ALGO_DISPLAY.get(a, a) for a in ordered["Algorithm"]],
        )

        ax.set_ylabel(ylabel)
        style_standard_axes(ax)
        save_figure(fig, f"{prefix}compare_{metric.lower()}", output_dir)


# =========================================================
# Half violins + boxplots + individual runs
# =========================================================
METRIC_LABELS = {
    "HV": "Normalized HV",
    "IGD": "Normalized IGD",
    "Run_Wall_Seconds": "Wall-clock runtime (s)",
}


def draw_metric_violin_on_ax(ax, raw, metric, *, compact=False):
    
    selected = _valid_comparisons(raw)
    column = metric
    if metric == "Run_Wall_Seconds" and column not in selected:
        column = "Runtime_Seconds"
    names = [name for name in ALGO_ORDER if name in set(selected.get("Algorithm", []))]
    samples = {}
    for name in names:
        values = pd.to_numeric(selected.loc[selected["Algorithm"] == name, column],
                               errors="coerce").to_numpy(dtype=float) if column in selected else np.array([])
        samples[name] = values[np.isfinite(values)]

    style_standard_axes(ax)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    rng = np.random.default_rng(42)
    for position, (name, values) in enumerate(samples.items(), start=1):
        color = ALGO_COLORS[name]["Best"]
        if not len(values):
            ax.text(position, 0.5, "No valid data", rotation=90, ha="center", va="center",
                    transform=ax.get_xaxis_transform(), fontsize=LEGEND_FONTSIZE)
            continue
        
        if len(values) > 1 and np.ptp(values) > 0:
            violin = ax.violinplot([values], positions=[position], widths=0.72,
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

        ax.boxplot(
            [values], positions=[position], widths=0.13,
            patch_artist=True, showfliers=False, showmeans=True, whis=1.5,
            manage_ticks=False,
            boxprops=dict(facecolor="white", edgecolor=color, linewidth=1.0),
            medianprops=dict(color=SPINE_COLOR, linewidth=1.2),
            whiskerprops=dict(color=color, linewidth=0.9),
            capprops=dict(color=color, linewidth=0.9),
            meanprops=dict(marker="D", markerfacecolor=color, markeredgecolor="white",
                           markeredgewidth=0.35, markersize=3.4),
            zorder=3,
        )
        jitter = rng.uniform(0.12, 0.34, size=len(values))
        ax.scatter(position + jitter, values, s=5 if compact else 9,
                   color=color, edgecolors="none", alpha=0.55, zorder=4)

    ax.set_xticks(np.arange(1, len(names) + 1))
    ax.set_xticklabels([ALGO_DISPLAY[name] for name in names],
                       rotation=60 if compact else 0, ha="right" if compact else "center")
    ax.set_xlim(0.5, max(len(names), 1) + 0.5)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.margins(y=0.08)
    if metric == "Run_Wall_Seconds":
        ax.yaxis.set_major_formatter(FuncFormatter(_comma_int_formatter))
    return samples


def plot_metric_boxplots(raw, output_dir=None, prefix="", title=None):
    
    if raw.empty:
        return
    for metric in ("HV", "IGD"):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        draw_metric_violin_on_ax(ax, raw, metric)
        save_figure(fig, f"{prefix}boxplot_{metric.lower()}", output_dir)


def plot_run_wall_violin(raw, output_dir=None, prefix=""):
    
    if raw.empty:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    draw_metric_violin_on_ax(ax, raw, "Run_Wall_Seconds")
    save_figure(fig, f"{prefix}compare_run_wall_seconds", output_dir)


def read_combined_groups(source_dir):
    
    source_dir = Path(source_dir)
    raw = pd.read_csv(source_dir / CSV_FILES["multi_algorithm"]["performance_metrics_raw"])
    columns = {"Experiment_ID", "Protocol", "Budget_ID", "Algorithm", "Run", "Seed",
               "Comparison_Valid", "Status", "Stage", "Front_Kind",
               "Total_Cost", "Total_Defender_Loss"}
    points = pd.read_csv(source_dir / CSV_FILES["multi_algorithm"]["all_pareto_points"],
                         usecols=lambda name: name in columns)
    representative = pd.read_csv(source_dir / CSV_FILES["multi_algorithm"]["best_pareto_fronts"],
                                 usecols=lambda name: name in columns)
    groups = {}
    for budget in COMBINED_BUDGETS:
        subset = raw.loc[(raw["Protocol"] == "equal_fe") & (raw["Budget_ID"] == f"fe_{budget}")]
        if subset.empty or subset["Experiment_ID"].nunique() != 1:
            raise ValueError(f"FE={budget} must contain data from exactly one experiment")
        key = (subset["Experiment_ID"].iloc[0], "equal_fe", f"fe_{budget}")
        all_fronts, fixed_fronts, seed = _representative_fronts(
            _group_frame(points, key), _group_frame(representative, key), subset)
        valid = _valid_comparisons(subset)
        if set(valid["Algorithm"]) != set(ALGO_ORDER):
            raise ValueError(f"FE={budget} is missing valid runs for some algorithms")
        groups[budget] = dict(raw=valid, all_fronts=all_fronts,
                              representative=fixed_fronts, seed=seed)
    if len({group["seed"] for group in groups.values()}) != 1:
        raise ValueError("Both FE budgets must use the same representative seed")
    if len({group["raw"]["Experiment_ID"].iloc[0] for group in groups.values()}) != 1:
        raise ValueError("Both FE budgets must come from the same experiment")
    return groups


def pareto_limits(groups):
    
    coordinates = pd.concat([frame for group in groups.values()
                             for frame in (group["all_fronts"], group["representative"])],
                            ignore_index=True)[["Total_Cost", "Total_Defender_Loss"]].to_numpy(dtype=float)
    coordinates = coordinates[np.isfinite(coordinates).all(axis=1)]
    if not len(coordinates):
        return FIG8_XLIM, FIG8_YLIM
    lower, upper = coordinates.min(axis=0), coordinates.max(axis=0)
    padding = np.maximum((upper - lower) * 0.06, np.maximum(np.abs(upper), 1.0) * 0.02)
    return tuple((float(lower[index] - padding[index]), float(upper[index] + padding[index]))
                 for index in range(2))


def plot_combined_pareto(groups, output_dir, width=COMBINED_FIGURE_WIDTH):
    
    figure, axes = plt.subplots(4, 2, figsize=(width, COMBINED_PARETO_HEIGHT))
    limits = pareto_limits(groups)
    for row, algorithm in enumerate(ALGO_ORDER):
        for column, budget in enumerate(COMBINED_BUDGETS):
            axis, group = axes[row, column], groups[budget]
            _plot_one_algorithm_on_ax(axis, group["all_fronts"],
                                      group["representative"], algorithm, limits)
            axis.xaxis.labelpad = 2
            axis.yaxis.labelpad = 2
            axis.tick_params(axis="both", which="major", pad=2)
            legend = axis.get_legend()
            if legend is not None:
                legend.remove()
            axis.annotate(
                f"({chr(97 + row * 2 + column)}) {ALGO_DISPLAY[algorithm]}, FE={budget}",
                xy=(0.5, 0), xycoords="axes fraction",
                xytext=(0, -COMBINED_PARETO_CAPTION_OFFSET), textcoords="offset points",
                ha="center", va="top", fontsize=PANEL_CAPTION_FONTSIZE,
                annotation_clip=False,
            )
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
    figure.subplots_adjust(**COMBINED_PARETO_ADJUST)
    save_paper_figure(figure, output_dir, "equal_fe_fe_500_1000_pareto_fronts_4x2")


def plot_combined_metrics(groups, output_dir, width=COMBINED_FIGURE_WIDTH):
    
    figure, axes = plt.subplots(3, 2, figsize=(width, COMBINED_METRICS_HEIGHT), sharey="row")
    for row, metric in enumerate(COMBINED_METRICS):
        for column, budget in enumerate(COMBINED_BUDGETS):
            axis = axes[row, column]
            draw_metric_violin_on_ax(axis, groups[budget]["raw"], metric, compact=True)
            axis.yaxis.labelpad = 2
            axis.tick_params(axis="both", which="major", pad=2)
            axis.tick_params(axis="y", which="both", left=True, labelleft=True)
            panel = f"({chr(97 + row * 2 + column)})"
            caption = (f"{panel} {COMBINED_METRIC_TITLES[metric]}\n(FE budget = {budget})"
                       if metric == "Run_Wall_Seconds"
                       else f"{panel} FE={budget}: {COMBINED_METRIC_TITLES[metric]}")
            axis.annotate(
                caption,
                xy=(0.5, 0), xycoords="axes fraction",
                xytext=(0, -COMBINED_METRICS_CAPTION_OFFSET), textcoords="offset points",
                ha="center", va="top", fontsize=PANEL_CAPTION_FONTSIZE,
                annotation_clip=False,
            )
            axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    figure.subplots_adjust(**COMBINED_METRICS_ADJUST)
    save_paper_figure(figure, output_dir, "equal_fe_fe_500_1000_metrics_violin_3x2")


# =========================================================
# Loss convergence vs FE
# =========================================================
def plot_loss_convergence(history, output_dir=None, prefix=""):
    if history.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for name, algorithm_history in history.groupby("Algorithm"):
        color = ALGO_COLORS.get(name, {}).get("Best", None)

        for _, one_run in algorithm_history.groupby("Run"):
            ordered = one_run.sort_values("Total_FE")
            ax.plot(
                ordered["Total_FE"],
                ordered["Minimum_Loss"],
                alpha=0.35,
                linewidth=1,
                color=color,
            )

        
        runs = list(algorithm_history.groupby("Run"))
        maximum_fe = algorithm_history["Total_FE"].max()
        minimum_fe = algorithm_history["Total_FE"].min()
        grid = np.linspace(minimum_fe, maximum_fe, 60)
        curves = []
        for _, one_run in runs:
            ordered = one_run.sort_values("Total_FE")
            x = ordered["Total_FE"].to_numpy(dtype=float)
            y = ordered["Minimum_Loss"].to_numpy(dtype=float)
            positions = np.searchsorted(x, grid, side="right") - 1
            curve = np.full(len(grid), np.nan)
            valid = positions >= 0
            curve[valid] = y[positions[valid]]
            curves.append(curve)

        if curves:
            with np.errstate(invalid="ignore"):
                mean = np.nanmean(np.asarray(curves), axis=0)
            ax.plot(
                grid, mean,
                linewidth=2,
                color=color,
                label=f"{ALGO_DISPLAY.get(name, name)} mean",
            )

    ax.set_xlabel("Function evaluations (FE)")
    ax.set_ylabel("Minimum feasible defender loss")
    style_standard_axes(ax)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    save_figure(fig, f"{prefix}loss_convergence_vs_fe", output_dir)


def plot_generation_runtime(history, output_dir=None, prefix=""):
    
    if history.empty or "Generation_Seconds" not in history:
        return
    names = [name for name in ALGO_ORDER if name in set(history["Algorithm"])]
    values = [pd.to_numeric(history.loc[history["Algorithm"] == name,
                                       "Generation_Seconds"], errors="coerce")
              .dropna().to_numpy(dtype=float) for name in names]
    values = [value[np.isfinite(value)] for value in values]
    if not names or not any(len(value) for value in values):
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    boxes = ax.boxplot(values, positions=np.arange(1, len(names) + 1),
                       widths=0.55, patch_artist=True, showfliers=True, whis=1.5)
    for index, algorithm in enumerate(names):
        boxes["boxes"][index].set(facecolor=EDGE_BEST[algorithm]["Best"],
                                   edgecolor=ALGO_COLORS[algorithm]["Best"], linewidth=1.0)
        boxes["medians"][index].set(color=ALGO_COLORS[algorithm]["Best"], linewidth=1.2)
    ax.set_xticks(np.arange(1, len(names) + 1))
    ax.set_xticklabels([ALGO_DISPLAY[name] for name in names])
    ax.set_ylabel("Measured generation runtime (seconds)")
    style_standard_axes(ax)
    save_figure(fig, f"{prefix}compare_generation_seconds", output_dir)


def _group_frame(frame, key):
    selected = frame
    for column, value in zip(("Experiment_ID", "Protocol", "Budget_ID"), key):
        if column in selected:
            selected = selected.loc[selected[column].astype(str) == str(value)]
    return selected.copy()


def _valid_comparisons(frame):
    selected = frame
    if "Comparison_Valid" in selected:
        valid = selected["Comparison_Valid"].fillna(False).astype(str).str.lower().isin(("true", "1"))
        selected = selected.loc[valid]
    elif "Status" in selected:
        selected = selected.loc[selected["Status"] == "ok"]
    return selected.copy()


def _final_strict(frame):
    selected = _valid_comparisons(frame)
    if "Front_Kind" in selected:
        selected = selected.loc[selected["Front_Kind"] == "strict"]
    if "Stage" in selected:
        selected = selected.loc[selected["Stage"] == "final"]
    return selected.copy()


def _representative_fronts(all_points, best_fronts, raw):
    points = _final_strict(all_points)
    selected = _final_strict(best_fronts)
    seeds = selected["Seed"].dropna().unique() if "Seed" in selected else []
    if len(seeds) > 1:
        raise ValueError("Representative fronts must use the same prespecified Seed for all algorithms")
    declared = raw["Representative_Seed"].dropna().unique() if "Representative_Seed" in raw else []
    if len(declared) > 1:
        raise ValueError("Representative fronts must declare the same Representative_Seed")
    if len(declared):
        seed = declared[0]
        if len(seeds) and seeds[0] != seed:
            raise ValueError("Seed and Representative_Seed disagree in the representative fronts")
    elif len(seeds):
        seed = seeds[0]
    elif "Seed" in raw and raw["Seed"].notna().any():
        seed = raw["Seed"].dropna().min()
    else:
        seed = "legacy"
    if "Seed" in selected:
        selected = selected.loc[selected["Seed"] == seed].copy()
    elif "Seed" in points:
        selected = points.loc[points["Seed"] == seed].copy()
    return points, selected, seed


def _plot_directory(input_excel, output_dir, experiment_id):
    if output_dir is not None:
        return Path(output_dir)
    base = get_feature_output_dir("multi_algorithm", input_excel)
    if experiment_id is not None:
        identifier = str(experiment_id)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", identifier) or identifier in (".", ".."):
            raise ValueError("experiment_id must be a single safe directory name")
        return base / "experiments" / identifier
    pointer = base / "latest_experiment.csv"
    if pointer.is_file():
        return Path(pd.read_csv(pointer).iloc[0]["Output_Directory"])
    return base


def plot(input_excel=None, output_dir=None, experiment_id=None):
    
    started = perf_counter()
    output_dir = _plot_directory(input_excel, output_dir, experiment_id)
    paths = CSV_FILES["multi_algorithm"]
    tables = {name: pd.read_csv(output_dir / paths[name]) for name in (
        "performance_summary", "performance_metrics_raw", "all_pareto_points",
        "best_pareto_fronts", "optimization_history",
    )}
    raw = tables["performance_metrics_raw"]
    group_columns = ("Experiment_ID", "Protocol", "Budget_ID")
    if set(group_columns).issubset(raw.columns):
        keys = list(raw[list(group_columns)].drop_duplicates().itertuples(index=False, name=None))
    else:
        keys = [("legacy", "equal_fe", "legacy")]
    with plt.rc_context():
        set_plot_style()
        combined_groups = {}
        for key in keys:
            grouped = {name: _group_frame(frame, key) for name, frame in tables.items()}
            group_raw = grouped["performance_metrics_raw"]
            all_points, representative, seed = _representative_fronts(
                grouped["all_pareto_points"], grouped["best_pareto_fronts"], group_raw)
            token = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{key[1]}_{key[2]}_seed{seed}")
            prefix = "" if key[0] == "legacy" else token + "_"
            plot_pareto_fronts_4in1(all_points, representative, output_dir,
                                   name=f"{prefix}best_pareto_fronts")
            plot_summary_bars(grouped["performance_summary"], output_dir, prefix=prefix)
            valid_raw = _valid_comparisons(group_raw)
            plot_metric_boxplots(valid_raw, output_dir, prefix=prefix)
            plot_run_wall_violin(valid_raw, output_dir, prefix=prefix)
            history = _valid_comparisons(grouped["optimization_history"])
            if "Front_Kind" in history:
                history = history.loc[history["Front_Kind"] == "strict"]
            plot_generation_runtime(history, output_dir, prefix=prefix)
            if "Minimum_Loss" in history:
                plot_loss_convergence(history, output_dir, prefix=prefix)
            for budget in COMBINED_BUDGETS:
                if key[1:3] == ("equal_fe", f"fe_{budget}"):
                    combined_groups[budget] = dict(raw=valid_raw, all_fronts=all_points,
                                                   representative=representative, seed=seed)
        if (set(combined_groups) == set(COMBINED_BUDGETS)
                and raw["Experiment_ID"].nunique() == 1
                and len({group["seed"] for group in combined_groups.values()}) == 1
                and all(set(group["raw"]["Algorithm"]) == set(ALGO_ORDER)
                        for group in combined_groups.values())):
            plot_combined_pareto(combined_groups, output_dir)
            plot_combined_metrics(combined_groups, output_dir)
    elapsed = perf_counter() - started
    pd.DataFrame([{"Experiment_ID": key[0], "Plot_Seconds": elapsed,
                   "Output_Directory": str(output_dir)} for key in keys[:1]]).to_csv(
        output_dir / "plot_timing.csv", index=False)
    print(f"✅ Multi-algorithm plots saved to {output_dir}")
    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-excel")
    parser.add_argument("--output-dir")
    parser.add_argument("--experiment-id")
    arguments = parser.parse_args()
    plot(input_excel=arguments.input_excel, output_dir=arguments.output_dir,
         experiment_id=arguments.experiment_id)
