

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import t

from common.config import CSV_FILES, get_feature_output_dir
from common.plotting import (
    TICK_LABELSIZE, normalize_figure_typography,
    save_figure, set_plot_style, style_axes,
)

from features.macomp.config import CONFIDENCE_LEVEL
from features.macomp.utils import mean_confidence_interval



GROUP_TITLES = {
    "A": "Local FE cap",
    "B": "Elite quota",
    "C": "Neighborhood",
    "D": "Search depth",
    "E": "Search trigger",
    "F": "Acceptance rule",
    "G": "Combined configurations",
    "S": "Target selection",
    "Legacy": "Existing configurations",
}


def _ranking(runs: pd.DataFrame, fe_budget: int) -> pd.DataFrame:
    
    subset = runs.loc[runs["Budget_FE"] == fe_budget]
    ranking = subset.groupby("Experiment", as_index=False).agg(
        HV_Mean=("HV", "mean"),
        IGD_Mean=("IGD", "mean"),
    )
    ranking["HV_Rank"] = ranking["HV_Mean"].rank(
        ascending=False,
        method="min",
    )
    ranking["IGD_Rank"] = ranking["IGD_Mean"].rank(
        ascending=True,
        method="min",
    )
    ranking["Baseline_Priority"] = (
        ranking["Experiment"] != "NSGA-II"
    )
    ranking = ranking.sort_values(
        ["HV_Mean", "IGD_Mean", "Baseline_Priority", "Experiment"],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranking.insert(0, "Rank", np.arange(1, len(ranking) + 1))
    return ranking


def _step_values(one_run: pd.DataFrame, grid, metric: str):
    
    ordered = one_run.sort_values("Total_FE").drop_duplicates(
        "Total_FE",
        keep="last",
    )
    x = ordered["Total_FE"].to_numpy(dtype=float)
    y = ordered[metric].to_numpy(dtype=float)

    positions = np.searchsorted(x, grid, side="right") - 1
    values = np.full(len(grid), np.nan, dtype=float)
    valid = (positions >= 0) & (grid <= x[-1])
    values[valid] = y[positions[valid]]
    return values


def _curve_confidence(curves):
    
    values = np.asarray(curves, dtype=float)
    count = np.isfinite(values).sum(axis=0)

    mean = np.full(values.shape[1], np.nan)
    valid_mean = count > 0
    mean[valid_mean] = (
        np.nansum(values[:, valid_mean], axis=0)
        / count[valid_mean]
    )

    lower = np.full(values.shape[1], np.nan)
    upper = np.full(values.shape[1], np.nan)
    valid_interval = count > 1

    if valid_interval.any():
        deviations = values[:, valid_interval] - mean[valid_interval]
        variance = (
            np.nansum(deviations ** 2, axis=0)
            / (count[valid_interval] - 1)
        )
        critical = t.ppf(
            (1.0 + CONFIDENCE_LEVEL) / 2.0,
            df=count[valid_interval] - 1,
        )
        half_width = (
            critical * np.sqrt(variance)
            / np.sqrt(count[valid_interval])
        )
        lower[valid_interval] = mean[valid_interval] - half_width
        upper[valid_interval] = mean[valid_interval] + half_width

    return mean, lower, upper


def _plot_curves(
    history: pd.DataFrame,
    experiments: list[str],
    group_name: str,
    metric: str,
    fe_budget: int,
    output_dir: Path,
) -> None:
    
    subset = history.loc[
        (history["Budget_FE"] == fe_budget)
        & history["Experiment"].isin(experiments)
    ]
    if subset.empty:
        return

    checkpoints = subset["Total_FE"].to_numpy(dtype=int)
    grid = np.unique(np.concatenate([
        np.linspace(
            checkpoints.min(),
            fe_budget,
            200,
        ).astype(int),
        checkpoints,
    ]))

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    palette = plt.get_cmap("tab10")

    for index, experiment in enumerate(experiments):
        experiment_history = subset.loc[
            subset["Experiment"] == experiment
        ]
        curves = [
            _step_values(one_run, grid, metric)
            for _, one_run in experiment_history.groupby("Run")
        ]
        if not curves:
            continue

        mean, lower, upper = _curve_confidence(curves)
        color = (
            "#222222"
            if experiment == "NSGA-II"
            else palette(index % 10)
        )

        ax.step(
            grid,
            mean,
            where="post",
            color=color,
            linewidth=1.7,
            label=experiment,
        )
        ax.fill_between(
            grid,
            lower,
            upper,
            step="post",
            color=color,
            alpha=0.12,
        )

    ax.set_xlabel("Total function evaluations (FE)")
    ax.set_ylabel(f"{metric} (common empirical normalization)")
    ax.set_title(
        f"{GROUP_TITLES.get(group_name, group_name)}; "
        f"budget={fe_budget} FE; mean and 95% CI"
    )
    style_axes(ax)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=8,
    )
    normalize_figure_typography(fig)
    fig.tight_layout()
    save_figure(
        fig,
        output_dir,
        f"{metric.lower()}_vs_total_fe_{group_name}_{fe_budget}",
    )


def _bar_chart(
    runs: pd.DataFrame,
    order: list[str],
    column: str,
    xlabel: str,
    basename: str,
    fe_budget: int,
    output_dir: Path,
) -> None:
    
    subset = runs.loc[runs["Budget_FE"] == fe_budget]
    means = []
    lower_errors = []
    upper_errors = []

    for experiment in order:
        values = subset.loc[
            subset["Experiment"] == experiment,
            column,
        ].to_numpy(dtype=float)
        mean, _, lower, upper = mean_confidence_interval(values)

        means.append(mean)
        lower_errors.append(
            max(0.0, mean - lower) if np.isfinite(lower) else 0.0
        )
        upper_errors.append(
            max(0.0, upper - mean) if np.isfinite(upper) else 0.0
        )

    means = np.asarray(means, dtype=float)
    y = np.arange(len(order))
    fig, ax = plt.subplots(
        figsize=(9.2, max(4.5, 0.30 * len(order) + 1.2))
    )

    colors = [
        "#555555" if name == "NSGA-II" else "#74add1"
        for name in order
    ]
    ax.barh(
        y,
        np.nan_to_num(means, nan=0.0),
        xerr=np.asarray([lower_errors, upper_errors]),
        color=colors,
        edgecolor="#333333",
        linewidth=0.5,
        capsize=2,
        zorder=3,
    )

    for position, mean in enumerate(means):
        if not np.isfinite(mean):
            ax.text(0.0, position, " N/A", va="center", fontsize=TICK_LABELSIZE)

    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=TICK_LABELSIZE)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(f"{fe_budget} FE; mean and 95% CI")
    style_axes(ax)
    ax.grid(False, axis="y")
    ax.grid(
        True,
        axis="x",
        linestyle="--",
        linewidth=0.4,
        color="#d9d9d9",
    )
    normalize_figure_typography(fig)
    fig.tight_layout()
    save_figure(fig, output_dir, f"{basename}_{fe_budget}")


def _ranking_heatmap(
    ranking: pd.DataFrame,
    fe_budget: int,
    output_dir: Path,
) -> None:
    
    color_values = np.column_stack([
        ranking["Rank"],
        ranking["HV_Rank"],
        ranking["IGD_Rank"],
        ranking["HV_Rank"],
        ranking["IGD_Rank"],
    ]).astype(float)

    labels = ["Overall rank", "HV mean", "IGD mean", "HV rank", "IGD rank"]
    fig, ax = plt.subplots(
        figsize=(8.2, max(4.5, 0.30 * len(ranking) + 1.2))
    )
    image = ax.imshow(
        color_values,
        aspect="auto",
        cmap="RdYlGn_r",
        vmin=1,
        vmax=max(2, len(ranking)),
    )

    for row_index, row in ranking.iterrows():
        annotations = [
            str(int(row["Rank"])),
            f"{row['HV_Mean']:.5f}",
            f"{row['IGD_Mean']:.5f}",
            str(int(row["HV_Rank"])),
            str(int(row["IGD_Rank"])),
        ]
        for column_index, text in enumerate(annotations):
            ax.text(
                column_index,
                row_index,
                text,
                ha="center",
                va="center",
                fontsize=TICK_LABELSIZE,
                color="#111111",
            )

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(ranking)))
    ax.set_yticklabels(ranking["Experiment"], fontsize=TICK_LABELSIZE)
    ax.set_title(
        f"Ranking at {fe_budget} FE: HV descending, IGD ascending"
    )
    style_axes(ax)
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="Rank; lower is better", fraction=0.04)
    normalize_figure_typography(fig)
    fig.tight_layout()
    save_figure(fig, output_dir, f"ranking_heatmap_{fe_budget}")


def plot(
    *,
    input_dir=None,
    fe_budget: int | None = None,
    input_excel=None,
) -> None:
    
    output_dir = get_feature_output_dir("macomp", input_excel) if input_dir is None else Path(input_dir)
    files = CSV_FILES["macomp"]

    runs = pd.read_csv(output_dir / files["runs"])
    history = pd.read_csv(output_dir / files["history"])

    if runs.empty or history.empty:
        raise ValueError("macomp CSV data are empty; complete solving and export first")

    required_runs = {
        "Experiment", "Run", "Budget_FE", "Total_FE", "Local_FE",
        "Generations", "Pareto_Size", "Local_Accepted", "HV", "IGD",
    }
    required_history = {
        "Experiment", "Run", "Budget_FE", "Total_FE", "HV", "IGD",
    }
    if not required_runs.issubset(runs.columns):
        raise ValueError("runs.csv is missing required columns")
    if not required_history.issubset(history.columns):
        raise ValueError("history.csv is missing required columns")

    if "Variant_Group" not in runs:
        runs["Variant_Group"] = "Legacy"

    runs["Local_FE_Percent"] = (
        100.0 * runs["Local_FE"] / runs["Total_FE"]
    )
    runs["Accepted_Per_1000_Local_FE"] = np.where(
        runs["Local_FE"] > 0,
        1000.0 * runs["Local_Accepted"] / runs["Local_FE"],
        np.nan,
    )

    available_budgets = set(runs["Budget_FE"].astype(int))
    if fe_budget is None:
        fe_budget = max(available_budgets)
    if fe_budget not in available_budgets:
        raise ValueError(
            f"No CSV data for FE={fe_budget}; "
            f"available values={sorted(available_budgets)}"
        )

    set_plot_style(legend_fontsize=9)
    ranking = _ranking(runs, fe_budget)
    order = ranking["Experiment"].tolist()
    all_names = set(runs["Experiment"])

    for group_name, group in runs.groupby("Variant_Group", sort=True):
        names = group["Experiment"].drop_duplicates().tolist()

        anchors = ["NSGA-II"]
        if group_name == "A":
            anchors.append("Elite-ARS")
        elif "A2-Cap20" in all_names:
            anchors.append("A2-Cap20")

        names = list(dict.fromkeys(
            [name for name in anchors if name in all_names] + names
        ))

        for metric in ("HV", "IGD"):
            _plot_curves(
                history,
                names,
                str(group_name),
                metric,
                fe_budget,
                output_dir,
            )

    for column, xlabel, basename in (
        (
            "Local_FE_Percent",
            "Local FE / Total FE (%)",
            "local_fe_ratio",
        ),
        ("Generations", "Completed generations", "generations"),
        ("Pareto_Size", "Unique objective Pareto points", "pareto_size"),
        (
            "Accepted_Per_1000_Local_FE",
            "Accepted local candidates per 1000 Local FE",
            "local_acceptance_efficiency",
        ),
    ):
        _bar_chart(
            runs,
            order,
            column,
            xlabel,
            basename,
            fe_budget,
            output_dir,
        )

    _ranking_heatmap(ranking, fe_budget, output_dir)
    print(f"✅ Ablation plots saved to {output_dir}")


def main() -> None:
    
    parser = argparse.ArgumentParser(description="Plot MA ablation CSV data")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--input-excel", type=Path, default=None)
    parser.add_argument("--fe-budget", type=int, default=None)
    args = parser.parse_args()
    plot(input_dir=args.input_dir, fe_budget=args.fe_budget, input_excel=args.input_excel)


if __name__ == "__main__":
    main()
