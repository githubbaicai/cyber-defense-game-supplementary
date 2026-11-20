"""Read E3 CSV tables and render diagnostic figures."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.plotting import (
    AXES_LINEWIDTH, GRID_COLOR, GRID_LINEWIDTH, SPINE_COLOR,
    PANEL_CAPTION_FONTSIZE, normalize_figure_typography, set_plot_style,
)
from .export_csv import COMMON_COLUMNS, _output_dir



@contextmanager
def _plot_style():
    with plt.rc_context():
        set_plot_style(kind="multi", legend_fontsize=8)
        yield


def _read(directory: Path, name: str) -> pd.DataFrame:
    path = directory / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"E3 CSV not found: {path}")
    return pd.read_csv(path)


def _save(fig, directory: Path, basename: str, *, pad_inches=0.1):
    directory.mkdir(parents=True, exist_ok=True)
    normalize_figure_typography(fig)
    for extension in ("png", "svg"):
        fig.savefig(directory / f"{basename}.{extension}", format=extension,
                    bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)


def _strategy_trace(trace: pd.DataFrame, directory: Path, basename="strategy_residual_trace"):
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    if not trace.empty and {"Iteration", "Probability"}.issubset(trace.columns):
        identity = trace[["Run_ID", "Node_Path"]].iloc[0]
        selected = trace.loc[trace["Run_ID"].eq(identity["Run_ID"]) & trace["Node_Path"].eq(identity["Node_Path"])].copy()
        for (distribution, action), group in selected.groupby(["Distribution", "Action_ID"], dropna=False):
            group = group.sort_values("Iteration")
            axes[0].plot(group["Iteration"], group["Probability"], label=f"{distribution}:{action}")
        axes[0].set_title(f"Run {identity['Run_ID']} | {identity['Node_Path']}")
    axes[0].set_ylabel("Strategy / event probability")
    axes[0].legend(frameon=True) if axes[0].lines else None
    if not trace.empty and {"Iteration", "Map_Residual"}.issubset(trace.columns):
        residual = selected.groupby("Iteration", as_index=False)["Map_Residual"].max()
        residual["Map_Residual"] = pd.to_numeric(residual["Map_Residual"], errors="coerce")
        axes[1].plot(residual["Iteration"], residual["Map_Residual"], color="#C44E52")
        if (residual["Map_Residual"] > 0).any():
            axes[1].set_yscale("symlog", linthresh=1e-12)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Map residual")
    for axis in axes:
        axis.grid(True, axis="y", color="#d9d9d9", linewidth=0.4)
    fig.tight_layout()
    _save(fig, directory, basename)


def _initialization_eta(sensitivity: pd.DataFrame, diagnostics: pd.DataFrame, directory: Path, basename="initialization_eta_comparison"):
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.6))
    for row_index, experiment_type in enumerate(("fixed_local_snapshot", "full_graph_replay")):
        frame = sensitivity.loc[sensitivity["Experiment_Type"].eq(experiment_type)].copy() if not sensitivity.empty else sensitivity.copy()
        if not frame.empty:
            frame["Learning_Rate"] = pd.to_numeric(frame["Learning_Rate"], errors="coerce")
            diagnostic = diagnostics.loc[diagnostics["Run_ID"].isin(frame["Run_ID"])].copy() if not diagnostics.empty else diagnostics
            if not diagnostic.empty:
                residual = diagnostic.groupby("Run_ID", as_index=False)["R_map"].max()
                merged = frame.drop_duplicates("Run_ID").merge(residual, on="Run_ID", how="left", validate="one_to_one")
                for label, group in merged.groupby("Initialization", dropna=False):
                    axes[row_index, 0].scatter(group["Learning_Rate"], group["R_map"], label=str(label))
                axes[row_index, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            observations = frame if experiment_type == "fixed_local_snapshot" else frame.drop_duplicates("Run_ID")
            categories = observations["Endpoint_Category"].fillna("unknown").astype(str)
            categories = categories.str.replace("observed_local_endpoint_group_", "g", regex=False).str.replace("observed_endpoint_group_", "g", regex=False)
            if experiment_type == "fixed_local_snapshot":
                node_labels = {node: f"s{number}" for number, node in enumerate(observations["Node_Path"].unique(), start=1)}
                categories = observations["Node_Path"].map(node_labels) + ":" + categories
            counts = categories.value_counts()
            axes[row_index, 1].bar(counts.index.astype(str), counts.values)
            axes[row_index, 1].tick_params(axis="x", rotation=35)
        axes[row_index, 0].set_title(experiment_type)
        axes[row_index, 0].set_xlabel("Learning rate")
        axes[row_index, 0].set_ylabel("Maximum map residual")
        axes[row_index, 1].set_xlabel("State index : endpoint group" if row_index == 0 else "Observed endpoint group")
        axes[row_index, 1].set_ylabel("State/checkpoint observations" if row_index == 0 else "Runs")
        if axes[row_index, 0].collections:
            axes[row_index, 0].legend(frameon=True)
    for axis in axes.flat:
        axis.grid(True, axis="y", color="#d9d9d9", linewidth=0.4)
    fig.tight_layout()
    _save(fig, directory, basename)


def _path_frequency_loss(path: pd.DataFrame, sensitivity: pd.DataFrame, metadata: pd.DataFrame, directory: Path, basename="path_frequency_loss"):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    full_graph = sensitivity.loc[sensitivity["Experiment_Type"].eq("full_graph_replay")].copy() if not sensitivity.empty else sensitivity
    success = metadata.loc[metadata["Status"].astype(str).str.lower().isin(("ok", "success")), "Run_ID"]
    if not full_graph.empty:
        full_graph = full_graph.loc[full_graph["Run_ID"].isin(success)].copy()
    run_ids = set(full_graph["Run_ID"].astype(str)) if not full_graph.empty else set()
    complete = path.loc[path["Run_ID"].astype(str).isin(run_ids) & path["Path_Type"].eq("complete_path")].copy() if not path.empty else path
    if not complete.empty:
        complete["Probability"] = pd.to_numeric(complete["Probability"], errors="coerce")
        coverage = complete.groupby("Run_ID")["Probability"].apply(lambda values: values.notna().all())
        covered_ids = set(coverage.loc[coverage].index)
        complete = complete.loc[complete["Run_ID"].isin(covered_ids)]
        main = complete["Is_Main_Path"].astype(str).str.lower().isin(("true", "1"))
        frequency = complete.loc[main].groupby("Path")["Run_ID"].nunique().sort_values(ascending=False).head(15)
        axes[0].barh(frequency.index.astype(str), frequency.values / max(1, len(covered_ids)))
        axes[0].invert_yaxis()
        axes[0].set_title(f"Complete path coverage: {len(covered_ids)}/{len(run_ids)} runs")
    if not full_graph.empty and "Total_Defender_Loss" in full_graph:
        loss = pd.to_numeric(full_graph.drop_duplicates("Run_ID")["Total_Defender_Loss"], errors="coerce").dropna()
        if len(loss):
            axes[1].hist(loss, bins=min(10, max(1, len(loss))), color="#4C72B0")
    axes[0].set_xlabel("Frequency as a most likely path (ties retained)")
    axes[0].set_ylabel("Path")
    axes[1].set_xlabel("Defender loss")
    axes[1].set_ylabel("Runs")
    for axis in axes:
        axis.grid(True, axis="y", color="#d9d9d9", linewidth=0.4)
    fig.tight_layout()
    _save(fig, directory, basename)


def _publication_initialization(runs, trace, directory, basename="equilibrium_initialization_summary"):
    """Paper summary: actual fixed-local rounds, tracked path and signed loss.

    Display signed relative loss changes over the requested [-1.0, 0.0]% range.
    """
    fig, axes = plt.subplots(1, 3, figsize=(6.50, 2.40))
    fig.subplots_adjust(left=.075, right=.99, top=.975, bottom=.24, wspace=.50)
    colors = ("#3C5488", "#E64B35", "#00A087", "#7E6148")
    portfolio_labels = {"P20": "PF20", "P24": "PF24"}
    tick_labels = None
    initialization_signature = None
    for index, (portfolio, cohort) in enumerate(runs.groupby("Portfolio_ID", sort=False)):
        cohort = cohort.sort_values("Initialization_Order")
        baseline = cohort.loc[cohort["Is_Baseline"].astype(str).str.lower().isin(("true", "1"))].iloc[0]
        selected = trace.loc[trace["Portfolio_ID"].eq(portfolio)]
        maximum = selected.groupby("Iteration")["Maximum_Attack_Strategy_Difference"].max().sort_index()
        color = colors[index % len(colors)]
        axes[0].plot(maximum.index, maximum.values, color=color,
                     label=portfolio_labels.get(str(portfolio), str(portfolio)), linewidth=1.2)
        labels = cohort["Initialization_Label"].tolist()
        signature = list(zip(labels, cohort["Initialization"], cohort["Seed"].fillna("none")))
        if initialization_signature is not None and signature != initialization_signature:
            raise ValueError("publication summary: portfolio initialization orders differ")
        tick_labels = labels
        initialization_signature = signature
        positions = np.arange(len(cohort))
        axes[1].plot(positions, cohort["Path_Weight"].to_numpy() * 100,
                     "o-", color=color, linewidth=1.2, markersize=3)
        if baseline["Total_Defender_Loss"] == 0:
            raise ValueError("publication summary: relative loss change is undefined for zero baseline loss")
        relative = (cohort["Total_Defender_Loss"].to_numpy() / baseline["Total_Defender_Loss"] - 1) * 100
        axes[2].plot(positions, relative, "o-", color=color, linewidth=1.2, markersize=3, clip_on=True)
    axes[0].set(xlabel="Generation", ylabel="Maximum strategy difference", ylim=(0, 1))
    axes[0].set_yticks(np.linspace(0, 1, 6))
    axes[0].legend(frameon=True, edgecolor=SPINE_COLOR, loc="upper right")
    axes[1].set(xlabel="Attack initialization", ylabel="Baseline-path weight (%)")
    axes[2].set(xlabel="Attack initialization", ylabel="Relative loss change (%)", ylim=(-1.0, 0.0))
    axes[2].set_yticks(np.linspace(-1.0, 0.0, 6))
    for axis in axes[1:]:
        axis.set_xticks(range(len(tick_labels)), tick_labels)
    for axis, caption in zip(axes, ("(a) Local convergence", "(b) Path weight", "(c) Loss change")):
        axis.text(.5, -.26, caption, transform=axis.transAxes, ha="center", va="top",
                  fontsize=PANEL_CAPTION_FONTSIZE)
        axis.grid(True, color=GRID_COLOR, linewidth=GRID_LINEWIDTH)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color(SPINE_COLOR)
            spine.set_linewidth(AXES_LINEWIDTH)
    _save(fig, directory, basename, pad_inches=.02)


def plot(input_excel=None, output_dir=None):
    """Read only E3 CSV files and save all figures as PNG and SVG."""
    directory = _output_dir(input_excel, output_dir)
    trace = _read(directory, "strategy_trace")
    sensitivity = _read(directory, "initialization_sensitivity")
    paths = _read(directory, "path_sensitivity")
    metadata = _read(directory, "run_metadata")
    diagnostics = _read(directory, "equilibrium_diagnostics")
    _validate_identity(metadata, (trace, sensitivity, paths, diagnostics))
    with _plot_style():
        portfolios = metadata[["Case_ID", "Experiment_ID", "Portfolio_ID"]].drop_duplicates()
        for index, identity in enumerate(portfolios.itertuples(index=False, name=None)):
            selection = ["Case_ID", "Experiment_ID", "Portfolio_ID"]
            selected_frames = []
            for frame in (trace, sensitivity, paths, diagnostics, metadata):
                mask = np.ones(len(frame), dtype=bool)
                for column, value in zip(selection, identity):
                    mask &= frame[column].astype(str).eq(str(value)).to_numpy()
                selected_frames.append(frame.loc[mask].copy())
            selected_trace, selected_sensitivity, selected_paths, selected_diagnostics, selected_metadata = selected_frames
            suffix = "" if index == 0 else "_" + hashlib.sha256("|".join(map(str, identity)).encode()).hexdigest()[:10]
            _strategy_trace(selected_trace, directory, "strategy_residual_trace" + suffix)
            _initialization_eta(selected_sensitivity, selected_diagnostics, directory, "initialization_eta_comparison" + suffix)
            _path_frequency_loss(selected_paths, selected_sensitivity, selected_metadata, directory, "path_frequency_loss" + suffix)
        # Publication evidence has its own experiment and input hash, so the
        # current paper cohort can coexist with earlier diagnostic exports.
        from features.equilibrium_validation.publication import read_tables
        publication = read_tables(directory)
        if publication is not None and not publication[0].empty:
            runs, publication_trace = publication
            for index, (identity, cohort) in enumerate(runs.groupby(["Case_ID", "Experiment_ID", "Input_SHA256"], sort=False)):
                selected = publication_trace
                selected = selected.merge(cohort[list(COMMON_COLUMNS)], on=list(COMMON_COLUMNS), how="inner", validate="many_to_one")
                suffix = "" if index == 0 else "_" + hashlib.sha256("|".join(map(str, identity)).encode()).hexdigest()[:10]
                _publication_initialization(cohort, selected, directory, "equilibrium_initialization_summary" + suffix)
    print(f"equilibrium_validation plots saved to {directory}")


def _validate_identity(metadata: pd.DataFrame, frames):
    if metadata.empty or not set(COMMON_COLUMNS).issubset(metadata.columns):
        raise ValueError("run_metadata: missing run identity")
    if metadata["Run_ID"].duplicated().any():
        raise ValueError("run_metadata: duplicate Run_ID")
    identity = metadata.copy()
    for column in COMMON_COLUMNS:
        identity[column] = identity[column].astype(str)
    identity = identity.set_index("Run_ID")
    for frame in frames:
        if not set(COMMON_COLUMNS).issubset(frame.columns):
            raise ValueError("E3 CSV: missing identity columns")
        if frame.empty:
            continue
        run_ids = frame["Run_ID"].astype(str)
        if not run_ids.isin(identity.index).all():
            raise ValueError("E3 CSV: unknown Run_ID")
        for column in COMMON_COLUMNS[:-1]:
            if not frame[column].astype(str).eq(run_ids.map(identity[column])).all():
                raise ValueError(f"E3 CSV: {column} disagrees with run_metadata identity")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot exported equilibrium-validation CSV")
    parser.add_argument("--input-excel", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    plot(input_excel=args.input_excel, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
