"""Plot exported model-validation evidence without solving or loading caches."""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.config import CSV_FILES, get_feature_output_dir
from common.case_config import get_input_excel
from common.plotting import (
    TICK_LABELSIZE, normalize_figure_typography,
    save_figure, set_plot_style, style_axes,
)
from features.model_validation.export_csv import TABLE_COLUMNS, validate_table_columns


def _load_tables(directory):
    tables = {}
    for name in TABLE_COLUMNS:
        path = directory / CSV_FILES["model_validation"][name]
        if not path.is_file():
            raise FileNotFoundError(f"Missing model-validation CSV: {path}")
        frame = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
        validate_table_columns(name, frame)
        tables[name] = frame
    summary = tables["validation_summary"]
    statuses = summary["Status"].astype(str).str.strip().str.lower()
    if summary.empty or not statuses.eq("pass").all():
        raise ValueError("Model-validation CSV checks must all pass before plotting")
    metadata = tables["run_metadata"]
    passed = metadata.loc[metadata["Key"].astype(str).str.lower().eq("passed"), "Value"]
    if len(passed) != 1 or str(passed.iloc[0]).strip().lower() != "true":
        raise ValueError("run_metadata must contain exactly one Passed=true record")
    return tables


def _numeric(frame, columns, name):
    values = frame.loc[:, columns].apply(pd.to_numeric, errors="raise").astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"{name} contains non-finite plotting values")
    return values


def _plot_loss_comparison(summary, directory, case_id):
    checks = ("production_vs_terminal", "prefix_vs_terminal")
    comparison = summary.loc[
        summary["Scenario_ID"].eq("workbook_case") & summary["Check"].isin(checks)
    ].copy()
    if len(comparison) != len(checks) or comparison["Check"].duplicated().any():
        raise ValueError("workbook_case requires one production and one prefix reconciliation record")
    comparison = comparison.set_index("Check")
    values = _numeric(comparison, ["Actual", "Expected"], "workbook_case reconciliation")
    totals = np.array([
        values.loc["production_vs_terminal", "Actual"],
        values.loc["prefix_vs_terminal", "Actual"],
        values.loc["production_vs_terminal", "Expected"],
    ])
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    bars = axis.bar(
        ["Production\naggregation", "Unique-prefix\ncontributions", "Stopping\ntrajectories"],
        totals, width=0.6, color=("#E69F00", "#56B4E9", "#009E73"),
        edgecolor="#333333", linewidth=0.5, zorder=3,
    )
    axis.bar_label(bars, labels=[f"{value:.6g}" for value in totals], padding=3)
    axis.set_ylabel("Expected loss")
    axis.set_title(f"{case_id}: expected-loss reconciliation")
    axis.set_ylim(min(0.0, float(totals.min())) * 1.18, max(float(totals.max()), 1.0) * 1.18)
    figure.text(
        0.5, 0.01, f"Workbook case only. Maximum difference: {np.ptp(totals):.3g}",
        ha="center", fontsize=TICK_LABELSIZE,
    )
    style_axes(axis)
    normalize_figure_typography(figure)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(figure, directory, "loss_reconciliation")


def _plot_prefix_contributions(prefixes, directory, case_id):
    selected = prefixes.loc[prefixes["Scenario_ID"].eq("workbook_case")].copy()
    if selected.empty or selected["Child_History"].duplicated().any():
        raise ValueError("workbook_case must contain unique prefix-contribution records")
    selected["Expected_Contribution"] = _numeric(
        selected, ["Expected_Contribution"], "workbook_case prefixes",
    )["Expected_Contribution"]
    selected = selected.loc[
        selected["Expected_Contribution"].abs().sort_values(ascending=False, kind="stable").index
    ]
    displayed = selected.iloc[:12]
    labels = [
        fill(history.replace(" → ", " / "), width=54, break_long_words=False, break_on_hyphens=False)
        for history in displayed["Child_History"]
    ]
    totals = displayed["Expected_Contribution"].tolist()
    colors = ["#56B4E9"] * len(totals)
    remaining = selected.iloc[len(displayed):]
    if not remaining.empty:
        labels.append(f"Remaining {len(remaining)} prefixes (sum)")
        totals.append(float(remaining["Expected_Contribution"].sum()))
        colors.append("#999999")
    figure, axis = plt.subplots(figsize=(11.5, max(4.8, len(labels) * 0.46 + 1.6)))
    positions = np.arange(len(labels))
    bars = axis.barh(
        positions, totals, color=colors,
        edgecolor="#333333", linewidth=0.5, zorder=3,
    )
    axis.bar_label(bars, labels=[f"{value:.6g}" for value in totals], padding=3)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Expected loss contribution")
    axis.set_ylabel("Component-choice prefix")
    axis.set_title(f"{case_id}: each unique prefix contributes once")
    axis.set_xlim(min(0.0, min(totals)) * 1.23, max(max(totals), 1.0) * 1.23)
    figure.text(
        0.5, 0.01,
        f"{len(selected)} unique prefixes; total = {selected['Expected_Contribution'].sum():.6g}. "
        "Complete details: prefix_contributions.csv.",
        ha="center", fontsize=TICK_LABELSIZE,
    )
    style_axes(axis)
    axis.grid(False, axis="y")
    axis.grid(True, axis="x", linestyle="--", linewidth=0.4, color="#d9d9d9", zorder=0)
    normalize_figure_typography(figure)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(figure, directory, "prefix_contributions")


def plot(input_excel=None, output_dir=None):
    """Read exported CSVs and plot only the selected workbook case."""
    directory = get_feature_output_dir("model_validation", input_excel) if output_dir is None else Path(output_dir)
    tables = _load_tables(directory)
    metadata = tables["run_metadata"].set_index("Key")["Value"]
    if "Input_SHA256" not in metadata or "Case_ID" not in metadata:
        raise ValueError("These CSVs predate workbook-case validation; solve and export again")
    case_id = str(metadata["Case_ID"])
    if output_dir is None and case_id != get_input_excel(input_excel).stem:
        raise ValueError("These CSVs belong to a different case; solve and export the selected case")
    for name in ("action_probabilities", "prefix_contributions", "terminal_trajectories", "validation_summary"):
        frame = tables[name]
        if "Case_ID" not in frame or not frame["Case_ID"].eq(case_id).all():
            raise ValueError(f"{name} mixes cases; solve and export again")
    with plt.rc_context():
        
        set_plot_style(legend_fontsize=9)
        _plot_loss_comparison(tables["validation_summary"], directory, case_id)
        _plot_prefix_contributions(tables["prefix_contributions"], directory, case_id)
    for extension in ("png", "svg"):
        (directory / f"fixed_prefix_contributions.{extension}").unlink(missing_ok=True)
    return {
        name: [directory / f"{name}.{extension}" for extension in ("png", "svg")]
        for name in ("loss_reconciliation", "prefix_contributions")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-excel", default=None)
    parser.add_argument("--output-dir", default=None)
    arguments = parser.parse_args()
    plot(input_excel=arguments.input_excel, output_dir=arguments.output_dir)


if __name__ == "__main__":
    main()
