"""Draw diagnostic and manuscript sensitivity figures from exported CSV files only."""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from common.plotting import (
    BASE_FONTSIZE, TICK_LABELSIZE, PANEL_CAPTION_FONTSIZE, TITLE_FONTSIZE,
    normalize_figure_typography, save_figure, set_plot_style, style_axes,
)
from features.sensitivity_analysis.export_csv import output_directory, read_tables
from features.sensitivity_analysis.parameters import PARAMETER_NAMES


# =========================================================

# =========================================================

COLORS = {
    "Fixed": "#3C5488",
    "Reoptimized": "#E64B35",
    "S1": "#4DBBD5",
    "ST": "#E64B35",
}
FIGURE_WIDTH = 11
PANEL_HEIGHT = 3.5
ERROR_CAPSIZE = 2
LOCAL_STYLE = {
    "legend.fontsize": 8,
    "axes.titlesize": TITLE_FONTSIZE,
    "savefig.dpi": 300,
}

PAPER_FIGURE_SIZE = (5.6, 7.2) 
PAPER_FONT_SIZE = BASE_FONTSIZE
PAPER_TICK_SIZE = TICK_LABELSIZE
PAPER_CAPTION_SIZE = PANEL_CAPTION_FONTSIZE

PAPER_LEGEND_SCALE = 2/ 3 
PAPER_LEGEND_SIZE = 7 * PAPER_LEGEND_SCALE

PAPER_CAPTION_Y = -0.07 
PAPER_XLIM = (-0.5, 1.0)
PAPER_XTICKS = (-0.5, 0.0, 0.5, 1.0)

PAPER_ADJUST = {
    "left": 0.07,
    "right": 0.975,
    "top": 0.99,
    "bottom": 0.05,
    "wspace": 0.22,
    "hspace": 0.15, 
}


PAPER_Y_PADDING = 0.7 

PAPER_BAR_STYLES = {
    "S1": {
        "facecolor": "#DCECF7",
        "edgecolor": "#1879B8",
        "hatch": "///",
    },
    "ST": {
        "facecolor": "#FDE3E5",
        "edgecolor": "#F11C26",
        "hatch": "\\" * 3,
    },
}

PAPER_PARAMETER_SYMBOLS = {
    "epss": r"$p_{v_{n_t,k}}$",
    "attack_duration": r"$\tau_{a_t}$",
    "attacker_cost": r"$\varsigma_{a_t}$",
    "attacker_monetary_benefit": r"$e^{A}_{v_{n_t,k}}$",
    "attacker_destruction_benefit": r"$d^{A}_{v_{n_t,k}}$",
    "attacker_preference": r"$\alpha$",
    "patch_effectiveness": r"$\lambda_{v_{n_t,k}}$",
    "firewall_effectiveness": r"$\rho$",
    "ids_effectiveness": r"$\epsilon$",
    "response_effectiveness": r"$\gamma$",
    "ids_detection_cycle": r"$T$",
    "prevention_cost": r"$c^{p}$",
    "detection_cost": r"$c^{d}$",
    "response_cost": r"$c^{r}$",
    "defender_monetary_loss": r"$e^{D}_{v_{n_t,k}}$",
    "defender_destruction_loss": r"$d^{D}_{v_{n_t,k}}$",
    "attacker_exposure_penalty_scale": r"$\varsigma_{a_t}^{\mathrm{det}}$",
    "response_penalty_scale": r"$\varsigma_{D,a_t}^{r}$",
}

PAPER_METRICS = {
    "Fixed_J": (
        "Total objective with fixed deployment",
        r"J(d_0)",
    ),
    "Fixed_Loss": (
        "Loss with fixed deployment",
        r"L(d_0)",
    ),
    "Reoptimized_J": (
        "Total objective after reoptimization",
        r"J(\widehat{d})",
    ),
    "Reoptimized_Loss": (
        "Loss after reoptimization",
        r"L(\widehat{d})",
    ),
    "Search_Gap": (
        "Objective gap after reoptimization",
        r"\Delta J",
    ),
    "Envelope_Loss_0.25B": (
        "Loss envelope at quarter budget",
        r"L^{\mathrm{env}}_{B/4}",
    ),
    "Envelope_Loss_0.5B": (
        "Loss envelope at half budget",
        r"L^{\mathrm{env}}_{B/2}",
    ),
    "Envelope_Loss_1B": (
        "Loss envelope at full budget",
        r"L^{\mathrm{env}}_{B}",
    ),
}


def _numeric(frame, name):
    return pd.to_numeric(
        frame.get(name, pd.Series(index=frame.index, dtype=float)),
        errors="coerce",
    )


def _records(frame, record_type):
    if "Record_Type" not in frame:
        return frame.iloc[0:0]
    return frame.loc[frame["Record_Type"].eq(record_type)]


def _parameter_order(tables):
    return list(PARAMETER_NAMES)


def _base_rows(frame):
    """Keep only A/B rows at the largest exported design level."""
    if frame.empty or "Sample_Role" not in frame:
        return frame.iloc[0:0]
    rows = frame.loc[frame["Sample_Role"].isin(("A", "B"))].copy()
    if rows.empty:
        return rows
    maximum = _numeric(rows, "Base_N").max()
    return rows.loc[_numeric(rows, "Base_N").eq(maximum)]


def _unavailable(axis, reason):
    axis.text(
        0.5,
        0.5,
        reason,
        ha="center",
        va="center",
        transform=axis.transAxes,
        wrap=True,
    )
    axis.set_xticks([])
    axis.set_yticks([])


def _finish(figure, directory, name, title):
    figure.suptitle(title, fontsize=TITLE_FONTSIZE)
    for axis in figure.axes:
        style_axes(axis)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(figure, directory, name)


def _context(metadata):
    row = metadata.iloc[0]
    title = (
        f"{row.get('Case_ID', 'Case unknown')} | "
        f"{row.get('Status', 'status unknown')}"
    )
    interpretation = row.get("Mode_Interpretation")
    if pd.notna(interpretation):
        title += f" | {interpretation}"
    try:
        configuration = json.loads(row.get("Configuration", "{}"))
        if configuration.get("n_samples") is not None:
            title += f" | N={configuration['n_samples']}"
    except (TypeError, json.JSONDecodeError):
        pass
    unit = str(row.get("Monetary_Unit", "unknown monetary unit"))
    return f"{title} | {unit}", unit


def _sobol_page(indices, directory, context, order, metrics, name):
    finite_n = _numeric(indices, "Base_N").dropna()
    rows = (
        indices.loc[indices["Parameter_2"].isna()].copy()
        if "Parameter_2" in indices
        else indices.copy()
    )
    if not finite_n.empty:
        rows = rows.loc[_numeric(rows, "Base_N").eq(finite_n.max())]

    rows["_order"] = pd.Categorical(
        rows.get("Parameter"),
        categories=order,
        ordered=True,
    )
    rows = rows.sort_values("_order")

    panel_height = max(PANEL_HEIGHT, 0.27 * len(order) + 1)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(FIGURE_WIDTH, panel_height * 2),
        squeeze=False,
    )

    for axis, metric in zip(axes.flat, metrics):
        group = rows.loc[
            rows.get(
                "Metric",
                pd.Series(index=rows.index, dtype=object),
            ).eq(metric)
        ]
        positions = np.arange(len(order))
        plotted = False

        for offset, field in ((-0.17, "S1"), (0.17, "ST")):
            values = _numeric(group, field)
            errors = _numeric(group, f"{field}_Conf")
            if values.empty:
                continue

            lookup = group.assign(
                _value=values,
                _error=errors,
            ).set_index("Parameter")

            values_by_parameter = np.array([
                lookup.loc[parameter, "_value"]
                if parameter in lookup.index
                else np.nan
                for parameter in order
            ])
            errors_by_parameter = np.array([
                lookup.loc[parameter, "_error"]
                if parameter in lookup.index
                else np.nan
                for parameter in order
            ])
            valid = np.isfinite(values_by_parameter)

            if valid.any():
                plotted = True
                axis.barh(
                    positions[valid] + offset,
                    values_by_parameter[valid],
                    height=0.32,
                    color=COLORS[field],
                    label=field,
                )
                with_ci = (
                    valid
                    & np.isfinite(errors_by_parameter)
                    & (errors_by_parameter >= 0)
                )
                if with_ci.any():
                    axis.errorbar(
                        values_by_parameter[with_ci],
                        positions[with_ci] + offset,
                        xerr=errors_by_parameter[with_ci],
                        fmt="none",
                        color="black",
                        capsize=ERROR_CAPSIZE,
                        linewidth=0.7,
                    )

        axis.set_title(
            f"{metric} | N="
            f"{int(finite_n.max()) if not finite_n.empty else 'unknown'}"
        )
        if plotted:
            axis.set_yticks(
                positions,
                ["EPSS" if parameter == "epss" else parameter for parameter in order],
            )
            axis.invert_yaxis()
            axis.axvline(0, color="black", linewidth=0.5)

            levels = _numeric(group, "CI_Level").dropna().unique()
            ci_label = f"{100 * levels[0]:g}% " if len(levels) == 1 else ""
            axis.set_xlabel(f"Sobol index; exported {ci_label}CI half-widths")
            axis.legend()
        else:
            _unavailable(axis, "Indices unavailable")

    _finish(fig, directory, name, context)
    return name


def _sobol_figures(indices, directory, context, order=None):
    order = list(order or PARAMETER_NAMES)
    return [
        _sobol_page(
            indices,
            directory,
            context,
            order,
            ("Fixed_J", "Fixed_Loss", "Reoptimized_J", "Reoptimized_Loss"),
            "sensitivity_sobol",
        ),
        _sobol_page(
            indices,
            directory,
            context,
            order,
            (
                "Search_Gap",
                "Envelope_Loss_0.25B",
                "Envelope_Loss_0.5B",
                "Envelope_Loss_1B",
            ),
            "sensitivity_sobol_additional_1",
        ),
    ]


def _diagnostic_figure(diagnostics, directory, context, order, unit):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FIGURE_WIDTH, max(PANEL_HEIGHT, 0.27 * len(order) + 1)),
    )
    convergence = _records(diagnostics, "convergence")
    convergence = convergence.loc[
        convergence.get(
            "Metric",
            pd.Series(index=convergence.index, dtype=object),
        ).eq("Reoptimized_J")
    ]
    values = convergence.assign(
        ST=_numeric(convergence, "ST"),
        Base_N=_numeric(convergence, "Base_N"),
    )
    grid = values.pivot_table(
        index="Parameter",
        columns="Base_N",
        values="ST",
        sort=False,
    )
    grid = grid.reindex(order).reindex(sorted(grid.columns), axis=1)

    if grid.empty:
        _unavailable(
            axes[0],
            "No finite convergence records; convergence unassessed.",
        )
    else:
        image = axes[0].imshow(
            grid.to_numpy(),
            aspect="auto",
            cmap="viridis",
        )
        axes[0].set_yticks(
            np.arange(len(grid)),
            [
                "EPSS" if parameter == "epss" else parameter
                for parameter in grid.index
            ],
        )
        axes[0].set_xticks(
            np.arange(len(grid.columns)),
            [f"{column:g}" for column in grid.columns],
        )
        axes[0].set(
            title="Reoptimized_J total effects by N",
            xlabel="Base N",
        )
        fig.colorbar(
            image,
            ax=axes[0],
            label="ST",
            fraction=0.045,
            pad=0.02,
        )
        if len(grid.columns) < 2:
            axes[0].set_title(
                "Reoptimized_J total effects by N\n"
                "Single N: convergence unassessed"
            )

    noise = _records(diagnostics, "optimization_noise")
    groups, labels = [], []

    if not noise.empty:
        for (design, sample), group in noise.groupby(
            ["Design_ID", "Sample_ID"],
            sort=False,
            dropna=False,
        ):
            values = _numeric(group, "Value").dropna().to_numpy()
            if values.size and len(groups) < 12:
                groups.append(values)
                labels.append(f"{design}\nsample {sample}")

    if groups:
        axes[1].boxplot(
            groups,
            tick_labels=labels,
            showfliers=False,
        )
        axes[1].tick_params(
            axis="x",
            labelrotation=25,
            labelsize=8,
        )
        axes[1].set_ylabel(f"Returned J ({unit})")
        axes[1].set_title(
            "Repeated-search J variation\n"
            "First 12 scenarios; descriptive"
        )
    else:
        _unavailable(axes[1], "No repeated-search noise records.")

    name = "sensitivity_diagnostics"
    _finish(fig, directory, name, context)
    return name


def _decision_figure(metrics, directory, context, unit):
    samples = _base_rows(_records(metrics, "sample"))
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(FIGURE_WIDTH, PANEL_HEIGHT * 2),
    )
    cost_loss, total, rates, change, gap, response = axes.flat

    for prefix in ("Fixed", "Reoptimized"):
        valid = (
            np.isfinite(_numeric(samples, f"{prefix}_Cost"))
            & np.isfinite(_numeric(samples, f"{prefix}_Loss"))
        )
        if valid.any():
            cost_loss.scatter(
                samples.loc[valid, f"{prefix}_Cost"],
                samples.loc[valid, f"{prefix}_Loss"],
                s=14,
                alpha=0.5,
                color=COLORS[prefix],
                label=prefix,
            )

    valid = (
        np.isfinite(_numeric(samples, "Fixed_J"))
        & np.isfinite(_numeric(samples, "Reoptimized_J"))
    )
    if valid.any():
        total.scatter(
            samples.loc[valid, "Fixed_J"],
            samples.loc[valid, "Reoptimized_J"],
            s=14,
            alpha=0.6,
        )
        bounds = [*total.get_xlim(), *total.get_ylim()]
        total.plot(
            [min(bounds), max(bounds)],
            [min(bounds), max(bounds)],
            "k--",
            linewidth=0.7,
        )

    cost_loss.set(
        xlabel=f"Cost ({unit})",
        ylabel=f"Loss ({unit})",
        title="Fixed and reoptimized decisions",
    )
    total.set(
        xlabel=f"Fixed J ({unit})",
        ylabel=f"Reoptimized J ({unit})",
        title="Paired outcomes",
    )

    if "Fixed_Feasible" in samples and "Resource_Reselected" in samples:
        rate_labels, rate_values = [], []
        for field in ("Fixed_Feasible", "Resource_Reselected"):
            values = samples[field].map({
                True: 1,
                False: 0,
                "True": 1,
                "False": 0,
            })
            values = pd.to_numeric(values, errors="coerce").dropna()
            if not values.empty:
                rate_labels.append(
                    "Fixed feasible"
                    if field == "Fixed_Feasible"
                    else "Original deployment retained"
                )
                rate_values.append(values.mean())

        if rate_values:
            rates.bar(
                np.arange(len(rate_values)),
                rate_values,
                color=COLORS["Fixed"],
            )
            rates.set_xticks(
                np.arange(len(rate_values)),
                rate_labels,
                rotation=25,
                ha="right",
            )

    rates.set(
        title="A/B-row descriptive rates",
        ylabel="Rate",
        ylim=(0, 1.05),
    )

    for axis, field, title, xlabel in (
        (
            change,
            "Hamming",
            "Deployment changes",
            "Hamming distance",
        ),
        (
            gap,
            "Search_Gap",
            "Search gap where baseline is feasible",
            f"J_fixed - J_reoptimized ({unit})",
        ),
        (
            response,
            "P_Response_Change",
            "Response propensity changes",
            "Absolute initial P_Response change",
        ),
    ):
        values = _numeric(samples, field).dropna()
        if not values.empty:
            axis.hist(values, bins="auto", alpha=0.7)
        axis.set(
            title=title,
            xlabel=xlabel,
            ylabel="A/B rows",
        )

    gap.axvline(0, color="black", linewidth=0.6)

    for axis in (cost_loss, total, rates, change, gap, response):
        if axis.get_legend_handles_labels()[0]:
            axis.legend()
        has_data = (
            bool(axis.collections)
            if axis in (cost_loss, total)
            else bool(axis.patches)
        )
        if not has_data:
            _unavailable(axis, "No finite A/B observations for this panel.")

    name = "sensitivity_decisions"
    _finish(fig, directory, name, context + " | A/B rows only; descriptive")
    return name


def _budget_figure(metrics, directory, context, unit):
    rows = _base_rows(_records(metrics, "budget_envelope"))
    fig, axis = plt.subplots(
        1,
        1,
        figsize=(FIGURE_WIDTH, PANEL_HEIGHT),
    )
    values = rows.assign(
        Budget_Threshold=_numeric(rows, "Budget_Threshold"),
        Envelope_Loss=_numeric(rows, "Envelope_Loss"),
    )
    summary = (
        values.groupby("Budget_Threshold")["Envelope_Loss"]
        .median()
        .dropna()
        .sort_index()
    )

    if summary.empty:
        _unavailable(axis, "No finite A/B budget-envelope records.")
    else:
        axis.plot(
            summary.index,
            summary,
            marker="o",
            color=COLORS["Reoptimized"],
        )

    axis.set(
        title="Common-budget loss envelope; descriptive median",
        xlabel=f"Budget threshold ({unit})",
        ylabel=f"Loss ({unit})",
    )
    name = "sensitivity_budgets"
    _finish(fig, directory, name, context)
    return name


def _paper_sobol_figure(indices, order, metrics, panel_start=0):
    unknown = [
        parameter
        for parameter in order
        if parameter not in PAPER_PARAMETER_SYMBOLS
    ]
    if unknown:
        raise ValueError(f"Missing manuscript symbols for factors: {unknown}")

    rows = (
        indices.loc[indices["Parameter_2"].isna()].copy()
        if "Parameter_2" in indices
        else indices.copy()
    )
    levels = _numeric(rows, "Base_N").dropna()
    if not levels.empty:
        rows = rows.loc[_numeric(rows, "Base_N").eq(levels.max())]

    figure, axes = plt.subplots(
        2,
        2,
        figsize=PAPER_FIGURE_SIZE,
        squeeze=False,
    )
    figure.subplots_adjust(**PAPER_ADJUST)
    positions = np.arange(len(order))

    for panel_index, (axis, metric) in enumerate(zip(axes.flat, metrics)):
        caption, output = PAPER_METRICS[metric]
        group = rows.loc[
            rows.get(
                "Metric",
                pd.Series(index=rows.index, dtype=object),
            ).eq(metric)
        ]
        if group["Parameter"].duplicated().any():
            plt.close(figure)
            raise ValueError(f"Duplicate Sobol rows for {metric}")

        lookup = group.set_index("Parameter").reindex(order)
        plotted = False

        for offset, field in ((-0.18, "S1"), (0.18, "ST")):
            values = _numeric(lookup, field).to_numpy(dtype=float)
            valid = np.isfinite(values)
            if valid.any():
                plotted = True
                axis.barh(
                    positions[valid] + offset,
                    values[valid],
                    height=0.32,
                    linewidth=0.65,
                    zorder=3,
                    **PAPER_BAR_STYLES[field],
                )

        axis.set_title("")
        axis.set_xlabel("")
        axis.set_ylabel("")

        axis.text(
            0.5,
            PAPER_CAPTION_Y,
            f"({chr(97 + panel_start + panel_index)}) {caption}",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=PAPER_CAPTION_SIZE,
        )
        axis.tick_params(
            axis="both",
            labelsize=PAPER_TICK_SIZE,
            length=2.5,
        )
        axis.set_xlim(PAPER_XLIM)
        axis.set_xticks(PAPER_XTICKS)
        axis.set_axisbelow(True)

        style_axes(axis)
        axis.grid(
            True,
            axis="x",
            linestyle="-",
            linewidth=0.35,
            color="#dedede",
        )

        if plotted:
            axis.set_yticks(
                positions,
                [PAPER_PARAMETER_SYMBOLS[parameter] for parameter in order],
            )

            axis.set_ylim(  
                len(order) - 1 + PAPER_Y_PADDING,
                -PAPER_Y_PADDING,
            )


            handles = [
                Patch(
                    **PAPER_BAR_STYLES[field],
                    linewidth=0.65 * PAPER_LEGEND_SCALE,
                    label=rf"$S_{{i,{output}}}^{{({exponent})}}$",
                )
                for field, exponent in (("S1", "1"), ("ST", "T"))
            ]

            axis.legend(
                handles=handles,
                loc="upper right",
                fontsize=PAPER_LEGEND_SIZE,
                handlelength=1.8,
                handleheight=1.0,
                borderpad=0.3,
                labelspacing=0.25,
                borderaxespad=0.35,
                framealpha=1.0,
            )
        else:
            _unavailable(axis, "Indices unavailable")

    return figure


def _save_paper_figure(figure, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    normalize_figure_typography(figure, kind="multi")
    try:
        with plt.rc_context({"savefig.bbox": None}):
            for extension in ("png", "svg"):
                figure.savefig(
                    directory / f"{name}.{extension}",
                    format=extension,
                    bbox_inches=None,
                    pad_inches=0,
                    dpi=300,
                )
    finally:
        plt.close(figure)


def _paper_sobol_figures(indices, directory, order):
    pages = (
        (
            "sensitivity_sobol_paper",
            ("Fixed_J", "Fixed_Loss", "Reoptimized_J", "Reoptimized_Loss"),
        ),
        (
            "sensitivity_sobol_additional_1_paper",
            (
                "Search_Gap",
                "Envelope_Loss_0.25B",
                "Envelope_Loss_0.5B",
                "Envelope_Loss_1B",
            ),
        ),
    )
    names = []

    with plt.rc_context({
        "hatch.linewidth": 0.6,
        "font.size": PAPER_FONT_SIZE,
    }):

        for page_index, (name, metrics) in enumerate(pages):
            figure = _paper_sobol_figure(
                indices,
                order,
                metrics,
                panel_start=4 * page_index,
            )
            _save_paper_figure(figure, directory, name)
            names.append(name)

    return names


def plot(input_excel=None, output_dir=None, *, paper_only=False):
    """Read exported CSVs; optionally save only the two manuscript figures."""
    directory = output_directory(input_excel, output_dir)
    tables = read_tables(directory)
    context, unit = _context(tables["run_metadata"])
    order = _parameter_order(tables)
    written = []

    with plt.rc_context():
        set_plot_style()
        plt.rcParams.update(LOCAL_STYLE)

        if not paper_only:
            written.extend(
                _sobol_figures(
                    tables["sobol_indices"],
                    directory,
                    context,
                    order,
                )
            )
            written.append(
                _diagnostic_figure(
                    tables["sobol_diagnostics"],
                    directory,
                    context,
                    order,
                    unit,
                )
            )
            written.append(
                _decision_figure(
                    tables["sample_metrics"],
                    directory,
                    context,
                    unit,
                )
            )
            written.append(
                _budget_figure(
                    tables["sample_metrics"],
                    directory,
                    context,
                    unit,
                )
            )

        written.extend(
            _paper_sobol_figures(
                tables["sobol_indices"],
                directory,
                order,
            )
        )

    print(f"Sensitivity figures read only CSV and saved to {directory}")
    return [
        directory / f"{name}.{extension}"
        for name in written
        for extension in ("png", "svg")
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-excel")
    parser.add_argument("--output-dir")
    parser.add_argument("--paper-only", action="store_true")
    arguments = parser.parse_args()
    plot(
        input_excel=arguments.input_excel,
        output_dir=arguments.output_dir,
        paper_only=arguments.paper_only,
    )