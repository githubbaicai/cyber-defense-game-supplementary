"""Plot validation losses and per-step evolutionary probabilities from CSV."""

from __future__ import annotations

import argparse
from html import escape
import itertools
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from common.config import CSV_FILES, get_feature_output_dir
from common.case_config import get_input_excel
from common.plotting import (
    TICK_LABELSIZE, TITLE_FONTSIZE, DPI_SAVE, normalize_figure_typography,
    plot_style, set_plot_style,
)
from features.pruning_validation.export_csv import EVOLUTION_FILES, evolution_directory


EVOLUTION_MARKERS = ("o", "s", "^", "v", "D", "X", "P", "*")
# =========================================================

# =========================================================
PLOT_STYLE = plot_style(legend_fontsize=10)


def _plot_evolution_axes(axis, group, title, max_generation=None, compact=False):
    last_generation = int(group["Generation"].max())
    limit = last_generation if max_generation is None else min(max_generation, last_generation)
    axis.set_xlim(-1, limit + 2)
    axis.set_ylim(-0.02, 1.05)
    axis.set_xlabel("Generation")
    axis.set_ylabel("Probability")
    axis.set_title(textwrap.fill(title, width=70), fontsize=TITLE_FONTSIZE)
    axis.grid(True, color="#d9d9d9", linewidth=0.4)
    markers = itertools.cycle(EVOLUTION_MARKERS)
    for _, line in group.groupby("Strategy_ID", sort=False):
        line = line.sort_values("Generation")
        final_probability = float(line["Probability"].iloc[-1])
        label = f"{line['Strategy'].iloc[0]}: {final_probability:.6g}"
        plotted, = axis.plot(
            line["Generation"], line["Probability"],
            linewidth=1.5, marker=next(markers), markersize=4,
            markevery=8, label=label,
        )
        if round(final_probability, 2) != 0:
            axis.text(
                limit * 0.55,
                final_probability - 0.065 if final_probability > 0.9 else final_probability + 0.03,
                f"{final_probability:.2f}",
                color=plotted.get_color(), ha="center", fontsize=TICK_LABELSIZE,
            )
    axis.legend(
        loc="upper right", frameon=True, edgecolor="black",
        fontsize=7 if compact else 9, labelspacing=0.15,
        title="Strategy: final probability", title_fontsize=7 if compact else 9,
    )


def _save_evolution_figure(figure, directory, name):
    normalize_figure_typography(figure)
    figure.tight_layout()
    for extension in ("png", "svg"):
        figure.savefig(directory / f"{name}.{extension}", dpi=DPI_SAVE, bbox_inches="tight")
    plt.close(figure)


def _plot_probability_axes(axis, group, title, id_column, label_column,
                           max_generation=None, compact=False):
    last_generation = int(group["Generation"].max())
    limit = last_generation if max_generation is None else min(max_generation, last_generation)
    axis.set_xlim(-1, limit + 2)
    axis.set_ylim(-0.02, 1.05)
    axis.set_xlabel("Generation")
    axis.set_ylabel("Probability")
    axis.set_title(textwrap.fill(title, width=70), fontsize=TITLE_FONTSIZE)
    axis.grid(True, color="#d9d9d9", linewidth=0.4)
    markers = itertools.cycle(EVOLUTION_MARKERS)
    for _, line in group.groupby(id_column, sort=False):
        line = line.sort_values("Generation")
        final_probability = float(line["Probability"].iloc[-1])
        label = f"{line[label_column].iloc[0]}: {final_probability:.6g}"
        axis.plot(line["Generation"], line["Probability"], linewidth=1.5,
                  marker=next(markers), markersize=4, markevery=8, label=label)
    axis.legend(loc="upper right", frameon=True, edgecolor="black",
                fontsize=7 if compact else 9, labelspacing=0.15,
                title="Final probability", title_fontsize=7 if compact else 9)


def _plot_defender_bundle(directory, groups, title, max_generation=None,
                          compact=False, suffix=""):
    attacker, defender, policy = groups
    figure, axes = plt.subplots(3, 1, figsize=(10, 12 if not compact else 10), squeeze=False)
    axes = axes[:, 0]
    _plot_evolution_axes(axes[0], attacker, f"Attacker: {title}", max_generation, compact)
    if not defender.empty:
        _plot_probability_axes(axes[1], defender, f"Defender induced states: {title}",
                               "Strategy_ID", "Strategy", max_generation, compact)
    else:
        axes[1].set_visible(False)
    if not policy.empty:
        _plot_probability_axes(axes[2], policy, f"Defender conditional response policy: {title}",
                               "Strategy_ID", "Strategy", max_generation, compact)
    else:
        axes[2].set_visible(False)
    _save_evolution_figure(figure, directory, f"evolution_bundle_{suffix}")


def export_evolution_report(input_excel=None, case_id=4,
                            include_unreachable=False):
    input_excel = get_input_excel(input_excel)
    directory = evolution_directory(input_excel, case_id)
    nodes = pd.read_csv(directory / EVOLUTION_FILES["evolution_nodes"])
    if not include_unreachable:
        nodes = nodes[nodes["Node_Reach_Probability"] > 1e-9]
    figures = {}
    for mode, mode_nodes in nodes.groupby("Mode", sort=False):
        for node_index, node in enumerate(mode_nodes.itertuples(index=False), start=1):
            relative_path = f"{mode}/evolution_bundle_node_{node_index:03d}.png"
            if (directory / relative_path).is_file():
                figures.setdefault(node.Node_Path, {})[mode] = (relative_path, node)
    summary = pd.read_csv(directory / "case_summary.csv").iloc[0]
    sections = []
    for node_index, (node_path, modes) in enumerate(figures.items(), start=1):
        panels = []
        for mode, title in (("unpruned", "Unpruned"), ("pruned", "Pruned")):
            if mode not in modes:
                panels.append(f"<section><h3>{title}</h3><p>This node was not retained or its arrival probability is below the display threshold.</p></section>")
                continue
            relative_path, node = modes[mode]
            panels.append(
                f"<section><h3>{title}</h3>"
                f"<p>Arrival probability: {node.Node_Reach_Probability:.9g}; "
                f"iterations: {node.Iterations}</p>"
                f'<a href="{escape(relative_path, quote=True)}">'
                f'<img loading="lazy" src="{escape(relative_path, quote=True)}" '
                f'alt="{escape(node_path, quote=True)} — {title}"></a></section>'
            )
        sections.append(
            f'<details{" open" if node_index == 1 else ""}>'
            f"<summary>{node_index}. {escape(node_path)}</summary>"
            f'<div class="panels">{"".join(panels)}</div></details>'
        )
    scope = "all nodes" if include_unreachable else "nodes with arrival probability greater than 1e-9"
    content = (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        f"<title>Case {case_id} evolution comparison</title>"
        '<style>body{font-family:system-ui,sans-serif;margin:24px;color:#222;}'
        'table{border-collapse:collapse;}th,td{padding:8px 16px;border:1px solid #ddd;}'
        'details{margin:16px 0;border:1px solid #ddd;padding:12px;}'
        'summary{cursor:pointer;overflow-wrap:anywhere;}.panels{display:grid;'
        'grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;}img{width:100%;}'
        '@media(max-width:900px){.panels{grid-template-columns:1fr;}}</style>'
        f"<h1>Case {case_id}: evolution with and without pruning</h1>"
        f"<p>Input: {escape(Path(input_excel).name)}. Showing {scope}, with {len(figures)} state paths. "
        "Complete node data are in the CSV files in the same directory. Click a node to expand it, or an image to view the original.</p>"
        "<p>Each figure shows, from top to bottom: attacker strategies, three detection states, and the response decision after detection. "
        "The three-state probabilities are jointly determined by detection rates and response decisions.</p>"
        "<table><tr><th>Mode</th><th>Simulated nodes</th><th>Attacker payoff</th><th>Defender loss</th></tr>"
        f"<tr><td>Unpruned</td><td>{int(summary.Node_Count_NoPrune)}</td>"
        f"<td>{summary.Attacker_Payoff_NoPrune:.12g}</td><td>{summary.Loss_NoPrune:.12g}</td></tr>"
        f"<tr><td>Pruned</td><td>{int(summary.Node_Count_Pruned)}</td>"
        f"<td>{summary.Attacker_Payoff_Pruned:.12g}</td><td>{summary.Loss_Pruned:.12g}</td></tr></table>"
        f'{"".join(sections)}</html>'
    )
    report_path = directory / "evolution_comparison.html"
    report_path.write_text(content, encoding="utf-8")
    return report_path


def plot_case_evolution(input_excel=None, case_id=4, include_unreachable=False):
    input_excel = get_input_excel(input_excel)
    directory = evolution_directory(input_excel, case_id)
    history_path = directory / EVOLUTION_FILES["game_strategy_history"]
    if not history_path.is_file():
        return {}
    history = pd.read_csv(history_path)
    nodes = pd.read_csv(directory / EVOLUTION_FILES["evolution_nodes"])
    if history.empty:
        return {}
    defender_path = directory / EVOLUTION_FILES.get("defender_strategy_history", "defender_strategy_history.csv")
    policy_path = directory / EVOLUTION_FILES.get("response_policy_history", "response_policy_history.csv")
    defender_history = pd.read_csv(defender_path) if defender_path.is_file() else pd.DataFrame()
    policy_history = pd.read_csv(policy_path) if policy_path.is_file() else pd.DataFrame()
    plotted_counts = {}
    with plt.rc_context(PLOT_STYLE):
        for mode, mode_nodes in nodes.groupby("Mode", sort=False):
            if not include_unreachable:
                mode_nodes = mode_nodes[mode_nodes["Node_Reach_Probability"] > 1e-9]
            if mode_nodes.empty:
                continue
            mode_directory = directory / mode
            mode_directory.mkdir(parents=True, exist_ok=True)
            groups = []
            for node_index, node in enumerate(mode_nodes.itertuples(index=False), start=1):
                group = history[(history["Mode"] == mode) & (history["Node_Path"] == node.Node_Path)]
                defender_group = defender_history[(defender_history["Mode"] == mode) & (defender_history["Node_Path"] == node.Node_Path)] if not defender_history.empty else pd.DataFrame()
                policy_group = policy_history[(policy_history["Mode"] == mode) & (policy_history["Node_Path"] == node.Node_Path)] if not policy_history.empty else pd.DataFrame()
                title = f"Case {case_id} / {mode} / Step {node.Step}\n{node.Node_Path}"
                figure, axis = plt.subplots(figsize=(8, 6))
                _plot_evolution_axes(axis, group, title)
                _save_evolution_figure(figure, mode_directory, f"evolution_node_{node_index:03d}")
                _plot_defender_bundle(mode_directory, (group, defender_group, policy_group), title,
                                      suffix=f"node_{node_index:03d}")
                groups.append((group, defender_group, policy_group, title))
            for start in range(0, len(groups), 12):
                page = groups[start:start + 12]
                page_suffix = "" if len(groups) <= 12 else f"_page_{start // 12 + 1}"
                for max_generation, suffix in ((None, ""), (20, "_first_20")):
                    row_count = (len(page) + 2) // 3
                    figure, axes = plt.subplots(row_count, 3, figsize=(18, 4.8 * row_count), squeeze=False)
                    for axis, (group, _, _, title) in zip(axes.flat, page):
                        _plot_evolution_axes(axis, group, title, max_generation, compact=True)
                    for axis in list(axes.flat)[len(page):]:
                        axis.set_visible(False)
                    _save_evolution_figure(figure, mode_directory, f"evolution_overview{page_suffix}{suffix}")
                for player_name, player_index in (("defender", 1), ("response_policy", 2)):
                    for max_generation, suffix in ((None, ""), (20, "_first_20")):
                        row_count = (len(page) + 2) // 3
                        figure, axes = plt.subplots(row_count, 3, figsize=(18, 4.8 * row_count), squeeze=False)
                        for axis, item in zip(axes.flat, page):
                            group = item[player_index]
                            if group.empty:
                                axis.set_visible(False)
                            else:
                                _plot_probability_axes(axis, group, item[3], "Strategy_ID", "Strategy", max_generation, compact=True)
                        for axis in list(axes.flat)[len(page):]:
                            axis.set_visible(False)
                        _save_evolution_figure(figure, mode_directory, f"{player_name}_evolution_overview{page_suffix}{suffix}")
            plotted_counts[mode] = len(groups)
            print(f"Case {case_id} {mode}: {len(groups)} evolutionary plots -> {mode_directory}")
    export_evolution_report(input_excel, case_id, include_unreachable)
    return plotted_counts


def _initial_probability_case_sort(value):
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _initial_probability_player_name(value):
    normalized = str(value).strip().lower()
    if normalized == "attacker":
        return "attacker"
    if normalized in {"induced_state", "defender_state", "defender states", "defender-state"}:
        return "defender_state"
    if normalized in {"response_policy", "response policy", "response-policy"}:
        return "response_policy"
    return normalized


def _initial_probability_distribution(value, run_id=""):
    normalized = str(value).strip().lower()
    run_normalized = str(run_id).strip().lower()
    if normalized.startswith("random") or run_normalized.startswith("random"):
        return "random"
    if normalized == "uniform" or run_normalized == "uniform":
        return "uniform"
    if normalized == "original" or run_normalized == "original":
        return "original"
    return normalized


def _initial_probability_run_for_plot(group, distribution):
    if group.empty:
        return group
    distribution_mask = group["Initial_Distribution"].map(
        lambda value: _initial_probability_distribution(value)
    ) == distribution
    candidates = group[distribution_mask]
    if distribution == "random":
        run_mask = group["Run_ID"].astype(str).str.lower().eq("random_0")
        random_zero = group[run_mask]
        if not random_zero.empty:
            return random_zero
    if candidates.empty:
        run_mask = group["Run_ID"].astype(str).str.lower().eq(distribution)
        candidates = group[run_mask]
    if candidates.empty:
        return candidates
    run_ids = sorted(candidates["Run_ID"].astype(str).unique())
    return candidates[candidates["Run_ID"].astype(str) == run_ids[0]]


def _initial_probability_node_key(case_id, node_path):
    return str(case_id), str(node_path)


def plot_initial_probability_comparison(directory, probability_tol=1e-3):
    directory = Path(directory)
    nested_directory = directory / "initial_probability_comparison"
    if not (directory / "evolution_history.csv").is_file() and nested_directory.is_dir():
        directory = nested_directory
    history_path = directory / "evolution_history.csv"
    report_path = directory / "initial_probability_comparison.html"
    if not history_path.is_file():
        raise FileNotFoundError(history_path)

    history = pd.read_csv(history_path)
    if history.empty:
        report_path.write_text(
            "<!doctype html><html lang='zh-CN'><meta charset='utf-8'>"
            "<title>Initial probability comparison</title><h1>Initial probability comparison</h1>"
            "<p>evolution_history.csv is empty.</p></html>",
            encoding="utf-8",
        )
        return report_path

    required_columns = {
        "Case_ID", "Node_Path", "Reference_Node_Reach_Probability",
        "Initial_Distribution", "Run_ID", "Player", "Strategy_ID",
        "Strategy", "Generation", "Probability",
    }
    missing_columns = required_columns.difference(history.columns)
    if missing_columns:
        raise ValueError(
            "evolution_history.csv is missing columns: " + ", ".join(sorted(missing_columns))
        )
    history["Player"] = history["Player"].map(_initial_probability_player_name)
    history["Generation"] = pd.to_numeric(history["Generation"], errors="coerce")
    history["Probability"] = pd.to_numeric(history["Probability"], errors="coerce")
    history = history.dropna(subset=["Generation", "Probability"])

    comparison_path = directory / "node_comparison.csv"
    comparison = pd.read_csv(comparison_path) if comparison_path.is_file() else pd.DataFrame()
    diagnostics_path = directory / "node_diagnostics.csv"
    diagnostics = pd.read_csv(diagnostics_path) if diagnostics_path.is_file() else pd.DataFrame()
    case_summary_path = directory / "case_summary.csv"
    case_summary = pd.read_csv(case_summary_path) if case_summary_path.is_file() else pd.DataFrame()
    reference_cases_path = directory / "reference_cases.csv"
    reference_cases = pd.read_csv(reference_cases_path) if reference_cases_path.is_file() else pd.DataFrame()
    if "Probability_Comparison_Tolerance" in reference_cases.columns:
        saved_tolerances = pd.to_numeric(
            reference_cases["Probability_Comparison_Tolerance"], errors="coerce"
        ).dropna()
        if not saved_tolerances.empty:
            probability_tol = float(saved_tolerances.iloc[0])
    max_iterations = 5000
    if "Max_Iterations" in reference_cases.columns:
        saved_iterations = pd.to_numeric(reference_cases["Max_Iterations"], errors="coerce").dropna()
        if not saved_iterations.empty:
            max_iterations = int(saved_iterations.max())
    if not case_summary.empty and "Scope" in case_summary.columns:
        visible_summary = case_summary[
            case_summary["Scope"] == "reference_reachable_above_1e-9"
        ]
        if not visible_summary.empty:
            case_summary = visible_summary

    metadata_columns = ["Case_ID", "Node_Path", "Reference_Node_Reach_Probability"]
    metadata = history[metadata_columns].drop_duplicates()
    if not comparison.empty:
        comparison_metadata = comparison[
            [column for column in metadata_columns if column in comparison.columns]
        ].drop_duplicates()
        metadata = pd.concat([metadata, comparison_metadata], ignore_index=True).drop_duplicates(
            subset=["Case_ID", "Node_Path"]
        )
    metadata = metadata.sort_values(
        by=["Case_ID", "Node_Path"],
        key=lambda column: column.map(_initial_probability_case_sort)
        if column.name == "Case_ID" else column.astype(str),
    )

    image_records = {}
    node_records = {}
    for (case_id, node_path), node_metadata in metadata.groupby(
        ["Case_ID", "Node_Path"], sort=False
    ):
        node_history = history[
            (history["Case_ID"].astype(str) == str(case_id))
            & (history["Node_Path"].astype(str) == str(node_path))
        ].copy()
        reach_values = pd.to_numeric(
            node_metadata["Reference_Node_Reach_Probability"], errors="coerce"
        )
        reach = float(reach_values.iloc[0]) if not reach_values.isna().all() else 0.0
        key = _initial_probability_node_key(case_id, node_path)
        node_records[key] = {"reach": reach}
        if node_history.empty or reach <= 1e-9:
            continue
        case_directory = directory / f"case_{str(case_id).zfill(2)}"
        case_directory.mkdir(parents=True, exist_ok=True)
        node_index = len([record for record in image_records if record[0] == str(case_id)]) + 1
        figure, axes = plt.subplots(3, 2, figsize=(16, 12), squeeze=False)
        selected_labels = (("original", "original"), ("uniform", "uniform"), ("random_0", "random"))
        for row_index, (display_label, distribution) in enumerate(selected_labels):
            for column_index, player in enumerate(("attacker", "defender_state")):
                axis = axes[row_index, column_index]
                player_history = node_history[node_history["Player"] == player]
                selected = _initial_probability_run_for_plot(player_history, distribution)
                if selected.empty:
                    axis.text(0.5, 0.5, "No trajectory for this initialization", ha="center", va="center")
                    axis.set_axis_off()
                    continue
                for _, line in selected.groupby("Strategy_ID", sort=False):
                    line = line.sort_values("Generation")
                    strategy_name = str(line["Strategy"].iloc[0])
                    axis.plot(
                        line["Generation"], line["Probability"],
                        linewidth=1.5, label=strategy_name,
                    )
                axis.set_ylim(-0.02, 1.05)
                axis.set_xlabel("Generation")
                axis.set_ylabel("Probability")
                axis.set_title(
                    f"{display_label} / "
                    + ("attacker p" if player == "attacker" else "defender state q (3 strategies)")
                )
                axis.grid(True, color="#d9d9d9", linewidth=0.4)
                axis.legend(fontsize=7, loc="best", title="Strategy", title_fontsize=7)
        figure.suptitle(
            f"Case {case_id} / {node_path}\nReference reach probability: {reach:.9g}",
            fontsize=TITLE_FONTSIZE,
        )
        normalize_figure_typography(figure)
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        image_name = f"node_{node_index:03d}_initial_probability_comparison"
        for extension in ("png", "svg"):
            figure.savefig(case_directory / f"{image_name}.{extension}", dpi=DPI_SAVE, bbox_inches="tight")
        plt.close(figure)
        image_records[key] = {
            "relative_path": f"case_{str(case_id).zfill(2)}/{image_name}.png",
            "reach": reach,
        }

    comparison_rows = {}
    if not comparison.empty and {"Case_ID", "Node_Path"}.issubset(comparison.columns):
        for row in comparison.to_dict("records"):
            key = _initial_probability_node_key(row["Case_ID"], row["Node_Path"])
            comparison_rows.setdefault(key, []).append(row)

    def format_value(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "—"
        if isinstance(value, bool):
            return "True" if value else "False"
        try:
            return f"{float(value):.6g}"
        except (TypeError, ValueError):
            return escape(str(value))

    case_sections = []
    case_ids = sorted({key[0] for key in node_records}, key=_initial_probability_case_sort)
    for case_id in case_ids:
        case_nodes = [key for key in node_records if key[0] == case_id]
        case_nodes.sort(key=lambda key: key[1])
        rows = []
        details = []
        for key in case_nodes:
            _, node_path = key
            node_comparisons = comparison_rows.get(key, [])
            row = max(node_comparisons, key=lambda entry: float(entry.get("Iterations", 0)), default={})
            checkpoint_cells = []
            for generation in (200, max_iterations):
                checkpoint_comparisons = [
                    entry for entry in node_comparisons
                    if float(entry.get("Iterations", 0)) == generation
                ]
                checkpoint = checkpoint_comparisons[0] if checkpoint_comparisons else {}
                checkpoint_cells.extend([
                    format_value(checkpoint.get("Max_Attacker_Probability_Difference")),
                    format_value(checkpoint.get("Max_Response_Policy_Difference")),
                ])
            for current_comparison in sorted(
                node_comparisons or [{}],
                key=lambda entry: float(entry.get("Iterations", 0)),
            ):
                rows.append(
                    "<tr><td>" + "</td><td>".join([
                        escape(str(node_path)),
                        format_value(current_comparison.get("Iterations")),
                        format_value(current_comparison.get("Max_Attacker_Probability_Difference")),
                        format_value(current_comparison.get("Max_Response_Policy_Difference")),
                        format_value(current_comparison.get("Max_Induced_State_Difference")),
                        escape(str(current_comparison.get("All_Candidate_Stationary", "—"))),
                        escape(str(current_comparison.get("Comparison_Status", "—"))),
                        *checkpoint_cells,
                    ]) + "</td></tr>"
                )
            image = image_records.get(key)
            if image:
                image_markup = (
                    f'<a href="{escape(image["relative_path"], quote=True)}">'
                    f'<img loading="lazy" src="{escape(image["relative_path"], quote=True)}" '
                    f'alt="{escape(str(node_path), quote=True)}"></a>'
                )
                reach_text = f"Reference reach probability: {image['reach']:.9g}."
            else:
                reach_text = f"Reference reach probability: {node_records[key]['reach']:.9g}."
                image_markup = (
                    "<p>This node has reach <= 1e-9; no plot is generated under the stated rule.</p>"
                    if node_records[key]["reach"] <= 1e-9
                    else "<p>No evolution-history data are available for this node.</p>"
                )
            run_table = ""
            if not diagnostics.empty:
                node_diagnostics = diagnostics[
                    (diagnostics["Case_ID"].astype(str) == str(case_id))
                    & (diagnostics["Node_Path"].astype(str) == str(node_path))
                ]
                diagnostic_columns = [
                    "Iterations", "Run_ID", "Candidate_Stationary",
                    "Attacker_Residual", "Defender_Residual", "Attacker_Invasion_Gap",
                    "Defender_Invasion_Gap", "Tail_Max_Change", "Tail_Max_Span",
                    "Attacker_Payoff", "Defender_Loss",
                ]
                available_diagnostic_columns = [
                    column for column in diagnostic_columns if column in diagnostics.columns
                ]
                run_table = (
                    "<h4>Stability evidence for each trial</h4><table><thead><tr>"
                    + "".join(f"<th>{escape(column)}</th>" for column in available_diagnostic_columns)
                    + "</tr></thead><tbody>"
                    + "".join(
                        "<tr>" + "".join(
                            f"<td>{format_value(run.get(column))}</td>"
                            for column in available_diagnostic_columns
                        ) + "</tr>"
                        for run in node_diagnostics.to_dict("records")
                    )
                    + "</tbody></table>"
                )
            details.append(
                f'<details><summary>{escape(str(node_path))}</summary>'
                f'<p>{reach_text} Max attacker difference: '
                f'{format_value(row.get("Max_Attacker_Probability_Difference"))}; '
                f'Max response-policy difference: '
                f'{format_value(row.get("Max_Response_Policy_Difference"))}; '
                f'Stationary candidates: {escape(str(row.get("All_Candidate_Stationary", "—")))}; '
                f'Status: {escape(str(row.get("Comparison_Status", "—")))}</p>'
                f'{image_markup}{run_table}</details>'
            )
        case_table = (
            "<table><thead><tr><th>Node_Path</th>"
            "<th>Iterations</th>"
            "<th>Max attacker Δ</th><th>Max response-policy Δ</th>"
            "<th>Max induced-state Δ</th><th>All stationary candidates</th>"
            "<th>Status</th><th>200 attacker Δ</th>"
            f"<th>200 response Δ</th><th>{max_iterations} attacker Δ</th>"
            f"<th>{max_iterations} response Δ</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>"
        )
        case_sections.append(
            f'<details class="case" open><summary>Case {escape(str(case_id))} '
            f'({len(case_nodes)} nodes)</summary>{case_table}{"".join(details)}</details>'
        )

    summary_table = ""
    if not case_summary.empty:
        summary_columns = [
            "Case_ID", "Iterations", "Scope", "Node_Count", "Visible_Node_Count",
            "Same_Attacker_Node_Count", "Same_Response_Node_Count",
            "Same_Induced_State_Node_Count", "Same_Stationary_Candidate_Node_Count",
            "Different_Stationary_Candidate_Node_Count", "Inconclusive_Node_Count",
        ]
        available = [column for column in summary_columns if column in case_summary.columns]
        summary_table = (
            "<h2>Case summary</h2><table><thead><tr>"
            + "".join(f"<th>{escape(column)}</th>" for column in available)
            + "</tr></thead><tbody>"
            + "".join(
                "<tr>" + "".join(f"<td>{format_value(row.get(column))}</td>" for column in available) + "</tr>"
                for row in case_summary.to_dict("records")
            )
            + "</tbody></table>"
        )
    content = (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        "<title>Evolution comparison across initial probabilities</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:24px;color:#222;}"
        "table{border-collapse:collapse;width:100%;margin:12px 0 20px;}"
        "th,td{padding:7px 10px;border:1px solid #ddd;text-align:left;}"
        "details{margin:14px 0;border:1px solid #ddd;padding:10px;}"
        "summary{cursor:pointer;overflow-wrap:anywhere;}img{display:block;max-width:100%;"
        "margin:12px 0;}@media(max-width:900px){table{font-size:12px;overflow:auto;display:block;}}"
        "</style>"
        "<h1>Evolution from original benefit-based, uniform and random initializations</h1>"
        f"<p>Local-game experiments under a fixed reference. All initializations use the same payoff matrix, detection rates and cumulative attack time; "
        f"random_0 is shown only as an illustrative trial, while statistics use all random trials. Only nodes with reach &gt; 1e-9 are plotted, "
        f"with probability-difference threshold {probability_tol:.6g}. q is an induced state; no ESS claim is made.</p>"
        f"<p>The reference comes from the unpruned 200-generation run with the original benefit-based initialization; with these inputs fixed, each local game is replayed for {max_iterations} generations. "
        "Delta is the maximum componentwise probability difference across the original, uniform and all random trials; numerical values are in node_comparison.csv. "
        "When the Scope column exists, Case summary prioritizes nodes with reference reach &gt; 1e-9; "
        "all nodes and random trials remain in the CSV files.</p>"
        "<p>Candidate stability requires normalized residuals and invasion gains both &lt; 1e-6, and probability span over the final 50 steps &lt; 1e-4. "
        "Results that do not satisfy these conditions are inconclusive_not_converged. "
        "The three rows show original, uniform and random_0; the two columns show attacker distribution p and three-state defense distribution q. "
        "The three q states are Not detected, Detected, no response, and Detected, response; "
        "the conditional two-strategy response distribution r remains in the CSV diagnostics and is not presented as three-state defense evolution.</p>"
        '<p>Complete data: <a href="node_diagnostics.csv">node_diagnostics.csv</a>, '
        '<a href="node_comparison.csv">node_comparison.csv</a>、'
        '<a href="case_summary.csv">case_summary.csv</a>、'
        '<a href="evolution_history.csv">evolution_history.csv</a>。</p>'
        f"{summary_table}{''.join(case_sections)}</html>"
    )
    report_path.write_text(content, encoding="utf-8")
    return report_path


def plot(input_excel=None, case_id=4):
    set_plot_style(legend_fontsize=10)
    input_excel = get_input_excel(input_excel)
    output_dir = get_feature_output_dir("pruning_validation", input_excel)
    case_directory = output_dir / input_excel.stem
    path = case_directory / "case_summary.csv"
    if not path.is_file():
        path = output_dir / CSV_FILES["pruning_validation"]["case_summary"]
    data = pd.read_csv(path)
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(data["Case_ID"], data["Loss_Pruned"], marker="o", label="Pruned")
    ax.plot(data["Case_ID"], data["Loss_NoPrune"], marker="x", label="No prune")
    ax.set_xlabel("Case_ID")
    ax.set_ylabel("Defense loss")
    ax.set_title("Pruned versus unpruned defense loss")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    count_axis = ax.twinx()
    count_axis.bar(
        data["Case_ID"], data["Pruned_Node_Count"],
        color="gray", alpha=0.15,
    )
    count_axis.set_ylabel("Pruned node count")
    count_axis.set_ylim(0, max(1, data["Pruned_Node_Count"].max()) * 1.3)
    for current_case_id, node_count in zip(data["Case_ID"], data["Pruned_Node_Count"]):
        count_axis.text(current_case_id, node_count, str(node_count), ha="center", va="bottom")
    normalize_figure_typography(fig)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_comparison.png", dpi=DPI_SAVE)
    fig.savefig(output_dir / "loss_comparison.svg")
    case_directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(case_directory / "loss_comparison.png", dpi=DPI_SAVE)
    fig.savefig(case_directory / "loss_comparison.svg")
    plt.close(fig)
    case_ids = (case_id, 11) if case_id == 4 else (case_id,)
    for current_case_id in case_ids:
        plot_case_evolution(input_excel, current_case_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot pruning validation evolutionary histories from CSV")
    parser.add_argument("--input-excel", type=Path, default=None)
    parser.add_argument("--case-id", type=int, nargs="+", default=[4, 11])
    parser.add_argument("--evolution-only", action="store_true")
    parser.add_argument("--include-unreachable", action="store_true")
    arguments = parser.parse_args()
    if arguments.evolution_only or arguments.include_unreachable:
        for case_id in arguments.case_id:
            plot_case_evolution(arguments.input_excel, case_id, arguments.include_unreachable)
    else:
        plot(arguments.input_excel, arguments.case_id[0])
        for case_id in arguments.case_id[1:]:
            if case_id != 11 or arguments.case_id[0] != 4:
                plot_case_evolution(arguments.input_excel, case_id)
