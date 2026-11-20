"""Evaluate three fixed portfolios resolved from the current upstream snapshot."""

import argparse
from dataclasses import asdict
import json
import warnings

import numpy as np
import pandas as pd

from common.attack_graph_data import build_attack_graph_tables
from common.case_config import get_input_excel
from common.defense_config import create_selected_configuration
from common.implementation import IMPLEMENTATION_VERSION, MODEL_PROTOCOL_VERSION
from common.model_options import ModelOptions
from common.parsers import (
    LEGACY_FIREWALL_POSITIONS, LEGACY_IDS_POSITIONS, component_vulnerability_names,
    position_by_label, vulnerability_patch_labels,
)
from common.problem_factory import load_problem_context
from common.simulation import compute_total_expected_values, run_simulation
from features.pareto_attack_paths.inputs import (
    assert_unchanged, decision_vector, file_hash, read_snapshot,
)
from features.pareto_attack_paths.path_analysis import analyze_paths


def _node_id(kind, *names):
    return json.dumps([kind, *names], ensure_ascii=False, separators=(",", ":"))


def _annotations(panel, selected, data):
    rows = []
    selected = set(selected)
    patch_mapping = vulnerability_patch_labels(data["Vulnerability"], data["Patch vulnerability"])
    distribution = component_vulnerability_names(data["Vulnerability"], data["Vulnerabilities distribution"])
    for component, vulnerabilities in distribution.items():
        for vulnerability in vulnerabilities:
            label = patch_mapping[vulnerability]
            if label in selected:
                rows.append({"Panel": panel, "Node_ID": _node_id("vulnerability", component, vulnerability),
                             "Annotation_Type": "patch", "Text": "(patch)", "Defense_Label": label,
                             "Scope": "vulnerability"})
    for sheet, legacy, kind, text in (
        ("Set firewall", LEGACY_FIREWALL_POSITIONS, "firewall", "Firewall"),
        ("Set IDS", LEGACY_IDS_POSITIONS, "ids", "IDS"),
    ):
        for label, component in position_by_label(data[sheet], legacy).items():
            if label in selected:
                network = component == "network"
                rows.append({"Panel": panel, "Node_ID": _node_id("component", "S" if network else component),
                             "Annotation_Type": kind, "Text": text + (" (network)" if network else ""),
                             "Defense_Label": label, "Scope": "network" if network else "component"})
    for label in sorted(selected.intersection({"RP1", "RP2", "RP3"})):
        rows.append({"Panel": panel, "Node_ID": _node_id("component", "S"),
                     "Annotation_Type": "response", "Text": f"Response ({label})", "Defense_Label": label,
                     "Scope": "network"})
    annotated = {row["Defense_Label"] for row in rows}
    if selected - annotated:
        raise ValueError(f"Selected defenses have no graph annotation mapping: {sorted(selected - annotated)}")
    return rows


def solve(input_excel=None, attack_path_config=None):
    """Return self-contained tables; no optimizer, file export, or drawing runs here."""
    input_excel = get_input_excel(input_excel)
    input_fingerprint = {"Source_File": str(input_excel), "SHA256": file_hash(input_excel)}
    options, points, settings, source_tables, fingerprints = read_snapshot(input_excel, attack_path_config)
    defense_items, data, graph = load_problem_context(input_excel, allow_no_attack=True)
    model_options = ModelOptions()
    nodes, edges = build_attack_graph_tables(graph.nodes, graph.adj_matrix, data["Vulnerability"],
                                            data["Vulnerabilities distribution"])
    portfolios = [("a", {"Selection_Type": "undefended", "Solution_ID": 0, "Total_Cost": 0.0,
                           "P_Response": 0.0, "labels": []}, np.zeros(len(defense_items), dtype=bool))]
    for panel, point in zip(("b", "c"), points):
        portfolios.append((panel, point, decision_vector(point, defense_items)))

    scenario_rows, decision_rows, annotation_rows = [], [], []
    steps, candidates, comparisons = [], [], []
    for panel, point, vector in portfolios:
        configuration = create_selected_configuration(vector, defense_items, data)
        if set(configuration["selected"]) != set(point["labels"]):
            raise ValueError(f"{point['Selection_Type']}: response normalization changed the source decision")
        has_response = any(label in configuration["selected"] for label in ("RP1", "RP2", "RP3"))
        results = run_simulation(
            state_graph=graph.state_paths, vuln_data=configuration["vuln_data_sim"],
            component_vuln_dist=data["Vulnerabilities distribution"],
            ids_config=configuration["ids_config"], firewall_config=configuration["firewall_config"],
            patch_config=configuration["patch_config"], has_response=has_response,
            p_response=point["P_Response"], iterations=settings["Game_Iterations"],
            learning_rate=settings["Game_Learning_Rate"], prune_tol=None, model_options=model_options,
        )
        payoff, loss, component_weights = compute_total_expected_values(results)
        analysis = analyze_paths(results, graph.attack_paths, component_weights)
        selection_type, solution_id = point["Selection_Type"], point["Solution_ID"]
        title = "Undefended" if panel == "a" else f"PF{solution_id} ({selection_type.replace('_', ' ')})"
        scenario_rows.append({"Panel": panel, "Selection_Type": selection_type, "Solution_ID": solution_id,
                              "Display_Title": title, "Total_Cost": point["Total_Cost"],
                              "Selected_Defenses": " | ".join(point["labels"]),
                              "P_Response": point["P_Response"], **settings, **analysis["summary"],
                              "Total_Attacker_Payoff": payoff, "Total_Defender_Loss": loss,
                              "Evaluation_Context": "current_model_reevaluation",
                              "Source_Tail_Stable": point.get("Tail_Stable", "not_applicable")})
        for (label, cost), enabled in zip(defense_items, vector):
            decision_rows.append({"Panel": panel, "Selection_Type": selection_type, "Solution_ID": solution_id,
                                  "Defense_Label": label, "Selected": int(enabled), "Defense_Cost": cost,
                                  "P_Response": point["P_Response"]})
        for key, destination in (("path_steps", steps), ("candidate_paths", candidates)):
            frame = analysis[key].copy()
            frame.insert(0, "Panel", panel)
            frame.insert(1, "Selection_Type", selection_type)
            frame.insert(2, "Solution_ID", solution_id)
            destination.append(frame)
        annotation_rows.extend(_annotations(panel, configuration["selected"], data))
        if panel != "a":
            for field, value in (("Resimulated_Defender_Loss", loss), ("Resimulated_Attacker_Payoff", payoff)):
                if field in point:
                    original = float(point[field])
                    comparisons.append({"Panel": panel, "Selection_Type": selection_type, "Solution_ID": solution_id,
                                        "Metric": field, "Node_Path": "", "Source_Value": original,
                                        "Current_Value": value, "Difference": value - original,
                                        "Source_Definition_Status": "source_protocol_unverified"})
            risk = source_tables.get("node_risk", pd.DataFrame())
            if {"Selection_Type", "Solution_ID", "Node_Path", "Compromise_Probability"} <= set(risk):
                linked = risk.loc[risk.Selection_Type.eq(selection_type)]
                if not linked.empty and not pd.to_numeric(linked.Solution_ID).eq(solution_id).all():
                    raise ValueError(f"{selection_type}: node_risk has a conflicting Solution_ID")
                for row in linked.itertuples(index=False):
                    if row.Node_Path in component_weights:
                        current = float(component_weights[row.Node_Path])
                        original = float(row.Compromise_Probability)
                        comparisons.append({"Panel": panel, "Selection_Type": selection_type, "Solution_ID": solution_id,
                                            "Metric": "Component_Selection_Probability", "Node_Path": row.Node_Path,
                                            "Source_Value": original, "Current_Value": current,
                                            "Difference": current - original,
                                            "Source_Definition_Status": "source_protocol_unverified_not_success_probability"})
        print(f"[{panel}] {title}: {analysis['summary']['Component_Path'] or analysis['summary']['Status']}; "
              f"successful vulnerability path probability={analysis['summary']['Path_Probability']:.9g}; loss={loss:.9g}")
    assert_unchanged([input_fingerprint, *fingerprints])
    metadata = {
        "Input_Excel": str(input_excel), "Input_SHA256": input_fingerprint["SHA256"],
        "Source_Directory": str(options["source_dir"]),
        "Source_Files": json.dumps(fingerprints, ensure_ascii=False),
        "Source_Selections": json.dumps([{key: point[key] for key in ("Selection_Type", "Solution_ID")}
                                          for point in points]),
        "Source_Protocol_Status": "unverified: legacy selected_points has no complete protocol/input fingerprint",
        "Implementation_Version": IMPLEMENTATION_VERSION, "Model_Protocol_Version": MODEL_PROTOCOL_VERSION,
        "Model_Options": json.dumps(asdict(model_options), sort_keys=True),
        "Model_Source_SHA256": json.dumps({name: file_hash(__import__(f"common.{name}", fromlist=["__file__"]).__file__)
                                           for name in ("simulation", "models", "game_theory")}, sort_keys=True),
        "Evaluation_Context": "current_model_reevaluation", "Pruning": "disabled_per_run",
        "Path_Probability_Definition": "product(final_action_probability * effective_success_probability(final_response_probability))",
        "Component_Weight_Definition": "product(sum(final_action_probability for target_component)); success excluded",
        "Tie_Break": "lexicographic component path, then vulnerability path",
        **settings,
    }
    comparison = pd.DataFrame(comparisons, columns=["Panel", "Selection_Type", "Solution_ID", "Metric",
                                                  "Node_Path", "Source_Value", "Current_Value", "Difference",
                                                  "Source_Definition_Status"])
    losses = comparison.loc[comparison.Metric.eq("Resimulated_Defender_Loss")] if not comparison.empty else comparison
    if not losses.empty and not np.allclose(losses.Difference, 0, rtol=0, atol=1e-6):
        warnings.warn("Current-model fixed-portfolio losses differ from upstream analysis; see source_comparison.csv. "
                      "Source protocol remains unverified.", UserWarning, stacklevel=2)
    return {"input_excel": str(input_excel), "attack_graph_nodes": nodes, "attack_graph_edges": edges,
            "scenarios": pd.DataFrame(scenario_rows), "decisions": pd.DataFrame(decision_rows),
            "path_steps": pd.concat(steps, ignore_index=True),
            "candidate_paths": pd.concat(candidates, ignore_index=True),
            "defense_annotations": pd.DataFrame(annotation_rows, columns=["Panel", "Node_ID", "Annotation_Type",
                                                                          "Text", "Defense_Label", "Scope"]),
            "source_comparison": comparison,
            "run_metadata": pd.DataFrame([{"Key": key, "Value": str(value)} for key, value in metadata.items()])}


if __name__ == "__main__":
    from common.pipeline import run_solve
    from features.pareto_attack_paths.export_csv import export

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-excel")
    parser.add_argument("--attack-path-config")
    args = parser.parse_args()
    run_solve("pareto_attack_paths", solve, export, input_excel=args.input_excel,
              attack_path_config=args.attack_path_config)
