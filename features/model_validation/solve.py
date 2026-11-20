"""Compare production loss aggregation with independent stopping trajectories."""

from __future__ import annotations

import argparse
import hashlib
from types import SimpleNamespace

import numpy as np
import pandas as pd

from common.case_config import get_input_excel, load_case_parameters
from common.parsers import parse_firewall_config, parse_ids_config, parse_patch_config
from common.problem_factory import load_problem_context
from common.implementation import IMPLEMENTATION_VERSION, MODEL_PROTOCOL_VERSION
from common.model_options import ModelOptions
from common.models import DoNothingStrategy
from common.simulation import run_simulation
from external.attack_graph import AttackGraph
from features.model_validation.diagnostics import validate_scenario


ABSOLUTE_TOLERANCE = 1e-10
RELATIVE_TOLERANCE = 1e-9
INTEGRATION_ITERATIONS = 40
INTEGRATION_LEARNING_RATE = 0.2


def _fixed_record(targets, probabilities, losses, stored_loss=0.0):
    strategies = [
        DoNothingStrategy() if target is None else SimpleNamespace(
            target_component=target,
            vulnerability=SimpleNamespace(name=f"fixed_{target}"),
        )
        for target in targets
    ]
    return {
        "strategies": strategies,
        "attacker_strategy": list(probabilities),
        "defender_loss": stored_loss,
        "attacker_payoff": 0.0,
        "game_result": {"final_defender_losses": np.array(losses, dtype=float)},
    }


def build_fixed_results():
    """Build the hand-calculated choice process with expected loss 52."""
    return {
        "S": _fixed_record(["A", None], [0.8, 0.2], [5.0, 0.0]),
        "S → A": _fixed_record(
            ["B", "C", None], [0.6, 0.3, 0.1], [50.0, 100.0, 0.0], 4.0,
        ),
        "S → A → B": _fixed_record([], [], [], 30.0),
        "S → A → C": _fixed_record([], [], [], 30.0),
    }


def build_integration_results(*, duplicate_prefixes=True):
    """Run the actual local games on a small synthetic S-A-(B/C)-D-T case."""
    nodes = ["S", "A", "B", "C", "D", "T"]
    adjacency = np.zeros((len(nodes), len(nodes)), dtype=int)
    for source, target in (
        ("S", "A"), ("A", "B"), ("A", "C"),
        ("B", "D"), ("C", "D"), ("D", "T"),
    ):
        adjacency[nodes.index(source), nodes.index(target)] = 1
    graph = AttackGraph(nodes, adjacency)
    graph.build_state_transition_graph()
    prefixes = graph.state_paths
    if not duplicate_prefixes:
        prefixes = list(dict.fromkeys(prefixes))
    vulnerability_data = {
        "Vulnerability": ["a1", "a2", "b1", "c1", "d1"],
        "Attack Success Probability": [0.5, 0.7, 0.6, 0.4, 0.8],
        "Attack time (h)": [1.0, 2.0, 1.0, 3.0, 1.0],
        "Attack cost (monetary)": [0.2] * 5,
        "Attack Benefit (monetary)": [2.0, 3.0, 2.0, 4.0, 3.0],
        "Attack benefit (destruction)": [0.0] * 5,
        "Weight_ABM": [1.0] * 5,
        "Weight_ABD": [0.0] * 5,
        "Defend loss (monetary)": [-10.0, -20.0, -40.0, -60.0, -30.0],
        "Defend loss (destruction)": [0.0] * 5,
        "Weight_DLM": [1.0] * 5,
        "Weight_DLD": [0.0] * 5,
        "Defense Success Probability after detecting (0)": [0.0] * 5,
    }
    return run_simulation(
        state_graph=prefixes,
        vuln_data=vulnerability_data,
        component_vuln_dist={
            "Component": ["S", "A", "B", "C", "D", "T"],
            "Vulnerability": [0, "a1,a2", "b1", "c1", "d1", 0],
        },
        ids_config={"network": {"eff": 0.25, "cycle": 2.0}},
        firewall_config={},
        patch_config={},
        iterations=INTEGRATION_ITERATIONS,
        learning_rate=INTEGRATION_LEARNING_RATE,
        has_response=False,
        p_response=0.0,
        prune_tol=None,
        model_options=ModelOptions(),
    )


def build_case_results(input_excel=None, *, parameters=None):
    """Run the production aggregation for the explicitly selected workbook case."""
    input_excel = get_input_excel(input_excel)
    if parameters is None:
        parameters = load_case_parameters(
            input_excel,
            names=("Game_Iterations", "Game_Learning_Rate"),
        )
    _, all_data, graph = load_problem_context(input_excel)
    return run_simulation(
        state_graph=graph.state_paths,
        vuln_data=all_data["Vulnerability"],
        component_vuln_dist=all_data["Vulnerabilities distribution"],
        ids_config=parse_ids_config(all_data["Set IDS"]),
        firewall_config=parse_firewall_config(all_data["Set firewall"]),
        patch_config=parse_patch_config(
            all_data["Patch vulnerability"], all_data["Vulnerability"],
        ),
        iterations=parameters["Game_Iterations"],
        learning_rate=parameters["Game_Learning_Rate"],
        prune_tol=None,
    )


def solve(input_excel=None):
    """Validate fixed examples and the data from one explicitly selected workbook."""
    input_excel = get_input_excel(input_excel)
    if not input_excel.is_file():
        raise FileNotFoundError(f"Input workbook does not exist: {input_excel}")
    tables = {name: [] for name in (
        "action_probabilities", "prefix_contributions", "terminal_trajectories",
        "validation_summary",
    )}
    case_parameters = load_case_parameters(
        input_excel,
        names=("Game_Iterations", "Game_Learning_Rate"),
    )
    scenarios = [
        ("fixed_example", build_fixed_results, 52.0, "actions"),
        ("production_integration", build_integration_results, None, "actions"),
        ("workbook_case", lambda: build_case_results(input_excel, parameters=case_parameters), None, "components"),
    ]
    for scenario_id, builder, reference, trajectory_mode in scenarios:
        try:
            scenario = validate_scenario(
                scenario_id, builder(), reference,
                atol=ABSOLUTE_TOLERANCE, rtol=RELATIVE_TOLERANCE,
                trajectory_mode=trajectory_mode,
            )
        except ValueError as error:
            tables["validation_summary"].append({
                "Scenario_ID": scenario_id,
                "Check": "scenario_input",
                "Actual": np.nan,
                "Expected": np.nan,
                "Abs_Error": np.nan,
                "Tolerance": ABSOLUTE_TOLERANCE,
                "Status": "FAIL",
                "Detail": str(error),
            })
            continue
        for name in tables:
            tables[name].extend(scenario[name])
    frames = {name: pd.DataFrame(rows) for name, rows in tables.items()}
    case_id = input_excel.stem
    source_by_scenario = {
        "fixed_example": "fixed synthetic example",
        "production_integration": "production API synthetic integration",
        "workbook_case": "explicit workbook case",
    }
    for name in ("action_probabilities", "prefix_contributions", "terminal_trajectories", "validation_summary"):
        frames[name]["Case_ID"] = case_id
        frames[name]["Input_Excel"] = str(input_excel)
        if "Scenario_ID" in frames[name].columns:
            frames[name]["Scenario_Source"] = frames[name]["Scenario_ID"].map(source_by_scenario)
    input_sha256 = hashlib.sha256(input_excel.read_bytes()).hexdigest()
    passed = bool(
        not frames["validation_summary"].empty
        and frames["validation_summary"]["Status"].eq("PASS").all()
    )
    metadata = {
        "Input_Excel": str(input_excel),
        "Input_Role": "explicit workbook case data and Strategy parameters are used for workbook_case; synthetic scenarios are independent",
        "Scenario_Source": "fixed/integration synthetic examples plus explicit workbook case",
        "Input_SHA256": input_sha256,
        "Schema_Version": 2,
        "Probability_Semantics": "component choice history; not successful compromise reach",
        "Local_Loss_Semantics": "expected local loss including local success and defense mixtures",
        "Expected_Loss_Formula": "sum_h,c rho(h) * sum_i:target(i)=c p(i|h) * ell(h,i)",
        "Terminal_Formula": "sum_omega P(omega) * sum_t ell(h_t,i_t)",
        "Stop_Reasons": "DoNothing; candidate history ends without further actions",
        "Failure_Absorption_Implemented": "false",
        "Prune_Tolerance": "None (disabled explicitly for exact reconciliation)",
        "Integration_Game_Iterations": INTEGRATION_ITERATIONS,
        "Integration_Game_Learning_Rate": INTEGRATION_LEARNING_RATE,
        "Integration_Has_Response": "false",
        "Integration_Model_Options": str(ModelOptions()),
        "Case_ID": input_excel.stem,
        "Case_Game_Iterations": case_parameters["Game_Iterations"],
        "Case_Game_Learning_Rate": case_parameters["Game_Learning_Rate"],
        "Case_Source": "explicit workbook graph, vulnerabilities, defenses, and Strategy parameters",
        "Case_Configuration": "workbook initial IDS/firewall/patch configuration and RP0; production response defaults",
        "Case_Trajectory_Mode": "components; conditional local loss=sum(p*ell)/sum(p)",
        "Absolute_Tolerance": ABSOLUTE_TOLERANCE,
        "Relative_Tolerance": RELATIVE_TOLERANCE,
        "Max_Trajectories_Per_Scenario": 10000,
        "Implementation_Version": IMPLEMENTATION_VERSION,
        "Model_Protocol": MODEL_PROTOCOL_VERSION,
        "Passed": str(passed).lower(),
    }
    return {
        "schema_version": 2,
        "input_excel": str(input_excel), "passed": passed, **frames,
        "run_metadata": pd.DataFrame(metadata.items(), columns=["Key", "Value"]),
    }


if __name__ == "__main__":
    from common.pipeline import run_solve
    from features.model_validation.export_csv import export

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-excel", default=None)
    arguments = parser.parse_args()
    run_solve("model_validation", solve, export, input_excel=arguments.input_excel)
