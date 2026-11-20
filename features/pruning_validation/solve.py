"""Compare pruned and unpruned simulation on deterministic decision vectors."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd

from common.case_config import get_input_excel, load_case_parameters
from common.defense_config import (
    create_selected_configuration,
    repair_response_selection,
)
from common.problem_factory import load_problem_context
from common.simulation import PRUNE_TOL_DEFAULT, compute_total_expected_values, run_simulation
from features.pruning_validation.stability import diagnose_distribution


PRUNE_TOL = PRUNE_TOL_DEFAULT
RNG_SEED = 42
BASE_CASE_COUNT = 10
NETWORK_IDS_CASE_ID = BASE_CASE_COUNT + 1
ABS_TOL = 1e-6
REL_TOL = 1e-9
EVOLUTION_CASE_ID = 4
EVOLUTION_CASE_IDS = (4, NETWORK_IDS_CASE_ID)
DEFENDER_STATES = ("Not detected", "Detected, no response", "Detected, response")
RESPONSE_POLICIES = ("No response after detection", "Respond after detection")


def _full_defense_vector(defense_items, all_data):
    categories = all_data["Allocation"].get("Category")
    if categories is not None:
        if len(categories) != len(defense_items):
            raise ValueError("Allocation Category/Label lengths differ")
        resource_labels = {
            label
            for (label, _), category in zip(defense_items, categories)
            if str(category).strip().lower() in ("prevention", "detection")
        }
    else:
        resource_labels = {
            label
            for sheet in ("Patch vulnerability", "Set firewall", "Set IDS")
            for label in all_data[sheet]["Label"]
        }
    response_labels = [
        label for label, _ in defense_items if label in ("RP1", "RP2", "RP3")
    ]
    if not response_labels:
        raise ValueError("Full-defense validation requires a response measure")
    highest_response = max(response_labels, key=lambda label: int(label[-1]))
    return np.array([
        label in resource_labels or label == highest_response
        for label, _ in defense_items
    ], dtype=bool)


def _decision_vectors(item_count: int, defense_items, all_data):
    rng = np.random.default_rng(RNG_SEED)
    vectors = [np.zeros(item_count, dtype=bool)]
    vectors.extend(
        (rng.random(item_count) < 0.3).astype(bool)
        for _ in range(BASE_CASE_COUNT - 1)
    )
    vectors.append(_full_defense_vector(defense_items, all_data))
    probe_labels = {"V1", "V2", "F0", "IDS0", "RP3"}
    available_labels = {label for label, _ in defense_items}
    missing_labels = probe_labels - available_labels
    if missing_labels:
        raise ValueError(
            f"Network IDS probe case requires missing labels: {sorted(missing_labels)}"
        )
    vectors.append(
        np.array(
            [label in probe_labels for label, _ in defense_items],
            dtype=bool,
        )
    )
    for vector in vectors:
        repair_response_selection(vector, defense_items)
    return vectors


def _simulate(configuration, graph, all_data, prune_tol, case_parameters=None):
    parameters = (
        load_case_parameters(names=("Game_Iterations", "Game_Learning_Rate"))
        if case_parameters is None else case_parameters
    )
    p_response = _initial_response_probability(configuration)
    started = perf_counter()
    results = run_simulation(
        state_graph=graph.state_paths,
        vuln_data=configuration["vuln_data_sim"],
        component_vuln_dist=all_data["Vulnerabilities distribution"],
        ids_config=configuration["ids_config"],
        firewall_config=configuration["firewall_config"],
        patch_config=configuration["patch_config"],
        iterations=parameters["Game_Iterations"],
        learning_rate=parameters["Game_Learning_Rate"],
        p_response=p_response,
        prune_tol=prune_tol,
    )
    elapsed = perf_counter() - started
    payoff, loss, _ = compute_total_expected_values(results)
    return results, float(payoff), float(loss), elapsed


def _initial_response_probability(configuration):
    return 0.5 if any(label in ("RP1", "RP2", "RP3") for label in configuration["selected"]) else 0.0


def _validate_history(history, width, node, player):
    probabilities = np.asarray(history, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != width:
        raise ValueError(f"{player} strategy/history dimensions differ at {node}")
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(probabilities.sum(axis=1), 1.0)
    ):
        raise ValueError(f"Invalid {player} evolutionary probabilities at {node}")
    return probabilities


def _collect_evolution(case_id, mode, results):
    _, _, availability = compute_total_expected_values(results)
    history_rows = []
    probability_rows = []
    node_rows = []
    defender_history_rows = []
    policy_history_rows = []
    defender_probability_rows = []
    for node, result in results.items():
        strategies = result["strategies"]
        history = result["evolution_history"]
        if not strategies or not history:
            continue
        game_result = result["game_result"]
        probabilities = _validate_history(history, len(strategies), node, "attacker")
        defender_probabilities = _validate_history(game_result["q_history"], 3, node, "defender state")
        policy_probabilities = _validate_history(game_result["response_policy_history"], 2, node, "defender policy")
        if len(probabilities) != len(defender_probabilities) or len(probabilities) != len(policy_probabilities):
            raise ValueError(f"Attacker/defender history generations differ at {node}")
        if not np.allclose(probabilities[-1], result["attacker_strategy"]) or not np.allclose(defender_probabilities[-1], result["defender_strategy"]):
            raise ValueError(f"Invalid evolutionary probabilities at {node}")
        metadata = {
            "Case_ID": case_id,
            "Mode": mode,
            "Step": node.count(" → ") + 1,
            "Node_Path": node,
        }
        node_rows.append({
            **metadata,
            "Node_Reach_Probability": float(availability.get(node, 0.0)),
            "Attack_Time_h": float(result["attack_time"]),
            "Strategy_Count": len(strategies),
            "Iterations": len(history) - 1,
            "Initial_Conditional_Response_Probability": float(policy_probabilities[0, 1]),
            "Final_Conditional_Response_Probability": float(policy_probabilities[-1, 1]),
            "Not_Detected_Probability": float(defender_probabilities[-1, 0]),
            "Detected_No_Response_Probability": float(defender_probabilities[-1, 1]),
            "Response_Probability": float(defender_probabilities[-1, 2]),
            "Effective_Strategies": float(1.0 / np.sum(probabilities[-1] ** 2)),
            "Entropy": float(-np.sum(probabilities[-1][probabilities[-1] > 0] * np.log(probabilities[-1][probabilities[-1] > 0]))),
            "Max_Probability": float(np.max(probabilities[-1])),
            "Min_Positive_Probability": float(np.min(probabilities[-1][probabilities[-1] > 0])),
            "Payoff_Scale": float(game_result["payoff_scale"]),
            "Defender_Effective_Policies": float(1.0 / np.sum(policy_probabilities[-1] ** 2)),
            "Defender_Max_Policy_Probability": float(np.max(policy_probabilities[-1])),
            "Node_Attacker_Payoff": float(game_result["attacker_payoff"]),
            "Node_Defender_Loss": float(game_result["defender_loss"]),
            "Diagnostic_Scope": "attacker_at_final_defender_policy_no_coupled_ESS_claim",
        })
        attacker = np.array([
            [strategy.get_utility_not_detected(), strategy.get_utility_detected_no_response(), strategy.get_utility_detected_with_response()]
            if hasattr(strategy, "vulnerability") else [0.0, 0.0, 0.0]
            for strategy in strategies
        ])
        detection = np.array([strategy.detected_prob for strategy in strategies])
        response = bool(game_result["has_response"])
        node_rows[-1].update(diagnose_distribution(
            probabilities[-1], attacker, detection, response,
            p_response=float(game_result["final_response_probability"]),
        ))
        for player, player_probabilities, names, rows in (
            ("state", defender_probabilities, DEFENDER_STATES, defender_history_rows),
            ("conditional_policy", policy_probabilities, RESPONSE_POLICIES, policy_history_rows),
        ):
            for strategy_id, name in enumerate(names):
                strategy_metadata = {
                    **metadata,
                    "Player": "Defender",
                    "Probability_Type": player,
                    "Strategy_ID": strategy_id,
                    "Strategy": name,
                }
                defender_probability_rows.append({
                    **strategy_metadata,
                    "Initial_Probability": float(player_probabilities[0, strategy_id]),
                    "Final_Probability": float(player_probabilities[-1, strategy_id]),
                })
                for generation, probability in enumerate(player_probabilities[:, strategy_id]):
                    rows.append({
                        **strategy_metadata,
                        "Generation": generation,
                        "Probability": float(probability),
                    })
        for strategy_id, strategy in enumerate(strategies):
            vulnerability = getattr(strategy, "vulnerability", None)
            component = getattr(strategy, "target_component", "")
            label = (
                f"Attack {component} ({vulnerability.name})"
                if vulnerability is not None
                else "Do Nothing"
            )
            strategy_metadata = {
                **metadata,
                "Strategy_ID": strategy_id,
                "Strategy": label,
            }
            probability_rows.append({
                **strategy_metadata,
                "Component": component,
                "Vulnerability": vulnerability.name if vulnerability else "",
                "Initial_Probability": float(probabilities[0, strategy_id]),
                "Final_Probability": float(probabilities[-1, strategy_id]),
                "Attack_Success_Probability": float(
                    strategy.compute_success_probability(
                        result["game_result"]["final_response_probability"],
                        has_response=result["game_result"]["has_response"],
                    )
                ) if vulnerability else 0.0,
                "Passive_Success_Probability": float(strategy.passive_success_prob) if vulnerability else 0.0,
                "Active_Success_Probability": float(strategy.active_success_prob) if vulnerability else 0.0,
                "Detected_Probability": float(strategy.detected_prob),
                "Expected_Utility": float(result["game_result"]["final_attacker_payoffs"][strategy_id]),
                "Expected_Defender_Loss": float(result["game_result"]["final_defender_losses"][strategy_id]),
            })
            for generation, probability in enumerate(probabilities[:, strategy_id]):
                history_rows.append({
                    **strategy_metadata,
                    "Generation": generation,
                    "Probability": float(probability),
                })
    return {
        "game_strategy_history": history_rows,
        "strategy_probabilities": probability_rows,
        "evolution_nodes": node_rows,
        "defender_strategy_history": defender_history_rows,
        "response_policy_history": policy_history_rows,
        "defender_strategy_probabilities": defender_probability_rows,
    }


def solve(input_excel=None, evolution_case_id=EVOLUTION_CASE_IDS):
    """Compare the original cases, full defense, and the network IDS probe."""
    input_excel = get_input_excel(input_excel)
    parameters = load_case_parameters(input_excel, names=("Game_Iterations", "Game_Learning_Rate"))
    items, all_data, graph = load_problem_context(input_excel)
    vectors = _decision_vectors(len(items), items, all_data)
    evolution_case_ids = () if evolution_case_id is None else (
        (evolution_case_id,) if isinstance(evolution_case_id, (int, np.integer)) else tuple(evolution_case_id)
    )
    if any(not isinstance(case_id, (int, np.integer)) or not 0 <= case_id < len(vectors) for case_id in evolution_case_ids):
        raise ValueError(f"evolution_case_id values must be in 0..{len(vectors) - 1}")
    summary_rows = []
    detail_rows = []
    timing_rows = []
    failures = []
    evolution_rows = {name: [] for name in (
        "game_strategy_history", "strategy_probabilities", "evolution_nodes",
        "defender_strategy_history", "response_policy_history", "defender_strategy_probabilities",
    )}

    for case_id, vector in enumerate(vectors):
        configuration = create_selected_configuration(vector, items, all_data)
        pruned, payoff_pruned, loss_pruned, time_pruned = _simulate(
            configuration, graph, all_data, PRUNE_TOL, parameters
        )
        unpruned, payoff_none, loss_none, time_none = _simulate(
            configuration, graph, all_data, None, parameters
        )
        if case_id in evolution_case_ids:
            for mode, results in (("pruned", pruned), ("unpruned", unpruned)):
                for name, rows in _collect_evolution(case_id, mode, results).items():
                    evolution_rows[name].extend(rows)
        abs_diff = abs(loss_pruned - loss_none)
        relative_diff = abs_diff / abs(loss_none) if loss_none != 0 else (0.0 if abs_diff == 0 else float("inf"))
        payoff_abs_diff = abs(payoff_pruned - payoff_none)
        payoff_relative_diff = payoff_abs_diff / abs(payoff_none) if payoff_none != 0 else (0.0 if payoff_abs_diff == 0 else float("inf"))
        loss_matches = abs_diff <= ABS_TOL or relative_diff <= REL_TOL
        payoff_matches = (
            payoff_abs_diff <= ABS_TOL or payoff_relative_diff <= REL_TOL
        )
        status = (
            "ok"
            if loss_matches and payoff_matches
            else "mismatch"
        )
        if status != "ok":
            failures.append(case_id)
        summary_rows.append({
            "Case_ID": case_id,
            "Case_Name": (
                "all_prevention_detection_max_response"
                if case_id == BASE_CASE_COUNT
                else "network_ids_probe"
                if case_id == NETWORK_IDS_CASE_ID
                else "no_defense" if case_id == 0 else f"random_{case_id}"
            ),
            "Selected_Count": int(vector.sum()),
            "Initial_Conditional_Response_Probability": _initial_response_probability(configuration),
            "Node_Count_Pruned": len(pruned),
            "Node_Count_NoPrune": len(unpruned),
            "Pruned_Node_Count": len(unpruned) - len(pruned),
            "Loss_Pruned": loss_pruned,
            "Loss_NoPrune": loss_none,
            "Abs_Diff": abs_diff,
            "Relative_Diff": relative_diff,
            "Attacker_Payoff_Pruned": payoff_pruned,
            "Attacker_Payoff_NoPrune": payoff_none,
            "Payoff_Abs_Diff": payoff_abs_diff,
            "Payoff_Relative_Diff": payoff_relative_diff,
            "Status": status,
        })
        timing_rows.append({
            "Case_ID": case_id,
            "Wall_Time_Pruned_s": time_pruned,
            "Wall_Time_NoPrune_s": time_none,
        })
        for (label, cost), selected in zip(items, vector):
            detail_rows.append({
                "Case_ID": case_id,
                "Defense_Label": label,
                "Cost": float(cost),
                "Is_Selected": int(bool(selected)),
            })

    return {
        "case_summary": pd.DataFrame(summary_rows),
        "case_details": pd.DataFrame(detail_rows),
        "case_timings": pd.DataFrame(timing_rows),
        "passed": not failures,
        "failures": failures,
        "input_excel": str(input_excel),
        "evolution_case_id": evolution_case_ids[0] if len(evolution_case_ids) == 1 else None,
        "evolution_case_ids": list(evolution_case_ids),
        **{name: pd.DataFrame(rows) for name, rows in evolution_rows.items()},
        "prune_tol": PRUNE_TOL,
    }


if __name__ == "__main__":
    from common.pipeline import run_solve
    from features.pruning_validation.export_csv import export

    result = run_solve("pruning_validation", solve, export)
    print(f"Pruning validation: {'PASS' if result['passed'] else 'FAIL'}; file: {result['input_excel']}")
