"""Run bounded local diagnostics and independent complete-graph E3 replays."""

from __future__ import annotations

import argparse
from importlib import metadata as package_metadata
import math
from pathlib import Path
import platform

import numpy as np

from common.case_config import get_input_excel, load_case_parameters
from common.defense_config import create_selected_configuration
from common.game_theory import (
    DEFENDER_STATE_NAMES, marginal_expected_values, replicator_dynamics,
    response_fitness,
)
from common.implementation import IMPLEMENTATION_VERSION, MODEL_PROTOCOL_VERSION
from common.problem import split_decision_vector
from common.problem_factory import load_problem_context
from common.simulation import PRUNE_TOL_DEFAULT, compute_total_expected_values, resolve_prune_tolerance, run_simulation
from features.equilibrium_validation.diagnostics import diagnose_game, resolve_thresholds
from features.equilibrium_validation.inputs import file_sha256, json_text, load_configuration, load_portfolios


TABLE_NAMES = (
    "game_definition", "equilibrium_diagnostics", "jacobian_eigenvalues",
    "initialization_sensitivity", "strategy_trace", "path_sensitivity", "run_metadata",
)

METHOD_DESCRIPTION = (
    "Synchronous discrete exponential adaptation of attacker p and conditional response r; "
    "detection occurs before execution of the conditional response. Adaptation rounds "
    "are numerical population updates, not sequential observations or Bayesian beliefs. "
    "Induced Nash/ESS refer only to the attacker population game with frozen response. "
    "Joint or sequential equilibrium is not established. Reach is choice-based and "
    "does not validate empirical compromise probability."
)


def _action_label(strategy):
    if hasattr(strategy, "vulnerability"):
        return f"{strategy.target_component}:{strategy.vulnerability.name}"
    return "DoNothing"


def _initializer(kind, seed):
    if kind == "original":
        return None

    def initialize(strategies, default_p):
        if kind == "uniform":
            return np.full(len(default_p), 1.0 / len(default_p))
        import hashlib

        identity = "|".join(_action_label(strategy) for strategy in strategies)
        state_seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")
        return np.random.default_rng(np.random.SeedSequence([int(seed), state_seed])).dirichlet(np.ones(len(default_p)))

    return initialize


def _design(config, iterations, learning_rate, response, base_prune_tol=PRUNE_TOL_DEFAULT):
    designs = [{"initialization": "original", "seed": None, "response": response,
                "learning_rate": learning_rate, "iterations": iterations,
                "design_factor": "baseline", "prune_tol": base_prune_tol}]
    for kind in config["attacker_initializations"]:
        if kind == "original":
            continue
        for seed in config["dirichlet_seeds"] if kind == "dirichlet" else [None]:
            designs.append({**designs[0], "initialization": kind, "seed": seed, "design_factor": "attacker_initialization"})
    for value in config["response_initializations"]:
        if value != "original" and value != response:
            designs.append({**designs[0], "response": float(value), "design_factor": "response_initialization"})
    for multiplier in config["eta_multipliers"]:
        if multiplier != 1:
            designs.append({**designs[0], "learning_rate": learning_rate * multiplier, "design_factor": "learning_rate"})
    for checkpoint in config["iteration_checkpoints"]:
        if checkpoint != iterations:
            designs.append({**designs[0], "iterations": checkpoint, "design_factor": "iteration_cap"})
    if config["prune_tol"] != base_prune_tol:
        designs.append({**designs[0], "design_factor": "pruning_scenario", "prune_tol": config["prune_tol"]})
    unique = []
    seen = set()
    for design in designs:
        design.setdefault("prune_tol", base_prune_tol)
        key = (design["initialization"], design["seed"], design["response"], design["learning_rate"], design["iterations"], design["prune_tol"])
        if key not in seen:
            seen.add(key)
            unique.append(design)
    return unique


def _provenance(input_excel):
    root = Path(__file__).resolve().parents[2]
    sources = ["common/models.py", "common/game_theory.py", "common/simulation.py", "common/defense_config.py",
               "common/config.py", "common/case_config.py", "common/parsers.py", "common/case_validation.py",
               "external/attack_graph.py", "external/excel_processor.py",
               "features/equilibrium_validation/inputs.py", "features/equilibrium_validation/solve.py",
               "features/equilibrium_validation/diagnostics.py", "features/equilibrium_validation/publication.py"]
    dependencies = {}
    for name in ("numpy", "pandas", "networkx", "matplotlib", "pymoo", "openpyxl"):
        try:
            dependencies[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            dependencies[name] = "not_installed"
    return {"input_excel": str(input_excel), "input_sha256": file_sha256(input_excel),
            "implementation_version": IMPLEMENTATION_VERSION, "model_protocol": MODEL_PROTOCOL_VERSION,
            "source_sha256": {name: file_sha256(root / name) for name in sources},
            "dependencies": dependencies, "python": platform.python_version(),
            "units": "case-defined monetary units; no unit conversion performed", "method": METHOD_DESCRIPTION,
            "dirichlet_seed_method": "SeedSequence([configured_seed, first_8_bytes_SHA256_of_ordered_action_labels]); regenerated per local state without shared RNG state",
            "threshold_units": {"map_tol": "probability", "fitness_tol": "payoff-scale normalized per-round fitness drift",
                                "gap_tol": "payoff-scale normalized unilateral invasion advantage", "tail_tol": "probability",
                                "jacobian_step": "simplex probability coordinates", "jacobian_tol": "relative matrix-entry tolerance",
                                "stability_margin": "dimensionless spectral radius", "support_tol": "probability"}}


def _metadata(keys, portfolio, design, config, provenance, status="ok", reason=""):
    return {**keys, "Input_Excel": provenance["input_excel"], "Input_SHA256": provenance["input_sha256"],
            "Model_Protocol": MODEL_PROTOCOL_VERSION, "Implementation_Version": IMPLEMENTATION_VERSION,
            "Decision_JSON": json_text(portfolio["decision"]), "Raw_Decision_JSON": json_text(portfolio["raw_decision"]),
            "Allocation_Labels_JSON": json_text(portfolio["allocation_labels"]),
            "Current_Total_Cost": portfolio.get("normalized_cost"),
            "Expected_Source_Cost": portfolio["expected_cost"], "Expected_Source_Loss": portfolio["expected_loss"],
            "P_Response": design["response"], "Learning_Rate": design["learning_rate"],
            "Iterations": design["iterations"], "Seed": design["seed"], "Initialization": design["initialization"],
            "Source_Status": portfolio["source"]["status"], "Replay_Status": "not_applicable",
            "Config_JSON": json_text(config), "Provenance_JSON": json_text({**provenance, "source": portfolio["source"]}),
            "Experiment_Type": design["experiment_type"], "Design_Factor": design["design_factor"],
            "Status": status, "Reason": reason}


def _discard_partial_rows(tables, keys, node_path=None):
    for name in TABLE_NAMES:
        if name == "run_metadata":
            continue
        tables[name][:] = [row for row in tables[name] if not (
            all(row.get(key) == value for key, value in keys.items())
            and (node_path is None or row.get("Node_Path") == node_path)
        )]


def _history_at(game, name, iteration):
    """Return the requested state, or the terminal state after early stopping."""
    history = game[name]
    return history[min(int(iteration), len(history) - 1)]


def _game_rows(tables, keys, node_path, strategies, game, design, config, checkpoints=None):
    inputs = game["game_inputs"]
    attacker = np.asarray(inputs["attacker_gain_matrix"])
    losses = np.asarray(inputs["defender_loss_matrix"])
    detection = np.asarray(inputs["f"])
    for index, strategy in enumerate(strategies):
        tables["game_definition"].append({
            **keys, "Node_Path": node_path, "Action_ID": index, "Action_Label": _action_label(strategy),
            "Target_Component": getattr(strategy, "target_component", ""),
            "Vulnerability": getattr(getattr(strategy, "vulnerability", None), "name", ""),
            "Detection_Probability": float(detection[index]), "A_Not_Detected": float(attacker[index, 0]),
            "A_Detected_No_Response": float(attacker[index, 1]), "A_Detected_Response": float(attacker[index, 2]),
            "L_Not_Detected": float(losses[index, 0]), "L_Detected_No_Response": float(losses[index, 1]),
            "L_Detected_Response": float(losses[index, 2]), "P_Initial": float(game["p_history"][0][index]),
            "Response_Feasible": game["has_response"], "Payoff_Scale": game["payoff_scale"],
            "Defender_Actions_JSON": json_text(["no_response", "respond_after_detection"] if game["has_response"] else ["no_response"]),
            "Response_Initial_JSON": json_text(np.asarray(game["response_policy_history"][0]).tolist()),
            "Event_Order_JSON": json_text(list(DEFENDER_STATE_NAMES)),
            "Loss_Scale": game["loss_scale"], "Method": METHOD_DESCRIPTION,
            "Previous_Attack_Time": getattr(strategy, "prev_time", None),
            "Strategy_Total_Attack_Time": getattr(strategy, "total_attack_time", None),
            "Passive_Success_Probability": getattr(strategy, "passive_success_prob", None),
            "Active_Success_Probability": getattr(strategy, "active_success_prob", None),
            "Game_Scope": design["experiment_type"],
        })
    final_iteration = len(game["p_history"]) - 1
    if checkpoints is None:
        checkpoints = [final_iteration]
    diagnostics = {}
    for checkpoint in checkpoints:
        effective_checkpoint = min(int(checkpoint), final_iteration)
        outcome = diagnose_game(
            attacker, losses, detection, _history_at(game, "p_history", checkpoint),
            _history_at(game, "response_policy_history", checkpoint),
            learning_rate=design["learning_rate"], has_response=game["has_response"],
            log_p=_history_at(game, "log_p_history", checkpoint),
            log_r=_history_at(game, "log_r_history", checkpoint),
            p_history=game["p_history"][:effective_checkpoint + 1],
            r_history=game["response_policy_history"][:effective_checkpoint + 1],
            thresholds=config["thresholds"],
        )
        diagnostic = {**keys, "Node_Path": node_path, "Checkpoint": checkpoint,
                      "Experiment_Type": design["experiment_type"], **outcome["diagnostics"]}
        tables["equilibrium_diagnostics"].append(diagnostic)
        for eigenvalue in outcome["eigenvalues"]:
            tables["jacobian_eigenvalues"].append({**keys, "Node_Path": node_path, "Checkpoint": checkpoint, **eigenvalue})
        diagnostics[checkpoint] = diagnostic
    for iteration in sorted(set(range(0, final_iteration + 1, config["trace_stride"])) | {final_iteration} | set(checkpoints)):
        current_p = np.asarray(_history_at(game, "p_history", iteration))
        current_r = np.asarray(_history_at(game, "response_policy_history", iteration))
        current_q = np.asarray(_history_at(game, "q_history", iteration))
        payoffs = marginal_expected_values(current_p, attacker, detection, current_r[1], game["has_response"])
        defender_fitness = response_fitness(current_p, losses, detection, game["has_response"])
        residual = diagnostics.get(iteration, {}).get("R_map")
        if residual is None and iteration < final_iteration:
            residual = max(float(np.max(np.abs(np.asarray(game["p_history"][iteration + 1]) - current_p))),
                           float(np.max(np.abs(np.asarray(game["response_policy_history"][iteration + 1]) - current_r))))
        for distribution, probabilities in (("p", current_p), ("r", current_r), ("q", current_q)):
            for action_id, probability in enumerate(probabilities):
                tables["strategy_trace"].append({
                    **keys, "Node_Path": node_path, "Iteration": iteration, "Distribution": distribution,
                    "Action_ID": action_id, "Probability": float(probability),
                    "Action_Label": _action_label(strategies[action_id]) if distribution == "p" else ("no_response", "respond_after_detection")[action_id] if distribution == "r" else DEFENDER_STATE_NAMES[action_id],
                    "Payoff": float(payoffs[action_id]) if distribution == "p" else None,
                    "Fitness": float(payoffs[action_id]) if distribution == "p" else float(defender_fitness[action_id]) if distribution == "r" else None,
                    "Map_Residual": residual,
                })
    return diagnostics


def _reach_status(path, results, availability):
    if path in availability:
        return "zero_probability" if availability[path] == 0 else "observed"
    parts = path.split(" → ")
    parent = " → ".join(parts[:-1])
    if parent not in results:
        return "ancestor_not_expanded"
    strategies = results[parent].get("strategies", [])
    matching = [(strategy, probability) for strategy, probability in zip(strategies, results[parent].get("attacker_strategy", []))
                if getattr(strategy, "target_component", None) == parts[-1]]
    if not matching:
        return "terminal_T_not_simulated" if parts[-1] == "T" else "no_feasible_transition"
    return "pruned_zero_probability" if sum(probability for _, probability in matching) == 0 else "pruned_positive_probability"


def _path_rows(keys, graph, results, availability, baseline, tolerance):
    rows = []
    for path in sorted(set(graph.state_graph.nodes) | {"S"}):
        probability = float(availability[path]) if path in availability else None
        rows.append({**keys, "Path_Type": "node", "Path": path, "Probability": probability,
                     "Status": _reach_status(path, results, availability),
                     "Probability_Definition": "choice-based prefix reach; not empirical compromise probability"})
    complete = []
    for path in graph.attack_paths:
        terminal = " → ".join(path)
        before_terminal = " → ".join(path[:-1])
        key = terminal if terminal in availability or path[-1] != "T" else before_terminal
        probability = float(availability[key]) if key in availability else None
        complete.append({**keys, "Path_Type": "complete_path", "Path": terminal, "Probability": probability,
                         "Status": "terminal_T_fallback" if terminal not in availability and probability is not None else _reach_status(terminal, results, availability),
                         "Probability_Definition": "choice-based path probability with documented terminal-T fallback"})
    probabilities = sorted({row["Probability"] for row in complete if row["Probability"] is not None}, reverse=True)
    best = probabilities[0] if probabilities else None
    complete_coverage = bool(complete) and all(row["Probability"] is not None for row in complete)
    for row in rows + complete:
        probability = row["Probability"]
        is_complete = row["Path_Type"] == "complete_path"
        row["Rank"] = 1 + sum(value > probability + tolerance for value in probabilities) if is_complete and probability is not None else None
        row["Is_Main_Path"] = bool(is_complete and best is not None and probability is not None and best - probability <= tolerance)
        row["Tied"] = bool(is_complete and probability is not None and sum(other["Probability"] is not None and abs(other["Probability"] - probability) <= tolerance for other in complete) > 1)
        row["Ranking_Scope"] = "all_complete_paths" if is_complete and complete_coverage else "observed_complete_paths_only" if is_complete else "not_applicable"
        row["Main_Path_Status"] = "established_for_current_choice_model" if is_complete and complete_coverage else "not_established_incomplete_path_coverage" if is_complete else "not_applicable"
        reference = baseline.get((row["Path_Type"], row["Path"]))
        row["Baseline_Probability"] = reference
        row["Probability_Delta"] = probability - reference if probability is not None and reference is not None else None
    return rows + complete


def _replay_status(portfolio, parameters, total_cost, total_loss, provenance):
    source = portfolio["source"]
    if source["type"] == "explicit":
        return "current_scenario_only"
    metadata = source["metadata"]
    comparisons = []
    for name in ("Game_Iterations", "Game_Learning_Rate"):
        value = metadata.get(name)
        if value is None or value == "":
            comparisons.append("unknown")
        else:
            try:
                comparisons.append("match" if math.isclose(float(value), parameters[name], rel_tol=0, abs_tol=1e-12) else "mismatch")
            except (ValueError, TypeError):
                comparisons.append("mismatch")
    source["historical_parameter_status"] = "mismatch" if "mismatch" in comparisons else "unknown" if "unknown" in comparisons else "match"
    objective_match = math.isclose(total_cost, portfolio["expected_cost"], rel_tol=1e-8, abs_tol=1e-8) and math.isclose(total_loss, portfolio["expected_loss"], rel_tol=1e-6, abs_tol=1e-6)
    source["objective_status"] = "match" if objective_match else "mismatch"
    source["model_status"] = "unknown" if not metadata.get("Model_Protocol_Version") else "match" if metadata["Model_Protocol_Version"] == provenance["model_protocol"] else "mismatch"
    source["pruning_status"] = "unknown"
    if not objective_match or "mismatch" in (source["historical_input_status"], source["historical_parameter_status"], source["model_status"]):
        return "mismatch_current_scenario_only"
    return "historical_identity_or_pruning_unknown"


def solve(input_excel=None, equilibrium_config=None, *, excel_path=None):
    """Return final E3 table rows without writing results or invoking another feature."""
    if input_excel is not None and excel_path is not None:
        raise ValueError("provide only one input workbook argument")
    input_excel = get_input_excel(input_excel if input_excel is not None else excel_path)
    config = load_configuration(equilibrium_config)
    config["thresholds"] = resolve_thresholds(config["thresholds"])
    parameters = load_case_parameters(input_excel, names=("Use_Pruning", "Game_Iterations", "Game_Learning_Rate"))
    if not config.get("_prune_tol_explicit", False):
        config["prune_tol"] = resolve_prune_tolerance(parameters.get("Use_Pruning", True))
    defense_items, all_data, graph = load_problem_context(input_excel)
    provenance = _provenance(input_excel)
    portfolios = load_portfolios(config, defense_items, provenance["input_sha256"])
    tables = {name: [] for name in TABLE_NAMES}
    publication_tables = {}
    if config.get("publication_summary") is not None:
        from features.equilibrium_validation.publication import TABLE_NAMES as PUBLICATION_TABLE_NAMES
        publication_tables = {name: [] for name in PUBLICATION_TABLE_NAMES}
    for portfolio in portfolios:
        prefix, response = split_decision_vector(portfolio["decision"], len(defense_items))
        portfolio["normalized_cost"] = float(np.dot(prefix.astype(float), [cost for _, cost in defense_items]))
        designs = _design(
            config, parameters["Game_Iterations"], parameters["Game_Learning_Rate"], response,
            base_prune_tol=config["prune_tol"],
        )
        full_designs = designs[:config["max_full_graph_runs"]]
        baseline_results = None
        baseline_paths = {}
        baseline_loss = None
        baseline_endpoint = {}
        endpoint_groups = []
        publication_records = []
        for run_number, scenario in enumerate(full_designs):
            design = {**scenario, "experiment_type": "full_graph_replay"}
            keys = {"Case_ID": input_excel.stem, "Experiment_ID": config["experiment_id"],
                    "Portfolio_ID": portfolio["portfolio_id"], "Run_ID": f"{portfolio['portfolio_id']}-full-{run_number:03d}"}
            try:
                selected = create_selected_configuration(prefix, defense_items, all_data)
                results = run_simulation(
                    graph.state_paths, selected["vuln_data_sim"], all_data["Vulnerabilities distribution"],
                    selected["ids_config"], selected["firewall_config"], selected["patch_config"],
                    iterations=design["iterations"], learning_rate=design["learning_rate"],
                    p_response=design["response"], prune_tol=design["prune_tol"],
                    attacker_initializer=_initializer(design["initialization"], design["seed"]), capture_diagnostics=True,
                )
                reward, loss, availability = compute_total_expected_values(results)
                if not np.isfinite([reward, loss]).all():
                    raise ValueError("nonfinite complete-graph result")
                local_diagnostics = []
                endpoint = {}
                response_feasible = any(state.get("game_result", {}).get("has_response", False) for state in results.values())
                effective_design = {**design, "response": design["response"] if response_feasible else 0.0}
                for node_path, state in results.items():
                    if "game_result" not in state:
                        continue
                    local_diagnostics.extend(_game_rows(tables, keys, node_path, state["strategies"], state["game_result"], effective_design, config).values())
                    endpoint[node_path] = np.concatenate((state["game_result"]["final_p"], state["game_result"]["final_response_policy"]))
                if run_number == 0:
                    baseline_results = results
                    baseline_loss = loss
                    baseline_endpoint = endpoint
                paths = _path_rows(keys, graph, results, availability, baseline_paths, config["path_tie_tolerance"])
                if run_number == 0:
                    baseline_paths = {(row["Path_Type"], row["Path"]): row["Probability"] for row in paths}
                    for row in paths:
                        row["Baseline_Probability"] = row["Probability"]
                        row["Probability_Delta"] = 0.0 if row["Probability"] is not None else None
                tables["path_sensitivity"].extend(paths)
                path_differences = [abs(row["Probability_Delta"]) for row in paths if row["Path_Type"] == "complete_path" and row["Probability_Delta"] is not None]
                path_distance = max(path_differences, default=None)
                path_coverage_match = all(row["Probability"] is not None and row["Baseline_Probability"] is not None for row in paths if row["Path_Type"] == "complete_path")
                if not path_coverage_match:
                    path_distance = None
                same_states = set(endpoint) == set(baseline_endpoint)
                distance = max((float(np.max(np.abs(endpoint[node] - baseline_endpoint[node]))) for node in endpoint), default=0.0) if same_states else None
                stationary = bool(local_diagnostics) and all(row["Stationary_Status"] == "yes" for row in local_diagnostics)
                category = "finite_endpoint_not_certified"
                stability_signature = {row["Node_Path"]: row["Discrete_Stability_Status"] for row in local_diagnostics}
                for group_index, (reference, reference_stability) in enumerate(endpoint_groups):
                    if reference_stability == stability_signature and set(reference) == set(endpoint) and all(np.max(np.abs(reference[node] - endpoint[node])) <= config["endpoint_tolerance"] for node in endpoint):
                        category = f"observed_endpoint_group_{group_index + 1}"
                        break
                else:
                    endpoint_groups.append((endpoint, stability_signature))
                    category = f"observed_endpoint_group_{len(endpoint_groups)}"
                tables["initialization_sensitivity"].append({
                    **keys, "Node_Path": "__full_graph__", "Checkpoint": design["iterations"],
                    "Experiment_Type": "full_graph_replay", "Initialization": design["initialization"],
                    "Response_Initial": effective_design["response"], "Seed": design["seed"],
                    "Learning_Rate": design["learning_rate"], "Iterations": design["iterations"],
                    "Stop_Reason": "iteration_cap_stationary_candidate" if stationary else "iteration_cap_not_converged",
                    "Total_Defender_Loss": loss, "Total_Attacker_Reward": reward,
                    "Loss_Delta": loss - baseline_loss, "Strategy_Distance": distance, "Path_Distance": path_distance,
                    "Endpoint_Category": category, "Endpoint_Is_Equilibrium": False,
                    "Observed_State_Count": len(endpoint), "Status": "ok", "Reason": "endpoint groups do not establish multiple equilibria",
                })
                metadata = _metadata(keys, portfolio, effective_design, config, provenance)
                metadata["Requested_Response_Initial"] = design["response"]
                if run_number == 0:
                    cost = float(np.dot(prefix.astype(float), [cost for _, cost in defense_items]))
                    metadata["Replay_Status"] = _replay_status(portfolio, parameters, cost, loss, provenance)
                    metadata["Provenance_JSON"] = json_text({**provenance, "source": portfolio["source"]})
                metadata.update({"Requested_Full_Graph_Runs": len(designs), "Executed_Full_Graph_Runs": len(full_designs),
                                 "Total_Defender_Loss": loss, "Total_Attacker_Reward": reward,
                                 "Actual_Design_JSON": json_text(full_designs), "Prune_Tolerance": design["prune_tol"]})
                tables["run_metadata"].append(metadata)
                if config.get("publication_summary") is not None and design["design_factor"] in ("baseline", "attacker_initialization"):
                    publication_records.append({"keys": keys, "design": design, "results": results, "loss": float(loss)})
            except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
                _discard_partial_rows(tables, keys)
                reason = f"{type(error).__name__}: {error}"
                tables["run_metadata"].append(_metadata(keys, portfolio, design, config, provenance, "failed", reason))
                tables["initialization_sensitivity"].append({**keys, "Node_Path": "__full_graph__", "Checkpoint": design["iterations"],
                    "Experiment_Type": "full_graph_replay", "Initialization": design["initialization"], "Response_Initial": design["response"],
                    "Learning_Rate": design["learning_rate"], "Iterations": design["iterations"], "Stop_Reason": "failed",
                    "Endpoint_Category": "failed", "Status": "failed", "Reason": reason})
        if baseline_results is not None:
            _local_experiments(tables, input_excel.stem, config, portfolio, designs, baseline_results, provenance)
            if config.get("publication_summary") is not None:
                from features.equilibrium_validation.publication import build_tables
                publication = build_tables(publication_records, graph, portfolio, config, provenance)
                for name, rows in publication.items():
                    publication_tables[name].extend(rows)
    return {"input_excel": str(input_excel), "experiment_id": config["experiment_id"], "config": config,
            "provenance": provenance, "tables": tables, "publication_tables": publication_tables}


def _local_experiments(tables, case_id, config, portfolio, designs, baseline_results, provenance):
    states = [(name, state) for name, state in baseline_results.items() if "game_result" in state]
    if config["max_local_states"] is not None:
        states = states[:config["max_local_states"]]
    checkpoints = sorted(set(config["iteration_checkpoints"] + [designs[0]["iterations"]]))
    local_designs = [design for design in designs if design["design_factor"] not in ("iteration_cap", "pruning_scenario")]
    local_endpoint_groups = {}
    for run_number, scenario in enumerate(local_designs):
        design = {**scenario, "experiment_type": "fixed_local_snapshot", "iterations": max(checkpoints)}
        keys = {"Case_ID": case_id, "Experiment_ID": config["experiment_id"], "Portfolio_ID": portfolio["portfolio_id"], "Run_ID": f"{portfolio['portfolio_id']}-local-{run_number:03d}"}
        response_feasible = any(state["game_result"]["has_response"] for _, state in states)
        requested_response = design["response"]
        design["response"] = design["response"] if response_feasible else 0.0
        metadata = _metadata(keys, portfolio, design, config, provenance)
        metadata["Requested_Response_Initial"] = requested_response
        for node_path, state in states:
            inputs = state["game_result"]["game_inputs"]
            initializer = _initializer(design["initialization"], design["seed"])
            initial = inputs["p_init"].copy()
            if initializer is not None:
                initial = initializer(state["strategies"], initial)
            try:
                game = replicator_dynamics(initial, inputs["attacker_gain_matrix"], inputs["defender_loss_matrix"], inputs["f"],
                    iterations=design["iterations"], learning_rate=design["learning_rate"], has_response=state["game_result"]["has_response"],
                    p_response=design["response"], capture_diagnostics=True)
                outcomes = _game_rows(tables, keys, node_path, state["strategies"], game, design, config, checkpoints)
                reference = state["game_result"]
                for checkpoint, diagnostic in outcomes.items():
                    distance = max(float(np.max(np.abs(np.asarray(_history_at(game, "p_history", checkpoint)) - reference["final_p"]))),
                                   float(np.max(np.abs(np.asarray(_history_at(game, "response_policy_history", checkpoint)) - reference["final_response_policy"]))))
                    endpoint = np.concatenate((_history_at(game, "p_history", checkpoint), _history_at(game, "response_policy_history", checkpoint)))
                    signature = diagnostic["Discrete_Stability_Status"]
                    groups = local_endpoint_groups.setdefault(node_path, [])
                    category = None
                    for group_index, (reference_endpoint, reference_signature) in enumerate(groups):
                        if signature == reference_signature and np.max(np.abs(endpoint - reference_endpoint)) <= config["endpoint_tolerance"]:
                            category = f"observed_local_endpoint_group_{group_index + 1}"
                            break
                    if category is None:
                        groups.append((endpoint, signature))
                        category = f"observed_local_endpoint_group_{len(groups)}"
                    tables["initialization_sensitivity"].append({
                        **keys, "Node_Path": node_path, "Checkpoint": checkpoint, "Experiment_Type": "fixed_local_snapshot",
                        "Initialization": design["initialization"], "Response_Initial": design["response"], "Seed": design["seed"],
                        "Learning_Rate": design["learning_rate"], "Iterations": checkpoint,
                        "Stop_Reason": "checkpoint_stationary_candidate" if diagnostic["Stationary_Status"] == "yes" else "checkpoint_not_converged",
                        "Total_Defender_Loss": None, "Total_Attacker_Reward": None,
                        "Local_Defender_Loss": _history_at(game, "defender_loss_history", checkpoint),
                        "Local_Attacker_Reward": _history_at(game, "attacker_payoff_history", checkpoint),
                        "Strategy_Distance": distance, "Endpoint_Category": category, "Endpoint_Is_Equilibrium": False,
                        "Status": "ok", "Reason": "A/L/f frozen from original full-graph replay; not a full-graph sensitivity result",
                    })
            except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
                _discard_partial_rows(tables, keys, node_path)
                metadata["Status"] = "partial_failure"
                metadata["Reason"] = f"{node_path}: {type(error).__name__}: {error}"
                tables["initialization_sensitivity"].append({**keys, "Node_Path": node_path, "Checkpoint": design["iterations"],
                    "Experiment_Type": "fixed_local_snapshot", "Initialization": design["initialization"], "Response_Initial": design["response"],
                    "Learning_Rate": design["learning_rate"], "Iterations": design["iterations"], "Stop_Reason": "failed",
                    "Endpoint_Category": "failed", "Status": "failed", "Reason": metadata["Reason"]})
        metadata["Observed_State_Count"] = len(states)
        metadata["Local_State_Limit"] = config["max_local_states"]
        tables["run_metadata"].append(metadata)
if __name__ == "__main__":
    from common.pipeline import run_solve
    from features.equilibrium_validation.export_csv import export

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--excel", dest="input_excel")
    parser.add_argument("--equilibrium-config", required=True)
    arguments = parser.parse_args()
    run_solve("equilibrium_validation", solve, export, input_excel=arguments.input_excel, equilibrium_config=arguments.equilibrium_config)
