"""One fresh process per complete, uncached objective evaluation."""

from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import sys
from time import perf_counter
import traceback

import numpy as np

from common.case_config import load_case_parameters
from common.game_termination import GameTermination
from common.model_options import ModelOptions
from common.parsers import component_vulnerability_names, defense_categories
from common.problem import CyberDefenseProblem
import common.problem as problem_module
from common.problem_factory import load_problem_context
from common.termination import FEASIBILITY_TOL


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def portfolio_decision(items, data, portfolio, response):
    categories = defense_categories(items, data)
    if portfolio not in ("none", "DR", "PDR"):
        raise ValueError(f"Unknown portfolio: {portfolio}")
    if portfolio != "none" and "RP3" not in {label for label, _ in items}:
        raise ValueError("RP3 is required for response portfolios")
    enabled = [
        portfolio != "none" and (
            label == "RP3" or category == "Detection"
            or (portfolio == "PDR" and category == "Prevention")
        )
        for (label, _), category in zip(items, categories)
    ]
    return np.asarray([*enabled, response if portfolio != "none" else 0.0], dtype=float)


@contextmanager
def capture_simulation():
    original = problem_module.run_simulation
    captured = []

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result)
        return result

    problem_module.run_simulation = record
    try:
        yield captured
    finally:
        problem_module.run_simulation = original


def workload(results):
    games = [state["game_result"] for state in results.values() if "game_result" in state]
    checked = [game for game in games if game["convergence_checked"]]
    return {
        "Visited_Prefixes": len(results),
        "Local_Games": len(games),
        "Total_Updates": sum(game["iterations_executed"] for game in games),
        "Iteration_Limit_Games": sum(game["stop_reason"] == "iteration_limit" for game in games),
        "Checked_Games": len(checked),
        "Unconverged_Games": sum(not game["converged"] for game in checked),
        "Fixed_Policy_Games": sum(game["stop_reason"] == "fixed_policy" for game in games),
        "Max_Residual": max((game["stationarity_residual"] for game in games), default=0.0),
        "Max_Invasion_Gap": max((game["invasion_gap"] for game in games), default=0.0),
        "Max_Local_Actions": max((len(state["strategies"]) for state in results.values()), default=0),
        "Diagnostic_Status": (
            "not_applicable_fixed_policy" if not checked
            else "all_passed" if all(game["converged"] for game in checked)
            else "limited_unconverged"
        ),
    }


def dimensions(items, data, graph):
    distribution = component_vulnerability_names(data["Vulnerability"], data["Vulnerabilities distribution"])
    return {
        "Assets": len([node for node in graph.nodes if node not in ("S", "T")]),
        "Vulnerabilities": len(data["Vulnerability"]["Vulnerability"]),
        "Component_Nodes": len(graph.nodes),
        "Component_Edges": sum(sum(row) for row in graph.adj_matrix),
        "Component_Paths": len(graph.attack_paths),
        "Potential_Vulnerability_Combinations": sum(
            math.prod(len(distribution.get(node, [])) for node in path if node not in ("S", "T"))
            for path in graph.attack_paths
        ),
        "Constructed_Prefix_Nodes": graph.state_graph.number_of_nodes(),
        "Constructed_Prefix_Edges": graph.state_graph.number_of_edges(),
        "Stored_State_Edge_Entries": len(graph.state_paths),
        "Binary_Measures": len(items), "Continuous_Variables": 1,
    }


def evaluate_once(task, settings):
    row = dict(task)
    row["Status"] = "error"
    row["Error"] = ""
    random.seed(task["Seed"])
    np.random.seed(task["Seed"])
    local_generator = np.random.default_rng(task["Seed"])
    row["RNG_Type"] = type(local_generator.bit_generator).__name__
    row["Input_SHA256"] = file_hash(task["Input_Excel"])
    setup_start = perf_counter()
    try:
        items, data, graph = load_problem_context(task["Input_Excel"])
        budget = load_case_parameters(task["Input_Excel"], names=("Budget",))["Budget"]
        decision = portfolio_decision(items, data, task["Portfolio"], settings["p_response"])
        options = ModelOptions(game_mode=task["Game_Mode"], **settings["model_options"])
        problem = CyberDefenseProblem(
            items, data, graph, budget=budget, use_constraint=True, use_cache=True,
            game_iterations=settings["game_iterations"],
            game_learning_rate=settings["game_learning_rate"],
            model_options=options, infer_response_from_effectiveness=False,
            prune_tol=settings["prune_tol"],
            game_termination=GameTermination(**settings["game_termination"]),
        )
        row["Setup_Seconds"] = perf_counter() - setup_start
        row.update(dimensions(items, data, graph))
        row["Budget_USD_Thousands"] = budget
        row["Selected_Measures"] = json.dumps([label for (label, _), enabled in zip(items, decision[:-1]) if enabled])
        row["Decision"] = json.dumps(decision.tolist())
        row["P_Response"] = float(decision[-1])
        output = {}
        with capture_simulation() as captured:
            evaluation_start = perf_counter()
            problem._evaluate(decision, output)
            row["Evaluation_Seconds"] = perf_counter() - evaluation_start
        if len(captured) != 1 or problem.cache_hits != 0 or problem.total_evals != 1 or len(problem.eval_cache) != 1:
            raise RuntimeError("Evaluation did not execute exactly one cold simulation")
        row.update(workload(captured[0]))
        row["Total_FE"] = problem.total_evals
        row["Cache_Hits"] = problem.cache_hits
        row["Uncached_Evaluations"] = len(problem.eval_cache)
        row["Cost_USD_Thousands"], row["Loss_USD_Thousands"] = map(float, output["F"])
        row["Feasible"] = bool(output["G"][0] <= FEASIBILITY_TOL)
        if not np.isfinite(output["F"]).all():
            row["Status"] = "nonfinite"
        elif not row["Feasible"]:
            row["Status"] = "infeasible"
        else:
            row["Status"] = "ok"
        row["Input_SHA256_After"] = file_hash(task["Input_Excel"])
        if row["Input_SHA256_After"] != row["Input_SHA256"]:
            raise RuntimeError("Input changed during evaluation")
    except Exception as error:
        row["Status"] = "error"
        row["Error"] = f"{type(error).__name__}: {error}"
        row["Traceback"] = traceback.format_exc()
    return row


def main():
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    row = evaluate_once(request["task"], request["settings"])
    Path(sys.argv[2]).write_text(json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
