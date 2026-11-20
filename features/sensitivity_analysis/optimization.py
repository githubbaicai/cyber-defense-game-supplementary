"""Decision evaluation and isolated, reproducible optimization for sensitivity runs."""

from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import random

import numpy as np
from pymoo.optimize import minimize

from common.defense_config import DefenseRepair, normalize_decision_vector
from common.memetic import MemeticNSGA2, get_local_search_statistics, is_feasible_individual
from common.model_options import resolve_model_options
from common.problem import CyberDefenseProblem
from common.simulation import PRUNE_TOL_DEFAULT, resolve_prune_tolerance
from common.sampling import FeasibleBinarySampling, ResponseAwareCrossover, ResponseAwareMutation
from common.termination import FEASIBILITY_TOL, MyTerminationSelector


def _value(config, name, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _canonical(problem, decision):
    if np.asarray(decision).size != problem.defense_count + 1:
        raise ValueError("sensitivity decisions must include the complete P_Response gene")
    return normalize_decision_vector(decision, problem.defense_items)


def _ids(problem, vector):
    labels = [str(label) for label, _ in problem.defense_items]
    prefix = [int(value >= 0.5) for value in vector[: len(labels)]]
    resource_payload = _json({"labels": labels, "binary": prefix}).encode("utf-8")
    resource_id = hashlib.sha256(resource_payload).hexdigest()
    response = float(vector[-1])
    decision_payload = (resource_id + ":" + response.hex()).encode("ascii")
    return resource_id, hashlib.sha256(decision_payload).hexdigest()


def _record(problem, vector, objective, constraint=None):
    vector = _canonical(problem, vector)
    objective = np.asarray(objective, dtype=float).reshape(-1)
    if objective.size != 2 or not np.isfinite(objective).all():
        raise ValueError("decision evaluation must return two finite objectives")
    cost, loss = map(float, objective)
    if constraint is None:
        constraint = cost - float(problem.budget)
    constraint = float(constraint)
    feasible = bool(np.isfinite(constraint) and constraint <= FEASIBILITY_TOL)
    labels = [str(label) for label, _ in problem.defense_items]
    selected = [label for enabled, label in zip(vector[: len(labels)], labels) if enabled >= 0.5]
    resource_id, decision_id = _ids(problem, vector)
    rp = [label for label in selected if label in ("RP1", "RP2", "RP3")]
    rp_level = rp[-1] if rp else "RP0"
    return {
        "Cost": cost,
        "Loss": loss,
        "J": cost + loss,
        "Constraint": constraint,
        "Feasible": feasible,
        "Resource_ID": resource_id,
        "Decision_ID": decision_id,
        "P_Response": float(vector[-1]),
        "RP_Level": rp_level,
        "Decision_Vector": _json([float(value) for value in vector]),
        "Allocation_Labels": _json(labels),
        "Selected_Defenses": _json(selected),
        "Exposure_Penalty_Ratio": resolve_model_options(problem.model_options).exposure_penalty_ratio,
        "Response_Penalty_Ratio": resolve_model_options(problem.model_options).response_penalty_ratio,
    }


def make_problem(data, graph, config, *, model_options=None):
    """Construct an independent evaluator, synchronizing costs from this data copy."""
    isolated_data = deepcopy(data)
    labels = list(isolated_data["Allocation"]["Label"])
    costs = list(isolated_data["Allocation"]["Cost"])
    if len(labels) != len(costs) or not labels:
        raise ValueError("Allocation Label/Cost must be nonempty and aligned")
    defense_items = list(zip(labels, costs))
    budget = _value(config, "budget")
    if budget is None:
        raise ValueError("sensitivity optimization requires a resolved budget")
    prune_tol = PRUNE_TOL_DEFAULT if config.use_pruning is None else resolve_prune_tolerance(config.use_pruning)
    return CyberDefenseProblem(
        defense_items,
        isolated_data,
        graph,
        budget=float(budget),
        use_constraint=True,
        use_cache=True,
        game_iterations=_value(config, "game_iterations"),
        game_learning_rate=_value(config, "game_learning_rate"),
        prune_tol=prune_tol,
        model_options=model_options,
        infer_response_from_effectiveness=True,
    )


def evaluate_decision(problem, decision, lambda_loss=1.0):
    """Evaluate a complete decision, including an over-budget decision unchanged."""
    try:
        lambda_loss = float(lambda_loss)
    except (TypeError, ValueError) as error:
        raise ValueError("lambda_loss must be finite and nonnegative") from error
    if not np.isfinite(lambda_loss) or lambda_loss < 0:
        raise ValueError("lambda_loss must be finite and nonnegative")
    vector = _canonical(problem, np.asarray(decision, dtype=float))
    output = {}
    problem._evaluate(vector, output)
    constraint = float(np.asarray(output.get("G", [output["F"][0] - problem.budget]))[0])
    result = _record(problem, output["X"], output["F"], constraint)
    result["J"] = result["Cost"] + lambda_loss * result["Loss"]
    return result


def _record_individual(problem, individual):
    constraint = None
    raw_g = individual.get("G")
    if raw_g is not None and np.asarray(raw_g).size:
        constraint = float(np.asarray(raw_g, dtype=float).reshape(-1)[0])
    record = _record(problem, individual.X, individual.F, constraint)
    record["Feasible"] = record["Feasible"] and is_feasible_individual(individual, problem.budget)
    return record


def _record_sort_key(record):
    return (float(record["J"]), float(record["Cost"]), float(record["Loss"]), str(record["Decision_ID"]))


def merge_fronts(fronts):
    """Deduplicate complete decisions, then retain the feasible cost-loss front."""
    candidates = []
    for front in fronts or []:
        for source in front or []:
            record = dict(source)
            if not record.get("Feasible", False):
                continue
            if not all(np.isfinite(float(record.get(key, np.nan))) for key in ("Cost", "Loss", "J")):
                continue
            candidates.append(record)
    by_decision = {}
    for record in sorted(candidates, key=_record_sort_key):
        by_decision.setdefault(str(record["Decision_ID"]), record)
    candidates = list(by_decision.values())
    front = []
    for candidate in candidates:
        dominated = any(
            other is not candidate
            and other["Cost"] <= candidate["Cost"]
            and other["Loss"] <= candidate["Loss"]
            and (other["Cost"] < candidate["Cost"] or other["Loss"] < candidate["Loss"])
            for other in candidates
        )
        if not dominated:
            front.append(candidate)
    return sorted(front, key=lambda record: (float(record["Cost"]), float(record["Loss"]), str(record["Decision_ID"])))


def select_decision(front):
    """Select the deterministic minimum-J point from a feasible front."""
    if not front:
        raise ValueError("cannot select a decision from an empty front")
    return min(front, key=_record_sort_key)


def _algorithm(problem, config):
    ma = deepcopy(_value(config, "ma_parameters", {}) or {})
    ma.setdefault("use_archive", True)
    pop_size = int(_value(config, "pop_size"))
    budget = float(_value(config, "budget"))
    sampling = FeasibleBinarySampling(problem.defense_items, budget, all_data=problem.all_data)
    return MemeticNSGA2(
        pop_size=pop_size,
        sampling=sampling,
        crossover=ResponseAwareCrossover(prob=float(_value(config, "crossover_prob", 0.9))),
        mutation=ResponseAwareMutation(prob=float(_value(config, "mutation_prob", 0.05))),
        repair=DefenseRepair(problem.defense_items),
        max_fe=int(_value(config, "max_fe")),
        **ma,
    )


def _empty_metrics(problem, status, error=None):
    metrics = {
        "Status": status, "Error": error, "Total_FE": int(problem.total_evals),
        "Local_FE": 0, "Cache_Hits": int(problem.cache_hits),
        "Simulation_Count": int(problem.total_evals - problem.cache_hits),
        "Best_J": np.nan, "Best_Cost": np.nan, "Best_Loss": np.nan,
        "Exposure_Penalty_Ratio": resolve_model_options(problem.model_options).exposure_penalty_ratio,
        "Response_Penalty_Ratio": resolve_model_options(problem.model_options).response_penalty_ratio,
    }
    return metrics


def optimize(data, graph, config, seed, *, model_options=None):
    """Run one isolated optimization and return its archived complete front."""
    seed = int(seed)
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    problem = make_problem(data, graph, config, model_options=model_options)
    algorithm = None
    try:
        np.random.seed(seed)
        random.seed(seed)
        with redirect_stdout(io.StringIO()):
            algorithm = _algorithm(problem, config)
            termination = MyTerminationSelector.get_termination(
                strategy="max_gen", max_gen=int(_value(config, "max_generations")),
                max_fe=int(_value(config, "max_fe")),
            )
            result = minimize(
                problem, algorithm, termination=termination, seed=seed,
                verbose=False, copy_algorithm=False,
            )
        candidates = []
        population = getattr(result.algorithm, "pop", None)
        if population is not None:
            candidates.append([_record_individual(problem, individual) for individual in population])
        archive = getattr(result.algorithm, "pareto_archive", None)
        if archive:
            candidates.append([_record_individual(problem, individual) for individual in archive])
        for candidate_front in candidates:
            for record in candidate_front:
                record["J"] = record["Cost"] + float(_value(config, "lambda_loss", 1.0)) * record["Loss"]
        front = merge_fronts(candidates)
        if not front:
            raise RuntimeError("optimization returned no feasible finite complete decision")
        if int(result.algorithm.evaluator.n_eval) != int(problem.total_evals):
            raise RuntimeError("optimization FE counters disagree")
        metrics = _empty_metrics(problem, "ok")
        metrics.update({
            "Total_FE": int(getattr(result.algorithm.evaluator, "n_eval", problem.total_evals)),
            "Local_FE": int(get_local_search_statistics(result.algorithm).get("Local_FE", 0)),
            "Cache_Hits": int(problem.cache_hits),
        })
        metrics["Simulation_Count"] = metrics["Total_FE"] - metrics["Cache_Hits"]
        metrics.update({
            "Seed": seed, "Crossover_Probability": float(_value(config, "crossover_prob", 0.9)),
            "Mutation_Probability": float(_value(config, "mutation_prob", 0.05)),
            "FE_Limit": int(_value(config, "max_fe")),
        })
        if front:
            best = select_decision(front)
            metrics.update({"Best_J": best["J"], "Best_Cost": best["Cost"], "Best_Loss": best["Loss"]})
        return {"front": front, "metrics": metrics}
    except Exception as error:
        metrics = _empty_metrics(problem, "failed", f"{type(error).__name__}: {error}")
        if algorithm is not None:
            metrics["Local_FE"] = int(get_local_search_statistics(algorithm).get("Local_FE", 0))
        metrics["Seed"] = seed
        return {"front": [], "metrics": metrics}
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
