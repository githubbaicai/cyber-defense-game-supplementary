"""Fixed decisions, matched-FE reoptimization, and common-model evaluation for E7."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
from copy import deepcopy
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import time
import uuid

import numpy as np
import pandas as pd
from pymoo import __version__ as pymoo_version
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from common.case_config import get_input_excel
from common.defense_config import DefenseRepair, create_selected_configuration, normalize_decision_vector
from common.implementation import IMPLEMENTATION_VERSION, MODEL_PROTOCOL_VERSION
from common.memetic import MemeticNSGA2, get_local_search_statistics, is_feasible_individual
from common.model_options import ModelOptions
from common.parsers import defense_categories
from common.problem import CyberDefenseProblem, compose_decision_vector, decision_key, split_decision_vector
from common.problem_factory import load_problem_context
from common.sampling import FeasibleBinarySampling, ResponseAwareCrossover, ResponseAwareMutation
from common.simulation import PRUNE_TOL_DEFAULT, compute_total_expected_values, run_simulation
from .config import resolve_config, scenarios
from .export_csv import REQUIRED_COLUMNS, STRUCTURAL_PROTOCOL_VERSION, ensure_output_source, validate_tables


_WORKER_CONTEXT = None


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _code_hash():
    root = Path(__file__).resolve().parents[2]
    paths = [root / "run.py"]
    for directory in (root / "common", root / "external", Path(__file__).parent):
        paths.extend(sorted(directory.glob("*.py")))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _decision_fields(x, items, categories):
    prefix, response = split_decision_vector(x, len(items))
    costs = np.asarray([cost for _, cost in items], dtype=float)
    fields = {
        "decision": _json(np.asarray(x, dtype=float).tolist()),
        "selected_labels": _json([str(label) for enabled, (label, _) in zip(prefix, items) if enabled]),
        "p_response": response, "purchase_cost": float(prefix @ costs),
    }
    for group in ("prevention", "detection", "response"):
        fields[f"{group}_cost"] = float(sum(costs[index] for index in categories[group] if prefix[index]))
    return fields


def _fixed_decisions(items, data, settings):
    category_labels = defense_categories(items, data)
    categories = {
        domain.lower(): [index for index, category in enumerate(category_labels) if category == domain]
        for domain in ("Prevention", "Detection", "Response")
    }
    costs = np.asarray([cost for _, cost in items], dtype=float)
    budget = settings["budget"]
    # A domain has one fixed set of measures in every built-in portfolio.
    # Budget feasibility must not silently change the other control factors.
    domain_indices = {
        "prevention": categories["prevention"],
        "detection": categories["detection"],
        "response": [index for index in categories["response"] if items[index][0] == "RP3"],
    }
    portfolios = []
    for portfolio_id, domains in (
        ("none", ()), ("P", ("prevention",)), ("D", ("detection",)),
        ("R", ("response",)), ("PD", ("prevention", "detection")),
        ("PR", ("prevention", "response")), ("DR", ("detection", "response")),
        ("PDR", ("prevention", "detection", "response")),
    ):
        prefix = np.zeros(len(items))
        status = "ok"
        for domain in domains:
            indices = domain_indices[domain]
            if not indices:
                status = "unavailable_domain"
            prefix[indices] = 1
        if status == "ok" and prefix @ costs > budget:
            status = "insufficient_budget"
        response = 0.5 if any(prefix[index] for index in categories["response"]) else 0.0
        portfolios.append((portfolio_id, compose_decision_vector(prefix, response), status, False))
    identifiers = {row[0] for row in portfolios}
    labels = {str(label): index for index, (label, _) in enumerate(items)}
    for specification in settings["fixed_portfolios"]:
        if not isinstance(specification, dict) or set(specification) - {"portfolio_id", "labels", "p_response"}:
            raise ValueError("each fixed portfolio needs portfolio_id, labels and optional p_response")
        name = specification.get("portfolio_id")
        selected = specification.get("labels")
        response = specification.get("p_response", 0.5)
        if not isinstance(name, str) or not name.strip() or name in identifiers:
            raise ValueError("fixed portfolio IDs must be nonempty and unique")
        if not isinstance(selected, list) or len(set(selected)) != len(selected) or set(selected) - set(labels):
            raise ValueError(f"{name}: labels must be unique known Allocation labels")
        if isinstance(response, bool) or not isinstance(response, (int, float)) or not np.isfinite(response) or not 0 <= response <= 1:
            raise ValueError(f"{name}: p_response must be in [0, 1]")
        prefix = np.asarray([str(label) in selected for label, _ in items], dtype=float)
        original = compose_decision_vector(prefix, response)
        decision = normalize_decision_vector(original, items)
        status = "ok" if decision[:-1] @ costs <= budget else "insufficient_budget"
        portfolios.append((name, decision, status, not np.array_equal(original, decision)))
        identifiers.add(name)
    return portfolios, categories


def _evaluate(x, items, data, graph, settings, options):
    prefix, response = split_decision_vector(x, len(items))
    configuration = create_selected_configuration(prefix, items, data)
    model_kwargs = {} if options is None else {"model_options": options}
    if options is not None and options != ModelOptions():
        model_kwargs["has_response"] = any(label in ("RP1", "RP2", "RP3") for label in configuration["selected"])
    results = run_simulation(
        state_graph=graph.state_paths, vuln_data=configuration["vuln_data_sim"],
        component_vuln_dist=data["Vulnerabilities distribution"],
        ids_config=configuration["ids_config"], firewall_config=configuration["firewall_config"],
        patch_config=configuration["patch_config"], iterations=settings["game_iterations"],
        learning_rate=settings["game_learning_rate"], p_response=response,
        game_termination=settings.get("game_termination"),
        prune_tol=settings.get("prune_tol", PRUNE_TOL_DEFAULT), **model_kwargs,
    )
    payoff, loss, availability = compute_total_expected_values(results)
    if not np.isfinite([payoff, loss, *availability.values()]).all():
        raise ValueError("simulation returned non-finite results")
    games = [state["game_result"] for state in results.values() if "game_result" in state]
    stop_reasons = sorted({game["stop_reason"] for game in games})
    metrics = {
        "defender_loss": float(loss), "attacker_payoff": float(payoff),
        "local_games": len(games),
        "unconverged_local_games": sum(not game["converged"] and game["stop_reason"] != "fixed_policy" for game in games),
        "fixed_policy_local_games": sum(game["stop_reason"] == "fixed_policy" for game in games),
        "iterations_executed": max((game["iterations_executed"] for game in games), default=0),
        "stop_reason": ";".join(stop_reasons) if stop_reasons else "no_local_games",
        "converged": bool(games) and all(game["converged"] for game in games),
        "stationarity_residual": max((game["stationarity_residual"] for game in games), default=0.0),
        "invasion_gap": max((game["invasion_gap"] for game in games), default=0.0),
    }
    return metrics, results, availability


def _path_rows(results, availability, variant, portfolio_id, source, seed=-1):
    rows = []
    for path, state in results.items():
        game = state.get("game_result")
        base = {
            "variant": variant, "portfolio_id": portfolio_id, "source": source, "seed": seed,
            "path": path, "probability": float(availability.get(path, 0.0)),
            "probability_semantics": "selection_path_probability",
            "elapsed_time": float(state["attack_time"]),
        }
        if game is None:
            rows.append({**base, "action": "terminal", "is_attack": False})
            continue
        initial = np.asarray(game["p_history"][0], dtype=float)
        response_history = game["response_policy_history"]
        base.update({
            "initial_p": _json(initial.tolist()), "final_p": _json(game["final_p"].tolist()),
            "initial_response_policy": _json(np.asarray(response_history[0]).tolist()),
            "final_response_policy": _json(np.asarray(response_history[-1]).tolist()),
            "final_q": _json(game["final_q"].tolist()), "state_names": _json(list(game["state_names"])),
            "iterations_executed": game["iterations_executed"],
            "stop_reason": game["stop_reason"], "converged": game["converged"],
            "stationarity_residual": game["stationarity_residual"],
            "invasion_gap": game["invasion_gap"],
            "detection_probability": game["detection_probability"],
            "response_execution_probability": game["response_execution_probability"],
        })
        for index, strategy in enumerate(state["strategies"]):
            attack = hasattr(strategy, "vulnerability")
            pruned = attack and f"{path} → {strategy.target_component}" not in results
            raw_loss_contribution = float(base["probability"] * game["final_p"][index] * game["final_defender_losses"][index])
            row = {
                **base, "action_index": index, "is_attack": attack,
                "action": f"{strategy.target_component}: {strategy.vulnerability.name}" if attack else "DoNothing",
                "conditional_action_probability": float(game["final_p"][index]),
                "conditional_attacker_payoff": float(game["final_attacker_payoffs"][index]),
                "conditional_defender_loss": float(game["final_defender_losses"][index]),
                "pruned": pruned, "raw_expected_loss_contribution": raw_loss_contribution,
                "expected_loss_contribution": 0.0 if pruned else raw_loss_contribution,
            }
            if attack:
                ids = strategy.ids_config.get("network", {})
                detection_time = strategy.base_attack_time if strategy.model_options.history_mode == "local_step" else strategy.total_attack_time
                row.update({
                    "base_attack_time": strategy.base_attack_time, "detection_time": detection_time,
                    "total_attack_time": strategy.total_attack_time,
                    "network_detection": 1 - (1 - ids.get("eff", 0.0)) ** (detection_time / ids.get("cycle", 100)),
                    "action_detection_probability": strategy.detected_prob,
                    "passive_success_probability": strategy.passive_success_prob,
                    "active_success_probability": strategy.active_success_prob,
                    "evaluated_success_probability": strategy.compute_success_probability(
                        game["final_response_probability"], has_response=game["has_response"]
                    ),
                    "response_disruption_cost": strategy.response_operational_disruption_cost,
                })
            rows.append(row)
    return rows


def _front(population, budget, items):
    feasible = [individual for individual in population if is_feasible_individual(individual, budget)]
    if not feasible:
        raise RuntimeError("optimization produced no feasible portfolio")
    indices = NonDominatedSorting().do(np.asarray([individual.F for individual in feasible]), only_non_dominated_front=True)
    selected, seen = [], set()
    for index in sorted(indices, key=lambda index: tuple(feasible[index].F)):
        individual = feasible[index]
        key = decision_key(individual.X, len(items), items)
        if key not in seen:
            selected.append(individual)
            seen.add(key)
    return selected


def _optimize(items, data, graph, settings, options, seed):
    problem = CyberDefenseProblem(
        items, data, graph, budget=settings["budget"],
        game_iterations=settings["game_iterations"], game_learning_rate=settings["game_learning_rate"],
        game_termination=settings.get("game_termination"),
        prune_tol=settings.get("prune_tol", PRUNE_TOL_DEFAULT),
        model_options=None if options == ModelOptions() else options,
    )
    algorithm = MemeticNSGA2(
        max_fe=settings["max_fe"], local_fe_reference=settings["max_fe"],
        pop_size=settings["pop_size"], eliminate_duplicates=False,
        sampling=FeasibleBinarySampling(items, settings["budget"], all_data=data),
        crossover=ResponseAwareCrossover(prob=settings["crossover_prob"]),
        mutation=ResponseAwareMutation(prob=settings["mutation_prob"]),
        repair=DefenseRepair(items), **deepcopy(settings["ma_parameters"]),
    )
    random_state = np.random.get_state()
    started = time.perf_counter()
    try:
        result = minimize(problem, algorithm, termination=("n_eval", settings["max_fe"]), seed=seed, copy_algorithm=False, verbose=False)
    finally:
        np.random.set_state(random_state)
    elapsed = time.perf_counter() - started
    total = int(result.algorithm.evaluator.n_eval)
    if total != settings["max_fe"] or problem.total_evals != total:
        raise RuntimeError(f"equal-FE contract failed: requested {settings['max_fe']}, evaluator {total}, problem {problem.total_evals}")
    local = get_local_search_statistics(result.algorithm)
    record = {
        "seed": seed, "max_fe": settings["max_fe"], "total_fe": total,
        "genetic_fe": total - local["Local_FE"], "local_fe": local["Local_FE"],
        "cache_hits": problem.cache_hits, "unique_simulations": total - problem.cache_hits,
        "wall_time_s": elapsed, "status": "ok", "postprocessing_fe": 0,
        "optimization_postprocessing_fe": 0,
    }
    return record, _front(result.algorithm.pop, settings["budget"], items)


def _initialize_worker(input_excel, expected_sha256, settings):
    global _WORKER_CONTEXT
    source = Path(input_excel)
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected_sha256:
        raise RuntimeError("input workbook changed before worker initialization")
    with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink):
        items, data, graph = load_problem_context(source)
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected_sha256:
        raise RuntimeError("input workbook changed during worker initialization")
    _WORKER_CONTEXT = items, data, graph, settings


def _optimization_payload(items, data, graph, settings, options, seed):
    record, front = _optimize(items, data, graph, settings, options, seed)
    return record, [(individual.X.copy(), individual.F.copy()) for individual in front]


def _run_worker(task):
    if _WORKER_CONTEXT is None:
        raise RuntimeError("structural ablation worker is not initialized")
    _, options, seed = task
    with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink):
        return _optimization_payload(*_WORKER_CONTEXT, options, seed)


def _optimization_results(source, input_sha256, items, data, graph, settings, protocol):
    if settings["fixed_only"]:
        print("[structural_ablation] fixed-only: skipping reoptimization", flush=True)
        return
    tasks = [(variant, options, seed) for variant, options, reoptimize in protocol if reoptimize
             for seed in settings["seeds"]]
    actual_workers = min(settings["workers"], len(tasks))
    print(f"[structural_ablation] {len(tasks)} optimization tasks, workers={actual_workers}", flush=True)
    if actual_workers == 1:
        for variant, options, seed in tasks:
            print(f"E7 {source.stem}: {variant}, seed={seed}, FE={settings['max_fe']}", flush=True)
            record, front = _optimization_payload(items, data, graph, settings, options, seed)
            yield variant, options, seed, record, front
        return
    outputs = {}
    with ProcessPoolExecutor(
        max_workers=actual_workers, mp_context=get_context("spawn"),
        initializer=_initialize_worker, initargs=(str(source), input_sha256, settings),
    ) as pool:
        futures = {pool.submit(_run_worker, task): task for task in tasks}
        try:
            for position, future in enumerate(as_completed(futures), 1):
                variant, _, seed = futures[future]
                outputs[(variant, seed)] = future.result()
                record, _ = outputs[(variant, seed)]
                print(f"[{position}/{len(tasks)}] E7 {source.stem}: {variant}, seed={seed}, "
                      f"FE={record['total_fe']}, time={record['wall_time_s']:.2f}s", flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    if hashlib.sha256(source.read_bytes()).hexdigest() != input_sha256:
        raise RuntimeError("input workbook changed during parallel optimization")
    for variant, options, seed in tasks:
        record, front = outputs[(variant, seed)]
        yield variant, options, seed, record, front


def solve(input_excel=None, structural_config=None, **overrides):
    source = get_input_excel(input_excel)
    ensure_output_source(source)
    settings = resolve_config(source, structural_config, **overrides)
    items, data, graph = load_problem_context(source)
    portfolios, categories = _fixed_decisions(items, data, settings)
    protocol = scenarios()
    resolved = {**settings, "scenarios": [{"variant": name, "model_options": asdict(options), "reoptimize": optimize} for name, options, optimize in protocol]}
    config_json = _json(resolved)
    identity = {
        "case_id": source.stem, "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "run_id": uuid.uuid4().hex, "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "code_sha256": _code_hash(), "implementation_version": IMPLEMENTATION_VERSION,
        "model_protocol_version": MODEL_PROTOCOL_VERSION,
    }
    rows = {name: [] for name in REQUIRED_COLUMNS}
    checks, fixed_evaluations, verification_evaluations, cross_evaluations = [], 0, 0, 0
    representative_evaluations = 0
    baseline_fixed = {}
    for variant, options, _ in protocol:
        for portfolio_id, decision, status, repaired in portfolios:
            fields = _decision_fields(decision, items, categories)
            record = {"variant": variant, "portfolio_id": portfolio_id, **fields, **asdict(options), "status": status, "decision_repaired": repaired}
            if status != "ok":
                rows["fixed_portfolio_results"].append({**record, "defender_loss": np.nan, "attacker_payoff": np.nan})
                continue
            metrics, results, availability = _evaluate(decision, items, data, graph, settings, options)
            fixed_evaluations += 1
            rows["fixed_portfolio_results"].append({**record, **metrics})
            path_records = _path_rows(results, availability, variant, portfolio_id, "fixed")
            rows["ablation_paths"].extend(path_records)
            loss_error = abs(sum(row.get("expected_loss_contribution", 0.0) for row in path_records) - metrics["defender_loss"])
            checks.append({"check": f"loss_ledger:{variant}:{portfolio_id}", "passed": loss_error <= 1e-9, "value": loss_error})
            if variant == "baseline":
                legacy, legacy_states, legacy_paths = _evaluate(decision, items, data, graph, settings, None)
                verification_evaluations += 1
                errors = [abs(metrics["defender_loss"] - legacy["defender_loss"])]
                for path, state in results.items():
                    errors.append(abs(availability.get(path, 0.0) - legacy_paths.get(path, 0.0)))
                    if "game_result" in state:
                        for field in ("p_history", "q_history", "response_policy_history"):
                            errors.append(float(np.max(np.abs(np.asarray(state["game_result"][field]) - np.asarray(legacy_states[path]["game_result"][field])))))
                checks.append({"check": f"baseline_legacy_equivalence:{portfolio_id}", "passed": max(errors) <= 1e-12, "value": max(errors)})
                baseline_fixed[portfolio_id] = fields["decision"]
            checks.append({
                "check": f"fixed_decision_identity:{variant}:{portfolio_id}",
                "passed": baseline_fixed[portfolio_id] == fields["decision"], "value": 0,
            })
            for path, state in results.items():
                game = state.get("game_result")
                if game is None:
                    continue
                histories = [np.asarray(game[field], dtype=float) for field in ("p_history", "q_history", "response_policy_history")]
                error = max(float(np.max(np.abs(history.sum(axis=1) - 1))) for history in histories)
                valid = all(np.isfinite(history).all() and (history >= -1e-12).all() and (history <= 1 + 1e-12).all() for history in histories)
                checks.append({"check": f"probabilities:{variant}:{portfolio_id}:{path}", "passed": valid and error <= 1e-10, "value": error})
                if variant == "no_egt":
                    valid = game["iterations_executed"] == 0 and all(len(history) == 1 for history in histories)
                    checks.append({"check": f"no_egt_frozen:{portfolio_id}:{path}", "passed": valid, "value": game["iterations_executed"]})
    failed_checks = [check for check in checks if not check["passed"]]
    if failed_checks:
        details = "; ".join(f"{check['check']} (value={check['value']!r})" for check in failed_checks)
        raise RuntimeError(f"model checks failed before reoptimization: {details}")
    for variant, options, seed, record, front in _optimization_results(
        source, identity["input_sha256"], items, data, graph, settings, protocol,
    ):
        rows["structural_ablation_runs"].append({"variant": variant, **record, **asdict(options), "budget": settings["budget"], "pop_size": settings["pop_size"], "game_iterations": settings["game_iterations"], "game_learning_rate": settings["game_learning_rate"]})
        for index, (decision, objectives) in enumerate(front, 1):
            rows["ablation_fronts"].append({
                "variant": variant, "seed": seed, "solution_id": index,
                **_decision_fields(decision, items, categories),
                "defender_loss": float(objectives[1]), **asdict(options),
            })
        solution_id, (decision, objectives) = min(enumerate(front, 1), key=lambda pair: float(np.sum(pair[1][1])))
        native_metrics, native_states, native_probabilities = _evaluate(decision, items, data, graph, settings, options)
        representative_evaluations += 1
        native_error = abs(native_metrics["defender_loss"] - float(objectives[1]))
        if native_error > 1e-9:
            raise RuntimeError("representative reevaluation disagrees with optimization")
        checks.append({"check": f"native_recheck:{variant}:{seed}:{solution_id}", "passed": True, "value": native_error})
        rows["ablation_paths"].extend(
            {**row, "source_variant": variant, "solution_id": solution_id}
            for row in _path_rows(native_states, native_probabilities, variant, f"representative:{seed}:{solution_id}", "native_representative", seed)
        )
        metrics, states, probabilities = _evaluate(decision, items, data, graph, settings, ModelOptions())
        cross_evaluations += 1
        rows["cross_evaluation"].append({
            "source_variant": variant, "evaluation_variant": "baseline", "seed": seed,
            "solution_id": solution_id, "selection_rule": "min_purchase_cost_plus_defender_loss",
            "native_defender_loss": float(objectives[1]),
            **_decision_fields(decision, items, categories), **metrics,
        })
        rows["ablation_paths"].extend(
            {**row, "source_variant": variant, "solution_id": solution_id}
            for row in _path_rows(states, probabilities, "baseline", f"{variant}:{seed}:{solution_id}", "cross_evaluation", seed)
        )
    rows["model_checks"] = checks
    rows["experiment_manifest"].append({
        "input_excel": str(source), "baseline_protocol": MODEL_PROTOCOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(), "resolved_config": config_json,
        "pymoo_version": pymoo_version, "fixed_evaluations": fixed_evaluations,
        "verification_evaluations": verification_evaluations, "cross_evaluations": cross_evaluations,
        "representative_evaluations": representative_evaluations,
        "optimization_fe": sum(row["total_fe"] for row in rows["structural_ablation_runs"]),
        "fixed_only": settings["fixed_only"], "requested_workers": settings["workers"],
        "optimization_workers": min(settings["workers"], sum(reoptimize for _, _, reoptimize in protocol) * len(settings["seeds"])) if not settings["fixed_only"] else 0,
        "probability_semantics": "selection_path_probability; not compromise probability",
        "no_history_scope": "network detection uses local attack time; prevention has no historical recursion in the baseline",
        "interaction_scope": "all actions settle against population-average detection and conditional response",
        "attacker_initializer": "attack-benefit proportional with DoNothing raw weight 0.01",
        "cross_selection_rule": "min_purchase_cost_plus_defender_loss; both objectives use workbook monetary units",
        "model_options_protocol": STRUCTURAL_PROTOCOL_VERSION,
    })
    tables = {}
    for name, records in rows.items():
        tables[name] = pd.DataFrame([{**identity, **record} for record in records])
    return {"input_excel": str(source), "tables": validate_tables(tables, expected_input=source)}


def main():
    parser = argparse.ArgumentParser(description="Run isolated E7 model ablations")
    parser.add_argument("--input-excel", default=None)
    parser.add_argument("--structural-config", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--fixed-only", action="store_true", default=None)
    arguments = parser.parse_args()
    from common.pipeline import run_solve
    from .export_csv import export
    run_solve("structural_ablation", solve, export, input_excel=arguments.input_excel,
              structural_config=arguments.structural_config, workers=arguments.workers, fixed_only=arguments.fixed_only)


if __name__ == "__main__":
    main()
