

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd

from pymoo import __version__ as pymoo_version
from pymoo.core.callback import Callback
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from common.config import (
    NSGA2_MA_CROSSOVER_PROBABILITY,
    NSGA2_MA_MUTATION_PROBABILITY,
    NSGA2_MA_SEED,
    NSGA2_MA_TERMINATION_STRATEGY,
    TERMINATION_HV_REFERENCE_POINT,
    TERMINATION_LOSS_SCALE,
    TERMINATION_METRIC,
    TERMINATION_MIN_FRONT_SIZE,
    TERMINATION_MIN_GENERATIONS,
    TERMINATION_ROBUST_PERIOD,
    TERMINATION_STABLE_GENERATIONS,
    TERMINATION_TOLERANCE,
)
from common.case_config import get_input_excel, load_case_parameters
from common.defense_config import (
    DefenseRepair,
    create_selected_configuration,
)
from common.memetic import (
    MemeticNSGA2,
    get_selected_ma_file,
    get_local_search_statistics,
    is_feasible_individual,
    load_selected_ma_parameters,
)
from common.problem import CyberDefenseProblem, split_decision_vector
from common.problem_factory import load_problem_context
from common.parsers import component_vulnerability_names, vulnerability_patch_labels
from common.sampling import (
    FeasibleBinarySampling,
    ResponseAwareCrossover,
    ResponseAwareMutation,
    defense_categories,
)
from common.simulation import (
    compute_total_expected_values,
    resolve_prune_tolerance,
    run_simulation,
)
from common.termination import (
    FrontWindowTermination,
    MyTerminationSelector,
    feasible_pareto_front,
)
from features.nsga2_ma.annotations import evaluate_reference_points, select_cost_breakdown_points


class OptimizationTrace(Callback):
    

    def __init__(self):
        super().__init__()
        self.data["rows"] = []
        self.data["generations"] = []

    def notify(self, algorithm):
        generation = int(algorithm.n_gen)
        total_fe = int(algorithm.evaluator.n_eval)
        generations = self.data["generations"]

        if generations and generations[-1] == generation:
            self.data["rows"] = [
                row
                for row in self.data["rows"]
                if row["Generation"] != generation
            ]
        else:
            generations.append(generation)
            self.data["main_loop_front"] = feasible_pareto_front(algorithm.pop, algorithm.problem.budget)

        for index, individual in enumerate(algorithm.pop, start=1):
            f = np.asarray(individual.F, dtype=float)
            self.data["rows"].append({
                "Generation": generation,
                "Total_FE": total_fe,
                "Solution_ID": index,
                "Total_Cost": float(f[0]),
                "Total_Defender_Loss": float(f[1]),
            })


def _pareto_individuals(population, budget, defense_count):
    
    feasible = [
        individual
        for individual in population
        if is_feasible_individual(individual, budget)
    ]
    if not feasible:
        return []

    values = np.asarray(
        [individual.F for individual in feasible],
        dtype=float,
    )
    indices = NonDominatedSorting().do(
        values,
        only_non_dominated_front=True,
    )
    candidates = sorted(
        [feasible[int(index)] for index in indices],
        key=lambda individual: (
            float(individual.F[0]),
            float(individual.F[1]),
        ),
    )

    seen = set()
    unique = []
    for individual in candidates:
        prefix, _ = split_decision_vector(individual.X, defense_count)
        key = tuple(bool(value) for value in prefix)
        if key not in seen:
            seen.add(key)
            unique.append(individual)

    return [
        individual
        for individual in unique
        if not any(
            other is not individual
            and other.F[1] <= individual.F[1] + 1e-3
            and other.F[0] < individual.F[0] - 1e-3
            for other in unique
        )
    ]


def _vulnerability_lookup(all_data):
    
    sheet = all_data["Vulnerability"]
    names = sheet["Vulnerability"]
    if not names:
        return {}
    probabilities = np.asarray(sheet["Attack Success Probability"], dtype=float)
    if probabilities.shape != (len(names),) or not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError("Vulnerability Attack Success Probability must provide finite [0, 1] values for every row")
    patch_sheet = all_data["Patch vulnerability"]
    labels = patch_sheet["Label"]
    effects = patch_sheet["Patching effectiveness"]
    if len(labels) != len(effects) or len(set(labels)) != len(labels):
        raise ValueError("Patch Label values must be unique and correspond row by row to Patching effectiveness")
    patch_effectiveness = dict(zip(labels, effects))

    patch_labels_by_name = vulnerability_patch_labels(sheet, patch_sheet)
    lookup = {}
    for index, description in enumerate(names):
        label = patch_labels_by_name[description]
        lookup[description] = {
            "Label": label,
            "BaseProb": float(probabilities[index]),
            "PatchEff": float(patch_effectiveness[label]) if label is not None else 0.0,
            "Desc": description,
        }

    return {
        component: [lookup[name] for name in references]
        for component, references in component_vulnerability_names(
            sheet, all_data["Vulnerabilities distribution"]
        ).items()
    }


def _strategy_name(strategy):
    
    if not hasattr(strategy, "target_component"):
        return "Do Nothing"

    name = f"Attack {strategy.target_component}"
    if hasattr(strategy, "vulnerability"):
        name += f" ({strategy.vulnerability.name})"
    return name


def _analyze_solution(
    solution_id,
    x,
    defense_items,
    all_data,
    graph,
    by_component,
    path_rows,
    game_rows,
    game_iterations,
    game_learning_rate,
    prune_tol,
    p_response=0.0,
):
    
    configuration = create_selected_configuration(
        x,
        defense_items,
        all_data,
    )

    try:
        simulated = run_simulation(
            state_graph=graph.state_paths,
            vuln_data=configuration["vuln_data_sim"],
            component_vuln_dist=all_data[
                "Vulnerabilities distribution"
            ],
            ids_config=configuration["ids_config"],
            firewall_config=configuration["firewall_config"],
            patch_config=configuration["patch_config"],
            iterations=game_iterations,
            learning_rate=game_learning_rate,
            p_response=p_response,
            prune_tol=prune_tol,
        )
        _, _, availability = compute_total_expected_values(simulated)

        for path, probability in availability.items():
            if path == "S" or probability <= 1e-6:
                continue

            for component in path.split(" → "):
                if component == "S":
                    continue

                firewall_effectiveness = configuration["firewall_config"].get(component, 0.0)
                network_firewall_effectiveness = configuration["firewall_config"].get("network", 0.0)

                for vulnerability in by_component.get(component, []):
                    patched = (
                        vulnerability["Label"]
                        in configuration["selected"]
                    )
                    patch_effectiveness = (
                        vulnerability["PatchEff"] if patched else 0.0
                    )
                    effective_probability = (
                        vulnerability["BaseProb"]
                        * (1.0 - patch_effectiveness)
                        * (1.0 - firewall_effectiveness)
                        * (1.0 - network_firewall_effectiveness)
                    )

                    path_rows.append({
                        "Solution_ID": solution_id,
                        "Path": path,
                        "Path_Probability": probability,
                        "Component": component,
                        "Vulnerability": vulnerability["Desc"],
                        "Description": vulnerability["Desc"],
                        "Base_Prob": vulnerability["BaseProb"],
                        "Patched?": "YES" if patched else "NO",
                        "Firewall?": (
                            "YES" if firewall_effectiveness > 0 or network_firewall_effectiveness > 0 else "NO"
                        ),
                        "Effective_Prob": effective_probability,
                    })

        for node, information in simulated.items():
            history = information.get("evolution_history", [])
            strategies = information.get("strategies", [])
            if len(history) == 0 or len(strategies) == 0:
                continue

            for generation, probabilities in enumerate(history):
                for strategy, probability in zip(
                    strategies,
                    probabilities,
                ):
                    game_rows.append({
                        "Solution_ID": solution_id,
                        "Node_Path": node,
                        "Generation": generation,
                        "Strategy": _strategy_name(strategy),
                        "Probability": float(probability),
                    })

    except Exception as exc:
        raise RuntimeError(f"Per-solution analysis failed: Solution_ID={solution_id}, decision={np.asarray(x).tolist()}") from exc


def solve(
    max_gen=None,
    pop_size=None,
    strategy=NSGA2_MA_TERMINATION_STRATEGY,
    tol=None,
    n_stable=None,
    period=None,
    budget=None,
    max_fe=None,
    game_iterations=None,
    game_learning_rate=None,
    *,
    metric=None,
    min_gen=None,
    min_front_size=None,
    loss_scale=None,
    ref_point=None,
    input_excel=None,
):
    
    if strategy not in ("convergence", "combined_or", "combined_and", "robust", "max_gen"):
        raise ValueError(f"Unknown termination strategy: {strategy}")
    input_excel = get_input_excel(input_excel)
    overrides = {
        name: value for name, value in (
            ("Max_Generations", max_gen), ("Population_Size", pop_size),
            ("Budget", budget), ("Termination_Tolerance", tol),
            ("Termination_Stable_Generations", n_stable),
            ("Termination_Robust_Period", period),
            ("Game_Iterations", game_iterations),
            ("Game_Learning_Rate", game_learning_rate),
        ) if value is not None
    }
    if strategy != "max_gen":
        overrides.update({
            name: value for name, value in (
                ("Termination_Metric", metric),
                ("Termination_Min_Generations", min_gen),
                ("Termination_Min_Front_Size", min_front_size),
                ("Termination_Loss_Scale", loss_scale),
            ) if value is not None
        })
        if ref_point is not None:
            reference = np.asarray(ref_point, dtype=float)
            if reference.shape != (2,):
                raise ValueError("ref_point must contain two normalized coordinates")
            overrides.update(zip(
                ("Termination_HV_Reference_Cost", "Termination_HV_Reference_Loss"),
                reference,
            ))
    parameter_names = ["Use_Pruning", "Budget", "Population_Size", "Max_Generations", "Game_Iterations", "Game_Learning_Rate"]
    if strategy != "max_gen":
        parameter_names.extend([
            "Termination_Tolerance", "Termination_Stable_Generations", "Termination_Robust_Period",
            "Termination_Metric", "Termination_Min_Generations", "Termination_Min_Front_Size",
            "Termination_Loss_Scale", "Termination_HV_Reference_Cost", "Termination_HV_Reference_Loss",
        ])
    parameters = load_case_parameters(
        input_excel, include_termination=strategy != "max_gen", names=parameter_names,
        overrides={name: value for name, value in overrides.items() if name in parameter_names},
    )
    max_gen = parameters["Max_Generations"]
    pop_size = parameters["Population_Size"]
    budget = parameters["Budget"]
    tol = parameters.get("Termination_Tolerance", TERMINATION_TOLERANCE)
    n_stable = parameters.get("Termination_Stable_Generations", TERMINATION_STABLE_GENERATIONS)
    period = parameters.get("Termination_Robust_Period", TERMINATION_ROBUST_PERIOD)
    game_iterations = parameters["Game_Iterations"]
    game_learning_rate = parameters["Game_Learning_Rate"]
    prune_tol = resolve_prune_tolerance(parameters["Use_Pruning"])
    metric = parameters.get("Termination_Metric", TERMINATION_METRIC)
    min_gen = parameters.get("Termination_Min_Generations", TERMINATION_MIN_GENERATIONS)
    min_front_size = parameters.get("Termination_Min_Front_Size", TERMINATION_MIN_FRONT_SIZE)
    loss_scale = parameters.get("Termination_Loss_Scale")
    if ref_point is None:
        ref_point = tuple(
            parameters[name] if parameters.get(name) is not None else default
            for name, default in zip(
                ("Termination_HV_Reference_Cost", "Termination_HV_Reference_Loss"),
                TERMINATION_HV_REFERENCE_POINT,
            )
        )
    if metric == "front" and strategy != "max_gen":
        if isinstance(period, (bool, np.bool_)) or period != 1:
            raise ValueError("metric=front uses the complete stability window and requires Termination_Robust_Period=1")
        if loss_scale is None:
            raise ValueError("metric=front requires an offline-calibrated Termination_Loss_Scale")
        termination = FrontWindowTermination(
            max_gen=max_gen, max_fe=max_fe, budget=budget,
            loss_scale=loss_scale, ref_point=ref_point, tol=tol,
            n_stable=n_stable, min_gen=min_gen, min_front_size=min_front_size,
        )
    else:
        termination = MyTerminationSelector.get_termination(
            strategy=strategy, max_gen=max_gen, tol=tol,
            n_stable=n_stable, period=period,
            metric=metric if metric != "front" else "both",
            budget=budget,
            loss_scale=TERMINATION_LOSS_SCALE if loss_scale is None else loss_scale,
            ref_point=ref_point, min_gen=min_gen, min_front_size=min_front_size,
            max_fe=max_fe,
        )
    print(f"[nsga2_ma] input={input_excel}; budget={budget}; pop_size={pop_size}; max_gen={max_gen}")
    if max_fe is not None and max_fe < pop_size:
        raise ValueError("max_fe must not be smaller than pop_size")

    items, all_data, graph = load_problem_context(input_excel)
    ma_parameters, override_metadata = load_selected_ma_parameters(
        budget=budget,
        pop_size=pop_size,
        game_iterations=game_iterations,
        game_learning_rate=game_learning_rate,
        input_excel=input_excel,
    )
    selected_ma_file = get_selected_ma_file(input_excel)
    try:
        selection_document = json.loads(selected_ma_file.read_text(encoding="utf-8"))
    except OSError as error:
        raise FileNotFoundError(f"Unable to read the current case selection: {selected_ma_file}") from error

    problem = CyberDefenseProblem(
        items,
        all_data,
        graph,
        budget=budget,
        use_constraint=True,
        use_cache=True,
        game_iterations=game_iterations,
        game_learning_rate=game_learning_rate,
        prune_tol=prune_tol,
    )

    local_fe_reference = (
        max_fe if max_fe is not None else pop_size * max_gen
    )
    algorithm = MemeticNSGA2(
        pop_size=pop_size,
        sampling=FeasibleBinarySampling(items, budget, all_data=all_data),
        crossover=ResponseAwareCrossover(
            prob=NSGA2_MA_CROSSOVER_PROBABILITY
        ),
        mutation=ResponseAwareMutation(
            prob=NSGA2_MA_MUTATION_PROBABILITY
        ),
        repair=DefenseRepair(items),
        eliminate_duplicates=False,
        max_fe=max_fe,
        local_fe_reference=local_fe_reference,
        **ma_parameters,
    )
    np.random.seed(NSGA2_MA_SEED)
    result = minimize(
        problem,
        algorithm,
        termination=termination,
        seed=NSGA2_MA_SEED,
        callback=OptimizationTrace(),
        save_history=False,
        verbose=False,
    )

    finished = result.algorithm
    actual_fe = int(finished.evaluator.n_eval)
    if int(problem.total_evals) != actual_fe:
        raise RuntimeError("Optimization-problem evaluation counts and pymoo FE disagree")
    if max_fe is not None and actual_fe > max_fe:
        raise RuntimeError("Optimization exceeded the global FE safety cap")

    selected = _pareto_individuals(finished.pop, budget, len(items))
    labels = [item[0] for item in items]
    categories = defense_categories(items, all_data)
    if len(categories) != len(items):
        raise ValueError("Measure-category results must correspond item by item to Allocation")
    by_component = _vulnerability_lookup(all_data)

    pareto_rows = []
    matrix_rows = []
    item_rows = []
    cost_rows = []
    path_rows = []
    game_rows = []

    for solution_id, individual in enumerate(selected, start=1):
        x, p_response = split_decision_vector(individual.X, len(items))
        f = np.asarray(individual.F, dtype=float)

        pareto_rows.append({
            "Solution_ID": solution_id,
            "Total_Cost": float(f[0]),
            "Total_Defender_Loss": float(f[1]),
            "P_Response": p_response,
        })

        matrix_row = {"Solution_ID": solution_id}
        matrix_row["P_Response"] = p_response
        category_costs = {
            "Prevention": 0.0,
            "Detection": 0.0,
            "Response": 0.0,
        }

        for index, (label, cost) in enumerate(items):
            active = bool(x[index])
            matrix_row[label] = int(active)
            if not active:
                continue

            category = categories[index]
            if category not in category_costs:
                raise ValueError(f"Invalid measure category: {label}={category!r}")
            item_rows.append({
                "Solution_ID": solution_id,
                "Category": category,
                "Defense_Label": label,
                "Cost": cost,
            })
            category_costs[category] += float(cost)

        total_cost = float(sum(category_costs.values()))
        if not np.isfinite([*category_costs.values(), total_cost, f[0]]).all() or not math.isclose(
            total_cost, float(f[0]), rel_tol=1e-9, abs_tol=1e-9,
        ):
            raise ValueError(
                f"Category costs and optimization cost disagree: Solution_ID={solution_id}, "
                f"categories={category_costs!r}, total={total_cost!r}, F[0]={float(f[0])!r}"
            )
        matrix_rows.append(matrix_row)
        cost_rows.append({
            **category_costs,
            "TotalCost": total_cost,
            "DefenderLoss": float(f[1]),
            "P_Response": p_response,
            "Solution_ID": solution_id,
        })

        _analyze_solution(
            solution_id,
            x,
            items,
            all_data,
            graph,
            by_component,
            path_rows,
            game_rows,
            game_iterations,
            game_learning_rate,
            prune_tol,
            p_response,
        )

    statistics = get_local_search_statistics(finished)
    elite_quota = 0
    if ma_parameters["local_search_prob"] > 0:
        elite_quota = (
            ma_parameters["elite_size"]
            if ma_parameters["elite_size"] is not None
            else math.ceil(
                pop_size * ma_parameters["local_search_prob"]
            )
        )

    callback_data = finished.callback.data
    metadata = {
        **override_metadata,
        "Budget": budget,
        "Seed": NSGA2_MA_SEED,
        "Population_Size": pop_size,
        "Termination_Strategy": strategy,
        "Safety_Max_Generations": max_gen,
        "Actual_Generations": len(callback_data["generations"]),
        "Total_FE": actual_fe,
        "Genetic_FE": actual_fe - statistics["Local_FE"],
        **statistics,
        "Cache_Hits": int(problem.cache_hits),
        "Unique_Simulations": len(problem.eval_cache),
        "Selection_Experiment": selection_document["experiment_name"],
        "Selection_Mode": ma_parameters["selection_mode"],
        "Neighborhood": ma_parameters["neighborhood"],
        "Local_Search_Prob": ma_parameters["local_search_prob"],
        "Elite_Size": ma_parameters["elite_size"],
        "Elite_Quota": elite_quota,
        "Max_Local_Search_Iters": ma_parameters[
            "max_local_search_iters"
        ],
        "No_Improve_Rounds": ma_parameters["no_improve_rounds"],
        "Selection_File": str(selected_ma_file),
        "Max_Local_FE_Ratio": ma_parameters["max_local_fe_ratio"],
        "Max_Local_FE_Per_Gen": ma_parameters["max_local_fe_per_gen"],
        "Local_Search_Trigger": ma_parameters["local_search_trigger"],
        "Local_Search_Start_Ratio": ma_parameters[
            "local_search_start_ratio"
        ],
        "Acceptance_Mode": ma_parameters["acceptance_mode"],
        "MA_Parameters_JSON": json.dumps(
            ma_parameters,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "Safety_Max_FE": finished.max_fe,
        "Game_Iterations": game_iterations,
        "Game_Learning_Rate": game_learning_rate,
        "Use_Pruning": parameters["Use_Pruning"],
        "Prune_Tolerance": prune_tol,
        "Pymoo_Version": pymoo_version,
    }

    output = {
        "input_excel": str(input_excel),
        "pareto_front": pd.DataFrame(pareto_rows),
        "defense_selection_matrix": pd.DataFrame(matrix_rows),
        "detailed_items_list": pd.DataFrame(item_rows),
        "cost_breakdown_data": pd.DataFrame(cost_rows),
        "optimization_history": pd.DataFrame(callback_data["rows"]),
        "game_strategy_history": pd.DataFrame(game_rows),
        "path_vulnerability_details": pd.DataFrame(path_rows),
        "defense_labels": labels,
        "metadata": metadata,
    }
    monitor = finished.termination
    history = monitor.history
    latest = history[-1]
    final_front = feasible_pareto_front(finished.pop, budget)
    snapshot = (
        callback_data["main_loop_front"] if strategy == "max_gen"
        else monitor.last_front
    )
    front_changed = not np.array_equal(snapshot, final_front)
    metadata.update({
        "Termination_Metric": metric if strategy != "max_gen" else "max_gen",
        "Main_Loop_Stop_Reason": monitor.stop_reason,
        "Main_Loop_Stop_Generation": latest["generation"],
        "Main_Loop_Total_FE": latest["total_fe"],
        "Postprocessing_FE": actual_fe - latest["total_fe"],
        "Postprocessing_Front_Changed": front_changed,
        "Final_Front_Window_Verified": False,
    })
    if strategy != "max_gen":
        metadata.update({
            "Termination_Min_Generations": min_gen,
            "Termination_Tolerance": tol,
            "Termination_Stable_Generations": n_stable,
            "Termination_Robust_Period": period,
            "Termination_Min_Front_Size": min_front_size,
            "Termination_Loss_Scale": TERMINATION_LOSS_SCALE if loss_scale is None else loss_scale,
            "Termination_HV_Reference_Cost": float(ref_point[0]),
            "Termination_HV_Reference_Loss": float(ref_point[1]),
        })
    output["termination_history"] = pd.DataFrame(history)
    if isinstance(monitor, FrontWindowTermination):
        exported_front = np.unique(
            np.asarray([individual.F for individual in selected], dtype=float).reshape(-1, 2),
            axis=0,
        )
        exported_front_unchanged = np.array_equal(monitor.last_front, exported_front)
        metadata.update({
            "Final_Front_Window_Verified": (
                monitor.stop_reason == "front_stable" and not front_changed and exported_front_unchanged
            ),
        })
        if front_changed or actual_fe != latest["total_fe"]:
            print("[nsga2_ma] finalize postprocessing changed the solution set or FE; the postprocessed results have not passed a new stability window")
        elif not exported_front_unchanged:
            print("[nsga2_ma] export filtering changed the front; the exported results have not passed a new stability window")
    output["cost_breakdown_selected_points"] = select_cost_breakdown_points(output["cost_breakdown_data"])
    output["cost_breakdown_reference_points"] = evaluate_reference_points(
        items,
        all_data,
        graph,
        game_iterations=game_iterations,
        game_learning_rate=game_learning_rate,
        prune_tol=prune_tol,
    )
    return output


def main():
    
    parser = argparse.ArgumentParser(
        description="Run NSGA-II+MA using the ablation-selected configuration"
    )
    parser.add_argument(
        "--max-gen", type=int, default=None
    )
    parser.add_argument(
        "--pop-size", type=int, default=None
    )
    parser.add_argument(
        "--strategy", default=NSGA2_MA_TERMINATION_STRATEGY
    )
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--tol", type=float, default=None)
    parser.add_argument("--n-stable", type=int, default=None)
    parser.add_argument("--period", type=int, default=None)
    parser.add_argument("--game-iterations", type=int, default=None)
    parser.add_argument("--game-learning-rate", type=float, default=None)
    parser.add_argument("--max-fe", type=int, default=None)
    parser.add_argument("--input-excel", default=None)
    args = parser.parse_args()

    from common.pipeline import run_solve
    from features.nsga2_ma.export_csv import export

    result = run_solve(
        "nsga2_ma",
        solve,
        export,
        max_gen=args.max_gen,
        pop_size=args.pop_size,
        strategy=args.strategy,
        budget=args.budget,
        max_fe=args.max_fe,
        tol=args.tol,
        n_stable=args.n_stable,
        period=args.period,
        game_iterations=args.game_iterations,
        game_learning_rate=args.game_learning_rate,
        input_excel=args.input_excel,
    )

    information = result["metadata"]
    print(
        f"✅ NSGA-II+MA：{information['Selection_Experiment']}；"
        f"Pareto={len(result['pareto_front'])}；"
        f"FE={information['Total_FE']}；"
        f"local FE={information['Local_FE']}"
    )


if __name__ == "__main__":
    main()
