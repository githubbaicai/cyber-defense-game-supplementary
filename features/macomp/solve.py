

from __future__ import annotations


import os

for thread_variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[thread_variable] = "1"





os.environ.setdefault("PYTHONHASHSEED", "0")

import argparse


if __name__ == "__main__":
    input_parser = argparse.ArgumentParser(add_help=False)
    input_parser.add_argument("--input-excel", default=None)
    bootstrap_args, _ = input_parser.parse_known_args()
    if bootstrap_args.input_excel is not None:
        os.environ["GAME_THEORY_EXCEL"] = os.path.abspath(bootstrap_args.input_excel)

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import asdict
from importlib.metadata import version
import json
import math
import warnings
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from common.config import (
    NSGA2_MA_CROSSOVER_PROBABILITY,
    NSGA2_MA_MUTATION_PROBABILITY,
)
from common.case_config import get_input_excel, load_case_parameters
from common.defense_config import DefenseRepair
from common.memetic import (
    MemeticNSGA2,
    get_local_search_statistics,
    is_feasible_individual,
    sha256_file,
)
from common.problem import CyberDefenseProblem, split_decision_vector
from common.problem_factory import load_problem_context
from common.implementation import IMPLEMENTATION_VERSION, MODEL_PROTOCOL_VERSION
from common.simulation import resolve_prune_tolerance
from common.sampling import (
    FeasibleBinarySampling,
    ResponseAwareCrossover,
    ResponseAwareMutation,
)

from features.macomp.config import (
    UNLIMITED_RATIO_CODE,
    VARIANTS,
    VARIANT_BY_NAME,
    Variant,
)
from features.macomp.utils import (
    EmpiricalReference,
    FrontTrace,
    mean_confidence_interval,
)


_WORKER_CONTEXT = None


def _initialize_worker(input_excel: str, expected_sha256: str) -> None:
    
    global _WORKER_CONTEXT

    with open(os.devnull, "w", encoding="utf-8") as sink:
        with redirect_stdout(sink):
            if sha256_file(input_excel) != expected_sha256:
                raise RuntimeError("The input Excel workbook changed during experiment startup")
            _WORKER_CONTEXT = load_problem_context(input_excel)


def _final_decisions(population, budget: float, defense_count) -> list[dict]:
    
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

    rows = []
    seen = set()

    for index in indices:
        individual = feasible[int(index)]
        prefix, response_probability = split_decision_vector(
            individual.X, defense_count
        )
        bits = tuple(int(value) for value in prefix)
        key = (bits, response_probability)
        if key in seen:
            continue

        seen.add(key)
        rows.append({
            "X_Bits": "".join(str(bit) for bit in bits),
            "P_Response": response_probability,
            "Total_Cost": float(individual.F[0]),
            "Total_Defender_Loss": float(individual.F[1]),
        })

    return sorted(
        rows,
        key=lambda row: (
            row["Total_Cost"],
            row["Total_Defender_Loss"],
            row["X_Bits"],
            row["P_Response"],
        ),
    )


def _run_one(task: tuple) -> dict:
    
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with redirect_stdout(sink):
            return _run_one_silent(task)


def _run_one_silent(task: tuple) -> dict:
    
    (
        variant,
        run_number,
        seed,
        fe_budget,
        pop_size,
        budget,
        game_iterations,
        game_learning_rate,
        prune_tol,
    ) = task

    if _WORKER_CONTEXT is None:
        raise RuntimeError("The worker-process problem context has not been initialized")

    items, all_data, graph = _WORKER_CONTEXT
    np.random.seed(seed)

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
        max_fe=fe_budget,
        **variant.ma_parameters(),
    )

    started = perf_counter()
    result = minimize(
        problem,
        algorithm,
        termination=("n_eval", fe_budget),
        seed=seed,
        callback=FrontTrace(budget),
        save_history=False,
        verbose=False,
    )
    elapsed = perf_counter() - started

    finished = result.algorithm
    actual_fe = int(finished.evaluator.n_eval)
    statistics = get_local_search_statistics(finished)
    steps = finished.callback.data["steps"]

    if actual_fe != fe_budget:
        raise RuntimeError(
            f"{variant.name} seed={seed}: "
            f"Actual FE={actual_fe}, budget={fe_budget}"
        )
    if int(problem.total_evals) != actual_fe:
        raise RuntimeError(
            "problem.total_evals and evaluator.n_eval disagree"
        )
    if len(problem.eval_cache) + int(problem.cache_hits) != actual_fe:
        raise RuntimeError("Simulation-cache statistics and FE counts disagree")
    if statistics["Local_FE"] != int(finished.n_local_evals):
        raise RuntimeError("Legacy and current local-FE statistics disagree")

    local_limit = statistics["Local_FE_Limit"]
    if local_limit is not None and statistics["Local_FE"] > local_limit:
        raise RuntimeError("Local search exceeded the total local-FE cap")

    if not steps:
        raise RuntimeError("The algorithm produced no generation history")

    final_front = steps[-1]["Front"].copy()
    if len(final_front) == 0:
        raise RuntimeError(
            f"{variant.name} seed={seed}: no feasible final front"
        )

    quota = 0
    if variant.local_search_prob > 0:
        quota = (
            variant.elite_size
            if variant.elite_size is not None
            else math.ceil(pop_size * variant.local_search_prob)
        )

    record = {
        "Experiment": variant.name,
        "Run": run_number,
        "Seed": seed,
        "Budget_FE": fe_budget,
        "Total_FE": actual_fe,
        "Genetic_FE": actual_fe - statistics["Local_FE"],
        **statistics,
        "Wall_Seconds": float(elapsed),
        "Generations": len(steps),
        "Pareto_Size": len(final_front),
        "Selection_Mode": variant.selection_mode,
        "Neighborhood": variant.neighborhood,
        "Local_Search_Prob": variant.local_search_prob,
        "Max_Local_Search_Iters": variant.max_local_search_iters,
        "No_Improve_Rounds": variant.no_improve_rounds,
        "Elite_Size": variant.elite_size,
        "Elite_Quota": quota,
        "Max_Local_FE_Ratio": variant.max_local_fe_ratio,
        "Local_FE_Ratio_Code": (
            UNLIMITED_RATIO_CODE
            if variant.max_local_fe_ratio is None
            else variant.max_local_fe_ratio
        ),
        "Max_Local_FE_Per_Gen": variant.max_local_fe_per_gen,
        "Local_Search_Trigger": variant.local_search_trigger,
        "Local_Search_Start_Ratio": variant.local_search_start_ratio,
        "HV_Stagnation_Generations": variant.hv_stagnation_generations,
        "Front_Stagnation_Generations": (
            variant.front_stagnation_generations
        ),
        "Trigger_HV_Tolerance": variant.trigger_hv_tolerance,
        "Trigger_Loss_Scale": variant.trigger_loss_scale,
        "Trigger_HV_Reference_Point": json.dumps(
            variant.trigger_hv_reference_point
        ),
        "Acceptance_Mode": variant.acceptance_mode,
        "Acceptance_Epsilon": variant.acceptance_epsilon,
        "Last_Gen_Local_FE_Ratio": variant.last_gen_local_fe_ratio,
        "Variant_Code": variant.code,
        "Variant_Group": variant.group,
        "Population_Size": pop_size,
        "Defense_Budget": budget,
        "Crossover_Prob": NSGA2_MA_CROSSOVER_PROBABILITY,
        "Mutation_Prob": NSGA2_MA_MUTATION_PROBABILITY,
        "Game_Iterations": game_iterations,
        "Game_Learning_Rate": game_learning_rate,
        "Use_Cache": True,
        "Eliminate_Duplicates": False,
        "Cache_Hits": int(problem.cache_hits),
        "Unique_Simulations": len(problem.eval_cache),
    }

    record["Local_FE_Ratio"] = statistics["Local_FE"] / actual_fe
    record["Local_Accepted_Per_1000_Local_FE"] = (
        1000.0 * statistics["Local_Accepted"] / statistics["Local_FE"]
        if statistics["Local_FE"] > 0
        else np.nan
    )

    return {
        "record": record,
        "front": final_front,
        "steps": steps,
        "decisions": _final_decisions(finished.pop, budget, len(items)),
    }


def _summary(runs_frame: pd.DataFrame) -> pd.DataFrame:
    
    metrics = (
        "HV",
        "IGD",
        "Wall_Seconds",
        "Total_FE",
        "Genetic_FE",
        "Local_FE",
        "Local_FE_Ratio",
        "Generations",
        "Pareto_Size",
        "Local_Accepted",
        "Local_Accepted_Per_1000_Local_FE",
        "Completed_Local_Rounds",
        "Interrupted_Local_Rounds",
        "Neighbor_Scans",
        "Local_Improved_Center",
        "Local_Entered_Final_Front",
        "Local_Dominates_Center",
        "Final_Refill_Genetic_FE",
        "Cache_Hits",
        "Unique_Simulations",
    )
    rows = []

    for (experiment, fe_budget), group in runs_frame.groupby(
        ["Experiment", "Budget_FE"],
        sort=True,
    ):
        row = {
            "Experiment": experiment,
            "Budget_FE": int(fe_budget),
            "Runs": len(group),
            "Variant_Code": group["Variant_Code"].iloc[0],
            "Variant_Group": group["Variant_Group"].iloc[0],
        }

        for metric in metrics:
            mean, deviation, lower, upper = mean_confidence_interval(
                group[metric].to_numpy(dtype=float)
            )
            row[f"{metric}_Mean"] = mean
            row[f"{metric}_Std"] = deviation
            row[f"{metric}_CI_Lower"] = lower
            row[f"{metric}_CI_Upper"] = upper

        rows.append(row)

    return pd.DataFrame(rows)


def _rank_and_select(
    summary: pd.DataFrame,
    runs: pd.DataFrame,
    variants: tuple[Variant, ...],
    context: dict,
    runs_per_variant: int,
) -> tuple[pd.DataFrame, dict]:
    
    from scipy.stats import wilcoxon

    alpha = 0.05
    if runs_per_variant < 6:
        warnings.warn(
            f"small sample: {runs_per_variant} paired runs per configuration; "
            "a two-sided exact Wilcoxon test with fewer than 6 nonzero pairs "
            "cannot attain p < 0.05. Selection/tie-break rules are unchanged.",
            RuntimeWarning, stacklevel=2,
        )
    baseline_name = "NSGA-II"
    budgets = sorted(
        int(value) for value in summary["Budget_FE"].unique()
    )

    significant_budgets: dict[str, list[int]] = {}
    for variant in variants:
        name = variant.name
        if name == baseline_name:
            significant_budgets[name] = []
            continue

        found = []
        for budget in budgets:
            baseline_runs = runs.loc[
                (runs["Experiment"] == baseline_name)
                & (runs["Budget_FE"] == budget)
            ].sort_values("Seed")
            variant_runs = runs.loc[
                (runs["Experiment"] == name)
                & (runs["Budget_FE"] == budget)
            ].sort_values("Seed")

            if len(baseline_runs) == 0:
                continue
            if len(baseline_runs) != len(variant_runs):
                continue
            if not np.array_equal(
                baseline_runs["Seed"].to_numpy(),
                variant_runs["Seed"].to_numpy(),
            ):
                continue

            left = baseline_runs["HV"].to_numpy(dtype=float)
            right = variant_runs["HV"].to_numpy(dtype=float)
            if not (np.isfinite(left).all() and np.isfinite(right).all()):
                continue

            difference = right - left
            if np.allclose(difference, 0.0):
                continue
            try:
                _, p_value = wilcoxon(left, right)
            except ValueError:
                continue

            if (
                float(p_value) < alpha
                and float(np.median(difference)) > 0.0
            ):
                found.append(budget)

        significant_budgets[name] = found

    summary["significant_budget_count"] = [
        len(significant_budgets.get(str(name), []))
        for name in summary["Experiment"]
    ]
    summary["significant_budgets"] = [
        json.dumps(significant_budgets.get(str(name), []))
        for name in summary["Experiment"]
    ]

    high_fe = int(summary["Budget_FE"].max())
    ranking = summary.loc[
        summary["Budget_FE"] == high_fe
    ].copy()

    if len(ranking) != len(variants):
        raise RuntimeError("The number of high-FE configurations is incomplete")
    if not (ranking["Runs"] == runs_per_variant).all():
        raise RuntimeError("The number of high-FE repeated runs is incomplete")
    if not np.isfinite(
        ranking[["HV_Mean", "IGD_Mean"]].to_numpy(dtype=float)
    ).all():
        raise RuntimeError("High-FE metrics are incomplete; automatic selection is unavailable")

    ranking["HV_Rank"] = ranking["HV_Mean"].rank(
        ascending=False,
        method="min",
    )
    ranking["IGD_Rank"] = ranking["IGD_Mean"].rank(
        ascending=True,
        method="min",
    )

    
    ranking["Combined_Rank"] = (
        ranking["HV_Rank"] + ranking["IGD_Rank"]
    )
    ranking["Deployment_Eligible"] = True
    ranking["Baseline_Priority"] = (
        ranking["Experiment"] != "NSGA-II"
    )

    
    ranking = ranking.sort_values(
        ["significant_budget_count", "HV_Mean", "Experiment"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranking.insert(0, "Rank", np.arange(1, len(ranking) + 1))

    variant_by_name = {variant.name: variant for variant in variants}
    winner = ranking.iloc[0]
    baseline = ranking.loc[
        ranking["Experiment"] == "NSGA-II"
    ].iloc[0]

    ma_names = {
        variant.name
        for variant in variants
        if variant.local_search_prob > 0
    }
    best_ma = ranking.loc[
        ranking["Experiment"].isin(ma_names)
    ].iloc[0]

    selected_variant = variant_by_name[str(winner["Experiment"])]

    document = {
        "schema_version": 1,
        "context": context,
        "selected": selected_variant.ma_parameters(),
        "experiment_name": selected_variant.name,
        "selection_rule": (
            "At each FE budget, first apply a seed-paired Wilcoxon "
            "signed-rank test (HV, p < 0.05 and positive median difference), "
            "and count the budgets where each configuration significantly outperforms NSGA-II. "
            "Select the configuration with the most significant budgets; break ties "
            "by descending mean HV at the highest FE budget, then by name. "
            "If no configuration is significant, use descending HV at the highest FE budget. "
            "Do not use rank sums or parallel runtime to break ties. "
            "If NSGA-II is selected, selected uses elite/all/prob=0 "
            "to indicate that local search is disabled."
        ),
        "reference_type": "pooled_empirical_observed_front",
        "high_fe_budget": high_fe,
        "significance_test": "wilcoxon_signed_rank_hv_paired_by_seed",
        "significance_alpha": alpha,
        "budgets_evaluated": budgets,
        "significant_budget_count": int(
            winner["significant_budget_count"]
        ),
        "significant_budgets": list(
            significant_budgets.get(str(winner["Experiment"]), [])
        ),
        "global_best_experiment": str(winner["Experiment"]),
        "best_ma_experiment": str(best_ma["Experiment"]),
        "selected_is_baseline": selected_variant.local_search_prob == 0,
        "selected_vs_nsga2": {
            "hv_mean_difference": float(
                winner["HV_Mean"] - baseline["HV_Mean"]
            ),
            "igd_mean_difference": float(
                winner["IGD_Mean"] - baseline["IGD_Mean"]
            ),
        },
        "best_ma_vs_nsga2": {
            "hv_mean_difference": float(
                best_ma["HV_Mean"] - baseline["HV_Mean"]
            ),
            "igd_mean_difference": float(
                best_ma["IGD_Mean"] - baseline["IGD_Mean"]
            ),
        },
    }

    return ranking, document


def solve(
    *,
    runs: int | None = None,
    pop_size: int | None = None,
    fe_budgets=None,
    seed_start: int | None = None,
    budget: float | None = None,
    game_iterations: int | None = None,
    game_learning_rate: float | None = None,
    input_excel=None,
    workers: int | None = None,
    variants=VARIANTS,
) -> dict:
    
    input_excel = get_input_excel(input_excel)
    overrides = {
        name: value for name, value in (
            ("Macomp_Runs", runs), ("Population_Size", pop_size),
            ("Macomp_FE_Budgets", fe_budgets), ("Macomp_Seed_Start", seed_start),
            ("Budget", budget), ("Game_Iterations", game_iterations),
            ("Game_Learning_Rate", game_learning_rate), ("Macomp_Workers", workers),
        ) if value is not None
    }
    parameters = load_case_parameters(
        input_excel,
        names=("Use_Pruning", "Macomp_Runs", "Population_Size", "Macomp_FE_Budgets", "Macomp_Seed_Start",
               "Budget", "Game_Iterations", "Game_Learning_Rate", "Macomp_Workers"),
        overrides=overrides,
    )
    runs = parameters["Macomp_Runs"]
    pop_size = parameters["Population_Size"]
    fe_budgets = parameters["Macomp_FE_Budgets"]
    seed_start = parameters["Macomp_Seed_Start"]
    budget = parameters["Budget"]
    game_iterations = parameters["Game_Iterations"]
    game_learning_rate = parameters["Game_Learning_Rate"]
    prune_tol = resolve_prune_tolerance(parameters["Use_Pruning"])
    workers = parameters["Macomp_Workers"]
    print(f"[macomp] input={input_excel}; budget={budget}; pop_size={pop_size}")
    integer_parameters = {
        "runs": (runs, 2),
        "pop_size": (pop_size, 2),
        "seed_start": (seed_start, 0),
        "game_iterations": (game_iterations, 1),
        "workers": (workers, 1),
    }
    for name, (value, minimum) in integer_parameters.items():
        if (
            isinstance(value, (bool, np.bool_))
            or int(value) != value
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")

    runs = int(runs)
    pop_size = int(pop_size)
    seed_start = int(seed_start)
    game_iterations = int(game_iterations)
    workers = int(workers)
    budget = float(budget)
    game_learning_rate = float(game_learning_rate)

    if not np.isfinite(budget) or budget <= 0:
        raise ValueError("budget must be positive and finite")
    if (
        not np.isfinite(game_learning_rate)
        or game_learning_rate <= 0
    ):
        raise ValueError("game_learning_rate must be positive and finite")
    if seed_start + runs - 1 > np.iinfo(np.uint32).max:
        raise ValueError("The random seed is outside the range of the legacy NumPy random interface")

    raw_budgets = tuple(fe_budgets)
    if not raw_budgets:
        raise ValueError("At least one FE budget is required")
    if any(
        isinstance(value, (bool, np.bool_))
        or int(value) != value
        or value < pop_size
        for value in raw_budgets
    ):
        raise ValueError("Each FE budget must be an integer greater than or equal to pop_size")

    fe_budgets = tuple(int(value) for value in raw_budgets)
    if list(fe_budgets) != sorted(set(fe_budgets)):
        raise ValueError("FE budgets must be strictly increasing without duplicates")

    variants = tuple(variants)
    names = [variant.name for variant in variants]

    if len(names) != len(set(names)):
        raise ValueError("Variant names must be unique")
    codes = [variant.code for variant in variants]
    if len(codes) != len(set(codes)):
        raise ValueError("Variant codes must be unique")
    if "NSGA-II" not in names:
        raise ValueError("The experiment must include the NSGA-II baseline")
    if not any(variant.local_search_prob > 0 for variant in variants):
        raise ValueError("The experiment requires at least one local-search variant")

    path = input_excel
    excel_digest = sha256_file(path)
    pymoo_version = version("pymoo")

    context = {
        "implementation_version": IMPLEMENTATION_VERSION,
        "model_protocol_version": MODEL_PROTOCOL_VERSION,
        "budget": budget,
        "pop_size": pop_size,
        "game_iterations": game_iterations,
        "game_learning_rate": game_learning_rate,
        "use_pruning": parameters["Use_Pruning"],
        "prune_tol": prune_tol,
        "excel_sha256": excel_digest,
    }

    tasks = [
        (
            variant,
            run_number,
            seed_start + run_number - 1,
            fe_budget,
            pop_size,
            budget,
            game_iterations,
            game_learning_rate,
            prune_tol,
        )
        for variant in variants
        for run_number in range(1, runs + 1)
        for fe_budget in fe_budgets
    ]
    actual_workers = min(workers, len(tasks))
    outputs = []

    print(
        f"[macomp] {len(variants)} configurations, {len(tasks)} tasks, "
        f"{actual_workers} spawn processes",
        flush=True,
    )

    with ProcessPoolExecutor(
        max_workers=actual_workers,
        mp_context=get_context("spawn"),
        initializer=_initialize_worker,
        initargs=(str(path), excel_digest),
    ) as pool:
        futures = [pool.submit(_run_one, task) for task in tasks]

        try:
            for position, future in enumerate(
                as_completed(futures),
                start=1,
            ):
                output = future.result()
                outputs.append(output)
                row = output["record"]

                print(
                    f"[{position}/{len(tasks)}] {row['Experiment']} "
                    f"seed={row['Seed']} FE={row['Total_FE']} "
                    f"genetic={row['Genetic_FE']} "
                    f"local={row['Local_FE']} "
                    f"generations={row['Generations']} "
                    f"time={row['Wall_Seconds']:.2f}s",
                    flush=True,
                )
        except BaseException:
            for future in futures:
                future.cancel()
            raise

    if sha256_file(path) != excel_digest:
        raise RuntimeError("The Excel workbook changed during the experiment; selection refused")

    outputs.sort(
        key=lambda output: (
            output["record"]["Experiment"],
            output["record"]["Run"],
            output["record"]["Budget_FE"],
        )
    )

    reference = EmpiricalReference(
        [
            step["Front"]
            for output in outputs
            for step in output["steps"]
        ],
        budget,
    )
    reference_meta = reference.metadata()

    run_rows = []
    history_rows = []
    decision_rows = []

    for output in outputs:
        base = output["record"]
        identifiers = {
            name: base[name]
            for name in (
                "Experiment",
                "Run",
                "Seed",
                "Budget_FE",
                "Variant_Code",
                "Variant_Group",
            )
        }

        run_rows.append({
            **base,
            **reference.metrics(output["front"]),
            **reference_meta,
            "Pymoo_Version": pymoo_version,
            "Excel_SHA256": excel_digest,
            "Implementation_Version": IMPLEMENTATION_VERSION,
            "Model_Protocol_Version": MODEL_PROTOCOL_VERSION,
        })

        for step in output["steps"]:
            front = step["Front"]
            history_rows.append({
                **identifiers,
                **{
                    name: value
                    for name, value in step.items()
                    if name != "Front"
                },
                "Pareto_Size": len(front),
                "Minimum_Cost": (
                    float(front[:, 0].min()) if len(front) else np.nan
                ),
                "Minimum_Loss": (
                    float(front[:, 1].min()) if len(front) else np.nan
                ),
                **reference.metrics(front),
            })

        for point_id, point in enumerate(output["decisions"], start=1):
            decision_rows.append({
                **identifiers,
                "Point_ID": point_id,
                **point,
            })

    runs_frame = pd.DataFrame(run_rows).sort_values(
        ["Experiment", "Run", "Budget_FE"]
    ).reset_index(drop=True)
    history_frame = pd.DataFrame(history_rows).sort_values(
        ["Experiment", "Run", "Budget_FE", "Generation"]
    ).reset_index(drop=True)

    if not np.isfinite(
        runs_frame[["HV", "IGD"]].to_numpy(dtype=float)
    ).all():
        raise RuntimeError("HV/IGD is invalid for at least one final run")

    decisions_frame = pd.DataFrame(
        decision_rows,
        columns=[
            "Experiment",
            "Run",
            "Seed",
            "Budget_FE",
            "Variant_Code",
            "Variant_Group",
            "Point_ID",
            "X_Bits",
            "P_Response",
            "Total_Cost",
            "Total_Defender_Loss",
        ],
    ).sort_values(
        ["Experiment", "Run", "Budget_FE", "Point_ID"]
    ).reset_index(drop=True)

    summary_frame = _summary(runs_frame)
    ranking_frame, selection = _rank_and_select(
        summary_frame,
        runs_frame,
        variants,
        context,
        runs,
    )

    reference_frame = pd.DataFrame({
        "Total_Cost": reference.front[:, 0],
        "Total_Defender_Loss": reference.front[:, 1],
        "Normalized_Cost": (
            reference.front[:, 0] - reference.ideal[0]
        ) / reference.span[0],
        "Normalized_Loss": (
            reference.front[:, 1] - reference.ideal[1]
        ) / reference.span[1],
    })

    manifest_frame = pd.DataFrame([{
        **context,
        **reference_meta,
        "Runs_Per_Configuration": runs,
        "Seed_Start": seed_start,
        "FE_Budgets": json.dumps(fe_budgets),
        "Configurations": len(variants),
        "Tasks": len(tasks),
        "Parallel_Processes": actual_workers,
        "Requested_Processes": workers,
        "Pymoo_Version": pymoo_version,
        "Input_Excel": str(path),
        "Crossover_Prob": NSGA2_MA_CROSSOVER_PROBABILITY,
        "Mutation_Prob": NSGA2_MA_MUTATION_PROBABILITY,
        "Use_Cache": True,
        "Eliminate_Duplicates": False,
        "Termination_Strategy": "exact_n_eval",
        "Factor_Anchor_Local_FE_Ratio": 0.2,
        "Selection_Rule": selection["selection_rule"],
        "FE_Definition": (
            "Number of individuals actually submitted to the evaluator, including problem-cache hits; "
            "direct reuse of known individuals in the current generation adds no FE"
        ),
    }])

    variant_rows = []
    for variant in variants:
        row = asdict(variant)
        row["local_fe_ratio_code"] = (
            UNLIMITED_RATIO_CODE
            if variant.max_local_fe_ratio is None
            else variant.max_local_fe_ratio
        )
        variant_rows.append(row)

    return {
        "runs.csv": runs_frame,
        "history.csv": history_frame,
        "final_decisions.csv": decisions_frame,
        "summary.csv": summary_frame,
        "ranking.csv": ranking_frame,
        "reference_front.csv": reference_frame,
        "experiment_manifest.csv": manifest_frame,
        "variants.csv": pd.DataFrame(variant_rows),
        "selected_ma": selection,
        "input_excel": str(path),
    }


def main() -> None:
    
    parser = argparse.ArgumentParser(
        description="MA ablation experiments with strict FE budgets"
    )
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument(
        "--pop-size", type=int, default=None
    )
    parser.add_argument(
        "--fe-budgets", type=int, nargs="+", default=None
    )
    parser.add_argument(
        "--seed-start", type=int, default=None
    )
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument(
        "--game-iterations",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--game-learning-rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--input-excel", type=Path, default=None
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANT_BY_NAME),
        default=None,
    )
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    from common.pipeline import run_solve
    from features.macomp.export_csv import export

    selected_variants = (
        VARIANTS
        if args.variants is None
        else tuple(VARIANT_BY_NAME[name] for name in args.variants)
    )

    result = run_solve(
        "macomp",
        solve,
        export,
        runs=args.runs,
        pop_size=args.pop_size,
        fe_budgets=args.fe_budgets,
        seed_start=args.seed_start,
        budget=args.budget,
        game_iterations=args.game_iterations,
        game_learning_rate=args.game_learning_rate,
        input_excel=args.input_excel,
        workers=args.workers,
        variants=selected_variants,
    )

    document = result["selected_ma"]
    print(
        f"✅ Selected configuration: {document['experiment_name']}; "
        f"best nonzero local-search configuration: {document['best_ma_experiment']}"
    )

    if document["selected_is_baseline"]:
        print(
            "Note: the baseline ranked first in this experiment, so selected_ma.json disables local search; "
            "a worse-performing MA was not forced into use."
        )

    if not args.skip_plots:
        from features.macomp.plot import plot
        plot(input_excel=result.get("input_excel"))


if __name__ == "__main__":
    main()
