

from __future__ import annotations

import os

THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

if __name__ == "__main__":
    for thread_variable in THREAD_VARIABLES:
        os.environ[thread_variable] = "1"
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import combinations
import json
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
import traceback
from uuid import uuid4

import numpy as np
import pandas as pd
from pymoo.optimize import minimize
from scipy.stats import mannwhitneyu

from common.case_config import get_input_excel, load_case_parameters
from common.config import NSGA2_MA_SEED, get_feature_output_dir
from common.implementation import IMPLEMENTATION_VERSION, MODEL_PROTOCOL_VERSION
from common.memetic import get_selected_ma_file, load_selected_ma_parameters, sha256_file
from common.problem_factory import load_problem_context
from common.simulation import resolve_prune_tolerance
from common.termination import MyTerminationSelector as MyTerminationSelector
from features.multi_algorithm.fronts import build_empirical_reference
from features.multi_algorithm.execution import run_task


ALGORITHMS = ("NSGA2_MA", "NSGA2", "SPEA2", "MOEA/D")
PROTOCOLS = ("equal_fe", "equal_wall_clock", "equal_termination")
COMPARISON_PROTOCOL_VERSION = "three-protocol-strict-front-v2"
WORKERS = 1
_WORKER_CONTEXT = None
GROUP_COLUMNS = ["Experiment_ID", "Protocol", "Budget_ID"]
EXTRA_SUMMARY_METRICS = (
    "Genetic_FE", "Local_Accepted", "Interrupted_Local_Rounds", "Neighbor_Scans",
    "Local_Improved_Center", "Local_Entered_Final_Front", "Local_Dominates_Center",
    "Final_Refill_Genetic_FE", "Cache_Hits", "Unique_Simulations",
)


def _initialize_worker(input_excel, expected_sha256=None):
    global _WORKER_CONTEXT
    if expected_sha256 is not None and sha256_file(input_excel) != expected_sha256:
        raise RuntimeError("The input workbook changed during the experiment")
    with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink):
        _WORKER_CONTEXT = load_problem_context(input_excel)


def _legacy_task(task):
    keys = ("name", "run", "seed", "pop_size", "max_fe", "budget", "ma_parameters",
            "local_fe_reference", "strategy", "max_gen", "tol", "n_stable", "period",
            "game_iterations", "game_learning_rate")
    values = dict(zip(keys, task))
    values.update(protocol="equal_fe" if values["strategy"] == "exact_n_eval" else "equal_termination", legacy=True)
    return values


def _run_one_silent(task):
    if _WORKER_CONTEXT is None:
        raise RuntimeError("The problem context has not been initialized")
    return run_task(_legacy_task(task) if isinstance(task, tuple) else task, _WORKER_CONTEXT, minimize_fn=minimize)


def _run_one(task):
    try:
        with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink):
            return _run_one_silent(task)
    except Exception as exc:
        name, run, seed = task[:3] if isinstance(task, tuple) else (task["name"], task["run"], task["seed"])
        raise RuntimeError(f"multi_algorithm task failed: algorithm={name}, run={run}, seed={seed}") from exc


def _comparison_mask(metrics):
    valid = metrics["Status"].eq("ok")
    if "Comparison_Valid" in metrics:
        valid &= metrics["Comparison_Valid"].fillna(False).eq(True)
    return valid


def _runtime_statistics(metrics):
    rows = []
    for keys, group in metrics.groupby([*GROUP_COLUMNS, "Algorithm"], sort=False):
        valid = group.loc[_comparison_mask(group)]
        attempts = pd.to_numeric(group["Runtime_Seconds"], errors="coerce")
        values = pd.to_numeric(valid["Runtime_Seconds"], errors="coerce")
        row = dict(zip([*GROUP_COLUMNS, "Algorithm"], keys))
        row.update(Attempts=len(group), Valid_Runs=len(valid), Invalid_Runs=len(group) - len(valid))
        for label, operation in (("Mean", "mean"), ("Std", "std"), ("Min", "min"), ("Max", "max")):
            row[f"{label}_Seconds"] = getattr(values, operation)()
            row[f"All_Attempts_{label}_Seconds"] = getattr(attempts, operation)()
        row["Sum_Attempt_Run_Wall_Seconds"] = attempts.sum(min_count=1)
        row["Sum_Valid_Run_Wall_Seconds"] = values.sum(min_count=1)
        rows.append(row)
    return pd.DataFrame(rows)


def _summary(metrics):
    names = ("HV", "IGD", "GD", "Runtime_Seconds", "Run_Wall_Seconds", "Pareto_Size",
             "Generations", "Total_FE", "Local_FE", "Initialization_Seconds", "Finalize_Seconds",
             *EXTRA_SUMMARY_METRICS)
    rows = []
    for keys, group in metrics.groupby([*GROUP_COLUMNS, "Algorithm"], sort=False):
        valid = group.loc[_comparison_mask(group)]
        row = dict(zip([*GROUP_COLUMNS, "Algorithm"], keys))
        row.update(Runs=len(group), Valid_Runs=len(valid), Failed_Runs=len(group) - len(valid))
        row["Solve_Failed_Runs"] = int(group["Status"].ne("ok").sum())
        row["Protocol_Incomplete_Runs"] = int((group["Status"].eq("ok") & ~_comparison_mask(group)).sum())
        for name in names:
            values = pd.to_numeric(valid.get(name, pd.Series(dtype=float)), errors="coerce")
            row[f"{name}_Mean"] = values.mean()
            row[f"{name}_Std"] = values.std(ddof=1)
            row[f"{name}_Valid_Runs"] = int(values.notna().sum())
            row[f"{name}_Invalid_Runs"] = len(group) - int(values.notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _statistical_tests(metrics):
    rows = []
    for keys, group in metrics.groupby(GROUP_COLUMNS, sort=False):
        for metric in ("HV", "IGD", "GD", "Runtime_Seconds"):
            for first, second in combinations(ALGORITHMS, 2):
                samples = [pd.to_numeric(group.loc[(group["Algorithm"] == name) & _comparison_mask(group), metric], errors="coerce").dropna().to_numpy() for name in (first, second)]
                statistic = p_value = np.nan
                status = "insufficient_samples"
                if min(map(len, samples)) >= 2:
                    result = mannwhitneyu(*samples, alternative="two-sided")
                    statistic, p_value, status = float(result.statistic), float(result.pvalue), "ok"
                rows.append({**dict(zip(GROUP_COLUMNS, keys)), "Metric": metric,
                             "Algorithm_A": first, "Algorithm_B": second,
                             "N_A": len(samples[0]), "N_B": len(samples[1]),
                             "Statistic_U": statistic, "P_Value_Unadjusted": p_value, "Status": status,
                             "Statistical_Design": "independent_samples; matching seed labels do not establish pairing"})
    return pd.DataFrame(rows)


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value) or int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
    return int(value)


def solve(runs=None, pop_size=None, max_gen=None, budget=None, max_fe=None, *,
          strategy="exact_n_eval", tol=None, n_stable=None, period=None, workers=None,
          game_iterations=None, game_learning_rate=None, input_excel=None,
          protocols=None, protocol=None, fe_budgets=None, wall_seconds=None,
          loss_scale=None, min_gen=None, min_front_size=None, ref_point=None,
          termination_local_fe_reference=None, seed_start=NSGA2_MA_SEED,
          representative_seed=None, experiment_id=None, output_dir=None):
    started = perf_counter()
    input_excel = get_input_excel(input_excel)
    legacy = protocols is None and protocol is None
    if protocols is not None and protocol is not None:
        raise ValueError("Provide either protocols or protocol, not both")
    selected_protocols = protocols if protocols is not None else protocol
    if selected_protocols is None:
        selected_protocols = ("equal_fe" if strategy in (None, "exact_n_eval") else "equal_termination",)
    if isinstance(selected_protocols, str):
        selected_protocols = PROTOCOLS if selected_protocols == "all" else (selected_protocols,)
    selected_protocols = tuple(selected_protocols)
    if not selected_protocols or len(set(selected_protocols)) != len(selected_protocols) or set(selected_protocols) - set(PROTOCOLS):
        raise ValueError(f"protocols must be selected from {PROTOCOLS} without duplicates")
    if strategy not in (None, "exact_n_eval", "convergence", "combined_or", "combined_and", "robust", "max_gen"):
        raise ValueError(f"Unknown stopping strategy: {strategy}")
    if ref_point is not None and (np.asarray(ref_point).shape != (2,) or not np.isfinite(np.asarray(ref_point, dtype=float)).all()):
        raise ValueError("ref_point must contain two finite numbers")
    names = ["Use_Pruning", "Budget", "Multi_Algorithm_Runs", "Multi_Algorithm_Population_Size", "Multi_Algorithm_Default_Max_Generations",
             "Multi_Algorithm_Reference_Loss", "Multi_Algorithm_Reference_Cost_Factor", "Game_Iterations", "Game_Learning_Rate"]
    front_mode = "equal_termination" in selected_protocols and not legacy
    if front_mode or ("equal_termination" in selected_protocols and strategy != "max_gen"):
        names += ["Termination_Tolerance", "Termination_Stable_Generations", "Termination_Robust_Period"]
    if front_mode:
        names += ["Termination_Loss_Scale", "Termination_Min_Generations", "Termination_Min_Front_Size", "Termination_HV_Reference_Cost", "Termination_HV_Reference_Loss"]
    overrides = {name: value for name, value in (
        ("Budget", budget), ("Multi_Algorithm_Runs", runs), ("Multi_Algorithm_Population_Size", pop_size),
        ("Multi_Algorithm_Default_Max_Generations", max_gen), ("Game_Iterations", game_iterations),
        ("Game_Learning_Rate", game_learning_rate), ("Termination_Tolerance", tol),
        ("Termination_Stable_Generations", n_stable), ("Termination_Robust_Period", period),
        ("Termination_Loss_Scale", loss_scale), ("Termination_Min_Generations", min_gen),
        ("Termination_Min_Front_Size", min_front_size),
        ("Termination_HV_Reference_Cost", ref_point[0] if ref_point is not None else None),
        ("Termination_HV_Reference_Loss", ref_point[1] if ref_point is not None else None),
    ) if value is not None and name in names}
    parameters = load_case_parameters(input_excel, names=names, include_termination=front_mode, overrides=overrides)
    runs, pop_size, max_gen = (parameters[name] for name in ("Multi_Algorithm_Runs", "Multi_Algorithm_Population_Size", "Multi_Algorithm_Default_Max_Generations"))
    budget, game_iterations, game_learning_rate = (parameters[name] for name in ("Budget", "Game_Iterations", "Game_Learning_Rate"))
    baseline_max_gen = load_case_parameters(
        input_excel,
        names=["Multi_Algorithm_Default_Max_Generations"],
    )["Multi_Algorithm_Default_Max_Generations"]
    prune_tol = resolve_prune_tolerance(parameters["Use_Pruning"])
    workers = _integer(1 if workers is None else workers, "workers")
    seed_start = _integer(seed_start, "seed_start", 0)
    if seed_start + runs - 1 > np.iinfo(np.uint32).max:
        raise ValueError("seed is outside the NumPy uint32 range")
    representative_seed = seed_start if representative_seed is None else _integer(representative_seed, "representative_seed", 0)
    if representative_seed not in range(seed_start, seed_start + runs):
        raise ValueError("representative_seed must be among the seeds of the current paired runs")
    if max_fe is not None:
        max_fe = _integer(max_fe, "max_fe", pop_size)
    if termination_local_fe_reference is not None:
        if "equal_termination" not in selected_protocols:
            raise ValueError("termination_local_fe_reference applies only to equal_termination")
        termination_local_fe_reference = _integer(
            termination_local_fe_reference,
            "termination_local_fe_reference",
            pop_size,
        )
    if front_mode:
        if parameters["Termination_Loss_Scale"] is None:
            raise ValueError("equal_termination requires a calibrated loss_scale")
        if parameters.get("Termination_Robust_Period", 1) != 1:
            raise ValueError("equal_termination front mode requires period=1")
    if "equal_wall_clock" in selected_protocols:
        if workers != 1:
            raise ValueError("equal_wall_clock runs serially and requires workers=1")
        if wall_seconds is None:
            raise ValueError("equal_wall_clock requires explicit wall_seconds")
        if any(os.environ.get(name) != "1" for name in THREAD_VARIABLES):
            raise ValueError("equal_wall_clock requires numerical-library thread environment variables set to 1 before Python starts; the module CLI sets them before importing numerical libraries")
    raw_fe = (max_fe if max_fe is not None else pop_size * max_gen,) if fe_budgets is None else tuple(fe_budgets)
    fe_values = tuple(_integer(value, "fe_budgets", pop_size) for value in raw_fe)
    if "equal_fe" in selected_protocols and (not fe_values or list(fe_values) != sorted(set(fe_values))):
        raise ValueError("fe_budgets must be nonempty and strictly increasing")
    time_values = () if wall_seconds is None else (wall_seconds,) if np.isscalar(wall_seconds) else tuple(wall_seconds)
    if "equal_wall_clock" in selected_protocols and not time_values:
        raise ValueError("wall_seconds must not be empty for equal_wall_clock")
    if any(isinstance(value, bool) or not np.isfinite(value) or value <= 0 for value in time_values):
        raise ValueError("wall_seconds must be positive and finite")
    if len(set(time_values)) != len(time_values):
        raise ValueError("wall_seconds must not contain duplicates")
    if termination_local_fe_reference is not None:
        termination_local_fe_reference_source = "explicit_termination_local_fe_reference"
    elif max_fe is not None:
        termination_local_fe_reference = max_fe
        termination_local_fe_reference_source = "max_fe"
    else:
        termination_local_fe_reference = pop_size * baseline_max_gen
        termination_local_fe_reference_source = "pop_size_times_workbook_max_gen"
    selection_parameters, selection_metadata = load_selected_ma_parameters(
        budget=budget, pop_size=None, game_iterations=game_iterations,
        game_learning_rate=game_learning_rate, input_excel=input_excel)
    selection_path = get_selected_ma_file(input_excel)
    selection_hash, input_hash = sha256_file(selection_path), sha256_file(input_excel)
    experiment_id = experiment_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid4().hex[:8]
    if Path(experiment_id).name != experiment_id or experiment_id in (".", ".."):
        raise ValueError("experiment_id must be a single directory name")
    directory = Path(output_dir).resolve() if output_dir is not None else get_feature_output_dir("multi_algorithm", input_excel) / "experiments" / experiment_id
    if directory.exists():
        raise FileExistsError(f"The experiment directory already exists; use a new experiment_id/output_dir: {directory}")
    groups = []
    for selected_protocol in selected_protocols:
        if selected_protocol == "equal_fe":
            groups.extend((selected_protocol, f"fe_{value}", value, None) for value in fe_values)
        elif selected_protocol == "equal_wall_clock":
            groups.extend((selected_protocol, "wall_" + repr(float(value)).replace(".", "p") + "s", max_fe, float(value)) for value in time_values)
        else:
            groups.append((selected_protocol, f"front_g{max_gen}_fe{max_fe if max_fe is not None else 'none'}", max_fe, None))
    tasks = []

    def local_reference_for_protocol(selected_protocol, fe_limit):
        if selected_protocol in ("equal_fe", "equal_wall_clock") and fe_limit is not None:
            return fe_limit, "max_fe"
        if selected_protocol == "equal_termination":
            return termination_local_fe_reference, termination_local_fe_reference_source
        return pop_size * baseline_max_gen, "pop_size_times_workbook_max_gen"

    for selected_protocol, budget_id, fe_limit, seconds in groups:
        task_local_fe_reference, task_local_fe_reference_source = local_reference_for_protocol(
            selected_protocol, fe_limit,
        )
        for run in range(1, runs + 1):
            for name in ALGORITHMS:
                tasks.append(dict(name=name, run=run, seed=seed_start + run - 1, protocol=selected_protocol,
                    budget_id=budget_id, pop_size=pop_size, max_fe=fe_limit, budget=budget,
                    ma_parameters=selection_parameters, local_fe_reference=task_local_fe_reference,
                    local_fe_reference_source=task_local_fe_reference_source,
                    strategy=(strategy or "exact_n_eval") if legacy else {"equal_fe": "exact_n_eval", "equal_wall_clock": "wall_clock", "equal_termination": "front"}[selected_protocol], legacy=legacy,
                    max_gen=max_gen, tol=parameters.get("Termination_Tolerance", 0.01),
                    n_stable=parameters.get("Termination_Stable_Generations", 5), period=parameters.get("Termination_Robust_Period", 1),
                    loss_scale=parameters.get("Termination_Loss_Scale"), min_gen=parameters.get("Termination_Min_Generations", 20),
                    min_front_size=parameters.get("Termination_Min_Front_Size", 2),
                    ref_point=(parameters.get("Termination_HV_Reference_Cost") or 1.1, parameters.get("Termination_HV_Reference_Loss") or 1.1),
                    game_iterations=game_iterations, game_learning_rate=game_learning_rate,
                    prune_tol=prune_tol, target_seconds=seconds))
    print(f"[multi_algorithm] {input_excel.name}: {len(tasks)} runs; protocols={selected_protocols}; workers={workers}", flush=True)
    loading_started = perf_counter()
    if workers == 1:
        _initialize_worker(input_excel, input_hash)
    common_loading_seconds = perf_counter() - loading_started
    outputs = []

    def collect(task, operation):
        try:
            entry = operation()
        except Exception as error:
            entry = {"Algorithm": task["name"], "Run": task["run"], "Seed": task["seed"], "Status": "failed",
                     "Error": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
                     "Front": np.empty((0, 2)), "History": [], "Decisions": []}
        entry.update(Experiment_ID=experiment_id, Protocol=task["protocol"], Budget_ID=task["budget_id"],
                     Target_FE=task["max_fe"] if task["protocol"] == "equal_fe" else None,
                     Safety_Max_FE=task["max_fe"], Target_Wall_Seconds=task["target_seconds"],
                     Safety_Max_Generations=task["max_gen"] if task["protocol"] != "equal_fe" else None,
                     Local_FE_Reference=task["local_fe_reference"],
                     Local_FE_Reference_Source=task["local_fe_reference_source"])
        if entry.get("Status") == "ok" and not len(entry["Front"]):
            entry["Status"] = "no_feasible_front"
        completed = entry.get("Protocol_Completed", entry.get("Status") == "ok")
        if task["protocol"] == "equal_wall_clock":
            completed = bool(entry.get("Wall_Budget_Completed", False))
        entry["Protocol_Completed"] = bool(completed)
        entry["Comparison_Valid"] = entry.get("Status") == "ok" and bool(completed)
        outputs.append(entry)
        print(f"[{len(outputs)}/{len(tasks)}] {task['protocol']}/{task['budget_id']} {task['name']} seed={task['seed']} status={entry.get('Status')} completed={completed} FE={entry.get('Total_FE')}", flush=True)

    if workers == 1:
        for task in tasks:
            collect(task, lambda current=task: _run_one(current))
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks)), mp_context=get_context("spawn"), initializer=_initialize_worker, initargs=(str(input_excel), input_hash)) as pool:
            futures = {pool.submit(_run_one, task): task for task in tasks}
            for future in as_completed(futures):
                collect(futures[future], future.result)
    if sha256_file(input_excel) != input_hash or sha256_file(selection_path) != selection_hash:
        raise RuntimeError("The workbook or selection file changed during the experiment; export refused")
    outputs.sort(key=lambda row: (row["Protocol"], row["Budget_ID"], row["Run"], ALGORITHMS.index(row["Algorithm"])))
    offline_started = perf_counter()
    observed = [row["Front"] for row in outputs]
    for row in outputs:
        observed.append(row.get("Main_Loop_Front", np.empty((0, 2))))
        observed.extend(step["Front"] for step in row["History"] if "Front" in step)
        observed.extend(row.get("Observed_Fronts", []))
    reference = build_empirical_reference(observed, budget, parameters["Multi_Algorithm_Reference_Cost_Factor"], parameters["Multi_Algorithm_Reference_Loss"])
    context = {
        "Input_Excel": str(input_excel), "Excel_SHA256": input_hash, "Budget": budget, "Population_Size": pop_size,
        "Game_Iterations": game_iterations, "Game_Learning_Rate": game_learning_rate, "Pymoo_Version": version("pymoo"),
        "Implementation_Version": IMPLEMENTATION_VERSION, "Model_Protocol_Version": MODEL_PROTOCOL_VERSION,
        "Comparison_Protocol_Version": COMPARISON_PROTOCOL_VERSION, "Selection_Path": str(selection_path),
        "Selection_SHA256": selection_hash, "Front_Kind": "strict", "Requested_Processes": workers,
        "Actual_Processes": min(workers, len(tasks)),
        "Thread_Configuration_JSON": json.dumps({name: os.environ.get(name, "unset") for name in THREAD_VARIABLES}, sort_keys=True),
        "Representative_Seed": representative_seed, "Representative_Selection_Rule": "fixed_seed_before_running",
        "Comparison_Validity_Rule": "solve ok and protocol completed; wall-clock requires deadline completion without earlier safety cap",
        "Statistical_Design": "independent_samples; Mann-Whitney U, two-sided, unadjusted p-values",
        **selection_metadata, **reference.metadata(),
    }
    metric_rows, history_rows, point_rows = [], [], []
    excluded = ("Front", "Display_Front", "History", "Decisions", "Display_Decisions", "Observed_Fronts", "Main_Loop_Front", "Main_Loop_Decisions")
    for entry in outputs:
        identifiers = {name: entry[name] for name in (*GROUP_COLUMNS, "Algorithm", "Run", "Seed")}
        identifiers.update(Status=entry["Status"], Comparison_Valid=entry["Comparison_Valid"], Protocol_Completed=entry["Protocol_Completed"])
        values = {name: value for name, value in entry.items() if name not in excluded}
        values["Termination_Parameters_JSON"] = json.dumps(entry.get("Termination_Parameters", {}), sort_keys=True)
        values.update(context)
        values.update(reference.evaluate(entry["Front"]))
        values["Pareto_Size"] = len(entry["Front"])
        if values.get("Status") == "ok" and not len(entry["Front"]):
            values["Status"] = "no_feasible_front"
        metric_rows.append(values)
        for step in entry["History"]:
            front = step.get("Front", np.empty((0, 2)))
            history_rows.append({**identifiers, **{name: value for name, value in step.items() if name not in excluded},
                **reference.metadata(), **reference.evaluate(front), "Front_Kind": "strict", "Pareto_Size": len(front), "Feasible_Pareto_Size": len(front)})
        stages = [("final", entry.get("Generation", len(entry["History"])), entry)]
        if "Main_Loop_Front" in entry:
            stages.append(("main_loop", len(entry["History"]), {"Front": entry["Main_Loop_Front"], "Decisions": entry["Main_Loop_Decisions"]}))
        stages.extend(("generation", step["Generation"], step) for step in entry["History"])
        for stage, generation, snapshot in stages:
            for kind, field, decision_field in (("strict", "Front", "Decisions"), ("display_compatible", "Display_Front", "Display_Decisions")):
                for point_id, point in enumerate(snapshot.get(field, []), start=1):
                    decision = snapshot.get(decision_field, [])[point_id - 1]
                    point_rows.append({**identifiers, "Stage": stage, "Generation": generation, "Front_Kind": kind, "Point_ID": point_id,
                        "Total_Cost": float(point[0]), "Total_Defender_Loss": float(point[1]), **decision})
    metrics = pd.DataFrame(metric_rows)
    for name in ("HV", "IGD", "GD", "Runtime_Seconds", "Run_Wall_Seconds", "Total_FE", "Local_FE", "Generations"):
        if name not in metrics:
            metrics[name] = np.nan
    points = pd.DataFrame(point_rows, columns=[*GROUP_COLUMNS, "Algorithm", "Run", "Seed", "Status", "Comparison_Valid", "Protocol_Completed", "Stage", "Generation", "Front_Kind", "Point_ID", "Total_Cost", "Total_Defender_Loss", "X_Bits", "P_Response", "X_JSON"])
    representatives = points.loc[(points["Seed"] == representative_seed) & (points["Stage"] == "final")].copy()
    representatives["Selection_Rule"] = "fixed_seed_before_running"
    runtime = _runtime_statistics(metrics)
    summary, tests = _summary(metrics), _statistical_tests(metrics)
    offline_seconds, experiment_seconds = perf_counter() - offline_started, perf_counter() - started
    manifest = []
    for selected_protocol, budget_id, fe_limit, seconds in groups:
        group = metrics.loc[(metrics["Protocol"] == selected_protocol) & (metrics["Budget_ID"] == budget_id)]
        valid_group = group.loc[_comparison_mask(group)]
        missing_algorithms = [name for name in ALGORITHMS if not valid_group["Algorithm"].eq(name).any()]
        group_status = "failed" if missing_algorithms else "partial_failure" if len(valid_group) != len(group) else "ok"
        manifest.append({"Experiment_ID": experiment_id, "Protocol": selected_protocol, "Budget_ID": budget_id, **context,
            "Target_FE": fe_limit if selected_protocol == "equal_fe" else None, "Safety_Max_FE": fe_limit, "Target_Wall_Seconds": seconds,
            "Safety_Max_Generations": max_gen if selected_protocol != "equal_fe" else None,
            "Max_Generations": max_gen, "Termination_Parameters_JSON": json.dumps({name: parameters.get(name) for name in names if name.startswith("Termination_")}, sort_keys=True),
            "Readiness_Rule": "all_algorithms_ready; no MA-only stability gate" if not legacy else "legacy_termination_compatibility",
            "MA_Parameters_JSON": json.dumps(selection_parameters, sort_keys=True),
            "Local_FE_Reference": local_reference_for_protocol(selected_protocol, fe_limit)[0],
            "Local_FE_Reference_Source": local_reference_for_protocol(selected_protocol, fe_limit)[1],
            "Max_Local_FE_Ratio": selection_parameters.get("max_local_fe_ratio"),
            "Max_Local_FE_Per_Generation": selection_parameters.get("max_local_fe_per_gen"),
            "Runs": runs, "Seed_Start": seed_start, "Algorithms_JSON": json.dumps(ALGORITHMS),
            "Valid_Runs": len(valid_group), "Failed_Runs": len(group) - len(valid_group),
            "Solve_Failed_Runs": int(group["Status"].ne("ok").sum()),
            "Protocol_Incomplete_Runs": int((group["Status"].eq("ok") & ~_comparison_mask(group)).sum()),
            "Group_Status": group_status, "Missing_Algorithms_JSON": json.dumps(missing_algorithms),
            "Experiment_Wall_Seconds": experiment_seconds, "Sum_Run_Wall_Seconds": metrics["Run_Wall_Seconds"].sum(min_count=1),
            "Sum_Valid_Run_Wall_Seconds": metrics.loc[_comparison_mask(metrics), "Run_Wall_Seconds"].sum(min_count=1),
            "Runtime_Statistics_Scope": "legacy runtime means use valid comparisons; All_Attempts columns retain all recorded attempts",
            "Common_Loading_Seconds": common_loading_seconds, "Offline_Metrics_Seconds": offline_seconds,
            "Common_Loading_Scope": "parent context load" if workers == 1 else "parent only; worker context loading included in experiment wall time",
            "Thread_Control": "CLI fixes numerical-library thread environment before imports; API uses recorded caller environment",
            "Experiment_Timing_Scope": "solve including configuration/loading/runs/offline metrics; export and plot timed separately",
            "FE_Definition": "Number of individuals actually submitted to the evaluator, including problem-cache hits; direct reuse of known individuals adds no FE"})
    reference_frame = reference.frame()
    reference_frame.insert(0, "Experiment_ID", experiment_id)
    experiment_status = "failed" if any(row["Group_Status"] == "failed" for row in manifest) else "partial_failure" if not _comparison_mask(metrics).all() else "ok"
    return {"input_excel": str(input_excel), "experiment_id": experiment_id, "output_dir": str(directory), "experiment_status": experiment_status,
        "performance_metrics_raw": metrics, "performance_summary": summary, "all_pareto_points": points,
        "best_pareto_fronts": representatives, "runtime_statistics": runtime, "statistical_tests": tests,
        "optimization_history": pd.DataFrame(history_rows), "reference_front": reference_frame,
        "experiment_manifest": pd.DataFrame(manifest)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare algorithms under equal FE, runtime or front-stopping rules")
    parser.add_argument("--input-excel", type=Path)
    parser.add_argument("--protocols", nargs="+", choices=(*PROTOCOLS, "all"))
    parser.add_argument("--fe-budgets", nargs="+", type=int)
    parser.add_argument("--wall-seconds", nargs="+", type=float)
    for name in ("runs", "pop-size", "max-gen", "max-fe", "workers", "seed-start", "representative-seed", "n-stable", "min-gen", "min-front-size", "period", "game-iterations"):
        parser.add_argument("--" + name, type=int, default=None)
    parser.add_argument(
        "--termination-local-fe-reference", "--local-fe-reference",
        dest="termination_local_fe_reference", type=int, default=None,
    )
    for name in ("budget", "tol", "loss-scale", "game-learning-rate"):
        parser.add_argument("--" + name, type=float, default=None)
    parser.add_argument("--strategy", choices=("exact_n_eval", "convergence", "combined_or", "combined_and", "robust", "max_gen"), default="exact_n_eval")
    parser.add_argument("--experiment-id")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-plots", action="store_true")
    arguments = vars(parser.parse_args(argv))
    skip_plots = arguments.pop("skip_plots")
    if arguments.get("protocols") == ["all"]:
        arguments["protocols"] = PROTOCOLS
    if arguments["seed_start"] is None:
        arguments.pop("seed_start")
    from features.multi_algorithm.export_csv import export
    result = solve(**arguments)
    export(result)
    metrics = result["performance_metrics_raw"]
    valid_count = int(_comparison_mask(metrics).sum())
    print(f"Comparison validation: {result['experiment_status']}; valid={valid_count}; invalid={len(metrics) - valid_count}; solve_failed={int(metrics['Status'].ne('ok').sum())}")
    if result["experiment_status"] == "failed":
        print(f"Required comparison groups lack valid algorithm runs; failure details exported to: {result['output_dir']}")
        return 1
    if not skip_plots:
        from features.multi_algorithm.plot import plot
        plot(output_dir=result["output_dir"])
    print(f"Comparison outputs: {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
