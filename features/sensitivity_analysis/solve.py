

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from uuid import uuid4

from joblib import Parallel, delayed, parallel_config
import numpy as np

from common.case_config import get_input_excel, load_case_parameters
from common.implementation import IMPLEMENTATION_VERSION, MODEL_PROTOCOL_VERSION
from common.problem_factory import load_problem_context
from features.sensitivity_analysis.analysis import analyze_outputs, build_design
from features.sensitivity_analysis.config import PROTOCOL_VERSION, resolve_config
from features.sensitivity_analysis.optimization import (
    evaluate_decision, make_problem, merge_fronts, optimize, select_decision,
)
from features.sensitivity_analysis.parameters import (
    parameter_problem, parameter_records, scenario_model_options, transform_data, validate_parameters,
)
from features.sensitivity_analysis.progress import SolveProgress


MAIN_METRICS = ("Fixed_J", "Fixed_Loss", "Reoptimized_J", "Reoptimized_Loss", "Search_Gap")


def _search(data, graph, config, seed, *, model_options=None):
    try:
        return optimize(data, graph, config, seed, model_options=model_options)
    except Exception as error:
        return {"front": [], "metrics": {"Status": "failed", "Total_FE": 0, "Seed": seed,
                "Error": f"{type(error).__name__}: {error}"}}


def _vector(record):
    values = record["Decision_Vector"]
    return json.loads(values) if isinstance(values, str) else list(values)


def _performance(record, prefix):
    return {f"{prefix}_{name}": record.get(name, np.nan) for name in ("Cost", "Loss", "J")}


def _rank_candidates(records):
    feasible = sorted(
        (row for row in records if row.get("Feasible") and row.get("Status") == "ok"),
        key=lambda row: (row["J"], row["Cost"], row["Loss"], row["Decision_ID"]),
    )
    for rank, row in enumerate(feasible, start=1):
        row["Candidate_Rank"] = rank
        row["Candidate_Nondominated"] = not any(
            other["Cost"] <= row["Cost"] and other["Loss"] <= row["Loss"]
            and (other["Cost"] < row["Cost"] or other["Loss"] < row["Loss"])
            for other in feasible
        )
    return feasible


def _evaluate_scenario(original, graph, config, factors, keys, fixed, candidates, searches=None):
    metrics = {"Record_Type": "sample", **keys, "Status": "ok"}
    metrics.update({name: np.nan for name in MAIN_METRICS})
    runs, fronts, evaluations, envelopes = [], [], [], []
    errors = []
    try:
        data, _ = transform_data(original, factors)
        model_options = scenario_model_options(factors)
        metrics["Exposure_Penalty_Ratio"] = model_options.exposure_penalty_ratio
        metrics["Response_Penalty_Ratio"] = model_options.response_penalty_ratio
        problem = make_problem(data, graph, config, model_options=model_options)
    except Exception as error:
        metrics.update(Status="failed_scenario", Error=f"{type(error).__name__}: {error}")
        return dict(metrics=metrics, runs=runs, fronts=fronts, evaluations=evaluations,
                    envelopes=envelopes, best=None)

    def evaluate(record, role):
        try:
            evaluated = evaluate_decision(problem, _vector(record), config.lambda_loss)
            evaluated.update(keys, Evaluation_Role=role, Status="ok")
        except Exception as error:
            evaluated = {**keys, "Decision_ID": record.get("Decision_ID"),
                         "Evaluation_Role": role, "Status": "failed_evaluation",
                         "Error": f"{type(error).__name__}: {error}"}
            errors.append(evaluated["Error"])
        evaluations.append(evaluated)
        return evaluated

    fixed_evaluation = evaluate(fixed, "fixed_baseline")
    fixed_count = problem.total_evals
    if fixed_evaluation["Status"] == "ok":
        metrics.update(_performance(fixed_evaluation, "Fixed"))
        metrics.update(Fixed_Feasible=fixed_evaluation["Feasible"],
                       Fixed_Over_Budget=max(0.0, fixed_evaluation["Constraint"]))

    for position, seed in enumerate(config.optimization_seeds):
        result = (_search(data, graph, config, seed, model_options=model_options)
                  if searches is None else searches[position])
        runs.append({**keys, "Record_Type": "optimization_run", "Seed": seed, **result["metrics"]})
        fronts.extend({**record, **keys, "Seed": seed, "Front_Role": "seed"}
                      for record in result["front"])
    complete = all(row["Status"] == "ok" for row in runs)
    merged = merge_fronts([fronts])
    fronts.extend({**record, **keys, "Seed": None, "Front_Role": "merged"} for record in merged)
    best = select_decision(merged) if merged and complete else None
    if best is not None:
        metrics.update(_performance(best, "Reoptimized"))
        metrics.update(Resource_ID=best["Resource_ID"], Decision_ID=best["Decision_ID"],
                       Reoptimized_Feasible=best["Feasible"],
                       Selected_Defenses=best["Selected_Defenses"])
    else:
        errors.extend(row.get("Error", "no_feasible_front") for row in runs if row["Status"] != "ok")
        metrics["Status"] = "failed_optimization"

    candidate_evaluations = [evaluate(candidate, "nominal_candidate") for candidate in candidates]
    ranked = _rank_candidates(candidate_evaluations)
    candidate_count = problem.total_evals - fixed_count
    metrics["Candidate_Ranking_Status"] = (
        "ok" if all(row["Status"] == "ok" for row in candidate_evaluations) else "incomplete"
    )
    if fixed_evaluation["Status"] == "ok":
        metrics["Baseline_Candidate_Rank"] = next(
            (row.get("Candidate_Rank") for row in ranked
             if row["Decision_ID"] == fixed_evaluation["Decision_ID"]), None)
        if best is not None:
            first = np.asarray(_vector(fixed)[:-1], dtype=bool)
            second = np.asarray(_vector(best)[:-1], dtype=bool)
            union = np.count_nonzero(first | second)
            metrics.update(
                Resource_Reselected=fixed_evaluation["Resource_ID"] == best["Resource_ID"],
                Jaccard=np.count_nonzero(first & second) / union if union else 1.0,
                Hamming=int(np.count_nonzero(first != second)),
                RP_Level_Change=int(best["RP_Level"][-1]) - int(fixed_evaluation["RP_Level"][-1]),
                P_Response_Change=abs(best["P_Response"] - fixed_evaluation["P_Response"]),
            )
            if fixed_evaluation["Feasible"]:
                metrics["Search_Gap"] = fixed_evaluation["J"] - best["J"]
                reference = min([best["J"], fixed_evaluation["J"]] + [row["J"] for row in ranked])
                metrics["Candidate_Reference_Gap"] = fixed_evaluation["J"] - reference

    fallback = evaluate({"Decision_Vector": [0.0] * problem.n_var}, "zero_deployment_fallback")
    for fraction in config.budget_fractions:
        threshold = fraction * config.budget
        options = [record for record in merged if record["Cost"] <= threshold + 1e-8] if complete else []
        if fallback["Status"] == "ok" and fallback["Feasible"] and complete:
            options.append(fallback)
        selected = min(options, key=lambda row: (row["Loss"], row["Cost"], row["Decision_ID"])) if options else None
        envelopes.append({**keys, "Record_Type": "budget_envelope", "Budget_Threshold": threshold,
                          "Budget_Fraction": fraction,
                          "Envelope_Loss": selected["Loss"] if selected else np.nan,
                          "Envelope_Source": ("zero_deployment_fallback" if selected is fallback
                                              else "searched_front" if selected else "unavailable"),
                          "Decision_ID": selected["Decision_ID"] if selected else None,
                          "Status": "ok" if selected else "invalid_design"})

    metrics.update(Fixed_Evaluations=fixed_count, Candidate_Evaluations=candidate_count,
                   Auxiliary_Evaluations=problem.total_evals - fixed_count - candidate_count,
                   Evaluation_Cache_Hits=problem.cache_hits,
                   Evaluation_Simulation_Count=problem.total_evals - problem.cache_hits,
                   Total_FE=sum(row.get("Total_FE", 0) for row in runs))
    if errors:
        metrics["Error"] = " | ".join(errors)
        if metrics["Status"] == "ok":
            metrics["Status"] = "partial_evaluation_failure"
    return dict(metrics=metrics, runs=runs, fronts=fronts, evaluations=evaluations,
                envelopes=envelopes, best=best)


def _raw_strategy_value(data, name):
    strategy = data.get("Strategy", {})
    for parameter, value in zip(strategy.get("Parameter", []), strategy.get("Value", [])):
        if str(parameter).strip().casefold() == name.casefold():
            return value if value is not None else "unknown"
    return "unknown"


def _search_status(metrics):
    message = f"status={metrics['Status']}; FE={metrics.get('Total_FE', 0)}"
    if metrics.get("Error"):
        message += f"; error={metrics['Error']}"
    return message


def _evaluate_samples(original, graph, config, tasks, fixed, candidates, progress):
    progress.samples(0, len(tasks), f"workers={config.workers}; seeds={list(config.optimization_seeds)}")
    results = []

    def receive(result):
        results.append(result)
        metrics = result["metrics"]
        progress.samples(len(results), len(tasks),
                         f"sample={metrics['Sample_ID']} ({metrics['Sample_Role']}); "
                         + _search_status(metrics))

    if config.workers == 1:
        for factors, keys in tasks:
            receive(_evaluate_scenario(original, graph, config, factors, keys, fixed, candidates))
    else:
        with parallel_config(backend="loky", inner_max_num_threads=1):
            with Parallel(n_jobs=config.workers, return_as="generator_unordered", batch_size=1) as pool:
                for result in pool(
                    delayed(_evaluate_scenario)(original, graph, config, factors, keys, fixed, candidates)
                    for factors, keys in tasks
                ):
                    receive(result)
    by_sample = {result["metrics"]["Sample_ID"]: result for result in results}
    return [by_sample[keys["Sample_ID"]] for _, keys in tasks]


def solve(mode="fast", n_samples=None, workers=None, budget=None, game_iterations=None,
          game_learning_rate=None, input_excel=None, *, sensitivity_config=None):
    
    input_excel = get_input_excel(input_excel)
    config = resolve_config(mode, n_samples, workers, sensitivity_config,
                            budget=budget, game_iterations=game_iterations,
                            game_learning_rate=game_learning_rate)
    with SolveProgress(input_excel.stem) as progress:
        return _solve(input_excel, config, progress)


def _solve(input_excel, config, progress):
    progress.log("loading case settings and workbook")
    overrides = {name: value for name, value in (
        ("Sensitivity_Budget", config.budget), ("Game_Iterations", config.game_iterations),
        ("Game_Learning_Rate", config.game_learning_rate),
    ) if value is not None}
    case = load_case_parameters(
        input_excel,
        names=("Use_Pruning", "Sensitivity_Budget", "Game_Iterations", "Game_Learning_Rate"),
        overrides=overrides,
        lenient_names=("Use_Pruning",),
    )
    config = replace(config, budget=case["Sensitivity_Budget"], game_iterations=case["Game_Iterations"],
                     game_learning_rate=case["Game_Learning_Rate"], use_pruning=case["Use_Pruning"])
    _, original, graph = load_problem_context(input_excel)
    validate_parameters(original)
    definition = parameter_problem(config)
    progress.log(f"building Saltelli design: N={config.n_samples}; D={definition['num_vars']}")
    design = build_design(definition, config.n_samples, calc_second_order=config.calc_second_order,
                          sample_seed=config.sampling_seed, scramble=config.scramble)
    workload = config.planned_workload(definition["num_vars"])
    progress.log(f"plan: workers={config.workers}; budget={config.budget}; "
                 f"max_FE/run={config.max_fe}; planned={workload}")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid4().hex[:8]
    common_keys = {"Case_ID": input_excel.stem, "Run_ID": run_id}
    protocol = config.to_dict()
    metadata = {**common_keys, "Protocol_Version": PROTOCOL_VERSION, "Input_Excel": str(input_excel),
                "Input_SHA256": hashlib.sha256(input_excel.read_bytes()).hexdigest(),
                "Config_SHA256": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
                "Configuration": json.dumps(protocol, sort_keys=True), "Planned_Workload": json.dumps(workload),
                "Dimension": definition["num_vars"], "Parameter_Names": json.dumps(definition["names"]),
                "Nominal_Model_Options": json.dumps(asdict(scenario_model_options()), sort_keys=True),
                "Response_Activation": "inferred from selected response effectiveness, preserving baseline semantics",
                "Implementation_Version": IMPLEMENTATION_VERSION, "Model_Protocol_Version": MODEL_PROTOCOL_VERSION,
                "Budget": config.budget, "Main_Budget": _raw_strategy_value(original, "Budget"),
                "Sensitivity_Budget": _raw_strategy_value(original, "Sensitivity_Budget"),
                "Monetary_Unit": "case monetary units; see provenance", "Currency": "unknown",
                "Evaluation_Period": "unknown", "Lambda_Loss": config.lambda_loss,
                "Objective_Assumption": "C and L use the same currency and evaluation period; empirical validation unknown",
                "Optimization_Target": "merged feasible fronts from fixed ordered seed set; approximate algorithm result",
                "Mode_Interpretation": "smoke" if config.mode == "fast" else "configured experiment; assess diagnostics",
                "Dependencies": json.dumps({name: version(name) for name in
                                             ("numpy", "pandas", "SALib", "pymoo", "scipy", "joblib", "matplotlib")})}
    tables = {name: [] for name in ("parameters", "parameter_samples", "sample_metrics", "fronts",
                                   "portfolio_evaluations", "sobol_indices", "critical_parameters", "sobol_diagnostics")}
    tables["parameters"] = parameter_records(original, config, input_excel=input_excel)
    nominal_data, _ = transform_data(original)
    nominal_options = scenario_model_options()
    baseline_searches = []
    for position, seed in enumerate(config.optimization_seeds, start=1):
        label = f"baseline optimization {position}/{config.repeats}: seed={seed}"
        progress.log(label + "; starting (serial)")
        search = _search(nominal_data, graph, config, seed, model_options=nominal_options)
        baseline_searches.append(search)
        progress.log(label + "; " + _search_status(search["metrics"]))
    candidates = merge_fronts([search["front"] for search in baseline_searches])
    if not candidates or any(search["metrics"]["Status"] != "ok" for search in baseline_searches):
        for seed, search in zip(config.optimization_seeds, baseline_searches):
            tables["sample_metrics"].append({**common_keys, "Sample_ID": 0,
                "Sample_Role": "nominal", "Record_Type": "optimization_run", "Seed": seed, **search["metrics"]})
            tables["fronts"].extend({**record, **common_keys, "Sample_ID": 0,
                "Sample_Role": "nominal", "Seed": seed, "Front_Role": "seed"} for record in search["front"])
        metadata.update(Status="baseline_failed", Error="No complete nominal search protocol; no Sobol estimates",
                        Total_Optimization_FE=sum(search["metrics"].get("Total_FE", 0) for search in baseline_searches),
                        Partial_Front_Count=len(tables["fronts"]))
        for records in tables.values():
            for row in records:
                row.update(common_keys)
        progress.log("finished: status=baseline_failed; no complete nominal search; "
                     "sample evaluation and Sobol analysis skipped")
        return {"input_excel": str(input_excel), "metadata": metadata, **tables}
    fixed = select_decision(candidates)
    metadata.update(Baseline_Decision_ID=fixed["Decision_ID"], Baseline_Resource_ID=fixed["Resource_ID"],
                    Nominal_Candidate_Count=len(candidates))
    metadata["Budget_Context"] = (
        "independent sensitivity budget scenario; do not compare fronts at a different budget"
        if metadata["Main_Budget"] != config.budget else "same as main case budget"
    )
    provenance = next((row for row in tables["parameters"] if row.get("Currency") != "unknown"), {})
    metadata["Currency"] = provenance.get("Currency", "unknown")
    metadata["Monetary_Unit"] = provenance.get("Amount_Unit", "unknown")

    def collect(result):
        tables["sample_metrics"].append(result["metrics"])
        tables["sample_metrics"].extend(result["runs"] + result["envelopes"])
        tables["fronts"].extend(result["fronts"])
        tables["portfolio_evaluations"].extend(result["evaluations"])

    nominal_keys = {**common_keys, "Sample_ID": 0, "Sample_Role": "nominal"}
    progress.log(f"evaluating nominal portfolios: candidates={len(candidates)}")
    collect(_evaluate_scenario(original, graph, config, None, nominal_keys,
                               fixed, candidates, searches=baseline_searches))
    tasks = []
    for values, design_row in zip(design["X"], design["rows"]):
        keys = {**common_keys, **design_row}
        factors = dict(zip(definition["names"], values.tolist()))
        options = scenario_model_options(factors)
        tables["parameter_samples"].append({**keys, **factors,
            "Exposure_Penalty_Ratio": options.exposure_penalty_ratio,
            "Response_Penalty_Ratio": options.response_penalty_ratio})
        tasks.append((factors, keys))
    results = _evaluate_samples(original, graph, config, tasks, fixed, candidates, progress)
    progress.log("collecting sample results and preparing Sobol outputs")
    for result in results:
        collect(result)
    outputs = {metric: np.asarray([result["metrics"].get(metric, np.nan) for result in results])
               for metric in MAIN_METRICS}
    for position, fraction in enumerate(config.budget_fractions):
        outputs[f"Envelope_Loss_{fraction:g}B"] = np.asarray([
            result["envelopes"][position]["Envelope_Loss"] if result["envelopes"] else np.nan
            for result in results])
    progress.log(f"Sobol analysis: outputs={len(outputs)}; N_levels={list(config.n_levels)}; "
                 f"bootstrap_resamples={config.bootstrap_resamples}")
    analyzed = analyze_outputs(design, outputs, bootstrap_seed=config.bootstrap_seed,
                               num_resamples=config.bootstrap_resamples, conf_level=config.confidence_level,
                               n_levels=config.n_levels)
    metadata["Sobol_Protocol"] = json.dumps(analyzed["metadata"], sort_keys=True)
    tables["sobol_indices"].extend({**common_keys, **row} for row in analyzed["sobol_indices"])
    tables["sobol_diagnostics"].extend({**common_keys, **row} for row in analyzed["diagnostics"])
    progress.log("Sobol analysis finished; starting optimization noise checks (serial)")
    base_indices = [index for index, row in enumerate(design["rows"]) if row["Sample_Role"] in ("A", "B")]
    noise_completed = 0
    noise_total = workload["Noise_Optimization_Runs"]
    for index in base_indices[:config.noise_scenarios]:
        factors = dict(zip(definition["names"], design["X"][index].tolist()))
        model_options = scenario_model_options(factors)
        keys = {**common_keys, **design["rows"][index],
                "Exposure_Penalty_Ratio": model_options.exposure_penalty_ratio,
                "Response_Penalty_Ratio": model_options.response_penalty_ratio}
        observations = []
        for run in results[index]["runs"]:
            value = run.get("Best_J", np.nan)
            observations.append(value)
            tables["sobol_diagnostics"].append({**run, **keys, "Record_Type": "optimization_noise",
                "Noise_Role": "protocol_seed", "Metric": "J", "Value": value})
        for seed in config.noise_seeds:
            label = (f"noise optimization {noise_completed + 1}/{noise_total}: "
                     f"sample={keys['Sample_ID']}; seed={seed}")
            progress.log(label + "; starting")
            try:
                data, _ = transform_data(original, factors)
                noise = optimize(data, graph, config, seed, model_options=model_options)
            except Exception as error:
                noise = {"metrics": {"Status": "failed", "Total_FE": 0,
                                      "Error": f"{type(error).__name__}: {error}"}}
            noise_completed += 1
            progress.log(label + "; " + _search_status(noise["metrics"]))
            value = noise["metrics"].get("Best_J", np.nan)
            observations.append(value)
            tables["sobol_diagnostics"].append({**keys, "Record_Type": "optimization_noise", "Seed": seed,
                "Noise_Role": "extra_seed", "Metric": "J", "Value": value, **noise["metrics"]})
        valid_noise = np.asarray(observations, dtype=float)
        valid_noise = valid_noise[np.isfinite(valid_noise)]
        tables["sobol_diagnostics"].append({**keys, "Record_Type": "noise_summary", "Metric": "J",
            "Seed_Count": len(valid_noise), "Failed_Seed_Count": len(observations) - len(valid_noise),
            "Mean": float(np.mean(valid_noise)) if len(valid_noise) else np.nan,
            "Standard_Deviation": float(np.std(valid_noise, ddof=1)) if len(valid_noise) > 1 else np.nan,
            "Range": float(np.ptp(valid_noise)) if len(valid_noise) else np.nan,
            "Reference_Merged_J": results[index]["metrics"]["Reoptimized_J"],
            "Status": "ok" if len(valid_noise) == len(observations) else "incomplete"})
    progress.log(f"optimization noise checks finished: {noise_completed}/{noise_total}; "
                 "summarizing decision robustness and FE counts")

    for metric in outputs:
        for base_n in config.n_levels:
            ranked = sorted((row for row in tables["sobol_indices"] if row["Base_N"] == base_n
                             and row["Metric"] == metric and row["Record_Type"] == "S1_ST"
                             and row["Status"] == "ok" and np.isfinite(row["ST"])),
                            key=lambda row: (-row["ST"], row["Parameter"]))
            tables["critical_parameters"].extend({**row, "Rank_By_ST": rank, "Rank": rank}
                                                 for rank, row in enumerate(ranked, start=1))
    sample_rows = [row for row in tables["sample_metrics"] if row["Record_Type"] == "sample"
                   and row.get("Sample_Role") in ("A", "B")]
    for metric in ("Fixed_Feasible", "Resource_Reselected", "Search_Gap", "Jaccard", "Hamming", "P_Response_Change"):
        values = np.asarray([row.get(metric, np.nan) for row in sample_rows], dtype=float)
        valid = values[np.isfinite(values)]
        tables["sample_metrics"].append({**common_keys, "Record_Type": "summary",
            "Metric": metric, "Summary_Base": "A/B rows; paired design; descriptive only",
            "Count": len(valid), "Missing_Count": len(values) - len(valid),
            "Mean": float(np.mean(valid)) if len(valid) else np.nan,
            "Min": float(np.min(valid)) if len(valid) else np.nan,
            "Max": float(np.max(valid)) if len(valid) else np.nan})
    valid_decisions = [row for row in sample_rows if row.get("Decision_ID")]
    for label in original["Allocation"]["Label"]:
        count = sum(label in json.loads(row["Selected_Defenses"]) for row in valid_decisions)
        tables["sample_metrics"].append({**common_keys, "Record_Type": "measure_frequency", "Defense_Label": label,
            "Summary_Base": "A/B rows; paired design; descriptive only",
            "Count": len(valid_decisions), "Missing_Count": len(sample_rows) - len(valid_decisions),
            "Selected_Count": count,
            "Frequency": count / len(valid_decisions) if valid_decisions else np.nan})
    metadata["Total_Optimization_FE"] = sum(row.get("Total_FE", 0) for row in tables["sample_metrics"]
                                            if row["Record_Type"] == "optimization_run")
    metadata["Noise_Optimization_FE"] = sum(row.get("Total_FE", 0) for row in tables["sobol_diagnostics"]
                                            if row["Record_Type"] == "optimization_noise"
                                            and row.get("Noise_Role") == "extra_seed")
    metadata["Failed_Sample_Count"] = sum(row["Status"] != "ok" for row in tables["sample_metrics"]
                                          if row["Record_Type"] == "sample")
    metadata["Status"] = "complete_with_failures" if metadata["Failed_Sample_Count"] else "complete"
    for records in tables.values():
        for row in records:
            row.update(common_keys)
    progress.log(f"finished: status={metadata['Status']}; samples={len(results)}; "
                 f"failed_samples={metadata['Failed_Sample_Count']}; "
                 f"optimization_FE={metadata['Total_Optimization_FE']}; "
                 f"noise_FE={metadata['Noise_Optimization_FE']}")
    return {"input_excel": str(input_excel), "metadata": metadata, **tables}


if __name__ == "__main__":
    import argparse
    from common.pipeline import run_solve
    from features.sensitivity_analysis.export_csv import export

    parser = argparse.ArgumentParser(description="Run paired sensitivity analysis")
    parser.add_argument("--input-excel", type=Path, default=None)
    parser.add_argument("--sensitivity-config", type=Path, default=None)
    parser.add_argument("--sample-mode", choices=("fast", "standard", "comprehensive"), default="fast")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--game-iterations", type=int, default=None)
    parser.add_argument("--game-learning-rate", type=float, default=None)
    arguments = vars(parser.parse_args())
    arguments["mode"] = arguments.pop("sample_mode")
    run_solve("sensitivity_analysis", solve, export, **arguments)