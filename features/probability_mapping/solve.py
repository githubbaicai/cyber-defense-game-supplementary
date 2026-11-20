"""Isolated mapping searches, followed by complete-decision cross-evaluation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from multiprocessing import get_context
import os
from pathlib import Path
import random
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd

from common.config import ROOT, get_feature_output_dir
from common.implementation import IMPLEMENTATION_VERSION, MODEL_PROTOCOL_VERSION
from common.memetic import sha256_file
from common.optimization_execution import run_task
from common.problem_factory import load_problem_context

from .config import MAPPINGS, PROTOCOL_VERSION, load_config
from .evaluation import decision_record, evaluate_pool, select_best, summarize
from .inputs import mapping_input_records, transform_data


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _task(settings, seed, run):
    return {
        "name": "NSGA2_MA", "run": run, "seed": seed,
        "protocol": "equal_fe", "strategy": "exact_n_eval",
        "pop_size": settings["pop_size"], "max_fe": settings["max_fe"],
        "budget": settings["budget"], "ma_parameters": settings["ma_parameters"],
        "local_fe_reference": settings["max_fe"],
        "max_gen": int(np.ceil(settings["max_fe"] / settings["pop_size"])),
        "tol": .001, "n_stable": 5, "period": 1,
        **{key: settings[key] for key in ("game_iterations", "game_learning_rate", "prune_tol", "game_termination")},
    }


def _worker(job):
    settings, mapping = job["settings"], job["mapping"]
    random_state, numpy_state = random.getstate(), np.random.get_state()
    started = perf_counter()
    try:
        if sha256_file(settings["input_excel"]) != job["input_hash"]:
            raise ValueError("Input workbook changed during the experiment")
        with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink):
            items, data, graph = load_problem_context(settings["input_excel"])
            context = (items, transform_data(data, mapping), graph)
            if job["stage"] == "search":
                random.seed(job["seed"])
                output = run_task(_task(settings, job["seed"], job["run"]), context)
                if not output["Comparison_Valid"]:
                    raise RuntimeError(f"Invalid search: {output['Status']}")
                front = [
                    {"Mapping_ID": mapping, "Seed": job["seed"],
                     **decision_record(items, json.loads(decision["X_JSON"]), objective, settings["lambda_loss"])}
                    for decision, objective in zip(output["Decisions"], output["Front"])
                ]
                best = select_best(front)
                metrics = {key: value for key, value in output.items()
                           if value is None or isinstance(value, (str, int, float, bool))}
                metrics.update(Mapping_ID=mapping, Best_J=best["J"], Best_Cost=best["Cost"],
                               Best_Loss=best["Loss"], Best_Decision_ID=best["Decision_ID"])
                history = [{"Mapping_ID": mapping, "Seed": job["seed"],
                            **{key: value for key, value in row.items()
                               if key not in ("Front", "Decisions", "Display_Front", "Display_Decisions")}}
                           for row in output["History"]]
                return {"front": front, "metrics": metrics, "history": history}
            rows, counts = evaluate_pool(context, job["candidates"], settings)
            return {"rows": [{**row, "Mapping_ID": mapping} for row in rows],
                    "metrics": {"Mapping_ID": mapping, **counts,
                                "Runtime_Seconds": perf_counter() - started, "Worker_PID": os.getpid()}}
    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)


def _execute(jobs, workers):
    if workers == 1:
        results = []
        for index, job in enumerate(jobs, 1):
            results.append(_worker(job))
            print(f"{job['stage']} {index}/{len(jobs)}: {job['mapping']} seed={job.get('seed', '-')}", flush=True)
        return results
    outputs = {}
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=get_context("spawn")) as executor:
        futures = {executor.submit(_worker, job): index for index, job in enumerate(jobs)}
        for future in as_completed(futures):
            index = futures[future]
            outputs[index] = future.result()
            job = jobs[index]
            print(f"{job['stage']} {len(outputs)}/{len(jobs)}: {job['mapping']} seed={job.get('seed', '-')}", flush=True)
    return [outputs[index] for index in range(len(jobs))]


def solve(input_excel=None, mapping_config=None, workers=None, seeds=None, max_fe=None,
          pop_size=None, budget=None, game_iterations=None, game_learning_rate=None):
    started = perf_counter()
    source_files = [ROOT / "run.py", *sorted((ROOT / "common").glob("*.py")),
                    *sorted((ROOT / "external").glob("*.py")), *sorted(Path(__file__).parent.glob("*.py"))]
    source_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in source_files}
    settings = load_config(mapping_config, input_excel=input_excel, workers=workers, seeds=seeds,
                           max_fe=max_fe, pop_size=pop_size, budget=budget,
                           game_iterations=game_iterations, game_learning_rate=game_learning_rate)
    input_path = settings["input_excel"]
    snapshot = input_path.read_bytes()
    input_hash = sha256(snapshot).hexdigest()
    with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink):
        items, data, _ = load_problem_context(input_path)
    inputs = mapping_input_records(data, input_path)
    identifier = f"mapping_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    output_dir = get_feature_output_dir("probability_mapping", input_path) / "experiments" / identifier
    common_job = {"settings": settings, "input_hash": input_hash}
    jobs = [{**common_job, "stage": "search", "mapping": mapping, "seed": seed, "run": index}
            for mapping in MAPPINGS for index, seed in enumerate(settings["seeds"], 1)]
    print(f"Probability mappings: {len(jobs)} searches, FE={settings['max_fe']} each, workers={min(settings['workers'], len(jobs))}", flush=True)
    searches = _execute(jobs, settings["workers"])
    candidates = [row for search in searches for row in search["front"]]
    pool = {row["Decision_ID"]: row for row in candidates}
    no_defense = decision_record(items, np.zeros(len(items) + 1), [0, 0], settings["lambda_loss"])
    pool.setdefault(no_defense["Decision_ID"], no_defense)
    unique = [pool[decision_id] for decision_id in sorted(pool)]
    evaluation_jobs = [{**common_job, "stage": "evaluate", "mapping": mapping, "candidates": unique}
                       for mapping in MAPPINGS]
    evaluated = _execute(evaluation_jobs, settings["workers"])
    cross = [row for result in evaluated for row in result["rows"]]
    for mapping in MAPPINGS:
        group = [row for row in cross if row["Mapping_ID"] == mapping]
        ranks = pd.Series([row["J"] for row in group]).rank(method="min").astype(int)
        for row, rank in zip(group, ranks):
            row["J_Rank"] = int(rank)
    lookup = {(row["Mapping_ID"], row["Decision_ID"]): row for row in cross}
    max_replay_error = 0.
    for candidate in candidates:
        replay = lookup[(candidate["Mapping_ID"], candidate["Decision_ID"])]
        error = max(abs(candidate[field] - replay[field]) for field in ("Cost", "Loss"))
        max_replay_error = max(max_replay_error, error)
        if not np.allclose([candidate["Cost"], candidate["Loss"]], [replay["Cost"], replay["Loss"]], rtol=1e-10, atol=1e-8):
            raise ValueError("Search and cross-evaluation objectives disagree")
    summary, selected = summarize(candidates, cross, no_defense["Decision_ID"])
    tables = {
        "mapping_inputs": pd.DataFrame(inputs), "mapping_runs": pd.DataFrame([result["metrics"] for result in searches]),
        "mapping_fronts": pd.DataFrame(candidates), "mapping_cross_evaluations": pd.DataFrame(cross),
        "mapping_summary": summary, "mapping_selected_decisions": selected,
        "mapping_history": pd.DataFrame([row for result in searches for row in result["history"]]),
        "mapping_evaluation_counts": pd.DataFrame([result["metrics"] for result in evaluated]),
    }
    for frame in tables.values():
        frame.insert(0, "Experiment_ID", identifier)
        frame.insert(0, "Case_ID", input_path.stem)
    if sha256_file(input_path) != input_hash:
        raise ValueError("Input workbook changed during the experiment")
    if any(sha256_file(ROOT / path) != expected for path, expected in source_hashes.items()):
        raise ValueError("Source code changed during the experiment; rerun with a fixed version")
    metadata = {
        "Experiment_ID": identifier, "Case_ID": input_path.stem, "Status": "complete",
        "Protocol_Version": PROTOCOL_VERSION, "Implementation_Version": IMPLEMENTATION_VERSION,
        "Model_Protocol_Version": MODEL_PROTOCOL_VERSION, "Input_SHA256": input_hash,
        "Configuration": json.loads(json.dumps(settings, default=_json_value)),
        "Source_SHA256": source_hashes,
        "Dependencies": {name: version(name) for name in ("numpy", "pandas", "pymoo", "matplotlib", "openpyxl")},
        "Optimization_FE": sum(result["metrics"]["Total_FE"] for result in searches),
        "Cross_Evaluation_FE": sum(result["metrics"]["Evaluation_FE"] for result in evaluated),
        "Max_Replay_Error": max_replay_error, "Runtime_Seconds": perf_counter() - started,
        "Candidate_Pool": "union of returned strict front representatives across mappings/seeds, plus no defense; complete decisions retained",
        "Decision_Criterion": "minimum C + lambda_loss * L; deterministic cost/loss/ID tie break",
        "Candidate_Gap": "same-target Fixed_J minus candidate-pool best J; not global optimality regret",
        "Calibration_Status": "scenario proxy mappings; not calibrated conditional attack success probabilities",
        "Other_Parameters": "held fixed; uncertainty analysis is a separate experiment",
        "Historical_Results_Reused": False,
    }
    return {"input_excel": input_path, "output_dir": output_dir, "tables": tables,
            "metadata": metadata, "input_snapshot": snapshot}


if __name__ == "__main__":
    raise SystemExit("Use: python run.py --feature probability_mapping --mode all --workers 9")
