

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import multiprocessing
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import pandas as pd
import pymoo

from common.config import get_feature_output_dir
from common.implementation import IMPLEMENTATION_VERSION, MODEL_PROTOCOL_VERSION
from common.optimization_execution import run_task
from common.problem_factory import load_problem_context

from .config import load_config
from .export_csv import FROZEN_TABLES, file_hash, freeze, load_frozen
from .inputs import load_source, replay_objectives


def _search(task, workbook):
    return run_task(task, load_problem_context(workbook))


def _replay_batch(source, workbook, game_termination, expected_hash):
    if file_hash(workbook) != expected_hash:
        raise ValueError("The input snapshot changed during historical-objective verification")
    return replay_objectives(source, load_problem_context(workbook), game_termination)


def _replay_source(source, settings):
    started = perf_counter()
    unique = source["points"].drop_duplicates("X_JSON").reset_index(drop=True)
    total = len(unique)

    def report(completed, count):
        print(f"Historical-objective verification {completed}/{count}, elapsed {perf_counter() - started:.1f}s", flush=True)

    if settings["workers"] == 1:
        return replay_objectives(source, load_problem_context(source["input_excel"]), settings["game_termination"], progress=report)
    batches = [unique.iloc[offset:offset + 200].copy() for offset in range(0, total, 200)]
    workers = min(settings["workers"], len(batches))
    print(f"Historical-objective verification: workers={workers}, batches={len(batches)}", flush=True)
    results = {}
    completed = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = {
            executor.submit(_replay_batch, {"points": batch, "settings": source["settings"], "budget": source["budget"]},
                            source["input_excel"], settings["game_termination"], source["excel_sha256"]): index
            for index, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            audit = future.result()
            results[futures[future]] = audit
            completed += len(audit)
            report(completed, total)
    return pd.concat([results[index] for index in range(len(batches))], ignore_index=True)


def _validate_protocol(settings, source):
    overlap = set(settings["seeds"]) & source["seeds"]
    if overlap:
        raise ValueError(f"Reference seeds overlap with the previous experiment: {sorted(overlap)}")
    metrics = source["metrics"]
    if not metrics["Protocol"].eq("equal_fe").all():
        raise ValueError("The current reevaluation entry point requires a single equal_fe experiment")
    source_fe = pd.to_numeric(metrics["Total_FE"], errors="raise").max()
    if settings["min_actual_fe"] < source_fe:
        raise ValueError(f"min_actual_fe must not be smaller than the maximum FE of the previous experiment ({source_fe:g})")
    if settings["max_fe"] < 5 * source_fe:
        raise ValueError(f"max_fe must be at least five times the maximum FE of the previous experiment ({source_fe:g})")
    if str(pymoo.__version__) != str(source["manifest"]["Pymoo_Version"].iloc[0]):
        raise ValueError("The current pymoo version differs from the previous experiment; use its corresponding environment")


def _tasks(settings, source):
    return [{
        "name": name, "run": run, "seed": seed,
        "pop_size": settings["pop_size"], "max_fe": settings["max_fe"],
        "budget": source["budget"], "ma_parameters": source["ma_parameters"],
        "local_fe_reference": settings["max_fe"], "strategy": "combined_or",
        "max_gen": settings["max_gen"], "tol": settings["tol"],
        "n_stable": settings["n_stable"], "period": 1,
        "min_gen": settings["min_gen"], "min_front_size": 2,
        "loss_scale": settings["loss_scale"], "ref_point": (1.1, 1.1),
        "protocol": "equal_termination", "legacy": False,
        "game_termination": settings["game_termination"], **source["settings"],
    } for name in settings["algorithms"] for run, seed in enumerate(settings["seeds"], 1)]


def _search_tables(runs, minimum_fe):
    summaries, candidates, histories = [], [], []
    excluded = {"Front", "Decisions", "Display_Front", "Display_Decisions", "History",
                "Main_Loop_Front", "Main_Loop_Decisions", "Termination_History"}
    import json

    for result in runs:
        if not result["Comparison_Valid"] or result["Total_FE"] < minimum_fe:
            raise ValueError(f"The reference run is invalid or below the minimum FE ({minimum_fe}): {result['Algorithm']}, seed={result['Seed']}, FE={result['Total_FE']}")
        identity = {name: result[name] for name in ("Algorithm", "Run", "Seed")}
        summaries.append({name: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                          for name, value in result.items() if name not in excluded})
        snapshots = [*result["History"], {"Generation": result["Generations"], "Front": result["Front"],
                                             "Decisions": result["Decisions"], "Stage": "final"}]
        for snapshot in snapshots:
            stage = snapshot.get("Stage", "generation")
            if stage == "generation":
                histories.append({**identity, **{name: value for name, value in snapshot.items() if name not in excluded}})
            for point_id, (point, decision) in enumerate(zip(snapshot["Front"], snapshot["Decisions"]), 1):
                candidates.append({**identity, "Stage": stage, "Generation": snapshot["Generation"],
                                   "Point_ID": point_id, "Front_Kind": "strict", "Total_Cost": point[0],
                                   "Total_Defender_Loss": point[1], **decision})
    return {"reference_runs": pd.DataFrame(summaries), "reference_candidates": pd.DataFrame(candidates),
            "reference_history": pd.DataFrame(histories)}


def _provenance(settings, source):
    search = {name: value for name, value in settings.items() if name not in ("config_path", "source_dir", "input_excel", "game_termination")}
    return {
        "source_dir": str(source["source_dir"]), "source_sha256": source["hashes"],
        "input_excel": str(source["input_excel"]), "excel_sha256": source["excel_sha256"],
        "implementation_version": IMPLEMENTATION_VERSION, "model_protocol_version": MODEL_PROTOCOL_VERSION,
        "pymoo_version": str(pymoo.__version__), "game_termination": asdict(settings["game_termination"]),
        "game_settings": source["settings"], "search_settings": search,
        "reference_pool": "independent strict generation and final fronts only",
        "termination": "full-front stability OR FE cap OR generation cap",
        "ma_parameters": source["ma_parameters"],
        "experiment_role": "pilot" if settings.get("pilot", False) else "multi_seed_reference",
    }


def solve(input_excel=None, reference_config=None, reference_dir=None, workers=None):
    from .analysis import freeze_reference, reevaluate

    settings = load_config(reference_config, input_excel=input_excel, workers=workers)
    source = load_source(settings["source_dir"], settings["input_excel"])
    _validate_protocol(settings, source)
    provenance = _provenance(settings, source)
    prefix = "pilot" if settings.get("pilot", False) else "reference"
    identifier = f"{prefix}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    output_dir = get_feature_output_dir("independent_reference", source["input_excel"]) / "experiments" / identifier
    started = perf_counter()
    print(f"Verifying all historical decisions: {source['points']['X_JSON'].nunique()}", flush=True)
    audit = _replay_source(source, settings)
    replay_seconds = perf_counter() - started
    search_started = perf_counter()
    if reference_dir is not None:
        reference, manifest = load_frozen(reference_dir)
        original = manifest["provenance"]
        for key in ("source_sha256", "excel_sha256", "implementation_version", "model_protocol_version",
                    "pymoo_version", "game_termination", "game_settings", "ma_parameters", "experiment_role"):
            if original[key] != provenance[key]:
                raise ValueError(f"The frozen reference is incompatible with the current experiment: {key}")
        provenance = original
        tables = {name: pd.read_csv(Path(reference_dir) / f"{name}.csv", float_precision="round_trip", keep_default_na=False,
                                   dtype={"X_Bits": str, "X_JSON": str}) for name in FROZEN_TABLES}
    else:
        tasks = _tasks(settings, source)
        runs = []
        search_workers = min(settings["workers"], len(tasks))
        print(f"Independent reference search: {len(tasks)} groups, workers={search_workers} (configured limit {settings['workers']})", flush=True)
        with ProcessPoolExecutor(max_workers=search_workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = {executor.submit(_search, task, source["input_excel"]): task for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                runs.append(result)
                print(f"Independent search {len(runs)}/{len(tasks)}: {result['Algorithm']}, seed={result['Seed']}, FE={result['Total_FE']}, stop={result['Stop_Reason']}", flush=True)
        runs.sort(key=lambda row: (row["Algorithm"], row["Run"], row["Seed"]))
        tables = _search_tables(runs, settings["min_actual_fe"])
        reference, tables["reference_front"] = freeze_reference(tables["reference_candidates"], source["budget"])
    result = {"input_excel": source["input_excel"], "output_dir": output_dir,
              "reference": reference, "provenance": provenance, "tables": tables}
    freeze(result)
    reference_seconds = perf_counter() - search_started
    reevaluation_started = perf_counter()
    result["tables"].update(reevaluate(source, reference))
    result["tables"]["objective_replay_audit"] = audit
    result["tables"]["source_experiment_manifest"] = source["manifest"].copy(deep=True)
    for name, expected in source["hashes"].items():
        if file_hash(source["source_dir"] / name) != expected:
            raise ValueError(f"A source file changed during reevaluation: {name}")
    if file_hash(source["input_excel"]) != source["excel_sha256"]:
        raise ValueError("The input snapshot changed during reevaluation")
    result["timing"] = {"objective_replay_seconds": replay_seconds,
                        "reference_build_or_load_seconds": reference_seconds,
                        "offline_reevaluation_seconds": perf_counter() - reevaluation_started,
                        "reused_reference_dir": None if reference_dir is None else str(Path(reference_dir).resolve())}
    return result


if __name__ == "__main__":
    from common.pipeline import run_solve
    from .export_csv import export
    run_solve("independent_reference", solve, export)
