

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common.config import CSV_FILES, get_feature_output_dir
from common.memetic import get_selected_ma_file


REQUIRED_COLUMNS = {
    "runs": {
        "Experiment", "Run", "Seed", "Budget_FE", "Total_FE",
        "Genetic_FE", "Local_FE", "Generations", "Pareto_Size",
        "HV", "IGD", "Local_Accepted", "Interrupted_Local_Rounds",
        "Neighbor_Scans", "Wall_Seconds", "Cache_Hits",
        "Unique_Simulations",
    },
    "history": {
        "Experiment", "Run", "Seed", "Budget_FE", "Generation",
        "Total_FE", "Local_FE", "Pareto_Size",
        "Minimum_Cost", "Minimum_Loss", "HV", "IGD",
    },
    "final_decisions": {
        "Experiment", "Run", "Seed", "Budget_FE", "Point_ID",
        "X_Bits", "Total_Cost", "Total_Defender_Loss",
    },
    "summary": {
        "Experiment", "Budget_FE", "Runs",
        "HV_Mean", "HV_Std", "IGD_Mean", "IGD_Std",
    },
    "ranking": {
        "Rank", "Experiment", "Budget_FE", "HV_Mean", "IGD_Mean",
        "HV_Rank", "IGD_Rank",
    },
    "reference_front": {
        "Total_Cost", "Total_Defender_Loss",
        "Normalized_Cost", "Normalized_Loss",
    },
    "experiment_manifest": {
        "budget", "pop_size", "game_iterations",
        "game_learning_rate", "excel_sha256",
        "FE_Budgets", "Configurations", "Parallel_Processes",
    },
    "variants": {
        "name", "selection_mode", "neighborhood",
        "local_search_prob", "max_local_search_iters",
        "no_improve_rounds", "elite_size",
    },
}


def _validate_selection(document: dict) -> None:
    
    required_document = {
        "schema_version", "context", "selected",
        "experiment_name", "selection_rule",
    }
    if not required_document.issubset(document):
        raise ValueError("The selection object is missing top-level contract fields")
    if document["schema_version"] != 1:
        raise ValueError("The selection object schema_version must be 1")

    required_context = {
        "budget", "pop_size", "game_iterations",
        "game_learning_rate", "excel_sha256",
    }
    required_selected = {
        "selection_mode", "neighborhood", "local_search_prob",
        "max_local_search_iters", "no_improve_rounds", "elite_size",
    }
    if not required_context.issubset(document["context"]):
        raise ValueError("The selection object context is incomplete")
    if not required_selected.issubset(document["selected"]):
        raise ValueError("The selection object selected field is incomplete")


def export(result: dict, *, output_dir=None) -> dict:
    
    files = CSV_FILES["macomp"]
    tables = {}

    for name, filename in files.items():
        frame = result.get(filename, result.get(name))
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"Missing DataFrame: {filename}")

        missing = REQUIRED_COLUMNS[name] - set(frame.columns)
        if missing:
            raise ValueError(
                f"{filename} is missing required contract columns: {sorted(missing)}"
            )

        
        tables[filename] = frame.copy()

    document = result.get("selected_ma")
    if not isinstance(document, dict):
        raise ValueError("The solver object is missing selected_ma")
    _validate_selection(document)

    json_text = json.dumps(
        document,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )

    input_excel = result.get("input_excel")
    if input_excel is None:
        manifest = tables[files["experiment_manifest"]]
        input_excel = manifest.iloc[0]["Input_Excel"]
    directory = (
        get_feature_output_dir("macomp", input_excel)
        if output_dir is None else Path(output_dir)
    )
    directory.mkdir(parents=True, exist_ok=True)
    staged_files = []
    for filename, frame in tables.items():
        temporary = directory / f"{filename}.tmp"
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        staged_files.append((temporary, directory / filename))

    selected_ma_file = get_selected_ma_file(input_excel)
    selected_ma_file.parent.mkdir(parents=True, exist_ok=True)
    json_temporary = selected_ma_file.with_suffix(".json.tmp")
    json_temporary.write_text(json_text, encoding="utf-8")

    for temporary, destination in staged_files:
        temporary.replace(destination)

    json_temporary.replace(selected_ma_file)
    print(f"macomp CSV files exported to {directory}; selection JSON: {selected_ma_file}")
    return tables


if __name__ == "__main__":
    from common.pipeline import run_export

    run_export("macomp", export)
