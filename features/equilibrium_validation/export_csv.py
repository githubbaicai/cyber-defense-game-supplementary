"""Export equilibrium-validation tables without running diagnostics."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common.config import get_feature_output_dir


TABLE_NAMES = (
    "game_definition",
    "equilibrium_diagnostics",
    "jacobian_eigenvalues",
    "initialization_sensitivity",
    "strategy_trace",
    "path_sensitivity",
    "run_metadata",
)

COMMON_COLUMNS = ("Case_ID", "Experiment_ID", "Portfolio_ID", "Run_ID")

REQUIRED_COLUMNS = {
    "game_definition": (
        "Node_Path", "Action_ID", "Action_Label", "Detection_Probability",
        "A_Not_Detected", "A_Detected_No_Response", "A_Detected_Response",
        "L_Not_Detected", "L_Detected_No_Response", "L_Detected_Response",
        "Response_Feasible",
    ),
    "equilibrium_diagnostics": (
        "Node_Path", "Checkpoint", "R_map", "Stationary_Status",
        "Induced_Nash_Status", "ESS_Status", "Discrete_Stability_Status",
        "Reason", "Game_Scope",
    ),
    "jacobian_eigenvalues": (
        "Node_Path", "Checkpoint", "Jacobian_Type", "Eigenvalue_Index",
        "Eigenvalue_Real", "Eigenvalue_Imag", "Spectral_Radius", "Step_Size",
        "Status", "Coordinate_Scope",
    ),
    "initialization_sensitivity": (
        "Experiment_Type", "Node_Path", "Checkpoint", "Initialization",
        "Response_Initial", "Learning_Rate", "Iterations", "Stop_Reason",
        "Endpoint_Category",
    ),
    "strategy_trace": (
        "Node_Path", "Iteration", "Distribution", "Action_ID", "Probability",
        "Map_Residual",
    ),
    "path_sensitivity": (
        "Path_Type", "Path", "Probability", "Rank", "Is_Main_Path", "Tied",
        "Status",
    ),
    "run_metadata": (
        "Input_Excel", "Input_SHA256", "Model_Protocol", "Decision_JSON",
        "P_Response", "Learning_Rate", "Iterations", "Seed", "Status", "Reason",
        "Source_Status", "Replay_Status", "Config_JSON", "Provenance_JSON",
    ),
}

NUMERIC_COLUMNS = {
    "game_definition": (
        "Detection_Probability", "A_Not_Detected", "A_Detected_No_Response",
        "A_Detected_Response", "L_Not_Detected", "L_Detected_No_Response",
        "L_Detected_Response", "Payoff_Scale", "Loss_Scale", "P_Initial",
    ),
    "equilibrium_diagnostics": (
        "Checkpoint", "R_map", "R_map_p", "R_map_r", "Fitness_Residual",
        "Attacker_Fitness_Residual", "Response_Fitness_Residual", "Tail_Variation",
        "Invasion_Gap", "Attacker_Invasion_Gap", "Response_Invasion_Gap",
        "Normalized_Invasion_Gap", "Spectral_Radius",
    ),
    "jacobian_eigenvalues": (
        "Checkpoint", "Eigenvalue_Index", "Eigenvalue_Real", "Eigenvalue_Imag",
        "Spectral_Radius", "Step_Size",
    ),
    "initialization_sensitivity": (
        "Checkpoint", "Learning_Rate", "Iterations", "Total_Defender_Loss",
        "Total_Attacker_Reward", "Strategy_Distance", "Path_Distance",
    ),
    "strategy_trace": (
        "Iteration", "Probability", "Payoff", "Fitness",
        "Map_Residual",
    ),
    "path_sensitivity": (
        "Probability", "Rank", "Baseline_Probability", "Probability_Delta",
        "Total_Defender_Loss",
    ),
    "run_metadata": ("P_Response", "Learning_Rate", "Iterations", "Seed"),
}

UNIQUE_COLUMNS = {
    "game_definition": (*COMMON_COLUMNS, "Node_Path", "Action_ID"),
    "equilibrium_diagnostics": (*COMMON_COLUMNS, "Node_Path", "Checkpoint"),
    "jacobian_eigenvalues": (
        *COMMON_COLUMNS, "Node_Path", "Checkpoint", "Jacobian_Type", "Eigenvalue_Index",
    ),
    "initialization_sensitivity": (
        *COMMON_COLUMNS, "Experiment_Type", "Node_Path", "Checkpoint", "Initialization",
    ),
    "strategy_trace": (*COMMON_COLUMNS, "Node_Path", "Iteration", "Distribution", "Action_ID"),
    "path_sensitivity": (*COMMON_COLUMNS, "Path_Type", "Path"),
    "run_metadata": (*COMMON_COLUMNS,),
}


def _output_dir(input_excel=None, output_dir=None) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    return get_feature_output_dir("equilibrium_validation", input_excel)


def _rows(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    if isinstance(value, dict):
        return [value]
    return list(value)


def _defaults(result: dict[str, Any]) -> dict[str, Any]:
    config_data = result.get("config") or {}
    return {
        "Case_ID": result.get("case_id", config_data.get("Case_ID", "unknown")),
        "Experiment_ID": result.get("experiment_id", "experiment"),
        "Portfolio_ID": result.get("portfolio_id", config_data.get("Portfolio_ID", "unknown")),
        "Run_ID": result.get("run_id", "run-1"),
    }


def _as_frame(name: str, rows: Any, defaults: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(_rows(rows))
    missing = [column for column in (*COMMON_COLUMNS, *REQUIRED_COLUMNS[name]) if column not in frame]
    if missing and len(frame) > 0:
        raise ValueError(f"{name}: missing required columns: {', '.join(missing)}")
    columns = list(dict.fromkeys((*COMMON_COLUMNS, *REQUIRED_COLUMNS[name], *frame.columns)))
    frame = frame.reindex(columns=columns)
    if len(frame) == 0:
        return frame
    for column in COMMON_COLUMNS:
        values = frame[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"{name}: {column} must be non-empty")
        frame[column] = values.astype(str)
    for column in NUMERIC_COLUMNS[name]:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].notna() & (~np.isfinite(values.to_numpy(dtype=float)))
        if invalid.any():
            raise ValueError(f"{name}: {column} contains non-finite numeric values")
        frame[column] = values
    unique_columns = UNIQUE_COLUMNS[name]
    for column in unique_columns:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{name}: key {column} must be non-empty")
    if frame.duplicated(list(unique_columns)).any():
        raise ValueError(f"{name}: duplicate key ({', '.join(unique_columns)})")
    required_numeric = {
        "game_definition": ("Detection_Probability", "A_Not_Detected", "A_Detected_No_Response", "A_Detected_Response", "L_Not_Detected", "L_Detected_No_Response", "L_Detected_Response"),
        "strategy_trace": ("Iteration", "Probability"),
        "run_metadata": ("P_Response", "Learning_Rate", "Iterations"),
    }.get(name, ())
    for column in required_numeric:
        if frame[column].isna().any():
            raise ValueError(f"{name}: {column} must be finite")
    for column in ("Probability", "Detection_Probability", "P_Response", "P_Initial"):
        if column in frame and ((frame[column] < 0) | (frame[column] > 1)).any():
            raise ValueError(f"{name}: {column} must be within [0, 1]")
    for column in ("Iteration", "Iterations", "Checkpoint", "Eigenvalue_Index"):
        if column in frame:
            values = frame[column].dropna()
            if ((values < 0) | (values % 1 != 0)).any():
                raise ValueError(f"{name}: {column} must contain non-negative integers")
    if "Learning_Rate" in frame and (frame["Learning_Rate"] < 0).any():
        raise ValueError(f"{name}: Learning_Rate must be non-negative")
    if name == "strategy_trace" and not frame["Distribution"].isin(("p", "r", "q")).all():
        raise ValueError("strategy_trace: Distribution must be p, r or q")
    if name == "initialization_sensitivity" and not frame["Experiment_Type"].isin(("fixed_local_snapshot", "full_graph_replay")).all():
        raise ValueError("initialization_sensitivity: unknown Experiment_Type")
    if name == "run_metadata":
        for column in ("Config_JSON", "Provenance_JSON", "Decision_JSON"):
            for value in frame[column]:
                try:
                    json.loads(value)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"run_metadata: invalid {column}") from error
    return frame


def _metadata_rows(result: dict[str, Any], defaults: dict[str, Any]):
    value = result.get("tables", {}).get("run_metadata")
    if value is None:
        value = result.get("metadata")
    rows = _rows(value)
    if not rows:
        rows = [{}]
    enriched = []
    for row in rows:
        item = dict(row)
        item.update({key: item.get(key, value) for key, value in defaults.items()})
        if "Config_JSON" not in item:
            item["Config_JSON"] = json.dumps(result.get("config", {}), ensure_ascii=False, sort_keys=True, default=str)
        if "Provenance_JSON" not in item:
            item["Provenance_JSON"] = json.dumps(result.get("provenance", {}), ensure_ascii=False, sort_keys=True, default=str)
        enriched.append(item)
    return enriched


def export(result: dict[str, Any], input_excel=None, output_dir=None):
    """Validate and persist the seven E3 CSV tables; never run a computation."""
    if not isinstance(result, dict):
        raise TypeError("result must be a dictionary")
    defaults = _defaults(result)
    tables = result.get("tables") or {}
    frames = {}
    for name in TABLE_NAMES:
        rows = _metadata_rows(result, defaults) if name == "run_metadata" else tables.get(name, [])
        frames[name] = _as_frame(name, rows, defaults)
    metadata = frames["run_metadata"]
    if metadata.empty:
        raise ValueError("run_metadata: at least one run is required")
    if metadata["Run_ID"].duplicated().any():
        raise ValueError("run_metadata: Run_ID must uniquely identify its metadata")
    metadata_ids = set(metadata["Run_ID"].astype(str))
    expected_experiment = str(result.get("experiment_id", metadata.iloc[0]["Experiment_ID"]))
    if not metadata["Experiment_ID"].eq(expected_experiment).all():
        raise ValueError("run_metadata: Experiment_ID does not match result")
    identity_by_run = metadata.set_index("Run_ID")[list(COMMON_COLUMNS[:-1])]
    source = input_excel or result.get("input_excel")
    if source is not None:
        expected_case = Path(source).stem
        if not metadata["Case_ID"].eq(expected_case).all():
            raise ValueError("run_metadata: Case_ID does not match input_excel stem")
    for case_id, source_path in zip(metadata["Case_ID"], metadata["Input_Excel"]):
        if pd.isna(source_path) or Path(str(source_path)).stem != case_id:
            raise ValueError("run_metadata: Case_ID does not match recorded Input_Excel")
    for name, frame in frames.items():
        if name != "run_metadata" and len(frame) and not set(frame["Run_ID"].astype(str)).issubset(metadata_ids):
            raise ValueError(f"{name}: Run_ID has no matching run_metadata row")
        if name != "run_metadata" and len(frame):
            for column in COMMON_COLUMNS[:-1]:
                expected = frame["Run_ID"].map(identity_by_run[column])
                if not frame[column].eq(expected).all():
                    raise ValueError(f"{name}: {column} disagrees with run_metadata identity")
    directory = _output_dir(input_excel or result.get("input_excel"), output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(directory / f"{name}.csv", index=False)
    if result.get("publication_tables"):
        from features.equilibrium_validation.publication import export_tables
        export_tables(result["publication_tables"], directory)
    return {**frames, "output_dir": directory}


def main():
    parser = argparse.ArgumentParser(description="Export equilibrium-validation CSV tables")
    parser.add_argument("--input-excel", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    if args.output_dir:
        cache = Path(args.output_dir).expanduser().resolve() / "solve_result.pkl"
        if not cache.is_file():
            raise FileNotFoundError(f"Solver-result cache not found: {cache}")
        with cache.open("rb") as stream:
            result = pickle.load(stream)
        export(result, input_excel=args.input_excel, output_dir=args.output_dir)
    else:
        from common.pipeline import run_export
        run_export("equilibrium_validation", export, input_excel=args.input_excel)


if __name__ == "__main__":
    main()
