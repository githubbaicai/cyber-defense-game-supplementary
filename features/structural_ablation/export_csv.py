"""Validate and export structural-ablation tables without running experiments."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from common.case_config import get_input_excel
from common.config import CSV_FILES, get_feature_output_dir
from common.implementation import MODEL_PROTOCOL_VERSION


STRUCTURAL_PROTOCOL_VERSION = "structural-ablation-v3-population-average-detection"
IDENTITY_COLUMNS = ("case_id", "input_sha256", "model_protocol_version")
CONVERGENCE_COLUMNS = (
    "iterations_executed", "stop_reason", "converged", "stationarity_residual", "invasion_gap",
)
REQUIRED_COLUMNS = {
    "experiment_manifest": ("input_excel", "run_id", "baseline_protocol", "model_options_protocol"),
    "structural_ablation_runs": ("variant", "seed", "max_fe", "total_fe", "wall_time_s", "status"),
    "fixed_portfolio_results": (
        "variant", "portfolio_id", "status", "purchase_cost", "defender_loss", "p_response",
        "prevention_cost", "detection_cost", "response_cost",
        *CONVERGENCE_COLUMNS, "local_games", "unconverged_local_games", "fixed_policy_local_games",
    ),
    "ablation_fronts": (
        "variant", "seed", "solution_id", "purchase_cost", "defender_loss",
        "p_response", "prevention_cost", "detection_cost", "response_cost",
    ),
    "cross_evaluation": (
        "source_variant", "evaluation_variant", "seed", "solution_id",
        "purchase_cost", "defender_loss",
        *CONVERGENCE_COLUMNS, "local_games", "unconverged_local_games", "fixed_policy_local_games",
    ),
    "ablation_paths": ("variant", "portfolio_id", "path", "probability", *CONVERGENCE_COLUMNS),
    "model_checks": ("check", "passed", "value"),
}
TABLE_NAMES = tuple(REQUIRED_COLUMNS)
NUMERIC_COLUMNS = {
    "seed", "max_fe", "total_fe", "wall_time_s", "purchase_cost", "defender_loss",
    "attacker_payoff", "p_response", "prevention_cost", "detection_cost", "response_cost", "probability",
}


def output_directory(input_excel=None, output_dir=None):
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    return get_feature_output_dir("structural_ablation", input_excel)


def ensure_output_source(input_excel, output_dir=None):
    """Reject stem collisions before either cache or CSV data is overwritten."""
    source = get_input_excel(input_excel)
    directory = output_directory(source, output_dir)
    manifest_path = directory / CSV_FILES["structural_ablation"]["experiment_manifest"]
    if manifest_path.is_file():
        previous = pd.read_csv(manifest_path, dtype={"case_id": str, "input_sha256": str})
        if len(previous) != 1 or "input_excel" not in previous or not previous["input_excel"].map(get_input_excel).eq(source).all():
            raise ValueError("output directory already belongs to another source path")
    return directory


def validate_tables(tables, *, expected_input=None):
    """Check the CSV contract and provenance without reading the source workbook."""
    frames = {}
    for name in TABLE_NAMES:
        if name not in tables:
            raise ValueError(f"{name}: missing required table")
        frame = pd.DataFrame(tables[name]).copy()
        columns = (*IDENTITY_COLUMNS, *REQUIRED_COLUMNS[name])
        missing = [column for column in columns if column not in frame]
        if missing and not frame.empty:
            raise ValueError(f"{name}: missing required columns: {', '.join(missing)}")
        frame = frame.reindex(columns=list(dict.fromkeys((*columns, *frame.columns))))
        for column in NUMERIC_COLUMNS.intersection(frame.columns):
            values = pd.to_numeric(frame[column], errors="coerce")
            allowed_missing = (
                name == "fixed_portfolio_results"
                and column in {"defender_loss", "attacker_payoff"}
                and "status" in frame
            )
            invalid_numeric = frame[column].notna() & values.isna()
            if invalid_numeric.any():
                raise ValueError(f"{name}: {column} must contain finite numbers")
            finite_mask = np.isfinite(values.to_numpy(dtype=float))
            if allowed_missing:
                finite_mask |= frame["status"].astype(str).ne("ok").to_numpy() & values.isna().to_numpy()
            if not finite_mask.all():
                raise ValueError(f"{name}: {column} must contain finite numbers")
            frame[column] = values
        for column in ("probability", "p_response"):
            if column in frame and not frame[column].between(0.0, 1.0).all():
                raise ValueError(f"{name}: {column} must be within [0, 1]")
        identity_fields = ("variant", "evaluation_variant", "portfolio_id", "path", "status", "check")
        if name != "ablation_paths":
            identity_fields += ("source_variant", "solution_id")
        for column in identity_fields:
            if column in frame and (frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any()):
                raise ValueError(f"{name}: {column} must be nonempty")
        frames[name] = frame
    manifest = frames["experiment_manifest"]
    if len(manifest) != 1:
        raise ValueError("experiment_manifest: exactly one row is required")
    identity = manifest.iloc[0]
    source = get_input_excel(str(identity["input_excel"]))
    if identity["case_id"] != source.stem:
        raise ValueError("experiment_manifest: case_id disagrees with input_excel")
    digest = str(identity["input_sha256"])
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("experiment_manifest: input_sha256 must be a lowercase SHA256")
    if expected_input is not None and source != get_input_excel(expected_input):
        raise ValueError("experiment_manifest: input_excel does not match the requested source")
    for column in ("run_id", "baseline_protocol"):
        if pd.isna(identity[column]) or not str(identity[column]).strip():
            raise ValueError(f"experiment_manifest: {column} must be nonempty")
    if identity["baseline_protocol"] != MODEL_PROTOCOL_VERSION or identity["model_protocol_version"] != MODEL_PROTOCOL_VERSION:
        raise ValueError("obsolete model protocol; rerun structural_ablation solve before export or plot")
    if identity["model_options_protocol"] != STRUCTURAL_PROTOCOL_VERSION:
        raise ValueError("obsolete structural-ablation protocol; rerun solve before export or plot")
    for name, frame in frames.items():
        for column in IDENTITY_COLUMNS:
            if not frame[column].eq(identity[column]).all():
                raise ValueError(f"{name}: {column} disagrees with experiment_manifest")
        if name in ("fixed_portfolio_results", "cross_evaluation", "ablation_paths"):
            active = frame["status"].eq("ok") if name == "fixed_portfolio_results" else pd.Series(True, index=frame.index)
            if name == "ablation_paths":
                active = frame["iterations_executed"].notna()
            for column in ("iterations_executed", "stationarity_residual", "invasion_gap"):
                values = pd.to_numeric(frame.loc[active, column], errors="coerce")
                if not np.isfinite(values.to_numpy(dtype=float)).all() or (values < 0).any():
                    raise ValueError(f"{name}: {column} must contain finite non-negative diagnostics")
            if frame.loc[active, "stop_reason"].isna().any():
                raise ValueError(f"{name}: local games require stop_reason")
    checks = frames["model_checks"]
    if not checks["passed"].map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError("model_checks: passed must contain booleans")
    return frames


def export(result, input_excel=None, output_dir=None):
    """Save all seven validated CSVs, preserving every additional result column."""
    if not isinstance(result, dict) or "input_excel" not in result or "tables" not in result:
        raise ValueError("result must contain input_excel and tables")
    source = get_input_excel(result["input_excel"])
    if input_excel is not None and get_input_excel(input_excel) != source:
        raise ValueError("input_excel disagrees with the solve result")
    frames = validate_tables(result["tables"], expected_input=source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != frames["experiment_manifest"].iloc[0]["input_sha256"]:
        raise ValueError("input_sha256 disagrees with the current source file")
    directory = ensure_output_source(source, output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(directory / CSV_FILES["structural_ablation"][name], index=False)
    return frames


def main():
    parser = argparse.ArgumentParser(description="Export saved structural-ablation results")
    parser.add_argument("--input-excel", default=None)
    arguments = parser.parse_args()
    from common.pipeline import run_export
    run_export("structural_ablation", export, input_excel=arguments.input_excel)


if __name__ == "__main__":
    main()
