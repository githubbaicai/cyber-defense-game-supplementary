"""Export model-validation evidence without running the validation again."""

from __future__ import annotations

import argparse

import pandas as pd

from common.case_config import get_input_excel
from common.config import CSV_FILES, get_feature_output_dir


TABLE_COLUMNS = {
    "action_probabilities": (
        "Scenario_ID", "History", "Action_ID", "Action", "Target_Component",
        "Conditional_Probability", "Local_Expected_Loss",
    ),
    "prefix_contributions": (
        "Scenario_ID", "Parent_History", "Target_Component", "Child_History",
        "Parent_Probability", "Conditional_Probability", "Child_Probability",
        "Local_Weighted_Loss", "Expected_Contribution",
    ),
    "terminal_trajectories": (
        "Scenario_ID", "Trajectory_ID", "Action_Trajectory", "Component_History",
        "Stop_Reason", "Probability", "Cumulative_Expected_Loss",
        "Expected_Contribution",
    ),
    "validation_summary": (
        "Scenario_ID", "Check", "Actual", "Expected", "Abs_Error",
        "Tolerance", "Status", "Detail",
    ),
    "run_metadata": ("Key", "Value"),
}


def validate_table_columns(name, frame):
    """Require the shared CSV contract while preserving additional evidence."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    missing = [column for column in TABLE_COLUMNS[name] if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def export(result):
    """Write all evidence before reporting a failed numerical validation."""
    if result.get("schema_version") != 2:
        raise ValueError("This cache predates workbook-case validation; run model_validation solve again")
    case_path = get_input_excel(result["input_excel"])
    metadata = result["run_metadata"]
    validate_table_columns("run_metadata", metadata)
    for key, expected in (("Case_ID", case_path.stem), ("Input_Excel", str(case_path))):
        stored = metadata.loc[metadata["Key"].eq(key), "Value"]
        if len(stored) != 1 or str(stored.iloc[0]) != expected:
            raise ValueError(f"Cached model-validation {key} does not match its output case; solve again")
    directory = get_feature_output_dir("model_validation", result["input_excel"])
    frames = {}
    for name, columns in TABLE_COLUMNS.items():
        frame = result[name].copy()
        if frame.empty:
            frame = frame.reindex(columns=tuple(dict.fromkeys((*columns, *frame.columns))))
        validate_table_columns(name, frame)
        if name != "run_metadata" and not frame.empty:
            if "Case_ID" not in frame or not frame["Case_ID"].eq(case_path.stem).all():
                raise ValueError(f"{name} contains a different case; solve again")
            if "Input_Excel" not in frame or not frame["Input_Excel"].eq(str(case_path)).all():
                raise ValueError(f"{name} contains a different workbook; solve again")
        extra_columns = [column for column in frame.columns if column not in columns]
        frames[name] = frame.loc[:, [*columns, *extra_columns]]

    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(
            directory / CSV_FILES["model_validation"][name],
            index=False,
            encoding="utf-8-sig",
        )

    summary = frames["validation_summary"]
    statuses = summary["Status"].astype(str).str.strip().str.lower()
    if not result["passed"] or summary.empty or not statuses.eq("pass").all():
        failures = summary.loc[~statuses.eq("pass"), ["Scenario_ID", "Check"]]
        detail = "; ".join(f"{row.Scenario_ID}: {row.Check}" for row in failures.itertuples())
        raise ValueError(f"Model validation failed; evidence exported to {directory}. {detail}".strip())
    return frames


def main():
    """Export only the existing result cache selected by the workbook path."""
    from common.pipeline import run_export

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-excel", default=None)
    arguments = parser.parse_args()
    run_export("model_validation", export, input_excel=arguments.input_excel)


if __name__ == "__main__":
    main()
