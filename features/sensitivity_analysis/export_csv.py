"""Export and read the CSV-only sensitivity-analysis contract."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common.config import CSV_FILES, get_feature_output_dir
from features.sensitivity_analysis.config import PROTOCOL_VERSION
from features.sensitivity_analysis.parameters import PARAMETER_NAMES


_IDENTITY = ["Case_ID", "Run_ID"]
TABLE_COLUMNS = {
    "parameters": _IDENTITY + [
        "Record_Type", "Factor_Index", "Parameter", "Display_Name", "Sheet", "Column",
        "Nominal_Value", "Lower_Bound", "Upper_Bound", "Distribution", "Transformation",
        "Dependence_Group", "Range_Source", "Unit", "Evaluation_Period", "Currency",
        "Source_Type", "Source_Location", "Extraction_Rule", "Calibration_Status",
        "Validation_Status", "CVE", "EPSS", "EPSS_Date", "EPSS_Probability_Lower", "EPSS_Probability_Upper",
    ],
    "parameter_samples": _IDENTITY + [
        "Design_ID", "Base_N", "Sample_ID", "Saltelli_Block_ID", "Base_Index", "Sample_Role",
        "Factor_Index", "D", "Calc_Second_Order", *PARAMETER_NAMES, "Exposure_Penalty_Ratio", "Response_Penalty_Ratio",
    ],
    "sample_metrics": _IDENTITY + [
        "Record_Type", "Design_ID", "Base_N", "Sample_ID", "Sample_Role", "Scenario_ID", "Seed",
        "Repeat", "Status", "Fixed_Cost", "Fixed_Loss", "Fixed_J", "Fixed_Feasible",
        "Reoptimized_Cost", "Reoptimized_Loss", "Reoptimized_J", "Resource_ID", "Decision_ID",
        "Resource_Reselected", "Jaccard", "Hamming", "P_Response_Change", "Search_Gap",
        "Budget_Threshold", "Envelope_Loss", "Total_FE", "Local_FE", "Cache_Hits",
        "Unique_Simulations", "Exposure_Penalty_Ratio", "Response_Penalty_Ratio", "Error",
    ],
    "sobol_indices": _IDENTITY + [
        "Design_ID", "Base_N", "Metric", "Parameter", "Parameter_2", "S1", "S1_Conf", "ST",
        "ST_Conf", "S2", "S2_Conf", "S1_CI_Lower", "S1_CI_Upper", "ST_CI_Lower", "ST_CI_Upper",
        "Status", "CI_Level", "CI_Convention",
    ],
    "critical_parameters": _IDENTITY + [
        "Design_ID", "Base_N", "Metric", "Parameter", "S1", "S1_Conf", "ST", "ST_Conf", "Rank", "Status",
    ],
    "run_metadata": _IDENTITY + [
        "Input_Excel", "Input_SHA256", "Config_SHA256", "Protocol_Version",
        "Implementation_Version", "Model_Protocol_Version", "Status",
        "Budget", "Main_Budget", "Sensitivity_Budget", "Monetary_Unit", "Lambda_Loss", "Dimension",
        "Parameter_Names", "Nominal_Model_Options", "Configuration", "Dependencies",
    ],
    "fronts": _IDENTITY + [
        "Design_ID", "Base_N", "Sample_ID", "Sample_Role", "Scenario_ID", "Seed", "Repeat", "Front_Role",
        "Resource_ID", "Decision_ID", "Allocation_Labels", "Selected_Defenses", "Decision_Vector", "P_Response",
        "RP_Level", "Cost", "Loss", "J", "Constraint", "Feasible", "Exposure_Penalty_Ratio", "Response_Penalty_Ratio", "Status",
    ],
    "portfolio_evaluations": _IDENTITY + [
        "Design_ID", "Base_N", "Sample_ID", "Sample_Role", "Scenario_ID", "Resource_ID", "Decision_ID",
        "Evaluation_Role", "P_Response", "RP_Level", "Allocation_Labels", "Selected_Defenses", "Decision_Vector",
        "Cost", "Loss", "J", "Constraint", "Feasible", "Candidate_Rank", "Candidate_Nondominated",
        "Exposure_Penalty_Ratio", "Response_Penalty_Ratio", "Status",
    ],
    "sobol_diagnostics": _IDENTITY + [
        "Record_Type", "Design_ID", "Base_N", "Sample_ID", "Sample_Role", "Scenario_ID", "Seed", "Metric",
        "Parameter", "S1", "S1_Conf", "ST", "ST_Conf", "Value", "Status", "Detail", "Reason",
        "Exposure_Penalty_Ratio", "Response_Penalty_Ratio",
    ],
}


def output_directory(input_excel=None, output_dir=None):
    if output_dir is not None:
        return Path(output_dir)
    return get_feature_output_dir("sensitivity_analysis", input_excel)


def _csv_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return value


def _table(result, name):
    values = result.get("metadata", {}) if name == "run_metadata" else result.get(name, [])
    if isinstance(values, pd.DataFrame):
        frame = values.copy()
    elif isinstance(values, dict):
        frame = pd.DataFrame([values]) if values else pd.DataFrame()
    else:
        frame = pd.DataFrame(values)
    columns = list(TABLE_COLUMNS[name])
    columns.extend(column for column in frame.columns if column not in columns)
    frame = frame.reindex(columns=columns)
    for column in frame.select_dtypes(include="object"):
        frame[column] = frame[column].map(_csv_value)
    return frame


def export(result, output_dir=None):
    """Write all nine tables; this function never samples, optimizes, or evaluates."""
    metadata = result.get("metadata", {})
    has_data = any(name in result for name in TABLE_COLUMNS if name != "run_metadata")
    if has_data and metadata.get("Protocol_Version") != PROTOCOL_VERSION:
        raise ValueError("Legacy or incompatible sensitivity cache protocol; rerun sensitivity solve.")
    tables = {name: _table(result, name) for name in TABLE_COLUMNS}
    _validate_tables(tables)
    directory = output_directory(result.get("input_excel"), output_dir if output_dir is not None else result.get("output_dir"))
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(directory / CSV_FILES["sensitivity_analysis"][name], index=False, float_format="%.17g")
    return tables


def _read_csv(path):
    return pd.read_csv(path, float_precision="round_trip", dtype={
        field: str for field in ("Case_ID", "Run_ID", "Design_ID", "Resource_ID", "Decision_ID")
    })


def _validate_tables(tables):
    metadata = tables["run_metadata"]
    if len(metadata) != 1 or not metadata["Protocol_Version"].eq(PROTOCOL_VERSION).all():
        raise ValueError("Sensitivity CSV protocol is missing or incompatible; rerun solve and export.")
    run_ids = metadata["Run_ID"].dropna().astype(str).unique()
    if len(run_ids) != 1:
        raise ValueError("Sensitivity CSV metadata must identify exactly one run.")
    case_ids = metadata["Case_ID"].dropna().astype(str).unique()
    if len(case_ids) != 1:
        raise ValueError("Sensitivity CSV metadata must identify exactly one case.")
    dimension = pd.to_numeric(metadata["Dimension"], errors="coerce").dropna()
    if dimension.empty or dimension.iloc[0] != len(PARAMETER_NAMES):
        raise ValueError("Sensitivity CSV dimension is stale; rerun sensitivity solve.")
    names = metadata.get("Parameter_Names", pd.Series(dtype=str)).dropna()
    if names.empty:
        raise ValueError("Sensitivity CSV parameter order is missing; rerun sensitivity solve.")
    try:
        parsed_names = json.loads(names.iloc[0])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Sensitivity CSV parameter order is invalid; rerun sensitivity solve.") from exc
    if not isinstance(parsed_names, list) or parsed_names != list(PARAMETER_NAMES):
        raise ValueError("Sensitivity CSV parameter order is stale; rerun sensitivity solve.")
    for name, frame in tables.items():
        if not frame.empty:
            for field, expected in (("Run_ID", run_ids[0]), ("Case_ID", case_ids[0])):
                if (field not in frame or frame[field].isna().any()
                        or not frame[field].astype(str).eq(expected).all()):
                    raise ValueError(f"{name}: missing, mixed or stale sensitivity {field}.")
        if any(column.startswith("Mapping_") for column in frame.columns):
            raise ValueError(f"{name}: legacy mapping columns are not valid for v3.")
    samples = tables["parameter_samples"]
    if not samples.empty:
        if not _sample_dimension_valid(samples):
            raise ValueError("Sensitivity parameter samples have a stale dimension or missing factors.")


def _sample_dimension_valid(samples):
    if "D" not in samples or not pd.to_numeric(samples["D"], errors="coerce").eq(len(PARAMETER_NAMES)).all():
        return False
    for name in (*PARAMETER_NAMES, "Exposure_Penalty_Ratio", "Response_Penalty_Ratio"):
        if name not in samples or pd.to_numeric(samples[name], errors="coerce").isna().any():
            return False
    return True


def read_tables(directory):
    """Read and validate one v3 run; old/mixed caches are deliberately rejected."""
    directory = Path(directory)
    tables = {name: _read_csv(directory / CSV_FILES["sensitivity_analysis"][name]) for name in TABLE_COLUMNS}
    _validate_tables(tables)
    return tables


if __name__ == "__main__":
    from common.pipeline import run_export
    run_export("sensitivity_analysis", export)
