

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
from time import perf_counter

import pandas as pd

from common.config import CSV_FILES, get_feature_output_dir
from common.memetic import LOCAL_STAT_COLUMNS
from common.selection_override import SELECTION_METADATA_COLUMNS
from common.implementation import IMPLEMENTATION_METADATA_COLUMNS


BASE_COLUMNS = {
    "performance_metrics_raw": [
        "Algorithm",
        "Run",
        "Seed",
        "Termination_Strategy",
        "Generations",
        "Total_FE",
        "Local_FE",
        "Cache_Hits",
        "Unique_Simulations",
        "MA_Parameters_JSON",
        "Pareto_Size",
        "Runtime_Seconds",
        "HV",
        "IGD",
        "GD",
        "Reference_Type",
        "HV_Reference_Cost",
        "HV_Reference_Loss",
        "Normalization_Ideal_Cost",
        "Normalization_Ideal_Loss",
        "Normalization_Span_Cost",
        "Normalization_Span_Loss",
    ],
    "performance_summary": [
        "Algorithm",
        "Runs",
        "HV_Mean",
        "HV_Std",
        "IGD_Mean",
        "IGD_Std",
        "GD_Mean",
        "GD_Std",
        "Runtime_Seconds_Mean",
        "Runtime_Seconds_Std",
        "Pareto_Size_Mean",
        "Pareto_Size_Std",
        "Generations_Mean",
        "Generations_Std",
        "Total_FE_Mean",
        "Total_FE_Std",
        "Local_FE_Mean",
        "Local_FE_Std",
    ],
    "all_pareto_points": [
        "Algorithm", "Run", "Point_ID",
        "Total_Cost", "Total_Defender_Loss", "X_Bits", "P_Response",
    ],
    "best_pareto_fronts": [
        "Algorithm", "Run", "Point_ID",
        "Total_Cost", "Total_Defender_Loss", "X_Bits", "P_Response",
    ],
    "runtime_statistics": [
        "Algorithm", "Mean_Seconds", "Std_Seconds",
        "Min_Seconds", "Max_Seconds",
    ],
    "statistical_tests": [
        "Metric", "Algorithm_A", "Algorithm_B",
        "N_A", "N_B", "Statistic_U",
        "P_Value_Unadjusted", "Status",
    ],
    "optimization_history": [
        "Algorithm", "Run", "Generation", "Total_FE",
        "Feasible_Pareto_Size", "Minimum_Cost", "Minimum_Loss",
    ],
}

EXTRA_SUMMARY_METRICS = (
    "Genetic_FE",
    "Local_Accepted",
    "Interrupted_Local_Rounds",
    "Neighbor_Scans",
    "Local_Improved_Center",
    "Local_Entered_Final_Front",
    "Local_Dominates_Center",
    "Final_Refill_Genetic_FE",
    "Cache_Hits",
    "Unique_Simulations",
)

COLUMNS = {
    name: list(columns)
    for name, columns in BASE_COLUMNS.items()
}

COLUMNS["performance_metrics_raw"].extend([
    "Genetic_FE",
    "Max_Local_FE_Ratio",
    "Local_Search_Trigger",
    "Acceptance_Mode",
    *[
        name
        for name in LOCAL_STAT_COLUMNS
        if name not in BASE_COLUMNS["performance_metrics_raw"]
    ],
    *SELECTION_METADATA_COLUMNS,
    "Main_Loop_Stop_Reason", "Main_Loop_Total_FE", "Postprocessing_FE",
    *IMPLEMENTATION_METADATA_COLUMNS,
])

COLUMNS["performance_summary"].extend([
    f"{metric}_{suffix}"
    for metric in EXTRA_SUMMARY_METRICS
    for suffix in ("Mean", "Std")
])

COLUMNS["optimization_history"].extend([
    "Genetic_FE",
    "Local_FE",
])

COLUMNS["reference_front"] = [
    "Experiment_ID", "Point_ID", "Total_Cost", "Total_Defender_Loss", "Reference_ID",
]
COLUMNS["experiment_manifest"] = [
    "Experiment_ID", "Protocol", "Budget_ID", "Input_Excel", "Excel_SHA256",
    "Implementation_Version", "Model_Protocol_Version", "Comparison_Protocol_Version",
    "Experiment_Wall_Seconds", "Sum_Run_Wall_Seconds", "Offline_Metrics_Seconds",
]


def result_directory(result):
    if result.get("output_dir"):
        return Path(result["output_dir"])
    directory = get_feature_output_dir("multi_algorithm", result.get("input_excel"))
    return directory / "experiments" / result["experiment_id"] if result.get("experiment_id") else directory


def export(result, output_dir=None):
    
    started = perf_counter()
    output_directory = Path(output_dir).expanduser().resolve() if output_dir is not None else result_directory(result)
    if output_dir is not None:
        result = {**result, "output_dir": str(output_directory)}
    output_directory.mkdir(parents=True, exist_ok=True)
    output = {}

    for name, columns in COLUMNS.items():
        frame = result.get(name, pd.DataFrame())
        columns = list(dict.fromkeys([*columns, *frame.columns]))
        frame = frame.reindex(columns=columns)

        frame.to_csv(
            output_directory / CSV_FILES["multi_algorithm"][name],
            index=False,
        )
        output[name] = frame

    if result.get("experiment_id") or output_dir is not None:
        with (output_directory / "solve_result.pkl").open("wb") as stream:
            pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
    if result.get("experiment_id"):
        pd.DataFrame([{
            "Experiment_ID": result["experiment_id"],
            "Export_Seconds": perf_counter() - started,
            "Timing_Scope": "CSV and experiment cache serialization; excludes optimization and plotting",
        }]).to_csv(output_directory / "export_timing.csv", index=False)
        case_directory = get_feature_output_dir("multi_algorithm", result.get("input_excel"))
        if output_directory.resolve().parent == (case_directory / "experiments").resolve():
            pd.DataFrame([{"Experiment_ID": result["experiment_id"], "Output_Directory": str(output_directory.resolve())}]).to_csv(
                case_directory / "latest_experiment.csv", index=False,
            )
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="Re-export saved comparison results only")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--input-excel", type=Path)
    args = parser.parse_args(argv)
    if args.output_dir is not None:
        with (args.output_dir / "solve_result.pkl").open("rb") as stream:
            return export(pickle.load(stream), output_dir=args.output_dir)
    else:
        pointer = get_feature_output_dir("multi_algorithm", args.input_excel) / "latest_experiment.csv"
        if pointer.is_file():
            directory = Path(pd.read_csv(pointer).iloc[0]["Output_Directory"])
            with (directory / "solve_result.pkl").open("rb") as stream:
                return export(pickle.load(stream), output_dir=directory)
        else:
            from common.pipeline import run_export
            return run_export("multi_algorithm", export, input_excel=args.input_excel)


if __name__ == "__main__":
    main()
