

from __future__ import annotations

import pandas as pd

from common.config import CSV_FILES, get_feature_output_dir
from common.memetic import LOCAL_STAT_COLUMNS
from common.selection_override import SELECTION_METADATA_COLUMNS
from common.implementation import IMPLEMENTATION_METADATA_COLUMNS
from features.nsga2_ma.annotations import (
    REFERENCE_POINT_COLUMNS,
    REFERENCE_POINTS_FILENAME,
    SELECTED_POINTS_FILENAME,
    select_cost_breakdown_points,
)


COLUMNS = {
    "pareto_front": [
        "Solution_ID", "Total_Cost", "Total_Defender_Loss",
        "P_Response",
    ],
    "detailed_items_list": [
        "Solution_ID", "Category", "Defense_Label", "Cost",
    ],
    "cost_breakdown_data": [
        "Prevention", "Detection", "Response", "TotalCost",
        "DefenderLoss", "P_Response", "Solution_ID",
    ],
    "optimization_history": [
        "Generation", "Total_FE", "Solution_ID",
        "Total_Cost", "Total_Defender_Loss",
    ],
    "game_strategy_history": [
        "Solution_ID", "Node_Path", "Generation",
        "Strategy", "Probability",
    ],
    "path_vulnerability_details": [
        "Solution_ID", "Path", "Path_Probability",
        "Component", "Vulnerability", "Description", "Base_Prob",
        "Patched?", "Firewall?", "Effective_Prob",
    ],
}

BASE_METADATA_COLUMNS = [
    "Budget",
    "Seed",
    "Population_Size",
    "Termination_Strategy",
    "Safety_Max_Generations",
    "Actual_Generations",
    "Total_FE",
    "Genetic_FE",
    "Local_FE",
    "Cache_Hits",
    "Unique_Simulations",
    "Selection_Experiment",
    "Selection_Mode",
    "Neighborhood",
    "Local_Search_Prob",
    "Elite_Size",
    "Elite_Quota",
    "Max_Local_Search_Iters",
    "No_Improve_Rounds",
    "Selection_File",
]

TERMINATION_COLUMNS = [
    "generation", "total_fe", "front_size", "window_ready",
    "front_window_distance", "min_generation_reached", "ma_ready",
    "hv", "hv_status", "hv_detail", "reference_valid", "stop_reason",
    "ideal_distance", "hv_range", "ideal_coordinate_range", "stability_met",
    "quality_met", "converged", "monitor_error",
]

TERMINATION_METADATA_COLUMNS = [
    "Termination_Metric", "Termination_Min_Generations",
    "Termination_Tolerance", "Termination_Stable_Generations",
    "Termination_Robust_Period", "Termination_Min_Front_Size",
    "Termination_Loss_Scale", "Termination_HV_Reference_Cost",
    "Termination_HV_Reference_Loss", "Main_Loop_Stop_Reason",
    "Main_Loop_Stop_Generation", "Main_Loop_Total_FE", "Postprocessing_FE",
    "Postprocessing_Front_Changed", "Final_Front_Window_Verified",
]

METADATA_COLUMNS = [
    *BASE_METADATA_COLUMNS,
    "Max_Local_FE_Ratio",
    "Max_Local_FE_Per_Gen",
    "Local_Search_Trigger",
    "Local_Search_Start_Ratio",
    "Acceptance_Mode",
    "MA_Parameters_JSON",
    *[
        name
        for name in LOCAL_STAT_COLUMNS
        if name not in BASE_METADATA_COLUMNS
    ],
    *SELECTION_METADATA_COLUMNS,
    *TERMINATION_METADATA_COLUMNS,
    *IMPLEMENTATION_METADATA_COLUMNS,
    "Safety_Max_FE",
    "Game_Iterations",
    "Game_Learning_Rate",
    "Pymoo_Version",
]


def _deduplicate_solution_tables(result):
    
    labels = list(result["defense_labels"])
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("defense_labels must be nonempty and unique")
    matrix_columns = ["Solution_ID", *result["defense_labels"], "P_Response"]
    front = result.get("pareto_front", pd.DataFrame()).reindex(columns=COLUMNS["pareto_front"])
    matrix = result.get("defense_selection_matrix", pd.DataFrame()).reindex(columns=matrix_columns)
    for name, frame in (("pareto_front", front), ("defense_selection_matrix", matrix)):
        if frame["Solution_ID"].isna().any() or not frame["Solution_ID"].is_unique:
            raise ValueError(f"{name}: Solution_ID values must be nonempty and unique")
    if set(front["Solution_ID"]) != set(matrix["Solution_ID"]):
        raise ValueError("Solution_ID values differ between the Pareto front and defense-selection matrix")
    if not matrix[labels].isin([0, 1]).all().all():
        raise ValueError("Measure columns in the defense-selection matrix must contain 0/1 values")

    ranked = front.merge(matrix[["Solution_ID", *labels]], on="Solution_ID", validate="one_to_one")
    representatives = (
        ranked.sort_values(["Total_Defender_Loss", "Total_Cost", "Solution_ID"], kind="stable")
        .drop_duplicates(subset=labels, keep="first")
        .sort_values(["Total_Cost", "Total_Defender_Loss", "Solution_ID"], kind="stable")
    )
    id_map = {old_id: new_id for new_id, old_id in enumerate(representatives["Solution_ID"], start=1)}
    all_ids = set(front["Solution_ID"])
    output = {}
    for key, columns in {
        **COLUMNS,
        "defense_selection_matrix": matrix_columns,
    }.items():
        frame = result.get(key, pd.DataFrame()).reindex(columns=columns)
        if key != "optimization_history":
            if frame["Solution_ID"].isna().any() or not set(frame["Solution_ID"]).issubset(all_ids):
                raise ValueError(f"{key}: invalid Solution_ID values")
            frame = frame.loc[frame["Solution_ID"].isin(id_map)].copy()
            frame["Solution_ID"] = frame["Solution_ID"].map(id_map).astype(int)
            frame = frame.sort_values("Solution_ID", kind="stable").reset_index(drop=True)
        output[key] = frame
    return output


def export(result):
    
    output = _deduplicate_solution_tables(result)
    annotations = {}
    if "cost_breakdown_selected_points" in result:
        annotations["cost_breakdown_selected_points"] = (
            SELECTED_POINTS_FILENAME,
            select_cost_breakdown_points(output["cost_breakdown_data"]),
        )
    if "cost_breakdown_reference_points" in result:
        annotations["cost_breakdown_reference_points"] = (
            REFERENCE_POINTS_FILENAME,
            result["cost_breakdown_reference_points"].reindex(columns=REFERENCE_POINT_COLUMNS),
        )
    output_directory = get_feature_output_dir("nsga2_ma", result.get("input_excel"))
    output_directory.mkdir(parents=True, exist_ok=True)

    for key, frame in output.items():
        frame.to_csv(
            output_directory / CSV_FILES["nsga2_ma"][key],
            index=False,
        )

    metadata = pd.DataFrame([
        result["metadata"]
    ]).reindex(columns=METADATA_COLUMNS)

    metadata.to_csv(
        output_directory / CSV_FILES["nsga2_ma"]["run_metadata"],
        index=False,
    )
    output["run_metadata"] = metadata
    if "termination_history" in result:
        history = result["termination_history"].reindex(columns=TERMINATION_COLUMNS)
        history.to_csv(
            output_directory / CSV_FILES["nsga2_ma"]["termination_history"],
            index=False,
        )
        output["termination_history"] = history
    for key, (filename, frame) in annotations.items():
        frame.to_csv(output_directory / filename, index=False)
        output[key] = frame
    return output


if __name__ == "__main__":
    from common.pipeline import run_export

    run_export("nsga2_ma", export)
