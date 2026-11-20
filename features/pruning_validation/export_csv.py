"""Export pruning validation tables without recomputing results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from common.config import CSV_FILES, get_feature_output_dir


COLUMNS = {
    "case_summary": [
        "Case_ID", "Case_Name", "Selected_Count", "Initial_Conditional_Response_Probability", "Node_Count_Pruned", "Node_Count_NoPrune",
        "Pruned_Node_Count", "Loss_Pruned", "Loss_NoPrune", "Abs_Diff",
        "Relative_Diff", "Attacker_Payoff_Pruned", "Attacker_Payoff_NoPrune",
        "Payoff_Abs_Diff", "Payoff_Relative_Diff", "Status",
    ],
    "case_details": ["Case_ID", "Defense_Label", "Cost", "Is_Selected"],
}


EVOLUTION_FILES = {
    "game_strategy_history": "game_strategy_history.csv",
    "strategy_probabilities": "strategy_probabilities.csv",
    "evolution_nodes": "evolution_nodes.csv",
    "defender_strategy_history": "defender_strategy_history.csv",
    "response_policy_history": "response_policy_history.csv",
    "defender_strategy_probabilities": "defender_strategy_probabilities.csv",
}


def evolution_directory(input_excel, case_id):
    return get_feature_output_dir("pruning_validation", input_excel) / Path(input_excel).stem / f"case_{case_id}_evolution"


def export(result):
    directory = get_feature_output_dir("pruning_validation", result.get("input_excel"))
    directory.mkdir(parents=True, exist_ok=True)
    output = {}
    for name, columns in COLUMNS.items():
        frame = result.get(name, pd.DataFrame()).reindex(columns=columns)
        path = directory / CSV_FILES["pruning_validation"][name]
        frame.to_csv(path, index=False)
        output[name] = frame
    case_directory = directory / Path(result["input_excel"]).stem
    case_directory.mkdir(parents=True, exist_ok=True)
    for name in ("case_summary", "case_details", "case_timings"):
        result[name].to_csv(case_directory / f"{name}.csv", index=False)
    case_ids = result.get("evolution_case_ids", ())
    if not case_ids and result.get("evolution_case_id") is not None:
        case_ids = (result["evolution_case_id"],)
    for case_id in case_ids:
        directory = evolution_directory(result["input_excel"], case_id)
        directory.mkdir(parents=True, exist_ok=True)
        for name, filename in EVOLUTION_FILES.items():
            frame = result[name]
            if "Case_ID" in frame.columns:
                frame = frame.loc[frame["Case_ID"] == case_id]
            frame.to_csv(directory / filename, index=False)
            output[name] = frame
        summary = result["case_summary"]
        summary.loc[summary["Case_ID"] == case_id].to_csv(
            directory / "case_summary.csv", index=False
        )
        details = result["case_details"]
        details.loc[details["Case_ID"] == case_id].to_csv(
            directory / "selected_defenses.csv", index=False
        )
    return output


if __name__ == "__main__":
    from common.pipeline import run_export
    run_export("pruning_validation", export)
