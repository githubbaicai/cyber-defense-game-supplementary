

import pandas as pd

from common.config import CSV_FILES, get_feature_output_dir


COLUMNS = {
    "selected_points": [
        "Selection_Type",
        "Solution_ID",
        "Selected_Defenses",
        "Number_of_Defenses",
        "Total_Cost",
        "Original_Defender_Loss",
        "Original_Cost_Plus_Loss",
        "Resimulated_Attacker_Payoff",
        "Resimulated_Defender_Loss",
        "Resimulated_Cost_Plus_Loss",
        "P_Response",
        "Game_Iterations",
        "Game_Learning_Rate",
        "Loss_Difference",
        "Max_Tail_Change",
        "Tail_Stable",
    ],
    "selected_defenses": [
        "Selection_Type",
        "Solution_ID",
        "Defense_Label",
        "Defense_Cost",
        "Total_Cost",
        "Original_Defender_Loss",
        "Resimulated_Defender_Loss",
    ],
    "node_risk": [
        "Selection_Type",
        "Solution_ID",
        "Node_Path",
        "Compromise_Probability",
        "Attacker_Payoff",
        "Defender_Loss",
        "Tail_Change",
        "Tail_Stable",
    ],
    "game_strategy_history": [
        "Selection_Type",
        "Solution_ID",
        "Node_Path",
        "Generation",
        "Strategy",
        "Probability",
        "Player",
        "Probability_Type",
    ],
}

SUMMARY_FILES = {
    "summary_step_selection": "summary_step_selection.csv",
    "useful_game_strategy_history": "useful_game_strategy_history.csv",
}
EXCEL_FILENAME = "useful_path_analysis.xlsx"

COLUMNS["summary_step_selection"] = [
    "Selection_Type",
    "Solution_ID",
    "Node_Path",
    "Reach_Probability",
    "Rounded_Reach_Probability",
    "Has_Attack_Strategy",
    "Final_Attack_Mass",
    "Has_Active_Attack",
    "Has_Evolution_History",
    "Included",
    "Step_Order",
    "Exclusion_Reason",
]

COLUMNS["useful_game_strategy_history"] = [
    *COLUMNS["game_strategy_history"],
    "Step_Order",
]


def _export_excel(output, path):
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    steps = output["summary_step_selection"]
    useful = steps.loc[steps["Included"].eq(True)]
    history = output["useful_game_strategy_history"]

    sheets = {
        "selected_points": output["selected_points"],
        "selected_defenses": output["selected_defenses"],
        "node_risk": output["node_risk"],
        "useful_steps": useful,
    }

    for selection_type in (
        output["selected_points"]["Selection_Type"]
        .drop_duplicates()
    ):
        sheets[f"{selection_type}_history"] = history.loc[
            history["Selection_Type"].eq(selection_type)
        ]

    sheets["step_selection"] = steps

    with pd.ExcelWriter(
        path,
        engine="openpyxl",
    ) as writer:
        for name, frame in sheets.items():
            frame.to_excel(
                writer,
                sheet_name=name,
                index=False,
            )

            sheet = writer.sheets[name]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            sheet.sheet_view.showGridLines = False
            sheet.row_dimensions[1].height = 32

            for cell in sheet[1]:
                cell.fill = PatternFill(
                    "solid",
                    fgColor="17365D",
                )
                cell.font = Font(
                    color="FFFFFF",
                    bold=True,
                )
                cell.alignment = Alignment(
                    vertical="center",
                    wrap_text=True,
                )

            for column_index, column in enumerate(
                frame.columns, 1
            ):
                width = 22

                if column in (
                    "Node_Path",
                    "Strategy",
                    "Selected_Defenses",
                ):
                    width = 36
                elif column == "Exclusion_Reason":
                    width = 42

                sheet.column_dimensions[
                    get_column_letter(column_index)
                ].width = width

                if pd.api.types.is_numeric_dtype(
                    frame[column]
                ):
                    number_format = "0.000000E+00"

                    if column in (
                        "Solution_ID",
                        "Generation",
                        "Game_Iterations",
                        "Number_of_Defenses",
                        "Step_Order",
                    ):
                        number_format = "0"
                    elif column == "Rounded_Reach_Probability":
                        number_format = "0.00"

                    for cells in sheet.iter_rows(
                        min_row=2,
                        min_col=column_index,
                        max_col=column_index,
                    ):
                        cells[0].number_format = number_format


def export(result):
    history = result.get(
        "game_strategy_history",
        pd.DataFrame(),
    )

    if not history.empty and (
        not {
            "Player",
            "Probability_Type",
        }.issubset(history.columns)
        or set(
            history.loc[
                history["Player"] == "Defender",
                "Probability_Type",
            ]
        ) != {"defender_state"}
    ):
        raise ValueError(
            "The cache still contains legacy two-action response histories; rerun "
            "python -m features.pareto_path_analysis.solve --plot"
        )

    directory = get_feature_output_dir(
        "pareto_path_analysis",
        result.get("input_excel"),
    )
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    filenames = {
        **CSV_FILES["pareto_path_analysis"],
        **SUMMARY_FILES,
    }
    output = {}

    for name, columns in COLUMNS.items():
        frame = result.get(
            name,
            pd.DataFrame(),
        ).reindex(columns=columns)

        frame.to_csv(
            directory / filenames[name],
            index=False,
        )
        output[name] = frame

    if all(
        name in result
        for name in SUMMARY_FILES
    ):
        _export_excel(
            output,
            directory / EXCEL_FILENAME,
        )

    return output


if __name__ == "__main__":
    from common.pipeline import run_export

    run_export(
        "pareto_path_analysis",
        export,
    )