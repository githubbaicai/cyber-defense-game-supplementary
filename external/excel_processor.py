

import pandas as pd
import numpy as np


REQUIRED_COLUMNS = {
    "Label": ("Label",),
    "Vulnerabilities distribution": ("Component", "Vulnerability"),
    "Allocation": ("Label", "Cost"),
    "Patch vulnerability": ("Label", "Patching effectiveness"),
    "Set firewall": ("Label", "Position", "Firewall effectiveness"),
    "Set IDS": ("Label", "Position", "Detection Cycle (h)", "Detection effectiveness"),
    "Vulnerability": (
        "Vulnerability", "Attack Success Probability", "Attack time (h)",
        "Attack cost (monetary)", "Attack Benefit (monetary)",
        "Attack benefit (destruction)", "Weight_ABM", "Weight_ABD",
        "Defend loss (monetary)", "Defend loss (destruction)", "Weight_DLM", "Weight_DLD",
        *(f"Defense Success Probability after detecting ({i})" for i in range(4)),
    ),
}


class ExcelProcessor:
    

    @staticmethod
    def process_adjacency_matrix(df):
        
        df = df.dropna(how="all").dropna(axis=1, how="all")
        nodes = [str(value).strip() for value in df.columns[1:]]
        rows = [str(value).strip() for value in df.iloc[:, 0]]
        if not nodes or len(set(nodes)) != len(nodes) or rows != nodes:
            raise ValueError("Attack graph must be a square matrix with unique row and column nodes in matching order")
        matrix = df.iloc[:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
        if matrix.shape != (len(nodes), len(nodes)) or not np.isin(matrix, [0, 1]).all():
            raise ValueError("Attack graph must be a 0/1 adjacency matrix without missing values")
        return nodes, matrix.astype(int).tolist()

    @staticmethod
    def process_standard_sheet(df):
        
        aligned = df.dropna(how="all").astype(object)
        aligned = aligned.where(pd.notna(aligned), None)
        return {column: aligned[column].tolist() for column in aligned.columns}

    @classmethod
    def import_excel(cls, file_path, *, allow_no_attack=False):
        
        all_sheets = pd.read_excel(file_path, sheet_name=None)

        canonical_names = {name.casefold(): name for name in ("Attack graph", *REQUIRED_COLUMNS, "Strategy")}
        sheets = {}
        for name, frame in all_sheets.items():
            clean_name = canonical_names.get(name.strip().casefold(), name.strip())
            if clean_name in sheets:
                raise ValueError(f"Duplicate sheet name: {clean_name}")
            frame = frame.rename(columns=lambda column: str(column).strip())
            if frame.columns.duplicated().any():
                raise ValueError(f"{clean_name}: duplicate column names")
            sheets[clean_name] = frame.dropna(how="all")
        if "Attack graph" not in sheets:
            raise ValueError("The Attack graph sheet is missing; the first sheet is no longer assumed to be the attack graph")
        for name, columns in REQUIRED_COLUMNS.items():
            if name not in sheets:
                raise ValueError(f"Missing sheet: {name}")
            missing = set(columns) - set(sheets[name].columns)
            if missing:
                raise ValueError(f"{name}: missing required columns {sorted(missing)}")
            nonempty_columns = [column for column in columns
                                if (name, column) != ("Vulnerabilities distribution", "Vulnerability")]
            if sheets[name][nonempty_columns].isna().any().any():
                raise ValueError(f"{name}: required columns contain missing values; complete the corresponding rows")

        adj_matrix_df = sheets.pop("Attack graph")
        nodes, adjacency_matrix = (
            cls.process_adjacency_matrix(adj_matrix_df)
        )

        other_sheets_data = {
            sheet_name: cls.process_standard_sheet(
                frame
            )
            for sheet_name, frame in sheets.items()
        }

        from common.case_validation import validate_case_data

        validate_case_data(nodes, adjacency_matrix, other_sheets_data, allow_no_attack=allow_no_attack)
        return nodes, adjacency_matrix, other_sheets_data
