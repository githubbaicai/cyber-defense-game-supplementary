

import warnings

import pandas as pd

from common.config import (
    CSV_FILES,
    get_feature_output_dir,
)
from features.state_visualization.attack_graph_data import EDGE_COLUMNS, NODE_COLUMNS

COLUMNS = {
    "attack_graph_nodes": NODE_COLUMNS,
    "attack_graph_edges": EDGE_COLUMNS,
    "node_metrics": [
        "Node_Path",
        "Depth",
        "Compromise_Probability",
        "Attack_Time_h",
        "Attacker_Payoff",
        "Defender_Loss",
    ],
    "edge_probabilities": [
        "Source_State",
        "Target_State",
        "Source_Probability",
        "Target_Probability",
    ],
    "main_attack_path": [
        "Step",
        "Node_Path",
        "Component",
        "Compromise_Probability",
        "Terminal_Path",
    ],
    "state_graph_edges": [
        "Source_State",
        "Target_State",
    ],
    "availability": [
        "Node_Path",
        "Probability",
        "Total_Attacker_Payoff",
        "Total_Defender_Loss",
    ],
}


def export(result):
    
    directory = get_feature_output_dir("state_visualization", result.get("input_excel"))
    directory.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {}
    attack_tables = {"attack_graph_nodes", "attack_graph_edges"}
    missing_attack_tables = attack_tables - result.keys()
    if missing_attack_tables:
        warnings.warn(
            "The solver cache is missing the new attack-graph data; rerun state_visualization solve/export; "
            "this export retains the available state-analysis data and clears both attack-graph CSV files to avoid reusing unrelated results.",
            UserWarning,
            stacklevel=2,
        )
    for name, columns in (
        COLUMNS.items()
    ):
        frame = (pd.DataFrame() if missing_attack_tables and name in attack_tables
                 else result.get(name, pd.DataFrame())).reindex(columns=columns)
        frame.to_csv(
            directory
            / CSV_FILES[
                "state_visualization"
            ][name],
            index=False
        )
        output[name] = frame

    return output


if __name__ == "__main__":
    from common.pipeline import (
        run_export,
    )

    run_export(
        "state_visualization",
        export
    )
