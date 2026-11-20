

import numpy as np
import pandas as pd

from common.case_config import get_input_excel, load_case_parameters
from common.parsers import (
    parse_firewall_config,
    parse_ids_config,
    parse_patch_config,
)
from common.simulation import (
    compute_total_expected_values,
    resolve_prune_tolerance,
    run_simulation,
)
from external.attack_graph import (
    AttackGraph,
)
from external.excel_processor import (
    ExcelProcessor,
)
from features.state_visualization.attack_graph_data import build_attack_graph_tables


def _path_depth(
    path
):
    
    return len(
        str(path).split(
            " → "
        )
    )


def solve(input_excel=None):
    
    input_excel = get_input_excel(input_excel)
    parameters = load_case_parameters(input_excel, names=("Use_Pruning", "Game_Iterations", "Game_Learning_Rate"))
    nodes, adjacency, data = (
        ExcelProcessor.import_excel(
            input_excel
        )
    )
    attack_graph_nodes, attack_graph_edges = build_attack_graph_tables(
        nodes, adjacency, data["Vulnerability"], data["Vulnerabilities distribution"]
    )

    graph = AttackGraph(
        nodes,
        adjacency
    )
    graph.build_state_transition_graph()

    results = run_simulation(
        state_graph=graph.state_paths,
        vuln_data=data[
            "Vulnerability"
        ],
        component_vuln_dist=data[
            "Vulnerabilities distribution"
        ],
        ids_config=parse_ids_config(
            data["Set IDS"]
        ),
        firewall_config=(
            parse_firewall_config(
                data["Set firewall"]
            )
        ),
        patch_config=(
            parse_patch_config(
                data[
                    "Patch vulnerability"
                ], data["Vulnerability"]
            )
        ),
        iterations=(
            parameters["Game_Iterations"]
        ),
        learning_rate=(
            parameters["Game_Learning_Rate"]
        ),
        prune_tol=resolve_prune_tolerance(parameters["Use_Pruning"]),
    )

    total_attacker, total_defender, availability = (
        compute_total_expected_values(
            results
        )
    )

    node_rows = []
    for path in sorted(
        set(
            availability
        )
        | set(
            results
        )
    ):
        info = results.get(
            path,
            {}
        )

        node_rows.append({
            "Node_Path": path,
            "Depth": (
                _path_depth(
                    path
                )
            ),
            "Compromise_Probability": float(
                availability.get(
                    path,
                    np.nan
                )
            ),
            "Attack_Time_h": float(
                info.get(
                    "attack_time",
                    np.nan
                )
            ),
            "Attacker_Payoff": float(
                info.get(
                    "attacker_payoff",
                    np.nan
                )
            ),
            "Defender_Loss": float(
                info.get(
                    "defender_loss",
                    np.nan
                )
            ),
        })

    edge_rows = []
    for source, target in (
        graph.state_paths
    ):
        edge_rows.append({
            "Source_State": source,
            "Target_State": target,
            "Source_Probability": (
                float(
                    availability.get(
                        source,
                        np.nan
                    )
                )
            ),
            "Target_Probability": (
                float(
                    availability.get(
                        target,
                        np.nan
                    )
                )
            ),
        })


    terminal_paths = [
        path
        for path in availability
        if str(path).split(" → ")[-1] == "T"
    ]

    reach_target_directly = True
    if not terminal_paths:
        reach_target_directly = False
        
        t_predecessors = {
            src.split(" → ")[-1]
            for src, dst in graph.state_paths
            if dst.split(" → ")[-1] == "T"
        }
        
        
        candidate_paths = [
            path for path in availability
            if path != "S" and path.split(" → ")[-1] in t_predecessors
        ]
        
        if not candidate_paths:
            
            candidate_paths = [path for path in availability if path != "S"]

        if not candidate_paths:
            raise ValueError("All attack states are unreachable; the main attack path is undefined")
            
        terminal_paths = candidate_paths

    
    main_terminal = max(
        terminal_paths,
        key=lambda path: (
            float(availability.get(path, 0.0)),
            -_path_depth(path),
            str(path)
        )
    )

    
    components = str(main_terminal).split(" → ")
    
    
    if not reach_target_directly and components[-1] != "T":
        target_terminal_path = f"{main_terminal} → T"
    else:
        target_terminal_path = main_terminal

    prefix_rows = []
    for index in range(len(components)):
        prefix = " → ".join(components[: index + 1])
        prefix_rows.append({
            "Step": index,
            "Node_Path": prefix,
            "Component": components[index],
            "Compromise_Probability": float(
                availability.get(prefix, np.nan)
            ),
            "Terminal_Path": target_terminal_path,
        })

    
    if not reach_target_directly and components[-1] != "T":
        last_prob = float(availability.get(main_terminal, 0.0))
        prefix_rows.append({
            "Step": len(components),
            "Node_Path": target_terminal_path,
            "Component": "T",
            "Compromise_Probability": last_prob,
            "Terminal_Path": target_terminal_path,
        })
    availability_rows = [
        {
            "Node_Path": path,
            "Probability": float(
                probability
            ),
            "Total_Attacker_Payoff": float(
                total_attacker
            ),
            "Total_Defender_Loss": float(
                total_defender
            ),
        }
        for path, probability in (
            availability.items()
        )
    ]

    
    state_graph_edges = [
        {
            "Source_State": row[
                "Source_State"
            ],
            "Target_State": row[
                "Target_State"
            ],
        }
        for row in edge_rows
    ]

    return {
        "input_excel": str(input_excel),
        "use_pruning": parameters["Use_Pruning"],
        "prune_tol": resolve_prune_tolerance(parameters["Use_Pruning"]),
        "attack_graph_nodes": attack_graph_nodes,
        "attack_graph_edges": attack_graph_edges,
        "node_metrics": pd.DataFrame(
            node_rows
        ),
        "edge_probabilities": (
            pd.DataFrame(
                edge_rows
            )
        ),
        "main_attack_path": (
            pd.DataFrame(
                prefix_rows
            )
        ),
        "state_graph_edges": (
            pd.DataFrame(
                state_graph_edges
            )
        ),
        "availability": pd.DataFrame(
            availability_rows
        ),
    }


if __name__ == "__main__":
    from common.pipeline import (
        run_solve,
    )
    from features.state_visualization.export_csv import (
        export,
    )

    run_solve(
        "state_visualization",
        solve,
        export
    )
