

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
from external.attack_graph import AttackGraph
from external.excel_processor import ExcelProcessor


def solve(input_excel=None):
    
    input_excel = get_input_excel(input_excel)
    parameters = load_case_parameters(input_excel, names=("Use_Pruning", "Game_Iterations", "Game_Learning_Rate"))
    nodes, adj_matrix, other_data = (
        ExcelProcessor.import_excel(input_excel)
    )

    ag = AttackGraph(nodes, adj_matrix)
    ag.build_state_transition_graph()

    ids_config = parse_ids_config(
        other_data["Set IDS"]
    )
    firewall_config = parse_firewall_config(
        other_data["Set firewall"]
    )
    patch_config = parse_patch_config(
        other_data["Patch vulnerability"], other_data["Vulnerability"]
    )

    results = run_simulation(
        state_graph=ag.state_paths,
        vuln_data=other_data["Vulnerability"],
        component_vuln_dist=other_data[
            "Vulnerabilities distribution"
        ],
        ids_config=ids_config,
        firewall_config=firewall_config,
        patch_config=patch_config,
        iterations=parameters["Game_Iterations"],
        learning_rate=parameters["Game_Learning_Rate"],
        prune_tol=resolve_prune_tolerance(parameters["Use_Pruning"]),
    )

    total_att, total_def, availability = (
        compute_total_expected_values(results)
    )

    return {
        "input_excel": str(input_excel),
        "results": results,
        "availability": availability,
        "total_attacker_payoff": total_att,
        "total_defender_loss": total_def,
        "use_pruning": parameters["Use_Pruning"],
        "prune_tol": resolve_prune_tolerance(parameters["Use_Pruning"]),
    }


if __name__ == "__main__":
    from common.pipeline import run_solve
    from features.initial_simulation.export_csv import export

    run_solve(
        "initial_simulation",
        solve,
        export
    )
    print("✅ Initial simulation CSV files exported.")
