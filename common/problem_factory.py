

from common.case_config import get_input_excel


def load_problem_context(input_excel=None, *, allow_no_attack=False):
    
    path = get_input_excel(input_excel)
    if not path.is_file():
        raise FileNotFoundError(
            f"Project input file not found: {path}"
        )

    from external.attack_graph import AttackGraph
    from external.excel_processor import ExcelProcessor

    nodes, adjacency, all_data = ExcelProcessor.import_excel(path, allow_no_attack=allow_no_attack)
    graph = AttackGraph(nodes, adjacency)
    graph.build_state_transition_graph()

    labels = all_data["Allocation"]["Label"]
    costs = all_data["Allocation"]["Cost"]
    if len(labels) != len(costs):
        raise ValueError("Allocation Label and Cost lengths differ")

    defense_items = list(zip(labels, costs))
    if not defense_items:
        raise ValueError("Allocation contains no defense measures")

    return defense_items, all_data, graph
