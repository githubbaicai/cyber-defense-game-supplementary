

import json

import numpy as np
import pandas as pd

from common.parsers import component_vulnerability_names

NODE_COLUMNS = [
    "Node_ID", "Display_Label", "Node_Type", "Component",
    "Vulnerability_ID", "Local_Index", "Order",
]
EDGE_COLUMNS = ["Source_ID", "Target_ID", "Edge_Type"]


def _node_id(kind, *names):
    return json.dumps([kind, *names], ensure_ascii=False, separators=(",", ":"))


def build_attack_graph_tables(nodes, adjacency, vuln_sheet, distribution):
    
    components = list(nodes)
    if (any(not isinstance(component, str) or not component.strip() for component in components)
            or len(set(components)) != len(components)):
        raise ValueError("Attack-graph component names must be nonempty and unique")
    matrix = np.asarray(adjacency)
    if matrix.shape != (len(components), len(components)) or not np.isin(matrix, (0, 1)).all():
        raise ValueError("The attack graph must provide a 0/1 adjacency matrix with rows corresponding to components")
    vulnerabilities = component_vulnerability_names(vuln_sheet, distribution)
    if set(vulnerabilities) != set(components):
        raise ValueError("Vulnerability-distribution components must match the attack-graph components")

    node_rows = []
    edge_rows = []
    seen_edges = set()

    def add_edge(source, target, kind):
        if (source, target) not in seen_edges:
            seen_edges.add((source, target))
            edge_rows.append({"Source_ID": source, "Target_ID": target, "Edge_Type": kind})

    for component in components:
        component_id = _node_id("component", component)
        node_rows.append({
            "Node_ID": component_id,
            "Display_Label": component,
            "Node_Type": {"S": "start", "T": "target"}.get(component, "component"),
            "Component": component,
            "Vulnerability_ID": "",
            "Local_Index": 0,
            "Order": len(node_rows),
        })
        for local_index, vulnerability in enumerate(vulnerabilities[component], start=1):
            vulnerability_id = _node_id("vulnerability", component, vulnerability)
            node_rows.append({
                "Node_ID": vulnerability_id,
                "Display_Label": vulnerability,
                "Node_Type": "vulnerability",
                "Component": component,
                "Vulnerability_ID": vulnerability,
                "Local_Index": local_index,
                "Order": len(node_rows),
            })
            add_edge(vulnerability_id, component_id, "compromises")

    for source_index, source in enumerate(components):
        source_id = _node_id("component", source)
        for target_index, target in enumerate(components):
            if not matrix[source_index, target_index]:
                continue
            if vulnerabilities[target]:
                for vulnerability in vulnerabilities[target]:
                    add_edge(source_id, _node_id("vulnerability", target, vulnerability), "exploits")
            else:
                add_edge(source_id, _node_id("component", target), "component_transition")

    return (pd.DataFrame(node_rows, columns=NODE_COLUMNS),
            pd.DataFrame(edge_rows, columns=EDGE_COLUMNS))
