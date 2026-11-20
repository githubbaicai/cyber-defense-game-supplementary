"""Backward-compatible exports for shared attack-graph tables."""

from common.attack_graph_data import (
    EDGE_COLUMNS, NODE_COLUMNS, _node_id, build_attack_graph_tables,
)

__all__ = ["EDGE_COLUMNS", "NODE_COLUMNS", "build_attack_graph_tables"]
