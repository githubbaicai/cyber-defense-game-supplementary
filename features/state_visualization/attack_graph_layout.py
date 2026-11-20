"""Backward-compatible exports for the shared deterministic layout."""

from common.attack_graph_layout import (
    AttackGraphLayout, _label, _rounded_path, build_layout,
)

__all__ = ["AttackGraphLayout", "build_layout"]
