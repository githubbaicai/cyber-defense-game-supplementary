

import math
from dataclasses import dataclass

import networkx as nx
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.textpath import TextPath

from common.plotting import FONT_SERIF, PANEL_CAPTION_FONTSIZE


@dataclass(frozen=True)
class AttackGraphLayout:
    positions: dict
    sizes: dict
    labels: dict
    paths: dict
    ranks: dict
    page_mm: tuple


def _label(row):
    if row.Node_Type != "vulnerability":
        return str(row.Display_Label)
    component = str(row.Component).replace("_", r"\_")
    return rf"$V_{{\mathrm{{{component}}},{int(row.Local_Index)}}}$"


def _rounded_path(points, radius):
    vertices = [points[0]]
    codes = [Path.MOVETO]
    for previous, corner, following in zip(points, points[1:], points[2:]):
        incoming = np.asarray(corner) - previous
        outgoing = np.asarray(following) - corner
        before = float(np.linalg.norm(incoming))
        after = float(np.linalg.norm(outgoing))
        if min(before, after) < 1e-8:
            continue
        distance = min(radius, before / 2, after / 2)
        entry = np.asarray(corner) - incoming / before * distance
        exit_point = np.asarray(corner) + outgoing / after * distance
        vertices.extend([entry, corner, exit_point])
        codes.extend([Path.LINETO, Path.CURVE3, Path.CURVE3])
    vertices.append(points[-1])
    codes.append(Path.LINETO)
    return Path(np.asarray(vertices), codes)


def build_layout(nodes, edges, *, font_size=PANEL_CAPTION_FONTSIZE, font_family=FONT_SERIF[0]):
    
    if nodes.empty:
        raise ValueError("The attack-graph node table is empty; rerun state_visualization solve/export")
    if nodes["Node_ID"].isna().any() or not nodes["Node_ID"].is_unique:
        raise ValueError("Attack-graph Node_ID values must be nonempty and unique")
    graph = nx.DiGraph()
    records = {row.Node_ID: row for row in nodes.itertuples(index=False)}
    graph.add_nodes_from(records)
    edge_pairs = list(edges[["Source_ID", "Target_ID"]].itertuples(index=False, name=None))
    if any(source not in records or target not in records for source, target in edge_pairs):
        raise ValueError("An attack-graph edge references a nonexistent node")
    if len(set(edge_pairs)) != len(edge_pairs):
        raise ValueError("The attack-graph edge table contains duplicate edges")
    graph.add_edges_from(edge_pairs)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("A layered attack-graph layout requires a directed acyclic topology; check Attack graph")

    compact = len(records) > 20
    max_size_mm = (156.0, 238.0) if compact else (105.0, 148.5)
    center = 0.0
    labels = {node: _label(row) for node, row in records.items()}
    font = FontProperties(family=font_family, math_fontfamily="stix")
    sizes = {}
    for node, row in records.items():
        width = TextPath((0, 0), labels[node], size=font_size, prop=font).get_extents().width * 25.4 / 72
        if row.Node_Type == "vulnerability":
            sizes[node] = (max(14, width + 2.8), 6.2)
        else:
            sizes[node] = (max(14, width + 6.5), 6.5)

    ranks = {}
    for node in nx.lexicographical_topological_sort(graph, key=lambda value: records[value].Order):
        ranks[node] = max((ranks[parent] + 1 for parent in graph.predecessors(node)), default=0)
    layers = {}
    for node, rank in ranks.items():
        layers.setdefault(rank, {}).setdefault(records[node].Component, []).append(node)
    group_widths = {}
    vulnerability_gap = 2.5
    group_gap = 4.0
    for groups in layers.values():
        for component, members in groups.items():
            width = sum(sizes[node][0] for node in members) + vulnerability_gap * (len(members) - 1)
            group_widths[component] = max(group_widths.get(component, 0), width)
    rank_gap = 9.8 if compact else 12.5
    positions = {}
    previous_centers = {}
    for rank, groups in sorted(layers.items()):
        def order_group(component, active_groups=groups):
            members = active_groups[component]
            parents = [positions[parent][0] for node in members for parent in graph.predecessors(node)]
            desired = previous_centers.get(component, float(np.mean(parents)) if parents else center)
            return desired, min(records[node].Order for node in members)

        components = sorted(groups, key=order_group)
        total_width = sum(group_widths[component] for component in components) + group_gap * (len(components) - 1)
        if total_width > max_size_mm[0]:
            raise ValueError("Attack-graph labels on the same layer exceed the available manuscript width; adjust the grouped layout")
        cursor = center - total_width / 2
        for component in components:
            members = sorted(groups[component], key=lambda node: records[node].Order)
            group_center = cursor + group_widths[component] / 2
            member_width = sum(sizes[node][0] for node in members) + vulnerability_gap * (len(members) - 1)
            member_cursor = group_center - member_width / 2
            for node in members:
                positions[node] = (member_cursor + sizes[node][0] / 2, rank * rank_gap)
                member_cursor += sizes[node][0] + vulnerability_gap
            previous_centers[component] = group_center
            cursor += group_widths[component] + group_gap

    def boundary(node, offset, direction):
        horizontal, vertical = positions[node]
        width, height = sizes[node]
        factor = 1 if records[node].Node_Type == "vulnerability" else math.sqrt(1 - (offset / (width / 2)) ** 2)
        return horizontal + offset, vertical + direction * height / 2 * factor

    source_ports, target_ports = {}, {}
    for node in graph:
        for neighbors, ports, outgoing in (
            (list(graph.successors(node)), source_ports, True),
            (list(graph.predecessors(node)), target_ports, False),
        ):
            neighbors.sort(key=lambda neighbor: (positions[neighbor][0], ranks[neighbor], records[neighbor].Order))
            offsets = np.linspace(-0.30, 0.30, len(neighbors)) if len(neighbors) > 1 else [0]
            for neighbor, offset in zip(neighbors, offsets):
                edge = (node, neighbor) if outgoing else (neighbor, node)
                ports[edge] = boundary(node, offset * sizes[node][0], 1 if outgoing else -1)

    row_heights = {rank: max(sizes[node][1] for members in groups.values() for node in members)
                   for rank, groups in layers.items()}
    ordered_edges = sorted(graph.edges(), key=lambda edge: (ranks[edge[0]], ranks[edge[1]], records[edge[1]].Order))
    long_edges = [edge for edge in ordered_edges if ranks[edge[1]] > ranks[edge[0]] + 1]
    exit_offsets, entry_offsets = {}, {}
    for endpoint_index, offsets, neighbor_direction in ((0, exit_offsets, 1), (1, entry_offsets, -1)):
        for rank in layers:
            members = [edge for edge in long_edges if ranks[edge[endpoint_index]] == rank]
            if not members:
                continue
            neighbor_height = row_heights.get(rank + neighbor_direction, row_heights[rank])
            free_gap = rank_gap - (row_heights[rank] + neighbor_height) / 2
            last_offset = min(free_gap - 0.8, 1.3 + 1.3 * (len(members) - 1))
            offsets.update(zip(members, np.linspace(1.3, last_offset, len(members))))
    lanes, paths = [], {}
    for source, target in ordered_edges:
        edge = (source, target)
        start, end = source_ports[edge], target_ports[edge]
        if ranks[target] == ranks[source] + 1:
            bend = (end[1] - start[1]) * 0.48
            paths[edge] = Path([start, (start[0], start[1] + bend), (end[0], end[1] - bend), end],
                               [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
            continue
        exit_y = positions[source][1] + row_heights[ranks[source]] / 2 + exit_offsets[edge]
        entry_y = positions[target][1] - row_heights[ranks[target]] / 2 - entry_offsets[edge]
        obstacles = [node for node in graph if ranks[source] < ranks[node] < ranks[target]]
        left = min(positions[node][0] - sizes[node][0] / 2 for node in obstacles) - 4
        right = max(positions[node][0] + sizes[node][0] / 2 for node in obstacles) + 4
        candidates = []
        for side, initial in ((-1, left), (1, right)):
            channel = initial
            while any(abs(channel - occupied) < 1.5 and exit_y < finish and entry_y > begin
                      for occupied, begin, finish in lanes):
                channel += side * 1.6
            if abs(channel) < max_size_mm[0] / 2:
                cost = abs(start[0] - channel) + abs(end[0] - channel)
                candidates.append((cost, channel))
        if not candidates:
            raise ValueError("The manuscript width leaves insufficient space for routing between layers; adjust the layout")
        _, channel = min(candidates)
        lanes.append((channel, exit_y, entry_y))
        paths[edge] = _rounded_path([start, (start[0], exit_y), (channel, exit_y),
                                    (channel, entry_y), (end[0], entry_y), end], 1.2)

    bounds = [(positions[node][0] - width / 2, positions[node][1] - height / 2,
               positions[node][0] + width / 2, positions[node][1] + height / 2)
              for node, (width, height) in sizes.items()]
    bounds.extend(tuple(path.get_extents().extents) for path in paths.values())
    left = min(bound[0] for bound in bounds) - 0.15
    top = min(bound[1] for bound in bounds) - 0.15
    right = max(bound[2] for bound in bounds) + 0.15
    bottom = max(bound[3] for bound in bounds) + 0.15
    page_mm = (right - left, bottom - top)
    if any(length > limit for length, limit in zip(page_mm, max_size_mm)):
        raise ValueError("The attack graph exceeds the manuscript size limits; adjust the layout rather than automatically reducing the font size")
    positions = {node: (horizontal - left, vertical - top)
                 for node, (horizontal, vertical) in positions.items()}
    paths = {edge: Path(path.vertices - (left, top), path.codes) for edge, path in paths.items()}
    return AttackGraphLayout(positions, sizes, labels, paths, ranks, page_mm)
