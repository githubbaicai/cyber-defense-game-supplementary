"""Render saved scenario CSVs without reading models, workbooks, or upstream results."""

from pathlib import Path
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import patheffects
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle

from common.attack_graph_layout import build_layout
from common.config import CSV_FILES, get_feature_output_dir


from common.plotting import (
    DPI_SAVE, FONT_SERIF, PANEL_CAPTION_FONTSIZE, TICK_LABELSIZE,
    plot_style as shared_plot_style,
)

PATH_COLOR = "#E78589"
BACKGROUND_NODE = "#F1EFEF"
BACKGROUND_EDGE = "#DEDCDC"
BACKGROUND_TEXT = "#A7A4A4"
ENDPOINT_COLOR = "#F8CD90"
PATCH_COLOR = "#22812B"
DEFENSE_COLOR = "#2348A3"


COMBINED_CAPTION_REFERENCE_WIDTH_MM = 160.0


def _highlight(steps, node_ids, edge_ids):
    """Validate and collect only the explicitly selected consecutive path edges."""
    highlighted_nodes, highlighted_edges = set(), set()
    for row in steps.itertuples(index=False):
        sequence = [str(row.Source_Node_ID)]
        vulnerability = str(row.Vulnerability_Node_ID)
        if vulnerability:
            sequence.append(vulnerability)
        sequence.append(str(row.Target_Node_ID))
        if any(node not in node_ids for node in sequence):
            raise ValueError("Selected path references an unknown graph node")
        for edge in zip(sequence, sequence[1:]):
            if edge not in edge_ids:
                raise ValueError("Selected path references a nonexistent directed graph edge")
            highlighted_edges.add(edge)
        highlighted_nodes.update(sequence)
    return highlighted_nodes, highlighted_edges


def _caption(scenario):
    title = str(scenario.Display_Title)
    panel = str(scenario.Panel).strip("()")
    if title.startswith(f"({panel})"):
        return title
    return f"({panel}) {title}"


def _geometry(layout, annotations):
    """Common millimetre canvas includes space for all panels' annotations."""
    component = annotations[annotations["Scope"].eq("component")]
    top_rows = annotations[annotations["Scope"].eq("network")].groupby("Panel").size()
    top = max(14.0, (float(top_rows.max()) if len(top_rows) else 0) * 4.2 + 5)
    right = 44.0 if not component.empty else 2.5
    return (layout.page_mm[0] + 4 + right, layout.page_mm[1] + top + 8), top


def _draw_panel(ax, nodes, edges, layout, scenario, steps, annotations, page_mm, top,
                caption_fontsize):
    ax.set_xlim(-2, page_mm[0] - 2)
    ax.set_ylim(page_mm[1] - top, -top)
    ax.set_aspect("equal")
    ax.axis("off")
    node_ids = set(nodes["Node_ID"])
    edge_ids = set(zip(edges["Source_ID"], edges["Target_ID"]))
    active_nodes, active_edges = _highlight(steps, node_ids, edge_ids)
    for active in (False, True):
        for edge, path in layout.paths.items():
            if (edge in active_edges) != active:
                continue
            arrow = FancyArrowPatch(
                path=path, arrowstyle="-|>", mutation_scale=8 if active else 6,
                linewidth=1.1 if active else 0.55,
                color=PATH_COLOR if active else BACKGROUND_EDGE, zorder=2 if active else 1,
            )
            if active:
                arrow.set_path_effects([
                    patheffects.Stroke(linewidth=2.2, foreground="white"),
                    patheffects.Normal(),
                ])
            arrow.set_gid("path-edge" if active else "background-edge")
            ax.add_patch(arrow)
    for row in nodes.itertuples(index=False):
        x, y = layout.positions[row.Node_ID]
        width, height = layout.sizes[row.Node_ID]
        active = row.Node_ID in active_nodes
        color = (ENDPOINT_COLOR if row.Node_Type in ("start", "target")
                 else PATH_COLOR if active else BACKGROUND_NODE)
        shape = (Rectangle((x - width / 2, y - height / 2), width, height)
                 if row.Node_Type == "vulnerability" else Ellipse((x, y), width, height))
        shape.set(facecolor=color, edgecolor="none", zorder=3)
        shape.set_gid(f"node:{row.Node_ID}")
        ax.add_patch(shape)
        label = ax.text(x, y, layout.labels[row.Node_ID], ha="center", va="center",
                        fontsize=PANEL_CAPTION_FONTSIZE,
                        color="black" if active or row.Node_Type in ("start", "target")
                        else BACKGROUND_TEXT, zorder=4)
        label.set_gid("node-label")
    network = annotations[annotations["Scope"].eq("network")]
    for index, row in enumerate(network.itertuples(index=False)):
        ax.text(layout.page_mm[0] / 2, -top + 3 + index * 4.2, str(row.Text),
                ha="center", va="center", color=DEFENSE_COLOR,
                fontsize=TICK_LABELSIZE, fontstyle="italic", zorder=5)
    components = []
    for row in annotations[~annotations["Scope"].eq("network")].itertuples(index=False):
        if row.Node_ID not in layout.positions:
            raise ValueError(f"Defense annotation references unknown graph node: {row.Node_ID}")
        x, y = layout.positions[row.Node_ID]
        width, height = layout.sizes[row.Node_ID]
        if row.Annotation_Type == "patch":
            ax.text(x, y + height / 2 + 1.0, str(row.Text), ha="center", va="top",
                    color=PATCH_COLOR, fontsize=TICK_LABELSIZE, fontstyle="italic", zorder=5,
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 0.2})
        else:
            components.append((y, x, width, str(row.Text)))
    last_y = float("-inf")
    for y, x, width, text in sorted(components):
        text_y = max(y, last_y + 4.1)
        text_x = layout.page_mm[0] + 3
        ax.annotate(text, xy=(x + width / 2, y), xytext=(text_x, text_y),
                    ha="left", va="center", fontsize=TICK_LABELSIZE, color=DEFENSE_COLOR,
                    fontstyle="italic", zorder=5,
                    arrowprops={"arrowstyle": "-", "color": "#B8C3DB", "lw": 0.45})
        last_y = text_y
    bottom = layout.page_mm[1] + 5.5
    caption = ax.text(layout.page_mm[0] / 2, bottom, _caption(scenario),
                      ha="center", va="center", fontsize=caption_fontsize)
    caption.set_gid("panel-caption")


def draw_attack_paths(nodes, edges, scenarios, steps, annotations, *, layout=None,
                      canvas_geometry=None):
    """Return one or more equal-scale panels using only supplied saved tables."""
    with mpl.rc_context(shared_plot_style(kind="multi")):
        if scenarios.empty:
            raise ValueError("No attack-path scenarios were exported")
        layout = layout or build_layout(nodes, edges, font_size=PANEL_CAPTION_FONTSIZE,
                                        font_family=FONT_SERIF[0])
        geometry = canvas_geometry or _geometry(layout, annotations)
        page_mm, top = geometry
        count = len(scenarios)
        gap = 5.0
        width = count * page_mm[0] + (count - 1) * gap
        caption_fontsize = PANEL_CAPTION_FONTSIZE
        if count > 1:
            caption_fontsize *= max(1.0, width / COMBINED_CAPTION_REFERENCE_WIDTH_MM)
            extra_bottom = (caption_fontsize - PANEL_CAPTION_FONTSIZE) * 25.4 / 72 / 2 + 0.3
            page_mm = (page_mm[0], page_mm[1] + extra_bottom)
        height = page_mm[1]
        figure = plt.figure(figsize=(width / 25.4, height / 25.4))
        for index, scenario in enumerate(scenarios.itertuples(index=False)):
            ax = figure.add_axes(((index * (page_mm[0] + gap)) / width,
                                  0, page_mm[0] / width, 1))
            panel = str(scenario.Panel)
            _draw_panel(ax, nodes, edges, layout, scenario,
                        steps[steps["Panel"].astype(str).eq(panel)],
                        annotations[annotations["Panel"].astype(str).eq(panel)], page_mm, top,
                        caption_fontsize)
        return figure


def _save(figure, directory, basename):
    outputs = []
    try:
        with mpl.rc_context({"savefig.bbox": None}):
            for extension in ("png", "svg", "pdf"):
                destination = directory / f"{basename}.{extension}"
                figure.savefig(destination, bbox_inches=None, pad_inches=0,
                               dpi=DPI_SAVE, facecolor="white")
                outputs.append(destination)
    finally:
        plt.close(figure)
    return outputs


def plot(input_excel=None, output_dir=None):
    """Read this feature's CSV snapshot and render the combined and individual panels."""
    directory = (Path(output_dir) if output_dir is not None
                 else get_feature_output_dir("pareto_attack_paths", input_excel))
    files = CSV_FILES["pareto_attack_paths"]
    tables = {name: pd.read_csv(directory / files[name], keep_default_na=False)
              for name in ("attack_graph_nodes", "attack_graph_edges", "scenarios",
                           "path_steps", "defense_annotations")}
    nodes, edges = tables["attack_graph_nodes"], tables["attack_graph_edges"]
    scenarios = tables["scenarios"].sort_values("Panel", kind="stable")
    steps, annotations = tables["path_steps"], tables["defense_annotations"]
    with mpl.rc_context(shared_plot_style(kind="multi")):
        layout = build_layout(nodes, edges, font_size=PANEL_CAPTION_FONTSIZE,
                              font_family=FONT_SERIF[0])
        geometry = _geometry(layout, annotations)
        kwargs = {"layout": layout, "canvas_geometry": geometry}
        figure = draw_attack_paths(nodes, edges, scenarios, steps, annotations, **kwargs)
        outputs = _save(figure, directory, "most_probable_attack_paths")
        for scenario in scenarios.itertuples(index=False):
            selected = scenarios[scenarios["Panel"].eq(scenario.Panel)]
            figure = draw_attack_paths(nodes, edges, selected, steps, annotations, **kwargs)
            if scenario.Selection_Type == "undefended":
                basename = "attack_path_undefended"
            else:
                selection = str(scenario.Selection_Type)
                if not re.fullmatch(r"[A-Za-z0-9_-]+", selection):
                    raise ValueError("Selection_Type is not safe for an output filename")
                basename = f"attack_path_{selection}_{int(scenario.Solution_ID)}"
            outputs.extend(_save(figure, directory, basename))
    return outputs


if __name__ == "__main__":
    plot()
