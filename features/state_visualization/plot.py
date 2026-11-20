

import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib import patheffects
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle

from common.config import CSV_FILES, get_feature_output_dir
from common.plotting import save_figure, style_axes
from features.state_visualization.attack_graph_layout import build_layout


from common.plotting import (
    FONT_FAMILY, FONT_SERIF, BASE_FONTSIZE, AX_LABELSIZE, TICK_LABELSIZE,
    AXES_LINEWIDTH, GRID_COLOR, GRID_LINEWIDTH, SPINE_COLOR, DPI_FIG, DPI_SAVE,
    MULTI_AXIS_LABELSIZE, PANEL_CAPTION_FONTSIZE, TITLE_FONTSIZE,
    plot_style as shared_plot_style,
)

LEGEND_FONTSIZE = 9
ATTACK_LABELSIZE = PANEL_CAPTION_FONTSIZE
ATTACK_COMPONENT_COLOR = "#F79A9C"
ATTACK_ENDPOINT_COLOR = "#F8CD90"
ATTACK_EDGE_COLOR = "#D9757C"
ATTACK_EDGE_WIDTH = 0.65
ATTACK_ARROW_SIZE = 8
FILES = CSV_FILES["state_visualization"]


def _plot_style():
    return shared_plot_style(legend_fontsize=LEGEND_FONTSIZE)


def draw_attack_graph(nodes, edges):
    
    with mpl.rc_context(_plot_style()):
        return _draw_attack_graph(nodes, edges)


def _draw_attack_graph(nodes, edges):
    layout = build_layout(nodes, edges, font_size=ATTACK_LABELSIZE, font_family=FONT_SERIF[0])
    fig = plt.figure(figsize=tuple(length / 25.4 for length in layout.page_mm))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, layout.page_mm[0])
    ax.set_ylim(layout.page_mm[1], 0)
    ax.set_aspect("equal")
    ax.axis("off")
    for path in layout.paths.values():
        arrow = FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=ATTACK_ARROW_SIZE,
                                linewidth=ATTACK_EDGE_WIDTH, color=ATTACK_EDGE_COLOR, zorder=1)
        arrow.set_path_effects([patheffects.Stroke(linewidth=ATTACK_EDGE_WIDTH + 1.1,
                                                 foreground="white"), patheffects.Normal()])
        ax.add_patch(arrow)
    for row in nodes.itertuples(index=False):
        horizontal, vertical = layout.positions[row.Node_ID]
        width, height = layout.sizes[row.Node_ID]
        color = ATTACK_ENDPOINT_COLOR if row.Node_Type in ("start", "target") else ATTACK_COMPONENT_COLOR
        if row.Node_Type == "vulnerability":
            shape = Rectangle((horizontal - width / 2, vertical - height / 2), width, height)
        else:
            shape = Ellipse((horizontal, vertical), width, height)
        shape.set(facecolor=color, edgecolor="none", zorder=2)
        ax.add_patch(shape)
        ax.text(horizontal, vertical, layout.labels[row.Node_ID], ha="center", va="center",
                fontsize=ATTACK_LABELSIZE, color="black", zorder=3)
    return fig


def _plot_attack_graph(directory):
    tables = [directory / FILES[key] for key in ("attack_graph_nodes", "attack_graph_edges")]
    if not all(path.is_file() for path in tables):
        warnings.warn("Attack-graph CSV files are missing; rerun state_visualization solve/export", UserWarning, stacklevel=2)
        return
    nodes, edges = [pd.read_csv(path, keep_default_na=False) for path in tables]
    if nodes.empty:
        warnings.warn("The attack-graph CSV comes from a legacy cache and contains no nodes; rerun state_visualization solve/export",
                      UserWarning, stacklevel=2)
        return
    fig = draw_attack_graph(nodes, edges)
    try:
        with mpl.rc_context({"savefig.bbox": None}):
            for extension in ("png", "svg", "pdf"):
                destination = directory / f"attack_graph.{extension}"
                try:
                    fig.savefig(destination, bbox_inches=None, pad_inches=0, dpi=DPI_SAVE, facecolor="white")
                except PermissionError:
                    alternate = directory / f"attack_graph_compact.{extension}"
                    fig.savefig(alternate, bbox_inches=None, pad_inches=0, dpi=DPI_SAVE, facecolor="white")
                    warnings.warn(f"Unable to overwrite {destination.name} (the file may be in use); the updated version was saved as {alternate.name}",
                                  UserWarning, stacklevel=2)
    finally:
        plt.close(fig)


def _plot_state_risk(nodes, directory):
    top = (nodes.dropna(subset=["Compromise_Probability"])
           .sort_values("Compromise_Probability", ascending=False).head(20))
    if top.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(top["Node_Path"].astype(str)[::-1], top["Compromise_Probability"][::-1],
            color="#4DBBD5", zorder=3)
    ax.set_xlabel("Compromise probability")
    ax.set_ylabel("State path")
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, directory, "state_risk")


def _plot_main_path(main, directory):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(main["Step"], main["Compromise_Probability"], "o-", color="#E64B35")
    ax.set_xticks(main["Step"])
    ax.set_xticklabels(main["Component"].astype(str))
    ax.set_xlabel("Main attack path component")
    ax.set_ylabel("Cumulative compromise probability")
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, directory, "main_attack_path")


def _plot_state_transitions(edges, main, directory):
    graph = nx.DiGraph()
    graph.add_edges_from(edges[["Source_State", "Target_State"]].itertuples(index=False, name=None))
    positions = nx.spring_layout(graph, seed=42)
    main_states = set(main["Node_Path"].astype(str))
    highlighted_edges = [edge for edge in graph.edges() if set(edge).issubset(main_states)]
    highlighted_set = set(highlighted_edges)
    ordinary_edges = [edge for edge in graph.edges() if edge not in highlighted_set]
    fig, ax = plt.subplots(figsize=(14, 9))
    nx.draw_networkx_nodes(graph, positions, node_color=["#E64B35" if node in main_states else "#91D1C2"
                                                      for node in graph.nodes()], node_size=750, ax=ax)
    nx.draw_networkx_labels(graph, positions, font_size=TICK_LABELSIZE, ax=ax)
    nx.draw_networkx_edges(graph, positions, edgelist=ordinary_edges, edge_color="#aaaaaa",
                           arrows=True, arrowsize=12, ax=ax)
    nx.draw_networkx_edges(graph, positions, edgelist=highlighted_edges, edge_color="#E64B35",
                           width=2.5, arrows=True, arrowsize=14, ax=ax)
    ax.axis("off")
    fig.tight_layout()
    save_figure(fig, directory, "state_transition_graph")


def plot(input_excel=None):
    
    directory = get_feature_output_dir("state_visualization", input_excel)
    with mpl.rc_context(_plot_style()):
        nodes = pd.read_csv(directory / FILES["node_metrics"])
        edges = pd.read_csv(directory / FILES["state_graph_edges"])
        main = pd.read_csv(directory / FILES["main_attack_path"])
        _plot_attack_graph(directory)
        if not nodes.empty:
            _plot_state_risk(nodes, directory)
        if not main.empty:
            _plot_main_path(main, directory)
        if not edges.empty:
            _plot_state_transitions(edges, main, directory)


if __name__ == "__main__":
    plot()
