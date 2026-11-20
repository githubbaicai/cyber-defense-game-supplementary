

import networkx as nx
from matplotlib import pyplot as plt
import pandas as pd

# =========================================================

# =========================================================
from common.plotting import PANEL_CAPTION_FONTSIZE, TITLE_FONTSIZE, set_plot_style


class AttackGraph:
    def __init__(self, nodes, adj_matrix):
        
        self.nodes = nodes
        self.adj_matrix = adj_matrix

        
        self.graph_dict = self.matrix_to_dict()

        
        self.attack_paths = self.find_attack_paths('S', 'T')

        
        self.state_graph = None
        self.state_paths = []

    def matrix_to_dict(self):
        
        graph = {}
        for i, node in enumerate(self.nodes):
            graph[node] = {}
            for j, neighbor in enumerate(self.nodes):
                graph[node][neighbor] = self.adj_matrix[i][j]
        return graph

    def find_attack_paths(self, start, end, path=[]):
        
        path = path + [start]
        if start == end:
            return [path]

        
        if start not in self.graph_dict:
            return []

        paths = []
        for node in self.graph_dict[start]:
            
            if self.graph_dict[start][node] == 1:
                
                if node not in path:
                    new_paths = self.find_attack_paths(
                        node,
                        end,
                        path
                    )
                    for p in new_paths:
                        paths.append(p)
        return paths

    def build_state_transition_graph(self):
        
        G = nx.DiGraph()
        self.state_paths = []

        for path in self.attack_paths:
            for i in range(len(path) - 1):
                current_state = ' → '.join(path[: i + 1])
                next_state = ' → '.join(path[: i + 2])

                G.add_edge(current_state, next_state)
                self.state_paths.append(
                    (current_state, next_state)
                )

        self.state_graph = G
        return G

    def draw_attack_graph(self):
        
        G = nx.DiGraph()
        for i, node in enumerate(self.nodes):
            for j, neighbor in enumerate(self.nodes):
                if self.adj_matrix[i][j] == 1:
                    G.add_edge(node, neighbor)

        pos = nx.spring_layout(G, seed=42)
        pos['S'] = (-0.3, 0)
        pos['T'] = (0.3, 0)

        for node in self.nodes:
            if node in ('S', 'T'):
                pos[node] = (
                    pos[node][0] * 0.8,
                    pos[node][1]
                )

        set_plot_style()
        plt.figure(figsize=(10, 6))
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color='lightblue',
            node_size=2500,
            font_size=PANEL_CAPTION_FONTSIZE,
            font_weight='bold',
            arrows=True,
            arrowstyle='-|>',
            arrowsize=20,
        )
        plt.title('Attack Graph', fontsize=TITLE_FONTSIZE)
        plt.tight_layout()
        plt.show()

    def draw_state_transition_graph(self):
        
        if self.state_graph is None:
            self.build_state_transition_graph()

        pos = {}
        y_offset = 0
        for current_state, next_state in self.state_paths:
            depth_current = len(
                current_state.split(' → ')
            )
            depth_next = len(
                next_state.split(' → ')
            )

            pos[current_state] = (
                depth_current,
                y_offset
            )
            pos[next_state] = (
                depth_next,
                y_offset - 1
            )
            y_offset -= 2

        set_plot_style()
        plt.figure(figsize=(14, 8))
        nx.draw(
            self.state_graph,
            pos,
            with_labels=True,
            node_color='lightgreen',
            node_size=3500,
            font_size=PANEL_CAPTION_FONTSIZE,
            font_weight='bold',
            arrows=True,
            arrowstyle='-|>',
            arrowsize=20,
        )
        plt.title('State Transition Diagram', fontsize=TITLE_FONTSIZE)
        plt.tight_layout()
        plt.show()

    def save_paths_to_excel(
        self,
        filename='r1_attack_graph_paths_save.xlsx'
    ):
        
        attack_rows = [
            ' , '.join([c, n])
            for c, n in [
                (node, '')
                for node in []
            ]
        ]

        with pd.ExcelWriter(filename) as writer:
            df_attack_paths = pd.DataFrame({
                'Attack Path': [
                    ' , '.join(p)
                    for p in self.attack_paths
                ]
            })
            df_attack_paths.to_excel(
                writer,
                sheet_name='Attack Paths',
                index=False
            )

            df_state_paths = pd.DataFrame({
                'State Transition Path': [
                    ' , '.join([c, n])
                    for c, n in self.state_paths
                ]
            })
            df_state_paths.to_excel(
                writer,
                sheet_name='State Transition Paths',
                index=False
            )