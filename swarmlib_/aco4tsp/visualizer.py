# ------------------------------------------------------------------------------------------------------
#  Copyright (c) Leo Hanisch. All rights reserved.
#  Licensed under the BSD 3-Clause License. See LICENSE.txt in the project root for license information.
# ------------------------------------------------------------------------------------------------------

import numpy as np
from typing import cast
from matplotlib import animation
from matplotlib import pyplot as plt
import networkx as nx

from swarmlib_.aco4tsp.dynamic_vrp_env.routes_graph import RoutesGraph

from ..util.visualizer_base import VisualizerBase
import seaborn as sns


class Visualizer(VisualizerBase):  # pylint:disable=too-many-instance-attributes
    def __init__(self, **kwargs):
        self._edge_widths = []
        self._edge_colors = []
        self._best_color, self._usual_color, self._node_color = sns.color_palette(
            'deep', 3)
        self._best_paths = []
        self._fitness = []

    def add_data(self, pheromone_map, best_path, fitness):
        self._fitness.append(fitness)
        best_path_edges = [
            (node, best_path[idx+1])
            for idx, node in enumerate(best_path[:-1])
        ]

        self._edge_widths.append(self.__scale_range(
            [v for k, v in pheromone_map.items() if k in best_path_edges]))
        self._best_paths.append(best_path_edges)

    def _colorize(self, path: list[tuple[int, int]], depot_node: int):
        colorized_path = {}
        n_vehicles = len([1 for _, dest in path if dest == depot_node])
        pallet = iter(sns.color_palette('deep', n_colors=n_vehicles))
        for edge in path:
            if edge[0] == depot_node:
                color = next(pallet)
            colorized_path[edge] = color
        return colorized_path

    def replay(self, **kwargs):
        """Draw the given graph."""
        fig, ax = plt.subplots(frameon=False)  # pylint:disable=invalid-name
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        sns.set_theme('paper')

        graph = cast(RoutesGraph, kwargs['graph'])
        node_pos = graph.node_coordinates
        depot = graph.depot

        def _update(num):
            ax.clear()
            title = f'Iteration: {num}'
            best_path = self._best_paths[num]
            best_path_colorized = self._colorize(best_path, depot)

            nodes_artist = nx.draw_networkx_nodes(
                graph.networkx_graph, pos=node_pos, ax=ax, node_color=self._node_color)
            labels_artist = nx.draw_networkx_labels(
                graph.networkx_graph, pos=node_pos, ax=ax)
            edges_artist = nx.draw_networkx_edges(
                graph.networkx_graph, pos=node_pos,
                width=self._edge_widths[num],
                edgelist=[e for e in best_path_colorized],
                edge_color=best_path_colorized.values(),
                ax=ax)
            return nodes_artist, labels_artist, edges_artist, ax.text(0.5, 0.9, title, horizontalalignment='center', color='k', verticalalignment='center', transform=ax.transAxes)

        a = animation.FuncAnimation(fig, _update, frames=len(
            self._best_paths), repeat=False)
        a._start()

    def plot_result(self, graph):
        fig, ax = plt.subplots(1, 1)
        best_path = self._best_paths[-1]
        node_pos = graph.node_coordinates
        depot = graph.depot
        best_path_colorized = self._colorize(best_path, depot)
       

        nx.draw_networkx_nodes(
                graph.networkx_graph, pos=node_pos, ax=ax, node_color=self._node_color)
        nx.draw_networkx_labels(
                graph.networkx_graph, pos=node_pos, ax=ax)
        nx.draw_networkx_edges(
                graph.networkx_graph, pos=node_pos,
                width=self._edge_widths[-1],
                edgelist=[e for e in best_path_colorized],
                edge_color=best_path_colorized.values(),
                ax=ax)




    def plot_fitness(self):
        fig, axs = plt.subplots(1, 1)
        axs.plot(self._fitness)
        plt.show()

    @staticmethod
    def __scale_range(seq, new_max=10, new_min=0.01):
        old_max = max(seq)
        old_min = min(seq)
        return [
            (((old_value - old_min) * (new_max - new_min)) /
             (old_max - old_min)) + new_min
            for old_value in iter(seq)
        ]
