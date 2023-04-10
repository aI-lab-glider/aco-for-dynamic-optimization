# ------------------------------------------------------------------------------------------------------
#  Copyright (c) Leo Hanisch. All rights reserved.
#  Licensed under the BSD 3-Clause License. See LICENSE.txt in the project root for license information.
# ------------------------------------------------------------------------------------------------------
from dataclasses import dataclass, asdict
from itertools import product
from math import sqrt
import networkx
import pandas as pd
import logging
from copy import deepcopy
from threading import RLock
import networkx as nx
import pandas as pd


LOGGER = logging.getLogger(__name__)


@dataclass
class Node:
    name: str
    demand: int
    service: int
    from_: int
    to: int
    coord: tuple
    is_depot: bool = False

    @classmethod
    def from_series(cls, series: pd.Series):
        return cls(
            name=series['cust'].iloc[0],
            demand=series['demand'].iloc[0],
            service=series['service'].iloc[0],
            from_=series['from'].iloc[0],
            to=series['to'].iloc[0],
            coord=(series['x'].iloc[0], series['y'].iloc[0]),
            is_depot=series['cust'].iloc[0] == 1
        )


class RoutesGraph:
    def __init__(self, df: pd.DataFrame):
        self.depot = 1
        self._df = df
        self.networkx_graph = self._create_graph(df)
        self.__lock = RLock()

    def max_demand(self):
        return self._df['demand'].max()

    def _create_graph(self, df: pd.DataFrame):
        G = networkx.Graph()

        # add basic graph metadata
        G.graph['name'] = 'ACO'
        # set up a map from original node name to new node name
        nodes = list(df['cust'].to_list())
        names = {n: n for n in nodes}

        coords = self._get_coords()
        # add every node with some associated metadata
        for n in nodes:
            is_depot = n == self.depot
            r = df[df['cust'] == n]
            G.add_node(names[n], **asdict(Node.from_series(r)))

        # add every edge with some associated metadata
        for a, b in product(df['cust'], df['cust']):
            (x, y) = coords[a]
            (x1, y1) = coords[b]
            dist = sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
            G.add_edge(names[a], names[b], weight=dist)
        return G

    def get_node(self, name: str):
        return Node(
            **self.networkx_graph.nodes[name]
        )

    def update_node(self, name: str, new_node: Node):
        self.networkx_graph.nodes[name].update(asdict(new_node))

    def add_node(self, new_node: Node):
        self.networkx_graph.add_node(new_node.name, **asdict(new_node))

    @property
    def node_coordinates(self):
        return self._get_coords()

    @property
    def name(self):
        return 'ACO'

    def _get_coords(self):
        df = self._df
        return {c: (x, y) for c, x, y in zip(df['cust'], df['x'], df['y'])}

    def get_nodes(self):
        """Get all nodes."""
        # with self.__lock:
        return [self.get_node(n) for n in self.networkx_graph.nodes()]

    def get_edges(self, node=None):
        """Get all edges connected to the given node. (u,v)"""
        # with self.__lock:
        return self.networkx_graph.edges(node)

    def set_pheromone(self, edge, value):
        """Set pheromone for the given edge.
        Edge is tuple (u,v)"""
        with self.__lock:
            self.networkx_graph.add_edge(*edge, pheromone=value)

    def get_connected_nodes(self, node):
        """Get the connected nodes of the given node"""
        # with self.__lock:
        return nx.node_connected_component(self.networkx_graph, node)

    def get_edge_pheromone(self, edge):
        """Get the pheromone value for the given edge"""
        sorted_edge = tuple(sorted(edge))
        return self.__get_edge_data(sorted_edge, 'pheromone')

    def get_vertex_demand(self, node):
        return nx.get_node_attributes(self.networkx_graph, 'demand')[node]

    def get_edge_length(self, edge):
        """Get the `rounded` length of the given edge."""
        return self.__get_edge_data(edge, 'weight')

    def __get_edge_data(self, edge, label):
        data = self.networkx_graph.get_edge_data(*edge)
        result = deepcopy(data.get(label, 0))
        return result

    def remove_node(self, node_name):
        self.networkx_graph.remove_node(node_name)
