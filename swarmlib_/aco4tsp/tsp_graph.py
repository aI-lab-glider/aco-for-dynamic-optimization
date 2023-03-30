# ------------------------------------------------------------------------------------------------------
#  Copyright (c) Leo Hanisch. All rights reserved.
#  Licensed under the BSD 3-Clause License. See LICENSE.txt in the project root for license information.
# ------------------------------------------------------------------------------------------------------
from itertools import product
from math import sqrt
import networkx
import pandas as pd
import logging
from copy import deepcopy
from threading import RLock
import networkx as nx


LOGGER = logging.getLogger(__name__)


class Graph:
    def __init__(self, df):
        self.depot = 1
        self._df = df
        self.networkx_graph = self._create_graph(df)
        self.__lock = RLock()

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
            G.add_node(names[n], coord=(r['x'].iloc[0], r['y'].iloc[0]),
                       demand=r['demand'].iloc[0],
                       from_=r['from'].iloc[0],
                       to_=r['to'].iloc[0],
                       service=r['service'].iloc[0],
                       is_depot=is_depot)

        # add every edge with some associated metadata
        for a, b in product(df['cust'], df['cust']):
            (x, y) = coords[a]
            (x1, y1) = coords[b]
            dist = sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
            G.add_edge(names[a], names[b], weight=dist)

        return G

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
        return list(self.networkx_graph.nodes())

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
        # with self.__lock:
        data = self.networkx_graph.get_edge_data(*edge)
        result = deepcopy(data.get(label, 0))

        # LOGGER.debug('Get data="%s", value="%s" for edge="%s"',
        #              label, result, edge)
        return result
