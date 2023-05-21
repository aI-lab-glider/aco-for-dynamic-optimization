# ------------------------------------------------------------------------------------------------------
#  Copyright (c) Leo Hanisch. All rights reserved.
#  Licensed under the BSD 3-Clause License. See LICENSE.txt in the project root for license information.
# ------------------------------------------------------------------------------------------------------

import logging
from threading import Thread
import numpy as np

from swarmlib_.aco4tsp.dynamic_vrp_env.routes_graph import RoutesGraph
from swarmlib_.aco4tsp.vehicle import VehiclePath, Vehicle
from .local_2_opt import run_2opt
logger = logging.getLogger(__name__)
# pylint: disable=attribute-defined-outside-init,too-many-instance-attributes,too-many-arguments,super-init-not-called,invalid-name


class Ant(Thread):
    def __init__(self, start_node, graph: RoutesGraph, alpha, beta, Q, use_2_opt, random, vehicles: list[Vehicle], capacity):
        """Initializes a new instance of the Ant class."""
        self.graph = graph
        self._alpha = alpha
        self._beta = beta
        self._Q = Q
        self._use_2_opt = use_2_opt
        self._random = random
        self._vehicles = vehicles
        self._load = 0
        self._current_node = start_node
        self._capacity = capacity
        self.initialize([[start_node] for _ in vehicles])

    def initialize(self, vehicle_paths: list[VehiclePath]):
        Thread.__init__(self)
        self.vehicle_paths = vehicle_paths
        self.traveled_edges = []
        self.traveled_distance = 0

    @property
    def travelled_nodes(self) -> VehiclePath:
        return [n for path in self.vehicle_paths for n in path]

    def run(self):
        """Run the ant."""
        # Possible locations where the ant can got to from the current node without the location it has already been.
        depot = self.graph.depot
        while self.graph.get_connected_nodes(depot).difference(self.travelled_nodes):
            for vehicle_id, vehicle in enumerate(self._vehicles):
                self._make_step_for_vehicle(vehicle_id, vehicle)

    def _make_step_for_vehicle(self, vehicle_id: int, vehicle: Vehicle):
        path = self.vehicle_paths[vehicle_id]
        self._current_node = path[-1]
        possible_locations = self.graph.get_connected_nodes(
            self._current_node).difference(self.travelled_nodes)
        a, b = self._select_edge(possible_locations)
        demand = self.graph.get_vertex_demand(b)
        if self._load + demand <= self._capacity:
            self._load += demand
            self._move_to_next_node((a, b), vehicle_id)
        else:
            self._move_to_next_node((a, self.graph.depot), vehicle_id)
            self._load = 0

    def _select_edge(self, possible_locations):
        """Select the edge where to go next."""
        logger.debug('Possible locations="%s"', possible_locations)
        attractiveness = {}
        overall_attractiveness = .0

        if not possible_locations:
            return (self._current_node, self.graph.depot)

        for node in possible_locations:
            edge = (self._current_node, node)

            edge_pheromone = self.graph.get_edge_pheromone(edge)
            # Gets the rounded distance.
            distance = self.graph.get_edge_length(edge)
            if distance == 0:
                continue
            attractiveness[node] = pow(
                edge_pheromone, self._alpha)*pow(1/distance, self._beta)
            overall_attractiveness += attractiveness[node]

        if overall_attractiveness == 0:
            selected_edge = (self._current_node,
                             self._random.choice(list(attractiveness.keys())))
        else:
            choice = self._random.choice(list(attractiveness.keys()), p=list(
                attractiveness.values())/np.sum(list(attractiveness.values())))
            selected_edge = (self._current_node, choice)

        logger.debug('Selected edge: %s', (selected_edge,))
        return selected_edge

    def _move_to_next_node(self, edge_to_travel: tuple[int, int], vehicle_id):
        """Move to the next node and update member."""
        self.traveled_distance += self.graph.get_edge_length(edge_to_travel)
        self.vehicle_paths[vehicle_id].append(edge_to_travel[1])
        self.traveled_edges.append(edge_to_travel)
        logger.debug('Traveled from node="%s" to node="%s". Overall traveled distance="%s"',
                     self._current_node, edge_to_travel[1], self.traveled_distance)
        self._current_node = edge_to_travel[1]

    def spawn_pheromone(self):
        # loop all except the last item
        # TODO: take different vehicles into account
        for idx in range(len(self.travelled_nodes) - 1):
            traveled_edge = (
                self.travelled_nodes[idx], self.travelled_nodes[idx+1])
            pheromone = self.graph.get_edge_pheromone(traveled_edge)
            pheromone += self._Q / \
                max(self.graph.get_edge_length(traveled_edge), 1)
            self.graph.set_pheromone(traveled_edge, pheromone)
