# pylint: disable=too-many-instance-attributes,invalid-name,too-many-locals

from dataclasses import dataclass, field
from itertools import accumulate
from swarmlib_.aco4tsp.dynamic_vrp_env.routes_graph import RoutesGraph

VechiclePath = list[int]


@dataclass
class Vehicle:
    """
    Vehicle class

    """
    capacity: int
    speed: int
    traveled_distance: float = 0

    @property
    def traveled_path(self):
        return self._traveled_path[:]

    @traveled_path.setter
    def traveled_path(self, value):
        self._traveled_path = value

    def __post_init__(self):
        self._traveled_path = []

    def travel(self, routes_graph: RoutesGraph, path: VechiclePath, time_units: int):
        """Extends traveled path with the nodes traveled in the given time units"""
        assert self.path_is_valid(
            path), f"First nodes in path should be the same as traveled_path. Instead, path is: {path} and traveled_path is: {self.traveled_path}"
        path_to_travel = path[len(self.traveled_path):]
        distances_between_points = self._calculate_path_distances(
            routes_graph, path_to_travel)
        cumulated_distances = accumulate(distances_between_points)

        not_commited_distance = self.traveled_distance - \
            sum(self._calculate_path_distances(
                routes_graph, self.traveled_path))
        distance_to_travel = self.speed * time_units + not_commited_distance

        path_to_travel = [node for node, dist in zip(
            path_to_travel, cumulated_distances) if dist <= distance_to_travel]
        self._traveled_path.extend(path_to_travel)
        self.traveled_distance += distance_to_travel

    def path_is_valid(self, path):
        return all(path_node == traveled_node for path_node, traveled_node in zip(path, self._traveled_path))

    def _calculate_path_distances(self, routes_graph, path):
        if len(path) == 1:
            return [0]
        return [routes_graph.get_edge_length(
            route) for route in zip(path, path[1:])]
