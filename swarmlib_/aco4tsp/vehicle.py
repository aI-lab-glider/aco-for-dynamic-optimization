# pylint: disable=too-many-instance-attributes,invalid-name,too-many-locals

from dataclasses import dataclass, field
from itertools import accumulate
from swarmlib_.aco4tsp.dynamic_vrp_env.routes_graph import RoutesGraph

VechiclePath = list[int]


@dataclass
class Vehicle:
    """
    Vehicle class

    Attributes:
        capacity (int): Vehicle capacity
        speed (int): Vehicle speed in distance units per time unit (i.e. speed 10 is equivalent to distance of 10 )
        path (list[int]): List of nodes to visit
        traveled_path (list[int]): List of nodes visited
    """
    capacity: int
    speed: int
    traveled_path: VechiclePath = field(default_factory=list)

    def travel(self, routes_graph: RoutesGraph, path: VechiclePath, time_units: int):
        """Extends traveled path with the nodes traveled in the given time units"""
        assert self.path_is_valid(path), f"First nodes in path should be the same as traveled_path. Instead, path is: {path} and traveled_path is: {self.traveled_path}"
        path_to_travel = path[len(self.traveled_path):]
        distances_between_points = [routes_graph.get_edge_length(route) for route in zip(path_to_travel, path_to_travel[1:])]
        cumulated_distances = accumulate(distances_between_points)
        traveled_distance = self.speed * time_units
        traveled_path = [node for node, dist in zip(path_to_travel, cumulated_distances) if dist <= traveled_distance] 
        self.traveled_path.extend(traveled_path)

    def path_is_valid(self, path):
        return all(path_node == traveled_node for path_node, traveled_node in zip(path, self.traveled_path))