# ------------------------------------------------------------------------------------------------------
#  Copyright (c) Leo Hanisch. All rights reserved.
#  Licensed under the BSD 3-Clause License. See LICENSE.txt in the project root for license information.
# ------------------------------------------------------------------------------------------------------
import logging
import pandas as pd
from swarmlib_.aco4tsp.vehicle import VechiclePath, Vehicle

from swarmlib_.aco4tsp.dynamic_vrp_env.env import DynamicVrpEnv
from .ant import Ant
from .dynamic_vrp_env.routes_graph import RoutesGraph
from .visualizer import Visualizer
from ..util.problem_base import ProblemBase
from pathlib import Path

logger = logging.getLogger(__name__)


class ACOAlgorithm(ProblemBase):
    def __init__(self, instance_path: Path, vehicles: list[Vehicle], commit_freq,  **kwargs):
        """Initializes a new instance of the `ACOProblem` class.

        Arguments:
        `ant_number` -- Number of ants used for solving

        Keyword arguments:
        `rho`           -- Evaporation rate (default 0.5)
        `alpha`         -- Relative importance of the pheromone (default 0.5)
        `beta`          -- Relative importance of the heuristic information (default 0.5)
        `q`             -- Constant Q. Used to calculate the pheromone, laid down on an edge (default 1)
        `iterations`    -- Number of iterations to execute (default 10)
        `plot_interval` -- Plot intermediate result after this amount of iterations (default 10)
        `two_opt`       -- Additionally use 2-opt local search after each iteration (default true)
        """
        super().__init__(**kwargs)
        self._ant_number = kwargs['ant_number']  # Number of ants

        self._env = DynamicVrpEnv(instance_path)

        self._rho = kwargs.get('rho', 0.5)  # evaporation rate
        self._alpha = kwargs.get('alpha', 0.5)  # used for edge detection
        self._beta = kwargs.get('beta', 0.5)  # used for edge detection
        self._Q = kwargs.get('q', 1)  # Hyperparameter Q
        self._num_iterations = kwargs.get(
            'iteration_number', 10)  # Number of iterations
        self._use_2_opt = kwargs.get('two_opt', False)

        self._visualizer = Visualizer(**kwargs)
        # self.ant_capacity = vehicle_capacity

        self._vehicles = vehicles
        for vehicle in self._vehicles:
            vehicle.traveled_path = [self._env.depot]

        # self._commited_paths = [(self._env._routes_graph.depot,)
        #                         for _ in self._vehicles]
        self._commit_freq = commit_freq
        self._previous_commit = 0
        self._history = []

    def solve(self):
        """
        Solve the given problem.
        """
        ants = [
            Ant(self._env.depot,
                self._env._routes_graph, self._alpha, self._beta, self._Q, self._use_2_opt, self._random, self._vehicles)
            for _ in range(self._ant_number)
        ]
        shortest_distance = float('inf')

        for i in range(1, self._num_iterations):
            self._run_ants(ants)
            self._decay_pheromone()
            for ant in ants:
                ant.spawn_pheromone()
                if ant.traveled_distance < shortest_distance:
                    shortest_distance = ant.traveled_distance
                    best_vehicle_paths = ant.vehicle_paths
                    logger.debug(
                        f'Updated shortest_distance="{shortest_distance}" and best_vehicle_paths="{best_vehicle_paths}"')
                # Reset ants' thread
                ant.initialize(
                    [v.traveled_path for v in self._vehicles])

            if i % self._commit_freq == 0:
                self._commit_paths(i, best_vehicle_paths)

            _, change = self._env.step()
            if change:
                logger.debug(
                    f'[{change.type}] [{change.from_node}] -> [{change.to_node}]')

            uncommited_paths = self._extract_uncommited_paths(
                best_vehicle_paths)
            fitness = self._calculate_distance(uncommited_paths)
            self._history.append(fitness)

        logger.info(f'Finish! Shortest_distance="%s" and best_vehicle_paths="%s"',
                    shortest_distance, best_vehicle_paths)
        return best_vehicle_paths, shortest_distance

    def _run_ants(self, ants: list[Ant]):
        for ant in ants:
            ant.start()
        # Wait for all ants to finish
        for ant in ants:
            ant.join()

    def _decay_pheromone(self):
        edges = self._env._routes_graph.get_edges()
        for edge in edges:
            pheromone = self._env._routes_graph.get_edge_pheromone(edge)
            pheromone *= 1 - self._rho
            self._env._routes_graph.set_pheromone(edge, pheromone)

    def _commit_paths(self, step, paths):
        uncommited_paths = self._extract_uncommited_paths(paths)
        for vehicle, path in zip(self._vehicles, uncommited_paths):
            vehicle.travel(self._env._routes_graph,
                           path, time_units=step - self._previous_commit)
        self._previous_commit = step

    def _extract_uncommited_paths(self, paths: list[VechiclePath]):
        return [full_path[len(vehicle.traveled_path)-1:]  # assumption is made, that commited and paths are in the same order
                for vehicle, full_path in zip(self._vehicles, paths)
                ]

    def _calculate_distance(self, paths: list[VechiclePath]):
        return sum(
            sum(self._env._routes_graph.get_edge_length((a, b))
                for a, b in zip(p, p[1:])
                ) for p in paths if len(p) > 1)

    def plot_result(self):
        return self._visualizer.plot_result(self._env._routes_graph)

    def replay(self):
        """
        Play the visualization of the problem
        """
        return self._visualizer.replay(graph=self._env._routes_graph)

    def plot_fitness(self):
        """
        Plot the fitness over time
        """
        return self._visualizer.plot_fitness()
