# ------------------------------------------------------------------------------------------------------
#  Copyright (c) Leo Hanisch. All rights reserved.
#  Licensed under the BSD 3-Clause License. See LICENSE.txt in the project root for license information.
# ------------------------------------------------------------------------------------------------------

import inspect
import logging
from os import path

import pandas as pd

from swarmlib_.aco4tsp.dynamic_vrp_env.env import DynamicVrpEnv
from .ant import Ant
from .dynamic_vrp_env.routes_graph import RoutesGraph
from .visualizer import Visualizer
from ..util.problem_base import ProblemBase
from pathlib import Path

LOGGER = logging.getLogger(__name__)

# pylint: disable=too-many-instance-attributes,invalid-name,too-many-locals


class ACOAlgorithm(ProblemBase):
    def __init__(self, instance_path: Path, ant_capacity, **kwargs):
        """Initializes a new instance of the `ACOProblem` class.

        Arguments:  \r
        `ant_number` -- Number of ants used for solving

        Keyword arguments:  \r
        `rho`           -- Evaporation rate (default 0.5)  \r
        `alpha`         -- Relative importance of the pheromone (default 0.5)  \r
        `beta`          -- Relative importance of the heuristic information (default 0.5)  \r
        `q`             -- Constant Q. Used to calculate the pheromone, laid down on an edge (default 1)  \r
        `iterations`    -- Number of iterations to execute (default 10)  \r
        `plot_interval` -- Plot intermediate result after this amount of iterations (default 10) \r
        `two_opt`       -- Additionally use 2-opt local search after each iteration (default true)
        """
        super().__init__(**kwargs)
        self.__ant_number = kwargs['ant_number']  # Number of ants

        self._env = DynamicVrpEnv(instance_path)

        self.__rho = kwargs.get('rho', 0.5)  # evaporation rate
        self.__alpha = kwargs.get('alpha', 0.5)  # used for edge detection
        self.__beta = kwargs.get('beta', 0.5)  # used for edge detection
        self.__Q = kwargs.get('q', 1)  # Hyperparameter Q
        self.__num_iterations = kwargs.get(
            'iteration_number', 10)  # Number of iterations
        self.__use_2_opt = kwargs.get('two_opt', False)

        self._visualizer = Visualizer(**kwargs)
        self.ant_capacity = ant_capacity

    def solve(self):
        """
        Solve the given problem.
        """

        ants = []
        shortest_distance = None
        best_path = None

        # Create ants
        ants = [
            Ant(self._env.depot,
                self._env._routes_graph, self.__alpha, self.__beta, self.__Q, self.__use_2_opt, self._random, self.ant_capacity)
            for _ in range(self.__ant_number)
        ]

        for _ in range(self.__num_iterations):
            # Start all multithreaded ants
            _, env_change = self._env.step()
            if env_change:
                print(
                    f'[{env_change.type}] [{env_change.from_node}] -> [{env_change.to_node}]')

            for ant in ants:
                ant.start()

            # Wait for all ants to finish
            for ant in ants:
                ant.join()

            # decay pheromone
            edges = self._env._routes_graph.get_edges()
            for edge in edges:
                pheromone = self._env._routes_graph.get_edge_pheromone(edge)
                pheromone *= 1-self.__rho
                self._env._routes_graph.set_pheromone(edge, pheromone)

            # Add each ant's pheromone
            for ant in ants:

                ant.spawn_pheromone()
                # Check for best path
                if not shortest_distance or ant.traveled_distance < shortest_distance:
                    shortest_distance = ant.traveled_distance
                    best_path = ant.traveled_nodes
                    print(
                        f'Updated shortest_distance="{shortest_distance}" and best_path="{best_path}"')

                # Reset ants' thread
                ant.initialize(self._env._routes_graph.depot)

            self._visualizer.add_data(
                best_path=best_path,
                pheromone_map={edge: self._env._routes_graph.get_edge_pheromone(
                    edge) for edge in edges},
                fitness=shortest_distance
            )

        LOGGER.info(f'Finish! Shortest_distance="%s" and best_path="%s"',
                    shortest_distance, best_path)
        return best_path, shortest_distance

    def replay(self):
        """
        Play the visualization of the problem
        """
        return self._visualizer.replay(graph=self._env._routes_graph)
    
    def plot_result(self):
        return self._visualizer.plot_result(self._env._routes_graph)

    def plot_fitness(self):
        """
        Plot the fitness over time
        """
        return self._visualizer.plot_fitness()
