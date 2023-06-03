# ------------------------------------------------------------------------------------------------------
#  Copyright (c) Leo Hanisch. All rights reserved.
#  Licensed under the BSD 3-Clause License. See LICENSE.txt in the project root for license information.
# ------------------------------------------------------------------------------------------------------
import logging
from typing import Callable
import seaborn as sns
import wandb
import pandas as pd
import numpy as np
from swarmlib_.aco4tsp.vehicle import VehiclePath, Vehicle

from swarmlib_.aco4tsp.dynamic_vrp_env.env import DynamicVrpEnv
from .ant import Ant
from ..util.problem_base import ProblemBase
from pathlib import Path
import plotly.graph_objects as go
import plotly

logger = logging.getLogger(__name__)


class ACOAlgorithm(ProblemBase):
    def __init__(self,
                 instance_path: Path,
                 vehicles: list[Vehicle],
                 commit_freq,
                 capacity,
                 scenario_input_file = None, #'./scenario.csv',
                 **kwargs):
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

        self._env = DynamicVrpEnv(instance_path, scenario_input_file=scenario_input_file)
        self._rho = kwargs.get('rho', 0.5)  # evaporation rate
        self._alpha = kwargs.get('alpha', 0.5)  # used for edge detection
        self._beta = kwargs.get('beta', 0.5)  # used for edge detection
        self._Q = kwargs.get('q', 1)  # Hyperparameter Q
        self._num_iterations = kwargs.get(
            'iteration_number', 10)  # Number of iterations
        self._vehicles = vehicles
        self._use_2_opt = True
        self._best_fitness = float('inf')
        for vehicle in self._vehicles:
            vehicle.traveled_path = [self._env.depot]
        
        self._capacity = capacity

        self._commit_freq = commit_freq
        self._previous_commit = 0
    
    def solve(self):
        """
        Solve the given problem.
        """
        ants = [
            Ant(self._env.depot,
                self._env._routes_graph,
                self._alpha,
                self._beta,
                self._Q,
                self._use_2_opt,
                self._random,
                self._vehicles,
                self._capacity)
            for _ in range(self._ant_number)
        ]

        for i in range(1, self._num_iterations):
            print('Iteration: ', i)
            shortest_distance = float('inf')
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
                # self._commit_paths(i, best_vehicle_paths)
                self.log_commit(best_vehicle_paths)

            _, change = self._env.step()
            if change:
                logger.debug(
                    f'[{change.type}] [{change.from_node}] -> [{change.to_node}]')

            self.log_iteration(best_vehicle_paths)

        logger.info(f'Finish! Shortest_distance="%s" and best_vehicle_paths="%s"',
                    shortest_distance, best_vehicle_paths)
        self._env.prepare_scenario_file()
        return best_vehicle_paths, shortest_distance

    def log_iteration(self, best_vehicle_paths):
        # uncommited_paths = self._extract_uncommited_paths(
        #    best_vehicle_paths)
        fitness = self._calculate_distance(best_vehicle_paths)
        best_fitness = min(self._best_fitness, fitness)
        self._best_fitness = best_fitness
        pheromone_ratio = self._calculate_pheromone_ratio()
        attractiveness_dispersion = self._calculate_attractiveness_dispersion()
        attractiveness_ratio = self._calculate_attractiveness_ratio(best_vehicle_paths)

        if wandb.run is not None:
            wandb.log({
                "fitness": fitness,
                "best_fitness": best_fitness,
                "pheromone_ratio": pheromone_ratio,
                "attractiveness_dispersion": attractiveness_dispersion,
                "attractiveness_ratio": attractiveness_ratio,
                "overall_demand": self._env.overall_demand()
            })

    def log_commit(self, best_vehicle_path):
        fig = self._create_figure(best_vehicle_path)
        if wandb.run is not None:
            wandb.log({'vehicle routes': fig})

    def _create_figure(self, best_vehicle_path):
        G = self._env._routes_graph
        edge_x = []
        edge_y = []
        vehicle_paths_traces = []
        colors = sns.color_palette("Set2", len(best_vehicle_path))
        for vehicle, color in zip(best_vehicle_path, colors):
            edge_x, edge_y = [], []
            for from_, to_ in zip(vehicle, vehicle[1:]):
                x0, y0 = G.get_node(from_).coord
                x1, y1 = G.get_node(to_).coord
                edge_x.append(x0)
                edge_x.append(x1)
                edge_x.append(None)
                edge_y.append(y0)
                edge_y.append(y1)
                edge_y.append(None)
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(
                    width=0.5, color=f'rgb({",".join(str(c*255) for c in color)})'),
                hoverinfo='none',
                mode='lines')
            vehicle_paths_traces.append(edge_trace)

        node_x = []
        node_y = []
        for node in G.get_nodes():
            x, y = node.coord
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Customers',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))

        return go.Figure(data=[node_trace, *vehicle_paths_traces],
                         layout=go.Layout(
            title='<br>Paths traveled by vehicles',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        )

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
        for vehicle, path in zip(self._vehicles, paths):
            vehicle.travel(self._env._routes_graph,
                           path, time_units=step - self._previous_commit)
        self._previous_commit = step

    def _extract_uncommited_paths(self, paths: list[VehiclePath]):
        return [full_path[len(vehicle.traveled_path)-1:]  # assumption is made, that commited and paths are in the same order
                for vehicle, full_path in zip(self._vehicles, paths)
                ]

    def _calculate_distance(self, paths: list[VehiclePath]):
        return sum(
            sum(self._env._routes_graph.get_edge_length((a, b))
                for a, b in zip(p, p[1:])
                ) for p in paths if len(p) > 1)
    
    def _calculate_threshold(self):
        # average pheromone level
        edges = self._env._routes_graph.get_edges()
        return sum(self._env._routes_graph.get_edge_pheromone(edge) for edge in edges) / len(edges)

    def _is_pheromone_level_significant(self, edge):
        if self._env._routes_graph.get_edge_pheromone(edge) > self._calculate_threshold():
          return True
        return False

    def _calculate_pheromone_ratio(self):
        edges = list(filter(self._is_pheromone_level_significant, self._env._routes_graph.get_edges()))
        return len(edges) / len(self._env._routes_graph.get_edges())
    
    def _calculate_attractiveness_dispersion(self):
        pheromone_values = list(map(self._env._routes_graph.get_edge_pheromone, self._env._routes_graph.get_edges()))
        return np.std(pheromone_values)

    def _calculate_attractiveness_ratio(self, best_paths: list[VehiclePath]):
        pheromone_values = list(map(self._env._routes_graph.get_edge_pheromone, self._env._routes_graph.get_edges()))
        best_path_pheromones_sum = sum(
            sum(self._env._routes_graph.get_edge_pheromone((a, b))
                for a, b in zip(p, p[1:])
                ) for p in best_paths if len(p) > 1)
        return (best_path_pheromones_sum / (sum(pheromone_values) - best_path_pheromones_sum))
