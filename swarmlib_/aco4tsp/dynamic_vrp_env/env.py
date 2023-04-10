from swarmlib_.aco4tsp.dynamic_vrp_env.routes_graph import Node, RoutesGraph
from pathlib import Path
import pandas as pd
import random
from dataclasses import asdict, dataclass


@dataclass
class ChangeDetails:
    type: str
    from_node: Node
    to_node: Node


class DynamicVrpEnv:
    def __init__(self, data_file: Path) -> None:
        super().__init__()
        self._initial_df = pd.read_csv(data_file)
        self._routes_graph = RoutesGraph(self._initial_df)

    @property
    def depot(self):
        return self._routes_graph.depot

    def step(self):
        change = self._change_routes_graph() if random.random() < 0.4 else None
        return self._routes_graph, change

    def _change_routes_graph(self):
        change = random.choices(
            population=[
                self._update_demand,
                self._remove_node,
                self._add_node],
            weights=[0.5, 0.25, 0.25],
            k=1
        )[0]
        return change()

    def _update_demand(self):
        new_demand = random.randint(0, self._routes_graph.max_demand())
        node = random.choice(self._routes_graph.get_nodes())
        new_node = Node(**(asdict(node) | {'demand': new_demand}))
        self._routes_graph.update_node(node.name, node)
        return ChangeDetails('increase_demand', from_node=node, to_node=new_node)

    def _remove_node(self):
        node = random.choice(self._routes_graph.get_nodes())
        self._routes_graph.remove_node(node.name)
        return ChangeDetails('remove_node', node, None)

    def _add_node(self):
        present = {n.name for n in self._routes_graph.get_nodes()}
        candidates = set(self._initial_df['cust'].to_list()) - present
        if not candidates:
            return ChangeDetails('add_node', None, None)
        chosen = random.choice(list(candidates))
        node = Node.from_series(
            self._initial_df[self._initial_df['cust'] == chosen])
        self._routes_graph.add_node(node)
        return ChangeDetails('add_node', None, node)
