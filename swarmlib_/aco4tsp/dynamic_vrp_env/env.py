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
    def __init__(self, data_file: Path, scenario_input_file = None) -> None:
        super().__init__()
        self._counter = 0
        self._next_step = None
        self._initial_df = pd.read_csv(data_file)
        if scenario_input_file:
            self._scenario = pd.read_csv(scenario_input_file)
            self._node_name_history = list(self._scenario['name'].to_list())
            self._node_demand_history = list(self._scenario['demand'].to_list())
            self._steps_history = list(self._scenario['step'].to_list())
            self._next_step = self._steps_history.pop(0)
        else:
            self._scenario = None
        self._routes_graph = RoutesGraph(self._initial_df)
        self._history = pd.DataFrame()

    def overall_demand(self):
        return sum(node.demand for node in self._routes_graph.get_nodes())

    @property
    def depot(self):
        return self._routes_graph.depot

    def step(self):
        change = None
        if self._scenario is None:
            change = self._change_routes_graph() if random.random() < 0.7 else None
        else:
            if (self._next_step is not None) and self._counter == self._next_step:
                change = self._change_routes_graph()
                self._next_step = self._steps_history.pop(0) if len(self._steps_history) else None
        
        self._counter = self._counter + 1
        return self._routes_graph, change
    
    def prepare_scenario_file(self):
        filepath = Path('./out.csv')  
        filepath.parent.mkdir(parents=True, exist_ok=True)  
        self._history.to_csv(filepath, index=False)  

    def _change_routes_graph(self):
        change = self._update_demand
        # random.choices(
        # population=[
        # self._update_demand,
        # self._remove_node,
        # self._add_node],
        # weights=[0.25],
        # k=1
        # )[0]
        return change()

    def _update_demand(self):
        if self._scenario is None:
            new_demand = random.randint(0, self._routes_graph.max_demand())
            node = random.choice(self._routes_graph.get_nodes())
            while (node.is_depot == True):
                node = random.choice(self._routes_graph.get_nodes())
        else:
            new_demand = self._node_demand_history.pop(0)
            node = self._routes_graph.get_node(self._node_name_history.pop(0))
        new_node = Node(**(asdict(node) | {'demand': new_demand}))
        self._history = pd.concat([self._history, pd.DataFrame([new_node]).assign(step=self._counter)], ignore_index=True)
        self._routes_graph.update_node(node.name, new_node)
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
