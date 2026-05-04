from copy import deepcopy
from abc import abstractmethod
from abc import ABC
from src.states.DAGCircuit_state_handler import DAGCircuitState
from src.states.DAGCircuit_state_handler import DAGCircuitStateHandler
import heapq


class DAGCircuitHeurisitc(ABC):
    def __init__(self, state_handler: DAGCircuitStateHandler):
        self.state_handler = state_handler

    @abstractmethod
    def __call__(self, state: DAGCircuitState) -> float:
        raise NotImplementedError


class CountHeuristic(DAGCircuitHeurisitc):
    def __call__(self, state: DAGCircuitState) -> float:
        return len(state.dag.op_nodes())


class DepthHeuristic(DAGCircuitHeurisitc):
    def __call__(self, state: DAGCircuitState) -> float:
        return state.dag.depth()

class SabreBasicHeuristic(DAGCircuitHeurisitc):        
    def __call__(self, state: DAGCircuitState) -> float:
        total_distance = 0
        for node in state.dag.front_layer(): # Calculate distance for gates in front layer only.
            qubits = node.qargs
            physical_qubits = [state.layout[q] for q in qubits]

            if len(physical_qubits) == 2:
                distance = self.state_handler.cm.distance(*physical_qubits)
                total_distance += distance

        return total_distance

class TotalQubitDistanceHeuristic(DAGCircuitHeurisitc):        
    def __call__(self, state: DAGCircuitState) -> float:
        total_distance = 0
        for node in state.dag.op_nodes(): # Calculate distance for all gates.
            qubits = node.qargs
            physical_qubits = [state.layout[q] for q in qubits]

            if len(physical_qubits) == 2:
                distance = self.state_handler.cm.distance(*physical_qubits)
                total_distance += distance

        return total_distance

class RelaxedDijkstraHeuristic(DAGCircuitHeurisitc):
    def __call__(self, state: DAGCircuitState) -> float:
        state = deepcopy(state)

        goal_dist = self.dijkstra(
            state,
            lambda s: [
                (self.get_relaxed_next_state(s, action), 0.5)
                for action in self.state_handler.get_possible_actions(s)
            ],
            self.state_handler.is_terminal,
        )

        return goal_dist

    def get_relaxed_next_state(
        self, state: DAGCircuitState, action: int
    ) -> DAGCircuitState:
        next_state = deepcopy(state)
        for node in next_state.dag.op_nodes():
            qubits = node.qargs
            physical_qubits = [next_state.layout[q] for q in qubits]

            if len(physical_qubits) == 1:
                next_state.dag.remove_op_node(node)

            elif len(physical_qubits) == 2:
                if any(
                    set(physical_qubits) == set(swap)
                    for swap in self.state_handler.swaps
                ):
                    next_state.dag.remove_op_node(node)

            else:
                raise ValueError(
                    "Only 1 or 2 qubit gates supported in relaxed heuristic"
                )

        next_state.layout.swap(*self.state_handler.swaps[action])
        return next_state

    @staticmethod
    def dijkstra(start, neighbors_fn, goal_fn):
        """
        start: initial state
        neighbors_fn: function(state) -> iterable of (next_state, cost)
        goal_fn: function(state) -> bool

        Returns:
        cost of the shortest path to a goal state.
        """

        counter = 0
        pq = [(0, counter, start)]  # (distance, counter, state)
        dist = {start: 0}
        prev = {}
        
        while pq:
            current_dist, _, state = heapq.heappop(pq)

            # Skip outdated entries
            if current_dist > dist[state]:
                continue

            # Early exit if goal reached
            if goal_fn(state):
                return current_dist

            for next_state, cost in neighbors_fn(state):
                new_dist = current_dist + cost

                if next_state not in dist or new_dist < dist[next_state]:
                    dist[next_state] = new_dist
                    prev[next_state] = state
                    counter += 1
                    heapq.heappush(pq, (new_dist, counter, next_state))

        raise ValueError("Goal state not reachable from start state")

if __name__ == "__main__":
    from src.circuit_generator import CircuitGenerator
    from qiskit.qpy import load, dump
    import torch
    from src.states.circuit_graph_state_handler import CircuitGraphStateHandler
    from src.model import BiCircuitGNN
    from src.routing.swap_inserter.swap_inserter import SwapInserter
    from src.states.DAGCircuit_state_handler import DAGCircuitStateHandler
    from src.routing.rl_routing_pass import RlRoutingPass
    from qiskit.transpiler import CouplingMap, PassManager
    from qiskit.quantum_info import Operator
    import time

    n_qubits = 6
    horizon = 100
    coupling_map = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

    circuit = CircuitGenerator.generate_random_circuit(6, gateset={"cx"}, num_gates=8)
    with open("circuits/dud2.qpy", "wb") as f:
        dump(circuit, f)

    # with open("circuits/dud2.qpy", "rb") as f:
    #     circuit = load(f)[0]

    print(circuit)

    state_handler = DAGCircuitStateHandler(n_qubits, coupling_map)
    #heuristic = RelaxedCountHeuristic(state_handler)
    heuristic = RelaxedDijkstraHeuristic(state_handler)

    root_state = state_handler.state_from(circuit)
    h_value = heuristic(root_state)
    
    print(h_value)
    

    
    
    
    
    