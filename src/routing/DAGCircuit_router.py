from typing import Optional
from typing import Callable
from src.states.state_handler import StateHandler
from qiskit import QuantumCircuit
from src.routing.router import Router
import heapq


class WeightedAStarSearch[S](Router):
    def __init__(self, state_handler: StateHandler[S], heuristic: Callable[[S], float], weight=1.0):
        self.state_handler = state_handler
        self.heuristic = heuristic
        self.weight = weight

    def solve(self, circuit: QuantumCircuit) -> list[int]:
        root_state = self.state_handler.state_from(circuit)
        path, _ = self.weighted_astar(
            root_state, self.state_handler, self.heuristic, self.weight
        )
        if path is None:
            raise ValueError("No solution found")
        
        return path
        

    def weighted_astar(
        self,
        start: S,
        state_handler: StateHandler[S],
        heuristic: Callable[[S], float],
        weight: float = 1.0,
    ) -> tuple[Optional[list[int]], dict[S, float]]:
        """
        Weighted A* search over an implicit graph.

        Args:
            start: initial state
            handler: instance of StateHandler[S]
            heuristic: function h(state) -> estimated cost to goal
            weight: w >= 1.0 (w=1 -> standard A*, w>1 -> greedier, faster, suboptimal)

        Returns:
            (path, dist)
            path: list of states from start to goal (or None if not found)
            dist: map of state -> best known g-cost
        """

        # Priority queue entries: (f, g, state)
        pq: list[tuple[float, float, int, S]] = []

        g_cost: dict[S, float] = {start: 0.0}

        # Store: state -> (parent_state, action_taken)
        parent: dict[S, tuple[S, int]] = {}

        start_f = weight * heuristic(start)
        counter = 0
        heapq.heappush(pq, (start_f, 0.0, counter, start))

        while pq:
            f, g, _, state = heapq.heappop(pq)

            # Skip stale entries
            if g > g_cost.get(state, float("inf")):
                continue
            
            if state_handler.is_terminal(state):
                return self.reconstruct_actions(parent, start, state), g_cost

            for action in state_handler.get_possible_actions(state):
                next_state = state_handler.get_next_state(state, action)
                step_cost = state_handler.get_action_cost(state, action)

                new_g = g + step_cost

                if new_g < g_cost.get(next_state, float("inf")):
                    g_cost[next_state] = new_g
                    parent[next_state] = (state, action)

                    h = heuristic(next_state)
                    new_f = new_g + weight * h
                    counter += 1
                    heapq.heappush(pq, (new_f, new_g, counter, next_state))

        return None, g_cost

    @staticmethod
    def reconstruct_actions(parent, start, goal):
        actions = []
        current = goal

        while current != start:
            entry = parent.get(current)
            if entry is None:
                return None

            prev_state, action = entry
            actions.append(action)
            current = prev_state

        actions.reverse()
        return actions


if __name__ == "__main__":
    from src.routing.relaxed_heurisitc import RelaxedCountHeuristic, RelaxedDijkstraHeuristic, RelaxedDepthHeuristic
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    circuit = CircuitGenerator.generate_random_circuit(6, gateset={"cx"}, num_gates=20)
    with open("circuits/dud4.qpy", "wb") as f:
        dump(circuit, f)

    # with open("circuits/dud2.qpy", "rb") as f:
    #     circuit = load(f)[0]

    print(circuit)

    state_handler = DAGCircuitStateHandler(n_qubits, coupling_map)
    #heuristic = RelaxedCountHeuristic(state_handler)
    heuristic = RelaxedDepthHeuristic(state_handler)
    #heuristic = RelaxedDijkstraHeuristic(state_handler)

    router = WeightedAStarSearch(state_handler, heuristic, weight=1.0)
    swap_inserter = SwapInserter(coupling_map, num_qubits=n_qubits)

    rl_pass = RlRoutingPass(router, swap_inserter)
    pm = PassManager(rl_pass)
    start = time.time()
    final_circuit = pm.run(circuit)
    end = time.time()

    print("Original circuit:")
    print(circuit)
    print(f"Routed circuit (found in {end - start:.2f} seconds):")
    print("Routed circuit depth:", final_circuit.depth())
    print("Routed circuit:")
    print(final_circuit)

    org_op = Operator.from_circuit(circuit)
    routed_op = Operator.from_circuit(final_circuit)

    assert routed_op.equiv(org_op), (
        "The original and routed circuits are not equivalent!"
    )
