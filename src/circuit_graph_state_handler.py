from state_handler import Batchable
from torch_geometric.loader import DataLoader
from state_handler import StateHandler
from circuit_graph import CircuitGraph
import torch
import random


class CircuitGraphStateHandler(StateHandler[CircuitGraph]):
    def __init__(self, n_qubits: int, topology: list[tuple[int, int]]):
        self.n_qubits = n_qubits
        self.topology = topology

    def get_possible_actions(self, state: CircuitGraph) -> list[int]:
        return list(range(len(self.topology)))

    def get_next_state(self, state: CircuitGraph, action: int) -> CircuitGraph:
        state = self.prune(state)[0]
        if state.x is None or state.edge_index is None or state.edge_attr is None:
            raise ValueError("State must have x, edge_index, and edge_attr defined")

        new_state = state.clone()

        # make type checker happy
        assert not (
            new_state.x is None
            or new_state.edge_index is None
            or new_state.edge_attr is None
        ), "State must have x, edge_index, and edge_attr defined"

        n_qubits = state.x.shape[1] // 2
        q1, q2 = self.topology[action]

        # swap first qubits
        new_state.x[:, q1] = state.x[:, q2]
        new_state.x[:, q2] = state.x[:, q1]

        # swap second qubits
        new_state.x[:, n_qubits + q1] = state.x[:, n_qubits + q2]
        new_state.x[:, n_qubits + q2] = state.x[:, n_qubits + q1]

        for i, att in enumerate(new_state.edge_attr):
            if att[q1] > 0:
                new_state.edge_attr[i][q1] = 0
                new_state.edge_attr[i][q2] = att[q1]
            elif att[q2] > 0:
                new_state.edge_attr[i][q2] = 0
                new_state.edge_attr[i][q1] = att[q2]

        return new_state

    def is_terminal(self, state: CircuitGraph) -> bool:
        if state.x is None:
            raise ValueError("State must have x defined")

        removed_gages = self.get_removed_gates(state)
        return (
            len(removed_gages) == state.x.shape[0] - 1
        )  # all gates removed, only global node left

    def get_action_cost(self, state: CircuitGraph, action: int) -> float:
        if state.x is None:
            raise ValueError("State must have x defined")

        removed_gates = self.get_removed_gates(state)
        n_qubits = state.x.shape[1] // 2
        q1, q2 = self.topology[action]
        for removed_gate in removed_gates[::-1]:
            gate_q1 = torch.where(state.x[removed_gate, :n_qubits] > 0)[0].item()
            gate_q2 = torch.where(state.x[removed_gate, n_qubits:] > 0)[0].item()
            if gate_q1 == q1 and gate_q2 == q2:  # exact match
                return 0.5
            if (
                gate_q1 == q2 or gate_q2 == q1
            ):  # partial match can not reduce cnot amount
                return 1.0

        return 1.0

    def get_removed_gates(self, state: CircuitGraph) -> list[int]:
        if state.x is None or state.edge_index is None or state.edge_attr is None:
            return []

        frontlayer = set(i for i in range(state.x.shape[0] - 1))
        n_directed_edges = (
            state.edge_index.shape[1] - (state.x.shape[0] - 1) * 2
        )  # Exclude global node edges
        for succ, prev in state.edge_index.t()[
            :n_directed_edges
        ]:  # loop over edges, excluding global node edges
            frontlayer.discard(
                int(succ.item())
            )  # if a gate is a successor, it is not in the frontlayer
            if len(frontlayer) == 0:
                break

        removed_gates: list[int] = []
        for gate_index in frontlayer:
            q1 = torch.where(state.x[gate_index, : state.x.shape[1] // 2] > 0)[0].item()
            q2 = torch.where(state.x[gate_index, state.x.shape[1] // 2 :] > 0)[0].item()
            if (q1, q2) in self.topology or (q2, q1) in self.topology:
                removed_gates.append(gate_index)

        return removed_gates

    def prune(self, state: CircuitGraph) -> tuple[CircuitGraph, int]:
        if state.x is None or state.edge_index is None or state.edge_attr is None:
            return state, 0

        removed_gates = self.get_removed_gates(state)

        if len(removed_gates) == 0:
            return state, 0

        x = torch.stack(
            [state.x[i] for i in range(state.x.shape[0]) if i not in removed_gates]
        )

        edge_index = []
        edge_attr = []

        # remove edges associated with removed gates
        n_directed_edges = (
            state.edge_index.shape[1] - (state.x.shape[0] - 1) * 2
        )  # Exclude global node edges
        for i, (succ, prev) in enumerate(state.edge_index.t()):
            if i < n_directed_edges and succ.item() in removed_gates:
                raise ValueError("Acedently removed gate not in frontlayer")

            if prev.item() in removed_gates or succ.item() in removed_gates:
                continue

            edge_index.append((succ.item(), prev.item()))
            edge_attr.append(state.edge_attr[i])

        if len(edge_index) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, state.edge_attr.shape[1]), dtype=torch.float)
            new_state = CircuitGraph(x=x, edge_index=edge_index, edge_attr=edge_attr)
            return new_state, len(removed_gates)

        # shift node indexes to account for removed gates
        for i in range(len(edge_index)):
            succ_index, prev_index = edge_index[i]
            for removed_gate in sorted(removed_gates):
                if succ_index > removed_gate:
                    succ_index -= 1
                if prev_index > removed_gate:
                    prev_index -= 1
            edge_index[i] = (succ_index, prev_index)

        edge_index = torch.tensor(edge_index).t()
        edge_attr = torch.stack(edge_attr)

        new_state = CircuitGraph(x=x, edge_index=edge_index, edge_attr=edge_attr)
        new_state, n_pruned_gates = self.prune(new_state)

        return new_state, len(removed_gates) + n_pruned_gates

    def get_random_state(self, max_difficulty: int) -> CircuitGraph:
        n_gates = random.randint(1, max_difficulty)

        qc = QuantumCircuit(self.n_qubits)
        for _ in range(n_gates):
            q1, q2 = random.sample(range(self.n_qubits), 2)
            while q1 == q2:
                q2 = random.choice(range(self.n_qubits))
            qc.cx(q1, q2)

        state = CircuitGraph.from_circuit(qc)
        if self.is_terminal(state):
            return self.get_random_state(max_difficulty)

        return state

    def get_random_states(self, batch_size: int, max_difficulty: int):
        return [self.get_random_state(max_difficulty) for _ in range(batch_size)]

    def batch_states(self, states: Batchable[CircuitGraph]) -> CircuitGraph:
        return next(
            combined_state
            for combined_state in DataLoader(states, batch_size=len(states))
        )


if __name__ == "__main__":
    from qiskit import QuantumCircuit
    from cnot_circuit import CNOTCircuit

    topology = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    n_qubits = 6
    game = CircuitGraphStateHandler(6, topology)

    circuit = QuantumCircuit(6)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.cx(3, 5)

    print(circuit)
    state = CircuitGraph.from_circuit(circuit)
    print(state.edge_index)
    new_state = game.get_next_state(state, 2)

    print(new_state.x)
    print(new_state.edge_index)
    print(new_state.edge_attr)
    new_circuit = CNOTCircuit.from_circuit_graph(new_state)
    print(new_circuit)

    print("new_state is terminal:", game.is_terminal(new_state))
