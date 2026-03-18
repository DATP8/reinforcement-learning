from multiprocessing.sharedctypes import Value
from sympy.printing.pytorch import torch

from .state_handler import Batchable, StateHandler
from .cnot_circuit import CNOTCircuit

from qiskit import QuantumCircuit
from state_handler import StateHandler
import random
import torch


class TensorStateHandler(StateHandler[torch.Tensor]):
    def __init__(self, n_qubits: int, horizon: int, topology: list[tuple[int, int]]):
        self.n_qubits = n_qubits
        self.horizon = horizon
        self.topology = topology
        self.mask = self.init_mask(n_qubits)

    def init_mask(self, n_qubits: int):
        mask = torch.ones((n_qubits, n_qubits), dtype=torch.float32)
        for q1, q2 in self.topology:
            mask[q1, q2] = 0.0
            mask[q2, q1] = 0.0
        return mask
    
    def get_topology(self):
        return self.topology

    def get_qubits(self):
        return self.n_qubits

    def get_possible_actions(self, state: torch.Tensor) -> list[int]:
        return list(range(len(self.topology)))

    def get_restricted_actions(self, state: torch.Tensor):
        pruned_state, _ = self.prune(state)
        frontlayer_qubits, _ = self.get_front_layer_qubits(pruned_state)
        return [
            i
            for i, (q1, q2) in enumerate(self.topology)
            if (q1 in frontlayer_qubits or q2 in frontlayer_qubits)
        ]

    def prune(self, state: torch.Tensor) -> tuple[torch.Tensor, int]:
        new_state = state.clone()
        layers_removed = 0
        new_state[:, :, 0] *= self.mask

        while (
            layers_removed < self.horizon - 1
            and torch.sum(new_state[:, :, layers_removed]) <= 1e-7
        ):
            if torch.sum(new_state[:, :, layers_removed + 1]) <= 1e-7:
                break
            new_state[:, :, layers_removed + 1] *= self.mask
            layers_removed += 1

        new_state[:, :, : self.horizon - layers_removed] = new_state[
            :, :, layers_removed:
        ]
        new_state[:, :, self.horizon - layers_removed :] = 0.0

        return new_state, layers_removed

    def get_next_state(self, state: torch.Tensor, action: int) -> torch.Tensor:
        # If the first layer is now empty, shift all layers by one
        new_state, layers_removed = self.prune(state)
        q1, q2 = self.topology[action]

        # Swap the qubits in the tensor representation
        new_state_temp = new_state.clone()
        new_state[q1, :, :] = new_state_temp[q2, :, :]
        new_state[q2, :, :] = new_state_temp[q1, :, :]

        new_state_temp = new_state.clone()
        new_state[:, q1, :] = new_state_temp[:, q2, :]
        new_state[:, q2, :] = new_state_temp[:, q1, :]

        return new_state

    def is_terminal(self, state: torch.Tensor) -> bool:
        pruned_state, _ = self.prune(state)
        return torch.sum(pruned_state).item() <= 1e-7

    def get_action_cost(self, state: torch.Tensor, action: int) -> float:
        _, layers_removed = self.prune(state)
        _, backlayer_gates = self.get_front_layer_qubits(
            torch.flip(state[:, :, :layers_removed], dims=[-1])
        )

        # check if swap is successor of removed cnot gates, if so return lower cost
        q1, q2 = self.topology[action]
        if (q1, q2) in backlayer_gates or (q2, q1) in backlayer_gates:
            return 0.5

        return 1.0

    def get_front_layer_qubits(
        self, state: torch.Tensor
    ) -> tuple[set[int], set[tuple[int, int]]]:
        visited_qubits = set()
        frontlayer_qubits = set()
        frontlayer_gates = set()

        for i in range(state.shape[-1]):
            for q in range(self.n_qubits):
                if state[q, :, i].sum().item() > 1e-7:
                    q2 = int(torch.where(state[q, :, i] > 0)[0].item())
                    if q not in visited_qubits and q2 not in visited_qubits:
                        frontlayer_qubits.add(q)
                        frontlayer_qubits.add(q2)
                        frontlayer_gates.add((q, q2))
                    visited_qubits.add(q)
                    visited_qubits.add(q2)
                if state[:, q, i].sum().item() > 1e-7:
                    q2 = int(torch.where(state[:, q, i] > 0)[0].item())
                    if q not in visited_qubits and q2 not in visited_qubits:
                        frontlayer_qubits.add(q)
                        frontlayer_qubits.add(q2)
                        frontlayer_gates.add((q, q2))
                    visited_qubits.add(q)
                    visited_qubits.add(q2)

            if len(visited_qubits) >= self.n_qubits - 1:
                break

        return frontlayer_qubits, frontlayer_gates

    def get_random_states_in_range(
        self, batch_size: int, min_difficulty: int, max_difficulty: int
    ):
        states = torch.zeros((batch_size, self.n_qubits, self.n_qubits, self.horizon))
        for i in range(batch_size):
            n_gates = random.randint(min_difficulty, max_difficulty)
            states[i] = self.get_random_state(n_gates)

        return states

    def get_random_state(self, difficulty: int):
        flag = False
        while not flag:
            qc = CNOTCircuit(self.n_qubits)
            for i in range(difficulty):
                q1, q2 = random.sample(range(self.n_qubits), 2)
                while q1 == q2:
                    q2 = random.choice(range(self.n_qubits))
                if ((q1, q2) not in self.topology) and ((q2, q1) not in self.topology):
                    flag = True
                qc.add_cnot(q1, q2)

        # pyrefly: ignore[unbound-name], qc will always be initialized
        state = qc.to_tensor(horizon=self.horizon)

        return state

    def batch_states(self, states: Batchable[torch.Tensor]) -> torch.Tensor:
        if type(states) is torch.Tensor:
            return states

        return torch.stack([state for state in states])

    def state_from(self, circuit: QuantumCircuit) -> torch.Tensor:
        # todo: This is shit
        cnot_circuit = CNOTCircuit(circuit.num_qubits)
        for gate in circuit.data:
            if gate.operation.num_qubits == 2:
                q1, q2 = gate.qubits
                cnot_circuit.add_cnot(q1._index, q2._index)

        return cnot_circuit.to_tensor(horizon=self.horizon)



if __name__ == "__main__":
    n_qubits = 6
    horizon = 10

    circuit = CNOTCircuit(n_qubits)
    circuit.add_cnot(0, 2)
    circuit.add_cnot(0, 1)
    circuit.add_cnot(0, 3)
    print(circuit)
    topology = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

    game = TensorStateHandler(n_qubits, horizon, topology)

    root_state = circuit.to_tensor(horizon=horizon)

    next_state = game.get_next_state(root_state, 0)
    print(CNOTCircuit.from_tensor(next_state))

    print(next_state.shape)
    print(game.get_action_cost(next_state, 0))
    print(game.get_action_cost(next_state, 1))
    print(game.get_possible_actions(next_state))
    print(game.is_terminal(next_state))
