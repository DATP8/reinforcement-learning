from src.states.state_handler import Batchable, StateHandler #pyrefly: ignore
from src.states.qtensor import Qtensor #pyrefly: ignore

import random
import torch

from cachetools import LFUCache
from qiskit import QuantumCircuit


class QtensorStateHandler(StateHandler[Qtensor]):
    def __init__(self, n_qubits: int, horizon: int, topology: list[tuple[int, int]]):
        self.n_qubits = n_qubits
        self.horizon = horizon
        self.topology = topology
        self.mask = torch.zeros((self.n_qubits), dtype=torch.float32)
        self.next_state_cache = LFUCache[tuple[int, int], Qtensor](maxsize=10000)
        self.is_terminal_cache = LFUCache[int, bool](maxsize=10000)
        self.action_cost_cache = LFUCache[tuple[int, int], float](maxsize=10000)

    def gate_to_tuple(self, tensor: Qtensor):
        q1 = -1
        q2 = -1
        for i in range(self.n_qubits):
            if tensor[i].item() == 1:
                if q1 == -1:
                    q1 = i
                else:
                    q2 = i
        return q1, q2

    def get_possible_actions(self, state: Qtensor) -> list[int]:
        return list(range(len(self.topology)))

    def prune(self, state: Qtensor) -> tuple[Qtensor, int]:
        new_state = state.clone()
        # pyrefly: ignore[no-matching-overload]
        removed_gates = torch.ones((self.horizon), dtype=bool)
        front_layer = set()
        layers_removed = 0
        for i in range(state.gates):
            q1, q2 = self.gate_to_tuple(new_state[:, i])
            if (
                ((q1, q2) in self.topology or (q2, q1) in self.topology)
                and q1 not in front_layer
                and q2 not in front_layer
            ):
                new_state[:, i] = self.mask
                removed_gates[i] = False
                layers_removed += 1
            else:
                front_layer.add(q1)
                front_layer.add(q2)
            if len(front_layer) >= self.n_qubits - 1:
                break

        # pyrefly:ignore [no-matching-overload]
        new_state = torch.cat(
            (
                new_state[:, removed_gates],
                torch.zeros((self.n_qubits, layers_removed), dtype=torch.float32),
            ),
            dim=1,
        )
        new_state.gates -= layers_removed
        return new_state, layers_removed

    def get_next_state(self, state: Qtensor, action: int) -> Qtensor:
        state_hash = hash(state)
        if (state_hash, action) in self.next_state_cache:
            return self.next_state_cache[(state_hash, action)]

        # If the first layer is now empty, shift all layers by one
        new_state, layers_removed = self.prune(state)
        q1, q2 = self.topology[action]

        # Swap the qubits in the tensor representation
        new_state_temp = new_state.clone()
        new_state[q1] = new_state_temp[q2]._t
        new_state[q2] = new_state_temp[q1]._t

        self.next_state_cache[(state_hash, action)] = new_state
        return new_state

    def is_terminal(self, state: Qtensor) -> bool:
        state_hash = hash(state)
        if state_hash in self.is_terminal_cache:
            return self.is_terminal_cache[state_hash]
        next_state, _ = self.prune(state)
        # pyrefly: ignore[no-matching-overload]
        is_terminal = torch.sum(next_state).item() <= 1e-7
        self.is_terminal_cache[state_hash] = is_terminal
        return is_terminal

    def get_action_cost(self, state: Qtensor, action: int) -> float:
        # todo: this is incomplete. Should have cost 0.5 if cnot reduction is possible.
        state_hash = hash(state)
        if (state_hash, action) in self.action_cost_cache:
            return self.action_cost_cache[(state_hash, action)]
        cost = 1.0
        self.action_cost_cache[(state_hash, action)] = cost

        return cost

    def get_random_states(
        self, batch_size: int, max_difficulty: int
    ) -> Batchable[Qtensor]:
        batch = []
        for _ in range(batch_size):
            difficulty = random.randint(1, max_difficulty)
            batch.append(self.get_random_state(difficulty))
        return batch

    def get_random_state(self, difficulty: int):
        flag = False
        while not flag:
            qc = QuantumCircuit(self.n_qubits)
            for _ in range(difficulty):
                q1, q2 = random.sample(range(self.n_qubits), 2)
                while q1 == q2:
                    q2 = random.choice(range(self.n_qubits))
                if ((q1, q2) not in self.topology) and ((q2, q1) not in self.topology):
                    flag = True

                qc.cx(q1, q2)

            state = Qtensor.from_circuit(qc, self.horizon)
        # pyrefly: ignore[unbound-name]
        return state

    def batch_states(self, states: Batchable[Qtensor]) -> Qtensor:
        # pyrefly: ignore[bad-argument-type]
        return Qtensor(torch.stack(states))

    def state_from(self, circuit: QuantumCircuit) -> Qtensor:
        return Qtensor.from_circuit(circuit, self.horizon)

    def get_num_qubits(self) -> int:
        return self.n_qubits

    def get_topology(self):
        return self.topology


if __name__ == "__main__":
    n_qubits = 6
    horizon = 10
    circuit = QuantumCircuit(n_qubits)
    circuit.cx(1, 2)
    circuit.cx(0, 1)
    circuit.cx(0, 3)
    circuit.cx(0, 1)
    circuit.cx(3, 5)
    circuit.cx(0, 4)
    circuit.cx(0, 1)
    print(circuit)
    topology = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

    game = QtensorStateHandler(n_qubits, horizon, topology)

    root_state = Qtensor.from_circuit(circuit, horizon)
    # print(root_state)
    next_state, gates_removed = game.prune(root_state)
    print(next_state)
    # print(CNOTCircuitSmall.from_tensor(next_state))
    # print(gates_removed)
    next_state = game.get_next_state(root_state, 0)
    print(next_state)
    # print(next_state.shape)
    print(game.get_action_cost(next_state, 0))
    print(game.get_action_cost(next_state, 1))
    print(game.get_possible_actions(next_state))
    print(game.is_terminal(next_state))
