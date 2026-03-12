import torch
from cnot_circuit import CNOTCircuit
from tensor_state_handler import TensorStateHandler
from qiskit import QuantumCircuit
from qtensor import Qtensor
from qtensor_state_handler import QtensorStateHandler
import unittest

n_qubits = 6
horizon = 10
topology = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
input_circuits = [
    [(1, 2), (0, 1), (3, 5), (0, 1), (1, 2)],
    [(0, 2), (0, 1), (0, 2)],
    [(0, 1), (4, 5), (2, 3), (3, 4), (1, 2)],
    [(0, 1), (4, 5), (2, 3), (3, 4), (1, 2), (3, 5), (2, 3), (0, 1), (1, 2), (2, 3)],
]
output_circuits_pruned = [
    [(3, 5)],
    [(0, 2), (0, 1), (0, 2)],
    [],
    [(3, 5), (2, 3), (1, 2), (2, 3)],
]

output_circuits_actions = [
    [(4, 5)],
    [(1, 2), (1, 0), (1, 2)],
    [],
    [(3, 4), (2, 3), (1, 2), (2, 3)],
]

actions = [3, 0, 1, 4]


class TestTensorStateHandler(unittest.TestCase):
    game = TensorStateHandler(n_qubits, horizon, topology)

    def test_tensor_prune(self):
        circuits = []
        for gate_list in input_circuits:
            circuit = CNOTCircuit(n_qubits)
            for q1, q2 in gate_list:
                circuit.add_cnot(q1, q2)
            circuits.append(circuit.to_tensor(horizon))
        for i, state in enumerate(circuits):
            pruned_state, _ = self.game.prune(state)
            pruned_circuit = CNOTCircuit(n_qubits)
            for q1, q2 in output_circuits_pruned[i]:
                pruned_circuit.add_cnot(q1, q2)
            # self.assertTrue(torch.equal(pruned_circuit.to_tensor(horizon), pruned_state))

    def test_tensor_generate_random_circuit(self):
        for _ in range(1000):
            state = self.game.get_random_state(10)
            pruned_state, _ = self.game.prune(state.clone())
            self.assertFalse(self.game.is_terminal(pruned_state))

    def test_tensor_get_next_state(self):
        states = []
        for gate_list in input_circuits:
            circuit = CNOTCircuit(n_qubits)
            for q1, q2 in gate_list:
                circuit.add_cnot(q1, q2)
            states.append(circuit.to_tensor(horizon))
        for i, state in enumerate(states):
            # next_state= self.game.get_next_state(state, actions[i])
            pruned_circuit = CNOTCircuit(n_qubits)
            for q1, q2 in output_circuits_pruned[i]:
                pruned_circuit.add_cnot(q1, q2)
            # self.assertTrue(torch.equal(pruned_circuit.to_tensor(horizon), next_state))


class TestQtensorStateHandler(unittest.TestCase):
    game = QtensorStateHandler(n_qubits, horizon, topology)

    def test_qtensor_prune(self):
        circuits = []
        for gate_list in input_circuits:
            circuit = QuantumCircuit(n_qubits)
            for q1, q2 in gate_list:
                circuit.cx(q1, q2)
            circuits.append(Qtensor.from_circuit(circuit, horizon))
        for i, state in enumerate(circuits):
            pruned_state, _ = self.game.prune(state)
            pruned_circuit = QuantumCircuit(n_qubits)
            for q1, q2 in output_circuits_pruned[i]:
                pruned_circuit.cx(q1, q2)
            self.assertTrue(
                torch.equal(
                    Qtensor.from_circuit(pruned_circuit, horizon)._t, pruned_state._t
                )
            )

    def test_qtensor_generate_random_circuit(self):
        for _ in range(1000):
            state = self.game.get_random_state(10)
            pruned_state, _ = self.game.prune(state)
            if self.game.is_terminal(pruned_state):
                print(state)
            self.assertFalse(self.game.is_terminal(pruned_state))

    def test_qtensor_get_next_state(self):
        states = []
        for gate_list in input_circuits:
            circuit = QuantumCircuit(n_qubits)
            for q1, q2 in gate_list:
                circuit.cx(q1, q2)
            states.append(Qtensor.from_circuit(circuit, horizon))
        for i, state in enumerate(states):
            next_state = self.game.get_next_state(state, actions[i])
            pruned_circuit = QuantumCircuit(n_qubits)
            for q1, q2 in output_circuits_pruned[i]:
                pruned_circuit.cx(q1, q2)
            self.assertTrue(
                torch.equal(
                    Qtensor.from_circuit(pruned_circuit, horizon)._t, next_state._t
                )
            )


if __name__ == "__main__":
    unittest.main()
