import unittest
import torch

from qiskit import QuantumCircuit

from states.tensor_state_handler import TensorStateHandler
from states.qtensor import Qtensor
from states.qtensor_state_handler import QtensorStateHandler

n_qubits = 6
horizon = 10
topology = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
input_circuits = [
    [(1, 2), (0, 1), (3, 5), (0, 1), (1, 2)],
    [(0, 2), (0, 1), (0, 2)],
    [(0, 1), (4, 5), (2, 3), (3, 4), (1, 2)],
    [(0, 1), (4, 5), (2, 3), (3, 4), (1, 2), (3, 5), (2, 3), (0, 1), (1, 2), (2, 3)],
    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (3, 4), (2, 3), (1, 2), (0, 1)],
]
output_circuits_pruned = [
    [(3, 5)],
    [(0, 2), (0, 1), (0, 2)],
    [],
    [(3, 5), (2, 3), (1, 2), (2, 3)],
    [],
]

output_circuits_actions = [
    [(4, 5)],
    [(1, 2), (1, 0), (1, 2)],
    [],
    [(3, 4), (2, 3), (1, 2), (2, 3)],
    [],
]

actions = [3, 0, 1, 4, 0]
class TestTensorStateHandler(unittest.TestCase):
    game = TensorStateHandler(n_qubits, horizon, topology)

    def test_tensor_prune(self):
        circuits = []
        for gate_list in input_circuits:
            circuit = QuantumCircuit(n_qubits)
            for q1, q2 in gate_list:
                circuit.cx(q1, q2)
            circuits.append(self.game.state_from(circuit))
        for i, state in enumerate(circuits):
            pruned_state, _ = self.game.prune(state)
            pruned_circuit = QuantumCircuit(n_qubits)
            for q1, q2 in output_circuits_pruned[i]:
                pruned_circuit.cx(q1, q2)
           # self.assertTrue(torch.equal(self.game.state_from(circuit), pruned_state))

    def test_tensor_generate_random_circuit(self):
        for _ in range(1000):
            state = self.game.get_random_state(10)
            pruned_state, _ = self.game.prune(state.clone())
            self.assertFalse(self.game.is_terminal(pruned_state))

    def test_tensor_get_next_state(self):
        states = []
        for gate_list in input_circuits:
            circuit = QuantumCircuit(n_qubits)
            for q1, q2 in gate_list:
                circuit.cx(q1, q2)
            states.append(self.game.state_from(circuit))
        for i, state in enumerate(states):
            # next_state = self.game.get_next_state(state, actions[i])
            pruned_circuit = QuantumCircuit(n_qubits)
            for q1, q2 in output_circuits_actions[i]:
                pruned_circuit.cx(q1, q2)
            # self.assertTrue(torch.equal(pruned_circuit.to_tensor(horizon), next_state))

    def test_tensor_is_terminal(self):
        states = []
        for gate_list in input_circuits:
            circuit = QuantumCircuit(n_qubits)
            for q1, q2 in gate_list:
                circuit.cx(q1, q2)
            states.append(self.game.state_from(circuit))
        for i, state in enumerate(states):
            self.assertEqual(
                len(output_circuits_pruned[i]) == 0, self.game.is_terminal(state)
            )

class TestQtensorStateHandler(unittest.TestCase):
    game = QtensorStateHandler(n_qubits, horizon, topology)

    def test_Qtensor_prune(self):
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

    def test_Qtensor_generate_random_circuit(self):
        for _ in range(1000):
            state = self.game.get_random_state(10)
            pruned_state, _ = self.game.prune(state)
            if self.game.is_terminal(pruned_state):
                print(state)
            self.assertFalse(self.game.is_terminal(pruned_state))

    def test_Qtensor_get_next_state(self):
        states = []
        for gate_list in input_circuits:
            circuit = QuantumCircuit(n_qubits)
            for q1, q2 in gate_list:
                circuit.cx(q1, q2)
            states.append(Qtensor.from_circuit(circuit, horizon))
        for i, state in enumerate(states):
            next_state = self.game.get_next_state(state, actions[i])
            pruned_circuit = QuantumCircuit(n_qubits)
            for q1, q2 in output_circuits_actions[i]:
                pruned_circuit.cx(q1, q2)
            self.assertTrue(
                torch.equal(
                    Qtensor.from_circuit(pruned_circuit, horizon)._t, next_state._t
                )
            )

    def test_Qtensor_is_terminal(self):
        circuits = []
        for gate_list in input_circuits:
            circuit = QuantumCircuit(n_qubits)
            for q1, q2 in gate_list:
                circuit.cx(q1, q2)
            circuits.append(Qtensor.from_circuit(circuit, horizon))
        for i, state in enumerate(circuits):
            self.assertEqual(
                len(output_circuits_pruned[i]) == 0, self.game.is_terminal(state)
            )


if __name__ == "__main__":
    unittest.main()
