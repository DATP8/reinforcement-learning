import unittest

from qiskit import QuantumCircuit

from routing.swap_inserter.swap_inserter import SwapInserter
from states.qtensor_state_handler import Qtensor, QtensorStateHandler

input_circuits = [
    [(1, 2), (0, 1), (3, 5), (0, 1), (1, 2)],
    [(0, 2), (0, 1), (0, 2)],
    [(0, 1), (4, 5), (2, 3), (3, 4), (1, 2)],
    [(0, 1), (4, 5), (2, 3), (3, 4), (1, 2), (3, 5), (2, 3), (0, 1), (1, 2), (2, 3)],
    [(0, 1), (4, 5), (2, 3), (3, 4), (1, 2), (3, 5), (2, 3), (0, 1), (1, 2), (2, 3)],
    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (3, 4), (2, 3), (1, 2), (0, 1)],
]

actions = [
    [4],
    [1],
    [],
    [3, 2, 1],
    [4],
    [],
]


class TestSwapInserter(unittest.TestCase):
    n_qubits = 6
    topology = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

    def test_build_circuit_from_solution(self):
        state_handler = QtensorStateHandler(self.n_qubits, 100, self.topology)
        swap_inserter = SwapInserter(self.topology, self.n_qubits)
        for gate_list, action_set in zip(input_circuits, actions):
            circuit = QuantumCircuit(self.n_qubits)
            for q1, q2 in gate_list:
                circuit.cx(q1, q2)
            routed_circuit, _, _ = swap_inserter.build_circuit_from_solution(
                action_set, circuit
            )
            circuit_tensor = Qtensor.from_circuit(routed_circuit, 100)
            self.assertTrue(state_handler.is_terminal(circuit_tensor))
