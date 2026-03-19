import unittest
from circuit_generator import CircuitGenerator

NUM_QUBITS = 4
NUM_GATES = 10
GATESET = ["cx", "h", "s"]


class TestCircuitGenerator(unittest.TestCase):
    def test_generate_random_circuit(self):
        rand_circuit = CircuitGenerator.generate_random_circuit(
            NUM_QUBITS, NUM_GATES, ["cx", "h", "s"]
        )
        self.assertEqual(NUM_QUBITS, rand_circuit.num_qubits)
        # TODO: add randomized gateset test
