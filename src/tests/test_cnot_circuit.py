import unittest
from cnot_circuit import CNOTCircuit
import random

n_qubits = 10
horizon = 100

class TestCNOTCircuit(unittest.TestCase):
    def test_to_and_from_tensor(self):
        for _ in range(1000):
            circuit = CNOTCircuit(n_qubits)
            for i in range(10):
                q1, q2 = random.sample(range(n_qubits), 2)
                while q1 == q2:
                    q2 = random.choice(range(n_qubits))
                circuit.add_cnot(q1, q2)
            circuit_tensor = circuit.to_tensor(horizon)
            circuit_new = CNOTCircuit.from_tensor(circuit_tensor)
            self.assertEqual(circuit, circuit_new)