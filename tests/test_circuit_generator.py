import unittest

import numpy as np

from src.circuit_generator import CircuitGenerator


class TestCircuitGenerator(unittest.TestCase):
    _MAX_QUBITS = 100
    _MAX_GATES = 100
    _NUM_TESTS = 100

    def test_generate_n_random_circuits(self):
        seed = np.random.randint(0, np.iinfo(int).max)
        print(f"Seed is: {seed}")
        rng = np.random.default_rng(seed)

        num_gates = rng.choice(range(self._MAX_GATES))
        gateset = set()
        num_qubits = 0
        while (
            len(gateset) == 0 and num_gates != 0
        ):  # Reroll until we get valid combination
            num_qubits = rng.choice(range(self._MAX_QUBITS))
            gateset = self._make_random_gateset(rng, num_qubits)

        rqcs = CircuitGenerator.generate_n_random_circuits(
            self._NUM_TESTS, num_qubits, num_gates, gateset, seed
        )
        for rqc in rqcs:
            self.assertEqual(num_qubits, rqc.num_qubits)

            gate_dict = rqc.count_ops()

            rqc_gate_count: int = sum(gate_dict.values())

            if num_qubits == 0:
                self.assertEqual(0, rqc_gate_count)
            else:
                self.assertEqual(num_gates, rqc_gate_count)

            rqc_gates = set(gate_dict.keys())
            self.assertLessEqual(rqc_gates, gateset)

    def _make_random_gateset(
        self, rng: np.random.Generator, num_qubits: int
    ) -> set[str]:
        if num_qubits == 0:
            return set()

        valid_gate_names = [
            name
            for name, gate_info in CircuitGenerator._GATE_MAP.items()
            if gate_info[1] <= num_qubits
        ]
        num_gate_types = rng.choice(range(len(valid_gate_names)))
        selected_gates = rng.choice(
            valid_gate_names, size=num_gate_types, replace=False
        )
        return set(selected_gates)
