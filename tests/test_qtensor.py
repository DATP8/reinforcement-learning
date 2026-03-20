from qiskit import QuantumCircuit
from qtensor import Qtensor
import unittest
import random
import torch


class TestQtensor(unittest.TestCase):
    n_qubits = 10
    horizon = 100
    functions_single = [
        torch.sum,
        torch.add,
        torch.mul,
        torch.max,
        torch.min,
        torch.floor,
    ]
    functions_binary = [torch.mul, torch.add, torch.equal]
    functions_list = [torch.cat, torch.stack]
    args_single = [[], [1], [3], [], [], []]
    args_binary = [[], [], []]
    args_list = [[], [1]]
    kwargs_single = [{}, {}, {}, {}, {}, {}]
    kwargs_binary = [{}, {}, {}]
    kwargs_list = [{"dim": 1}, {}]

    def single_function(self, tensor, qtensor):
        for function, args, kwargs in zip(
            self.functions_single, self.args_single, self.kwargs_single
        ):
            # pyrefly: ignore[bad-argument-type, no-matching-overload]
            tensor_result = function(tensor, *args, **kwargs)
            # pyrefly: ignore[bad-argument-type, no-matching-overload]
            qtensor_result = function(qtensor, *args, **kwargs)
            self.assertTrue(torch.equal(tensor_result, qtensor_result))

    def binary_function(self, tensor, qtensor):
        for function, args, kwargs in zip(
            self.functions_binary, self.args_binary, self.kwargs_binary
        ):
            # pyrefly: ignore[bad-argument-type, no-matching-overload]
            tensor_result = function(tensor, tensor, *args, **kwargs)
            # pyrefly: ignore[bad-argument-type, no-matching-overload]
            qtensor_result = function(qtensor, tensor, *args, **kwargs)
            if isinstance(tensor_result, torch.Tensor):
                # pyrefly: ignore[bad-argument-type]
                self.assertTrue(torch.equal(tensor_result, qtensor_result))
            else:
                self.assertEqual(tensor_result, qtensor_result)

    def list_function(self, tensor, qtensor):
        for function, args, kwargs in zip(
            self.functions_list, self.args_list, self.kwargs_list
        ):
            # pyrefly: ignore[bad-argument-type, no-matching-overload]
            tensor_result = function((tensor, tensor), *args, **kwargs)
            # pyrefly: ignore[bad-argument-type, no-matching-overload]
            qtensor_result = function((qtensor, tensor), *args, **kwargs)
            self.assertTrue(torch.equal(tensor_result, qtensor_result))

    def test_from_circuit(self):
        for _ in range(1000):
            tensor = torch.zeros((self.n_qubits, self.horizon), dtype=torch.float32)
            circuit = QuantumCircuit(self.n_qubits)
            for i in range(10):
                q1, q2 = random.sample(range(self.n_qubits), 2)
                while q1 == q2:
                    q2 = random.choice(range(self.n_qubits))
                circuit.cx(q1, q2)
                tensor[q1, i] = 1
                tensor[q2, i] = 1
            circuit_tensor = Qtensor.from_circuit(circuit, self.horizon)
            self.assertTrue(torch.equal(circuit_tensor._t, tensor))

    def test_torch_functions(self):
        for _ in range(1000):
            tensor = torch.zeros((self.n_qubits, self.horizon), dtype=torch.float32)
            for j in range(10):
                q1, q2 = random.sample(range(self.n_qubits), 2)
                while q1 == q2:
                    q2 = random.choice(range(self.n_qubits))
                tensor[q1, j] = 1
                tensor[q2, j] = 1
            qtensor = Qtensor(tensor)
            self.single_function(tensor, qtensor)
            self.binary_function(tensor, qtensor)
            self.list_function(tensor, qtensor)
