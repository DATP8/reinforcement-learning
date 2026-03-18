from qiskit.converters import dag_to_circuit
from qiskit.circuit.library import SwapGate
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.quantumcircuit import QuantumCircuit
from .state_handler import StateHandler
from .circuit_graph import CircuitGraph
from collections import defaultdict
import itertools
import torch
import random

class CNOTCircuit(QuantumCircuit):
    def __init__(self, n_qubits: int):
        super().__init__(n_qubits)
        self.layers = defaultdict(list)

    def add_cnot(self, q1, q2):
        if q1 == q2:
            raise ValueError("Control and target qubits must be different.")

        self.cx(q1, q2)
        layer = max(self.qubit_layers[q1], self.qubit_layers[q2]) + 1
        self.qubit_layers[q1] = layer
        self.qubit_layers[q2] = layer
        self.layers[layer].append((q1, q2))

    def to_tensor(self, horizon=None):
        depth = self.depth() if horizon is None else horizon
        tensor = torch.zeros(
            (self.num_qubits, self.num_qubits, depth), dtype=torch.float32
        )

        layer = 0
        for layer in sorted(self.layers.keys()):
            for q1, q2 in self.layers[layer]:
                if layer < depth:
                    tensor[q1, q2, layer] = 1.0
            layer += 1
            if layer > depth:
                assert horizon is not None, (
                    f"Reached depth {layer} which exceeds circuit depth {self.depth()} without hitting specified horizon {horizon}."
                )
                break

        if horizon is None:
            tensor = tensor[:, :, :layer]

        return tensor

    @staticmethod
    def from_tensor(tensor: torch.Tensor):
        assert tensor.dim() == 3, (
            "Input tensor must be 3-dimensional (n_qubits, n_qubits, depth)."
        )

        n_qubits, n_qubits2, depth = tensor.size()
        assert n_qubits == n_qubits2, (
            f"The first two dimensions of the tensor must be equal (square). Got {n_qubits} and {n_qubits2}."
        )

        qc = CNOTCircuit(n_qubits)

        for i in range(depth):
            lefts, rights = torch.where(tensor[:, :, i] == 1.0)
            pairs = list(zip(lefts.tolist(), rights.tolist()))
            for q1, q2 in pairs:
                qc.add_cnot(q1, q2)

        return qc

    def from_circuit_graph(dag: CircuitGraph):
        if dag.x is None:
            return None

        circuit = QuantumCircuit(dag.x.shape[1] // 2)
        for x in dag.x[:-1]:  # Exclude global node
            q1 = torch.where(x[: dag.x.shape[1] // 2] > 0)[0].item()
            q2 = torch.where(x[dag.x.shape[1] // 2 :] > 0)[0].item()
            circuit.cx(q1, q2)

        return circuit


def generate_random_circuit(n_qubits: int, n_gates: int):
    qc = CNOTCircuit(n_qubits)

    for i in range(n_gates):
        q1, q2 = random.sample(range(n_qubits), 2)
        while q1 == q2:
            q2 = random.choice(range(n_qubits))

        qc.add_cnot(q1, q2)
    return qc


if __name__ == "__main__":
    circuit = generate_random_circuit(n_qubits=10, n_gates=20)

    print(circuit)
    print(circuit.depth())

    tensor = circuit.to_tensor()
    # print(tensor)
    print(tensor.size())

    new_circuit = CNOTCircuit.from_tensor(tensor)
    print(new_circuit)
    print(new_circuit.depth())

    assert circuit == new_circuit, (
        "The original and reconstructed circuits do not match!"
    )
    assert torch.equal(circuit.to_tensor(), new_circuit.to_tensor()), (
        "The original and reconstructed circuits do not match!"
    )
