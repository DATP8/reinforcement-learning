import hashlib
from qiskit.circuit.quantumcircuit import QuantumCircuit
from torch_geometric.data import Data
import torch
import torch.nn.functional as F


class DenseCircuitGraph(Data):
    def __init__(self, **args):
        super().__init__(**args)
        self.positional_encoding = None

    def __hash__(self):
        assert (
            self.x is not None
            and self.edge_index is not None
            and self.edge_attr is not None
        ), "State must have x, edge_index, and edge_attr defined"
        return (
            self.tensor_hash(self.x)
            ^ self.tensor_hash(self.edge_index)
            ^ self.tensor_hash(self.edge_attr)
        )

    @staticmethod
    def tensor_hash(t: torch.Tensor) -> int:
        return hash(hashlib.blake2b(t.numpy().tobytes(), digest_size=8).digest())

    @classmethod
    def from_circuit(cls, qc: QuantumCircuit):
        n_qubits = qc.num_qubits
        two_qubit_gates = []
        for gate in qc.data:
            if gate.operation.num_qubits == 2:
                q1, q2 = gate.qubits
                two_qubit_gates.append((q1._index, q2._index))

        x = torch.zeros((len(two_qubit_gates) + 1, n_qubits * 2), dtype=torch.float)
        edge_index = []
        edge_attr = []

        qubit_deps = {i: [] for i in range(n_qubits)}
        for gate, (q1, q2) in enumerate(two_qubit_gates):
            qubit_deps[q1].append(gate)
            qubit_deps[q2].append(gate)
            x[gate, q1] = 1.0
            x[gate, n_qubits + q2] = 1.0

            for i, prev_gate in enumerate(qubit_deps[q1][:-1]):
                edge_index.append((gate, prev_gate))
                dist = len(qubit_deps[q1]) - i - 1
                normalised_dist = 1.0 / dist
                edge_feature = torch.cat(
                    [
                        F.one_hot(torch.tensor(q1), num_classes=n_qubits),
                        torch.tensor([normalised_dist], dtype=torch.float),
                    ]
                )
                edge_attr.append(edge_feature)

            for i, prev_gate in enumerate(qubit_deps[q2][:-1]):
                edge_index.append((gate, prev_gate))
                dist = len(qubit_deps[q2]) - i - 1
                normalised_dist = 1.0 / dist
                edge_feature = torch.cat(
                    [
                        F.one_hot(torch.tensor(q2), num_classes=n_qubits),
                        torch.tensor([normalised_dist], dtype=torch.float),
                    ]
                )
                edge_attr.append(edge_feature)

        edge_index.append((len(two_qubit_gates), len(two_qubit_gates))) # add dummpy self loop for global node
        edge_attr.append(torch.zeros(n_qubits + 1, dtype=torch.float)) # add dummpy edge attr for global node self loop
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attr)

        return cls(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def to_circuit(self):
        raise NotImplementedError


if __name__ == "__main__":
    circuit = QuantumCircuit(6)

    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.cx(3, 5)

    print(circuit)

    graph = DenseCircuitGraph.from_circuit(circuit)

    print(graph.x)
    print(graph.edge_index)
    print(graph.edge_attr)
