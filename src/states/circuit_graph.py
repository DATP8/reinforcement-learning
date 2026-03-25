import hashlib
from qiskit.circuit.quantumcircuit import QuantumCircuit
from torch_geometric.data import Data
import torch
import torch.nn.functional as F


class CircuitGraph(Data):
    def __init__(self, **args):
        super().__init__(**args)

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
    def from_circuit(cls, qc: QuantumCircuit, horizon: int = 0):
        n_qubits = qc.num_qubits
        two_qubit_gates = []
        for gate in qc.data:
            if gate.operation.num_qubits == 2:
                q1, q2 = gate.qubits
                two_qubit_gates.append((q1._index, q2._index))

        x = torch.zeros((len(two_qubit_gates) + 1, n_qubits * 2), dtype=torch.float)
        edge_index = []
        edge_attr = []

        qubit_deps = {}

        for succ, (q1, q2) in enumerate(two_qubit_gates):
            prev_gate_1 = qubit_deps[q1] if q1 in qubit_deps else None
            prew_gate_2 = qubit_deps[q2] if q2 in qubit_deps else None
            qubit_deps[q1] = succ
            qubit_deps[q2] = succ
            x[succ, q1] = 1.0
            x[succ, n_qubits + q2] = 1.0
            if prev_gate_1 is not None:
                # todo: Make sure this is actualy correct
                # Add edge to previous gate that acted on q1 so information flows to front_layer
                edge_index.append((succ, prev_gate_1))
                edge_attr.append(F.one_hot(torch.tensor(q1), num_classes=n_qubits))

            if prew_gate_2 is not None:
                # Add edge to previous gate that acted on q2 so information flows to front_layer
                edge_index.append((succ, prew_gate_2))
                edge_attr.append(F.one_hot(torch.tensor(q2), num_classes=n_qubits))

        # Global node that connects to all gates in the first layer
        # todo: currently global node is zerro
        for i in range(len(two_qubit_gates)):
            edge_index.append((len(two_qubit_gates), i))
            edge_index.append((i, len(two_qubit_gates)))
            edge_attr.append(torch.zeros(n_qubits, dtype=torch.float))
            edge_attr.append(torch.zeros(n_qubits, dtype=torch.float))

        edge_index = torch.tensor(edge_index).t()
        edge_attr = (
            torch.stack(edge_attr)
            if len(edge_attr) > 0
            else torch.empty((0, n_qubits), dtype=torch.float)
        )

        return cls(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def to_circuit(self):
        raise NotImplementedError


if __name__ == "__main__":
    qs = QuantumCircuit(6)

    qs.cx(0, 1)
    qs.cx(1, 3)
    qs.cx(4, 2)

    print(qs)

    graph = CircuitGraph.from_circuit(qs)

    print(graph.x)
    print(graph.edge_index)
    print(graph.edge_attr)
