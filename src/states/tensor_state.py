from qiskit.converters import circuit_to_dag
from qiskit import QuantumCircuit
from collections import defaultdict


import torch


class TensorState(torch.Tensor):
    @staticmethod
    def from_quantum_circuit(qc: QuantumCircuit, horizon=None):
        depth = qc.depth() if horizon is None else horizon
        tensor = torch.zeros((qc.num_qubits, qc.num_qubits, depth), dtype=torch.float32)

        layers = defaultdict(list)
        qubit_layers = [-1 for _ in range(qc.num_qubits)]

        # idk was lazy
        dag = circuit_to_dag(qc)
        for g in dag.topological_op_nodes():
            if g.op.num_qubits == 2:
                q1, q2 = (dag.find_bit(q).index for q in g.qargs)
                layer = max(qubit_layers[q1], qubit_layers[q2]) + 1
                qubit_layers[q1] = layer
                qubit_layers[q2] = layer
                layers[layer].append((q1, q2))

        layer = 0
        for layer in sorted(layers.keys()):
            for q1, q2 in layers[layer]:
                if layer < depth:
                    tensor[q1, q2, layer] = 1.0
            layer += 1
            if layer > depth:
                assert horizon is not None, (
                    f"Reached depth {layer} which exceeds circuit depth {depth} without hitting specified horizon {horizon}."
                )
                break

        if horizon is None:
            tensor = tensor[:, :, :layer]

        return tensor

    def to_quantum_circuit(tensor: torch.Tensor):
        assert tensor.dim() == 3, (
            "Input tensor must be 3-dimensional (n_qubits, n_qubits, depth)."
        )

        n_qubits, n_qubits2, depth = tensor.size()
        assert n_qubits == n_qubits2, (
            f"The first two dimensions of the tensor must be equal (square). Got {n_qubits} and {n_qubits2}."
        )

        qc = QuantumCircuit(n_qubits)

        for i in range(depth):
            lefts, rights = torch.where(tensor[:, :, i] == 1.0)
            pairs = list(zip(lefts.tolist(), rights.tolist()))
            for q1, q2 in pairs:
                qc.cx(q1, q2)

        return qc

