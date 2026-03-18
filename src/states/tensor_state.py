from qiskit.converters import circuit_to_dag
from qiskit import QuantumCircuit
from collections import defaultdict

from .state_handler import StateHandler

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
                    f"Reached depth {layer} which exceeds circuit depth {depth()} without hitting specified horizon {horizon}."
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

    def _add_gate(self, dag, g, new_qc, inverse):
        new_qargs = []
        for q in g.qargs:
            qubit_idx = dag.find_bit(q).index
            new_idx = inverse[qubit_idx]
            new_qargs.append(new_qc.qubits[new_idx])

        new_qc.append(g.op, new_qargs)
        return new_qc

    def _make_topological_connection_list(self, num_qubits, topology):
        topological_connection_list = [[] for _ in range(num_qubits)]

        for pair in topology:
            q1, q2 = pair
            topological_connection_list[q1].append(q2)
            topological_connection_list[q2].append(q1)

        return topological_connection_list

    def _prune(self, qc, dag, inverse, new_qc, topological_connection_list):
        wait_list = [[] for _ in range(qc.num_qubits)]

        for q in range(qc.num_qubits):
            for g in dag.nodes_on_wire(qc.qubits[q], only_ops=True):
                if g.op.num_qubits == 1:
                    new_qc = self._add_gate(dag, g, new_qc, inverse)
                    dag.remove_op_node(g)
                    continue

                q1 = dag.find_bit(g.qargs[0]).index
                q2 = dag.find_bit(g.qargs[1]).index

                q_is_q1 = q1 == q
                if not q_is_q1:
                    q2 = q1
                    q1 = q

                q1 = inverse[q1]
                q2 = inverse[q2]

                if wait_list[q2] and wait_list[q2][0] == q1:
                    wait_list[q2].pop(0)
                    new_qc = self._add_gate(dag, g, new_qc, inverse)
                    dag.remove_op_node(g)
                    self._prune(qc, dag, inverse, new_qc, topological_connection_list)

                elif any(x == q1 for x in topological_connection_list[q2]):
                    wait_list[q1].append(q2)

                break

        return new_qc, dag

    def insert_swaps(
        self, qc: QuantumCircuit, path: list, topology: list, game: StateHandler
    ):
        dag = circuit_to_dag(qc)
        new_qc = QuantumCircuit(qc.num_qubits)

        topological_connection_list = self._make_topological_connection_list(
            qc.num_qubits, topology
        )

        mapping = list(range(qc.num_qubits))
        inverse = list(range(qc.num_qubits))

        while dag.topological_op_nodes():
            new_qc, dag = self._prune(
                qc, dag, inverse, new_qc, topological_connection_list
            )
            if not path:
                break
            q1, q2 = topology[path.pop(0)]
            new_qc.swap(q1, q2)
            v1, v2 = mapping[q1], mapping[q2]
            mapping[q1] = v2
            mapping[q2] = v1
            inverse[v1] = q2
            inverse[v2] = q1

        return new_qc, inverse

    def insert_swaps_old(
        self, qc: QuantumCircuit, path: list, topology: list, game: StateHandler
    ):
        if not path:
            return qc

        state, layers_removed = game.prune(self)
        depth_list = []

        for action in path:
            state, temp_layers_removed = game.prune(state)
            state = game.get_next_state(state, action)
            layers_removed += temp_layers_removed
            depth_list.append(layers_removed)

        org_dag = circuit_to_dag(qc)
        new_qc = QuantumCircuit(qc.num_qubits)

        mapping = list(range(qc.num_qubits))
        inverse = list(range(qc.num_qubits))

        swap_list = []

        two_gate_count = 0
        for i, layer in enumerate(org_dag.layers()):
            while depth_list and i >= depth_list[0]:
                swap_list.append(topology[path.pop(0)])
                depth_list.pop(0)

            for org_g in layer["graph"].topological_op_nodes():
                if org_g.op.num_qubits == 2:
                    two_gate_count += 1
                    for swap in swap_list:
                        q1, q2 = swap
                        new_qc.swap(q1, q2)

                    for swap in swap_list:
                        q1, q2 = swap
                        v1, v2 = mapping[q1], mapping[q2]
                        mapping[q1] = v2
                        mapping[q2] = v1
                        inverse[v1] = q2
                        inverse[v2] = q1
                    swap_list = []

                new_qc = self._add_gate(org_dag, org_g, new_qc, inverse)

        return new_qc


if __name__ == "__main__":
    import random

    import time

    from .model import ValueModel
    from .tensor_state_handler import TensorStateHandler
    from .batch_weighted_astar_search import BWAS

    random.seed(42)

    qc = QuantumCircuit(6)
    qc.cx(2, 5)
    qc.cx(4, 2)
    qc.cx(5, 1)
    qc.cx(1, 0)
    qc.cx(2, 3)

    n_qubits = 6
    horizon = 100
    topology = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

    game = TensorStateHandler(n_qubits, horizon, topology)
    model = ValueModel(n_qubits, horizon, len(topology))
    model.load_state_dict(
        torch.load(
            "/home/vind/code/P8/project/reinforcement-learning/models/difficulty17_iteration95270.pt",
            map_location="cpu",
        )
    )
    model.to("cpu")
    bwas = BWAS(model, game, batch_size=1)

    print(qc)

    state = TensorState.from_quantum_circuit(qc, horizon=horizon)
    root_state, _ = game.prune(state)

    start_time = time.time()
    path = bwas.search(root_state)
    end_time = time.time()

    state = TensorState(state)

    new_qc, _ = state.as_subclass(TensorState).insert_swaps(qc, path, topology, game)
    print(new_qc)
