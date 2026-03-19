from qiskit.transpiler import CouplingMap
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
                if (not q_is_q1):
                    q2 = q1
                    q1 = q

                q1 = inverse[q1]
                q2 = inverse[q2]

                if wait_list[q2] and wait_list[q2][0] == q1 :
                    wait_list[q2].pop(0)
                    new_qc = self._add_gate(dag, g, new_qc, inverse)
                    dag.remove_op_node(g)
                    return self._prune(qc, dag, inverse, new_qc, topological_connection_list)   

                elif any(x == q1 for x in topological_connection_list[q2]):
                    wait_list[q1].append(q2)
               
                break

        return new_qc, dag

    def insert_swaps(self, qc: QuantumCircuit, path: list, topology: list):
        dag = circuit_to_dag(qc)
        new_qc = QuantumCircuit(qc.num_qubits)
        
        topological_connection_list = self._make_topological_connection_list(qc.num_qubits, topology)

        mapping = list(range(qc.num_qubits))
        inverse = list(range(qc.num_qubits))
        
        while dag.topological_op_nodes():
            new_qc, dag = self._prune(qc, dag, inverse, new_qc, topological_connection_list)
            if (not path):
                break
            q1, q2 = topology[path.pop(0)] 
            new_qc.swap(q1, q2)  
            v1, v2 = mapping[q1], mapping[q2]
            mapping[q1] = v2 
            mapping[q2] = v1
            inverse[v1] = q2 
            inverse[v2] = q1
        return new_qc, inverse

    def insert_swaps2(
        self, actions: list[int], qc: QuantumCircuit, coupling_map: CouplingMap
    ) -> tuple[QuantumCircuit, list[int], list[int]] | None:
        """Reconstruct a routed circuit from the solution (coupling-map edge indices).

        The solution records which coupling-map edge was swapped at each step.
        This method replays those SWAPs, placing original gates whenever their
        qubits become adjacent, and returns the routed circuit plus layouts.

        Returns (routed_circuit, init_layout, final_layout) or None on failure.
        """
      
        num_qubits = qc.num_qubits
        from qiskit.transpiler import CouplingMap as CM

        dists = coupling_map.distance_matrix.astype(int)

        # Build gate list with virtual-qubit indices
        gates: list[tuple[list[int], object]] = []
        for inst in qc.data:
            qubits = [qc.find_bit(q).index for q in inst.qubits]
            gates.append((qubits, inst))

        # Per-qubit chains: ordered gate indices touching each virtual qubit
        qubit_chains: list[list[int]] = [[] for _ in range(num_qubits)]
        for gate_idx, (qubits, _) in enumerate(gates):
            for q in qubits:
                qubit_chains[q].append(gate_idx)

        # Predecessor count for each gate (dependency = same qubit, earlier gate)
        remaining = [0] * len(gates)
        for chain in qubit_chains:
            for i in range(1, len(chain)):
                remaining[chain[i]] += 1

        front = {i for i in range(len(gates)) if remaining[i] == 0}
        placed = [False] * len(gates)

        # Layout tracking (identity start)
        locations = list(range(num_qubits))   # virtual -> physical
        qubits_map = list(range(num_qubits))  # physical -> virtual

        out = QuantumCircuit(num_qubits, qc.num_clbits)

        def _activate_successors(gate_idx: int):
            qs, _ = gates[gate_idx]
            for q in qs:
                chain = qubit_chains[q]
                pos = chain.index(gate_idx)
                if pos + 1 < len(chain):
                    succ = chain[pos + 1]
                    remaining[succ] -= 1
                    if remaining[succ] == 0:
                        front.add(succ)

        def _place(gate_idx: int):
            qs, inst = gates[gate_idx]
            phys_qubits = [out.qubits[locations[q]] for q in qs]
            clbits = [out.clbits[qc.find_bit(c).index] for c in inst.clbits]
            out.append(inst.operation, phys_qubits, clbits)
            placed[gate_idx] = True
            _activate_successors(gate_idx)

        def _execute_ready():
            changed = True
            while changed:
                changed = False
                to_place = []
                for gidx in list(front):
                    qs, _ = gates[gidx]
                    if len(qs) <= 1:
                        to_place.append(gidx)
                    elif len(qs) == 2:
                        if dists[locations[qs[0]]][locations[qs[1]]] <= 1:
                            to_place.append(gidx)
                for gidx in to_place:
                    front.discard(gidx)
                    _place(gidx)
                    changed = True

        # Execute initially adjacent gates
        _execute_ready()

        # Process SWAPs from the solution
        for edge_idx in actions:
            l1, l2 = coupling_map[edge_idx]
            out.swap(l1, l2)

            vq1, vq2 = qubits_map[l1], qubits_map[l2]
            locations[vq1], locations[vq2] = l2, l1
            qubits_map[l1], qubits_map[l2] = vq2, vq1

            _execute_ready()

        init_layout = list(range(num_qubits))
        final_layout = list(locations)
        return out, init_layout, final_layout


if __name__ == "__main__":
    import random

    import time

    from ..model import ValueModel
    from .tensor_state_handler import TensorStateHandler
    from ..batch_weighted_astar_search import BWAS

    def generate_random_2qubit_circuit(num_qubits: int, num_gates: int) -> QuantumCircuit:
        if num_qubits < 2:
            raise ValueError("Number of qubits must be at least 2 for 2-qubit gates.")

        qc = QuantumCircuit(num_qubits)

        for _ in range(num_gates):
            control = random.randint(0, num_qubits - 1)
            target = random.randint(0, num_qubits - 1)
            while target == control:
                target = random.randint(0, num_qubits - 1)

            qc.cx(control, target)

        return qc


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
    bwas = BWAS(model, game)


    state = TensorState.from_quantum_circuit(qc, horizon=horizon)

    path = bwas.search(qc)
    state = TensorState(state)

    coupling_map = CouplingMap(topology)
    coupling_map.make_symmetric()


    iterations = 100
    my_time_list = []
    ibm_time_list = []

    bwas = BWAS(model, game)
    for _ in range(100):
        qc = generate_random_2qubit_circuit(6, 14)
        #print(qc)

        state = TensorState.from_quantum_circuit(qc, horizon=horizon)

        path = bwas.search(qc)

        start_time = time.time()
        new_qc, _ = state.as_subclass(TensorState).insert_swaps(qc, path, topology)
        end_time = time.time()
        my_time_list.append(end_time - start_time)

        start_time = time.time()
        new_qc2 = state.as_subclass(TensorState).insert_swaps2(path, qc, coupling_map)
        end_time = time.time()
        ibm_time_list.append(end_time - start_time)
        #print(new_qc)
        #print(new_qc2)

    print(f"my method takes on average {sum(my_time_list) / len(my_time_list)} s")
    print(f"IBM method takes on average {sum(ibm_time_list) / len(ibm_time_list)} s")
