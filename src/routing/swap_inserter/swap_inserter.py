from qiskit.circuit import CircuitInstruction
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap as CM


class SwapInserter:
    def __init__(self, coupling_map: list[tuple[int, int]] | CM, num_qubits: int):
        self.coupling_map = (
            CM(coupling_map) if isinstance(coupling_map, list) else coupling_map
        )
        self.num_qubits = num_qubits

    def build_circuit_from_solution(
        self, actions: list[int], input_circuit: QuantumCircuit
    ) -> tuple[QuantumCircuit, list[int], list[int]]:
        """Reconstruct a routed circuit from the solution (coupling-map edge indices).

        The solution records which coupling-map edge was swapped at each step.
        This method replays those SWAPs, placing original gates whenever their
        qubits become adjacent, and returns the routed circuit plus layouts.

        Returns (routed_circuit, init_layout, final_layout).
        """

        dists = self.coupling_map.distance_matrix.astype(  # pyrefly: ignore[missing-attribute]
            int
        )

        # Build gate list with virtual-qubit indices
        gates: list[tuple[list[int], object]] = []
        for inst in input_circuit.data:
            qubits = [input_circuit.find_bit(q).index for q in inst.qubits]
            gates.append((qubits, inst))

        # Per-qubit chains: ordered gate indices touching each virtual qubit
        qubit_chains: list[list[int]] = [[] for _ in range(self.num_qubits)]
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
        locations = list(range(self.num_qubits))  # virtual -> physical
        qubits_map = list(range(self.num_qubits))  # physical -> virtual

        out = QuantumCircuit(self.num_qubits, input_circuit.num_clbits)

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
            inst: CircuitInstruction  # type:ignore
            phys_qubits = [out.qubits[locations[q]] for q in qs]
            clbits = [out.clbits[input_circuit.find_bit(c).index] for c in inst.clbits]  # pyrefly: ignore[missing-attribute]
            out.append(inst.operation, phys_qubits, clbits)  # pyrefly: ignore[missing-attribute]
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
            l1, l2 = self.coupling_map.get_edges()[edge_idx]
            out.swap(l1, l2)

            vq1, vq2 = qubits_map[l1], qubits_map[l2]
            locations[vq1], locations[vq2] = l2, l1
            qubits_map[l1], qubits_map[l2] = vq2, vq1

            _execute_ready()

        init_layout = list(range(self.num_qubits))
        final_layout = list(locations)
        return out, init_layout, final_layout


if __name__ == "__main__":
    coupling_map = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    router = SwapInserter(coupling_map, num_qubits=6)

    circuit = QuantumCircuit(6)
    circuit.cx(0, 1)
    circuit.cx(0, 2)
    circuit.cx(0, 3)
    circuit.cx(0, 4)

    actions = [0, 1, 2]  # Example edge indices to swap
    routed_circuit, init_layout, final_layout = router.build_circuit_from_solution(
        actions, circuit
    )

    print("Original Circuit:")
    print(circuit)
    print("Routed Circuit:")
    print(routed_circuit)
    print("Initial Layout:", init_layout)
    print("Final Layout:", final_layout)
    print("number of gates:", len(routed_circuit.data))
    print("depth:", routed_circuit.depth())
