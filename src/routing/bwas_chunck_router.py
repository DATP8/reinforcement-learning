from qiskit.transpiler import Layout
from qiskit import QuantumCircuit
from src.routing.bwas_router import BWASRouter


class ChunkRouter(BWASRouter):
    def __init__(self, chunk_size: int, **kwargs):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size

    def solve(self, circuit: QuantumCircuit) -> list[int]:
        circuit_chunks = self._chunk_circuit(circuit)

        # Start with a trivial layout
        layout = Layout.from_qubit_list(circuit.qubits, range(len(circuit.qubits)))

        all_actions = []
        for chunk in circuit_chunks:
            chunk = self._apply_layout_to_circuit(chunk, layout)
            root_state = self.state_handler.state_from(
                chunk
            )  # update chuck to reflect preveus swaps
            actions = self.search(root_state)
            all_actions.extend(actions)
            layout = self._update_layout(layout, actions)

        return all_actions

    def _apply_layout_to_circuit(
        self, circuit: QuantumCircuit, layout: Layout
    ) -> QuantumCircuit:
        new_circ = QuantumCircuit(circuit.num_qubits)

        for instr, qargs, cargs in circuit.data:
            physical_qargs = [layout[q] for q in qargs]
            new_circ.append(instr, physical_qargs, cargs)

        return new_circ

    def _chunk_circuit(self, circuit: QuantumCircuit) -> list[QuantumCircuit]:
        chunks = []
        current_chunk = QuantumCircuit(circuit.num_qubits)
        for gate in circuit.data:
            if len(current_chunk.data) >= self.chunk_size:
                chunks.append(current_chunk)
                current_chunk = QuantumCircuit(circuit.num_qubits)
            current_chunk.append(gate.operation, gate.qubits, gate.clbits)

        chunks.append(current_chunk)
        return chunks

    def _update_layout(self, layout: Layout, actions: list[int]) -> Layout:
        topology = self.state_handler.get_topology()
        for action in actions:
            q1, q2 = topology[action]
            layout.swap(q1, q2)

        return layout


if __name__ == "__main__":
    from qiskit.qpy import load, dump
    import torch
    from src.states.circuit_graph_state_handler import CircuitGraphStateHandler
    from src.circuit_generator import CircuitGenerator
    from src.model import BiCircuitGNN
    from src.routing.swap_inserter.swap_inserter import SwapInserter

    n_qubits = 6
    horizon = 100
    topology = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # circuit = CircuitGenerator.generate_random_circuit(6, gateset={"cx"}, num_gates=3)
    # with open("circuits/dud2.qpy", "wb") as f:
    #     dump(circuit, f)

    with open("circuits/dud2.qpy", "rb") as f:
        circuit = load(f)[0]

    state_handler = CircuitGraphStateHandler(6, topology)
    model = BiCircuitGNN(n_qubits)
    model.load_state_dict(
        torch.load("models/graph/difficulty45_iteration12510.pt", map_location=device)
    )

    router = ChunkRouter(chunk_size=1, state_handler=state_handler, model=model)
    actions = router.solve(circuit)
    print("Actions:", actions)
    swap_inserter = SwapInserter(topology, num_qubits=n_qubits)
    final_circuit, initial_layout, final_layout = (
        swap_inserter.build_circuit_from_solution(actions, circuit)
    )

    print("Original circuit:")
    print(circuit)
    print("Initial layout:", initial_layout)
    print("Routed circuit:")
    print(final_circuit)
    print("Final layout:", final_layout)
