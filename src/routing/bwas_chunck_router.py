from qiskit import QuantumCircuit
from routing.bwas_router import BWASRouter

class ChunkRouter(BWASRouter):
    def __init__(self, chunk_size: int, **kwargs):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size

    def solve(self, circuit: QuantumCircuit) -> list[int]:
        circuit_chunks = self.chunk_circuit(circuit)
        mapping = {q: q for q in range(circuit.num_qubits)} # Identity mapping at the start
        
        all_actions = []
        for chunk in circuit_chunks:
            mapped_chunk = self.apply_mapping(chunk, mapping)
            root_state = self.state_handler.state_from(mapped_chunk)
            actions = self.search(root_state)
            all_actions.extend(actions)
            mapping = self.update_mapping(mapping, actions)
        
        return all_actions    
        
    def _chunk_circuit(self, circuit: QuantumCircuit) -> list[QuantumCircuit]:
        chunks = []
        current_chunk = QuantumCircuit(circuit.num_qubits)
        for gate in circuit.data:
            if len(current_chunk.data) >= self.chunk_size:
                chunks.append(current_chunk)
                current_chunk = QuantumCircuit(circuit.num_qubits)
            current_chunk.append(gate[0], gate[1], gate[2])

        return chunks
    
    def _update_mapping(self, mapping: dict[int, int], actions: list[int]) -> dict[int, int]:
        topology = self.state_handler.get_topology()
        for action in actions:
            q1, q2 = topology[action]
            mapping[q1], mapping[q2] = mapping[q2], mapping[q1]
        return mapping
    
    def _apply_mapping(self, circuit: QuantumCircuit, mapping: dict[int, int]) -> QuantumCircuit:
        new_circuit = QuantumCircuit(circuit.num_qubits)
        for gate in circuit.data:
            new_qargs = [mapping[q.index] for q in gate[1]]
            new_circuit.append(gate[0], new_qargs, gate[2])
        return new_circuit
    
    
if __name__ == "__main__":
    from qiskit.qpy import load, dump
    import torch
    from states.circuit_graph_state_handler import CircuitGraphStateHandler
    from circuit_generator import CircuitGenerator
    from model import BiCircuitGNN
    from model import model
    from routing.swap_inserter.swap_inserter import SwapInserter
    
    n_qubits = 6
    horizon = 100
    topology = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    circuit = CircuitGenerator.generate_random_circuit(6, gateset={"cx"}, num_gates=30)
    with open("circuits/dud2.qpy", "wb") as f:
        dump(circuit, f)
        
    # with open("circuits/dud2.qpy", "rb") as f:
    #     circuit = load(f)[0]
    

    state_handler = CircuitGraphStateHandler(6, topology)
    model = BiCircuitGNN(n_qubits)
    model.load_state_dict(torch.load("models/graph/difficulty45_iteration12510.pt", map_location=device))
    
    router = ChunkRouter(chunk_size=10, state_handler=state_handler, model=model)
    actions = router.solve(circuit)
    print("Actions:", actions)
    swap_inserter = SwapInserter(topology, num_qubits=n_qubits)

    