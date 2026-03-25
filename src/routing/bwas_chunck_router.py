from copy import deepcopy
from qiskit import QuantumCircuit
from src.routing.bwas_router import BWASRouter

class ChunkRouter(BWASRouter):
    def __init__(self, chunk_size: int, **kwargs):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size

    def solve(self, circuit: QuantumCircuit) -> list[int]:
        circuit_chunks = self._chunk_circuit(circuit)
        mapping = [q for q in range(circuit.num_qubits)] # Identity mapping at the start
        all_actions = []
        for chunk in circuit_chunks:
            mapped_chunk = self._apply_mapping(chunk, mapping)
            # print(mapping)
            # print(chunk)
            # print(mapped_chunk)
            root_state = self.state_handler.state_from(mapped_chunk)
            actions = self.search(root_state)
            # print(actions)
            all_actions.extend(actions)
            mapping = self._update_mapping(mapping, actions)
        
        return all_actions    
        
    def _chunk_circuit(self, circuit: QuantumCircuit) -> list[QuantumCircuit]:
        chunks = []
        current_chunk = QuantumCircuit(circuit.num_qubits)
        for gate in circuit.data:
            if len(current_chunk.data) >= self.chunk_size:
                chunks.append(current_chunk)
                current_chunk = QuantumCircuit(circuit.num_qubits)
            current_chunk.append(gate.operation, gate.qubits, gate.clbits)
            
        chunks.append(current_chunk)
        # print("UUUUUUUU")
        # print(len(chunks))
        # for chuck in chunks:
            # print(len(chuck.data))
        return chunks
    
    def _update_mapping(self, mapping: list[int], actions: list[int]) -> list[int]:
        topology = self.state_handler.get_topology()
        for action in actions:
            q1, q2 = topology[action]
            # print(q1, q2)
            # print("Before swap:", mapping)
            mapping[q1], mapping[q2] = mapping[q2], mapping[q1]
            # print("After swap:", mapping)
            

        # print("Updated mapping:", mapping)
        # print("Using actions:", actions)
        return mapping
    
    def _apply_mapping(self, circuit: QuantumCircuit, mapping: list[int]) -> QuantumCircuit:
        new_circuit = QuantumCircuit(circuit.num_qubits)
        for gate in circuit.data:
            mapped_qubits = [mapping.index(q._index) for q in gate.qubits]
            new_circuit.append(gate.operation, mapped_qubits, gate.clbits)
        return new_circuit
    
    
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    # circuit = CircuitGenerator.generate_random_circuit(6, gateset={"cx"}, num_gates=3)
    # with open("circuits/dud2.qpy", "wb") as f:
    #     dump(circuit, f)
        
    with open("circuits/dud2.qpy", "rb") as f:
        circuit = load(f)[0]
    

    state_handler = CircuitGraphStateHandler(6, topology)
    model = BiCircuitGNN(n_qubits)
    model.load_state_dict(torch.load("models/graph/difficulty45_iteration12510.pt", map_location=device))
    
    router = ChunkRouter(chunk_size=1, state_handler=state_handler, model=model)
    actions = router.solve(circuit)
    print("Actions:", actions)
    swap_inserter = SwapInserter(topology, num_qubits=n_qubits)
    final_circuit, initial_layout, final_layout = swap_inserter.build_circuit_from_solution(actions, circuit)
    
    print("Original circuit:")
    print(circuit)
    print("Initial layout:", initial_layout)
    print("Routed circuit:")
    print(final_circuit)
    print("Final layout:", final_layout)
