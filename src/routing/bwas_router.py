from qiskit import QuantumCircuit

from .rl_router import RlRouter
from ..routing.swap_inserter.swap_inserter import SwapInserter
from ..model import BiCircuitGNN

import qiskit
import torch
import heapq
import random
import time


class BWASNode:
    def __init__(self, state, g: float):
        self.state = state
        self.g = g
        self.parent_node = None
        self.action = None
    
    def get_child(self, next_state: torch.Tensor, action: int, action_cost: float):
        child_node = BWASNode(next_state, self.g + action_cost)
        child_node.parent_node = self
        child_node.action = action
        return child_node

class BWASRouter[S, To](RlRouter): 
    def search(self, root_state: S) -> list[int]:
        device = next(self.model.parameters()).device
        with torch.no_grad():
            h = self.model(self.state_handler.batch_states([root_state]).to(device)).item()
        
        counter = 0
        root_node = BWASNode(root_state, 0)
        open_set = [] 
        heapq.heappush(open_set, (h, counter, root_node))

        f_lookup = {root_state.__hash__(): h}
        closed_set = set()

        while open_set:
            nodes_to_expand = [heapq.heappop(open_set)[-1] for _ in range(min(self.batch_size, len(open_set)))]
            closed_set.update(node.state.__hash__() for node in nodes_to_expand)
            new_nodes = []
            for parent_node in nodes_to_expand:
                if self.state_handler.is_terminal(parent_node.state):
                    return self.reconstruct_path(parent_node)
                
                for action in self.state_handler.get_possible_actions(parent_node.state):
                    next_state = self.state_handler.get_next_state(parent_node.state, action)
                    next_node = parent_node.get_child(next_state, action, self.state_handler.get_action_cost(parent_node.state, action))
                    new_nodes.append(next_node)
            
            states = self.state_handler.batch_states([node.state for node in new_nodes])
            with torch.no_grad():
                h_values = self.model(states.to(device))
            
            for node, h in zip(new_nodes, h_values):
                f = self.weight * node.g +  h.item()
                if node.state.__hash__() in closed_set and f >= f_lookup[node.state.__hash__()]:
                    continue
                
                counter += 1
                heapq.heappush(open_set, (f, counter, node))
                f_lookup[node.state.__hash__()] = f
                
        return []
                
    def reconstruct_path(self, node):
        path = []
        while node.action is not None:
            path.append(node.action)
            node = node.parent_node
        return path[::-1]
    
    
if __name__ == "__main__":
    random.seed(42)
    from ..model import ValueModel, BiCircuitGNN
    from qiskit.transpiler.coupling import CouplingMap as CM
    from ..states.circuit_graph_state_handler import CircuitGraphStateHandler
    from qiskit.qpy import dump
    import random
    import time

    def generate_random_circuit(n_qubits: int, n_gates: int):
        qc = QuantumCircuit(n_qubits)
        
        for i in range(n_gates):
            q1, q2 = random.sample(range(n_qubits), 2)
            while q1 == q2:
                q2 = random.choice(range(n_qubits))

            qc.cx(q1, q2)
        
        return qc
    
    qc = QuantumCircuit(6)
    qc.cz(2, 3)  # 4
    qc.cx(3, 4)  # 5
    qc.cz(4, 5)  # 6
    qc.cx(5, 0)  # 7
    qc.cx(1, 3)  # 8
    qc.cz(2, 4)  # 9
    qc.cx(3, 5)  # 10
    qc.h(2)  # 11

    n_qubits = 6
    horizon = 100
    topology = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    state_handler = CircuitGraphStateHandler(n_qubits, topology)
    model = BiCircuitGNN(n_qubits)
    model.load_state_dict(torch.load("models/graph/difficulty43_updates9_iteration13490.pt", map_location=device))
    #model.load_state_dict(torch.load("models/graph/difficulty17_updates9_iteration3060.pt", map_location=device))
    
    # state_handler = QtensorStateHandler(n_qubits, horizon, topology)
    # model = ValueModelFlat(n_qubits, horizon, len(topology))
    # model.load_state_dict(torch.load("models/qtensor/difficulty99_iteration2510.pt", map_location=device))
    
    # state_handler = TensorStateHandler(n_qubits, horizon, topology)
    # model = ValueModel(n_qubits, horizon, len(topology))
    # model.load_state_dict(torch.load("models/davi_diff_tensor/difficulty17_iteration95270.pt", map_location=device))
    
    
    bwas = BWASRouter(model.to(device), state_handler)
    swap_inserter = SwapInserter(topology, num_qubits=n_qubits)
        
    # circuit = generate_random_circuit(n_qubits, n_gates=30)
    # with open("circuits/dud2.qpy", "wb") as f:
    #     dump(circuit, f)
    
    with open("circuits/dud.qpy", "rb") as f:
        circuit = load(f)[0]
    
    # circuit = QuantumCircuit(6)
    # circuit.cx(0,2)
    # circuit.h(1)
    # circuit.cx(0,1)
    # circuit.cs(0,3)
    # circuit.h(3)
    # circuit.cx(1,3)
    # circuit.h(3)
    
    print(circuit)
    t0 = time.time()
    state = state_handler.state_from(circuit)
    actions = bwas.search(state)
    t1 = time.time()
    
    routed_circuit, init_layout, final_layout = swap_inserter.build_circuit_from_solution(actions, circuit)
    
    optimized_circuit = qiskit.transpile(routed_circuit, optimization_level=3, coupling_map=CM(topology))
    
    print(f"Search time: {t1 - t0:.4f} seconds")
    print("Routed Circuit:")
    print(routed_circuit)
    print("Initial Layout:", init_layout)
    print("Final Layout:", final_layout)
    print("Path found:", actions)
    print("Path length:", len(actions))
    
    print(optimized_circuit)
    print("number of gates:", len(optimized_circuit.data))
    print("depth:", optimized_circuit.depth())

                    