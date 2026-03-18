from deep_approximate_value_iteration import To
from model import BiCircuitGNN
from state_handler import StateHandler
from qiskit import QuantumCircuit
import heapq
import torch



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

class BWAS[S: To]:
    def __init__(self, model, state_handler: StateHandler[S]):
        self.model = model
        self.state_handler = state_handler
    
    def search(self, circuit: QuantumCircuit, batch_size=64, weight=0.3) -> list[int]:
        root_state = self.state_handler.state_from(circuit)
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
            nodes_to_expand = [
                heapq.heappop(open_set)[-1]
                for _ in range(min(batch_size, len(open_set)))
            ]
            closed_set.update(node.state.__hash__() for node in nodes_to_expand)
            new_nodes = []
            for parent_node in nodes_to_expand:
                if self.state_handler.is_terminal(parent_node.state):
                    return self.reconstruct_path(parent_node)

                for action in self.state_handler.get_possible_actions(parent_node.state):
                    next_state = self.state_handler.get_next_state(parent_node.state, action)
                    next_node = parent_node.get_child(
                        next_state,
                        action,
                        self.state_handler.get_action_cost(parent_node.state, action),
                    )
                    new_nodes.append(next_node)

            states = torch.stack([node.state for node in new_nodes])
            with torch.no_grad():
                h_values = self.model(states).squeeze()

            for node, h in zip(new_nodes, h_values):
                f = weight * node.g + h.item()
                if (
                    node.state.__hash__() in closed_set
                    and f >= f_lookup[node.state.__hash__()]
                ):
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
    import random
    from circuit_graph_state_handler import CircuitGraphStateHandler
    import time
    from qiskit.qpy import dump
    
    def generate_random_circuit(n_qubits: int, n_gates: int):
        qc = QuantumCircuit(n_qubits)
        
        for i in range(n_gates):
            q1, q2 = random.sample(range(n_qubits), 2)
            while q1 == q2:
                q2 = random.choice(range(n_qubits))
            
            qc.cx(q1, q2)
        
        return qc
    
    n_qubits = 6
    horizon = 100
    topology = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    state_handler = CircuitGraphStateHandler(n_qubits, topology)
    model = BiCircuitGNN(n_qubits)
    #model.load_state_dict(torch.load("models/graph/difficulty43_updates9_iteration13490.pt", map_location=device))
    model.load_state_dict(torch.load("models/graph/difficulty17_updates9_iteration3060.pt", map_location=device))
    
    # state_handler = QtensorStateHandler(n_qubits, horizon, topology)
    # model = ValueModelFlat(n_qubits, horizon, len(topology))
    # model.load_state_dict(torch.load("models/qtensor/difficulty99_iteration2510.pt", map_location=device))
    
    # state_handler = TensorStateHandler(n_qubits, horizon, topology)
    # model = ValueModel(n_qubits, horizon, len(topology))
    # model.load_state_dict(torch.load("models/davi_diff_tensor/difficulty17_iteration95270.pt", map_location=device))
    
    
    bwas = BWAS(model.to(device), state_handler)
    
        
    circuit = generate_random_circuit(n_qubits, n_gates=16)
    with open("circuits/dud.qpy", "wb") as f:
        dump(circuit, f)
    
    # with open("circuits/dud.qpy", "rb") as f:
    #     circuit = load(f)[0]
    
    print(circuit)
    t0 = time.time()
    path = bwas.search(circuit, batch_size=64, weight=0.3)
    t1 = time.time()
    print(f"Search time: {t1 - t0:.4f} seconds")
    print("Path found:", path)
    print("Path length:", len(path))

                    
