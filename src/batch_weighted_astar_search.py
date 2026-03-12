import heapq
import torch
import time

class BWASNode:
    def __init__(self, state, g: float):
        self.state = state
        self.g = g
        self.parent_node = None
        self.action = None
    
    def get_child(self, next_state: torch.Tensor, action: int, cost: float):
        child_node = BWASNode(next_state, self.g + cost)
        child_node.parent_node = self
        child_node.action = action
        return child_node

class BWAS:
    def __init__(self, model, game, batch_size=64, weight=0.3):
        self.model = model
        self.game = game
        self.batch_size = batch_size
        self.weight = weight
    
    def search(self, root_state) -> list[int]:
        with torch.no_grad():
            h = self.model(root_state.unsqueeze(0)).item()
        
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
                if self.game.is_terminal(parent_node.state):
                    return self.reconstruct_path(parent_node)
                
                for action in self.game.get_possible_actions(parent_node.state):
                    next_state = self.game.get_next_state(parent_node.state, action)
                    next_node = parent_node.get_child(next_state, action, self.game.get_action_cost(parent_node.state, action))
                    new_nodes.append(next_node)
            
            states = torch.stack([node.state for node in new_nodes])
            with torch.no_grad():
                h_values = self.model(states).squeeze()
            
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
    from model import ValueModel
    from tensor_state_handler import TensorStateHandler, CNOTCircuit
    import random
    
    def generate_random_circuit(game, n_qubits: int, n_gates: int, horizon: int):
        qc = CNOTCircuit(n_qubits)
        
        for i in range(n_gates):
            q1, q2 = random.sample(range(n_qubits), 2)
            while q1 == q2:
                q2 = random.choice(range(n_qubits))
            
            qc.add_cnot(q1, q2)
            
        state = qc.to_tensor(horizon=horizon)
        state, _ = game.prune(state)
        
        if game.is_terminal(state):
            return generate_random_circuit(game, n_qubits, n_gates, horizon)
            
        return state
    
    n_qubits = 6
    horizon = 100
    topology = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    
    game = TensorStateHandler(n_qubits, horizon, topology)
    model = ValueModel(n_qubits, horizon, len(topology))
    model.load_state_dict(torch.load("models/value_model_deep_cube_a_exp_relu/difficulty11_iteration6790.pt"))
    
    # circuit = CNOTCircuit(n_qubits)
    # circuit.add_cnot(0, 2)
    # circuit.add_cnot(1, 2)
    # circuit.add_cnot(2, 3)
    # circuit.add_cnot(4, 0)
    
    bwas = BWAS(model, game, batch_size=1)
    #root_state = generate_random_circuit(game, n_qubits, 11, horizon)
    root_state = torch.load("circuits/dud.pt")
    circuit = CNOTCircuit.from_tensor(root_state)
    #root_state = circuit.to_tensor(horizon=horizon)
    
    print(circuit)
    
    #torch.save(root_state, "circuits/dud.pt")
    
    start_time = time.time()
    path = bwas.search(root_state)
    end_time = time.time()
    
    print(f"search time: {end_time - start_time:.4f}s")
    state = root_state
    for action in path:
        state = game.get_next_state(state, action)

        
    new_circuit = CNOTCircuit.from_tensor(state)
    print(new_circuit)
    
    print("Found path:", path)
                
                    