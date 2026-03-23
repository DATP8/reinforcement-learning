from circuit_generator import CircuitGenerator
from routing.swap_inserter.swap_inserter import SwapInserter
from states.circuit_graph_state_handler import CircuitGraphStateHandler
from utils.to import To
from states.state_handler import StateHandler
from torch import nn
import torch

class PolicySearch[S: To]:
    def __init__(self, model: nn.Module, state_handler: StateHandler[S], temperature=1.0):
        self.model = model
        self.state_handler = state_handler  
        self.temperature = temperature
    
    def search(self, state: S) -> list[int]:
        states = {hash(state)}
        device = next(self.model.parameters()).device
        actions = []
        while not self.state_handler.is_terminal(state):
            with torch.no_grad():
                logits = self.model(self.state_handler.batch_states([state]).to(device))
            policy = torch.softmax(logits / self.temperature, dim=-1).squeeze(0)
            print("Policy:", policy.cpu().numpy())
            action = torch.multinomial(policy, num_samples=1).item()
            assert type(action) == int, f"Expected action to be an int, got {type(action)}"
            state = self.state_handler.get_next_state(state, action)
            while hash(state) in states:
                policy[action] = 0
                if policy.sum() == 0:
                    raise ValueError("No more valid actions to sample")
                policy = policy / policy.sum()
                action = torch.multinomial(policy, num_samples=1).item()
                state = self.state_handler.get_next_state(state, action)
                
            states.add(hash(state))
            actions.append(action)

        return actions
    

if __name__ == "__main__":
    from model import BiCircuitGNNPolicy
    from qiskit import QuantumCircuit
    import time
    from qiskit.qpy import dump, load
    
    n_qubits = 6
    horizon = 100
    topology = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    state_handler = CircuitGraphStateHandler(6, topology)
    model = BiCircuitGNNPolicy(n_qubits, len(topology)).to(device)
    model.load_state_dict(torch.load("models/policy/iteration799.pt", map_location=device))
    router = SwapInserter(topology, num_qubits=n_qubits)
    policy_search = PolicySearch(model, state_handler, temperature=0.5)
    
    circuit = CircuitGenerator.generate_random_circuit(n_qubits, gateset={"cx"}, num_gates=30)
    with open("circuits/dud2.qpy", "wb") as f:
        dump(circuit, f)
    
    # with open("circuits/dud.qpy", "rb") as f:
    #     circuit = load(f)[0]
    
    actions = policy_search.search(state_handler.state_from(circuit))
    routed_circuit, init_layout, final_layout = router.build_circuit_from_solution(actions, circuit)
    
    print("Original circuit:")
    print(circuit)
    print("Routed circuit:")
    print(routed_circuit)
    print("Gate count: " + str(len(routed_circuit.data)))
    print("Depth: " + str(routed_circuit.depth()))