from deep_approximate_value_iteration import To
from batch_weighted_astar_search import BWAS
from state_handler import StateHandler
from torch import nn
import torch
from multiprocessing import Pool
import os


class PolicyTrainer[S: To]:
    def __init__(self, heurisitic_model: nn.Module, state_handler: StateHandler[S]):
        self.heuristic_model = heurisitic_model
        self.state_handler = state_handler
        self.bwas = BWAS(heurisitic_model, state_handler)
        
    def search(self, state: S) -> tuple[S, list[int]]:
        return state, self.bwas.search(state, batch_size=64, weight=0.3)
        
    def train(self, policy_model: nn.Module, batch_size=64, num_iterations=100000, weight=0.3, max_difficulty=43):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        policy_model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(policy_model.parameters())
        cpu_count = os.cpu_count()
        print(cpu_count, "CPU cores detected. Using multiprocessing for data generation.")
        
        for iteration in range(num_iterations):
            states = self.state_handler.get_random_states_in_range(batch_size, 1, max_difficulty)
            action_list = []
            state_list = []
            
            with Pool(cpu_count) as pool:
                state_action_groups = pool.map(self.search, states)
            
            for state, actions in state_action_groups:
                action_list.extend(actions)
                state_list.extend(self.construct_states_from_actions(state, actions))

            X = self.state_handler.batch_states(state_list).to(device)
            y = torch.tensor(action_list, dtype=torch.long).to(device)
            
            logits = policy_model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            print(f"Iteration {iteration+1}/{num_iterations}, Loss: {loss.item():.4f}")
            torch.save(policy_model.state_dict(), f"models/policy/iteration{iteration}.pt")

    def construct_states_from_actions(self, state: S, actions: list[int]) -> list[S]:
        states = [state]
        current_state = state
        for action in actions[:-1]: # Exclude the last action as terminal state should not be included
            current_state = self.state_handler.get_next_state(current_state, action)
            states.append(current_state)

        return states
            

if __name__ == "__main__":
    from model import BiCircuitGNN, BiCircuitGNNPolicy
    from circuit_graph_state_handler import CircuitGraphStateHandler

    n_qubits = 6
    horizon = 10
    topology = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    state_handler = CircuitGraphStateHandler(n_qubits, topology)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    policy_model = BiCircuitGNNPolicy(n_qubits, n_actions=len(topology), hidden_dim=128)
    heurisitc_model = BiCircuitGNN(n_qubits, hidden_dim=128)
    heurisitc_model.load_state_dict(torch.load("models/graph/difficulty45_iteration12510.pt", map_location=device))
    
    trainer = PolicyTrainer(heurisitc_model, state_handler)
    trainer.train(policy_model, batch_size=os.cpu_count(), num_iterations=1000000, weight=0.3, max_difficulty=43)

