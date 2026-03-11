from collections import defaultdict
from basegame import BaseGame
import torch

class Swapper(BaseGame[torch.Tensor]):
    def __init__(self, n_qubits: int, horizon: int, topology: list[tuple[int, int]]):
        self.n_qubits = n_qubits
        self.horizon = horizon
        self.topology = topology
        self.actions = list(range(len(topology)))
    
    def get_possible_actions(self, state: torch.Tensor) -> list[int]:
        return self.actions
    
    def get_next_state(self, state: torch.Tensor, action: int) -> torch.Tensor:
        state = state.clone()
        i, j = self.topology[action]
        state[i], state[j] = state[j], state[i]
        
        return state
    
    def is_terminal(self, state: torch.Tensor) -> bool:
        return torch.sum(state).item() < 1e-6
    
    def get_action_cost(self, state: torch.Tensor, action: int) -> float:
        # todo: experiment with using lower cost for swaps that allign with cnots in front layer
        return 1.0
    
    def prune(self, state: torch.Tensor) -> tuple[torch.Tensor, int]:
        new_state = state.clone()
        frontlayer_qubits = defaultdict(lambda: float('inf'))
        frontlayer = {}

        for q in range(self.n_qubits):
            for i in range(self.horizon):
                if state[q, i] > 0:
                    [q1, q2] = torch.where(state[:, i] > 0)[0].tolist()
                    if q1 < frontlayer_qubits[q1] and q2 < frontlayer_qubits[q2]:
                        frontlayer_qubits[q1] = q1
                        frontlayer_qubits[q2] = q2
                        frontlayer[(q1, q2)] = i
                    break

        for (q1, q2), i in frontlayer.items():
            if (q1, q2) in self.topology or (q2, q1) in self.topology:
                new_state[q1, i] = 0
                new_state[q2, i] = 0
                
        if torch.equal(state, new_state):
            return state, 0
        
        new_state, removed_gates = self.prune(new_state)
    
        return new_state, removed_gates + 1
    
    
if __name__ == "__main__":
    topology = [(0, 1), (1, 2), (2, 3)]
    n_qubits = 4
    horizon = 10
    
    swapper = Swapper(n_qubits, horizon, topology)
    
    state = torch.zeros((n_qubits, horizon), dtype=torch.float32)
    state[0,1] = 1.0
    state[2,1] = 1.0
    state[0,0] = 1.0
    state[1,0] = 1.0
    
    print(state)
    
    new_state = swapper.prune(state)[0]
    
    print(new_state)