from torch import Size
from sympy.printing.pytorch import torch
from cnot_circuit import CNOTCircuit
import torch
from basegame import BaseGame

class SwapOptimizer(BaseGame[torch.Tensor]):
    def __init__(self, n_qubits: int, horizon: int, topology: list[tuple[int, int]]):
        self.horizon = horizon
        self.topology = topology
        self.mask = self.init_mask(n_qubits)
        
    def init_mask(self, n_qubits: int):
        mask = torch.ones((n_qubits, n_qubits), dtype=torch.float32)
        for (q1, q2) in self.topology:
            mask[q1, q2] = 0.0
            mask[q2, q1] = 0.0
        return mask
    
    def get_possible_actions(self, state: torch.Tensor):
        return list(range(len(self.topology)))

    def prune(self, state: torch.Tensor) -> tuple[torch.Tensor, int]:
        new_state = state.clone()
        layers_removed = 0
        new_state[:, :, 0] *= self.mask
            
        while layers_removed < self.horizon - 1 and torch.sum(new_state[:, :, layers_removed]) <= 1e-7:
            new_state[:, :, layers_removed + 1] *= self.mask
            layers_removed += 1
        
        new_state[:,:,:self.horizon - layers_removed] = new_state[:, :, layers_removed:]
        new_state[:,:,self.horizon - layers_removed:] = 0.0
        
        return new_state, layers_removed

    def get_next_state(self, state: torch.Tensor, action: int) -> torch.Tensor:
        new_state = state.clone()
        q1, q2 = self.topology[action]
        
        # Swap the qubits in the tensor representation
        new_state[q1, :, :] = state[q2, :, :].clone()
        new_state[q2, :, :] = state[q1, :, :].clone()

        new_state[:, q1, :], new_state[:, q2, :] = new_state[:, q2, :].clone(), new_state[:, q1, :].clone()
    
    
        # If the first layer is now empty, shift all layers by one  
        new_state, layers_removed = self.prune(new_state)

        return new_state

    def is_terminal(self, state: torch.Tensor) -> bool:
        return torch.sum(state).item() <= 1e-7    

    def get_action_cost(self, state: torch.Tensor, action: int) -> float:
        return 1.0


if __name__ == "__main__":
    n_qubits = 6
    horizon = 10
    
    circuit = CNOTCircuit(n_qubits)
    circuit.add_cnot(0, 2)
    circuit.add_cnot(1, 2)
    
    topology = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    
    game = SwapOptimizer(n_qubits, horizon, topology)
    
    root_state = circuit.to_tensor(horizon=horizon)    
    
    print(circuit)
    #new_state, layers_removed = game.prune(root_state)
    #print("Layers removed:", layers_removed)
    
    new_state = game.get_next_state(root_state, 0)
    new_circuit = CNOTCircuit.from_tensor(new_state)
    print(new_circuit)
    