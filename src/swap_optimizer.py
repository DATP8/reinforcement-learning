from torch import Size
from sympy.printing.pytorch import torch
from cnot_circuit import CNOTCircuit
import torch
from basegame import BaseGame

class SwapOptimizer(BaseGame[torch.Tensor]):
    def __init__(self, circuit_tensor: torch.Tensor, topology: list[tuple[int, int]]):
        self.topology = topology
        self.horizon = circuit_tensor.size(3)
        self.mask = self.init_mask(circuit_tensor.size(1))
        self.root_state, _ = self.prune(circuit_tensor)
        self.root_state[:,:,:,0] *= self.mask
        self.circuit = CNOTCircuit.from_tensor(circuit_tensor)
        self.state_depths = {self.root_state.__hash__(): self.circuit.depth()}
        
    def init_mask(self, n_qubits: int):
        mask = torch.ones((1, n_qubits, n_qubits), dtype=torch.float32)
        for (q1, q2) in self.topology:
            mask[:, q1, q2] = 0.0
            mask[:, q2, q1] = 0.0
        return mask
        
    def get_initial_state(self) -> torch.Tensor:
        return self.root_state
    

    def get_possible_actions(self, state: torch.Tensor) -> list[int]:
        return list(range(len(self.topology)))
    

    def prune(self, state: torch.Tensor) -> tuple[torch.Tensor, int]:
        new_state = state.clone()
        layers_removed = 0
        new_state[:, :, :, 0] *= self.mask
            
        while layers_removed < self.horizon - 1 and torch.sum(new_state[0, :, :, layers_removed]) <= 1e-7:
            new_state[:, :, :, layers_removed + 1] *= self.mask
            layers_removed += 1
        
        new_state[:,:,:,:self.horizon - layers_removed] = new_state[:, :, :, layers_removed:]
        new_state[:, :, :,self.horizon - layers_removed:] = 0.0
        
        return new_state, layers_removed

    def get_next_state(self, state: torch.Tensor, action: int) -> torch.Tensor:
        depth = self.state_depths.get(state.__hash__(), None)
        new_state = state.clone()
        q1, q2 = self.topology[action]
        
        # Swap the qubits in the tensor representation
        new_state[0, q1, :, :] = state[0, q2, :, :].clone()
        new_state[0, q2, :, :] = state[0, q1, :, :].clone()

        new_state[0, :, q1, :], new_state[0, :, q2, :] = new_state[0, :, q2, :].clone(), new_state[0, :, q1, :].clone()
    
    
        # If the first layer is now empty, shift all layers by one  
        new_state, layers_removed = self.prune(new_state)
        
        depth = min((depth - layers_removed), 0) if depth is not None else None

        new_state_hash = new_state.__hash__()
        if depth is not None:
            self.state_depths[new_state_hash] = depth

        return new_state

    def is_terminal(self, state: torch.Tensor) -> bool:
        # if state.__hash__() not in self.state_depths:
        #     raise ValueError(f"State {state} not found in state_depths. This should not happen if get_next_state is implemented correctly.")
        
        #return self.state_depths[state.__hash__()] <= 0
        return torch.sum(state).item() <= 1e-7
    

    def get_reward(self, state: torch.Tensor) -> float:
        raise NotImplementedError
    

    def get_action_cost(self, state: torch.Tensor, action: int) -> float:
        return 1.0


if __name__ == "__main__":
    n_qubits = 6
    horizon = 10
    
    circuit = CNOTCircuit(n_qubits)
    circuit.add_cnot(0, 2)
    circuit.add_cnot(1, 2)
    
    topology = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    
    game = SwapOptimizer(circuit.to_tensor(horizon=horizon), topology)
    
    root_state = circuit.to_tensor()
    game = SwapOptimizer(root_state.clone(), topology=topology)
    
    
    print(circuit)
    #new_state, layers_removed = game.prune(root_state)
    #print("Layers removed:", layers_removed)
    
    new_state = game.get_next_state(root_state, 2)
    new_circuit = CNOTCircuit.from_tensor(new_state)
    print(new_circuit)
    