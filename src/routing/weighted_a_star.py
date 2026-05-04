from qiskit import QuantumCircuit
from src.states.state_handler import StateHandler
from src.routing.router import Router

class WeightedAStar(Router):
    def __init__(self, model, state_handler: StateHandler, weight=1.0):
        self.model = model
        self.state_handler = state_handler
        self.weight = weight
        
    
    def solve(self, circuit: QuantumCircuit) -> list[int]:
        raise NotImplementedError
