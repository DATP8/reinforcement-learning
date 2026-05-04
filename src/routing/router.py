from src.states.state_handler import StateHandler
from qiskit import QuantumCircuit
from abc import ABC, abstractmethod


class Router(ABC):
    state_handler: StateHandler

    @abstractmethod
    def solve(self, circuit: QuantumCircuit) -> list[int]:
        raise NotImplementedError
