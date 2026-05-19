from abc import ABC, abstractmethod

from qiskit import QuantumCircuit

from src.states.state_handler import StateHandler


class Router(ABC):
    state_handler: StateHandler

    @abstractmethod
    def solve(self, circuit: QuantumCircuit) -> list[int]:
        raise NotImplementedError
