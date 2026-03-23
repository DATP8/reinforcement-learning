from qiskit import QuantumCircuit
from abc import ABC, abstractmethod

class Router(ABC):
    @abstractmethod
    def solve(self, circuit: QuantumCircuit) -> list[int]:
        raise NotImplementedError