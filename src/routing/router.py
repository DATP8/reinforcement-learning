from abc import ABC, abstractmethod

from qiskit import QuantumCircuit


class Router(ABC):
    @abstractmethod
    def solve(self, circuit: QuantumCircuit) -> list[int]:
        raise NotImplementedError
