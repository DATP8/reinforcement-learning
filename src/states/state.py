from abc import ABC, abstractmethod
from qiskit import QuantumCircuit

class State(ABC):

    @classmethod
    @abstractmethod
    def from_circuit(cls, qc: QuantumCircuit, horizon: int=0):
        raise NotImplementedError

    @abstractmethod
    def to_circuit(self) -> QuantumCircuit: 
        raise NotImplementedError
  
