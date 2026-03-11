from abc import ABC, abstractmethod
from typing import Sequence, Union, Iterable, Protocol

class BaseGame[S](ABC):        
    @abstractmethod
    def get_topology(self) -> list[int]:
        raise NotImplementedError

class Batchable[S](Protocol, Iterable[S]):
    def __len__(self) -> int: ...

class StateHandler[S](ABC):      
    
    @abstractmethod
    def get_topology(self) -> list:
        raise NotImplementedError
    
    @abstractmethod
    def get_qubits(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_possible_actions(self, state: S) -> list[int]:
        raise NotImplementedError
    
    @abstractmethod
    def get_next_state(self, state: S, action: int) -> S:
        raise NotImplementedError
    
    @abstractmethod
    def is_terminal(self, state: S) -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def get_action_cost(self, state: S, action: int) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def prune(self, state: S) -> tuple[S, int]:
        raise NotImplementedError
    
    @abstractmethod
    def get_random_states(self, batch_size: int, max_difficulty: int) -> Batchable[S]:
        raise NotImplementedError
    
    @abstractmethod
    def batch_states(self, states: Batchable[S]) -> S:
        """
        `batch_size`: None if all states should be combined
        """
        
        raise NotImplementedError