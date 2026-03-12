import random
import torch
from typing import overload
from typing import List
from typing import SupportsIndex
from abc import ABC, abstractmethod
from typing import Iterable, Protocol

class Batchable[S](Protocol, Iterable[S]):
    def __len__(self) -> int: ...
    @overload
    def __getitem__ (self, i: SupportsIndex, /) -> S:...
    @overload
    def __getitem__ (self, s: slice, /) -> 'Batchable[S]': ...
    

class StateHandler[S](ABC):        
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
    
    def get_random_states_in_range(self, batch_size: int, min_difficulty: int, max_difficulty: int) -> Batchable[S]:
        """Default implementation"""
        return [self.get_random_state(random.randint(min_difficulty, max_difficulty)) for _ in range(batch_size)]
    
    def get_random_states_at_difficulty(self, batch_size: int, difficulty: int) -> Batchable[S]:
        """Default implementation"""
        return [self.get_random_state(difficulty) for _ in range(batch_size)]
    
    @abstractmethod
    def get_random_state(self, difficulty: int) -> S:
        raise NotImplementedError
    
    @abstractmethod
    def batch_states(self, states: Batchable[S]) -> S:
        raise NotImplementedError