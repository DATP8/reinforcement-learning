from abc import ABC, abstractmethod

class BaseGame[S](ABC):        
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
    