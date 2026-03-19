from .router import Router
from abc import abstractmethod
from typing import Protocol

from ..states.state_handler import StateHandler

import torch


class To(Protocol):
    def to(self, device: torch.device) -> "To": ...


class RlRouter[S: To](Router):
    def __init__(
        self, model, state_handler: StateHandler[S], batch_size=64, weight=0.3
    ):
        self.model = model
        self.state_handler = state_handler
        self.batch_size = batch_size
        self.weight = weight

    @abstractmethod
    def search(self, root_state: S) -> list[int]:
        pass
