import torch

from typing import Protocol


class To(Protocol):
    def to(self, device: torch.device) -> "To": ...
