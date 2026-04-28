from typing import Protocol

import torch


class To(Protocol):
    def to(self, device: torch.device) -> "To": ...
