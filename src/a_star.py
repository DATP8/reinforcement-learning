from model import PVModel
from state_handler import StateHandler
import torch


class AStarPolicySearch:
    def __init__(self):
        self.g = {}  # cost from start to current node
        self.f = {}  # estimated total cost from start to goal through current node
        self.parent = {}  # parent of each node in the path
        self.frontier = set[torch.Tensor]()  # frontier states

    def expand(
        self,
        state: torch.Tensor,
        game: StateHandler[torch.Tensor],
        model: PVModel,
        n_steps=100,
        k=10,
    ) -> float:
        if game.is_terminal(state):
            return self.g[state.__hash__()]

        raise NotImplementedError

    def search(
        self,
        state: torch.Tensor,
        game: StateHandler[torch.Tensor],
        model: PVModel,
        n_steps=100,
        k=10,
    ):
        self.g[state.__hash__()] = 0
        policy, v = model.predict(state)

        self.frontier.add(state)
        self.g[state.__hash__()] = 0

        for _ in range(n_steps):
            if len(self.frontier) == 0:
                break

            current = min(self.frontier, key=lambda s: self.f.get(s, float("inf")))
            policy, v = model.predict(current)
            self.f[current] = self.g[current.__hash__()] + v.item()
