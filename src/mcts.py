from math import sqrt
from cachetools import LRUCache

import torch
import uuid

from model import PVModel
from states.state_handler import StateHandler


class MCTSNode:
    def __init__(self):
        self.parent: None | MCTSNode = None
        self.action: None | int = None
        self.visit_count = 0
        self.huristic = 0.0
        self.policy = None
        self.children = {}  # action -> MCTSNode
        self.uuid = uuid.uuid4().int

    def get_child(self, action: int) -> "MCTSNode":
        if action in self.children:
            return self.children[action]

        child = MCTSNode()
        child.parent = self
        child.action = action
        self.children[action] = child

        return child

    def increase_visit_count(self):
        self.visit_count += 1

    def get_visit_count(self) -> int:
        return self.visit_count


class MCTS:
    def __init__(
        self, game: StateHandler[torch.Tensor], model: PVModel, cache_size=100
    ):
        self.model = model
        self.state_cache = LRUCache(maxsize=cache_size)  # nodeid ->
        # self.root_state = game.get_initial_state()
        self.actions = game.get_possible_actions(self.root_state)
        self.game = game
        self.root = MCTSNode()
        self.terminal_nodes = set()

    def update_root(self, action):
        if action not in self.root.children:
            raise ValueError("Action not found in root children.")

        state = self.get_state(self.root.children[action])
        self.root = self.root.children[action]
        self.root.parent = None
        self.root.action = None
        self.root_state = state

        return state

    def run(self, num_simulations=100, exploration_factor=1.0) -> torch.Tensor:
        with torch.no_grad():
            for _ in range(num_simulations):
                self.iteration(self.root, exploration_factor=exploration_factor)

        return self.get_mcts_policy(self.root, temperature=1.0)

    def iteration(self, current_node: MCTSNode, exploration_factor=1.0) -> float:
        current_node.increase_visit_count()

        if current_node.uuid in self.terminal_nodes or self.is_terminal(current_node):
            self.terminal_nodes.add(current_node.uuid)
            return 0.0

        # If the node is a leaf, evaluate it using the model
        if current_node.get_visit_count() == 1:
            state = self.get_state(current_node)
            policy, v = self.model.predict(state)
            current_node.policy = policy[
                0
            ]  # todo: the current implementation only supports batch size of 1
            current_node.huristic = v.item()
            return current_node.huristic

        # Select the action with the highest UCB score
        best_score = float("inf")
        best_action = -1
        for action in self.actions:
            child = current_node.get_child(action)
            ucb_score = self.get_ucb_score(child, current_node, exploration_factor)
            if ucb_score < best_score:
                best_score = ucb_score
                best_action = action

        # Recur on the selected child
        child = current_node.get_child(best_action)
        new_huristic = self.iteration(
            child, exploration_factor=exploration_factor
        ) + self.game.get_action_cost(self.get_state(current_node), best_action)

        # NOTE: experiment with using average instead of min (min should make huristic more admissable and average should be closer to expected value)
        new_huristic = (
            min(new_huristic, current_node.huristic)
            if current_node.huristic is not None
            else new_huristic
        )
        current_node.huristic = new_huristic

        return new_huristic

    def is_terminal(self, node: MCTSNode) -> bool:
        state = self.get_state(node)
        return self.game.is_terminal(state)

    def get_state(self, node: MCTSNode) -> torch.Tensor:
        state = self.state_cache.get(node.uuid)
        if state is not None:
            return state

        if node.parent is None:
            return self.root_state

        if node.action is None:
            raise ValueError("Action must be set for non-root nodes.")

        return self.game.get_next_state(self.get_state(node.parent), node.action)

    def get_ucb_score(
        self, child: MCTSNode, parent: MCTSNode, exploration_factor=1.0
    ) -> float:
        if parent.policy is None:
            raise ValueError("Parent node policy is not set.")

        return self.get_Q(child, parent) - exploration_factor * parent.policy[
            child.action
        ] * sqrt(parent.get_visit_count()) / (1 + child.get_visit_count())

    def get_Q(self, child: MCTSNode, parent: MCTSNode) -> float:
        h_min = min([child.huristic for child in parent.children.values()])
        h_max = max([child.huristic for child in parent.children.values()])

        return (child.huristic - h_min) / (
            h_max - h_min + 1e-4
        )  # Normalize Q value to [0, 1]

    @staticmethod
    def get_mcts_policy(node: MCTSNode, temperature=1.0) -> torch.Tensor:
        visit_counts = torch.tensor(
            [child.get_visit_count() for child in node.children.values()],
            dtype=torch.float32,
        )
        logits = visit_counts ** (1 / temperature)
        probs = logits / logits.sum()
        return probs
