from random import shuffle
from collections import defaultdict
from model import Model
from cnot_circuit import CNOTCircuit
from swap_optimizer import SwapOptimizer
from model import PVModel
from mcts import MCTS
import random
import torch


class Trainer:
    def __init__(self, model: PVModel, topology: list[tuple[int, int]]):
        self.topology = topology
        self.model = model
        self.n_qubits = model.n_qubits
        self.horizon = model.horizon
        self.n_actions = model.n_actions

        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.01, weight_decay=1e-4
        )
        self.kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
        self.mse_loss = torch.nn.MSELoss()

        assert self.n_actions == len(topology), (
            "Number of actions must match the number of edges in the topology."
        )

    def train(
        self,
        num_episodes=1000,
        num_simulations=100,
        exploration_factor=1.0,
        mcts_cache_size=1000,
    ):

        difficulty = 1
        upper_bound = self.n_qubits * 2
        # alpha = 0.986
        alpha = 0.966
        # circuit = CNOTCircuit(self.n_qubits)
        # circuit.add_cnot(0, 2)
        # circuit.add_cnot(1, 2)
        # root_state = circuit.to_tensor(horizon=self.horizon)

        success_rate = 0.0
        for episode in range(num_episodes):
            mcts_policies = []
            state = self.generate_random_circuit(difficulty)
            # state = root_state.clone()

            game = SwapOptimizer(state, self.topology)
            state = game.get_initial_state()
            print("circuit depth:", game.circuit.depth())
            if game.is_terminal(state):
                print("Generated a trivial circuit. Skipping episode.")
                continue

            mcts = MCTS(game, self.model, cache_size=mcts_cache_size)
            success = 0.0
            for _ in range(upper_bound):
                mcts_policy = mcts.run(
                    num_simulations=num_simulations,
                    exploration_factor=exploration_factor,
                ).unsqueeze(0)
                mcts_policies.append((state, mcts_policy))

                action = torch.multinomial(mcts_policy[0], num_samples=1).item()

                state = mcts.update_root(action).clone()

                if mcts.is_terminal(mcts.root):
                    print("Episode succeeded!: steps taken:", len(mcts_policies))
                    success = 1.0
                    self.update_model(mcts_policies)
                    break

            success_rate = alpha * success_rate + (1 - alpha) * success
            if success_rate > 0.8:
                difficulty += 1
                upper_bound += self.n_qubits * 2
                success_rate = 0.0

            if episode % 1 == 0:
                print(
                    f"Episode {episode}, Success Rate: {success_rate:.2f}, Difficulty: {difficulty}"
                )

    def update_model(self, mcts_policies: list[tuple[torch.Tensor, torch.Tensor]]):
        data = self.filter_mcts_policies(mcts_policies)
        shuffle(data)

        for state, policy, value in data:
            self.optimizer.zero_grad()
            predicted_probs, predicted_value = self.model.predict(state)

            print(policy)
            print(predicted_probs)
            print(value)
            print(predicted_value)
            value_loss = self.mse_loss(predicted_value, value)
            policy_loss = self.kl_loss(torch.log(predicted_probs), policy)
            loss = value_loss + 0.1 * policy_loss
            loss.backward()
            self.optimizer.step()

    def filter_mcts_policies(
        self, mcts_policies: list[tuple[torch.Tensor, torch.Tensor]]
    ):
        data = []

        shortests_paths = defaultdict(lambda: float("inf"))
        prev_state_id = mcts_policies[-1][0].__hash__()
        shortests_paths[prev_state_id] = 1.0

        for state, pi in mcts_policies[::-1]:
            state_id = state.__hash__()
            h = min(shortests_paths[state_id], shortests_paths[prev_state_id] + 1.0)
            data.append((state, pi, torch.tensor([h]).unsqueeze(0)))

        return data

    def generate_random_circuit(self, n_gates: int):
        qc = CNOTCircuit(self.n_qubits)

        for i in range(n_gates):
            q1, q2 = random.sample(range(self.n_qubits), 2)
            while q1 == q2:
                q2 = random.choice(range(self.n_qubits))

            qc.add_cnot(q1, q2)

        return qc.to_tensor(horizon=self.horizon)


if __name__ == "__main__":
    topology = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    model = Model(n_qubits=6, horizon=100, n_actions=len(topology))
    trainer = Trainer(model, topology)
    trainer.train(
        num_episodes=100000,
        num_simulations=100,
        exploration_factor=1.0,
        mcts_cache_size=1000,
    )
