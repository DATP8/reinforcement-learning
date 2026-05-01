from src.states.dense_circuit_graph_state_handler import DenseCircuitGraphStateHandler
from src.model import BiCircuitGNNDense
from src.states.state_handler import StateHandler
from src.states.circuit_graph_state_handler import CircuitGraphStateHandler
from src.states.qtensor_state_handler import QtensorStateHandler
from src.model import ValueModel, BiCircuitGNN
from torch import nn

import torch
import matplotlib

from utils.to import To

matplotlib.use("TkAgg")


class DAVI[S: To]:
    def __init__(
        self,
        training_model: nn.Module,
        evaluation_model: nn.Module,
        state_handler: StateHandler[S],
    ):
        self.train_model = training_model
        self.evaluation_model = evaluation_model
        self.state_handler = state_handler

    def train(
        self,
        batchsize=100,
        initial_difficulty=1,
        num_iterations=1000,
        update_frequency=10,
        max_difficulty=100,
        loss_threshold=0.06,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        self.train_model.to(device)
        self.evaluation_model.to(device).eval()

        optimizer = torch.optim.Adam(self.train_model.parameters())
        mse_loss = nn.MSELoss()

        difficulty = initial_difficulty
        previous_states = None
        for iteration in range(num_iterations):
            states = self.state_handler.get_random_states_in_range_keep(
                batchsize, 1, difficulty, previous_states, 30
            )
            previous_states = states
            y = torch.full((batchsize, 1), float("inf")).squeeze(-1).to(device)

            next_states = []
            next_state_actions = []
            for i, state in enumerate(states):
                for action in self.state_handler.get_possible_actions(state):
                    next_state = self.state_handler.get_next_state(state, action)
                    if self.state_handler.is_terminal(next_state):
                        y[i] = self.state_handler.get_action_cost(state, action)
                        break
                    next_states.append(next_state)
                    next_state_actions.append((i, action))

            with torch.no_grad():
                next_state_values = self.evaluation_model(
                    self.state_handler.batch_states(next_states).to(device)
                )

            for state_index, (i, action) in enumerate(next_state_actions):
                y[i] = torch.min(
                    self.state_handler.get_action_cost(state, action)
                    + next_state_values[state_index],
                    y[i],
                )

            X = self.state_handler.batch_states(states).to(device)
            optimizer.zero_grad()
            loss = mse_loss(self.train_model(X), y)
            loss.backward()
            optimizer.step()

            print(
                f"Difficulty: {difficulty}, Iteration {iteration}, Loss: {loss.item():.4f}"
            )

            if iteration % update_frequency == 0 and loss.item() < loss_threshold:
                self.evaluation_model.load_state_dict(self.train_model.state_dict())
                difficulty = min(max_difficulty, 1 + difficulty)
                torch.save(
                    self.train_model.state_dict(),
                    f"models/davi/difficulty{difficulty}_iteration{iteration}.pt",
                )


def qtensor():
    n_qubits = 6
    horizon = 100
    topology = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    game = QtensorStateHandler(n_qubits, horizon, topology)
    training_model = ValueModel(n_qubits, horizon, len(topology))
    evaluation_model = ValueModel(n_qubits, horizon, len(topology))

    trainer = DAVI(training_model, evaluation_model, game)

    trainer.train(
        batchsize=1000,
        initial_difficulty=1,
        num_iterations=100000,
        update_frequency=10,
        max_difficulty=1000,
        loss_threshold=0.08,
    )


def graph():
    n_qubits = 6
    topology = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    game = DenseCircuitGraphStateHandler(n_qubits, topology)
    training_model = BiCircuitGNNDense(n_qubits)
    evaluation_model = BiCircuitGNNDense(n_qubits)

    trainer = DAVI(training_model, evaluation_model, game)

    trainer.train(
        batchsize=1000,
        initial_difficulty=1,
        num_iterations=100000,
        update_frequency=10,
        max_difficulty=1000,
        loss_threshold=0.08,
    )


if __name__ == "__main__":
    graph()
    # qtensor()
