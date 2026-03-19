from .routing.swap_inserter.swap_inserter import SwapInserter
from .states.tensor_state_handler import TensorStateHandler
from .states.state_handler import StateHandler
from .states.circuit_graph_state_handler import CircuitGraphStateHandler
from .states.qtensor_state_handler import QtensorStateHandler
from .model import ValueModel, BiCircuitGNN
from .routing.rl_routing_pass import RlRoutingPass
from .routing.bwas_router import BWASRouter

from itertools import product
from qiskit.transpiler import CouplingMap
from multiprocessing import Process, set_start_method
from datetime import datetime
from torch import nn

import sys
import torch
import os
import matplotlib


from .benchmark.benchmarker import Benchmarker

matplotlib.use("TkAgg")

BENCHMARK_PATH_RESULTS = "src/benchmark/results"


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
        p_list = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        self.train_model.to(device)
        self.evaluation_model.to(device).eval()

        now = datetime.now()
        start_time = now.strftime("%Y-%m-%dT%H:%M:%S")

        optimizer = torch.optim.Adam(self.train_model.parameters())
        mse_loss = nn.MSELoss()

        difficulty = initial_difficulty
        last_model_path = ""

        for iteration in range(num_iterations):
            states = self.state_handler.get_random_states_in_range(
                batchsize, 1, difficulty
            )
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

                last_model_path = (
                    f"models/davi/difficulty{difficulty}_iteration{iteration}.pt"
                )
                tmp = last_model_path + ".tmp"
                torch.save(self.train_model.state_dict(), tmp)
                os.rename(tmp, last_model_path)

                topology = self.state_handler.get_topology()
                p_list.append(
                    Process(
                        target=bench_process,
                        args=(
                            self.state_handler.get_qubits(),
                            last_model_path,
                            difficulty,
                            topology,
                            start_time,
                        ),
                    )
                )
                p_list[-1].start()

        for p in p_list:
            p.join()

        torch.save(
            self.train_model.state_dict(),
            f"models/davi/difficulty{difficulty}_iteration{iteration}.pt",
        )

        return last_model_path


def bench_process(n_qubits, rel_model_path, difficulty, topology, start_time_str=""):

    o = sys.stdout

    bench_iterations = 1

    initial_layouts = ["qiskit"]
    forward_backward = ["none", "sabre"]
    final_routers = ["sabre", "rl"]

    configs = list(product(initial_layouts, forward_backward, final_routers))

    coupling_map = CouplingMap(topology)
    coupling_map.make_symmetric()

    cwd = os.getcwd()
    model_path = os.path.join(cwd, rel_model_path)
    out_path = os.path.join(
        cwd, BENCHMARK_PATH_RESULTS, f"benchmark{start_time_str}.md"
    )

    bench = Benchmarker(model_path, n_qubits, difficulty, coupling_map)

    with open(out_path, "a") as f:
        sys.stdout = f
        print("\n#", rel_model_path)
        bench.run_rand_benchmarks(configs, bench_iterations)

    sys.stdout = o


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
    game = CircuitGraphStateHandler(n_qubits, topology)
    # training_model = BiCircuitGNN(n_qubits, len(topology))
    # evaluation_model = BiCircuitGNN(n_qubits, len(topology))
    training_model = BiCircuitGNN(n_qubits)
    evaluation_model = BiCircuitGNN(n_qubits)

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
    set_start_method("spawn", force=True)

    from qiskit.transpiler.passes import TrivialLayout, SabreLayout, SabreSwap
    # graph()
    # qtensor()

    n_qubits = 6
    horizon = 100
    topology = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    game1 = TensorStateHandler(n_qubits, horizon, topology)
    model1 = ValueModel(n_qubits, horizon, len(topology))

    game2 = TensorStateHandler(n_qubits, horizon, topology)
    model2 = ValueModel(n_qubits, horizon, len(topology))

    path1 = "/home/vind/code/P8/project/reinforcement-learning/models/difficulty17_iteration95270.pt"
    path2 = "/home/vind/code/P8/project/reinforcement-learning/models/increment14_iteration77940_difficulty17.pt"
    model1.load_state_dict(torch.load(path1, map_location="cpu"))
    model2.load_state_dict(torch.load(path2, map_location="cpu"))

    bench_iterations = 100
    coupling_map = CouplingMap(topology)
    coupling_map.make_symmetric()

    initial_layouts = [TrivialLayout(coupling_map)]

    forward_backward = [SabreLayout(coupling_map)]

    swap_inserter = SwapInserter(coupling_map, n_qubits)
    router1 = BWASRouter(model1, game1)
    router2 = BWASRouter(model2, game2)

    final_routers = [
        SabreSwap(coupling_map),
        RlRoutingPass(router1, swap_inserter, "diff17"),
        RlRoutingPass(router2, swap_inserter, "incr14"),
    ]

    configs = list(product(initial_layouts, [None], final_routers))

    coupling_map.make_symmetric()

    bench = Benchmarker(n_qubits, 14, coupling_map)
    bench.run_rand_benchmarks(configs, bench_iterations, True)
