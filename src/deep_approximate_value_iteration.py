from .model import ValueModel
from .swap_optimizer import SwapOptimizer
from .cnot_circuit import CNOTCircuit
from .basegame import BaseGame
from torch import nn
from itertools import product
from qiskit.transpiler import CouplingMap
from multiprocessing import Process, set_start_method
from datetime import datetime

import sys
import torch
import os
import random
import matplotlib

matplotlib.use("TkAgg")
from .benchmark.benchmarker import Benchmarker

BENCHMARK_PATH_RESULTS = "src/benchmark/results"

class DAVI:
    def __init__(self, training_model, evaluation_model, n_qubits, horizon, game: BaseGame[torch.Tensor]):
        self.train_model = training_model
        self.evaluation_model = evaluation_model
        self.n_qubits = n_qubits
        self.horizon = horizon
        self.game = game
        
    def train(self, batchsize=100, initial_difficulty=1, num_iterations=1000, update_frequency=10, max_difficulty=100, loss_threshold=0.06):
        p_list = []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        self.train_model.to(device)

        now = datetime.now()
        start_time = now.strftime("%Y-%m-%dT%H:%M:%S")

        optimizer = torch.optim.Adam(self.train_model.parameters())
        mse_loss = nn.MSELoss()
        difficulty = initial_difficulty
        last_model_path = ""
        
        for iteration in range(num_iterations):
            X = self.get_random_states(batchsize, difficulty)
            y = torch.zeros((batchsize, 1))
            
            for i in range(batchsize):
                min_cost = float('inf')
                for action in self.game.get_possible_actions(X[i]):
                    next_state = self.game.get_next_state(X[i], action)
                    if self.game.is_terminal(next_state):
                        y[i] = self.game.get_action_cost(X[i], action)
                        break
                    with torch.no_grad():
                        cost = self.game.get_action_cost(X[i], action) + self.evaluation_model.predict(next_state.unsqueeze(0)).item()
                    if cost < min_cost:
                        min_cost = cost
                        y[i] = cost
            
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = mse_loss(self.train_model(X), y)
            loss.backward()
            optimizer.step()
            
            print(f"Difficulty: {difficulty}, Iteration {iteration}, Loss: {loss.item():.4f}")
            
            if iteration % update_frequency == 0 and loss.item() < loss_threshold:
                self.evaluation_model.load_state_dict(self.train_model.state_dict())
                difficulty = min(max_difficulty, 1 + difficulty)

                last_model_path = f"models/davi/difficulty{difficulty}_iteration{iteration}.pt"
                tmp = last_model_path + ".tmp"
                torch.save(self.train_model.state_dict(), tmp)
                os.rename(tmp, last_model_path)

                topology = self.game.get_topology()
                p_list.append(Process(target=bench_process, args=(self.n_qubits, last_model_path, difficulty, topology, start_time)))
                p_list[-1].start()
         
        for p in p_list:
            p.join()
                
        return last_model_path

        
    def get_random_states(self, batchsize, difficulty):
        states = torch.zeros((batchsize, self.n_qubits, self.n_qubits, self.horizon))
        for i in range(batchsize):
            n_gates = random.randint(1, difficulty)
            states[i] = self.generate_random_circuit(n_gates)

        return states

    def generate_random_circuit(self, n_gates: int):
        qc = CNOTCircuit(self.n_qubits)
        
        for i in range(n_gates):
            q1, q2 = random.sample(range(self.n_qubits), 2)
            while q1 == q2:
                q2 = random.choice(range(self.n_qubits))
            
            qc.add_cnot(q1, q2)
            
        state = qc.to_tensor(horizon=self.horizon)
        
        if self.game.is_terminal(state):
            return self.generate_random_circuit(n_gates)
            
        return state

def bench_process(n_qubits, rel_model_path, difficulty, topology, start_time_str=""):

    o = sys.stdout

    bench_iterations = 1

    initial_layouts = ["qiskit"]
    forward_backward = ["none", "sabre"]
    final_routers = ["sabre", "rl"]

    configs = list(product(
        initial_layouts,
        forward_backward,
        final_routers
    ))

    coupling_map = CouplingMap(topology)
    coupling_map.make_symmetric()


    cwd = os.getcwd()
    model_path = os.path.join(cwd, rel_model_path)
    out_path = os.path.join(cwd, BENCHMARK_PATH_RESULTS, f"benchmark{start_time_str}.md")
    
    bench = Benchmarker(model_path, n_qubits, difficulty, coupling_map)

    with open(out_path, 'a') as f:
        sys.stdout = f
        print("\n#", rel_model_path)
        bench.run_rand_benchmarks(configs, bench_iterations)

    sys.stdout = o
    
if __name__ == "__main__":
    set_start_method("spawn", force=True)

    n_qubits = 6
    horizon = 100
    topology = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    game = SwapOptimizer(n_qubits, horizon, topology)
    training_model = ValueModel(n_qubits, horizon, len(topology))
    evaluation_model = ValueModel(n_qubits, horizon, len(topology))
    #training_model.load_state_dict(torch.load("models/value_model_deep_cube_a_exp_relu/difficulty5_iteration320.pt"))
    #evaluation_model.load_state_dict(torch.load("models/value_model_deep_cube_a_exp_relu/difficulty5_iteration320.pt"))
    #training_model.load_state_dict(torch.load("models/value_model_deep_cube_a_exp_relu/difficulty5_iteration320.pt"))
    #evaluation_model.load_state_dict(torch.load("models/value_model_deep_cube_a_exp_relu/difficulty5_iteration320.pt"))
    
    trainer = DAVI(training_model, evaluation_model, n_qubits, horizon, game)
    
    path = trainer.train(batchsize=1000, initial_difficulty=100000, num_iterations=5, update_frequency=1, max_difficulty=1000, loss_threshold=0.08)
