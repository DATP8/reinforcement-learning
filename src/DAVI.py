from cnot_circuit import CNOTCircuit
from basegame import BaseGame
from torch import nn
import torch
import random

class DAVI:
    def __init__(self, training_model, evaluation_model, n_qubits, horizon, game: BaseGame[torch.Tensor]):
        self.train_model = training_model
        self.evaluation_model = evaluation_model
        self.n_qubits = n_qubits
        self.horizon = horizon
        self.game = game
        
    def train(self, batchsize=100, num_iterations=1000, update_frequency=10, K=100, loss_threshold=0.01):
        optimizer = torch.optim.Adam(self.train_model.parameters())
        mse_loss = nn.MSELoss()
        dificulty = 1
        
        for iteration in range(num_iterations):
            X = self.get_random_states(batchsize, dificulty)
            y = torch.zeros((batchsize, 1))
            
            for i in range(batchsize):
                min_cost = float('inf')
                for action in self.game.get_possible_actions(X[i]):
                    next_state = self.game.get_next_state(X[i], action)
                    if self.game.is_terminal(next_state):
                        y[i] = 1.0
                        break
                    
                    cost = self.game.get_action_cost(X[i], action) + self.evaluation_model.predict(next_state)[1].item()
                    if cost < min_cost:
                        min_cost = cost
                        y[i] = cost
            
            optimizer.zero_grad()
            loss = mse_loss(self.train_model(X), y)
            loss.backward()
            optimizer.step()
            
            print(f"Iteration {iteration}, Difficulty: {dificulty}, Loss: {loss.item():.4f}")
            
            if iteration % update_frequency == 0 and loss.item() < loss_threshold:
                self.evaluation_model.load_state_dict(self.train_model.state_dict())
                dificulty += 1

        
    def get_random_states(self, batchsize, dificulty):
        states = torch.zeros((batchsize, self.n_qubits, self.n_qubits, self.horizon))
        for i in range(batchsize):
            n_gates = random.randint(1, dificulty)
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
        state, _ = self.game.prune(state)
        
        if self.game.is_terminal(state):
            return self.generate_random_circuit(n_gates)
            
        return state


if __name__ == "__main__":
    n_qubits = 6
    horizon = 100
    topology = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]