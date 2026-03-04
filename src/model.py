from abc import abstractmethod
from abc import ABC
from cnot_circuit import generate_random_circuit
from tornado import gen
from cnot_circuit import CNOTCircuit
import torch.nn as nn
import torch.nn.functional as F
import torch

# class AttentionModel(nn.Module):
#     def __init__(self, n_qubits, d_model, nhead, num_layers):
#         super().__init__()
#         self.n_qubits = n_qubits
#         self.d_model = d_model
#         self.nhead = nhead
#         self.num_layers = num_layers

#         self.scale = nn.Linear(n_qubits, d_model)
#         self.pos_encoding = PositionalEncoding(d_model)
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
#         self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
#     def forward(self, x):
#         x = F.relu(self.scale(x))
#         x = x.view(x.size(0), 1, self.d_model)
#         x = self.pos_encoding(x)
#         out = self.transformer_encoder(x)
#         return out

class PVModel(nn.Module):
    def __init__(self, n_qubits: int, horizon: int, n_actions: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.horizon = horizon
        self.n_actions = n_actions
        
        
    @abstractmethod
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """_summary_
        Args:
            x (torch.Tensor): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (value, logits, probabilities)
        """
        raise NotImplementedError("Subclasses must implement this method.")
    

class Model(PVModel):
    def __init__(self, n_qubits, horizon, n_actions):
        super().__init__(n_qubits, horizon, n_actions)
        self.conv1 = nn.Conv1d(n_qubits * n_qubits, n_qubits * 4, kernel_size=3, padding=1)
        self.linear1 = nn.Linear(4 * n_qubits * horizon, n_actions * 4)
        self.distribution = nn.Linear(n_actions * 4, n_actions)
        self.estimate = nn.Linear(n_actions * 4, 1)
    
    

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.flatten(start_dim=1, end_dim=2)
        x = self.conv1(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = F.relu(x)
        logits = self.distribution(x)
        probs = F.softmax(logits, dim=-1)
        v = F.softplus(self.estimate(x))
        
        return probs, v
    
    def predict(self, x: torch.Tensor):
        return self.forward(x)
    
class ValueModel(nn.Module):
    def __init__(self, n_qubits, horizon, n_actions):
        super().__init__()
        self.conv1 = nn.Conv1d(n_qubits * n_qubits, n_qubits * 4, kernel_size=3, padding=1)
        self.linear1 = nn.Linear(4 * n_qubits * horizon, n_actions * 4)
        self.estimate = nn.Linear(n_actions * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1, end_dim=2)
        x = self.conv1(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = F.relu(x)
        #v = F.softplus(self.estimate(x))
        v = torch.exp(self.estimate(x))
        
        return v
    
    def predict(self, x: torch.Tensor):
        return self.forward(x)


class RetardModel(PVModel):
    def __init__(self, n_qubits, horizon, n_actions):
        super().__init__(n_qubits=0, horizon=0, n_actions=n_actions)
        self.n_actions = n_actions
        
    
    def predict(self, x: torch.Tensor):
        probs = torch.ones((x.size(0), self.n_actions)) / self.n_actions
        v = torch.zeros((x.size(0), 1))
        return probs, v


if __name__ == "__main__":
    horizon = 10
    model = ValueModel(n_qubits=5, horizon=10, n_actions=20)
    
    torch.save(model.state_dict(), "models/test/pik.pt")

