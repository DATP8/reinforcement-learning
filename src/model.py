
import torch.nn as nn
import torch.nn.functional as F
import torch

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
