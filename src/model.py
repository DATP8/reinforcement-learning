from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, GINEConv, global_add_pool

from src.states.circuit_graph import CircuitGraph

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
        self.conv1 = nn.Conv1d(
            n_qubits * n_qubits, n_qubits * 4, kernel_size=3, padding=1
        )
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
        self.conv1 = nn.Conv1d(
            n_qubits * n_qubits, n_qubits * 4, kernel_size=3, padding=1
        )
        self.linear1 = nn.Linear(4 * n_qubits * horizon, n_actions * 4)
        self.estimate = nn.Linear(n_actions * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1, end_dim=2)
        x = self.conv1(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = F.relu(x)
        # v = F.softplus(self.estimate(x))
        v = torch.exp(self.estimate(x))

        return v.squeeze(-1)

    def predict(self, x: torch.Tensor):
        return self.forward(x)


class ValueModelFlat(nn.Module):
    def __init__(self, n_qubits, horizon, n_actions):
        super().__init__()
        self.conv1 = nn.Conv1d(n_qubits, n_qubits * 4, kernel_size=3, padding=1)
        self.linear1 = nn.Linear(4 * n_qubits * horizon, n_actions * 4)
        self.estimate = nn.Linear(n_actions * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = F.relu(x)
        # v = F.softplus(self.estimate(x))
        v = torch.exp(self.estimate(x))

        return v.squeeze(-1)

    def predict(self, x: torch.Tensor):
        return self.forward(x)


class BiCircuitGNN(nn.Module):
    def __init__(self, n_qubits, hidden_dim=128):
        super().__init__()

        self.node_encoder = nn.Linear(n_qubits * 2, hidden_dim)
        self.edge_encoder = nn.Linear(n_qubits, hidden_dim)

        def make_mlp():
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        # forward DAG propagation
        self.f_conv1 = GINEConv(make_mlp(), flow="source_to_target")
        self.f_conv2 = GINEConv(make_mlp(), flow="source_to_target")
        self.f_bn1 = BatchNorm(hidden_dim)
        self.f_bn2 = BatchNorm(hidden_dim)

        # backward propagation
        self.b_conv1 = GINEConv(make_mlp(), flow="target_to_source")
        self.b_conv2 = GINEConv(make_mlp(), flow="target_to_source")
        self.b_bn1 = BatchNorm(hidden_dim)
        self.b_bn2 = BatchNorm(hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, data: Data):

        x = self.node_encoder(data.x)
        edge_attr = self.edge_encoder(data.edge_attr)

        edge_index = data.edge_index
        batch = data.batch

        # forward pass
        xf = F.relu(self.f_bn1(self.f_conv1(x, edge_index, edge_attr)))
        xf = F.relu(self.f_bn2(self.f_conv2(xf, edge_index, edge_attr)))

        # backward pass
        xb = F.relu(self.b_bn1(self.b_conv1(x, edge_index, edge_attr)))
        xb = F.relu(self.b_bn2(self.b_conv2(xb, edge_index, edge_attr)))

        # combine
        x = torch.cat([xf, xb], dim=1)

        x = global_add_pool(x, batch)

        out = self.head(x)

        return out.squeeze(-1)


class BiCircuitGNNDense(nn.Module):
    def __init__(self, n_qubits, hidden_dim=128):
        super().__init__()

        self.node_encoder = nn.Linear(n_qubits * 2, hidden_dim)
        self.edge_encoder = nn.Linear(n_qubits + 1, hidden_dim)

        def make_mlp():
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        # forward DAG propagation
        self.f_conv1 = GINEConv(make_mlp(), flow="source_to_target")
        self.f_conv2 = GINEConv(make_mlp(), flow="source_to_target")
        self.f_bn1 = BatchNorm(hidden_dim)
        self.f_bn2 = BatchNorm(hidden_dim)

        # backward propagation
        self.b_conv1 = GINEConv(make_mlp(), flow="target_to_source")
        self.b_conv2 = GINEConv(make_mlp(), flow="target_to_source")
        self.b_bn1 = BatchNorm(hidden_dim)
        self.b_bn2 = BatchNorm(hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, data: Data):

        x = self.node_encoder(data.x)
        edge_attr = self.edge_encoder(data.edge_attr)

        edge_index = data.edge_index
        batch = data.batch

        # forward pass
        xf = F.relu(self.f_bn1(self.f_conv1(x, edge_index, edge_attr)))
        xf = F.relu(self.f_bn2(self.f_conv2(xf, edge_index, edge_attr)))

        # backward pass
        xb = F.relu(self.b_bn1(self.b_conv1(x, edge_index, edge_attr)))
        xb = F.relu(self.b_bn2(self.b_conv2(xb, edge_index, edge_attr)))

        # combine
        x = torch.cat([xf, xb], dim=1)

        x = global_add_pool(x, batch)

        out = self.head(x)

        return out.squeeze(-1)


class RetardModel(PVModel):
    def __init__(self, n_qubits, horizon, n_actions):
        super().__init__(n_qubits=0, horizon=0, n_actions=n_actions)
        self.n_actions = n_actions

    def predict(self, x: torch.Tensor):
        probs = torch.ones((x.size(0), self.n_actions)) / self.n_actions
        v = torch.zeros((x.size(0), 1))
        return probs, v


if __name__ == "__main__":
    n_qubits = 6
    model = BiCircuitGNN(n_qubits)

    qc = QuantumCircuit(n_qubits)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(1, 2)
    qc.cx(1, 2)
    qc.cx(1, 2)
    qc.cx(1, 2)
    qc.cx(1, 2)
    qc.cx(1, 2)
    qc.cx(1, 2)
    qc.cx(1, 2)

    qc2 = QuantumCircuit(n_qubits)
    qc2.cx(0, 1)

    graph1 = CircuitGraph.from_circuit(qc)
    graph2 = CircuitGraph.from_circuit(qc2)

    data = next(x for x in DataLoader([graph1, graph2], batch_size=2))

    print(model(data))
