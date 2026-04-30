import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData
from torch_geometric.nn import GCNConv, global_mean_pool

from src.model import BiCircuitGNNDense
from src.states.dense_circuit_graph import DenseCircuitGraph


class SimpleExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)

        input_dim = observation_space.shape[0] * observation_space.shape[1]

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        return self.net(obs)


class SimpleGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, batch):
        # x: (B, N, F)
        # edge_index: (B, 2, E)

        B, N, F = x.shape

        # Flatten nodes
        x = x.view(B * N, F)

        # Fix edge indices
        edge_index_list = []
        for i in range(B):
            ei = edge_index[i]  # (2, E)
            ei = ei + i * N
            edge_index_list.append(ei)

        edge_index = torch.cat(edge_index_list, dim=1)

        valid = (edge_index[0] != 0) | (edge_index[1] != 0)
        edge_index = edge_index[:, valid]

        # Build batch vector
        batch = torch.arange(B, device=x.device).repeat_interleave(N)

        # GNN
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()

        return global_mean_pool(x, batch)


class HybridExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)

        matrix_shape = observation_space["matrix"].shape
        matrix_dim = matrix_shape[0] * matrix_shape[1]

        # MLP for your matrix
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(matrix_dim, 128),
            nn.ReLU(),
        )

        # GNN for graph
        self.gnn = SimpleGNN(in_dim=3, hidden_dim=64)

        # Final projection
        self.final = nn.Sequential(
            nn.Linear(128 + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        # --- matrix ---
        matrix = obs["matrix"]
        matrix_feat = self.mlp(matrix)

        # --- graph ---
        x = obs["graph_x"]
        edge_index = obs["graph_edge_idx"].long()

        # Create batch (single graph → all zeros)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        graph_feat = self.gnn(x, edge_index, batch)

        # --- combine ---
        combined = torch.cat([matrix_feat, graph_feat], dim=-1)

        return self.final(combined)


# class DenseDagExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space, features_dim=128):
#         super().__init__(observation_space, features_dim)
#
#         self.model = BiCircuitGNNDense(6, ppo_mode=True)
#
#     def forward(self, obs):
#         # --- matrix ---
#         matrix = obs["matrix"]
#         x = obs["graph_x"]
#         edge_index = obs["graph_edge_index"]
#         edge_attr = obs["graph_edge_attr"]
#         print("[Extractor] Shapes:")
#         print(x.shape)
#         print(edge_index.shape)
#         print(edge_attr.shape)
#         data = DenseCircuitGraph.from_tensors(x,edge_index,edge_attr)
#         print("[Extractor] Graph Shapes:")
#         print(data.x.shape)
#         print(data.edge_index.shape)
#         print(data.edge_attr.shape)
#         return self.model(data)


class DenseDagExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int):
        super().__init__(observation_space, features_dim=features_dim)
        self.gnn = BiCircuitGNNDense(6, ppo_mode=True)

    def forward(self, obs: dict) -> torch.Tensor:
        # All values arrive as (N, ...) tensors from PPO
        N = obs["graph_x"].shape[0]

        data_list: list[BaseData] = []
        for i in range(N):
            num_nodes = int(obs["graph_num_nodes"][i, 0].item())
            num_edges = int(obs["graph_num_edges"][i, 0].item())

            x = obs["graph_x"][i, :num_nodes]  # (num_nodes, n_qubits*2)
            ei = obs["graph_edge_index"][i, :num_edges].t().long()  # (2, num_edges)
            ea = obs["graph_edge_attr"][
                i, :num_edges
            ].float()  # (num_edges, n_qubits+1)

            data_list.append(
                DenseCircuitGraph.from_tensors(x=x, edge_index=ei, edge_attr=ea)
            )

        batch = Batch.from_data_list(data_list).to(obs["graph_x"].device)
        return self.gnn(batch)  # (N, hidden_dim)
