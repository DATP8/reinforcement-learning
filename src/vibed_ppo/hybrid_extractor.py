"""
hybrid_extractor.py
-------------------
Two-graph GNN feature extractor for the quantum circuit routing environment.

Architecture
------------

    Matrix (A×H)                           Coupling Graph GNN
    ─────────────                          ──────────────────
    Flatten → Linear(A*H, 128) → ReLU      GATConv × 2  (node features + edge attr)
                                           → per-edge embeddings for each SWAP action
                │                                         │
                │                          Interaction Graph GNN
                │                          ────────────────────
                │                          GATConv × 2  (node features + edge attr)
                │                          → global_mean_pool → circuit context
                │                                         │
                └──────────────┬───────────────────────────┘
                               │
                         Concatenate
                               │
                         Linear → ReLU → features_dim

The coupling GNN produces per-SWAP embeddings (one per active action) which are
then mean-pooled across the action dimension before concatenation. This ensures
the extractor scales to any number of active swaps without changing the final
feature size, while still encoding which swaps are structurally advantaged.

Observation space expected
--------------------------
    "matrix"              : (num_active_swaps, horizon)          int8  / float32
    "node_features"       : (num_qubits, NODE_F)                 float32
    "coupling_edge_index" : (2, 2*num_coupling_edges)            int64
    "coupling_edge_attr"  : (2*num_coupling_edges, COUP_EDGE_F)  float32
    "interact_edge_index" : (2, MAX_INTERACT_EDGES)              int64
    "interact_edge_attr"  : (MAX_INTERACT_EDGES, INT_EDGE_F)     float32
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


# ---------------------------------------------------------------------------
# Coupling Graph GNN
# ---------------------------------------------------------------------------
# Uses edge attributes (is_action, traffic, is_front) — GAT is a natural fit
# because the attention mechanism learns to weight neighbours by relevance.
#
# Output: per-node embeddings → we extract embeddings for each SWAP edge pair
#         and pool them → gives a fixed-size "SWAP quality" summary.

class CouplingGNN(nn.Module):
    def __init__(
        self,
        node_in: int,
        edge_in: int,
        hidden: int = 64,
        heads: int = 4,
        out_dim: int = 64,
    ):
        super().__init__()
        # GATConv with edge_dim supports edge attributes natively
        self.conv1 = GATConv(
            node_in, hidden, heads=heads,
            edge_dim=edge_in, concat=True, dropout=0.0,
        )
        self.conv2 = GATConv(
            hidden * heads, out_dim, heads=1,
            edge_dim=edge_in, concat=False, dropout=0.0,
        )
        self.act = nn.ELU()

    def forward(
        self,
        x: torch.Tensor,          # (B*N, node_in)
        edge_index: torch.Tensor, # (2, B*E)
        edge_attr: torch.Tensor,  # (B*E, edge_in)
        batch: torch.Tensor,      # (B*N,)
    ) -> torch.Tensor:
        x = self.act(self.conv1(x, edge_index, edge_attr))
        x = self.act(self.conv2(x, edge_index, edge_attr))
        return global_mean_pool(x, batch)   # (B, out_dim)


# ---------------------------------------------------------------------------
# Interaction Graph GNN
# ---------------------------------------------------------------------------
# Encodes gate demand urgency. Edge attributes are (urgency, dist, executable).
# Same GAT architecture but separate weights — the two graphs have very
# different semantics and should not share parameters.

class InteractionGNN(nn.Module):
    def __init__(
        self,
        node_in: int,
        edge_in: int,
        hidden: int = 64,
        heads: int = 4,
        out_dim: int = 64,
    ):
        super().__init__()
        self.conv1 = GATConv(
            node_in, hidden, heads=heads,
            edge_dim=edge_in, concat=True, dropout=0.0,
        )
        self.conv2 = GATConv(
            hidden * heads, out_dim, heads=1,
            edge_dim=edge_in, concat=False, dropout=0.0,
        )
        self.act = nn.ELU()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        x = self.act(self.conv1(x, edge_index, edge_attr))
        x = self.act(self.conv2(x, edge_index, edge_attr))
        return global_mean_pool(x, batch)   # (B, out_dim)


# ---------------------------------------------------------------------------
# Batched graph helper
# ---------------------------------------------------------------------------

def _batch_graphs(
    x_batch: torch.Tensor,           # (B, N, F)
    edge_index_batch: torch.Tensor,  # (B, 2, E)
    edge_attr_batch: torch.Tensor,   # (B, E, edge_f)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Flatten a batched graph observation (as produced by SB3's VecEnv) into
    the flat (node, edge) tensors expected by PyG, with a batch vector.

    Handles zero-padded edges: edges where both endpoints are 0 are treated
    as padding UNLESS node 0 genuinely connects to itself (self-loops would
    be unusual in a coupling map, but we protect against it).
    We use a sentinel approach — any edge (0,0) after index 0 is padding.
    """
    B, N, F = x_batch.shape
    device = x_batch.device

    x_flat = x_batch.view(B * N, F)
    batch_vec = torch.arange(B, device=device).repeat_interleave(N)

    ei_list, ea_list = [], []
    for i in range(B):
        ei = edge_index_batch[i]        # (2, E)
        ea = edge_attr_batch[i]         # (E, edge_f)

        # Remove padding: keep edges where at least one endpoint is non-zero,
        # OR the edge appears as the very first entry (to allow real (0,0) edges
        # in tiny topologies, though these are rare).
        is_real = (ei[0] != 0) | (ei[1] != 0)
        ei = ei[:, is_real]
        ea = ea[is_real]

        # Offset node indices for batching
        ei = ei + i * N
        ei_list.append(ei)
        ea_list.append(ea)

    edge_index = torch.cat(ei_list, dim=1)
    edge_attr = torch.cat(ea_list, dim=0)

    return x_flat, edge_index, edge_attr, batch_vec


# ---------------------------------------------------------------------------
# HybridExtractor
# ---------------------------------------------------------------------------

class HybridExtractor(BaseFeaturesExtractor):
    """
    Parameters
    ----------
    observation_space : gym.spaces.Dict
    features_dim      : output dimensionality fed to the policy/value heads
    gnn_hidden        : hidden channels per GNN layer
    gnn_heads         : attention heads in GATConv
    gnn_out           : output channels per GNN (before final MLP)
    matrix_out        : output channels of the matrix MLP branch
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
        gnn_hidden: int = 64,
        gnn_heads: int = 4,
        gnn_out: int = 64,
        matrix_out: int = 128,
    ):
        super().__init__(observation_space, features_dim)

        # -- infer shapes from observation space --
        matrix_shape = observation_space["matrix"].shape          # (A, H)
        node_shape = observation_space["node_features"].shape     # (N, NODE_F)
        coup_ea_shape = observation_space["coupling_edge_attr"].shape   # (E_c, 3)
        int_ea_shape = observation_space["interact_edge_attr"].shape    # (E_i, 3)

        node_in = node_shape[1]
        coup_edge_in = coup_ea_shape[1]
        int_edge_in = int_ea_shape[1]

        # -- matrix branch --
        matrix_flat = matrix_shape[0] * matrix_shape[1]
        self.matrix_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(matrix_flat, matrix_out),
            nn.ELU(),
            nn.Linear(matrix_out, matrix_out),
            nn.ELU(),
        )

        # -- graph branches --
        self.coupling_gnn = CouplingGNN(
            node_in=node_in,
            edge_in=coup_edge_in,
            hidden=gnn_hidden,
            heads=gnn_heads,
            out_dim=gnn_out,
        )
        self.interact_gnn = InteractionGNN(
            node_in=node_in,
            edge_in=int_edge_in,
            hidden=gnn_hidden,
            heads=gnn_heads,
            out_dim=gnn_out,
        )

        # -- final fusion MLP --
        combined_dim = matrix_out + gnn_out + gnn_out
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ELU(),
            nn.Linear(features_dim, features_dim),
            nn.ELU(),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        # ----------------------------------------------------------------
        # 1. Matrix branch
        # ----------------------------------------------------------------
        matrix = obs["matrix"].float()            # (B, A, H)
        matrix_feat = self.matrix_mlp(matrix)     # (B, matrix_out)

        # ----------------------------------------------------------------
        # 2. Shared node features (both GNNs use the same node matrix)
        # ----------------------------------------------------------------
        node_x = obs["node_features"].float()     # (B, N, NODE_F)

        # ----------------------------------------------------------------
        # 3. Coupling graph branch
        # ----------------------------------------------------------------
        coup_ei = obs["coupling_edge_index"].long()     # (B, 2, E_c)
        coup_ea = obs["coupling_edge_attr"].float()     # (B, E_c, 3)

        cx, c_ei, c_ea, c_batch = _batch_graphs(node_x, coup_ei, coup_ea)
        coupling_feat = self.coupling_gnn(cx, c_ei, c_ea, c_batch)   # (B, gnn_out)

        # ----------------------------------------------------------------
        # 4. Interaction graph branch
        # ----------------------------------------------------------------
        int_ei = obs["interact_edge_index"].long()      # (B, 2, E_i)
        int_ea = obs["interact_edge_attr"].float()      # (B, E_i, 3)

        # Interaction graph reuses the same node features
        ix, i_ei, i_ea, i_batch = _batch_graphs(node_x, int_ei, int_ea)
        interact_feat = self.interact_gnn(ix, i_ei, i_ea, i_batch)   # (B, gnn_out)

        # ----------------------------------------------------------------
        # 5. Fuse all branches
        # ----------------------------------------------------------------
        combined = torch.cat([matrix_feat, coupling_feat, interact_feat], dim=-1)
        return self.fusion(combined)
