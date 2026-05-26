"""
bipartite_extractor.py
----------------------
GNN feature extractor operating on the action-layer bipartite graph.

The GNN produces per-action-node embeddings which are passed to the policy
head in place of (or alongside) the flat A×H matrix. This preserves the
direct action↔feature correspondence the matrix has, while adding relational
context from the bipartite graph structure.

Architecture
------------

    Bipartite graph
         │
    HeteroGNN (GATConv × num_layers)
    — message passing between action ↔ layer nodes —
         │
    Action node embeddings  (num_active_swaps, gnn_out)
         │
    [Optional: concatenate flat matrix row  (horizon,)]
         │
    Per-action MLP  →  scalar logit per action
         │                    (num_active_swaps, action_out)
         │
    [Optional: concatenate coupling/interaction GNN global features]
         │
    Flatten  →  Final MLP  →  features_dim

Ablation flags
--------------
    use_bipartite : bool   — include the bipartite GNN branch
    use_matrix    : bool   — concatenate raw matrix row to each action embedding

At least one must be True. When use_bipartite=False and use_matrix=True this
reduces to a per-action MLP on the raw matrix rows, similar to your original
model but with an explicit per-action structure.
"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.nn import GATConv

from src.ppo_models.bipartite.graph_obs import EDGE_F, NODE_F

# ---------------------------------------------------------------------------
# Bipartite GNN
# ---------------------------------------------------------------------------


class BipartiteGNN(nn.Module):
    """
    Runs GATConv message passing on the combined action+layer node graph.

    Uses a node-type embedding to let the GNN distinguish action nodes from
    layer nodes, since they carry different semantic meaning despite sharing
    the same feature width (zero-padded).

    After message passing, only action node embeddings are returned —
    shape (B * num_active_swaps, gnn_out).
    """

    def __init__(
        self,
        node_in: int = NODE_F,
        edge_in: int = EDGE_F,
        hidden: int = 64,
        heads: int = 4,
        out_dim: int = 64,
        num_layers: int = 2,
        num_node_types: int = 2,
        type_embed_dim: int = 8,
    ):
        super().__init__()

        # Small embedding to distinguish action vs layer nodes
        self.type_embed = nn.Embedding(num_node_types, type_embed_dim)

        actual_in = node_in + type_embed_dim

        self.convs = nn.ModuleList()
        self.acts = nn.ModuleList()

        in_dim = actual_in
        for i in range(num_layers):
            is_last = i == num_layers - 1
            out = out_dim if is_last else hidden
            heads_here = 1 if is_last else heads
            concat = False if is_last else True
            self.convs.append(
                GATConv(
                    in_dim,
                    out,
                    heads=heads_here,
                    edge_dim=edge_in,
                    concat=concat,
                    dropout=0.0,
                )
            )
            self.acts.append(nn.ELU())
            in_dim = out * heads_here if concat else out

    def forward(
        self,
        x: torch.Tensor,  # (B*N_total, node_in)
        node_type: torch.Tensor,  # (B*N_total,)  int64
        edge_index: torch.Tensor,  # (2, B*E)
        edge_attr: torch.Tensor,  # (B*E, edge_in)
        num_active_swaps: int,  # action nodes per graph
        B: int,  # batch size
    ) -> torch.Tensor:
        # Augment node features with type embedding
        type_emb = self.type_embed(node_type)  # (B*N_total, type_embed_dim)
        x = torch.cat([x, type_emb], dim=-1)  # (B*N_total, node_in + type_embed_dim)

        for conv, act in zip(self.convs, self.acts):
            x = act(conv(x, edge_index, edge_attr))

        # Extract only action node embeddings
        # Action nodes are at positions [i*N_total : i*N_total + num_active_swaps]
        # for each graph i in the batch
        N_total = x.shape[0] // B

        # Reshape and extract only the first num_active_swaps nodes per batch
        x_reshaped = x.view(B, N_total, -1)
        action_rows = x_reshaped[
            :, :num_active_swaps, :
        ]  # (B, num_active_swaps, gnn_out)

        # Flatten back to match original output shape
        action_rows = action_rows.contiguous().view(B * num_active_swaps, -1)

        return action_rows


# ---------------------------------------------------------------------------
# Batched bipartite graph helper
# ---------------------------------------------------------------------------


def _batch_bipartite(
    x_batch: torch.Tensor,  # (B, N_total, NODE_F)
    node_type_batch: torch.Tensor,  # (B, N_total)
    edge_index_batch: torch.Tensor,  # (B, 2, E)
    edge_attr_batch: torch.Tensor,  # (B, E, EDGE_F)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten batched graph tensors into PyG format."""
    B, N_total, F = x_batch.shape

    x_flat = x_batch.contiguous().view(B * N_total, F)
    node_type_flat = node_type_batch.contiguous().view(B * N_total)

    # Vectorized offset computation:
    # edge_index_batch is shaped (B, 2, E). We add i * N_total to each graph.
    offsets = (
        torch.arange(
            B, device=edge_index_batch.device, dtype=edge_index_batch.dtype
        ).view(B, 1, 1)
        * N_total
    )

    # Add offsets, transpose to (2, B, E) and then flatten to (2, B * E)
    edge_index = (edge_index_batch + offsets).transpose(0, 1).contiguous().view(2, -1)

    # Flatten edge_attr to (B * E, EDGE_F)
    edge_attr = edge_attr_batch.contiguous().view(-1, edge_attr_batch.shape[-1])

    return x_flat, node_type_flat, edge_index, edge_attr


# ---------------------------------------------------------------------------
# BipartiteExtractor
# ---------------------------------------------------------------------------


class BipartiteExtractor(BaseFeaturesExtractor):
    """
    Parameters
    ----------
    observation_space : gym.spaces.Dict
    features_dim      : final output dimensionality
    gnn_hidden        : hidden channels in intermediate GATConv layers
    gnn_heads         : attention heads (must divide gnn_hidden)
    gnn_out           : output channels per action node from the GNN
    gnn_layers        : number of GATConv layers (2 recommended)
    action_mlp_hidden : hidden dim of per-action MLP after GNN
    action_out        : per-action scalar summary dim before flatten
    use_bipartite     : include the bipartite GNN branch
    use_matrix        : concatenate raw matrix row to each action embedding
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
        gnn_hidden: int = 64,
        gnn_heads: int = 4,
        gnn_out: int = 64,
        gnn_layers: int = 2,
        action_mlp_hidden: int = 64,
        action_out: int = 16,
        use_bipartite: bool = True,
        use_matrix: bool = False,
    ):
        # print("Init bipartite graph")
        assert use_bipartite or use_matrix, "At least one branch must be enabled"
        super().__init__(observation_space, features_dim)

        self.use_bipartite = use_bipartite
        self.use_matrix = use_matrix

        matrix_shape: tuple = observation_space["matrix"].shape  # pyrefly: ignore  # (num_active_swaps, horizon)
        self.num_active_swaps = matrix_shape[0]
        self.horizon = matrix_shape[1]

        bipartite_x_shape: tuple = observation_space["bipartite_x"].shape  # pyrefly: ignore  # (N_total, NODE_F)
        self.N_total = bipartite_x_shape[0]  # num_active_swaps + horizon

        # Per-action input dim to the action MLP
        action_embed_in = 0
        if use_bipartite:
            action_embed_in += gnn_out
        if use_matrix:
            action_embed_in += self.horizon

        # Bipartite GNN
        if use_bipartite:
            self.bipartite_gnn = BipartiteGNN(
                node_in=NODE_F,
                edge_in=EDGE_F,
                hidden=gnn_hidden,
                heads=gnn_heads,
                out_dim=gnn_out,
                num_layers=gnn_layers,
            )

        # Per-action MLP: maps each action embedding → action_out scalar summary
        # Applied identically to each action node (shared weights)
        self.action_mlp = nn.Sequential(
            nn.Linear(action_embed_in, action_mlp_hidden),
            nn.ELU(),
            nn.Linear(action_mlp_hidden, action_out),
            nn.ELU(),
        )

        # Final MLP: flatten all action summaries → features_dim
        self.final_mlp = nn.Sequential(
            nn.Linear(self.num_active_swaps * action_out, features_dim),
            nn.ELU(),
            nn.Linear(features_dim, features_dim),
            nn.ELU(),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        B = obs["matrix"].shape[0]

        parts = []

        # ----------------------------------------------------------------
        # Bipartite GNN branch
        # ----------------------------------------------------------------
        if self.use_bipartite:
            x = obs["bipartite_x"].float()  # (B, N_total, NODE_F)
            nt = obs["bipartite_node_type"].long()  # (B, N_total)
            ei = obs["bipartite_edge_index"].long()  # (B, 2, E)
            ea = obs["bipartite_edge_attr"].float()  # (B, E, EDGE_F)

            x_flat, nt_flat, ei_flat, ea_flat = _batch_bipartite(x, nt, ei, ea)

            action_emb = self.bipartite_gnn(
                x_flat,
                nt_flat,
                ei_flat,
                ea_flat,
                num_active_swaps=self.num_active_swaps,
                B=B,
            )  # (B * num_active_swaps, gnn_out)

            # Reshape to (B, num_active_swaps, gnn_out)
            action_emb = action_emb.view(B, self.num_active_swaps, -1)
            parts.append(action_emb)

        # ----------------------------------------------------------------
        # Raw matrix branch (per action row)
        # ----------------------------------------------------------------
        if self.use_matrix:
            matrix = obs["matrix"].float()  # (B, num_active_swaps, horizon)
            parts.append(matrix)

        # ----------------------------------------------------------------
        # Per-action MLP  (shared weights across action dimension)
        # ----------------------------------------------------------------
        # Concatenate along feature dim: (B, num_active_swaps, action_embed_in)
        per_action = torch.cat(parts, dim=-1)

        # Apply shared MLP to each action independently
        # Flatten batch+action → (B * num_active_swaps, action_embed_in)
        per_action = per_action.view(B * self.num_active_swaps, -1)
        per_action = self.action_mlp(per_action)  # (B * A, action_out)
        per_action = per_action.view(B, -1)  # (B, num_active_swaps * action_out)

        return self.final_mlp(per_action)  # (B, features_dim)
