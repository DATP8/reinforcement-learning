"""
bipartite_integration_guide.py
-------------------------------
Minimal diffs to wire the bipartite graph into your existing environment
and training script.
"""

# ===========================================================================
# 1. OBSERVATION SPACE  (in __init__)
# ===========================================================================
#
# Add these four keys alongside your existing "matrix" key.
# Sizes are fully determined by num_active_swaps and horizon.

import numpy as np
from gymnasium import spaces

from src.ppo_models.bipartite.graph_obs import EDGE_F, NODE_F


def make_bipartite_observation_space(num_active_swaps, horizon, num_qubits):
    N_total = num_active_swaps + horizon
    E_total = 2 * num_active_swaps * horizon  # both directions, all pairs

    return spaces.Dict(
        {
            # existing key — keep as-is
            "matrix": spaces.Box(
                low=-2,
                high=2,
                shape=(num_active_swaps, horizon),
                dtype=np.int8,
            ),
            # new bipartite graph keys
            "bipartite_x": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(N_total, NODE_F),
                dtype=np.float32,
            ),
            "bipartite_node_type": spaces.Box(
                low=0,
                high=1,
                shape=(N_total,),
                dtype=np.int64,
            ),
            "bipartite_edge_index": spaces.Box(
                low=0,
                high=N_total - 1,
                shape=(2, E_total),
                dtype=np.int64,
            ),
            "bipartite_edge_attr": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(E_total, EDGE_F),
                dtype=np.float32,
            ),
        }
    )


# ===========================================================================
# 2. __init__  — precompute coupling degrees once
# ===========================================================================
#
# In your env __init__, after setting up _cmap_edges:
#
#   self._coupling_degrees = compute_coupling_degrees(
#       self._num_qubits, self._cmap_edges
#   )


# ===========================================================================
# 3. _update_obs
# ===========================================================================
#
# from bipartite_graph_obs import build_bipartite_obs
#
# def _update_obs(self):
#     matrix = self._build_matrix()   # existing — also sets self._active_swaps
#     action_mask = self.valid_action_mask()
#
#     layers = list(self._dag.layers())   # reuse if already computed in _build_matrix
#
#     bipartite = build_bipartite_obs(
#         matrix           = matrix,
#         active_swaps     = self._active_swaps,
#         cmap_edges       = self._cmap_edges,
#         num_qubits       = self._num_qubits,
#         action_mask      = action_mask,
#         swap_cancellation= self._obs.get("swap_cancellation",
#                               np.zeros(len(self._active_swaps), dtype=bool)),
#         coupling_degrees = self._coupling_degrees,
#         layers           = layers,
#         l2p              = self.l2p,
#         qubit_indices    = self._qubit_indices,
#         distance_matrix  = self._distance_matrix,
#         horizon          = self._horizon,
#     )
#
#     self._obs = {
#         "matrix":                matrix,
#         **bipartite,
#     }
#
# Note: swap_cancellation should come from your _build_cancellation method
# once implemented. The fallback to zeros above is a safe placeholder.


# ===========================================================================
# 4. TRAINING SCRIPT
# ===========================================================================
#
# from bipartite_extractor import BipartiteExtractor
# from sb3_contrib import MaskablePPO
#
# # Bipartite GNN only (replace matrix branch)
# policy_kwargs = dict(
#     features_extractor_class=BipartiteExtractor,
#     features_extractor_kwargs=dict(
#         features_dim=256,
#         gnn_hidden=64,
#         gnn_heads=4,
#         gnn_out=64,
#         gnn_layers=2,
#         action_mlp_hidden=64,
#         action_out=16,
#         use_bipartite=True,
#         use_matrix=False,    # ← toggle this
#     ),
#     net_arch=[256, 256],
# )
#
# # Matrix only (ablation — equivalent to a per-action MLP on matrix rows)
# policy_kwargs_matrix_only = dict(
#     features_extractor_class=BipartiteExtractor,
#     features_extractor_kwargs=dict(
#         features_dim=256,
#         action_mlp_hidden=64,
#         action_out=16,
#         use_bipartite=False,  # ← GNN disabled
#         use_matrix=True,
#     ),
#     net_arch=[256, 256],
# )
#
# # Both branches (augmented)
# policy_kwargs_both = dict(
#     features_extractor_class=BipartiteExtractor,
#     features_extractor_kwargs=dict(
#         features_dim=256,
#         gnn_hidden=64,
#         gnn_heads=4,
#         gnn_out=64,
#         gnn_layers=2,
#         action_mlp_hidden=64,
#         action_out=16,
#         use_bipartite=True,
#         use_matrix=True,     # ← both active
#     ),
#     net_arch=[256, 256],
# )


# ===========================================================================
# 5. RAY TUNE SEARCH SPACE ADDITIONS
# ===========================================================================
#
# "use_bipartite":       tune.choice([True, False]),
# "use_matrix":          tune.choice([True, False]),
# "gnn_layers":          tune.choice([1, 2, 3]),
# "action_out":          tune.choice([8, 16, 32]),
# "action_mlp_hidden":   tune.choice([32, 64, 128]),
#
# Add a trial filter to ensure at least one branch is active:
#
# def trial_filter(config):
#     return config["use_bipartite"] or config["use_matrix"]
