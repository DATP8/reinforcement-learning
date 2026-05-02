"""
integration_guide.py
--------------------
Concrete diffs showing what to change in your environment and training script.
Read this as a guide — not a drop-in replacement for your full env file.
"""

# ===========================================================================
# 1. OBSERVATION SPACE  (in __init__)
# ===========================================================================
#
# Replace the old placeholder graph keys with the five new ones.
# Sizes are derived from your topology constants.
#
# For ibm_torino (133 qubits, ~160 coupling edges):
#   coupling edges (both dirs): ~320
#   MAX_INTERACT_EDGES: 200 (tune upward if you have deep circuits)
#
# For 6-qubit linear (5 coupling edges):
#   coupling edges (both dirs): 10
#   MAX_INTERACT_EDGES: 50 is plenty

from gymnasium import spaces
import numpy as np

# ---- constants (set these from your topology) ----
NUM_QUBITS = 6              # or 133 for ibm_torino
NUM_COUPLING_EDGES = 5      # undirected; we store both dirs → *2 in the space
NODE_F = 5
COUP_EDGE_F = 3
INT_EDGE_F = 3
MAX_INTERACT_EDGES = 200

# In __init__ of your env:
def make_observation_space(
    num_active_swaps, horizon, num_qubits,
    num_coupling_edges,   # undirected count
    max_interact_edges=MAX_INTERACT_EDGES,
):
    return spaces.Dict({
        # --- existing ---
        "matrix": spaces.Box(
            low=-2, high=2,
            shape=(num_active_swaps, horizon),
            dtype=np.int8,
        ),
        # --- new graph observations ---
        "node_features": spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(num_qubits, NODE_F),
            dtype=np.float32,
        ),
        "coupling_edge_index": spaces.Box(
            low=0, high=num_qubits - 1,
            shape=(2, num_coupling_edges * 2),   # both directions
            # shape=(2, num_active_swaps * 2),   # both directions
            dtype=np.int64,
        ),
        "coupling_edge_attr": spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(num_coupling_edges * 2, COUP_EDGE_F),
            # shape=(num_active_swaps * 2, COUP_EDGE_F),
            dtype=np.float32,
        ),
        "interact_edge_index": spaces.Box(
            low=0, high=num_qubits - 1,
            shape=(2, max_interact_edges),
            dtype=np.int64,
        ),
        "interact_edge_attr": spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(max_interact_edges, INT_EDGE_F),
            dtype=np.float32,
        ),
    })


# ===========================================================================
# 2. _update_obs  (call build_graph_obs here)
# ===========================================================================
#
# from graph_obs import build_graph_obs
#
# def _update_obs(self):
#     matrix = self._build_matrix()      # existing — also sets self._active_swaps
#
#     graph = build_graph_obs(
#         num_qubits      = self._num_qubits,
#         l2p             = self.l2p,
#         p2l             = self._p2l,
#         cmap_edges      = self._cmap_edges,
#         active_swaps    = self._active_swaps,
#         dag             = self._dag,
#         qubit_indices   = self._qubit_indices,
#         distance_matrix = self._distance_matrix,
#         horizon         = self._horizon,
#     )
#
#     self._obs = {
#         "matrix":               matrix,
#         "node_features":        graph["node_features"],
#         "coupling_edge_index":  graph["coupling_edge_index"],
#         "coupling_edge_attr":   graph["coupling_edge_attr"],
#         "interact_edge_index":  graph["interact_edge_index"],
#         "interact_edge_attr":   graph["interact_edge_attr"],
#     }
#
# def _get_obs(self):
#     return self._obs


# ===========================================================================
# 3. TRAINING SCRIPT  (policy_kwargs)
# ===========================================================================
#
# from hybrid_extractor import HybridExtractor
# from sb3_contrib import MaskablePPO
#
# policy_kwargs = dict(
#     features_extractor_class=HybridExtractor,
#     features_extractor_kwargs=dict(
#         features_dim=256,
#         gnn_hidden=64,
#         gnn_heads=4,
#         gnn_out=64,
#         matrix_out=128,
#     ),
#     net_arch=[256, 256],   # policy/value MLP after extractor
# )
#
# model = MaskablePPO(
#     "MultiInputPolicy",
#     env,
#     policy_kwargs=policy_kwargs,
#     verbose=1,
#     ...
# )


# ===========================================================================
# 4. NOTES ON PADDING AND VARIABLE TOPOLOGY
# ===========================================================================
#
# ibm_torino vs 6-qubit linear have very different sizes.
# Since you train one model per topology, you can hard-code the coupling
# edge count per training run and there is NO padding needed for coupling
# edges — the coupling graph is fully determined by the fixed topology.
#
# Interaction edges DO vary each step (different gates in the horizon),
# which is why they are zero-padded to MAX_INTERACT_EDGES.
# Set MAX_INTERACT_EDGES conservatively:
#   worst case ≈ horizon × max_gates_per_layer × 2 (both dirs)
#   For H=5 layers, up to ~10 2q-gates each → 100 edges → 200 with both dirs.
#
# The _batch_graphs helper in hybrid_extractor.py strips the zero-padding
# before passing edges to PyG, so no spurious self-loops are introduced.


# ===========================================================================
# 5. RECOMMENDED HYPERPARAMETERS (starting point)
# ===========================================================================
#
# MaskablePPO:
#   learning_rate     = 3e-4
#   n_steps           = 2048
#   batch_size        = 256
#   n_epochs          = 10
#   gamma             = 0.99
#   gae_lambda        = 0.95
#   clip_range        = 0.2
#   ent_coef          = 0.01   ← keep some entropy; routing has many valid paths
#   vf_coef           = 0.5
#
# GNN (HybridExtractor):
#   gnn_hidden        = 64
#   gnn_heads         = 4     ← reduces to 2 for the 6-qubit topology
#   gnn_out           = 64
#   matrix_out        = 128
#   features_dim      = 256
#
# For ibm_torino (133 qubits) consider:
#   gnn_hidden        = 128
#   gnn_heads         = 4
#   gnn_out           = 128
#   features_dim      = 512
