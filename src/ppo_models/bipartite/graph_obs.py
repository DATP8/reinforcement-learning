"""
bipartite_graph_obs.py
----------------------
Builds a bipartite action-layer graph observation for the quantum circuit
routing environment.

Graph structure
---------------
- Action nodes : num_active_swaps  (one per row in the A×H matrix)
- Layer nodes  : horizon           (one per column in the A×H matrix)
- Edges        : all (action, layer) pairs — fully connected bipartite,
                 both directions. Total edges = 2 × num_active_swaps × horizon.

Node features
-------------
Action nodes (ACTION_NODE_F = 6):
  0  p0 normalised            (physical qubit index / num_qubits)
  1  p1 normalised
  2  coupling degree of p0    (normalised by max degree)
  3  coupling degree of p1
  4  is_masked                {0, 1}  — action currently disallowed
  5  will_cancel              {0, 1}  — swap causes CNOT cancellation

Layer nodes (LAYER_NODE_F = 4):
  0  layer_norm               (layer_idx / horizon)
  1  mean_distance            (mean physical dist of gate pairs, / diameter)
  2  fraction_executable      (fraction of gates with distance == 1)
  3  num_gates_norm           (num 2q-gates / (num_qubits // 2))

Edge features (EDGE_F = 3):
  0  delta                    raw A×H value  [-2, 2]
  1  layer_urgency            1 - (layer_idx / horizon)
  2  is_nonzero               {0, 1}  — swap actually affects this layer

Observation space keys
----------------------
  "bipartite_x"          : (num_action_nodes + horizon, max(ACTION_NODE_F, LAYER_NODE_F))
                           float32  — zero-padded so both node types share one matrix.
                           Action nodes occupy rows [0 : num_active_swaps],
                           layer nodes occupy rows  [num_active_swaps : num_active_swaps + horizon].
  "bipartite_node_type"  : (num_action_nodes + horizon,)  int64
                           0 = action node, 1 = layer node.
  "bipartite_edge_index" : (2, 2 * num_active_swaps * horizon)  int64
  "bipartite_edge_attr"  : (2 * num_active_swaps * horizon, EDGE_F)  float32
"""

from __future__ import annotations

import numpy as np

ACTION_NODE_F = 6
LAYER_NODE_F = 4
NODE_F = max(ACTION_NODE_F, LAYER_NODE_F)   # shared feature width, zero-padded
EDGE_F = 3


# ---------------------------------------------------------------------------
# Action node features
# ---------------------------------------------------------------------------

def _build_action_nodes(
    active_swaps: list[int],
    cmap_edges: list[tuple[int, int]] | np.ndarray,
    num_qubits: int,
    action_mask: np.ndarray,        # bool, shape (num_active_swaps,)
    swap_cancellation: np.ndarray,  # bool, shape (num_active_swaps,) — from your new obs key
    degrees: np.ndarray,            # precomputed coupling degree per physical qubit
) -> np.ndarray:
    """Return float32 array of shape (num_active_swaps, ACTION_NODE_F)."""
    max_deg = degrees.max() if degrees.max() > 0 else 1.0
    n = len(active_swaps)
    x = np.zeros((n, ACTION_NODE_F), dtype=np.float32)

    for slot, edge_idx in enumerate(active_swaps):
        p0, p1 = cmap_edges[edge_idx]
        x[slot, 0] = p0 / max(num_qubits - 1, 1)
        x[slot, 1] = p1 / max(num_qubits - 1, 1)
        x[slot, 2] = degrees[p0] / max_deg
        x[slot, 3] = degrees[p1] / max_deg
        x[slot, 4] = float(not action_mask[slot])   # is_masked
        x[slot, 5] = float(swap_cancellation[slot])  # will_cancel

    return x


# ---------------------------------------------------------------------------
# Layer node features
# ---------------------------------------------------------------------------

def _build_layer_nodes(
    layers: list,
    l2p: list[int] | np.ndarray,
    qubit_indices: dict,
    distance_matrix: np.ndarray,
    horizon: int,
    num_qubits: int,
) -> np.ndarray:
    """Return float32 array of shape (horizon, LAYER_NODE_F)."""
    diam = distance_matrix.max() if distance_matrix.max() > 0 else 1.0
    max_gates = max(num_qubits // 2, 1)
    x = np.zeros((horizon, LAYER_NODE_F), dtype=np.float32)

    for h in range(horizon):
        x[h, 0] = h / max(horizon - 1, 1)   # layer_norm

        if h >= len(layers):
            # No layer at this depth — leave distance/executable/count as 0
            continue

        graph = layers[h]["graph"]
        gate_nodes = [n for n in graph.op_nodes() if len(n.qargs) == 2]

        if not gate_nodes:
            continue

        distances = []
        executable = 0
        for node in gate_nodes:
            log0, log1 = [qubit_indices[q] for q in node.qargs]
            p0, p1 = l2p[log0], l2p[log1]
            d = distance_matrix[p0, p1]
            distances.append(d)
            if d == 1:
                executable += 1

        x[h, 1] = np.mean(distances) / diam                    # mean_distance
        x[h, 2] = executable / len(gate_nodes)                  # fraction_executable
        x[h, 3] = len(gate_nodes) / max_gates                   # num_gates_norm

    return x


# ---------------------------------------------------------------------------
# Edge construction
# ---------------------------------------------------------------------------

def _build_edges(
    matrix: np.ndarray,        # (num_active_swaps, horizon) — actual size
    num_active_swaps: int,      # actual number of active swaps
    max_active_swaps: int,      # fixed offset for layer nodes
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    total_edges = 2 * num_active_swaps * horizon
    edge_index = np.zeros((2, total_edges), dtype=np.int64)
    edge_attr = np.zeros((total_edges, EDGE_F), dtype=np.float32)

    idx = 0
    for slot in range(num_active_swaps):
        for h in range(horizon):
            action_node = slot
            layer_node = max_active_swaps + h    # ← fixed offset

            delta = float(matrix[slot, h])
            urgency = 1.0 - (h / max(horizon - 1, 1))
            is_nonzero = float(matrix[slot, h] != 0)
            feat = [delta, urgency, is_nonzero]

            edge_index[0, idx] = action_node
            edge_index[1, idx] = layer_node
            edge_attr[idx] = feat
            idx += 1

            edge_index[0, idx] = layer_node
            edge_index[1, idx] = action_node
            edge_attr[idx] = feat
            idx += 1

    return edge_index, edge_attr


# ---------------------------------------------------------------------------
# Precompute coupling degrees  (call once in env __init__)
# ---------------------------------------------------------------------------

def compute_coupling_degrees(
    num_qubits: int,
    cmap_edges: list[tuple[int, int]] | np.ndarray,
) -> np.ndarray:
    """Return float32 array of shape (num_qubits,) with coupling map degree."""
    degrees = np.zeros(num_qubits, dtype=np.float32)
    for p0, p1 in cmap_edges:
        degrees[p0] += 1
        degrees[p1] += 1
    return degrees


# ---------------------------------------------------------------------------
# Master builder — call from _update_obs
# ---------------------------------------------------------------------------

def build_bipartite_obs(
    matrix: np.ndarray,             # (num_active_swaps, horizon)  already built
    active_swaps: list[int],
    max_active_swaps: int,
    cmap_edges: list[tuple[int, int]] | np.ndarray,
    num_qubits: int,
    action_mask: np.ndarray,        # bool (num_active_swaps,)
    swap_cancellation: np.ndarray,  # bool (num_active_swaps,)
    coupling_degrees: np.ndarray,   # precomputed, shape (num_qubits,)
    layers: list,                   # dag.layers() — call once and pass in
    l2p: list[int] | np.ndarray,
    qubit_indices: dict,
    distance_matrix: np.ndarray,
    horizon: int,
) -> dict:
    """
    Returns
    -------
    dict with keys:
        bipartite_x          : (num_active_swaps + horizon, NODE_F)  float32
        bipartite_node_type  : (num_active_swaps + horizon,)         int64
        bipartite_edge_index : (2, 2 * num_active_swaps * horizon)   int64
        bipartite_edge_attr  : (2 * num_active_swaps * horizon, 3)   float32
    """
    num_active_swaps = len(active_swaps)
    N_total_actual = num_active_swaps + horizon
    N_total_max = max_active_swaps + horizon   # declared shape

    print("Build action nodes")
    action_x = _build_action_nodes(
        active_swaps, cmap_edges, num_qubits,
        action_mask, swap_cancellation, coupling_degrees,
    )   # (num_active_swaps, ACTION_NODE_F)

    print("Build layer nodes")
    layer_x = _build_layer_nodes(
        layers, l2p, qubit_indices, distance_matrix, horizon, num_qubits,
    )   # (horizon, LAYER_NODE_F)


    # Pad to max size
    x = np.zeros((N_total_max, NODE_F), dtype=np.float32)
    x[:num_active_swaps, :ACTION_NODE_F] = action_x
    x[max_active_swaps:max_active_swaps + horizon, :LAYER_NODE_F] = layer_x
    #  ^^^ layer nodes always start at max_active_swaps, not num_active_swaps

    node_type = np.zeros(N_total_max, dtype=np.int64)
    node_type[max_active_swaps:] = 1   # layer nodes at fixed offset

    # Edges: build with actual swaps, pad remainder with (0,0) / zeros
    E_max = 2 * max_active_swaps * horizon
    edge_index = np.zeros((2, E_max), dtype=np.int64)
    edge_attr = np.zeros((E_max, EDGE_F), dtype=np.float32)

    print("Build edges")
    if num_active_swaps > 0:
        ei, ea = _build_edges(matrix, num_active_swaps, max_active_swaps, horizon)
        n_edges = ei.shape[1]
        edge_index[:, :n_edges] = ei
        edge_attr[:n_edges] = ea

    print("Obs done")
    return {
        "bipartite_x":          x,
        "bipartite_node_type":  node_type,
        "bipartite_edge_index": edge_index,
        "bipartite_edge_attr":  edge_attr,
    }
