"""
graph_obs.py
------------
Builds the two-graph observation for the quantum circuit routing environment.

Two graphs are constructed every step:

  1. COUPLING GRAPH  (hardware topology)
     - Nodes  : physical qubits
     - Edges  : valid CNOT connections from the coupling map
     - Encodes where we are and what the hardware looks like

  2. INTERACTION GRAPH  (circuit demand)
     - Nodes  : physical qubits (same node set, different edges)
     - Edges  : upcoming 2-qubit gates within the horizon, weighted by urgency
     - Encodes what the circuit needs

Both graphs share the same node feature matrix so they can be fed through
two separate GNN branches that then get combined with the matrix features.

Observation space additions
---------------------------
Replace the old placeholder graph keys with:

    "coupling_edge_index" : (2, num_coupling_edges),  int64
    "coupling_edge_attr"  : (num_coupling_edges, 3),  float32
    "interact_edge_index" : (2, MAX_INTERACT_EDGES),  int64   (zero-padded)
    "interact_edge_attr"  : (MAX_INTERACT_EDGES, 3),  float32 (zero-padded)
    "node_features"       : (num_qubits, NODE_F),     float32

NODE_F = 5  (see _build_node_features)
COUPLING edge features = 3
INTERACTION edge features = 3
"""

from __future__ import annotations

import numpy as np
from qiskit.converters import circuit_to_dag


# ---------------------------------------------------------------------------
# Constants – tune these to your largest topology (ibm_torino has 133 qubits)
# ---------------------------------------------------------------------------
NODE_F = 5          # number of node feature channels
COUP_EDGE_F = 3     # coupling edge feature channels
INT_EDGE_F = 3      # interaction edge feature channels
MAX_INTERACT_EDGES = 200   # pad/clip interaction edges to this fixed size


# ---------------------------------------------------------------------------
# Node features  (per physical qubit)
# ---------------------------------------------------------------------------
#
#  0  normalised degree in coupling map          [0, 1]
#  1  normalised logical qubit index at this     [0, 1]  (0 if idle)
#  2  is this qubit part of a front-layer gate?  {0, 1}
#  3  min distance to any front-layer gate pair  [0, 1]  (normalised by diam)
#  4  gate count in horizon (demand)             [0, 1]  (normalised by H)
#
def build_node_features(
    num_qubits: int,
    p2l: list[int],               # physical → logical mapping
    cmap_edges: list[tuple[int, int]],
    front_layer_pairs: list[tuple[int, int]],   # physical qubit pairs
    horizon_gate_count: np.ndarray,             # shape (num_qubits,)
    distance_matrix: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Return float32 array of shape (num_qubits, NODE_F)."""

    x = np.zeros((num_qubits, NODE_F), dtype=np.float32)

    # -- 0: normalised degree --
    degrees = np.zeros(num_qubits, dtype=np.float32)
    for p0, p1 in cmap_edges:
        degrees[p0] += 1
        degrees[p1] += 1
    max_deg = degrees.max() if degrees.max() > 0 else 1.0
    x[:, 0] = degrees / max_deg

    # -- 1: normalised logical index (0 = idle qubit) --
    num_logical = num_qubits  # same size; idle qubits map to themselves
    for phys, log in enumerate(p2l):
        x[phys, 1] = log / max(num_logical - 1, 1)

    # -- 2 & 3: front-layer gate membership and proximity --
    front_phys = set()
    for p0, p1 in front_layer_pairs:
        front_phys.add(p0)
        front_phys.add(p1)

    diam = distance_matrix.max() if distance_matrix.max() > 0 else 1.0

    for q in range(num_qubits):
        # -- 2: If the phys qubit is in front layer
        x[q, 2] = 1.0 if q in front_phys else 0.0

        # -- 3: Distance to a front-layer qubit
        if front_layer_pairs:
            min_d = min(
                min(distance_matrix[q, p0], distance_matrix[q, p1])
                for p0, p1 in front_layer_pairs
            )
            x[q, 3] = min_d / diam
        else:
            x[q, 3] = 1.0   # no active gates → maximally far

    # -- 4: normalised gate demand within horizon --
    max_count = horizon_gate_count.max() if horizon_gate_count.max() > 0 else 1.0
    x[:, 4] = horizon_gate_count / max_count

    return x


# ---------------------------------------------------------------------------
# Coupling graph edges
# ---------------------------------------------------------------------------
#
# Edge features (per coupling map edge, both directions):
#  0  is this edge a candidate SWAP action?   {0, 1}
#  1  current gate "traffic": gates needing   [0, 1]  this edge in horizon
#  2  is this edge adjacent to a front-layer  {0, 1}  gate pair?
#
def build_coupling_graph(
    num_qubits: int,
    cmap_edges: list[tuple[int, int]],
    active_swaps: list[int],            # indices into cmap_edges
    horizon_traffic: np.ndarray,        # shape (len(cmap_edges),) gate counts
    front_layer_pairs: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    edge_index : (2, 2*E)  int64   (both directions)
    edge_attr  : (2*E, 3)  float32
    """
    active_set = set(active_swaps)
    front_set = set(map(tuple, front_layer_pairs))
    max_traffic = horizon_traffic.max() if horizon_traffic.max() > 0 else 1.0

    rows, cols, attrs = [], [], []

    for idx, (p0, p1) in enumerate(cmap_edges):
        is_action = float(idx in active_set)
        traffic = horizon_traffic[idx] / max_traffic
        is_front = float(
            (p0, p1) in front_set or (p1, p0) in front_set
        )
        feat = [is_action, traffic, is_front]

        # both directions
        rows += [p0, p1]
        cols += [p1, p0]
        attrs += [feat, feat]

    edge_index = np.array([rows, cols], dtype=np.int64)
    edge_attr = np.array(attrs, dtype=np.float32)
    return edge_index, edge_attr


# ---------------------------------------------------------------------------
# Interaction graph edges
# ---------------------------------------------------------------------------
#
# Edge features (per upcoming gate pair within horizon):
#  0  normalised layer index (urgency: 0 = front, 1 = far away)  [0, 1]
#  1  current physical distance between the two qubits            [0, 1]
#  2  is this gate executable right now (distance == 1)?          {0, 1}
#
def build_interaction_graph(
    layers: list,                       # dag.layers() output
    l2p: list[int],                     # logical → physical mapping
    qubit_indices: dict,                # Qubit → int index
    distance_matrix: np.ndarray,
    horizon: int,
    max_edges: int = MAX_INTERACT_EDGES,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns fixed-size zero-padded arrays:
    edge_index : (2, max_edges)  int64
    edge_attr  : (max_edges, 3)  float32
    """
    diam = distance_matrix.max() if distance_matrix.max() > 0 else 1.0

    rows, cols, attrs = [], [], []
    num_layers = min(len(layers), horizon)

    for h in range(num_layers):
        graph = layers[h]["graph"]
        gate_nodes = [n for n in graph.op_nodes() if len(n.qargs) == 2]

        for node in gate_nodes:
            log0, log1 = [qubit_indices[q] for q in node.qargs]
            p0, p1 = l2p[log0], l2p[log1]

            urgency = h / max(horizon - 1, 1)
            dist = distance_matrix[p0, p1] / diam
            executable = float(distance_matrix[p0, p1] == 1)
            feat = [urgency, dist, executable]

            # both directions so message passing is symmetric
            rows += [p0, p1]
            cols += [p1, p0]
            attrs += [feat, feat]

            if len(rows) >= max_edges:
                break
        if len(rows) >= max_edges:
            break

    # Pad to fixed size
    n = len(rows)
    edge_index = np.zeros((2, max_edges), dtype=np.int64)
    edge_attr = np.zeros((max_edges, INT_EDGE_F), dtype=np.float32)

    if n > 0:
        edge_index[0, :n] = rows[:max_edges]
        edge_index[1, :n] = cols[:max_edges]
        edge_attr[:n] = attrs[:max_edges]

    return edge_index, edge_attr


# ---------------------------------------------------------------------------
# Horizon gate count helper  (used in node features)
# ---------------------------------------------------------------------------

def compute_horizon_gate_count(
    layers: list,
    l2p: list[int],
    qubit_indices: dict,
    num_qubits: int,
    horizon: int,
) -> np.ndarray:
    """Count how many gates each physical qubit participates in within the horizon."""
    counts = np.zeros(num_qubits, dtype=np.float32)
    for h in range(min(len(layers), horizon)):
        graph = layers[h]["graph"]
        for node in graph.op_nodes():
            if len(node.qargs) == 2:
                for q in node.qargs:
                    log = qubit_indices[q]
                    counts[l2p[log]] += 1
    return counts


# ---------------------------------------------------------------------------
# Horizon traffic helper  (used in coupling edge features)
# ---------------------------------------------------------------------------

def compute_horizon_traffic(
    layers: list,
    l2p: list[int],
    qubit_indices: dict,
    cmap_edges: list[tuple[int, int]],
    distance_matrix: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """
    For each coupling edge, count how many upcoming gates 'need' to cross it.
    A gate needs an edge if performing that SWAP would reduce the gate's distance.
    This mirrors the same logic used in _build_matrix.
    """
    traffic = np.zeros(len(cmap_edges), dtype=np.float32)
    for h in range(min(len(layers), horizon)):
        graph = layers[h]["graph"]
        gate_nodes = [n for n in graph.op_nodes() if len(n.qargs) == 2]
        for idx, (p0_a, p1_a) in enumerate(cmap_edges):
            for node in gate_nodes:
                indices = [qubit_indices[q] for q in node.qargs]
                p0_b = l2p[indices[0]]
                p1_b = l2p[indices[1]]
                # any qubit overlap means this edge is relevant
                if p0_b in (p0_a, p1_a) or p1_b in (p0_a, p1_a):
                    traffic[idx] += 1
    return traffic


# ---------------------------------------------------------------------------
# Front-layer gate pairs helper
# ---------------------------------------------------------------------------

def get_front_layer_pairs(
    layers: list,
    l2p: list[int],
    qubit_indices: dict,
) -> list[tuple[int, int]]:
    """Return physical qubit pairs from the first (front) layer."""
    if not layers:
        return []
    graph = layers[0]["graph"]
    pairs = []
    for node in graph.op_nodes():
        if len(node.qargs) == 2:
            log0, log1 = [qubit_indices[q] for q in node.qargs]
            pairs.append((l2p[log0], l2p[log1]))
    return pairs


# ---------------------------------------------------------------------------
# Master builder — call this from _update_obs
# ---------------------------------------------------------------------------

def build_graph_obs(
    num_qubits: int,
    l2p: list[int],
    p2l: list[int],
    cmap_edges: list[tuple[int, int]],
    active_swaps: list[int],
    dag,                          # Qiskit DAGCircuit
    qubit_indices: dict,
    distance_matrix: np.ndarray,
    horizon: int,
) -> dict:
    """
    Returns a dict with keys:
        node_features        : (num_qubits, NODE_F)           float32
        coupling_edge_index  : (2, 2*num_coupling_edges)      int64
        coupling_edge_attr   : (2*num_coupling_edges, 3)      float32
        interact_edge_index  : (2, MAX_INTERACT_EDGES)        int64
        interact_edge_attr   : (MAX_INTERACT_EDGES, 3)        float32
    """
    layers = list(dag.layers())

    front_pairs = get_front_layer_pairs(layers, l2p, qubit_indices)
    gate_counts = compute_horizon_gate_count(
        layers, l2p, qubit_indices, num_qubits, horizon
    )
    traffic = compute_horizon_traffic(
        layers, l2p, qubit_indices, cmap_edges, distance_matrix, horizon
    )

    node_features = build_node_features(
        num_qubits, p2l, cmap_edges, front_pairs, gate_counts,
        distance_matrix, horizon,
    )
    coup_ei, coup_ea = build_coupling_graph(
        num_qubits, cmap_edges, active_swaps, traffic, front_pairs,
    )
    int_ei, int_ea = build_interaction_graph(
        layers, l2p, qubit_indices, distance_matrix, horizon,
    )

    return {
        "node_features": node_features,
        "coupling_edge_index": coup_ei,
        "coupling_edge_attr": coup_ea,
        "interact_edge_index": int_ei,
        "interact_edge_attr": int_ea,
    }
