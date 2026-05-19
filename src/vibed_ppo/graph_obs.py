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

import numpy as np
from numpy import ndarray
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Constants – tune these to your largest topology (ibm_torino has 133 qubits)
# ---------------------------------------------------------------------------
NODE_F = 5  # number of node feature channels
BASE_COUP_EDGE_F = 2  # coupling edge feature channels
BASE_INT_EDGE_F = 2  # interaction edge feature channels
MAX_INTERACT_EDGES = 200  # pad/clip interaction edges to this fixed size


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
    p2l: NDArray[np.int64],  # physical → logical mapping
    cmap_edges: NDArray[np.int64],
    front_layer_pairs: list[tuple[int, int]],  # physical qubit pairs
    horizon_gate_count: np.ndarray,  # shape (num_qubits,)
    distance_matrix: np.ndarray,
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
            x[q, 3] = 1.0  # no active gates → maximally far

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
#
def build_coupling_graph(
    cmap_edges: ndarray,
    active_swaps: list[int],  # indices into cmap_edges
    horizon_traffic: np.ndarray,  # shape (len(cmap_edges),) gate counts
    horizon: int,
    action_layer_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    edge_index : (2, 2*E)  int64   (both directions)
    edge_attr  : (2*E, 3)  float32
    """
    active_set = set(active_swaps)
    max_traffic = horizon_traffic.max() if horizon_traffic.max() > 0 else 1.0

    rows, cols, attrs = [], [], []

    for idx, (p0, p1) in enumerate(cmap_edges):
        is_action = float(idx in active_set)
        traffic = horizon_traffic[idx] / max_traffic

        try:
            mask = ((cmap_edges[:, 0] == p0) & (cmap_edges[:, 1] == p1)) | (
                (cmap_edges[:, 0] == p1) & (cmap_edges[:, 1] == p0)
            )

            cmap_edge_index = np.where(mask)[0].min()
            action_index = active_swaps.index(cmap_edge_index)
            action_dists = action_layer_matrix[action_index, :]
        except Exception:
            action_dists = np.zeros(horizon)

        feat = [is_action, traffic] + list(action_dists)

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
#  2  is this gate executable right now (distance == 1)?          {0, 1} # TODO: Replace with action impact [-2, 2]
#
def build_interaction_graph(
    layers: list,  # dag.layers() output
    l2p: NDArray[np.int64],  # logical → physical mapping
    qubit_indices: dict,  # Qubit → int index
    distance_matrix: ndarray,
    horizon: int,
    action_gate_matrix: np.ndarray,
    num_active_swaps: int,
    max_edges: int = MAX_INTERACT_EDGES,
) -> tuple[ndarray, ndarray]:
    """
    Returns fixed-size zero-padded arrays:
    edge_index : (2, max_edges)  int64
    edge_attr  : (max_edges, 3)  float32
    """
    diam = distance_matrix.max() if distance_matrix.max() > 0 else 1.0

    rows, cols, attrs = [], [], []
    num_layers = min(len(layers), horizon)

    # Flatten layers into gates
    gate_nodes = []
    for h in range(num_layers):
        graph = layers[h]["graph"]
        gate_nodes += [n for n in graph.op_nodes() if len(n.qargs) == 2]

    for idx, gate in enumerate(gate_nodes):
        log0, log1 = [qubit_indices[q] for q in gate.qargs]
        p0, p1 = l2p[log0], l2p[log1]

        urgency = h / max(horizon - 1, 1)
        dist = distance_matrix[p0, p1] / diam
        action_distances = list(action_gate_matrix[:, idx])

        feat = [urgency, dist] + action_distances

        # both directions so message passing is symmetric
        rows += [p0, p1]
        cols += [p1, p0]
        attrs += [feat, feat]

        if len(rows) >= max_edges:
            break

    # Pad to fixed size
    n = len(rows)
    edge_index = np.zeros((2, max_edges), dtype=np.int64)
    edge_attr = np.zeros(
        (max_edges, BASE_INT_EDGE_F + num_active_swaps), dtype=np.float32
    )

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
    l2p: NDArray[np.int64],
    qubit_indices: dict,
    num_qubits: int,
    horizon: int,
) -> ndarray:
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
    l2p: NDArray[np.int64],
    qubit_indices: dict,
    cmap_edges: ndarray,
    horizon: int,
) -> ndarray:
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
    l2p: NDArray[np.int64],
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
# Action Gate Distance Matrix
# ---------------------------------------------------------------------------
def build_action_gate_matrix(
    layers: list,
    active_swaps: list[int],
    num_active_swaps: int,
    cmap_edges: ndarray,
    horizon: int,
    qubit_indices: dict,
    l2p: ndarray,
    distance_matrix: np.ndarray,
) -> np.ndarray:
    gate_nodes = []
    num_layers = min(len(layers), horizon)
    for h in range(num_layers):
        graph = layers[h]["graph"]
        gate_nodes += [n for n in graph.op_nodes() if len(n.qargs) == 2]

    matrix = np.zeros((num_active_swaps, len(gate_nodes)), dtype=np.int8)

    for slot, gate in enumerate(gate_nodes):
        for action, edge_idx in enumerate(active_swaps):
            p0_a, p1_a = cmap_edges[edge_idx]
            improvement = 0

            indices = [qubit_indices[q] for q in gate.qargs]
            p0_b = l2p[indices[0]]
            p1_b = l2p[indices[1]]

            if p0_b == p0_a and p1_b != p1_a:
                improvement += distance_matrix[p0_a, p1_b] - distance_matrix[p1_a, p1_b]
            elif p0_b == p1_a and p1_b != p0_a:
                improvement += distance_matrix[p1_a, p1_b] - distance_matrix[p0_a, p1_b]
            elif p1_b == p0_a and p0_b != p1_a:
                improvement += distance_matrix[p0_a, p0_b] - distance_matrix[p1_a, p0_b]
            elif p1_b == p1_a and p0_b != p0_a:
                improvement += distance_matrix[p1_a, p0_b] - distance_matrix[p0_a, p0_b]
            assert -1 <= improvement <= 1, "Improvement our of range"
            matrix[action, slot] = improvement

    return matrix


# ---------------------------------------------------------------------------
# Action Layer Distance Matrix
# ---------------------------------------------------------------------------
def build_action_layer_matrix(
    layers: list,
    active_swaps: list[int],
    num_active_swaps: int,
    cmap_edges: ndarray,
    horizon: int,
    qubit_indices: dict,
    l2p: ndarray,
    distance_matrix: np.ndarray,
) -> np.ndarray:
    matrix = np.zeros((num_active_swaps, horizon), dtype=np.int8)
    num_layers = min(len(layers), horizon)

    for h in range(num_layers):
        graph = layers[h]["graph"]
        gate_nodes = [n for n in graph.op_nodes() if len(n.qargs) == 2]

        if not gate_nodes:
            continue

        for slot, edge_idx in enumerate(active_swaps):
            p0_a, p1_a = cmap_edges[edge_idx]
            improvement = 0
            for node in gate_nodes:
                indices = [qubit_indices[q] for q in node.qargs]
                p0_b = l2p[indices[0]]
                p1_b = l2p[indices[1]]

                if p0_b == p0_a and p1_b != p1_a:
                    improvement += (
                        distance_matrix[p0_a, p1_b] - distance_matrix[p1_a, p1_b]
                    )
                elif p0_b == p1_a and p1_b != p0_a:
                    improvement += (
                        distance_matrix[p1_a, p1_b] - distance_matrix[p0_a, p1_b]
                    )
                elif p1_b == p0_a and p0_b != p1_a:
                    improvement += (
                        distance_matrix[p0_a, p0_b] - distance_matrix[p1_a, p0_b]
                    )
                elif p1_b == p1_a and p0_b != p0_a:
                    improvement += (
                        distance_matrix[p1_a, p0_b] - distance_matrix[p0_a, p0_b]
                    )

            assert -2 <= improvement <= 2, "Improvement out of range"
            matrix[slot, h] = improvement

    return matrix


# ---------------------------------------------------------------------------
# Master builder — call this from _update_obs
# ---------------------------------------------------------------------------
def build_graph_obs(
    num_qubits: int,
    l2p: NDArray[np.int64],
    p2l: NDArray[np.int64],
    cmap_edges: NDArray[np.int64],
    active_swaps: list[int],
    num_active_swaps: int,
    dag,  # Qiskit DAGCircuit
    qubit_indices: dict,
    distance_matrix: ndarray,
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
    traffic = compute_horizon_traffic(layers, l2p, qubit_indices, cmap_edges, horizon)
    action_gate_matrix = build_action_gate_matrix(
        layers,
        active_swaps,
        num_active_swaps,
        cmap_edges,
        horizon,
        qubit_indices,
        l2p,
        distance_matrix,
    )
    action_layer_matrix = build_action_layer_matrix(
        layers,
        active_swaps,
        num_active_swaps,
        cmap_edges,
        horizon,
        qubit_indices,
        l2p,
        distance_matrix,
    )
    node_features = build_node_features(
        num_qubits,
        p2l,
        cmap_edges,
        front_pairs,
        gate_counts,
        distance_matrix,
    )
    coup_ei, coup_ea = build_coupling_graph(
        cmap_edges, active_swaps, traffic, horizon, action_layer_matrix
    )
    int_ei, int_ea = build_interaction_graph(
        layers,
        l2p,
        qubit_indices,
        distance_matrix,
        horizon,
        action_gate_matrix,
        num_active_swaps,
    )

    return {
        "node_features": node_features,
        "coupling_edge_index": coup_ei,
        "coupling_edge_attr": coup_ea,
        "interact_edge_index": int_ei,
        "interact_edge_attr": int_ea,
    }
