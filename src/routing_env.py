import gymnasium
import numpy as np
from gymnasium import spaces
from numba import njit
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import CouplingMap
from torch import Tensor

from src.policy_types import ActorCriticPolicyType
from src.states.dense_circuit_graph import DenseCircuitGraph
from src.vibed_ppo.graph_obs import build_graph_obs
from src.vibed_ppo.integration import make_observation_space as make_vibed_obs_space


@njit(cache=True)
def _numba_compute_improvements(
    matrix: np.ndarray,
    active_edges: np.ndarray,
    layer_pairs: np.ndarray,
    layer_indices: np.ndarray,
    l2p: np.ndarray,
    dist_matrix: np.ndarray,
) -> None:
    """
    Highly optimized machine-code kernel computing swap improvements
    across all active candidate slots and pre-extracted topological layers.
    """
    num_swaps = active_edges.shape[0]
    num_pairs = layer_pairs.shape[0]

    for p_idx in range(num_pairs):
        h = layer_indices[p_idx]
        l0 = layer_pairs[p_idx, 0]
        l1 = layer_pairs[p_idx, 1]

        p0_b = l2p[l0]
        p1_b = l2p[l1]

        for s_idx in range(num_swaps):
            p0_a = active_edges[s_idx, 0]
            p1_a = active_edges[s_idx, 1]

            imp = 0
            if p0_b == p0_a and p1_b != p1_a:
                imp = dist_matrix[p0_a, p1_b] - dist_matrix[p1_a, p1_b]
            elif p0_b == p1_a and p1_b != p0_a:
                imp = dist_matrix[p1_a, p1_b] - dist_matrix[p0_a, p1_b]
            elif p1_b == p0_a and p0_b != p1_a:
                imp = dist_matrix[p0_a, p0_b] - dist_matrix[p1_a, p0_b]
            elif p1_b == p1_a and p0_b != p0_a:
                imp = dist_matrix[p1_a, p0_b] - dist_matrix[p0_a, p0_b]

            matrix[s_idx, h] += imp


@njit(cache=True)
def _numba_build_graph_features(
    interaction_counts: np.ndarray,
    l2p: np.ndarray,
    dist_matrix: np.ndarray,
    num_q: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compiles node feature calculations and edge extraction down to raw pointer arithmetic.
    """
    x = np.zeros((num_q, 3), dtype=np.float32)

    temp_edges = np.zeros((2, num_q * num_q), dtype=np.int64)
    edge_count = 0

    for q in range(num_q):
        phys = l2p[q]
        total_interactions = 0.0
        sum_dist = 0.0
        interacting_neighbors = 0

        for other_q in range(num_q):
            count = interaction_counts[q, other_q]
            if count > 0:
                total_interactions += count
                p2 = l2p[other_q]
                sum_dist += dist_matrix[phys, p2]
                interacting_neighbors += 1

                temp_edges[0, edge_count] = q
                temp_edges[1, edge_count] = other_q
                edge_count += 1

        avg_dist = (
            (sum_dist / interacting_neighbors) if interacting_neighbors > 0 else 0.0
        )
        x[q, 0] = phys
        x[q, 1] = total_interactions
        x[q, 2] = avg_dist

    max_edges = min(edge_count, 100)
    edge_index = np.zeros((2, 100), dtype=np.int64)
    for i in range(max_edges):
        edge_index[0, i] = temp_edges[0, i]
        edge_index[1, i] = temp_edges[1, i]

    return x, edge_index


class RoutingEnv(gymnasium.Env):
    def __init__(
        self,
        coupling_map: CouplingMap,
        num_active_swaps: int,
        horizon: int,
        initial_difficulty: int,
        max_difficulty: int,
        diff_slope: float,
        layout_exponent: float,
        policy_type: ActorCriticPolicyType,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self._num_qubits = len(coupling_map.physical_qubits)
        self._num_logic_qubits = self._num_qubits
        self._num_phys_qubits = self._num_qubits
        self._coupling_map = coupling_map
        self._num_active_swaps = num_active_swaps
        self._horizon = horizon
        self._current_difficulty = initial_difficulty
        self._max_difficulty = max_difficulty
        self._diff_slope = diff_slope
        self._layout_exponent = layout_exponent
        self._policy_type = policy_type
        self._render_mode = render_mode
        self._distance_matrix: np.ndarray = coupling_map.distance_matrix  # pyrefly: ignore
        self._build_dist_pairs()

        unique_edges = list({tuple(sorted(edge)) for edge in coupling_map.get_edges()})
        self._cmap_edges = np.array(unique_edges, dtype=np.int64)
        self._edge_set = frozenset(unique_edges)
        self._num_edges = len(self._cmap_edges)

        self._physical_to_edges = [[] for _ in range(self._num_qubits)]
        for i, (q1, q2) in enumerate(self._cmap_edges):
            self._physical_to_edges[q1].append(i)
            self._physical_to_edges[q2].append(i)

        self._active_swaps = []
        self.l2p: np.ndarray = np.arange(self._num_qubits, dtype=np.int64)
        self._p2l: np.ndarray = np.arange(self._num_qubits, dtype=np.int64)
        self._routed_q2idx: dict = {}
        self.action_space = spaces.Discrete(self._num_active_swaps)

        match policy_type:
            case ActorCriticPolicyType.BASIC | ActorCriticPolicyType.SIMPLE_MLP:
                self.observation_space = spaces.Box(
                    low=-2,
                    high=2,
                    shape=(self._num_active_swaps, self._horizon),
                    dtype=np.int8,
                )
            case ActorCriticPolicyType.BASIC_CANCEL:
                self.observation_space = spaces.Dict(
                    {
                        "matrix": spaces.Box(
                            low=-2,
                            high=2,
                            shape=(self._num_active_swaps, self._horizon),
                            dtype=np.int8,
                        ),
                        "swap_cancellation": spaces.MultiBinary(self._num_active_swaps),
                    }
                )
            case ActorCriticPolicyType.DENSE_GRAPH_GNN:
                self.observation_space = spaces.Dict(
                    {
                        "matrix": spaces.Box(
                            low=-2,
                            high=2,
                            shape=(self._num_active_swaps, self._horizon),
                            dtype=np.int8,
                        ),
                        "graph_x": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(horizon + 1, self._num_qubits * 2),
                            dtype=np.float32,
                        ),
                        "graph_edge_index": spaces.Box(
                            low=0,
                            high=self._num_active_swaps,
                            shape=(100, 2),
                            dtype=np.int64,
                        ),
                        "graph_edge_attr": spaces.Box(
                            low=0,
                            high=self._num_active_swaps,
                            shape=(100, self._num_qubits + 1),
                            dtype=np.int64,
                        ),
                        "graph_num_nodes": spaces.Box(
                            low=1, high=horizon + 1, shape=(1,), dtype=np.int64
                        ),
                        "graph_num_edges": spaces.Box(
                            low=1, high=100, shape=(1,), dtype=np.int64
                        ),
                    }
                )
            case ActorCriticPolicyType.VIBE_GRAPH:
                self.observation_space = make_vibed_obs_space(
                    self._num_active_swaps,
                    self._horizon,
                    self._num_qubits,
                    len(self._cmap_edges),
                )
            case _:
                self.observation_space = spaces.Dict(
                    {
                        "matrix": spaces.Box(
                            low=-2,
                            high=2,
                            shape=(self._num_active_swaps, self._horizon),
                            dtype=np.int8,
                        ),
                        "graph_x": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(self._num_qubits, 3),
                            dtype=np.float32,
                        ),
                        "graph_edge_idx": spaces.Box(
                            low=0,
                            high=self._num_active_swaps,
                            shape=(2, 100),
                            dtype=np.int64,
                        ),
                    }
                )

        self._completion_reward = 1.0
        self._swap_penalty = 0.01
        self._cancellation_discount_factor = 1 / 3
        self._visited_layouts = set()

    def _build_dist_pairs(self) -> None:
        q1_idx, q2_idx = np.triu_indices(self._num_qubits, k=1)
        distances = self._distance_matrix[q1_idx, q2_idx]

        self._dist_pairs: dict[int, list[tuple[int, int]]] = {}
        for dist, q1, q2 in zip(distances, q1_idx, q2_idx):
            self._dist_pairs.setdefault(dist, []).append((q1, q2))

        self._all_dists = sorted(self._dist_pairs.keys())

    def _reset_internals(self):
        self.l2p = np.arange(self._num_qubits, dtype=np.int64)
        self._p2l = np.arange(self._num_qubits, dtype=np.int64)
        self._active_swaps = []

    def set_difficulty(self, difficulty: int) -> None:
        self._current_difficulty = difficulty

    def get_difficulty(self) -> int:
        return self._current_difficulty

    def _compute_depth(self, sampled_diff: float) -> int:
        return round(self._diff_slope * sampled_diff)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}

        # Sample a random difficuly for eval_env when curriculum learning done
        if False and self._current_difficulty >= self._max_difficulty:
            sampled_diff = int(self.np_random.integers(1, self._max_difficulty + 1))
        else:
            sampled_diff = self._current_difficulty

        self._depth = self._compute_depth(sampled_diff)
        self._remaining_swaps = self._depth

        self.routed_circuit = QuantumCircuit(self._num_qubits)
        self._routed_q2idx = {q: i for i, q in enumerate(self.routed_circuit.qubits)}

        provided_circuit: QuantumCircuit = options.get("circuit")  # pyrefly: ignore
        if provided_circuit is not None:
            self._circuit = provided_circuit.copy()
        else:
            self._circuit = self._generate_random_circuit_from_diff(sampled_diff)

        self._qubit_indices = {q: i for i, q in enumerate(self._circuit.qubits)}
        self._dag = circuit_to_dag(self._circuit)
        self._reset_internals()

        if provided_circuit is None:
            self._apply_layout(sampled_diff)
            self._p2l[self.l2p] = np.arange(self._num_qubits, dtype=np.int64)

        self._execute_front_layer()
        self._visited_layouts = {self._p2l.tobytes()}

        self._update_obs()
        return self._get_obs(), {}

    def _apply_layout(self, sampled_diff: int) -> None:
        if self._current_difficulty >= self._max_difficulty:
            self.np_random.shuffle(self.l2p)
            return

        num_qubits = self._num_qubits
        num_swaps = int(
            ((sampled_diff / self._max_difficulty) ** self._layout_exponent)
            * num_qubits
        )
        for i in range(num_qubits - 1, num_qubits - num_swaps - 1, -1):
            j = self.np_random.integers(0, i + 1)
            self.l2p[i], self.l2p[j] = self.l2p[j], self.l2p[i]

    def _generate_random_circuit_from_diff(self, difficulty: int) -> QuantumCircuit:
        qc = QuantumCircuit(self._num_qubits)
        remaining = difficulty
        while remaining > 0:
            valid_dists = [dist for dist in self._all_dists if dist <= remaining]
            if not valid_dists:
                break

            dist_idx = self.np_random.integers(0, len(valid_dists))
            next_dist = valid_dists[dist_idx]

            pairs = self._dist_pairs[next_dist]
            pair_idx = self.np_random.integers(0, len(pairs))

            q1, q2 = pairs[pair_idx]
            qc.cx(q1, q2)
            remaining -= next_dist

        return qc

    def step(self, action: int | np.ndarray):
        if self.is_terminal():
            return self._get_obs(), 0.0, True, False, {}

        action = int(action)
        assert action < len(self._active_swaps), (
            f"Invalid action {action}, only {len(self._active_swaps)} active swaps"
        )

        edge_idx = self._active_swaps[action]
        p0, p1 = self._cmap_edges[edge_idx]

        l0, l1 = self._p2l[p0], self._p2l[p1]
        self._p2l[p0], self._p2l[p1] = l1, l0
        self.l2p[l0] = p1
        self.l2p[l1] = p0

        cancelled_nodes = self._pop_recent_cx(p0, p1)
        cancellation = False
        if cancelled_nodes:
            phys_ctrl, phys_trgt = cancelled_nodes
            self.routed_circuit.cx(phys_trgt, phys_ctrl)
            self.routed_circuit.cx(phys_ctrl, phys_trgt)
            cancellation = True
        else:
            self.routed_circuit.swap(p0, p1)

        gates_executed = self._execute_front_layer()
        if gates_executed > 0:
            self._visited_layouts.clear()

        self._visited_layouts.add(self._p2l.tobytes())
        self._remaining_swaps = max(0, self._remaining_swaps - 1)
        terminated = self.is_terminal()
        truncated = self._remaining_swaps == 0 and not terminated

        cancellation_discount_factor = (
            self._cancellation_discount_factor if cancellation else 1.0
        )
        achieved = self._completion_reward if terminated else 0.0
        penalty = self._swap_penalty * cancellation_discount_factor
        reward = achieved - penalty

        self._update_obs()

        if not terminated and not truncated:
            mask = self.valid_action_mask()
            if not mask.any():
                truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def _pop_recent_cx(self, p0: int, p1: int, dry_run=False) -> tuple[int, int] | None:
        for i in range(len(self.routed_circuit.data) - 1, -1, -1):
            instruction = self.routed_circuit.data[i]
            qargs = instruction.qubits

            if len(qargs) == 2:
                q0 = self._routed_q2idx[qargs[0]]
                q1 = self._routed_q2idx[qargs[1]]

                if (q0 == p0 and q1 == p1) or (q0 == p1 and q1 == p0):
                    if instruction.operation.name == "cx":
                        if not dry_run:
                            del self.routed_circuit.data[i]
                        return q0, q1
                    return None

                if q0 == p0 or q0 == p1 or q1 == p0 or q1 == p1:
                    return None

            elif len(qargs) == 1:
                q0 = self._routed_q2idx[qargs[0]]
                if q0 == p0 or q0 == p1:
                    return None
            else:
                for q in qargs:
                    if self._routed_q2idx[q] in (p0, p1):
                        return None
        return None

    def _update_obs(self):
        self._matrix = self._build_matrix()
        self._cancellation = self._build_cancellation()
        if (
            self._policy_type is not ActorCriticPolicyType.BASIC
            and self._policy_type is not ActorCriticPolicyType.SIMPLE_MLP
        ):
            self._gnn = self._build_graph()
        if self._policy_type is ActorCriticPolicyType.VIBE_GRAPH:
            graph = build_graph_obs(
                num_qubits=self._num_qubits,
                l2p=self.l2p,
                p2l=self._p2l,
                cmap_edges=self._cmap_edges,
                active_swaps=self._active_swaps,
                num_active_swaps=self._num_active_swaps,
                dag=self._dag,
                qubit_indices=self._qubit_indices,
                distance_matrix=self._distance_matrix,
                horizon=self._horizon,
            )

            self._graph_obs = {
                "matrix": self._matrix,
                "swap_cancellation": self._cancellation,
                "node_features": graph["node_features"],
                "coupling_edge_index": graph["coupling_edge_index"],
                "coupling_edge_attr": graph["coupling_edge_attr"],
                "interact_edge_index": graph["interact_edge_index"],
                "interact_edge_attr": graph["interact_edge_attr"],
            }

    def _execute_front_layer(self) -> int:
        progress = True
        gates_executed = 0
        while progress:
            progress = False
            for node in list(self._dag.front_layer()):
                indices = [self._qubit_indices[q] for q in node.qargs]
                if len(indices) == 1:
                    p0 = self.l2p[indices[0]]
                    self.routed_circuit.append(node.op, [p0])
                    self._dag.remove_op_node(node)
                    gates_executed += 1
                    progress = True
                else:
                    p0, p1 = self.l2p[indices[0]], self.l2p[indices[1]]
                    if (p0, p1) in self._edge_set or (p1, p0) in self._edge_set:
                        self.routed_circuit.append(node.op, [p0, p1])
                        self._dag.remove_op_node(node)
                        gates_executed += 1
                        progress = True

        return gates_executed

    def _dense_graph_to_obs(
        self, graph: DenseCircuitGraph, horizon: int, n_qubits: int
    ) -> dict:
        max_nodes = horizon + 1
        max_edges = 100

        x: Tensor = graph.x  # pyrefly: ignore
        edge_index: Tensor = graph.edge_index  # pyrefly: ignore
        edge_attr: Tensor = graph.edge_attr  # pyrefly: ignore

        num_nodes = x.shape[0]
        num_edges = edge_index.shape[1]

        x_padded = np.zeros((max_nodes, n_qubits * 2), dtype=np.float32)
        x_padded[:num_nodes] = x.numpy()

        ei_padded = np.zeros((max_edges, 2), dtype=np.int64)
        ei_padded[:num_edges] = edge_index.t().numpy()

        ea_padded = np.zeros((max_edges, n_qubits + 1), dtype=np.int64)
        ea_padded[:num_edges] = edge_attr.numpy()

        return {
            "matrix": self._matrix,
            "graph_x": x_padded,
            "graph_edge_index": ei_padded,
            "graph_edge_attr": ea_padded,
            "graph_num_nodes": np.array([num_nodes], dtype=np.int64),
            "graph_num_edges": np.array([num_edges], dtype=np.int64),
        }

    def _get_obs(self):
        match self._policy_type:
            case ActorCriticPolicyType.BASIC | ActorCriticPolicyType.SIMPLE_MLP:
                return self._matrix
            case ActorCriticPolicyType.BASIC_CANCEL:
                return {"matrix": self._matrix, "swap_cancellation": self._cancellation}
            case ActorCriticPolicyType.HYBRID_GNN:
                graph_x, graph_edge_idx = self._gnn
                return {
                    "matrix": self._matrix,
                    "graph_x": graph_x,
                    "graph_edge_idx": graph_edge_idx,
                }
            case ActorCriticPolicyType.DENSE_GRAPH_GNN:
                circuit = dag_to_circuit(self._dag)
                graph = DenseCircuitGraph.from_circuit(circuit)
                return self._dense_graph_to_obs(graph, self._horizon, self._num_qubits)
            case ActorCriticPolicyType.VIBE_GRAPH:
                return self._graph_obs

    def _build_matrix(self) -> np.ndarray:
        qubit_depth = np.zeros(self._num_qubits, dtype=int)
        custom_layers = []

        temp_pairs = []
        temp_layer_indices = []

        for node in self._dag.op_nodes():
            indices = [self._qubit_indices[q] for q in node.qargs]
            layer = 0
            for idx in indices:
                if qubit_depth[idx] > layer:
                    layer = qubit_depth[idx]
            for idx in indices:
                qubit_depth[idx] = layer + 1

            if len(indices) == 2:
                while len(custom_layers) <= layer:
                    custom_layers.append([])
                custom_layers[layer].append(indices)

                if layer < self._horizon:
                    temp_pairs.append(indices)
                    temp_layer_indices.append(layer)

        self._active_swaps = self._select_active_swaps(custom_layers)

        matrix = np.zeros((self._num_active_swaps, self._horizon), dtype=np.int8)
        if not self._active_swaps or not temp_pairs:
            return matrix

        layer_pairs = np.array(temp_pairs, dtype=np.int64)
        layer_indices = np.array(temp_layer_indices, dtype=np.int64)
        active_edges = self._cmap_edges[self._active_swaps]

        _numba_compute_improvements(
            matrix,
            active_edges,
            layer_pairs,
            layer_indices,
            self.l2p,
            self._distance_matrix,
        )

        return matrix

    def _select_active_swaps(self, custom_layers: list) -> list[int]:
        if self.is_terminal():
            return []

        num_layers = min(len(custom_layers), self._horizon)
        active_swaps = []
        seen_edges = set()

        for layer_gates in custom_layers[:num_layers]:
            for l0, l1 in layer_gates:
                for l_idx in (l0, l1):
                    p0 = self.l2p[l_idx]
                    for edge_idx in self._physical_to_edges[p0]:
                        if edge_idx not in seen_edges:
                            active_swaps.append(edge_idx)
                            seen_edges.add(edge_idx)
                            if len(active_swaps) >= self._num_active_swaps:
                                break
                if len(active_swaps) >= self._num_active_swaps:
                    break
            if len(active_swaps) >= self._num_active_swaps:
                break

        self.np_random.shuffle(active_swaps)
        return active_swaps

    def _build_graph(self):
        num_q = self._num_logic_qubits
        interaction_counts = np.zeros((num_q, num_q), dtype=np.float32)

        for node in self._dag.op_nodes():
            qargs = node.qargs
            if len(qargs) == 2:
                q1 = self._qubit_indices[qargs[0]]
                q2 = self._qubit_indices[qargs[1]]
                interaction_counts[q1, q2] += 1
                interaction_counts[q2, q1] += 1

        return _numba_build_graph_features(
            interaction_counts,
            self.l2p,
            self._distance_matrix,
            num_q,
        )

    def _build_cancellation(self):
        cancellation = np.zeros(self._num_active_swaps, dtype=np.int8)
        for slot, edge_idx in enumerate(self._active_swaps):
            p0, p1 = self._cmap_edges[edge_idx]
            if self._pop_recent_cx(p0, p1, dry_run=True):
                cancellation[slot] = True
        return cancellation

    def valid_action_mask(self) -> np.ndarray:
        mask = np.zeros(self._num_active_swaps, dtype=bool)
        for slot, edge_idx in enumerate(self._active_swaps):
            p0, p1 = self._cmap_edges[edge_idx]
            l0, l1 = self._p2l[p0], self._p2l[p1]
            self._p2l[p0], self._p2l[p1] = l1, l0
            already_seen = self._p2l.tobytes() in self._visited_layouts
            self._p2l[p0], self._p2l[p1] = l0, l1

            if not already_seen:
                mask[slot] = True

        return mask

    def is_terminal(self) -> bool:
        return not bool(self._dag.op_nodes())

    def render(self) -> None:
        if self._render_mode == "ansi":
            print("--- Original ---")
            print(self._circuit)
            print("\n--- Routed ---")
            print(self.routed_circuit)
            print()
