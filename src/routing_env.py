import gymnasium
import numpy as np
from gymnasium import spaces
from numba import njit
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import CouplingMap
from torch import Tensor

from src.policy_types import ActorCriticPolicyType
from src.ppo_models.bipartite.graph_obs import (
    build_bipartite_obs,
    compute_coupling_degrees,
)
from src.ppo_models.bipartite.integration import make_bipartite_observation_space
from src.ppo_models.vibed.graph_obs import build_graph_obs
from src.ppo_models.vibed.integration import (
    make_observation_space as make_vibed_obs_space,
)
from src.states.dense_circuit_graph import DenseCircuitGraph


@njit(cache=True)
def _numba_compute_improvements(
    matrix: np.ndarray,
    active_edges: np.ndarray,
    layer_pairs: np.ndarray,
    layer_indices: np.ndarray,
    l2p: np.ndarray,
    dist_matrix: np.ndarray,
) -> None:
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

            new_val = matrix[s_idx, h] + imp
            if new_val > 2:
                matrix[s_idx, h] = 2
            elif new_val < -2:
                matrix[s_idx, h] = -2
            else:
                matrix[s_idx, h] = new_val


@njit(cache=True)
def _numba_build_graph_features(
    interaction_counts: np.ndarray,
    l2p: np.ndarray,
    dist_matrix: np.ndarray,
    num_q: int,
) -> tuple[np.ndarray, np.ndarray]:
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
        gamma: float,
        shaping_coef: float,
        sample_diff: bool = True,
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
        self._gamma = gamma
        self._shaping_coef = shaping_coef
        self._sample_diff = sample_diff
        self._render_mode = render_mode
        self._distance_matrix: np.ndarray = (
            coupling_map.distance_matrix  # pyrefly: ignore
        )
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
        self._action_history = []
        self.l2p: np.ndarray = np.arange(self._num_qubits, dtype=np.int64)
        self._p2l: np.ndarray = np.arange(self._num_qubits, dtype=np.int64)
        self.action_space = spaces.Discrete(self._num_active_swaps)

        self._coupling_degrees = compute_coupling_degrees(
            self._num_qubits, self._cmap_edges
        )

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
            case ActorCriticPolicyType.BIPARTITE:
                self.observation_space = make_bipartite_observation_space(
                    self._num_active_swaps, self._horizon, self._num_qubits
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
        self._action_history = []

        gate_list = []
        self._gate_ops = []
        for inst in self._circuit.data:
            if len(inst.qubits) == 2:
                q0 = self._qubit_indices[inst.qubits[0]]
                q1 = self._qubit_indices[inst.qubits[1]]
                gate_list.append((q0, q1))
                self._gate_ops.append(inst.operation)
            elif len(inst.qubits) == 1:
                q0 = self._qubit_indices[inst.qubits[0]]
                gate_list.append((q0, -1))
                self._gate_ops.append(inst.operation)

        self._num_gates = len(gate_list)
        self._gates = np.array(gate_list, dtype=np.int32)
        self._gate_executed = np.zeros(self._num_gates, dtype=bool)

        schedule_list: list[list[int]] = [[] for _ in range(self._num_qubits)]
        for gate_idx, (q0, q1) in enumerate(self._gates):
            schedule_list[q0].append(gate_idx)
            if q1 != -1:
                schedule_list[q1].append(gate_idx)

        max_ops = max((len(s) for s in schedule_list), default=0)
        self._qubit_schedules = np.full((self._num_qubits, max_ops), -1, dtype=np.int32)
        for q, sched in enumerate(schedule_list):
            self._qubit_schedules[q, : len(sched)] = sched

        self._q_pointers = np.zeros(self._num_qubits, dtype=np.int32)

    def set_difficulty(self, difficulty: int) -> None:
        self._current_difficulty = difficulty

    def get_difficulty(self) -> int:
        return self._current_difficulty

    def _compute_depth(self, sampled_diff: float) -> int:
        return round(self._diff_slope * sampled_diff)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}

        if self._sample_diff and self._current_difficulty >= self._max_difficulty:
            sampled_diff = int(self.np_random.integers(1, self._max_difficulty + 1))
        else:
            sampled_diff = self._current_difficulty

        self._depth = self._compute_depth(sampled_diff)
        self._remaining_swaps = self._depth

        provided_circuit: QuantumCircuit = options.get("circuit")  # pyrefly: ignore
        if provided_circuit is not None:
            self._circuit = provided_circuit.copy()
        else:
            self._circuit = self._generate_random_circuit_from_diff(sampled_diff)

        self._qubit_indices = {q: i for i, q in enumerate(self._circuit.qubits)}
        self._reset_internals()

        if provided_circuit is None:
            self._apply_layout(sampled_diff)
            self._p2l[self.l2p] = np.arange(self._num_qubits, dtype=np.int64)

        self._execute_front_layer()
        self._visited_layouts = {self._p2l.tobytes()}

        self._update_obs()
        return self._get_obs(), {}

    def set_circuit(self, circuit: QuantumCircuit, *, seed: int | None = None):
        return self.reset(seed=seed, options={"circuit": circuit})

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

        phi_before = self._compute_potential()

        action = int(action)
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
            self._action_history.append(("cx", phys_trgt, phys_ctrl))
            self._action_history.append(("cx", phys_ctrl, phys_trgt))
            cancellation = True
        else:
            self._action_history.append(("swap", p0, p1))

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
        phi_after = self._compute_potential()
        shaping = self._shaping_coef * (self._gamma * phi_after - phi_before)
        reward = achieved - penalty + shaping

        self._update_obs()

        is_looping = False
        if not terminated and not truncated:
            mask = self.valid_action_mask()
            truncated = not mask.any()
            is_looping = truncated

        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            {"is_looping": is_looping},
        )

    def _compute_potential(self) -> float:
        if self._shaping_coef == 0.0:
            return 0.0

        mask = ~self._gate_executed
        q0s = self._gates[mask, 0]
        q1s = self._gates[mask, 1]

        two_qubit = q1s != -1
        q0s = q0s[two_qubit]
        q1s = q1s[two_qubit]

        if len(q0s) == 0:
            return 0.0

        p0s = self.l2p[q0s]
        p1s = self.l2p[q1s]
        return -float(self._distance_matrix[p0s, p1s].mean())

    def _pop_recent_cx(self, p0: int, p1: int, dry_run=False) -> tuple[int, int] | None:
        for i in range(len(self._action_history) - 1, -1, -1):
            op, q0, q1 = self._action_history[i]
            if getattr(op, "name", op) == "cx":
                if (q0 == p0 and q1 == p1) or (q0 == p1 and q1 == p0):
                    if not dry_run:
                        self._action_history.pop(i)
                    return q0, q1
                if q0 == p0 or q0 == p1 or q1 == p0 or q1 == p1:
                    return None
            elif op == "swap":
                if q0 == p0 or q0 == p1 or q1 == p0 or q1 == p1:
                    return None
            else:
                if q0 == p0 or q0 == p1 or q1 == p0 or q1 == p1:
                    return None
        return None

    def _get_remaining_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self._num_qubits)
        for gate_idx in range(self._num_gates):
            if not self._gate_executed[gate_idx]:
                op = self._gate_ops[gate_idx]
                if self._gates[gate_idx, 1] == -1:
                    qc.append(op, [int(self._gates[gate_idx, 0])])
                else:
                    qc.append(
                        op,
                        [int(self._gates[gate_idx, 0]), int(self._gates[gate_idx, 1])],
                    )
        return qc

    def _update_obs(self):
        self._matrix = self._build_matrix()
        self._cancellation = self._build_cancellation()

        if self._policy_type not in (
            ActorCriticPolicyType.BASIC,
            ActorCriticPolicyType.SIMPLE_MLP,
        ):
            if self._policy_type is not ActorCriticPolicyType.VIBE_GRAPH:
                self._gnn = self._build_graph()

        if self._policy_type in (
            ActorCriticPolicyType.VIBE_GRAPH,
            ActorCriticPolicyType.DENSE_GRAPH_GNN,
        ):
            self._temp_remaining_circuit = self._get_remaining_circuit()

        if self._policy_type is ActorCriticPolicyType.VIBE_GRAPH:
            temp_dag = circuit_to_dag(self._temp_remaining_circuit)
            graph = build_graph_obs(
                num_qubits=self._num_qubits,
                l2p=self.l2p,
                p2l=self._p2l,
                cmap_edges=self._cmap_edges,
                active_swaps=self._active_swaps,
                num_active_swaps=self._num_active_swaps,
                dag=temp_dag,
                qubit_indices={
                    q: i for i, q in enumerate(self._temp_remaining_circuit.qubits)
                },
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
        if self._policy_type is ActorCriticPolicyType.BIPARTITE:
            # print("Building pipartite obs")
            bipartite = build_bipartite_obs(
                matrix=self._matrix,
                active_swaps=self._active_swaps,
                max_active_swaps=self._num_active_swaps,
                cmap_edges=self._cmap_edges,
                num_qubits=self._num_qubits,
                action_mask=self.valid_action_mask(),
                swap_cancellation=self._cancellation,
                coupling_degrees=self._coupling_degrees,
                layers=self._custom_layers,
                l2p=self.l2p,
                distance_matrix=self._distance_matrix,
                horizon=self._horizon,
            )
            self._bipartite_obs = {"matrix": self._matrix, **bipartite}

    def _execute_front_layer(self) -> int:
        progress = True
        gates_executed = 0
        while progress:
            progress = False
            for q in range(self._num_qubits):
                ptr = self._q_pointers[q]
                if ptr >= self._qubit_schedules.shape[1]:
                    continue

                gate_idx = self._qubit_schedules[q, ptr]
                if gate_idx == -1 or self._gate_executed[gate_idx]:
                    continue

                q0, q1 = self._gates[gate_idx]
                if q1 == -1:
                    if q == q0:
                        self._q_pointers[q0] += 1
                        self._gate_executed[gate_idx] = True
                        op = self._gate_ops[gate_idx]
                        p0 = self.l2p[q0]
                        self._action_history.append((op, p0, -1))
                        gates_executed += 1
                        progress = True
                    continue

                other_q = q1 if q == q0 else q0
                ptr_other = self._q_pointers[other_q]

                if (
                    ptr_other >= self._qubit_schedules.shape[1]
                    or self._qubit_schedules[other_q, ptr_other] != gate_idx
                ):
                    continue

                p0, p1 = self.l2p[q0], self.l2p[q1]
                if (p0, p1) in self._edge_set or (p1, p0) in self._edge_set:
                    self._q_pointers[q0] += 1
                    self._q_pointers[q1] += 1
                    self._gate_executed[gate_idx] = True
                    op = self._gate_ops[gate_idx]
                    self._action_history.append((op, p0, p1))
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
                graph = DenseCircuitGraph.from_circuit(self._temp_remaining_circuit)
                return self._dense_graph_to_obs(graph, self._horizon, self._num_qubits)
            case ActorCriticPolicyType.VIBE_GRAPH:
                return self._graph_obs
            case ActorCriticPolicyType.BIPARTITE:
                return self._bipartite_obs

    def _build_matrix(self) -> np.ndarray:
        qubit_depth = np.zeros(self._num_qubits, dtype=int)
        custom_layers = []
        temp_pairs = []
        temp_layer_indices = []

        for gate_idx in range(self._num_gates):
            if self._gate_executed[gate_idx]:
                continue

            q0, q1 = self._gates[gate_idx]
            if q1 == -1:
                qubit_depth[q0] += 1
                continue

            layer = max(qubit_depth[q0], qubit_depth[q1])
            qubit_depth[q0] = layer + 1
            qubit_depth[q1] = layer + 1

            while len(custom_layers) <= layer:
                custom_layers.append([])
            custom_layers[layer].append((q0, q1))

            if layer < self._horizon:
                temp_pairs.append((q0, q1))
                temp_layer_indices.append(layer)

        self._active_swaps = self._select_active_swaps(custom_layers)

        matrix = np.zeros((self._num_active_swaps, self._horizon), dtype=np.int8)
        self._custom_layers = custom_layers
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
            if len(active_swaps) >= self._num_active_swaps:
                break

        self.np_random.shuffle(active_swaps)
        return active_swaps

    def _build_graph(self):
        num_q = self._num_logic_qubits
        interaction_counts = np.zeros((num_q, num_q), dtype=np.float32)

        for gate_idx in range(self._num_gates):
            if not self._gate_executed[gate_idx]:
                q1, q2 = self._gates[gate_idx]
                if q2 == -1:
                    continue
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
        return bool(np.all(self._gate_executed)) if self._num_gates > 0 else True

    def render(self) -> None:
        if self._render_mode == "ansi":
            print("--- Original ---")
            print(self._circuit)
            print("\n--- Routed ---")
            print(self.get_routed_circuit())
            print()

    def get_routed_circuit(self) -> QuantumCircuit:
        routed_qc = QuantumCircuit(self._num_qubits, self._circuit.num_clbits)
        for op, p0, p1 in self._action_history:
            if getattr(op, "name", op) == "cx":
                routed_qc.cx(p0, p1)
            elif op == "swap":
                routed_qc.swap(p0, p1)
            elif p1 == -1:
                routed_qc.append(op, [int(p0)])
            else:
                routed_qc.append(op, [int(p0), int(p1)])
        return routed_qc

    def get_final_mapping(self) -> dict:
        return {self._circuit.qubits[i]: int(p) for i, p in enumerate(self.l2p)}
