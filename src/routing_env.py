import gymnasium
import numpy as np
from gymnasium import spaces
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import CouplingMap

from src.policy_types import ActorCriticPolicyType


class RoutingEnv(gymnasium.Env):
    def __init__(
        self,
        coupling_map: CouplingMap,
        num_active_swaps: int,
        horizon: int,
        initial_difficulty: int,
        max_difficulty: int,
        diff_slope: int,
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
        self._cmap_edges = np.array(unique_edges)
        self._edge_set = frozenset(unique_edges)
        self._num_edges = len(self._cmap_edges)

        self._physical_to_edges = [[] for _ in range(self._num_qubits)]
        self._physical_to_edges = [[] for _ in range(self._num_qubits)]
        for i, (q1, q2) in enumerate(self._cmap_edges):
            self._physical_to_edges[q1].append(i)
            self._physical_to_edges[q2].append(i)

        self._active_swaps = []
        self.l2p: np.ndarray = np.arange(self._num_qubits, dtype=np.int64)
        self._p2l: np.ndarray = np.arange(self._num_qubits, dtype=np.int64)
        self.action_space = spaces.Discrete(self._num_active_swaps)

        if policy_type is ActorCriticPolicyType.BASIC or policy_type is ActorCriticPolicyType.SIMPLE_MLP:
            self.observation_space = spaces.Box(
                low=-2,
                high=2,
                shape=(self._num_active_swaps, self._horizon),
                dtype=np.int8,
            )
        else:
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
        self._cancellation_bonus = self._swap_penalty / 3
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
        self._visited_layouts = set()

    def set_difficulty(self, difficulty: int) -> None:
        self._current_difficulty = difficulty

    def get_difficulty(self) -> int:
        return self._current_difficulty

    def _compute_depth(self, sampled_diff: int) -> int:
        return self._diff_slope * sampled_diff

    def _compute_remaining_swaps(self) -> int:
        return self._depth

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}

        # Sample a random difficulty when curriculum learning done
        if self._current_difficulty >= self._max_difficulty:
            sampled_diff = int(self.np_random.integers(1, self._max_difficulty + 1))
        else:
            sampled_diff = self._current_difficulty

        self._depth = self._compute_depth(sampled_diff)
        self._remaining_swaps = self._compute_remaining_swaps()

        self.routed_circuit = QuantumCircuit(self._num_qubits)

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

        for logical_q, physical_q in enumerate(self.l2p):
            self._p2l[physical_q] = logical_q

        self._execute_front_layer()
        self._visited_layouts = {tuple(self._p2l)}

        self._update_obs()
        return self._get_obs(), {}

    def _apply_layout(self, sampled_diff: int) -> None:
        if self._current_difficulty >= self._max_difficulty:
            self.np_random.shuffle(self.l2p)
            return

        num_qubits = self._num_qubits
        num_swaps = int(
            (sampled_diff / self._max_difficulty) ** self._layout_exponent * num_qubits
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
        action = int(action)
        if action >= len(self._active_swaps):
            self._remaining_swaps = max(0, self._remaining_swaps - 1)
            terminated = self.is_terminal()
            truncated = self._remaining_swaps == 0 and not terminated
            achieved = self._completion_reward if terminated else 0.0
            self._update_obs()
            return self._get_obs(), achieved, terminated, truncated, {}

        edge_idx = self._active_swaps[action]
        p0, p1 = self._cmap_edges[edge_idx]

        l0, l1 = self._p2l[p0], self._p2l[p1]
        self._p2l[p0], self._p2l[p1] = l1, l0
        self.l2p[l0] = p1
        self.l2p[l1] = p0

        cancelled_nodes = self._pop_recent_cx(p0, p1)
        cancellation = 0
        if cancelled_nodes:
            phys_ctrl, phys_trgt = cancelled_nodes
            self.routed_circuit.cx(phys_trgt, phys_ctrl)
            self.routed_circuit.cx(phys_ctrl, phys_trgt)
            cancellation = 1
        else:
            self.routed_circuit.swap(p0, p1)

        gates_executed = self._execute_front_layer()
        if gates_executed > 0:
            self._visited_layouts.clear()

        self._visited_layouts.add(tuple(self._p2l))
        self._remaining_swaps = max(0, self._remaining_swaps - 1)
        terminated = self.is_terminal()
        truncated = self._remaining_swaps == 0 and not terminated

        cancellation_bonus = cancellation * self._cancellation_bonus
        achieved = (self._completion_reward if terminated else 0.0) + cancellation_bonus
        penalty = self._swap_penalty

        reward = achieved - penalty

        self._update_obs()
        return self._get_obs(), reward, terminated, truncated, {}

    def _pop_recent_cx(self, p0: int, p1: int) -> tuple[int, int] | None:
        """
        Reverse iterate over recent added gates to routed circuit and then if it matches wires
        where swap just added it checks if CX gate. If cx gate it removes it from circuit and
        returns orientation of cancelled gate
        """
        for i in range(len(self.routed_circuit.data) - 1, -1, -1):
            instruction = self.routed_circuit.data[i]
            indices = {self.routed_circuit.qubits.index(q) for q in instruction.qubits}

            if indices == {p0, p1}:
                if instruction.operation.name == "cx":
                    ctrl = self.routed_circuit.qubits.index(instruction.qubits[0])
                    tgt = self.routed_circuit.qubits.index(instruction.qubits[1])
                    del self.routed_circuit.data[i]
                    return ctrl, tgt
                return None

            if p0 in indices or p1 in indices:
                return None
        return None

    def _update_obs(self):
        if self._policy_type is not ActorCriticPolicyType.BASIC and self._policy_type is not ActorCriticPolicyType.SIMPLE_MLP:
            self._gnn = self._build_graph()
        self._matrix = self._build_matrix()

    def _execute_front_layer(self) -> int:
        progress = True
        gates_executed = 0
        while progress:
            progress = False
            for node in list(self._dag.front_layer()):
                indices = [self._qubit_indices[q] for q in node.qargs]
                if len(indices) == 1:
                    p0 = self.l2p[indices[0]]
                    self.routed_circuit._append(node.op, [p0])  # pyrefly: ignore
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

    def _get_obs(self):
        match self._policy_type:
            case ActorCriticPolicyType.BASIC | ActorCriticPolicyType.SIMPLE_MLP:
                return self._matrix

        graph_x, graph_edge_idx = self._gnn
        return {
            "matrix": self._matrix,
            "graph_x": graph_x,
            "graph_edge_idx": graph_edge_idx,
        }

    def _build_matrix(self) -> np.ndarray:
        layers = list(self._dag.layers())
        self._active_swaps = self._select_active_swaps(layers)

        matrix = np.zeros((self._num_active_swaps, self._horizon), dtype=np.int8)
        num_layers = min(len(layers), self._horizon)

        for h in range(num_layers):
            graph = layers[h]["graph"]
            gate_nodes = [n for n in graph.op_nodes() if len(n.qargs) == 2]

            if not gate_nodes:
                continue

            for slot, edge_idx in enumerate(self._active_swaps):
                p0_a, p1_a = self._cmap_edges[edge_idx]
                improvement = 0
                for node in gate_nodes:
                    indices = [self._qubit_indices[q] for q in node.qargs]
                    p0_b = self.l2p[indices[0]]
                    p1_b = self.l2p[indices[1]]

                    if p0_b == p0_a and p1_b != p1_a:
                        improvement += (
                            self._distance_matrix[p0_a, p1_b]
                            - self._distance_matrix[p1_a, p1_b]
                        )
                    elif p0_b == p1_a and p1_b != p0_a:
                        improvement += (
                            self._distance_matrix[p1_a, p1_b]
                            - self._distance_matrix[p0_a, p1_b]
                        )
                    elif p1_b == p0_a and p0_b != p1_a:
                        improvement += (
                            self._distance_matrix[p0_a, p0_b]
                            - self._distance_matrix[p1_a, p0_b]
                        )
                    elif p1_b == p1_a and p0_b != p0_a:
                        improvement += (
                            self._distance_matrix[p1_a, p0_b]
                            - self._distance_matrix[p0_a, p0_b]
                        )

                assert -2 <= improvement <= 2, "Improvement out of range"
                matrix[slot, h] = improvement

        return matrix

    def _select_active_swaps(self, layers: list) -> list[int]:
        if self.is_terminal():
            return []

        num_layers = min(len(layers), self._horizon)
        active_swaps = self._search_active_swaps(layers, num_layers)
        self.np_random.shuffle(active_swaps)
        return active_swaps

    def _search_active_swaps(self, layers, num_layers: int) -> list[int]:
        active_swaps = []
        seen_edges = set()
        for layer_idx in range(num_layers):
            graph = layers[layer_idx]["graph"]
            for node in graph.op_nodes():
                if len(node.qargs) == 2:
                    indices = [self._qubit_indices[q] for q in node.qargs]
                    for l0 in indices:
                        p0 = self.l2p[l0]
                        for edge_idx in self._physical_to_edges[p0]:
                            if edge_idx not in seen_edges:
                                active_swaps.append(edge_idx)
                                seen_edges.add(edge_idx)
                                if len(active_swaps) >= self._num_active_swaps:
                                    return active_swaps

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

        x = np.zeros((self._num_phys_qubits, 3), dtype=np.float32)

        for q in range(num_q):
            phys = self.l2p[q]
            total_interactions = interaction_counts[q].sum()

            if total_interactions > 0:
                dists = []
                for other_q in range(num_q):
                    if interaction_counts[q, other_q] > 0:
                        p1 = self.l2p[q]
                        p2 = self.l2p[other_q]
                        dists.append(self._distance_matrix[p1, p2])
                avg_dist = np.mean(dists) if len(dists) > 0 else 0.0
            else:
                avg_dist = 0.0

            x[q] = [phys, total_interactions, avg_dist]

        edges = []
        for q1 in range(num_q):
            for q2 in range(num_q):
                if interaction_counts[q1, q2] > 0:
                    edges.append((q1, q2))

        MAX_EDGES = 100
        edge_index = np.zeros((2, MAX_EDGES), dtype=np.int64)
        if edges:
            edge_array = np.array(edges, dtype=np.int64).T
            n_edges = min(edge_array.shape[1], MAX_EDGES)
            edge_index[:, :n_edges] = edge_array[:, :n_edges]

        return x, edge_index

    def valid_action_mask(self) -> np.ndarray:
        mask = np.zeros(self._num_active_swaps, dtype=bool)
        if self.is_terminal() or not self._active_swaps:
            return mask

        for slot, edge_idx in enumerate(self._active_swaps):
            p0, p1 = self._cmap_edges[edge_idx]
            l0, l1 = self._p2l[p0], self._p2l[p1]
            self._p2l[p0], self._p2l[p1] = l1, l0
            already_seen = tuple(self._p2l) in self._visited_layouts
            self._p2l[p0], self._p2l[p1] = l0, l1

            if not already_seen:
                mask[slot] = True

        if not mask.any():
            self._visited_layouts.clear()
            self._visited_layouts.add(tuple(self._p2l))
            mask = np.ones(self._num_active_swaps, dtype=bool)

        assert mask.any(), "No valid action"

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
