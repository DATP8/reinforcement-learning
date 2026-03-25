from gymnasium import spaces
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.converters import circuit_to_dag
import numpy as np
import gymnasium


class RoutingEnv(gymnasium.Env):
    def __init__(
        self,
        cmap: CouplingMap,
        horizon: int,
        render_mode: str | None = None,
        initial_difficulty=1,
    ) -> None:
        super().__init__()
        self.cmap_edges = list({tuple(sorted(edge)) for edge in cmap.get_edges()})
        self.edge_set = frozenset(self.cmap_edges)
        self.distance_matrix: np.ndarray = cmap.distance_matrix  # pyrefly: ignore

        self.num_phys_qubits = cmap.size()
        self.horizon = horizon
        self.render_mode = render_mode

        self.num_logic_qubits = 0
        self.current_difficulty = initial_difficulty

        self.action_space = spaces.Discrete(len(self.cmap_edges))
        self.observation_space = spaces.Dict(
            {
                "matrix": spaces.Box(
                    low=-2,
                    high=2,
                    shape=(len(self.cmap_edges), self.horizon),
                    dtype=np.int8,
                ),
                "graph_x": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_phys_qubits, 3),  # example features
                    dtype=np.float32,
                ),
                "graph_edge_idx": spaces.Box(
                    low=0,
                    high=self.num_phys_qubits,
                    shape=(2, 100),  # max edges (pad if needed)
                    dtype=np.int64,
                ),
            }
        )

        self.completion_reward = 10
        self.gate_cost = -0.1
        self.reduced_gate_cost = -0.1

        self.locked_actions = set()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        while True:
            self.num_logic_qubits = int(
                self.np_random.integers(2, self.num_phys_qubits + 1)
            )

            self.circuit = self._generate_random_circuit(
                self.num_logic_qubits, self.current_difficulty
            )

            self.qubit_indices = {q: i for i, q in enumerate(self.circuit.qubits)}
            self.dag = circuit_to_dag(self.circuit)
            self.routed_circuit = QuantumCircuit(self.num_phys_qubits)
            self.logical_to_physical, self.physical_to_logical = (
                self._generate_random_mapping()
            )

            self._execute_front_layer()
            if not self._is_terminal():
                break

        self.locked_actions = set()
        return self._get_obs(), {}

    def _build_graph(self):
        num_q = self.num_logic_qubits

        # Interaction tracking
        interaction_counts = np.zeros((num_q, num_q), dtype=np.float32)

        # Iterate over remaining DAG nodes
        for node in self.dag.op_nodes():
            qargs = node.qargs

            if len(qargs) == 2:
                # PERF: This is O(N)
                q1 = self.dag.qubits.index(qargs[0])
                q2 = self.dag.qubits.index(qargs[1])

                interaction_counts[q1, q2] += 1
                interaction_counts[q2, q1] += 1

        # Node features
        # Features per logical qubit:
        # [physical_position, total_interactions, avg_distance_to_partners]

        # Fill for phys qubits, then just update for logical ones
        x = np.zeros((self.num_phys_qubits, 3), dtype=np.float32)

        for q in range(num_q):
            phys = self.logical_to_physical[q]

            total_interactions = interaction_counts[q].sum()

            # Compute avg distance to interacting partners
            if total_interactions > 0:
                dists = []

                for other_q in range(num_q):
                    if interaction_counts[q, other_q] > 0:
                        p1 = self.logical_to_physical[q]
                        p2 = self.logical_to_physical[other_q]

                        d = self.distance_matrix[p1][p2]
                        dists.append(d)

                avg_dist = np.mean(dists) if len(dists) > 0 else 0.0
            else:
                avg_dist = 0.0

            x[q] = [
                phys,
                total_interactions,
                avg_dist,
            ]

        # Edge index
        edges = []
        for q1 in range(num_q):
            for q2 in range(num_q):
                if interaction_counts[q1, q2] > 0:
                    edges.append((q1, q2))

        # Padding (IMPORTANT for SB3)
        MAX_EDGES = 100
        edge_index = np.zeros((2, MAX_EDGES), dtype=np.int64)
        for i, (u, v) in enumerate(edges[:MAX_EDGES]):
            edge_index[0, i] = u
            edge_index[1, i] = v

        return x, edge_index

    def _generate_random_mapping(self):
        chosen = self.np_random.choice(
            self.num_phys_qubits, size=self.num_logic_qubits, replace=False
        ).astype(np.int64)
        l2p = chosen

        p2l = np.full(self.num_phys_qubits, -1, dtype=np.int64)
        p2l[l2p] = np.arange(self.num_logic_qubits, dtype=np.int64)  # pyrefly: ignore
        return l2p, p2l

    def _generate_random_circuit(
        self, num_qubits: int, num_gates: int
    ) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        # gate_set = ["h", "s", "cx"]
        gate_set = ["cx"]
        for _ in range(num_gates):
            gate = self.np_random.choice(gate_set)
            if gate == "cx" and num_qubits > 1:
                q0, q1 = self.np_random.choice(num_qubits, size=2, replace=False)
                qc.cx(int(q0), int(q1))
            elif gate == "h":
                qc.h(int(self.np_random.choice(num_qubits)))
            else:
                qc.s(int(self.np_random.choice(num_qubits)))

        return qc

    def step(self, action: int | np.ndarray):
        action = int(action)

        self._apply_swap(action)
        gates_executed, active_qubits = self._execute_front_layer()

        obs = self._get_obs()
        terminated = self._is_terminal()

        if terminated:
            reward = self.completion_reward
        else:
            reward = self.gate_cost

        if gates_executed > 0:
            to_unlock = {
                act
                for act in self.locked_actions
                if any(q in active_qubits for q in self.cmap_edges[act])
            }
            self.locked_actions -= to_unlock
        else:
            self.locked_actions.add(action)
        self.last_action = action

        return obs, reward, terminated, False, {}

    def _apply_swap(self, action: int) -> None:
        p0, p1 = self.cmap_edges[action]
        l0, l1 = self.physical_to_logical[p0], self.physical_to_logical[p1]

        self.routed_circuit.swap(p0, p1)
        self.physical_to_logical[p0], self.physical_to_logical[p1] = l1, l0

        if l0 != -1:
            self.logical_to_physical[l0] = p1
        if l1 != -1:
            self.logical_to_physical[l1] = p0

    def _execute_front_layer(self) -> tuple[int, set[int]]:
        progress = True
        gates_done = 0
        active_qubits = set()

        while progress:
            progress = False
            for node in list(self.dag.front_layer()):
                indices = [self.qubit_indices[q] for q in node.qargs]
                if len(indices) == 1:
                    p0 = self.logical_to_physical[indices[0]]
                    self.routed_circuit._append(node.op, [p0])  # pyrefly: ignore
                    self.dag.remove_op_node(node)

                    active_qubits.add(p0)
                    gates_done += 1
                    progress = True

                elif len(indices) == 2:
                    p0, p1 = (
                        self.logical_to_physical[indices[0]],
                        self.logical_to_physical[indices[1]],
                    )
                    if (p0, p1) in self.edge_set or (p1, p0) in self.edge_set:
                        self.routed_circuit.append(node.op, [p0, p1])
                        self.dag.remove_op_node(node)

                        active_qubits.update([p0, p1])
                        gates_done += 1
                        progress = True

        return gates_done, active_qubits

    def _get_obs(self) -> dict:
        matrix = np.zeros((len(self.cmap_edges), self.horizon), dtype=np.int8)

        layers = list(self.dag.layers())

        future_layers_gates = []
        for i in range(min(len(layers), self.horizon)):
            layer_gates = []
            for node in layers[i]["graph"].op_nodes():
                if len(node.qargs) == 2:
                    layer_gates.append(
                        (
                            self.qubit_indices[node.qargs[0]],
                            self.qubit_indices[node.qargs[1]],
                        )
                    )
            future_layers_gates.append(layer_gates)

        for edge_idx, (p0, p1) in enumerate(self.cmap_edges):
            for h, gates_in_layer in enumerate(future_layers_gates):
                layer_delta = 0
                for l1, l2 in gates_in_layer:
                    curr_p_a = self.logical_to_physical[l1]
                    curr_p_b = self.logical_to_physical[l2]

                    dist_before = self.distance_matrix[curr_p_a, curr_p_b]

                    new_p_a = (
                        p1 if curr_p_a == p0 else (p0 if curr_p_a == p1 else curr_p_a)
                    )
                    new_p_b = (
                        p1 if curr_p_b == p0 else (p0 if curr_p_b == p1 else curr_p_b)
                    )

                    dist_after = self.distance_matrix[new_p_a, new_p_b]

                    layer_delta += dist_after - dist_before

                matrix[edge_idx, h] = layer_delta

        graph_x, graph_edge_idx = self._build_graph()

        return {
            "matrix": matrix,
            "graph_x": graph_x,
            "graph_edge_idx": graph_edge_idx,
        }

    def _is_terminal(self) -> bool:
        return not self.dag.op_nodes()

    def render(self) -> None:
        if self.render_mode == "ansi":
            print("--- Original ---")
            print(self.circuit)
            print("\n--- Routed ---")
            print(self.routed_circuit)
            print()

    def valid_action_mask(self) -> np.ndarray:
        mask = np.zeros(len(self.cmap_edges), dtype=bool)

        front_physical_qubits = {
            self.logical_to_physical[self.qubit_indices[q]]
            for node in self.dag.front_layer()
            for q in node.qargs
        }

        for i, (p0, p1) in enumerate(self.cmap_edges):
            if p0 in front_physical_qubits or p1 in front_physical_qubits:
                mask[i] = True

        if self.locked_actions:
            mask[list(self.locked_actions)] = False

        if not mask.any():
            self.locked_actions.clear()
            for i, (p0, p1) in enumerate(self.cmap_edges):
                if p0 in front_physical_qubits or p1 in front_physical_qubits:
                    mask[i] = True

        assert mask.any(), "No valid action in mask"

        return mask
