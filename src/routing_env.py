from gymnasium import spaces
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.converters import circuit_to_dag
import numpy as np
import gymnasium

class RoutingEnv(gymnasium.Env):
    def __init__(
        self,
        num_qubits: int,
        coupling_map: CouplingMap,
        num_active_swaps: int,
        horizon: int,
        initial_difficulty: int,
        max_difficulty: int,
        depth_slope: int,
        max_depth: int,
        render_mode: str | None = None
    ) -> None:
        super().__init__()
        self._num_qubits = num_qubits
        self._num_logic_qubits = num_qubits
        self._num_phys_qubits = num_qubits
        self._coupling_map = coupling_map
        self._num_active_swaps = num_active_swaps
        self._horizon = horizon
        self._difficulty = initial_difficulty
        self._max_difficulty = max_difficulty
        self._depth_slope = depth_slope
        self._max_depth = max_depth
        self._render_mode = render_mode
        self._distance_matrix: np.ndarray = coupling_map.distance_matrix # pyrefly: ignore
        
        unique_edges = list({tuple(sorted(edge)) for edge in coupling_map.get_edges()})
        self._cmap_edges = np.array(unique_edges)
        self._edge_set = frozenset(unique_edges)
        
        self._active_swaps = []
        self.locations = np.arange(self._num_qubits, dtype=np.int64)
        self._qubits = np.arange(self._num_qubits, dtype=np.int64)
        
        self.action_space = spaces.Discrete(self._num_active_swaps)
        self.observation_space = spaces.Box(
            low=-2,
            high=2,
            shape=(len(self._cmap_edges), self._horizon),
            dtype=np.int8,
        )
        
        #self.observation_space = spaces.Dict(
        #    {
        #        "matrix": spaces.Box(
        #            low=-2,
        #            high=2,
        #            shape=(len(self._cmap_edges), self._horizon),
        #            dtype=np.int8,
        #        ),
        #        "graph_x": spaces.Box(
        #            low=-np.inf,
        #            high=np.inf,
        #            shape=(self._num_phys_qubits, 3),
        #            dtype=np.float32,
        #        ),
        #        "graph_edge_idx": spaces.Box(
        #            low=0,
        #            high=self._num_phys_qubits,
        #            shape=(2, 100),
        #            dtype=np.int64,
        #        ),
        #    }
        #)

        self._reward_value = 1.0

        self._visited_layouts = set()
        self._inserted_swaps = 0

    def _reset_internals(self):
        self.locations = np.arange(self._num_qubits, dtype=np.int64)
        self._qubits = np.arange(self._num_qubits, dtype=np.int64)
        self._active_swaps = []
        self._visited_layouts = set()
        self._sucess = not bool(self._dag.op_nodes())
        
        # Scale the completion reward with the circuit depth to offset swap penalties
        self._completion_reward = 1.0 + (self._depth * 0.5)
        self._reward_value = self._completion_reward if self.is_terminal() else 0.0

    def set_difficulty(self, difficulty: int):
        self._difficulty = difficulty

    def get_difficulty(self) -> int:
        return self._difficulty

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}

        # Sample a random difficulty when curriculum learning done
        if self._difficulty >= self._max_difficulty:
            sampled_diff = int(self.np_random.integers(1, self._max_difficulty + 1))
        else:
            sampled_diff = self._difficulty
            
        self._depth = min(self._depth_slope * sampled_diff, self._max_depth)
        
        provided_circuit: QuantumCircuit = options.get("circuit") # pyrefly: ignore
        self._is_eval_circuit = provided_circuit is not None
        
        if self._is_eval_circuit:
            self._remaining_swaps = len(provided_circuit) * 10
        else:
            self._remaining_swaps = self._depth * self._num_qubits

        self.routed_circuit = QuantumCircuit(self._num_qubits)
        self._inserted_swaps = 0

        while True:
            if provided_circuit is not None:
                self._circuit = provided_circuit.copy()
            else:
                self._circuit = self._generate_random_circuit(self._depth)
            self._qubit_indices = {q: i for i, q in enumerate(self._circuit.qubits)}
            self._dag = circuit_to_dag(self._circuit)
            self._reset_internals()

            if provided_circuit is None:
                # Randomly shuffle layout initially during training
                self.np_random.shuffle(self.locations)
                
            for q, loc in enumerate(self.locations):
                self._qubits[loc] = q

            self._execute_front_layer()
            self._visited_layouts = {tuple(self._qubits)}
            self._reward_value = self._completion_reward if self.is_terminal() else 0.0
            
            if provided_circuit is not None or not self.is_terminal():
                break
        
        self._cache_observation()
        return self._get_obs(), {}

    def _generate_random_circuit(self, num_gates: int) -> QuantumCircuit:
        qc = QuantumCircuit(self._num_qubits)
        for _ in range(num_gates):
            q0, q1 = self.np_random.choice(self._num_qubits, size=2, replace=False)
            qc.cx(int(q0), int(q1))
        return qc

    def step(self, action: int | np.ndarray):
        action = int(action)
        penalty = 0.0

        if action < len(self._cmap_edges):
            p0, p1 = self._cmap_edges[action]
            l0, l1 = self._qubits[p0], self._qubits[p1]

            penalty = 0.1
            
            self.routed_circuit.swap(p0, p1)
            self._qubits[p0], self._qubits[p1] = l1, l0
            self.locations[l0] = p1
            self.locations[l1] = p0

            self._inserted_swaps += 1

        gates_executed = self._execute_front_layer()
        
        if gates_executed > 0:
            self._visited_layouts.clear()
            
        self._visited_layouts.add(tuple(self._qubits))

        self._remaining_swaps = max(0, self._remaining_swaps - 1)

        achieved = self._completion_reward if self.is_terminal() else 0.0
        self._reward_value = achieved - penalty

        self._cache_observation()

        terminated = self.is_terminal()
        truncated = False
        if self._remaining_swaps == 0 and not self.is_terminal():
            truncated = True

        return self._get_obs(), self._reward_value, terminated, truncated, {}

    def _cache_observation(self):
        #self._cached_gnn_obs = self._build_graph()
        self._cached_matrix = self._build_matrix()

    def _execute_front_layer(self) -> int:
        progress = True
        gates_executed = 0
        while progress:
            progress = False
            for node in list(self._dag.front_layer()):
                indices = [self._qubit_indices[q] for q in node.qargs]
                if len(indices) == 1:
                    p0 = self.locations[indices[0]]
                    self.routed_circuit._append(node.op, [p0]) # pyrefly: ignore
                    self._dag.remove_op_node(node)
                    progress = True
                else:
                    p0, p1 = self.locations[indices[0]], self.locations[indices[1]]
                    if (p0, p1) in self._edge_set or (p1, p0) in self._edge_set:
                        self.routed_circuit.append(node.op, [p0, p1]) # pyrefly: ignore
                        self._dag.remove_op_node(node)
                        gates_executed += 1
                        progress = True
        return gates_executed

    def _get_obs(self):
        return self._cached_matrix
        #graph_x, graph_edge_idx = self._cached_gnn_obs
        #return {
        #    "matrix": self._cached_matrix,
        #    "graph_x": graph_x,
        #    "graph_edge_idx": graph_edge_idx,
        #}

    def _build_matrix(self):
        E = len(self._cmap_edges)
        matrix = np.zeros((E, self._horizon), dtype=np.int8)

        layers = list(self._dag.layers())
        p0_arr = self._cmap_edges[:, 0, None]
        p1_arr = self._cmap_edges[:, 1, None]

        for h in range(min(len(layers), self._horizon)):
            layer = layers[h]
            graph = layer["graph"] if isinstance(layer, dict) else layer
            gate_nodes = [n for n in graph.op_nodes() if len(n.qargs) == 2]
            if not gate_nodes:
                continue

            num_gates = len(gate_nodes)
            l1_arr = np.fromiter(
                (self._qubit_indices[n.qargs[0]] for n in gate_nodes),
                count=num_gates,
                dtype=np.int64,
            )
            l2_arr = np.fromiter(
                (self._qubit_indices[n.qargs[1]] for n in gate_nodes),
                count=num_gates,
                dtype=np.int64,
            )

            curr_pa = self.locations[l1_arr]
            curr_pb = self.locations[l2_arr]
            dist_before = self._distance_matrix[curr_pa, curr_pb]
            pa_exp = np.tile(curr_pa, (E, 1))
            pb_exp = np.tile(curr_pb, (E, 1))
            new_pa = np.where(pa_exp == p0_arr, p1_arr, np.where(pa_exp == p1_arr, p0_arr, pa_exp))
            new_pb = np.where(pb_exp == p0_arr, p1_arr, np.where(pb_exp == p1_arr, p0_arr, pb_exp))
            dist_after = self._distance_matrix[new_pa, new_pb]

            matrix[:, h] = np.sum(dist_after - dist_before, axis=1)
        return matrix
    
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
            phys = self.locations[q]
            total_interactions = interaction_counts[q].sum()

            if total_interactions > 0:
                dists = []
                for other_q in range(num_q):
                    if interaction_counts[q, other_q] > 0:
                        p1 = self.locations[q]
                        p2 = self.locations[other_q]
                        dists.append(self._distance_matrix[p1][p2])
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
        front_logical = [
            self._qubit_indices[q] for node in self._dag.front_layer() for q in node.qargs
        ]
        front_physical = self.locations[front_logical]
        mask = np.any(np.isin(self._cmap_edges, front_physical), axis=1)

        for i in range(len(self._cmap_edges)):
            if mask[i]:
                p0, p1 = self._cmap_edges[i]
                l0, l1 = self._qubits[p0], self._qubits[p1]
                new_qubits = self._qubits.copy()
                new_qubits[p0], new_qubits[p1] = l1, l0
                if tuple(new_qubits) in self._visited_layouts:
                    mask[i] = False

        if not mask.any():
            self._visited_layouts.clear()
            self._visited_layouts.add(tuple(self._qubits))
            mask = np.any(np.isin(self._cmap_edges, front_physical), axis=1)
            
        assert mask.any(), "No valid action in mask"
        
        # Pad up to num_active_swaps if needed
        diff = self._num_active_swaps - len(mask)
        if diff > 0:
            mask = np.pad(mask, (0, diff), constant_values=False)
        return mask[:self._num_active_swaps]
    
    def is_terminal(self) -> bool:
        return not bool(self._dag.op_nodes())

    def render(self) -> None:
        if self._render_mode == "ansi":
            print("--- Original ---")
            print(self._circuit)
            print("\n--- Routed ---")
            print(self.routed_circuit)
            print()
