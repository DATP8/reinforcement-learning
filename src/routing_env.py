from numpy import ndarray
from qiskit.transpiler import Layout
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
        unique_edges = list({tuple(sorted(edge)) for edge in cmap.get_edges()})
        self._cmap_edges = np.array(unique_edges)
        self._edge_set = frozenset(unique_edges)
        self._distance_matrix: np.ndarray = cmap.distance_matrix  # pyrefly: ignore

        self._num_phys_qubits = cmap.size()
        self._horizon = horizon
        self._render_mode = render_mode

        self._num_logic_qubits = 0
        self._current_difficulty = initial_difficulty

        self.action_space = spaces.Discrete(len(self._cmap_edges))
        self.observation_space = spaces.Dict(
            {
                "matrix": spaces.Box(
                    low=-2,
                    high=2,
                    shape=(len(self._cmap_edges), self._horizon),
                    dtype=np.int8,
                ),
                "graph_x": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self._num_phys_qubits, 3),  # example features
                    dtype=np.float32,
                ),
                "graph_edge_idx": spaces.Box(
                    low=0,
                    high=self._num_phys_qubits,
                    shape=(2, 100),  # max edges (pad if needed)
                    dtype=np.int64,
                ),
            }
        )

        self._completion_reward = 10
        self._gate_cost = -0.1
        self._reduced_gate_cost = -0.1
        
        self._locked_actions = set()
        self._inserted_swaps = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}

        if "circuit" in options and "layout" in options:
            self.circuit = options["circuit"]
            self._num_logic_qubits = len(self.circuit.qubits)
            self.qubit_indices = {q: i for i, q in enumerate(self.circuit.qubits)}
            
            self.logical_to_physical = np.zeros(self._num_logic_qubits, dtype=np.int64)
            self.physical_to_logical = np.full(self._num_phys_qubits, -1, dtype=np.int64)

            virtual_bits = options["layout"].get_virtual_bits()
            for v_qubit, p_idx in virtual_bits.items():
                if v_qubit in self.qubit_indices:
                    l_idx = self.qubit_indices[v_qubit]
                    self.logical_to_physical[l_idx] = p_idx
                    self.physical_to_logical[p_idx] = l_idx

            self.dag = circuit_to_dag(self.circuit)
            self.routed_circuit = QuantumCircuit(self._num_phys_qubits)
            self._execute_front_layer()
        else:
            while True:
                self._num_logic_qubits = int(self.np_random.integers(2, self._num_phys_qubits + 1))
                self.circuit = self._generate_random_circuit(self._num_logic_qubits, self._current_difficulty)
                self.qubit_indices = {q: i for i, q in enumerate(self.circuit.qubits)}
                self.dag = circuit_to_dag(self.circuit)
                self.routed_circuit = QuantumCircuit(self._num_phys_qubits)
                self.logical_to_physical, self.physical_to_logical = self._generate_random_mapping()

                self._execute_front_layer()
                if not self.is_terminal():
                    break

        self._locked_actions = set()
        self._inserted_swaps = 0
        return self._get_obs(), {}

    def _build_graph(self):
        num_q = self._num_logic_qubits

        # Interaction tracking
        interaction_counts = np.zeros((num_q, num_q), dtype=np.float32)

        # Iterate over remaining DAG nodes
        for node in self.dag.op_nodes():
            qargs = node.qargs

            if len(qargs) == 2:
                q1 = self.qubit_indices[qargs[0]]
                q2 = self.qubit_indices[qargs[1]]

                interaction_counts[q1, q2] += 1
                interaction_counts[q2, q1] += 1

        # Node features
        # Features per logical qubit:
        # [physical_position, total_interactions, avg_distance_to_partners]

        # Fill for phys qubits, then just update for logical ones
        x = np.zeros((self._num_phys_qubits, 3), dtype=np.float32)

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

                        d = self._distance_matrix[p1][p2]
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
        if edges:
            edge_array = np.array(edges, dtype=np.int64).T
            n_edges = min(edge_array.shape[1], MAX_EDGES)
            edge_index[:, :n_edges] = edge_array[:, :n_edges]

        return x, edge_index

    def _generate_random_mapping(self):
        chosen = self.np_random.choice(
            self._num_phys_qubits, size=self._num_logic_qubits, replace=False
        ).astype(np.int64)
        l2p = chosen

        p2l = np.full(self._num_phys_qubits, -1, dtype=np.int64)
        p2l[l2p] = np.arange(self._num_logic_qubits, dtype=np.int64)  # pyrefly: ignore
        return l2p, p2l

    def _generate_random_circuit(
        self, num_qubits: int, num_gates: int
    ) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        # gate_set = ["h", "s", "cx"]
        for _ in range(num_gates):
            q0, q1 = self.np_random.choice(num_qubits, size=2, replace=False)
            qc.cx(int(q0), int(q1))
        return qc

    def step(self, action: int | np.ndarray):
        action = int(action)

        self._apply_swap(action)
        gates_executed, active_qubits = self._execute_front_layer()

        obs = self._get_obs()
        terminated = self.is_terminal()

        if terminated:
            reward = self._completion_reward
        else:
            reward = self._gate_cost

        if gates_executed > 0:
            to_unlock = {
                act
                for act in self._locked_actions
                if any(q in active_qubits for q in self._cmap_edges[act])
            }
            self._locked_actions -= to_unlock
        else:
            self._locked_actions.add(action)
        self.last_action = action

        max_swaps = min(2*self._current_difficulty,128)
        truncated = self._inserted_swaps == max_swaps

        return obs, reward, terminated, truncated, {}

    def _apply_swap(self, action: int) -> None:
        p0, p1 = self._cmap_edges[action]
        l0, l1 = self.physical_to_logical[p0], self.physical_to_logical[p1]

        self.routed_circuit.swap(p0, p1)
        self.physical_to_logical[p0], self.physical_to_logical[p1] = l1, l0

        if l0 != -1:
            self.logical_to_physical[l0] = p1
        if l1 != -1:
            self.logical_to_physical[l1] = p0
            
        self._inserted_swaps += 1

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
                    if (p0, p1) in self._edge_set or (p1, p0) in self._edge_set:
                        self.routed_circuit.append(node.op, [p0, p1])
                        self.dag.remove_op_node(node)

                        active_qubits.update([p0, p1])
                        gates_done += 1
                        progress = True

        return gates_done, active_qubits

    def _get_obs(self) -> dict:
        matrix = self._build_matrix()
        graph_x, graph_edge_idx = self._build_graph()

        return {
            "matrix": matrix,
            "graph_x": graph_x,
            "graph_edge_idx": graph_edge_idx,
        }
        
    def _build_matrix(self):
        E = len(self._cmap_edges)
        matrix = np.zeros((E, self._horizon), dtype=np.int8)

        layers = list(self.dag.layers())
        
        p0_arr = self._cmap_edges[:, 0, None]
        p1_arr = self._cmap_edges[:, 1, None]

        for h in range(min(len(layers), self._horizon)):
            gate_nodes = [n for n in layers[h]["graph"].op_nodes() if len(n.qargs) == 2]
            if not gate_nodes:
                continue
                
            num_gates = len(gate_nodes)
            l1_arr = np.fromiter((self.qubit_indices[n.qargs[0]] for n in gate_nodes), count=num_gates, dtype=np.int64)
            l2_arr = np.fromiter((self.qubit_indices[n.qargs[1]] for n in gate_nodes), count=num_gates, dtype=np.int64)
            
            curr_pa = self.logical_to_physical[l1_arr]
            curr_pb = self.logical_to_physical[l2_arr]
            
            dist_before = self._distance_matrix[curr_pa, curr_pb]
            
            pa_exp = np.tile(curr_pa, (E, 1))
            pb_exp = np.tile(curr_pb, (E, 1))
            
            new_pa = np.where(pa_exp == p0_arr, p1_arr, np.where(pa_exp == p1_arr, p0_arr, pa_exp))
            new_pb = np.where(pb_exp == p0_arr, p1_arr, np.where(pb_exp == p1_arr, p0_arr, pb_exp))
            
            dist_after = self._distance_matrix[new_pa, new_pb]
            
            matrix[:, h] = np.sum(dist_after - dist_before, axis=1)
        return matrix

    def is_terminal(self) -> bool:
        return not self.dag.op_nodes()

    def render(self) -> None:
        if self._render_mode == "ansi":
            print("--- Original ---")
            print(self.circuit)
            print("\n--- Routed ---")
            print(self.routed_circuit)
            print()

    def valid_action_mask(self) -> np.ndarray:
        front_logical = [
            self.qubit_indices[q]
            for node in self.dag.front_layer()
            for q in node.qargs
        ]
        front_physical = self.logical_to_physical[front_logical]

        mask = np.any(np.isin(self._cmap_edges, front_physical), axis=1)

        if self._locked_actions:
            mask[list(self._locked_actions)] = False

        if not mask.any():
            self._locked_actions.clear()
            mask = np.any(np.isin(self._cmap_edges, front_physical), axis=1)

        assert mask.any(), "No valid action in mask"

        return mask
