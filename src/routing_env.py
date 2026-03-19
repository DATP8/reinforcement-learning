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
    ) -> None:
        super().__init__()
        self.cmap_edges = list({tuple(sorted(edge)) for edge in cmap.get_edges()})
        self.edge_set = frozenset(self.cmap_edges)
        self.distance_matrix: np.ndarray = cmap.distance_matrix  # pyrefly: ignore

        self.num_phys_qubits = cmap.size()
        self.horizon = horizon
        self.render_mode = render_mode

        self.num_logic_qubits = None
        self.current_difficulty = 1

        self.action_space = spaces.Discrete(len(self.cmap_edges))
        self.observation_space = spaces.Box(
            low=-2, high=2, shape=(len(self.cmap_edges), self.horizon), dtype=np.int8
        )

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
        gate_set = ["h", "s", "cx"]
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
        dist_before = self._get_total_distance_to_next_gates()

        self._apply_swap(action)
        gates_executed, active_qubits = self._execute_front_layer()
        dist_after = self._get_total_distance_to_next_gates()

        obs = self._get_obs()
        terminated = self._is_terminal()

        if gates_executed > 0:
            reward = gates_executed * 10.0
            to_unlock = {
                act
                for act in self.locked_actions
                if any(q in active_qubits for q in self.cmap_edges[act])
            }
            self.locked_actions -= to_unlock
        else:
            reward = (dist_before - dist_after) - 0.1
            self.locked_actions.add(action)

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
                    self.routed_circuit.append(node.op, [p0])
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

    def _get_total_distance_to_next_gates(self) -> float:
        total_dist = 0.0
        count = 0
        for node in self.dag.topological_op_nodes():
            if len(node.qargs) == 2:
                l0 = self.qubit_indices[node.qargs[0]]
                l1 = self.qubit_indices[node.qargs[1]]
                p0 = self.logical_to_physical[l0]
                p1 = self.logical_to_physical[l1]

                total_dist += self.distance_matrix[p0, p1].item()
                count += 1

                if count >= self.horizon:
                    break

        return total_dist

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((len(self.cmap_edges), self.horizon), dtype=np.int8)

        future_gates = []
        for node in self.dag.topological_op_nodes():
            if len(node.qargs) == 2:
                future_gates.append(
                    (
                        self.qubit_indices[node.qargs[0]],
                        self.qubit_indices[node.qargs[1]],
                    )
                )
            if len(future_gates) == self.horizon:
                break

        for edge_idx, (p0, p1) in enumerate(self.cmap_edges):
            for layer_idx, (l_a, l_b) in enumerate(future_gates):
                curr_p_a = self.logical_to_physical[l_a]
                curr_p_b = self.logical_to_physical[l_b]

                dist_before = self.distance_matrix[curr_p_a, curr_p_b]

                new_p_a = p1 if curr_p_a == p0 else (p0 if curr_p_a == p1 else curr_p_a)
                new_p_b = p1 if curr_p_b == p0 else (p0 if curr_p_b == p1 else curr_p_b)

                dist_after = self.distance_matrix[new_p_a, new_p_b]

                obs[edge_idx, layer_idx] = dist_before - dist_after

        return obs

    def _is_terminal(self) -> bool:
        return not self.dag.op_nodes()

    def render(self) -> None:
        if self.render_mode == "human":
            print("--- Original ---")
            print(self.circuit)
            print("\n--- Routed ---")
            print(self.routed_circuit)
            print()

    def valid_action_mask(self) -> np.ndarray:
        mask = np.zeros(len(self.cmap_edges), dtype=bool)

        front_physical_qubits = set()
        for node in self.dag.front_layer():
            for q in node.qargs:
                l_idx = self.qubit_indices[q]
                p_idx = self.logical_to_physical[l_idx]
                front_physical_qubits.add(p_idx)

        for i, (p0, p1) in enumerate(self.cmap_edges):
            if p0 in front_physical_qubits or p1 in front_physical_qubits:
                mask[i] = True

        for act in self.locked_actions:
            mask[act] = False

        if not mask.any():
            mask.fill(True)
            for act in self.locked_actions:
                mask[act] = False

        if not mask.any():
            mask.fill(True)
            self.locked_actions.clear()

        return mask
