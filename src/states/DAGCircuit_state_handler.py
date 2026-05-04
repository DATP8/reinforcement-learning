from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ApplyLayout
from qiskit.transpiler.passes import SetLayout
from src.states.state_handler import Batchable
from copy import deepcopy
from qiskit.transpiler import Layout
from src.circuit_generator import CircuitGenerator
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import CouplingMap
from src.states.state_handler import StateHandler
from qiskit.dagcircuit import DAGCircuit, DAGOpNode


class DAGCircuitState:
    def __init__(self, dag: DAGCircuit, layout: Layout | None = None):
        self.dag = dag
        self.layout = (
            Layout.from_qubit_list(dag.qubits, range(len(dag.qubits)))
            if layout is None
            else layout
        )
        self.pm = PassManager([SetLayout(self.layout), ApplyLayout()])

    def __str__(self) -> str:
        return f"DAGCircuitState:\n{self.get_circuit()}"

    def get_dag(self) -> DAGCircuit:
        return self.dag

    def get_layout(self) -> Layout:
        return self.layout

    def get_circuit(self) -> QuantumCircuit:
        return self.pm.run(dag_to_circuit(self.dag))


class DAGCircuitStateHandler(StateHandler[DAGCircuitState]):
    def __init__(
        self, num_qubits: int, coupling_map: list[tuple[int, int]] | CouplingMap
    ):
        self.num_qubits = num_qubits
        self.cm = (
            coupling_map
            if isinstance(coupling_map, CouplingMap)
            else CouplingMap(coupling_map)
        )
        self.cm.make_symmetric()
        self.swaps = list(
            dict.fromkeys(frozenset(edge) for edge in self.cm.get_edges())
        )
        self.actions = list(range(len(self.swaps)))

    def get_possible_actions(self, state: DAGCircuitState) -> list[int]:
        return self.actions

    def get_next_state(self, state: DAGCircuitState, action: int) -> DAGCircuitState:
        new_state, _ = self._get_removed_gates(state)
        new_state.layout.swap(*self.swaps[action])
        return new_state

    def is_terminal(self, state: DAGCircuitState) -> bool:
        for node in state.dag.topological_op_nodes():
            qargs = node.qargs
            physical_qubits = [state.layout[q] for q in qargs]
            if not any(set(physical_qubits) <= set(edge) for edge in self.swaps):
                return False

        return True

    def get_action_cost(self, state: DAGCircuitState, action: int) -> float:
        _, resolved_gates = self._get_removed_gates(state)
        locked_qubits = {i: False for i in range(self.num_qubits)}
        for node in resolved_gates[::-1]:
            if all(locked_qubits.values()):
                return 1.0

            qubits = [state.layout[q] for q in node.qargs]
            
            if any(locked_qubits[q] for q in qubits):
                locked_qubits.update({q: True for q in qubits})
                continue

            if len(qubits) == 2:
                if set(qubits) == set(self.swaps[action]):
                    return 0.5

            locked_qubits.update({q: True for q in qubits})

        return 1.0

    def _get_removed_gates(
        self, state: DAGCircuitState
    ) -> tuple[DAGCircuitState, list[DAGOpNode]]:
        new_state = deepcopy(state)
        resolved_gates = []
        flag = True
        while flag:
            local_resolved_gates = []
            for node in new_state.dag.front_layer():
                qargs = node.qargs
                physical_qargs = [new_state.layout[q] for q in qargs]
                if any(set(physical_qargs) <= set(edge) for edge in self.swaps):
                    new_state.dag.remove_op_node(node)
                    local_resolved_gates.append(node)

            if len(local_resolved_gates) == 0:
                flag = False

            resolved_gates.extend(local_resolved_gates)

        return new_state, resolved_gates

    def prune(self, state: DAGCircuitState) -> tuple[DAGCircuitState, int]:
        new_state, resolved_gates = self._get_removed_gates(state)
        return new_state, len(resolved_gates)

    def get_random_state(self, difficulty: int) -> DAGCircuitState:
        return DAGCircuitState(
            CircuitGenerator.generate_random_circuit(
                self.num_qubits, difficulty, {"cx"}
            ).to_dag()
        )

    def state_from(self, circuit: QuantumCircuit) -> DAGCircuitState:
        return DAGCircuitState(circuit_to_dag(circuit))

    def get_num_qubits(self) -> int:
        return self.num_qubits

    def batch_states(self, states: Batchable[DAGCircuitState]) -> DAGCircuitState:
        raise NotImplementedError

    def get_topology(self) -> list[tuple[int, int]]:
        raise NotImplementedError


if __name__ == "__main__":
    n_qubits = 6
    coupling_map = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    state_handler = DAGCircuitStateHandler(n_qubits, coupling_map)

    qc = QuantumCircuit(n_qubits)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.cx(0, 3)
    # qc.cx(0, 1)
    # qc.cx(3, 5)
    # qc.cx(0, 4)
    # qc.cx(0, 1)

    root_state = state_handler.state_from(qc)
    next_state1 = state_handler.get_next_state(root_state, 0)
    next_state2 = state_handler.get_next_state(next_state1, 1)
    next_state3, _ = state_handler.prune(next_state2)

    print(qc)
    print(next_state1)
    print(next_state2)
    print(next_state3)
    print(state_handler.is_terminal(root_state))
    print(state_handler.is_terminal(next_state1))
    print(state_handler.is_terminal(next_state2))
    print(state_handler.is_terminal(next_state3))
