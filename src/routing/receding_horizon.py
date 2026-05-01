from qiskit.converters import circuit_to_dag
from src.circuit_generator import CircuitGenerator
from src.model import BiCircuitGNNDense
from src.routing.bwas_router import BWASRouter
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ApplyLayout, SetLayout
from qiskit.transpiler.layout import Layout
from qiskit import QuantumCircuit
from src.routing.router import Router
from qiskit.dagcircuit import DAGCircuit


class RecedingHorizon(Router):
    def __init__(self, horizon_length: int, step_size: int, router: Router):
        self.horizon_length = horizon_length
        self.step_size = step_size
        self.router = router

    def solve(self, circuit: QuantumCircuit) -> list[int]:
        circuit_dag = circuit_to_dag(circuit)
        layout = Layout.from_qubit_list(circuit.qubits, range(len(circuit.qubits)))
        pm = PassManager()
        pm.append(SetLayout(layout))
        pm.append(ApplyLayout())

        all_actions = []

        while not self._is_terminal(circuit_dag, layout):
            window_circuit = self._get_window(circuit_dag, self.horizon_length)
            actions = self.router.solve(pm.run(window_circuit.to_circuit()))
            for action in actions[: self.step_size]:
                all_actions.append(action)
                self._prune(circuit_dag, layout)
                layout.swap(*self.router.state_handler.get_topology()[action])

        return all_actions

    def _is_terminal(self, circuit_dag: DAGCircuit, layout: Layout) -> bool:
        for node in circuit_dag.topological_op_nodes():
            qargs = node.qargs
            physical_qubits = [layout[q] for q in qargs]
            if not any(
                set(physical_qubits) <= set(edge)
                for edge in self.router.state_handler.get_topology()
            ):
                return False

        return True

    def _prune(self, circuit_dag: DAGCircuit, layout: Layout) -> int:
        resolved_gates = 0
        for node in circuit_dag.front_layer():
            # if node.type == "op":
            qargs = node.qargs
            physical_qargs = [layout[q] for q in qargs]
            if any(
                set(physical_qargs) <= set(edge)
                for edge in self.router.state_handler.get_topology()
            ):
                circuit_dag.remove_op_node(node)
                resolved_gates += 1

        if resolved_gates == 0:
            return 0

        return resolved_gates + self._prune(circuit_dag, layout)

    def _get_window(self, circuit: DAGCircuit, horizon: int) -> DAGCircuit:
        window_dag = circuit.copy_empty_like()

        num_gates_added = 0
        for node in circuit.topological_op_nodes():
            window_dag.apply_operation_back(node.op, node.qargs, node.cargs)
            num_gates_added += 1
            if num_gates_added == horizon:
                break

        return window_dag


if __name__ == "__main__":
    from qiskit.qpy import load, dump
    import torch
    from src.states.circuit_graph_state_handler import CircuitGraphStateHandler
    from src.model import BiCircuitGNN
    from src.routing.swap_inserter.swap_inserter import SwapInserter
    from src.states.dense_circuit_graph_state_handler import (
        DenseCircuitGraphStateHandler,
    )
    from src.routing.rl_routing_pass import RlRoutingPass
    from qiskit.transpiler import CouplingMap, PassManager
    from qiskit.quantum_info import Operator

    n_qubits = 6
    horizon = 100
    coupling_map = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    circuit = CircuitGenerator.generate_random_circuit(6, gateset={"cx"}, num_gates=30)
    with open("circuits/dud2.qpy", "wb") as f:
        dump(circuit, f)

    # with open("circuits/dud3.qpy", "rb") as f:
    #     circuit = load(f)[0]

    print(circuit)

    state_handler = DenseCircuitGraphStateHandler(n_qubits, coupling_map)
    model = BiCircuitGNNDense(n_qubits)
    model.load_state_dict(
        torch.load(
            "models/dense_graph/difficulty18_iteration82480.pt", map_location=device
        )
    )

    router = RecedingHorizon(
        horizon_length=18,
        step_size=9,
        router=BWASRouter(model.to(device), state_handler),
    )

    swap_inserter = SwapInserter(coupling_map, num_qubits=n_qubits)

    rl_pass = RlRoutingPass(router, swap_inserter)
    pm = PassManager(rl_pass)
    final_circuit = pm.run(circuit)

    print("Original circuit:")
    print(circuit)
    print("Routed circuit:")
    print(final_circuit)

    org_op = Operator.from_circuit(circuit)
    routed_op = Operator.from_circuit(final_circuit)

    assert routed_op.equiv(org_op), (
        "The original and routed circuits are not equivalent!"
    )
