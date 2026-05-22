from src.routing.router import Router
from qiskit.transpiler import Layout
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.basepasses import TransformationPass

from src.routing.swap_inserter.swap_inserter import SwapInserter  # pyrefly: ignore

from qiskit.converters import dag_to_circuit


class RlRoutingPass(TransformationPass):
    def __init__(
        self,
        router: Router,
        swap_inserter: SwapInserter,
    ):
        super().__init__()
        self.router = router
        self.swap_inserter = swap_inserter

    def run(self, dag):
        qc = dag_to_circuit(dag)
        actions = self.router.solve(qc)
        new_qc, init, final = self.swap_inserter.build_circuit_from_solution(
            actions, qc
        )

        new_dag = circuit_to_dag(new_qc)

        layout = Layout(
            {dag.qubits[org_q]: final_q for org_q, final_q in enumerate(final)}
        )

        self.property_set["final_layout"] = (
            layout
            if (prev := self.property_set["final_layout"]) is None
            else prev.compose(layout, new_dag.qubits)
        )

        return new_dag