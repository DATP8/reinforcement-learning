from qiskit.transpiler import Layout
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.basepasses import TransformationPass

from src.routing.swap_inserter.swap_inserter import SwapInserter  # pyrefly: ignore

from qiskit.converters import dag_to_circuit


class RlRoutingPass(TransformationPass):
    def __init__(
        self,
        router,
        swap_inserter: SwapInserter,
        name: str,
    ):
        super().__init__()
        self.router = router
        self.swap_inserter = swap_inserter
        self.model_name = name

    def get_name(self):
        return self.model_name

    def run(self, dag):
        qc = dag_to_circuit(dag)
        actions = self.router.solve(qc)
        new_qc, init, final = self.swap_inserter.build_circuit_from_solution(
            actions, qc
        )

        self.property_set["final_layout"] = Layout(
            {dag.qubits[org_q]: final_q for org_q, final_q in enumerate(final)}
        )
        return circuit_to_dag(new_qc)
