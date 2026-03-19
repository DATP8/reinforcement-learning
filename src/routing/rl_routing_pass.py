from qiskit.transpiler import Layout
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.basepasses import TransformationPass

import time

from .rl_router import RlRouter
from .swap_inserter.swap_inserter import SwapInserter

from qiskit.converters import dag_to_circuit 
class RlRoutingPass(TransformationPass):
    def __init__(
        self,
        router: RlRouter,
        swap_inserter: SwapInserter,
        name: str,
    ):
        super().__init__()
        self.router = router
        self.swap_inserter = swap_inserter
        self.last_time = 0.0
        self.model_name = name


    def get_name(self):
        return self.model_name

    def run(self, dag):
        qc = dag_to_circuit(dag)

        state = self.router.state_handler.state_from(qc)
        root_state, _ = self.router.state_handler.prune(state)

        start_time = time.time()
        path = self.router.search(root_state)
        end_time = time.time()

        self.last_time = end_time - start_time
        new_qc, init, final = self.swap_inserter.build_circuit_from_solution(path, qc)

        self.property_set["final_layout"] = Layout(
            {dag.qubits[org_q]: final_q for org_q, final_q in enumerate(final)}
        )
        return circuit_to_dag(new_qc)
