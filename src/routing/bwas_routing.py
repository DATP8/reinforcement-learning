from qiskit.transpiler import Layout
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.target import Target
from qiskit.transpiler.coupling import CouplingMap

import time
import torch.nn as nn

from ..states.tensor_state import TensorState
from ..states.state_handler import StateHandler
from ..batch_weighted_astar_search import BWAS
from .router import Router


from qiskit.converters import dag_to_circuit


class BWASRouting(TransformationPass):
    def __init__(
        self,
        model: nn.Module,
        state_handler: StateHandler,
        router: Router,
        name: str,
    ):
        super().__init__()
        self.state_handler = state_handler
        self.router = router
        self.model = model
        self.batch_size = 1
        self.last_time = 0.0
        self.model_name = name


    def get_name(self):
        return self.model_name

    def run(self, dag):
        qc = dag_to_circuit(dag)
        bwas = BWAS(self.model, self.state_handler)

        start_time = time.time()
        path = bwas.search(qc)
        end_time = time.time()

        self.last_time = end_time - start_time
        new_qc, init, final = self.router.build_circuit_from_solution(path, qc)

        self.property_set["final_layout"] = Layout(
            {dag.qubits[org_q]: final_q for org_q, final_q in enumerate(final)}
        )
        return circuit_to_dag(new_qc)
