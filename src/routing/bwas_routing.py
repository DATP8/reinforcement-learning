from qiskit.transpiler import Layout
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.target import Target
from qiskit.transpiler.coupling import CouplingMap

import time
import torch.nn as nn

from ..tensor_state import TensorState
from ..batch_weighted_astar_search import BWAS
from ..state_handler import StateHandler


from qiskit.converters import dag_to_circuit


class BWASRouting(TransformationPass):

    def __init__(self, coupling_map: CouplingMap, horizon: int, model: nn.Module, state_handler: StateHandler, name: str):
        super().__init__()
        self.horizon       = horizon
        self.state_handler = state_handler
        self.model         = model
        self.batch_size    = 1
        self.last_time     = 0.0
        self.model_name    = name
        
        if isinstance(coupling_map, Target):
            self.target = coupling_map
            self.coupling_map = self.target.build_coupling_map()
        else:
            self.target = None
            self.coupling_map = coupling_map
    
    def get_name(self):
        return self.model_name

    def run(self, dag):
        qc  = dag_to_circuit(dag)
        topology = [edge for edge in self.coupling_map.get_edges() if edge[0] < edge[1]]
     
        bwas = BWAS(self.model, self.state_handler, batch_size=self.batch_size)

        state = TensorState.from_quantum_circuit(qc, horizon=self.horizon)
        root_state, _ = self.state_handler.prune(state)
        
        start_time = time.time()
        path = bwas.search(root_state)
        end_time = time.time()
        
        self.last_time = end_time - start_time
        new_qc, qubit_map = state.as_subclass(TensorState).insert_swaps(qc, path, topology, self.state_handler)

        self.property_set['final_layout'] = Layout({
            dag.qubits[org_q]: final_q
            for org_q, final_q in enumerate(qubit_map)
        })

        return circuit_to_dag(new_qc)




