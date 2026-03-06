from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.target import Target
from qiskit.transpiler.coupling import CouplingMap

import torch
import time
from ..model import ValueModel
from ..swap_optimizer import SwapOptimizer, CNOTCircuit
from ..swap_optimizer import SwapOptimizer, CNOTCircuit
from ..batch_weighted_astar_search import BWAS

from qiskit.converters import circuit_to_dag, dag_to_circuit
import torch


class BWASRouting(TransformationPass):

    def __init__(self, coupling_map: CouplingMap, horizon: int, model_path: str, device: str):
        super().__init__()
        self.horizon    = horizon
        self.model_path = model_path
        self.batch_size = 1
        self.device     = device
        self.last_time  = 0.0
        
        if isinstance(coupling_map, Target):
            self.target = coupling_map
            self.coupling_map = self.target.build_coupling_map()
        else:
            self.target = None
            self.coupling_map = coupling_map

    def run(self, dag):
        qc  = dag_to_circuit(dag)
        topology = [edge for edge in self.coupling_map.get_edges() if edge[0] < edge[1]]
        game  = SwapOptimizer(qc.num_qubits, self.horizon, topology)
        model = ValueModel(qc.num_qubits, self.horizon, len(topology))
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        
        bwas = BWAS(model, game, batch_size=self.batch_size)
        cnot_c = CNOTCircuit.from_quantum_circuit(qc)

        state = cnot_c.to_tensor(horizon=self.horizon)
        root_state, _ = game.prune(state)
        
        start_time = time.time()
        path = bwas.search(root_state)
        end_time = time.time()
        
        self.last_time = end_time - start_time
        cnot_c = bwas.insert_swaps(qc=cnot_c, path=path, horizon=self.horizon, topology=topology)
        cnot_c.reconstruct_with_swaps()
        return circuit_to_dag(cnot_c.reconstruct_with_swaps())




