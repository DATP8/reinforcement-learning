from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.basepasses import TransformationPass
from sb3_contrib import MaskablePPO

from src.ppo_util import route_circuit


class AgenticRlRoutingPass(TransformationPass):
    def __init__(
        self,
        model: MaskablePPO,
        coupling_map: CouplingMap,
        samples: int | None = None,
        time_limit_s: float | None = None,
    ):
        super().__init__()
        self.model = model
        self.cmap = coupling_map
        self._samples = samples
        self._time_limit_s = time_limit_s

    def run(self, dag: DAGCircuit):
        new_dag, layout = route_circuit(
            self.model, dag, self._samples, self._time_limit_s
        )
        self.property_set["final_layout"] = layout
        return new_dag
