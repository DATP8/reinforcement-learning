from qiskit.transpiler import CouplingMap
from sb3_contrib import MaskablePPO
from src.gym_test import route_circuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit


class AgenticRlRoutingPass(TransformationPass):
    def __init__(self, model: MaskablePPO, cmap: CouplingMap):
        super().__init__()
        self.model = model
        self.cmap = cmap

    def run(self, dag: DAGCircuit):
        new_dag, layout = route_circuit(self.model, dag)
        self.property_set["final_layout"] = layout
        return new_dag
