from qiskit.transpiler import CouplingMap
from sb3_contrib import MaskablePPO
from src.gym_test import route_circuit
from qiskit.transpiler import Layout
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.basepasses import TransformationPass


class AgenticRlRoutingPass(TransformationPass):
    def __init__(
        self,
        model: MaskablePPO,
        cmap: CouplingMap
    ):
        super().__init__()
        self.model = model
        self.cmap = cmap

    def run(self, dag):
        qc = dag_to_circuit(dag)
        layout: Layout = self.property_set["final_layout"]
        qc, layout = route_circuit(self.model, qc, layout)
        new_dag = circuit_to_dag(qc)
        self.property_set["final_layout"] = layout

        #layout = Layout(self.env.unwrapped.get_final_mapping())
#
        #self.property_set["final_layout"] = (
        #    layout
        #    if (prev := self.property_set["final_layout"]) is None
        #    else prev.compose(layout, new_dag.qubits)
        #)

        return new_dag
