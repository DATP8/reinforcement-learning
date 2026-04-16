
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler import CouplingMap
from qiskit.converters import circuit_to_dag, dag_to_circuit
from mqt.qmap.sc import Architecture, Method
from mqt.qmap.plugins.qiskit.sc import compile_

class SatRoutingPass(TransformationPass):
    def __init__(self, coupling_map: CouplingMap):
        super().__init__()
        self.coupling_map = coupling_map

    def run(self, dag):
        qc = dag_to_circuit(dag)
        
        arch = Architecture(
            self.coupling_map.size(),
            {(a, b) for a, b in self.coupling_map.get_edges()},
        )

        new_qc, results = compile_(
            qc, 
            arch, 
            method=Method.exact, 
            add_measurements_to_mapped_circuit=False
        )

        if new_qc.layout is not None:
            initial_layout = new_qc.layout.initial_layout
            self.property_set["layout"] = initial_layout
            
            final_layout = new_qc.layout.final_layout
            if final_layout:
                self.property_set["final_layout"] = final_layout
            else:
                self.property_set["final_layout"] = initial_layout

        return circuit_to_dag(new_qc)