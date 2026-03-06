from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.layout import Layout
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.transpiler.target import Target
from qiskit.transpiler.passes.layout import disjoint_utils


class NaiveSabre(TransformationPass):

    def __init__(self, coupling_map):
        super().__init__()
        if isinstance(coupling_map, Target):
            self.target = coupling_map
            self.coupling_map = self.target.build_coupling_map()
        else:
            self.target = None
            self.coupling_map = coupling_map

    def run(self, dag):
        new_dag = dag.copy_empty_like()

        if self.coupling_map is None:
            raise TranspilerError("NaiveSabre cannot run with coupling_map=None")
        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("NaiveSabre runs on physical circuits only")
        if len(dag.qubits) > len(self.coupling_map.physical_qubits):
            raise TranspilerError("The layout does not match the amount of qubits in the DAG")
        disjoint_utils.require_layout_isolated_to_component(
            dag, self.coupling_map if self.target is None else self.target
        )

        canonical_register = dag.qregs["q"]
        current_layout = Layout.generate_trivial_layout(canonical_register)

        front_list = dag.front_layer()
        executed_set = set()  

        while front_list:
            execute_list = [] 
            for g in front_list:
                if g.num_qubits != 2:
                    execute_list.append(g)
                    continue

                physical_q0 = current_layout[g.qargs[0]]
                physical_q1 = current_layout[g.qargs[1]]

                if self.coupling_map.distance(physical_q0, physical_q1) == 1:
                    execute_list.append(g)

            if execute_list:
                for eg in execute_list:
                    new_dag.apply_operation_back(eg.op, eg.qargs, eg.cargs)
                    executed_set.add(eg)
                    front_list.remove(eg)

                    for suc in dag.op_successors(eg):
                        if self._dependencies_resolved(dag, suc, executed_set):
                            front_list.append(suc)

            else:
                best_cand = None
                best_dist = float("inf")

                for g in front_list:
                    physical_q0 = current_layout[g.qargs[0]]
                    physical_q1 = current_layout[g.qargs[1]]


                    q0_neighbors = list(self.coupling_map.neighbors(physical_q0))
                    q1_neighbors = list(self.coupling_map.neighbors(physical_q1))
                    candidates = (
                        [(physical_q0, n) for n in q0_neighbors ] +
                        [(physical_q1, n) for n in q1_neighbors ]
                    )

                    for q0, q1 in candidates:
                        dist = self.coupling_map.distance(q0, q1)
                        if dist < best_dist:
                            best_dist = dist
                            best_cand = (q0, q1)
                if not best_cand:
                    continue 

                swap_layer = DAGCircuit()
                swap_layer.add_qreg(canonical_register)
                swap_qubit_pair = [canonical_register[i] for i in best_cand]
                swap_layer.apply_operation_back(SwapGate(), swap_qubit_pair, cargs=(), check=False)
                order = current_layout.reorder_bits(new_dag.qubits)
                new_dag.compose(swap_layer, qubits=order)
                current_layout.swap(best_cand[0], best_cand[1])

        if self.property_set["final_layout"] is None:
            self.property_set["final_layout"] = current_layout
        else:
            self.property_set["final_layout"] = self.property_set["final_layout"].compose(
                current_layout, dag.qubits
            )

        return new_dag

    def _dependencies_resolved(self, dag, gate, executed_set):
        for pred in dag.op_predecessors(gate):
            if pred not in executed_set:
                return False
        return True