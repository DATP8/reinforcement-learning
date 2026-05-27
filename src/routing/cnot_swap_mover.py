from qiskit.dagcircuit.dagnode import DAGOpNode
from qiskit.transpiler import TransformationPass
from qiskit.circuit.library import CXGate
from qiskit.circuit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit


class CNOTSwapMover(TransformationPass):
    def run(self, dag):
        for node in list(dag.topological_nodes())[::-1]:
            if not isinstance(node, DAGOpNode):
                continue
            
            if node.name != "swap":
                continue

            self._find_pattern(dag, node)
            

        return dag
    
    def _find_pattern(self, dag, swap_node):
        qargs = swap_node.qargs
        
        predecessors = list(dag.predecessors(swap_node))
        
        cnot_match = None
        total_single_qubit_nodes = []
        
        if len(predecessors) == 1 and isinstance(predecessors[0], DAGOpNode) and predecessors[0].name == "cx" and set(predecessors[0].qargs) == set(qargs):
            cnot_match = predecessors[0]
            return self._replace_pattern(dag, cnot_match, [], swap_node)

            
        for anc in predecessors:
            if not isinstance(anc, DAGOpNode):
                continue
            
            if len(anc.qargs) == 2:
                if anc.name == "cx" and set(anc.qargs) == set(qargs):
                    cnot_match = anc
                    continue

                return

            cnot, single_qubit_nodes = self._find_2q_gate(dag, anc, qargs)
            
            if cnot is None:
                return
            
            if cnot is not None and single_qubit_nodes is not None:
                if cnot_match is not None:
                    assert cnot_match == cnot, "Multiple different 2-qubit gates found in pattern, get fucked"

                cnot_match = cnot
                total_single_qubit_nodes.extend(single_qubit_nodes)

        if cnot_match is not None:
            self._replace_pattern(dag, cnot_match, total_single_qubit_nodes, swap_node)
            
        return
    
    def _replace_pattern(self, dag, cx_node, single_qubit_nodes, swap_node):
        q0, q1 = cx_node.qargs

        # Create a fresh 2-qubit register for the sub-DAG
        qr = QuantumRegister(2, "q")
        sub_dag = DAGCircuit()
        sub_dag.add_qreg(qr)
        
        # CNOT Swap cancel
        sub_dag.apply_operation_back(cx_node.op, [qr[1], qr[0]], cx_node.cargs)
        sub_dag.apply_operation_back(cx_node.op, [qr[0], qr[1]], cx_node.cargs)

        dag.remove_op_node(swap_node)
        
        for node in single_qubit_nodes:
            dag.remove_op_node(node)
            sub_dag.apply_operation_back(node.op, [qr[1] if node.qargs[0] == q0 else qr[0]], node.cargs)
            
        
        dag.substitute_node_with_dag(cx_node, sub_dag, wires={qr[0]: q0, qr[1]: q1})


    def _find_2q_gate(self, dag, node, qargs):
        predecessors = list(dag.predecessors(node))

        predecessors = [p for p in predecessors if isinstance(p, DAGOpNode)]   
        
        assert len(predecessors) <= 1, "Pattern is not linear, get fucked"
    
        
        if len(predecessors) == 0:
            return None, None
        
        predecessor = predecessors[0]
        
        if not isinstance(predecessor, DAGOpNode):
            return None, None
        
        if len(predecessor.qargs) == 2:
            if predecessor.name == "cx" and set(predecessor.qargs) == set(qargs):
                return predecessor, [node]
            
            return None, None
        
        cnot, single_qubit_nodes = self._find_2q_gate(dag, predecessor, qargs)
        return cnot, single_qubit_nodes + [node] if single_qubit_nodes is not None else None



if __name__ == "__main__":
    from qiskit import QuantumCircuit
    from qiskit.transpiler import PassManager
    from qiskit.qpy import load
    from qiskit.quantum_info import Operator

    
    pm = PassManager([CNOTSwapMover()])


    with open("circuits/debug_circuit (1).qpy", "rb") as f:
        qc = load(f)[0]
        
    # qc = QuantumCircuit(3)
    # qc.cx(1, 0)
    # qc.h(1)
    # qc.h(1)
    # qc.swap(0, 1)
    # qc.swap(1, 2)

    print("Original Circuit:")
    print(qc)
    routed = pm.run(qc)
    
    print("Routed Circuit:")
    print(routed)
    
    org_op = Operator.from_circuit(qc)
    routed_op = Operator.from_circuit(routed)
    assert routed_op.equiv(org_op), (
        f"quantum circuits was not equal: \noriginal:\n{qc} routed: \n{routed}\n"
    )
    
    
