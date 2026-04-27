from qiskit.transpiler import TransformationPass
from qiskit.circuit.library import CXGate
from qiskit.circuit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit


class CNOTSwapCancelation(TransformationPass):
    def run(self, dag):
        # Iterate over all CX nodes
        for node in list(dag.op_nodes()):
            if node.name != "cx":
                continue

            qargs = node.qargs

            # Look at successors
            successors = list(dag.successors(node))
            predecessors = list(dag.predecessors(node))

            if len(successors) == 1:
                succ = successors[0]

                # Check it's a SWAP on same qubits
                if succ.name == "swap":
                    if set(succ.qargs) == set(qargs):
                        # Replace CX + SWAP with two CXs
                        self._replace_pattern(dag, node, succ)
                        continue

            if len(predecessors) == 1:
                pred = predecessors[0]

                # Check it's a SWAP on same qubits
                if pred.name == "swap":
                    if set(pred.qargs) == set(qargs):
                        # Replace SWAP + CX with two CXs
                        self._replace_pattern(dag, node, pred, reverse=True)

        return dag

    def _replace_pattern(self, dag, cx_node, swap_node, reverse=False):
        q0, q1 = cx_node.qargs

        # Create a fresh 2-qubit register for the sub-DAG
        qr = QuantumRegister(2, "q")
        sub_dag = DAGCircuit()
        sub_dag.add_qreg(qr)

        # Build the 2-CX replacement
        if not reverse:
            sub_dag.apply_operation_back(CXGate(), [qr[1], qr[0]])
            sub_dag.apply_operation_back(CXGate(), [qr[0], qr[1]])
        else:
            sub_dag.apply_operation_back(CXGate(), [qr[0], qr[1]])
            sub_dag.apply_operation_back(CXGate(), [qr[1], qr[0]])

        # Remove the original CX
        dag.remove_op_node(cx_node)

        # Replace SWAP in-place with the sub-DAG
        dag.substitute_node_with_dag(swap_node, sub_dag, wires={qr[0]: q0, qr[1]: q1})


if __name__ == "__main__":
    from qiskit import QuantumCircuit
    from qiskit.transpiler import PassManager

    pm = PassManager([CNOTSwapCancelation()])

    qc = QuantumCircuit(4)

    qc.swap(0, 1)
    qc.cx(0, 1)
    # qc.swap(0, 1)
    qc.cx(0, 1)
    # qc.swap(0, 1)
    # qc.swap(0, 1)

    print(qc)
    print(pm.run(qc))
