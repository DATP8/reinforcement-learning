from .circuit_graph import CircuitGraph
from collections import defaultdict
from qiskit.circuit.quantumcircuit import QuantumCircuit
import torch
import random
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Operator
 

class CNOTCircuit(QuantumCircuit):
    def __init__(self, n_qubits: int):
        super().__init__(n_qubits)
        self.layers = defaultdict(list)
        self.qubit_layers = [-1 for _ in range(n_qubits)]
        self.org_qc = None

    @staticmethod
    def from_quantum_circuit(qc: QuantumCircuit):
        dag = circuit_to_dag(qc)
        cnot_c = CNOTCircuit(qc.num_qubits)
        cnot_c.org_qc = qc

        for g in dag.topological_op_nodes():
            if g.op.num_qubits == 2:
                q1, q2 = (dag.find_bit(q).index for q in g.qargs)
                cnot_c.add_cnot(q1, q2)

        return cnot_c  

    def reconstruct_with_swaps(self):
        cx_dag = circuit_to_dag(self)
        org_dag = circuit_to_dag(self.org_qc)
        new_qc = QuantumCircuit(self.num_qubits)
        
        mapping = list(range(self.num_qubits))
        inverse = list(range(self.num_qubits))

        org_node_iter = org_dag.topological_op_nodes()
        swap = (0, 0)
        for cx_g in cx_dag.topological_op_nodes():
            if cx_g.op.name == 'swap': 
                q1, q2 = (cx_dag.find_bit(q).index for q in cx_g.qargs)
                swap = (q1, q2)
                
            elif cx_g.op.name == 'cx':
                for org_g in org_node_iter:
                    new_qargs = []
                    is_last = org_g.op.num_qubits == 2 

                    if is_last and swap is not None:
                        q1, q2 = swap
                        new_qc.swap(q1, q2)

                        v1, v2 = mapping[q1], mapping[q2]
                        mapping[v1] = q2  ## Feel like this should be possible without a reverse map
                        mapping[v2] = q1
                        inverse[v1] = q2 
                        inverse[v2] = q1

                    for q in org_g.qargs:
                        qubit_idx = org_dag.find_bit(q).index
                        new_idx = mapping[qubit_idx]
                        new_qargs.append(new_qc.qubits[new_idx])

                    new_qc.append(org_g.op, new_qargs)

                    if not is_last:
                        continue

                    swap = None
                    break

        self = new_qc
        return self

    def add_cnot(self, q1, q2):
        if q1 == q2:
            raise ValueError("Control and target qubits must be different.")
        
        self.cx(q1, q2)
        layer = max(self.qubit_layers[q1], self.qubit_layers[q2]) + 1
        self.qubit_layers[q1] = layer
        self.qubit_layers[q2] = layer
        self.layers[layer].append((q1, q2))

    def to_tensor(self, horizon=None):
        depth = self.depth() if horizon is None else horizon
        tensor = torch.zeros((self.num_qubits, self.num_qubits, depth), dtype=torch.float32)
        
        layer = 0
        for layer in sorted(self.layers.keys()):
            for q1, q2 in self.layers[layer]:
                if layer < depth:
                    tensor[q1, q2, layer] = 1.0
            layer += 1
            if layer > depth:
                assert horizon is not None, f"Reached depth {layer} which exceeds circuit depth {self.depth()} without hitting specified horizon {horizon}."
                break
        
        if horizon is None:
            tensor = tensor[:, :, :layer]
        
        
        return tensor
    
    @staticmethod
    def from_tensor(tensor: torch.Tensor):        
        assert tensor.dim() == 3, "Input tensor must be 3-dimensional (n_qubits, n_qubits, depth)."
        
        n_qubits, n_qubits2, depth = tensor.size()
        assert n_qubits == n_qubits2, f"The first two dimensions of the tensor must be equal (square). Got {n_qubits} and {n_qubits2}."
        
        qc = CNOTCircuit(n_qubits)
        
        for i in range(depth):
            lefts, rights = torch.where(tensor[:, :, i] == 1.0)
            pairs = list(zip(lefts.tolist(), rights.tolist()))
            for q1, q2 in pairs:
                qc.add_cnot(q1, q2)

        return qc
    
    def from_circuit_graph(dag: CircuitGraph):
        if dag.x is None:
            return None
        
        circuit = QuantumCircuit(dag.x.shape[1] // 2)
        for x in dag.x[:-1]: # Exclude global node
            q1 = torch.where(x[:dag.x.shape[1] // 2] > 0)[0].item()
            q2 = torch.where(x[dag.x.shape[1] // 2:] > 0)[0].item()
            circuit.cx(q1, q2)
        
        return circuit


def generate_random_circuit(n_qubits: int, n_gates: int):
    qc = CNOTCircuit(n_qubits)
    
    for i in range(n_gates):
        q1, q2 = random.sample(range(n_qubits), 2)
        while q1 == q2:
            q2 = random.choice(range(n_qubits))
        
        qc.add_cnot(q1, q2)
    return qc
    
if __name__ == "__main__":
    circuit = generate_random_circuit(n_qubits=10, n_gates=20)
    
    print(circuit)
    print(circuit.depth())
    
    tensor = circuit.to_tensor()
    #print(tensor)
    print(tensor.size())
    
    new_circuit = CNOTCircuit.from_tensor(tensor)
    print(new_circuit)
    print(new_circuit.depth())
    
    assert circuit == new_circuit, "The original and reconstructed circuits do not match!"
    assert torch.equal(circuit.to_tensor(), new_circuit.to_tensor()), "The original and reconstructed circuits do not match!"

    # Example: from_quantum_circuit
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(1, 3)
    qc.rz(0.5, 2)
    qc.cz(2, 3)
    qc.h(1)
    qc.cx(0, 2)
    print("\n--- Original QuantumCircuit ---")
    print(qc)

    cnot_c = CNOTCircuit.from_quantum_circuit(qc)
    print("\n--- CNOTCircuit from_quantum_circuit ---")
    print(cnot_c)
    print("Sub-circuits:")
    compose_qc = QuantumCircuit(4)
    for i, sc in enumerate(cnot_c.sub_circuit_list):
        print(f"  Sub-circuit {i}:")
        print(sc)
        compose_qc = compose_qc.compose(sc)

    print(f"The sub circuits {"equal" if Operator(compose_qc) == Operator(qc) else "does not equal" } the original circuit")

    print(cnot_c.reconstruct_with_swaps())