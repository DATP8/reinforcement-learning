from qiskit.converters import circuit_to_dag
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap as CM
from .swap_inserter import SwapInserter

class BadSwapINserter(SwapInserter):
    def __init__(self, coupling_map: list[tuple[int, int]] | CM, num_qubits: int):
        self.coupling_map = CM(coupling_map) if isinstance(coupling_map, list) else coupling_map
        self.num_qubits = num_qubits
        
    def _add_gate(self, dag, g, new_qc, inverse):
        new_qargs = []
        for q in g.qargs:
            qubit_idx = dag.find_bit(q).index
            new_idx = inverse[qubit_idx]
            new_qargs.append(new_qc.qubits[new_idx])

        new_qc.append(g.op, new_qargs)
        return new_qc

    def _make_topological_connection_list(self, num_qubits, topology):
        topological_connection_list = [[] for _ in range(num_qubits)]

        for pair in topology:
            q1, q2 = pair
            topological_connection_list[q1].append(q2)
            topological_connection_list[q2].append(q1)

        return topological_connection_list

    def _prune(self, qc, dag, inverse, new_qc, topological_connection_list):
        wait_list = [[] for _ in range(qc.num_qubits)]
        
        for q in range(qc.num_qubits):
            for g in dag.nodes_on_wire(qc.qubits[q], only_ops=True):
                if g.op.num_qubits == 1:
                    new_qc = self._add_gate(dag, g, new_qc, inverse)
                    dag.remove_op_node(g)
                    continue
                
                q1 = dag.find_bit(g.qargs[0]).index 
                q2 = dag.find_bit(g.qargs[1]).index 

                q_is_q1 = q1 == q
                if (not q_is_q1):
                    q2 = q1
                    q1 = q

                q1 = inverse[q1]
                q2 = inverse[q2]

                if wait_list[q2] and wait_list[q2][0] == q1 :
                    wait_list[q2].pop(0)
                    new_qc = self._add_gate(dag, g, new_qc, inverse)
                    dag.remove_op_node(g)
                    return self._prune(qc, dag, inverse, new_qc, topological_connection_list)   

                elif any(x == q1 for x in topological_connection_list[q2]):
                    wait_list[q1].append(q2)
               
                break

        return new_qc, dag
        
    def build_circuit_from_solution(self, actions: list, input_circuit: QuantumCircuit):
        dag = circuit_to_dag(input_circuit)
        new_qc = QuantumCircuit(input_circuit.num_qubits)

        topology = list(self.coupling_map.get_edges())
        
        topological_connection_list = self._make_topological_connection_list(input_circuit.num_qubits, topology)

        mapping = list(range(input_circuit.num_qubits))
        inverse = list(range(input_circuit.num_qubits))
        
        while dag.topological_op_nodes():
            new_qc, dag = self._prune(input_circuit, dag, inverse, new_qc, topological_connection_list)
            if (not actions):
                break
            q1, q2 = topology[actions.pop(0)] 
            new_qc.swap(q1, q2)  
            v1, v2 = mapping[q1], mapping[q2]
            mapping[q1] = v2 
            mapping[q2] = v1
            inverse[v1] = q2 
            inverse[v2] = q1
        return new_qc, list(range(input_circuit.num_qubits)), inverse



if __name__ == "__main__":
    import random
    import torch
    import time

    from ...model import ValueModel
    from ...states.tensor_state import TensorState
    from ...states.tensor_state_handler import TensorStateHandler
    from ...batch_weighted_astar_search import BWAS

    def generate_random_2qubit_circuit(num_qubits: int, num_gates: int) -> QuantumCircuit:
        if num_qubits < 2:
            raise ValueError("Number of qubits must be at least 2 for 2-qubit gates.")

        qc = QuantumCircuit(num_qubits)

        for _ in range(num_gates):
            control = random.randint(0, num_qubits - 1)
            target = random.randint(0, num_qubits - 1)
            while target == control:
                target = random.randint(0, num_qubits - 1)

            qc.cx(control, target)

        return qc


    random.seed(42)

    qc = QuantumCircuit(6)
    qc.cx(2, 5)
    qc.cx(4, 2)
    qc.cx(5, 1)
    qc.cx(1, 0)
    qc.cx(2, 3)

    n_qubits = 6
    horizon = 100
    topology = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

    game = TensorStateHandler(n_qubits, horizon, topology)
    model = ValueModel(n_qubits, horizon, len(topology))
    model.load_state_dict(
        torch.load(
            "/home/vind/code/P8/project/reinforcement-learning/models/difficulty17_iteration95270.pt",
            map_location="cpu",
        )
    )
    model.to("cpu")

    coupling_map = CM(topology)
    coupling_map.make_symmetric()

    iterations = 100
    my_time_list = []
    ibm_time_list = []

    bwas = BWAS(model, game)

    my_router = BadSwapINserter(coupling_map, n_qubits)
    ibm_router = SwapInserter(coupling_map, n_qubits)
    for _ in range(100):
        qc = generate_random_2qubit_circuit(6, 14)

        state = TensorState.from_quantum_circuit(qc, horizon=horizon)

        path = bwas.search(state, 1)

        start_time = time.time()
        new_qc, _, _ = my_router.build_circuit_from_solution(path, qc)
        end_time = time.time()
        my_time_list.append(end_time - start_time)

        start_time = time.time()
        new_qc2, _, _ = ibm_router.build_circuit_from_solution(path, qc)
        end_time = time.time()
        ibm_time_list.append(end_time - start_time)

    print(f"my method takes on average {sum(my_time_list) / len(my_time_list)} s")
    print(f"IBM method takes on average {sum(ibm_time_list) / len(ibm_time_list)} s")
