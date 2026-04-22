from src.model import BiCircuitGNNDense
from src.states.dense_circuit_graph_state_handler import DenseCircuitGraphStateHandler
from src.model import BiCircuitGNN
import heapq
from matplotlib.pyplot import step
from qiskit import QuantumCircuit
from src.routing.bwas_router import BWASNode
from qiskit.transpiler import Layout
from src.routing.router import Router
import heapq
from src.states.state_handler import StateHandler
import torch



class BeamSearch[S, To](Router):
    def __init__(
        self, model, state_handler: StateHandler[S], horizon=32, step_size=10, beam_with=100, max_steps=1000
    ):
        self.max_steps = max_steps
        self.beam_with = beam_with
        self.horizon = horizon
        self.step_size = step_size
        self.model = model
        self.state_handler = state_handler

    def search(self, root_state: S) -> list[int]:
        
        with torch.no_grad():
            h = self.model(root_state)
        
        root_node = BWASNode(root_state, 0)
        beam = []
        heapq.heappush(beam, (h, root_node))

        for _ in range(self.max_steps):
            candidates = []
            for _, node in beam:
                if self.state_handler.is_terminal(node.state):
                    return self.reconstruct_path(node)
                for action in self.state_handler.get_possible_actions(node.state):
                    next_state = self.state_handler.get_next_state(node.state, action)
                    action_cost = self.state_handler.get_action_cost(node.state, action)
                    with torch.no_grad():
                        h = self.model(next_state).item()
                    next_node = node.get_child(next_state, action, action_cost)
                    f = next_node.g + h
                    candidates.append((f, next_node))
                    
            beam = heapq.nsmallest(self.beam_with, candidates, key=lambda x: x[0])
        
        raise NotImplemented
        best = heapq.heappop(beam)[-1]
        return self.reconstruct_path(best)
        

    def reconstruct_path(self, node):
        path = []
        while node.action is not None:
            path.append(node.action)
            node = node.parent_node
        return path[::-1]

    
    def solve(self, circuit: QuantumCircuit):
        actions = []
        windows = []
        n = len(circuit.data)
        print(n)
        for i in range(0, n, self.step_size):
            qc = QuantumCircuit(circuit.num_qubits).from_instructions(
                circuit.data[i:i+self.horizon]
            )
            windows.append((i, qc))
        print(len(windows))
        layout = Layout.from_qubit_list(circuit.qubits, range(len(circuit.qubits)))
        for step, window in windows:
            window = self._apply_layout_to_circuit(window, layout)
            state = self.state_handler.state_from(window)
            local_actions = self.search(state)[:self.step_size]
            print("FUUUCKKK")
            print(local_actions)
            layout = self._update_layout(layout, local_actions)
            actions.extend(local_actions)
            
        return actions
            
    def _apply_layout_to_circuit(
        self, circuit: QuantumCircuit, layout: Layout
    ) -> QuantumCircuit:
        new_circ = QuantumCircuit(circuit.num_qubits)

        for instruction in circuit.data:
            instr = instruction.operation
            qargs = instruction.qubits
            cargs = instruction.clbits

            physical_qargs = [layout[q] for q in qargs]
            new_circ.append(instr, physical_qargs, cargs)

        return new_circ
            
    def _update_layout(self, layout: Layout, actions: list[int]) -> Layout:
        topology = self.state_handler.get_topology()
        for action in actions:
            q1, q2 = topology[action]
            layout.swap(q1, q2)

        return layout


if __name__ == "__main__":
    import random
    from qiskit.transpiler.coupling import CouplingMap as CM
    from src.states.circuit_graph_state_handler import CircuitGraphStateHandler
    from qiskit.qpy import load
    import qiskit
    import random
    import time

    
    num_qubits = 6
    coupling_map = [(0,1),(1,2),(2,3),(3,4),(4,5)]
    state_handler = DenseCircuitGraphStateHandler(num_qubits, coupling_map)
    model = BiCircuitGNNDense(num_qubits)
    path_dense = "models/dense_graph/difficulty18_iteration82480.pt"
    beam_search = BeamSearch(model, state_handler, horizon=18, step_size=5, beam_with=100, max_steps=1000)
    model.load_state_dict(torch.load(path_dense, map_location="cpu"))
    
    circuit = QuantumCircuit(num_qubits)
    with open("circuits/dud.qpy", "rb") as f:
        circuit = load(f)[0]
    
    actions = beam_search.solve(circuit)
    print(circuit)
    print(actions)
    