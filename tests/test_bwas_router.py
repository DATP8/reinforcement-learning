import unittest
import torch

from qiskit.quantum_info import Operator
from src.model import BiCircuitGNN#, ValueModel, ValueModelFlat
from src.routing.bwas_chunck_router import ChunkRouter
from src.routing.bwas_router import BWASRouter
from src.states.circuit_graph_state_handler import CircuitGraphStateHandler
from src.routing.swap_inserter.swap_inserter import SwapInserter
from src.states.circuit_graph import CircuitGraph
from src.circuit_generator import CircuitGenerator
# from src.states.qtensor_state_handler import QtensorStateHandler
# from src.states.qtensor import Qtensor
# from src.states.tensor_state import TensorState
# from src.states.tensor_state_handler import TensorStateHandler

class TestBWAS(unittest.TestCase):
    n_qubits = 6
    horizon = 100
    topology = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    swap_inserter = SwapInserter(topology, n_qubits)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_bwas_graph(self):
        state_handler = CircuitGraphStateHandler(self.n_qubits, self.topology)
        model = BiCircuitGNN(self.n_qubits)
        model.load_state_dict(
            torch.load(
                "test_models/graph/difficulty62_updates7_iteration25150.pt",
                map_location=self.device,
            )
        )
        bwas = BWASRouter(model.to(self.device), state_handler)
        circuits = CircuitGenerator.generate_n_random_circuits(
            50, self.n_qubits, 10, {"cx"}
        )
        for circuit in circuits:
            pre_operator = Operator.from_circuit(circuit)
            state = state_handler.state_from(circuit)
            actions = bwas.search(state)
            routed_circuit, _, _ = self.swap_inserter.build_circuit_from_solution(
                actions, circuit
            )
            post_operator = Operator.from_circuit(circuit)
            circuit_graph = CircuitGraph.from_circuit(routed_circuit)
            if not state_handler.is_terminal(circuit_graph):
                print(routed_circuit)
            self.assertTrue(state_handler.is_terminal(circuit_graph))
            self.assertTrue(pre_operator == post_operator)

    """
    def test_bwas_qtensor(self):
        state_handler = QtensorStateHandler(self.n_qubits, self.horizon, self.topology)
        model = ValueModelFlat(self.n_qubits, self.horizon, len(self.topology))
        model.load_state_dict(
            torch.load(
                "models/graph/difficulty43_updates9_iteration13490.pt", map_location=self.device
            )
        )
        bwas = BWASRouter(model.to(self.device), state_handler)
        circuits = CircuitGenerator.generate_n_random_circuits(100, self.n_qubits, 20, {"cx"})
        for circuit in circuits:
            pre_operator = Operator.from_circuit(circuit)
            state = state_handler.state_from(circuit)
            actions = bwas.search(state)
            routed_circuit, _, _ = (
                self.swap_inserter.build_circuit_from_solution(actions, circuit)
            )
            post_operator = Operator.from_circuit(circuit)
            circuit_tensor = Qtensor.from_circuit(routed_circuit)
            if not state_handler.is_terminal(circuit_tensor):
                print(routed_circuit)
            self.assertTrue(state_handler.is_terminal(circuit_tensor))
            self.assertTrue(pre_operator == post_operator)
    
    def test_bwas_tensor(self):
        state_handler = TensorStateHandler(self.n_qubits, self.horizon, self.topology,)
        model = ValueModel(self.n_qubits, self.horizon, len(self.topology))
        model.load_state_dict(
            torch.load(
                "models/graph/difficulty43_updates9_iteration13490.pt", map_location=self.device
            )
        )
        bwas = BWASRouter(model.to(self.device), state_handler)
        circuits = CircuitGenerator.generate_n_random_circuits(100, self.n_qubits, 20, {"cx"})
        for circuit in circuits:
            pre_operator = Operator.from_circuit(circuit)
            state = state_handler.state_from(circuit)
            actions = bwas.search(state)
            routed_circuit, _, _ = (
                self.swap_inserter.build_circuit_from_solution(actions, circuit)
            )
            post_operator = Operator.from_circuit(circuit)
            circuit_tensor = TensorState.from_circuit(routed_circuit)
            if not state_handler.is_terminal(circuit_tensor):
                print(routed_circuit)
            self.assertTrue(state_handler.is_terminal(circuit_tensor))
            self.assertTrue(pre_operator == post_operator)
            """
            
class TestBWASChunking(unittest.TestCase):
    n_qubits = 6
    horizon = 100
    topology = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    swap_inserter = SwapInserter(topology, n_qubits)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_bwas_chunking_graph(self):
        state_handler = CircuitGraphStateHandler(self.n_qubits, self.topology)
        model = BiCircuitGNN(self.n_qubits)
        model.load_state_dict(
            torch.load(
                "test_models/graph/difficulty62_updates7_iteration25150.pt",
                map_location=self.device,
            )
        )
        router = ChunkRouter(chunk_size=1, state_handler=state_handler, model=model)
        circuits = CircuitGenerator.generate_n_random_circuits(
            10, self.n_qubits, 200, {"cx"}
        )
        for circuit in circuits:
            pre_operator = Operator.from_circuit(circuit)
            actions = router.solve(circuit)
            routed_circuit, _, _ = self.swap_inserter.build_circuit_from_solution(
                actions, circuit
            )
            post_operator = Operator.from_circuit(circuit)
            circuit_graph = CircuitGraph.from_circuit(routed_circuit)
            if not state_handler.is_terminal(circuit_graph):
                print(routed_circuit)
            self.assertTrue(state_handler.is_terminal(circuit_graph))
            self.assertTrue(pre_operator == post_operator)

    """
    def test_bwas_chunking_qtensor(self):
        state_handler = QtensorStateHandler(self.n_qubits, self.horizon, self.topology)
        model = ValueModelFlat(self.n_qubits, self.horizon, len(self.topology))
        model.load_state_dict(
            torch.load(
                "models/graph/difficulty43_updates9_iteration13490.pt", map_location=self.device
            )
        )
        router = ChunkRouter(chunk_size=1, state_handler=state_handler, model=model)
        circuits = CircuitGenerator.generate_n_random_circuits(100, self.n_qubits, 20, {"cx"})
        for circuit in circuits:
            pre_operator = Operator.from_circuit(circuit)
            actions = router.solve(circuit)
            routed_circuit, _, _ = (
                self.swap_inserter.build_circuit_from_solution(actions, circuit)
            )
            post_operator = Operator.from_circuit(circuit)
            circuit_tensor = Qtensor.from_circuit(routed_circuit)
            if not state_handler.is_terminal(circuit_tensor):
                print(routed_circuit)
            self.assertTrue(state_handler.is_terminal(circuit_tensor))
            self.assertTrue(pre_operator == post_operator)
    
    def test_bwas_chunking_tensor(self):
        state_handler = TensorStateHandler(self.n_qubits, self.horizon, self.topology,)
        model = ValueModel(self.n_qubits, self.horizon, len(self.topology))
        model.load_state_dict(
            torch.load(
                "models/graph/difficulty43_updates9_iteration13490.pt", map_location=self.device
            )
        )
        router = ChunkRouter(chunk_size=1, state_handler=state_handler, model=model)
        circuits = CircuitGenerator.generate_n_random_circuits(100, self.n_qubits, 20, {"cx"})
        for circuit in circuits:
            pre_operator = Operator.from_circuit(circuit)
            actions = router.solve(circuit)
            routed_circuit, _, _ = (
                self.swap_inserter.build_circuit_from_solution(actions, circuit)
            )
            post_operator = Operator.from_circuit(circuit)
            circuit_tensor = TensorState.from_circuit(routed_circuit)
            if not state_handler.is_terminal(circuit_tensor):
                print(routed_circuit)
            self.assertTrue(state_handler.is_terminal(circuit_tensor))
            self.assertTrue(pre_operator == post_operator)
            """
