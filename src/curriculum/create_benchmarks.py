from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ApplyLayout
from qiskit.converters import dag_to_circuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import SabreSwap
from qiskit.transpiler.passes import SabreLayout
from src.circuit_generator import CircuitGenerator
from src.curriculum.curriculum_helper import RoutedCircuitMetrics
from src.curriculum.curriculum_helper import CircuitMetrics
from src.curriculum.curriculum_helper import CurriculumHelper
import random
import numpy as np
from tqdm import tqdm

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from qiskit.transpiler import TransformationPass
from qiskit.passmanager.flow_controllers import ConditionalController, DoWhileController

def generate_random_circuit(num_qubits, num_gates, gateset, seed=None):
    qc = QuantumCircuit(num_qubits)
    gate_set = ["cx"]
    for _ in range(num_gates):
        gate = np.random.choice(gate_set)
        if gate == "cx" and num_qubits > 1:
            q0, q1 = np.random.choice(num_qubits, size=2, replace=False)
            qc.cx(int(q0), int(q1))
        elif gate == "h":
            qc.h(int(np.random.choice(num_qubits)))
        else:
            qc.s(int(np.random.choice(num_qubits)))
    return qc

def extract_passes(tasks):
    passes = []
    for task in tasks:
        if isinstance(task, list):
            passes + extract_passes(task)
        elif isinstance(task, (ConditionalController, DoWhileController)):
            passes + extract_passes(task.tasks)
        elif isinstance(task, TransformationPass):
            print("Found pass", task.__class__.__name__)
            passes.append((task.__class__.__name__, task))
    return passes

def run_experiment(
    num_circuits: int,
    num_qubits: int,
    num_gates: int,
    gateset: set[str],
    cmap: CouplingMap,
    helper: CurriculumHelper,
):
    result = []

    for i in tqdm(range(num_circuits)):
        seed = random.randint(0, 1_000_000_000)
        # rng = np.random.default_rng(seed)
        # num_logical_qubits = rng.integers(2, num_qubits + 1)
        num_logical_qubits = num_qubits
        qc = CircuitGenerator.generate_random_circuit(
            num_qubits=num_logical_qubits,
            num_gates=num_gates,
            gateset=gateset,
            seed=seed,
        )
        metrics = CircuitMetrics(
            seed=seed,
            difficulty=num_gates,
            num_qubits=num_qubits,
            num_logical_qubits=num_logical_qubits,
            depth=qc.depth(),
            gate_set=gateset,
            routed=[],
        )
        # Testing on pass managers across all optimization levels showed they always just use SabreLayout and SabreSwap
        # IBM routing models according to their paper used VF2Layout instead of SabreLayout, so we may try that instead
        pm = PassManager([
            SabreLayout(cmap, seed=seed),
            SabreSwap(cmap, seed=seed),
            ApplyLayout(),
        ])
        routed = pm.run(qc)
        # print(routed)
        routed_circuit = RoutedCircuitMetrics(
            optimization_level=2, # Arbitrary number now
            depth=routed.depth(),
            num_gates=routed.size(),
        )
        metrics.routed.append(routed_circuit)
        result.append(metrics)

    helper.save_circuits(result)


if __name__ == "__main__":
    max_difficulty = 20
    qubits = 10
    cmap = CouplingMap.from_line(qubits)
    cmap_name = "10-L"
    gateset = {"cx"}

    helper = CurriculumHelper(cmap_name, qubits, gateset)

    for d in tqdm(range(max_difficulty)):
        run_experiment(
            num_circuits=1000,
            num_qubits=10,
            num_gates=d + 1,
            gateset=gateset,
            cmap=cmap,
            helper=helper,
        )
