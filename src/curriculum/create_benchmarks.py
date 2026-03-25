from curriculum.curriculum_helper import RoutedCircuitMetrics
from curriculum.curriculum_helper import CircuitMetrics
from curriculum.curriculum_helper import CurriculumHelper
import random
import numpy as np
from tqdm import tqdm

from qiskit import transpile, QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.providers.fake_provider import GenericBackendV2


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


def run_experiment(
    num_circuits: int,
    num_qubits: int,
    num_gates: int,
    gateset: set[str],
    cmap: CouplingMap,
    helper: CurriculumHelper,
):
    backend = GenericBackendV2(num_qubits=num_qubits, coupling_map=cmap)

    result = []

    for i in tqdm(range(num_circuits)):
        seed = np.random.randint(0, np.iinfo(int).max)
        qc = generate_random_circuit(
            num_qubits=num_qubits,
            num_gates=num_gates,
            gateset=gateset,
            seed=seed,
        )
        metrics = CircuitMetrics(
            seed=seed,
            difficulty=num_gates,
            num_qubits=num_qubits,
            depth=qc.depth(),
            gate_set=gateset,
            routed=[],
        )
        for opt_level in range(4):
            routed = transpile(
                qc,
                backend=backend,
                optimization_level=opt_level,
                seed_transpiler=seed,
            )
            routed_circuit = RoutedCircuitMetrics(
                optimization_level=opt_level,
                depth=routed.depth(),
                num_gates=routed.size(),
            )
            metrics.routed.append(routed_circuit)
        result.append(metrics)

    helper.save_circuits(result)


if __name__ == "__main__":
    max_difficulty = 200
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
