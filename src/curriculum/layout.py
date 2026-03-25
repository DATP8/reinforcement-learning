from qiskit import QuantumCircuit
from qiskit.transpiler.passes import ApplyLayout
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import SabreLayout
from qiskit.converters import circuit_to_dag
from src.circuit_generator import CircuitGenerator
from qiskit.transpiler import CouplingMap
from qiskit.converters import dag_to_circuit

cmap = CouplingMap.from_line(4)
qc = QuantumCircuit(4)

qc = CircuitGenerator.generate_random_circuit(
    num_qubits=4,
    num_gates=6,
    gateset={ "cx" },
    seed=42,
)

print(qc)
# dag = circuit_to_dag(qc)
# layout_pass= SabreLayout(cmap, seed=45)

pm = PassManager([
    SabreLayout(cmap, seed=45, skip_routing = True),
    ApplyLayout()
])
qc = pm.run(qc)

# dag = layout_pass.run(dag)
# qc = dag_to_circuit(dag)


# layout = pm.property_set["layout"]
# print(layout)

print(qc)
print("layout:", qc.layout)
