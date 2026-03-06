from qiskit.transpiler import CouplingMap
from qiskit import QuantumCircuit, transpile

# coupling = CouplingMap.from_line(5)
# coupling = CouplingMap.from_line(5)
coupling = CouplingMap([
    [0,2],
    [1,2],
    [3,2],
    [4,2],
    [2,0],
    [2,1],
    [2,3],
    [2,4],
])

qc = QuantumCircuit(5)
qc.cx(0, 1)
qc.cx(1, 2)
qc.cx(2, 3)
qc.cx(3, 4)

print(qc)

routed = transpile(
    qc,
    coupling_map=coupling,
    # initial_layout=initial_layout,
    routing_method="sabre",  # basic | lookahead | sabre
    layout_method="trivial",
    optimization_level=3
)

print(routed)
print(routed.layout)

