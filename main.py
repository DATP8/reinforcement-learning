from typing import Tuple
from mqt.bench import get_benchmark, BenchmarkLevel
from mqt.bench.targets import get_target_for_gateset, get_device
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import TrivialLayout, FullAncillaAllocation, EnlargeWithAncilla, ApplyLayout, SabreSwap
from qiskit.circuit.random import random_circuit
from qiskit import QuantumCircuit

from src.routing.bwas_routing import BWASRouting


def count_swaps(qc):
    return sum(1 for inst in qc.data if inst.operation.name == "swap")


def run_benchmark(algo="ghz", num_qubits=6, custom_coupling_map=None):
    horizon = 100

    coupling_map = CouplingMap([[0,1],[1,2],[2,3],[3,4],[4,5]])

    # Pre-routing circuit (input to your router)
    native_target = get_target_for_gateset("ibm_falcon", num_qubits)
    pre_routed = get_benchmark(algo, BenchmarkLevel.NATIVEGATES, num_qubits, native_target)

    # MQT's routed circuit (your baseline to compare against)
    mqt_device = get_device("ibm_falcon_27")
    mqt_routed = get_benchmark(algo, BenchmarkLevel.MAPPED, num_qubits, mqt_device)

    print("=== Pre-Routing Circuit ===")
    print(pre_routed.draw())

    # Your router via PassManager
    pm = PassManager([
        TrivialLayout(coupling_map),
        FullAncillaAllocation(coupling_map),
        EnlargeWithAncilla(),
        ApplyLayout(),
        #NaiveSabre(coupling_map=coupling_map)
        BWASRouting(coupling_map=coupling_map, horizon=horizon),
    ])
    my_routed = pm.run(pre_routed)

    

    print("=== MQT Routed Circuit ===")
    print(mqt_routed.draw())
    print("\n=== My Routed Circuit ===")
    print(my_routed.draw())

    print(f"\n{'Metric':<20} {'MQT':>10} {'Mine':>10}")
    print(f"{'Depth':<20} {mqt_routed.depth():>10} {my_routed.depth():>10}")
    print(f"{'SWAPs':<20} {count_swaps(mqt_routed):>10} {count_swaps(my_routed):>10}")
    print(f"{'Gate count':<20} {mqt_routed.size():>10} {my_routed.size():>10}")


def run_random_benchmark(iterations=10, num_qubits=6, max_gates=10, seed=42, model_path="", batch_size=1):
    horizon = 100
    coupling_map = CouplingMap([[0,1],[1,2],[2,3],[3,4],[4,5]])
    coupling_map.make_symmetric()

    # Standard Qiskit SABRE routing
    pm_sabre = PassManager([
        TrivialLayout(coupling_map),
        FullAncillaAllocation(coupling_map),
        EnlargeWithAncilla(),
        ApplyLayout(),
        SabreSwap(coupling_map),
    ])

    # BWAS routing
    pm = PassManager([
        TrivialLayout(coupling_map),
        FullAncillaAllocation(coupling_map),
        EnlargeWithAncilla(),
        ApplyLayout(),
        BWASRouting(coupling_map=coupling_map, 
                    horizon=horizon, 
                    model_path=model_path, 
                    batch_size=batch_size,
                    device="cpu"),
    ])

    sabre_dict = {}
    my_dict    = {}
    for _ in range(iterations):
        qc = random_circuit(num_qubits=num_qubits, depth=max_gates, seed=seed)
        my_routed = pm.run(qc)
        sabre_routed = pm_sabre.run(qc)
        sabre_dict["depth"] = sabre_dict["depth"] + sabre_routed.depth()
        my_dict["depth"]    = my_dict["depth"] + my_routed.depth()
        sabre_dict["swaps"] = sabre_dict["swaps"] + count_swaps(sabre_routed)
        my_dict["swaps"]    = my_dict["swaps"] + count_swaps(my_routed)
        sabre_dict["size"] = sabre_dict["size"] + sabre_routed.size()
        my_dict["size"]    = my_dict["size"] + my_routed.size()

    print(f"\n{'Metric':<20} {'SABRE':>10} {'Mine':>10}")
    print(f"{'Depth':<20} {sabre_dict["depth"]/iterations:>10} {my_dict["depth"]/iterations:>10}")
    print(f"{'SWAPs':<20} {sabre_dict["swaps"]/iterations:>10} {my_dict["swaps"]/iterations:>10}")
    print(f"{'Gate count':<20} {sabre_dict["size"]/iterations:>10} {my_dict["size"]/iterations:>10}")


def run_random_benchmark(iterations=10, num_qubits=6, max_gates=10, seed=42, model_path="", batch_size=1):
    horizon = 100
    coupling_map = CouplingMap([[0,1],[1,2],[2,3],[3,4],[4,5]])
    coupling_map.make_symmetric()

    # Standard Qiskit SABRE routing
    pm_sabre = PassManager([
        TrivialLayout(coupling_map),
        FullAncillaAllocation(coupling_map),
        EnlargeWithAncilla(),
        ApplyLayout(),
        SabreSwap(coupling_map),
    ])

    # BWAS routing
    pm = PassManager([
        TrivialLayout(coupling_map),
        FullAncillaAllocation(coupling_map),
        EnlargeWithAncilla(),
        ApplyLayout(),
        BWASRouting(coupling_map=coupling_map, 
                    horizon=horizon, 
                    model_path=model_path, 
                    device="cpu"),
    ])

    sabre_dict = {}
    my_dict    = {}
    for _ in range(iterations):
        qc = random_circuit(num_qubits=num_qubits, depth=max_gates, seed=seed)
        my_routed = pm.run(qc)
        sabre_routed = pm_sabre.run(qc)
        sabre_dict["depth"] = sabre_dict["depth"] + sabre_routed.depth()
        my_dict["depth"]    = my_dict["depth"] + my_routed.depth()
        sabre_dict["swaps"] = sabre_dict["swaps"] + count_swaps(sabre_routed)
        my_dict["swaps"]    = my_dict["swaps"] + count_swaps(my_routed)
        sabre_dict["size"] = sabre_dict["size"] + sabre_routed.size()
        my_dict["size"]    = my_dict["size"] + my_routed.size()

    print(f"\n{'Metric':<20} {'SABRE':>10} {'Mine':>10}")
    print(f"{'Depth':<20} {sabre_dict["depth"]/iterations:>10} {my_dict["depth"]/iterations:>10}")
    print(f"{'SWAPs':<20} {sabre_dict["swaps"]/iterations:>10} {my_dict["swaps"]/iterations:>10}")
    print(f"{'Gate count':<20} {sabre_dict["size"]/iterations:>10} {my_dict["size"]/iterations:>10}")

def test_bwas_routing(model_path):
    horizon = 100
    coupling_map = CouplingMap([[0,1],[1,2],[2,3],[3,4],[4,5]])
    coupling_map.make_symmetric()

    qc = QuantumCircuit(6)
    qc.h(0)
    qc.cx(1, 3)
    qc.rz(0.5, 2)
    qc.cz(2, 3)
    qc.h(1)
    qc.cx(0, 2)

    pm = PassManager([
        TrivialLayout(coupling_map),
        FullAncillaAllocation(coupling_map),
        EnlargeWithAncilla(),
        ApplyLayout(),
        BWASRouting(coupling_map=coupling_map, 
                    horizon=horizon, 
                    model_path=model_path, 
                    device="cpu"),
    ])

    my_routed = pm.run(qc)
    print(my_routed)


if __name__ == "__main__":
    model_path="/home/vind/code/P8/project/reinforcement-learning/models/difficulty9_iteration6500.pt"
    test_bwas_routing(model_path=model_path)
    #run_random_benchmark(num_qubits=6, max_gates=4, seed=42, model_path=model_path, iterations=100)
