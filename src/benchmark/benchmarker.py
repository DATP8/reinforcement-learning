from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.circuit.random import random_circuit
from qiskit import QuantumCircuit

from itertools import product

import time 
import random

from ..routing.bwas_routing import BWASRouting


class Benchmarker: 
    def __init__(self, qubits, max_gates, coupling_map):
        self.qubits = qubits
        self.max_depth = max_gates
        self.coupling_map = coupling_map

    def _collect_metrics(self, routed_circuit, transpile_time):
        ops = routed_circuit.count_ops()

        swaps = ops.get("swap", 0)
        cx = ops.get("cx", 0)

        metrics = {
            "transpile_time": transpile_time,
            "swap_count": swaps,
            "cx_count": cx,
            "two_qubit_total": swaps * 3 + cx,
            "depth": routed_circuit.depth(),
            "size": routed_circuit.size(),
            "two_qubit_depth": routed_circuit.depth(
                filter_function=lambda inst: inst.operation.name in ["cx","swap"]
            )
        }
        return metrics

    def _build_pass_manager(self, init, fb, final):

        passes = [p for p in [init, fb, final] if p is not None]

        return PassManager(passes)
  
    def generate_random_2qubit_circuit(self, num_qubits: int, num_gates: int) -> QuantumCircuit:
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

    def run_rand_benchmarks(self, configs, iterations, is_set_difficulty=False):
        results = {}
        for i in range(iterations):

            if is_set_difficulty:
                qc = self.generate_random_2qubit_circuit(self.qubits, self.max_depth)
            else:
                qc = random_circuit(self.qubits, self.max_depth)
        
            runs = self.run_bench(qc, configs)
            for run in runs:
                    
                key = type(run["initial"]).__name__ if run["forward_backward"] is not None else " " +'_'+ \
                      type(run["forward_backward"]).__name__  if run["forward_backward"] is not None else " " + "_" + \
                      run["final_router"].get_name() if type(run["final_router"]) is BWASRouting else type(run["final_router"]).__name__ 

                if key in results:
                    results[key] = self._add_metrics(results[key], run)
                else:
                    results[key] = run

        for key, val in results.items():
            self._pretty_print(key, self._scalar_divide_metrics(val, iterations))



    def run_bench(self, qc, configs, printing=False):
        rows = []
    

        org_op = Operator.from_circuit(qc)
        for init, fb, final in configs:
            if printing:
                print("Running with:", init, fb, final)
        
            pm =self._build_pass_manager(
                init,
                fb,
                final
            )
        
            start = time.perf_counter()
            routed = pm.run(qc)
            end = time.perf_counter()

            transpile_time = end - start

            routed_op = Operator.from_circuit(routed)

            if routed_op != org_op:
                print("\n\n", init, fb, final," was not eqivivalent\n\n")

            if printing:
                print(qc)
                print(routed)
        
            metrics = self._collect_metrics(
                routed,
                transpile_time
            )
        
            row = {
                "initial": init,
                "forward_backward": fb,
                "final_router": final,
                **metrics
            }
        
            rows.append(row)
        return rows

    def _add_metrics(self, metric1, metric2):
        return {
            "transpile_time":   metric1["transpile_time"]  + metric2["transpile_time"],
            "swap_count":       metric1["swap_count"]      + metric2["swap_count"],
            "cx_count":         metric1["cx_count"]        + metric2["cx_count"],
            "two_qubit_total":  metric1["two_qubit_total"] + metric2["two_qubit_total"],
            "depth":            metric1["depth"]           + metric2["depth"],
            "size":             metric1["size"]            + metric2["size"],
            "two_qubit_depth":  metric1["two_qubit_depth"] + metric2["two_qubit_depth"]
        }

    def _scalar_divide_metrics(self, metric, scalar):
        return {
            "transpile_time":   metric["transpile_time"]  / scalar,
            "swap_count":       metric["swap_count"]      / scalar,
            "cx_count":         metric["cx_count"]        / scalar,
            "two_qubit_total":  metric["two_qubit_total"] / scalar,
            "depth":            metric["depth"]           / scalar,
            "size":             metric["size"]            / scalar,
            "two_qubit_depth":  metric["two_qubit_depth"] / scalar
        }

    def _pretty_print(self, name, metrics):
        print(f"\n{'=' * 50}")
        print(f"  {name}")
        print(f"{'=' * 50}")
        print(f"  {'Transpile time':<20} {metrics['transpile_time']:.4f} s")
        print(f"  {'Swap count':<20} {metrics['swap_count']:.2f}")
        print(f"  {'CX count':<20} {metrics['cx_count']:.2f}")
        print(f"  {'2Q total':<20} {metrics['two_qubit_total']:.2f}")
        print(f"  {'Depth':<20} {metrics['depth']:.2f}")
        print(f"  {'Size':<20} {metrics['size']:.2f}")
        print(f"  {'2Q depth':<20} {metrics['two_qubit_depth']:.2f}")
        print(f"{'=' * 50}")


if __name__ == "__main__":

    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    
    initial_layouts = ["qiskit"]
    forward_backward = ["none", "sabre"]
    final_routers = ["sabre", "rl"]

    configs = list(product(
        initial_layouts,
        forward_backward,
        final_routers
    ))

    qubits = 6
    max_gates = 9
    coupling_map = CouplingMap([[0,1],[1,2],[2,3],[3,4],[4,5]])
    path = "/home/vind/code/P8/project/reinforcement-learning/models/difficulty9_iteration6500.pt"
    bench = Benchmarker(path, qubits, max_gates, coupling_map)
    # Run each combination
    rows = bench.run_rand_benchmarks(configs, 50)

    
    
