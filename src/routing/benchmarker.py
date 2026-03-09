from typing import override
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes import TrivialLayout, VF2Layout, SabreLayout, SabreSwap
from qiskit.circuit.random import random_circuit
from itertools import product
import time
import matplotlib
matplotlib.use("TkAgg")
from src.routing.bwas_routing import BWASRouting

class Benchmarker: 
    def __init__(self, model_path, qubits, max_gates, coupling_map):
        self.model_path = model_path
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

    def _build_pass_manager(self, init, fb, final, coupling_map):

        passes = []

        # ---------- initial layout ----------
        if init == "trivial":
            passes.append(TrivialLayout(coupling_map))

        elif init == "qiskit":
            passes.append(VF2Layout(coupling_map))

        # elif init == "rl":
        #     passes.append(RLInitialLayout(coupling_map, model))

        # ---------- forward/backward ----------
        if fb == "sabre":
            passes.append(SabreLayout(coupling_map))

        # elif fb == "rl":
        #     passes.append(RLForwardBackward(coupling_map, model))

        elif fb == "none":
            pass

        # ---------- final routing ----------
        if final == "sabre":
            passes.append(SabreSwap(coupling_map))

        elif final == "rl":
            passes.append(RLSwapRouter(coupling_map, self.model_path))
        return PassManager(passes)


    def run_rand_benchmarks(self, configs, iterations):
        results = {}
        for i in range(iterations):
            qc = random_circuit(self.qubits, self.max_depth)

            runs = self.run_bench(qc, configs)
            for run in runs:
                key = run["initial"]+'_'+run["forward_backward"]+'_'+run["final_router"]
                if key in results:
                    results[key] = self._add_metrics(results[key], run)
                else:
                    results[key] = run

        for key, val in results.items():
            self._pretty_print(key, self._scalar_divide_metrics(val, iterations))



    def run_bench(self, qc, configs, printing=False):
        rows = []
        for init, fb, final in configs:
            if printing:
                print("Running with:", init, fb, final)
        
            pm =self._build_pass_manager(
                init,
                fb,
                final,
                self.coupling_map
            )
        
            start = time.perf_counter()
            routed = pm.run(qc)
            end = time.perf_counter()

            if printing:
                print(routed)
        
            transpile_time = end - start
        
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



class RLInitialLayout(AnalysisPass):

    def __init__(self, coupling_map, model):
        super().__init__()
        self.coupling_map = coupling_map
        self.model = model

    @override
    def run(self, dag):
        candidates = vf2_candidate_layouts(dag, self.coupling_map)

        best_layout = None
        best_score = float("inf")

        for layout in candidates:

            tensor = circuit_to_tensor(dag, layout)
            score = self.model.predict(tensor)

            if score < best_score:
                best_layout = layout
                best_score = score

        self.property_set["layout"] = best_layout

class RLForwardBackward(TransformationPass):
    @override
    def run(self, dag):
        pass


class RLSwapRouter(TransformationPass):
    def __init__(self, coupling_map, model_path):
        super().__init__()
        self.coupling_map = coupling_map 
        self.coupling_map.make_symmetric()
        self.model_path = model_path
        self.horizon = 100
        self.device = "cpu"

        self.bwas = BWASRouting(
            coupling_map=self.coupling_map, 
            horizon=self.horizon, 
            model_path=self.model_path, 
            device=self.device
        )

    @override
    def run(self, dag):
        return self.bwas.run(dag)



if __name__ == "__main__":
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

    
    
