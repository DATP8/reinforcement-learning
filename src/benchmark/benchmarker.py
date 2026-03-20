from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.circuit.random import random_circuit
from qiskit import QuantumCircuit
from itertools import product
from collections import defaultdict
from scipy import stats

import numpy as np
import time
import random

from ..routing.rl_routing_pass import RlRoutingPass

METRIC_KEYS = [
    "transpile_time",
    "swap_count",
    "cx_count",
    "two_qubit_total",
    "depth",
    "size",
    "two_qubit_depth",
]


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
                filter_function=lambda inst: inst.operation.name in ["cx", "swap"]
            ),
        }
        return metrics

    def _build_pass_manager(self, init, fb, final):
        passes = [p for p in [init, fb, final] if p is not None]
        return PassManager(passes)

    def generate_random_2qubit_circuit(
        self, num_qubits: int, num_gates: int
    ) -> QuantumCircuit:
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

    def _config_key(self, run):
        if run["forward_backward"] is not None:
            return (
                type(run["initial"]).__name__
                + "_"
                + type(run["forward_backward"]).__name__
            )
        elif type(run["final_router"]) is RlRoutingPass:
            return type(run["initial"]).__name__ + "_" + run["final_router"].get_name()
        else:
            return (
                type(run["initial"]).__name__ + "_" + type(run["final_router"]).__name__
            )

    def run_rand_benchmarks(
        self, configs, iterations, is_set_difficulty=False, confidence=0.95
    ):
        # Accumulate raw per-iteration metric values per config key
        raw: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

        for i in range(iterations):
            if is_set_difficulty:
                qc = self.generate_random_2qubit_circuit(self.qubits, self.max_depth)
            else:
                qc = random_circuit(self.qubits, self.max_depth)

            runs = self.run_bench(qc, configs)
            for run in runs:
                key = self._config_key(run)
                for metric in METRIC_KEYS:
                    raw[key][metric].append(run[metric])

        # Compute mean and confidence intervals using numpy + scipy
        for key, metric_lists in raw.items():
            summary = {}
            for metric, values in metric_lists.items():
                arr = np.array(values, dtype=float)
                mean = arr.mean()
                n = len(arr)
                se = stats.sem(arr)  # standard error
                ci = se * stats.t.ppf((1 + confidence) / 2, df=n - 1)  # t-based CI
                summary[metric] = (mean, ci)
            self._pretty_print(key, summary, confidence)

    def run_bench(self, qc, configs, printing=False):
        rows = []

        org_op = Operator.from_circuit(qc)
        for init, fb, final in configs:
            if printing:
                print("Running with:", init, fb, final)

            pm = self._build_pass_manager(init, fb, final)

            start = time.perf_counter()
            routed = pm.run(qc)
            end = time.perf_counter()

            transpile_time = end - start

            routed_op = Operator.from_circuit(routed)

            assert routed_op == org_op, (
                f"\n\nFor the following configuration {init, fb, final}\n"
                f"quantum circuits was not equal: \noriginal:\n{qc} routed: \n{routed}\n"
            )

            if printing:
                print(qc)
                print(routed)

            metrics = self._collect_metrics(routed, transpile_time)

            row = {
                "initial": init,
                "forward_backward": fb,
                "final_router": final,
                **metrics,
            }

            rows.append(row)
        return rows

    def _pretty_print(self, name, summary, confidence):
        pct = int(confidence * 100)
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")
        print(f"  {'Metric':<22} {'Mean':>10}  {'±CI ({pct}%)':>12}".format(pct=pct))
        print(f"  {'-' * 46}")
        labels = {
            "transpile_time": "Transpile time (s)",
            "swap_count": "Swap count",
            "cx_count": "CX count",
            "two_qubit_total": "2Q total",
            "depth": "Depth",
            "size": "Size",
            "two_qubit_depth": "2Q depth",
        }
        for key, label in labels.items():
            mean, ci = summary[key]
            print(f"  {label:<22} {mean:>10.4f}  ±{ci:>10.4f}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    from ..routing.swap_inserter.swap_inserter import SwapInserter
    from ..states.tensor_state_handler import TensorStateHandler
    from ..model import ValueModel
    from ..routing.rl_routing_pass import RlRoutingPass
    from ..routing.bwas_router import BWASRouter

    import torch

    from qiskit.transpiler.passes import TrivialLayout, SabreLayout, SabreSwap

    n_qubits = 6
    horizon = 100
    topology = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    game1 = TensorStateHandler(n_qubits, horizon, topology)
    model1 = ValueModel(n_qubits, horizon, len(topology))

    game2 = TensorStateHandler(n_qubits, horizon, topology)
    model2 = ValueModel(n_qubits, horizon, len(topology))

    path1 = "/home/vind/code/P8/project/reinforcement-learning/models/difficulty17_iteration95270.pt"
    path2 = "/home/vind/code/P8/project/reinforcement-learning/models/increment14_iteration77940_difficulty17.pt"
    model1.load_state_dict(torch.load(path1, map_location="cpu"))
    model2.load_state_dict(torch.load(path2, map_location="cpu"))

    bench_iterations = 10
    coupling_map = CouplingMap(topology)
    coupling_map.make_symmetric()

    initial_layouts = [TrivialLayout(coupling_map)]

    forward_backward = [SabreLayout(coupling_map)]

    swap_inserter = SwapInserter(coupling_map, n_qubits)
    router1 = BWASRouter(model1, game1)
    router2 = BWASRouter(model2, game2)

    final_routers = [
        SabreSwap(coupling_map),
        RlRoutingPass(router1, swap_inserter, "diff17"),
        RlRoutingPass(router2, swap_inserter, "incr14"),
    ]

    configs = list(product(initial_layouts, [None], final_routers))

    coupling_map.make_symmetric()

    bench = Benchmarker(n_qubits, 14, coupling_map)
    bench.run_rand_benchmarks(configs, bench_iterations, True)
