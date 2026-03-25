from tqdm import tqdm
from qiskit import generate_preset_pass_manager
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap, PassManager
from qiskit import QuantumCircuit
from collections import defaultdict
from scipy import stats
import numpy as np

import time
import random


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

    def run_rand_benchmarks(
        self,
        configs: list[tuple[str, PassManager]],
        iterations: int,
        confidence: float = 0.95,
    ):
        raw: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

        qc_list = []

        for i in tqdm(range(iterations), desc="List of random circuits"):
            qc_list.append(
                self.generate_random_2qubit_circuit(self.qubits, self.max_depth)
            )

        for config in configs:
            title, pm = config
            runs = self.bench_pass(qc_list, config)
            for run in runs:
                for metric in METRIC_KEYS:
                    raw[title][metric].append(run[metric])

        for title, metric_lists in raw.items():
            summary = {}
            for metric, values in metric_lists.items():
                arr = np.array(values, dtype=float)
                mean = arr.mean()
                n = len(arr)
                se = stats.sem(arr)  # standard error
                ci = se * stats.t.ppf((1 + confidence) / 2, df=n - 1)  # pyrefly: ignore
                summary[metric] = (mean, ci)
            self._pretty_print(title, summary, confidence)

    def bench_pass(self, qc_list, config):
        runs = []

        title, pm = config

        for qc in tqdm(qc_list, desc=title):
            org_op = Operator.from_circuit(qc)

            start = time.perf_counter()
            routed = pm.run(qc)
            end = time.perf_counter()

            transpile_time = end - start

            routed_op = Operator.from_circuit(routed)

            assert routed_op == org_op, (
                f"\n\nFor the following configuration {title}\n"
                f"quantum circuits was not equal: \noriginal:\n{qc} routed: \n{routed}\n"
            )

            runs.append(self._collect_metrics(routed, transpile_time))
        return runs

    def _pretty_print(self, title, summary, confidence):
        pct = int(confidence * 100)
        print(f"\n{'=' * 60}")
        print(f"  Title: {title}")
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
    from src.routing.swap_inserter.swap_inserter import SwapInserter
    from src.states.tensor_state_handler import TensorStateHandler
    from src.model import ValueModel
    from src.routing.path_rl_routing_pass import PathRlRoutingPass
    from src.routing.bwas_router import BWASRouter

    import torch

    from qiskit.transpiler.passes import SabreSwap

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
    # model2.load_state_dict(torch.load(path2, map_location="cpu"))

    coupling_map = CouplingMap(topology)
    coupling_map.make_symmetric()

    swap_inserter = SwapInserter(coupling_map, n_qubits)
    router1 = BWASRouter(model1, game1)
    # router2 = BWASRouter(model2, game2)

    routers = [
        ("sabre", SabreSwap(coupling_map=coupling_map)),
        ("diff17", PathRlRoutingPass(router1, swap_inserter)),
    ]

    def _make_staged_pass_manager(transPass):
        pm = generate_preset_pass_manager(
            optimization_level=1, coupling_map=coupling_map
        )
        pm.routing = PassManager([transPass])
        return pm

    #### Standard qiskit pass manager inserted router
    configs = [(title, _make_staged_pass_manager(router)) for title, router in routers]
    configs.append(
        (
            "Op1 qiskit",
            generate_preset_pass_manager(
                optimization_level=1, coupling_map=coupling_map
            ),
        )
    )

    #### Pass manager with only routing stage
    # configs = [(title, PassManager([router])) for title, router in routers]

    bench_iterations = 10
    bench_circut_gate_count = 10
    bench = Benchmarker(n_qubits, bench_circut_gate_count, coupling_map)
    bench.run_rand_benchmarks(configs, bench_iterations)  # pyrefly: ignore
