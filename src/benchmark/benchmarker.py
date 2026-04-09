from qiskit.transpiler.passes import (
    FilterOpNodes,
    SabreLayout,
    ApplyLayout,
    BarrierBeforeFinalMeasurements,
)
from qiskit import generate_preset_pass_manager
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap, PassManager
from qiskit import QuantumCircuit
from collections import defaultdict
from scipy import stats
from tqdm import tqdm
import numpy as np

from mqt.bench import BenchmarkLevel, get_benchmark
from mqt.bench.benchmarks import get_available_benchmark_names

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
    def __init__(
        self,
        qubits,
        max_gates,
        coupling_map,
        decompose_before_routing=True,
        decompose_reps=2,
    ):
        self.qubits = qubits
        self.max_gates = max_gates
        self.coupling_map = coupling_map
        self.decompose_before_routing = decompose_before_routing
        self.decompose_reps = decompose_reps

    def _prepare_for_routing(self, qc: QuantumCircuit) -> QuantumCircuit:
        if not self.decompose_before_routing:
            return qc
        return qc.decompose(reps=5)

    def _collect_metrics(self, routed_circuit, transpile_time):
        ops = routed_circuit.count_ops()

        swaps = ops.get("swap", 0)
        cx = ops.get("cx", 0)

        metrics = {
            "transpile_time": transpile_time,
            "swap_count": swaps,
            "cx_count": cx,
            "two_qubit_total": swaps + cx,
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

    def run_mqt_benchmarks(
        self,
        configs: list[tuple[str, PassManager]],
    ):
        algorithm_name_list = get_available_benchmark_names()
        for algorithm_name in algorithm_name_list:
            success = False
            for q in range(1, self.qubits).__reversed__():
                if success:
                    break
                try:
                    qc = get_benchmark(algorithm_name, BenchmarkLevel.INDEP, self.qubits)
                    qc = self._prepare_for_routing(qc)
                    if qc.num_qubits > self.qubits or qc.size() > self.max_gates:
                        # print(f"{algorithm_name} had qubits ({qc.num_qubits} > {self.qubits}) and {qc.size()} > {self.max_gates}")
                        continue

                    runs = self.bench_circuit(qc, configs, algorithm_name)
                    self._pretty_print_by_config(algorithm_name, runs)
                    success = True

                except AssertionError:
                    raise
                except Exception:
                    pass

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
                self.generate_random_2qubit_circuit(self.qubits, self.max_gates)
            )

        for config in configs:
            title, pm = config
            runs = self.bench_config(qc_list, config)
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

    def bench_pass(self, qc, pm, title):

        has_classical_ops = any(len(inst.clbits) > 0 for inst in qc.data)
        if has_classical_ops:
            qc = qc.remove_final_measurements(inplace=False)

        start = time.perf_counter()
        routed = pm.run(qc)
        end = time.perf_counter()

        transpile_time = end - start

        org_op = Operator.from_circuit(qc)
        routed_op = Operator.from_circuit(routed)
        assert routed_op.equiv(org_op), (
            f"\n\nFor the following configuration {title}\n"
            f"quantum circuits was not equal: \noriginal:\n{qc} routed: \n{routed}\n"
        )

        return self._collect_metrics(routed, transpile_time)

    def bench_circuit(
        self, qc: QuantumCircuit, configs: list[tuple[str, PassManager]], title
    ):
        runs = {}

        for config in tqdm(configs, desc=title):
            config_title, pm = config
            runs[config_title] = self.bench_pass(qc, pm, config_title)

        return runs

    def bench_config(self, qc_list, config):
        runs = []

        title, pm = config

        for qc in tqdm(qc_list, desc=title):
            runs.append(self.bench_pass(qc, pm, title))

        return runs

    def _pretty_print(self, title, summary, confidence=None):
        ci_header = ""
        if confidence is not None:
            pct = int(confidence * 100)
            ci_header = f"{'±CI (' + str(pct) + '%)':>12}"

        print(f"\n{'=' * 60}")
        print(f"  Title: {title}")
        print(f"{'=' * 60}")
        print(f"  {'Metric':<22} {'Mean':>10}" + ci_header)
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
            print(f"  {label:<22} {mean:>10.4f}±{ci:>10.4f}" if confidence else "")
        print(f"{'=' * 60}")

    def _pretty_print_by_config(self, algorithm_name, runs_by_config):
        print(f"\n{'=' * 120}")
        print(f"  Algorithm: {algorithm_name}")
        print(f"{'=' * 120}")

        header = (
            f"  {'Config':<30}"
            f" {'Transpile(s)':>12}"
            f" {'Swap':>8}"
            f" {'CX':>8}"
            f" {'2Q total':>10}"
            f" {'Depth':>8}"
            f" {'Size':>8}"
            f" {'2Q depth':>10}"
        )
        print(header)
        print(f"  {'-' * (len(header) - 2)}")

        for config_name, metrics in runs_by_config.items():
            print(
                f"  {config_name:<30}"
                f" {metrics['transpile_time']:>12.4f}"
                f" {metrics['swap_count']:>8.0f}"
                f" {metrics['cx_count']:>8.0f}"
                f" {metrics['two_qubit_total']:>10.0f}"
                f" {metrics['depth']:>8.0f}"
                f" {metrics['size']:>8.0f}"
                f" {metrics['two_qubit_depth']:>10.0f}"
            )

        print(f"{'=' * 120}")


if __name__ == "__main__":
     from src.routing.swap_inserter.swap_inserter import SwapInserter
    from qiskit.transpiler.passes import ApplyLayout
    from qiskit.transpiler.passes import SabreLayout
    from qiskit.transpiler.passes import TrivialLayout
    from src.routing.bwas_chunck_router import ChunkRouter
    from src.states.circuit_graph_state_handler import CircuitGraphStateHandler
    from qiskit.transpiler import CouplingMap
    from src.model import BiCircuitGNN
    from src.routing.rl_routing_pass import RlRoutingPass
    from qiskit.transpiler.passes import SabreSwap
    import torch

    n_qubits = 6
    horizon = 100
    topology = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    state_handler = CircuitGraphStateHandler(n_qubits, topology)

    path = "models/graph/difficulty62_updates7_iteration25150.pt"
    model = BiCircuitGNN(n_qubits)
    model.load_state_dict(torch.load(path, map_location="cpu"))

    coupling_map = CouplingMap(topology)
    coupling_map.make_symmetric()

    swap_inserter = SwapInserter(coupling_map, n_qubits)

    chuck_size = 16
    chunk_router = ChunkRouter(
        chunk_size=chuck_size, model=model, state_handler=state_handler
    )
    chunck_swap_pass = RlRoutingPass(chunk_router, swap_inserter)

    trivial_layout = TrivialLayout(coupling_map)
    sabre_layout = SabreLayout(coupling_map=coupling_map, skip_routing=True)

    pm = PassManager([trivial_layout, SabreSwap(coupling_map=coupling_map)])
    #### Standard qiskit pass manager inserted router
    configs = [
        (
            "TrivialLayout_SabreSwap",
            PassManager(
                [trivial_layout, ApplyLayout(), SabreSwap(coupling_map=coupling_map)]
            ),
        ),
        (
            f"TrivialLayout_Chunking_{chuck_size}",
            PassManager([trivial_layout, ApplyLayout(), chunck_swap_pass]),
        ),
        (
            "SabreLayout_SabreSwap",
            PassManager(
                [
                    sabre_layout,
                    ApplyLayout(),
                    SabreSwap(coupling_map=coupling_map),
                ]
            ),
        ),
        (
            f"SabreLayout_Chunking_{chuck_size}",
            PassManager(
                [
                    sabre_layout,
                    ApplyLayout(),
                    chunck_swap_pass,
                ]
            ),
        ),
        (
            "Op1 qiskit",
            generate_preset_pass_manager(
                optimization_level=1, coupling_map=coupling_map
            ),
        ),
    ]

    #### Pass manager with only routing stage
    # configs = [(title, PassManager([router])) for title, router in routers]

    bench_iterations = 10
    bench_circut_gate_count = 64
    bench = Benchmarker(n_qubits, bench_circut_gate_count, coupling_map)
    bench.run_mqt_benchmarks(configs) # pyrefly: ignore
    bench.run_rand_benchmarks(configs, bench_iterations)  # pyrefly: ignore
