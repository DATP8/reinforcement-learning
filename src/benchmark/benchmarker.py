from src.gym_test import make_env
from sb3_contrib import MaskablePPO
from src.routing.agentic_rl_routing_pass import AgenticRlRoutingPass
from qiskit import generate_preset_pass_manager
from qiskit.quantum_info import Operator
from qiskit.transpiler import PassManager
from qiskit import QuantumCircuit
from collections import defaultdict
from scipy import stats
from tqdm import tqdm
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
    from qiskit.transpiler.passes import ApplyLayout
    from qiskit.transpiler.passes import SabreLayout
    from qiskit.transpiler.passes import TrivialLayout
    from src.states.circuit_graph_state_handler import CircuitGraphStateHandler
    from qiskit.transpiler import CouplingMap

    # from src.routing.rl_routing_pass import RlRoutingPass
    from qiskit.transpiler.passes import SabreSwap

    n_qubits = 6
    horizon = 100
    topology = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    state_handler = CircuitGraphStateHandler(n_qubits, topology)

    # path = "models/graph/difficulty62_updates7_iteration25150.pt"
    # model = BiCircuitGNN(n_qubits)
    # model.load_state_dict(torch.load(path, map_location="cpu"))

    coupling_map = CouplingMap(topology)
    coupling_map.make_symmetric()

    swap_inserter = SwapInserter(coupling_map, n_qubits)

    # chuck_size = 16
    # chunk_router = ChunkRouter(
    #    chunk_size=chuck_size, model=model, state_handler=state_handler
    # )

    horizon = 18
    ppo_env = make_env(n_qubits, coupling_map, horizon, None)
    ppo_model = MaskablePPO.load("test_model", ppo_env)
    agentic_router = AgenticRlRoutingPass(ppo_model, coupling_map)

    # chunck_swap_pass = RlRoutingPass(chunk_router, swap_inserter)

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
            f"TrivialLayout_MaskablePPO_{horizon}",
            PassManager([trivial_layout, ApplyLayout(), agentic_router]),
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
            f"SabreLayout_MaskablePPO_{horizon}",
            PassManager(
                [
                    sabre_layout,
                    ApplyLayout(),
                    agentic_router,
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

    bench_iterations = 100
    bench_circut_gate_count = 100
    bench = Benchmarker(n_qubits, bench_circut_gate_count, coupling_map)
    bench.run_rand_benchmarks(configs, bench_iterations)  # pyrefly: ignore
