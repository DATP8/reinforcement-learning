from src.routing.cnot_swap_cancel import CNOTSwapCancelation
from qiskit.transpiler.passes import (
    SabreLayout,
    ApplyLayout,
)
from src.states.dense_circuit_graph_state_handler import DenseCircuitGraphStateHandler
from src.model import BiCircuitGNNDense
from qiskit import generate_preset_pass_manager
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import CommutativeCancellation
from qiskit import QuantumCircuit
from scipy import stats
from tqdm import tqdm
import numpy as np

from mqt.bench import BenchmarkLevel, get_benchmark
from mqt.bench.benchmarks import get_available_benchmark_names

import time
import random


METRIC_KEYS = [
    ("Transpile", 10),
    ("Swap", 10),
    ("CX", 10),
    ("Depth", 10),
    ("Size", 10),
    ("Decomposed Depth", 10),
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
        self.commutative_cancellation = CommutativeCancellation()

    def _print_header(self, title: str, confidence: float | None = None) -> None:
        header: str = f"{'Config':<30}"
        underline: str = "".ljust(30, "-")
        if confidence is not None:
            for label, width in METRIC_KEYS:
                header += f"{label:>{width}}{'  ±CI':<10}"
                underline += f"{'-' * width}{'-' * 10}"
        else:
            for label, width in METRIC_KEYS:
                header += f"{label:>{width}}"
                underline += f"{'-' * width}"
        print("\n")
        print(underline)
        print(title)
        print(header)
        print(underline)

    def _print_row(self, name, metrics, ci=None):
        row = f"{name:<30}"
        for label, width in METRIC_KEYS:
            value = metrics[label]
            value_str = f"{value:>{width}.4f}"
            ci_val = ci[label] if ci is not None else None
            if ci_val is not None:
                ci_str = f" ±{ci_val:<8.4f}"
                value_str += ci_str
            else:
                value_str
            row += value_str
        print(row)

    def _prepare_for_routing(self, qc: QuantumCircuit) -> QuantumCircuit:
        if not self.decompose_before_routing:
            return qc
        return qc.decompose(reps=5)

    def _collect_metrics(self, routed_circuit, transpile_time):
        ops = routed_circuit.count_ops()

        decomposed_circuit = self.commutative_cancellation.run(
            routed_circuit.decompose(gates_to_decompose="swap").to_dag()
        ).to_circuit()

        swaps = ops.get("swap", 0)
        cx = ops.get("cx", 0)

        metrics = {
            METRIC_KEYS[0][0]: transpile_time,
            METRIC_KEYS[1][0]: swaps,
            METRIC_KEYS[2][0]: cx,
            METRIC_KEYS[3][0]: routed_circuit.depth(),
            METRIC_KEYS[4][0]: routed_circuit.size(),
            METRIC_KEYS[5][0]: decomposed_circuit.depth(),
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
                    qc = get_benchmark(
                        algorithm_name, BenchmarkLevel.INDEP, self.qubits
                    )
                    qc = self._prepare_for_routing(qc)
                    if qc.num_qubits > self.qubits or qc.size() > self.max_gates:
                        continue

                    runs = self.bench_circuit(qc, configs, algorithm_name)
                    self._print_header(f"Algorithm: {algorithm_name}")
                    for config_name, metrics in runs.items():
                        self._print_row(config_name, metrics)
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
        qc_list = []

        for _ in tqdm(
            range(iterations), desc="List of random circuits", position=0, leave=False
        ):
            qc_list.append(
                self.generate_random_2qubit_circuit(self.qubits, self.max_gates)
            )

        self._print_header(
            title=f"{len(qc_list)} random circuits", confidence=confidence
        )
        for config in configs:
            mean_dic = {}
            ci_dic = {}
            title, _ = config
            runs = self.bench_config(qc_list, config)
            for metric, _ in METRIC_KEYS:
                metric_values = [run[metric] for run in runs]
                arr = np.array(metric_values, dtype=float)
                n = len(arr)
                se = stats.sem(arr) if n > 1 else 0.0
                ci_val = (
                    se * stats.t.ppf((1 + confidence) / 2, df=n - 1) if n > 1 else 0.0
                )
                mean_dic[metric] = arr.mean()
                ci_dic[metric] = ci_val
            self._print_row(title, metrics=mean_dic, ci=ci_dic)

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

        for config in tqdm(configs, desc=title, position=0, leave=False):
            config_title, pm = config
            runs[config_title] = self.bench_pass(qc, pm, config_title)

        return runs

    def bench_config(self, qc_list, config):
        runs = []

        title, pm = config

        for qc in tqdm(qc_list, desc=title, position=0, leave=False):
            runs.append(self.bench_pass(qc, pm, title))

        return runs


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
    state_handler_dense = DenseCircuitGraphStateHandler(n_qubits, topology)

    path = "models/graph/difficulty62_updates7_iteration25150.pt"
    model = BiCircuitGNN(n_qubits)
    model.load_state_dict(torch.load(path, map_location="cpu"))

    path_dense = "models/dense_graph/difficulty18_iteration82480.pt"
    # path_dense = "models/dense_graph/difficulty32_iteration15040.pt"
    model_dense = BiCircuitGNNDense(n_qubits)
    model_dense.load_state_dict(torch.load(path_dense, map_location="cpu"))

    coupling_map = CouplingMap(topology)
    coupling_map.make_symmetric()

    swap_inserter = SwapInserter(coupling_map, n_qubits)

    chuck_size = 18
    chunk_router = ChunkRouter(
        chunk_size=chuck_size, model=model, state_handler=state_handler
    )
    chunck_swap_pass = RlRoutingPass(chunk_router, swap_inserter)

    chuck_router_dense = ChunkRouter(
        chunk_size=chuck_size, model=model_dense, state_handler=state_handler_dense
    )
    chunck_swap_pass_dense = RlRoutingPass(chuck_router_dense, swap_inserter)

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
        # (
        #     f"TrivialLayout_Chunking_{chuck_size}",
        #     PassManager([trivial_layout, ApplyLayout(), chunck_swap_pass]),
        # ),
        (
            "TrivialLayout_SabreSwap_cancel",
            PassManager(
                [
                    trivial_layout,
                    ApplyLayout(),
                    SabreSwap(coupling_map=coupling_map),
                    CNOTSwapCancelation(),
                ]
            ),
        ),
        (
            f"TrivialLayout_Dense_Chunking_cancel{chuck_size}",
            PassManager(
                [
                    trivial_layout,
                    ApplyLayout(),
                    chunck_swap_pass_dense,
                    CNOTSwapCancelation(),
                ]
            ),
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
        # (
        #     f"SabreLayout_Chunking_{chuck_size}",
        #     PassManager(
        #         [
        #             sabre_layout,
        #             ApplyLayout(),
        #             chunck_swap_pass,
        #         ]
        #     ),
        # ),
        (
            f"SabreLayout_Dense_Chunking_{chuck_size}",
            PassManager(
                [
                    sabre_layout,
                    ApplyLayout(),
                    chunck_swap_pass_dense,
                ]
            ),
        ),
        (
            "Op1 qiskit",
            generate_preset_pass_manager(
                optimization_level=1, coupling_map=coupling_map
            ),
        ),
        (
            "Op2 qiskit",
            generate_preset_pass_manager(
                optimization_level=2, coupling_map=coupling_map
            ),
        ),
        (
            "Op3 qiskit",
            generate_preset_pass_manager(
                optimization_level=3, coupling_map=coupling_map
            ),
        ),
    ]

    #### Pass manager with only routing stage
    # configs = [(title, PassManager([router])) for title, router in routers]

    bench_iterations = 10
    bench_circut_gate_count = 100
    bench = Benchmarker(n_qubits, bench_circut_gate_count, coupling_map)
    bench.run_mqt_benchmarks(configs)  # pyrefly: ignore
    print("\n")
    bench.run_rand_benchmarks(configs, bench_iterations)  # pyrefly: ignore
