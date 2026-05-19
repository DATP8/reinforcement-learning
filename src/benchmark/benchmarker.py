from src.policy_types import ActorCriticPolicyType
from src.benchmark.passmanager_creaters import PPOBuilder
from cmath import sqrt
# from mqt.bench import BenchmarkLevel, get_benchmark
# from mqt.bench.benchmarks import get_available_benchmark_names
import random
import time

import numpy as np
from qiskit import QuantumCircuit, generate_preset_pass_manager
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap, PassManager
from qiskit_ibm_transpiler.ai.routing import AIRouting
from sb3_contrib import MaskablePPO
from scipy import stats
from tqdm import tqdm
import numpy as np

import pickle
import json
import subprocess
import time
from pathlib import Path

from src.circuit_generator import CircuitGenerator

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent

# Define the absolute paths
BRIDGE_DIR = ROOT_DIR / "tools" / "mqt_bridge"
BRIDGE_PYTHON = str(BRIDGE_DIR / ".venv" / "bin" / "python")
WORKER_SCRIPT = str(BRIDGE_DIR / "mqt_worker.py")

METRIC_KEYS = [
    ("Transpile", 10),
    ("Swap", 10),
    ("CX", 10),
    ("Depth", 10),
    ("Size", 10),
    ("Decomposed Gates", 10),
]

EVAL_SEED = 2026  # np.random.randint(0, 2**31 - 1)
random.seed(EVAL_SEED)
np.random.seed(EVAL_SEED)

EVAL_TRIALS = 12

MAX_EQUIV_CHECK_QUBITS = 12


class Benchmarker:
    def __init__(
        self,
        qubits,
        coupling_map,
        num_gates,
        decompose_before_routing=True,
        decompose_reps=2,
    ):
        self.qubits = qubits
        self.coupling_map = coupling_map
        self.num_gates = num_gates
        self.decompose_before_routing = decompose_before_routing
        self.decompose_reps = decompose_reps

    def _print_header(
        self, title: str, confidence: float | None = None, title_size: int = 30
    ) -> None:
        header: str = f"{'Config':<{title_size}}"
        underline: str = "".ljust(title_size, "-")
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

    def _print_row(self, name, metrics, ci=None, title_size: int = 30):
        row = f"{name:<{title_size}}"
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
        qc_prep = qc.decompose(reps=5) if self.decompose_before_routing else qc

        # Build a clean anonymous circuit with no classical registers
        num_physical = max(qc_prep.num_qubits, self.coupling_map.size())
        qc_clean = QuantumCircuit(num_physical)
        for inst in qc_prep.data:
            if len(inst.clbits) == 0:
                new_qubits = [
                    qc_clean.qubits[qc_prep.find_bit(q).index] for q in inst.qubits
                ]
                qc_clean.append(inst.operation, new_qubits)

        return qc_clean

    def _collect_metrics(self, routed_circuit: QuantumCircuit, transpile_time: float):
        ops = routed_circuit.count_ops()

        swaps = ops.get("swap", 0)
        cx = ops.get("cx", 0)
        decomposed_depth = swaps * 3 + cx

        metrics = {
            METRIC_KEYS[0][0]: transpile_time,
            METRIC_KEYS[1][0]: swaps,
            METRIC_KEYS[2][0]: cx,
            METRIC_KEYS[3][0]: routed_circuit.depth(),
            METRIC_KEYS[4][0]: routed_circuit.size(),
            METRIC_KEYS[5][0]: decomposed_depth,
        }
        return metrics

    def _get_mqt_circuit_via_bridge(self, algo_name: str, qubits: int):
        result = subprocess.run(
            [BRIDGE_PYTHON, WORKER_SCRIPT, "circuit", algo_name, str(qubits)],
            capture_output=True,
            check=True,
        )
        if result.stdout.strip() == b"err":
            raise RuntimeError("err")
        return pickle.loads(result.stdout)

    def _get_available_names_via_bridge(self):
        try:
            result = subprocess.run(
                [BRIDGE_PYTHON, WORKER_SCRIPT, "names"],
                capture_output=True,
                text=True,
                check=True,
            )
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            print("\n--- BRIDGE CRASHED ---")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)  # THIS is where the real error is hidden
            raise e

    def run_mqt_benchmarks(self, configs: list[tuple[str, PassManager]]):
        name_lengths = [len(elem[0]) for elem in configs]
        name_size = max(name_lengths)

        algorithm_name_list = self._get_available_names_via_bridge()
        results = {}
        for algorithm_name in algorithm_name_list:
            try:
                qc = self._get_mqt_circuit_via_bridge(algorithm_name, self.qubits)
                qc = self._prepare_for_routing(qc)
                qc.remove_final_measurements()
                runs = self.bench_circuit(qc, configs, algorithm_name)
                self._print_header(f"Algorithm: {algorithm_name}", title_size=name_size)
                results[algorithm_name] = {}
                for config_name, metrics in runs.items():
                    results[algorithm_name][config_name] = metrics
                    self._print_row(config_name, metrics, title_size=name_size)

            except AssertionError:
                raise
            except RuntimeError:
                pass
            except Exception as e:
                raise e

        return results

    def run_rand_benchmarks(
        self,
        configs: list[tuple[str, PassManager]],
        iterations: int,
        title: str,
        confidence: float = 0.95,
        is_printing: bool = True,
        seed=1,
    ):
        name_lengths = [len(elem[0]) for elem in configs]
        name_size = max(name_lengths)
        qc_list = CircuitGenerator.generate_n_random_cx_circuits(
            n=iterations,
            num_qubits=self.qubits,
            num_gates=self.num_gates,
            seed=seed,
        )

        if is_printing:
            self._print_header(title=title, confidence=confidence, title_size=name_size)
        results = {}
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

            results[title] = (mean_dic, ci_dic)

            if is_printing:
                self._print_row(
                    title, metrics=mean_dic, ci=ci_dic, title_size=name_size
                )
        return results

    def bench_pass(self, qc, pm, title):

        has_classical_ops = any(len(inst.clbits) > 0 for inst in qc.data)
        if has_classical_ops:
            qc = qc.remove_final_measurements(inplace=False)

        start = time.perf_counter()
        routed = pm.run(qc)
        end = time.perf_counter()

        transpile_time = end - start

        try:
            org_op = Operator.from_circuit(qc)
            routed_op = Operator.from_circuit(routed)
            assert routed_op.equiv(org_op), (
                f"\n\nFor the following configuration {title}\n"
                f"quantum circuits was not equal: \noriginal:\n{qc} routed: \n{routed}\n"
            )
        except Exception as e:
            if isinstance(e, AssertionError):
                raise
            pass

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
    from src.benchmark.passmanager_creaters import (
        IbmRlBuilder,
        SabreBuilder,
        QiskitTranspiler,
    )

    start_qubits = 6
    end_qubits = 7

    sqrt_qubits = int(sqrt(end_qubits).real)

    coupling_map_list = []
    # coupling_map_list.extend([("grid",            CouplingMap().from_grid(x, y))              for x in range(start_qubits, sqrt_qubits) for y in range(start_qubits, sqrt_qubits)])
    # coupling_map_list.extend([("hex_lattice",     CouplingMap().from_hexagonal_lattice(x, y)) for x in range(start_qubits, sqrt_qubits) for y in range(start_qubits, sqrt_qubits)])
    # coupling_map_list.extend([("hex_heavy",       CouplingMap().from_heavy_hex(x))            for x in range(start_qubits, end_qubits) if x % 2 == 1])
    # coupling_map_list.extend([("hex_square",      CouplingMap().from_heavy_square(x))         for x in range(start_qubits, end_qubits) if x % 2 == 1])
    coupling_map_list.extend(
        [("ring", CouplingMap().from_ring(x)) for x in range(start_qubits, end_qubits)]
    )
    coupling_map_list.extend(
        [("line", CouplingMap().from_line(x)) for x in range(start_qubits, end_qubits)]
    )

    results = {}

    for title, coupling_map in coupling_map_list:
        coupling_map.make_symmetric()

        trivial_ai_ibm = IbmRlBuilder(op_level=3).build(coupling_map)

        sabre_ai_ibm = IbmRlBuilder(op_level=3, use_sabre_layout=True).build(
            coupling_map
        )

        trivial_sabre = SabreBuilder(use_sabre_layout=False).build(coupling_map)

        sabre_sabre = SabreBuilder(use_sabre_layout=False).build(coupling_map)

        qiskit_transpiler = QiskitTranspiler(op_level=0).build(coupling_map)


        our_ppo = PPOBuilder(
            num_active_swaps=5,
            horizon=8,
            initial_difficulty=256,
            max_difficulty=256,
            diff_slope=0.9,
            layout_exponent=1.0,
            policy_type=ActorCriticPolicyType.BASIC,
            seed=42,
            model_path="models/best_model_basic.zip",
            use_sabre_layout=False,
        ).build(coupling_map)

        configs = [
            ("trivial layout ai routing (ibm)", trivial_ai_ibm),
            ("trivial layout sabre", trivial_sabre),
            ("trivial layout ppo", our_ppo),
            ("sabre layout ai routing (ibm)", sabre_ai_ibm),
            ("sabre layout sabre", sabre_sabre),
            ("optimization level 0 qiskit standard transpiler", qiskit_transpiler),
        ]

        bench_iterations = 10
        bench_circut_gate_count = 100
        n_qubits = coupling_map.size()
        bench = Benchmarker(qubits=n_qubits, coupling_map=coupling_map, num_gates=bench_circut_gate_count)
        temp_results = bench.run_mqt_benchmarks(configs)  # pyrefly: ignore

        results[title] = temp_results

        results_dir = ROOT_DIR / "results"
        results_dir.mkdir(exist_ok=True)
        results_file = results_dir / "benchmark_mqt_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)


        # temp_results = bench.run_rand_benchmarks(
        #     configs,
        #     bench_iterations,
        #     title=f"{title} | Qubits: {n_qubits} | Random circuits: {bench_iterations}",
        #     is_printing=True,
        # )  # pyrefly: ignore
        # if title not in results:
        #     results[title] = {}

        # for config in temp_results:
        #     if config not in results[title]:
        #         results[title][config] = {}

        #     mean, ci = temp_results[config]
        #     results[title][config][n_qubits] = {
        #         metric: {"mean": mean[metric], "ci": ci[metric]}
        #         for metric, _ in METRIC_KEYS
        #     }

        # results_dir = ROOT_DIR / "results"
        # results_dir.mkdir(exist_ok=True)
        # results_file = results_dir / "benchmark_results.json"
        # with open(results_file, "w") as f:
        #     json.dump(results, f, indent=2)
