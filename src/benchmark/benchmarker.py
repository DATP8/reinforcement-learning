from src.routing.agentic_rl_routing_pass import AgenticRlRoutingPass
from sb3_contrib import MaskablePPO
from src.ppo_util import make_env
from qiskit.transpiler.passes import (
    SabreLayout,
    ApplyLayout,
)
from qiskit import generate_preset_pass_manager
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap, PassManager
from qiskit_ibm_transpiler.ai.routing import AIRouting
from qiskit import QuantumCircuit
from qiskit import qpy
from scipy import stats
from tqdm import tqdm
import numpy as np

import io
import json
import subprocess
import time
import random
from pathlib import Path

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
    ("2Q Depth", 10),
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
        qc_prep = qc.decompose(reps=5) if self.decompose_before_routing else qc

        # Build a clean anonymous circuit with no classical registers
        num_physical = max(qc_prep.num_qubits, self.coupling_map.size())
        qc_clean = QuantumCircuit(num_physical)
        for inst in qc_prep.data:
            if len(inst.clbits) == 0:
                new_qubits = [qc_clean.qubits[qc_prep.find_bit(q).index] for q in inst.qubits]
                qc_clean.append(inst.operation, new_qubits)

        return qc_clean

    def _collect_metrics(self, routed_circuit, transpile_time):
        ops = routed_circuit.count_ops()

        swaps = ops.get("swap", 0)
        cx = ops.get("cx", 0)

        metrics = {
            METRIC_KEYS[0][0]: transpile_time,
            METRIC_KEYS[1][0]: swaps,
            METRIC_KEYS[2][0]: cx,
            METRIC_KEYS[3][0]: routed_circuit.depth(),
            METRIC_KEYS[4][0]: routed_circuit.size(),
            METRIC_KEYS[5][0]: routed_circuit.depth(
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

    def _get_mqt_circuit_via_bridge(self, algo_name: str, qubits: int):
        """Bridge call to the sidecar environment using QPY."""
        result = subprocess.run(
            [BRIDGE_PYTHON, WORKER_SCRIPT, "circuit", algo_name, str(qubits)], 
            capture_output=True, check=True
        )
        if result.stdout.strip() == b"err":
            raise RuntimeError("err")
        # Load QuantumCircuit from QPY bytes
        import io
        buf = io.BytesIO(result.stdout)
        circuits = list(qpy.load(buf))
        if not circuits:
            raise RuntimeError("No circuit returned from bridge.")
        return circuits[0]

    def _get_available_names_via_bridge(self):
        try:
            result = subprocess.run(
                [BRIDGE_PYTHON, WORKER_SCRIPT, "names"], 
                capture_output=True, text=True, check=True
            )
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            print("\n--- BRIDGE CRASHED ---")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr) # THIS is where the real error is hidden
            raise e

    def run_mqt_benchmarks(self, configs: list[tuple[str, PassManager]]):
        algorithm_name_list = self._get_available_names_via_bridge()
        for algorithm_name in algorithm_name_list:
            try:
                qc = self._get_mqt_circuit_via_bridge(algorithm_name, self.qubits)    
                qc = self._prepare_for_routing(qc) 
                qc.remove_final_measurements()
                runs = self.bench_circuit(qc, configs, algorithm_name)
                self._print_header(f"Algorithm: {algorithm_name}")
                for config_name, metrics in runs.items():
                    self._print_row(config_name, metrics)
    
            except AssertionError:
                raise
            except RuntimeError:
                pass
            except Exception as e:
                raise e
    
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
    from src.routing.swap_inserter.swap_inserter import SwapInserter
    from qiskit.transpiler.passes import ApplyLayout, SabreLayout, TrivialLayout
    from src.states.circuit_graph_state_handler import CircuitGraphStateHandler

    from qiskit.transpiler.passes import SabreSwap

    n_qubits = 6
    topology = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    state_handler = CircuitGraphStateHandler(n_qubits, topology)

    coupling_map = CouplingMap(topology)
    coupling_map.make_symmetric()

    swap_inserter = SwapInserter(coupling_map, n_qubits)

    ai_routing = AIRouting(
        coupling_map=coupling_map,
        optimization_level=1,
        layout_mode="KEEP",
        local_mode=True,
    )

    ai_routing_op = AIRouting(
        coupling_map=coupling_map,
        optimization_level=1,
        local_mode=True,
    )


    horizon = 32
    ppo_env = make_env(
        n_qubits, coupling_map, num_active_swaps=6, horizon=horizon, diff_slope=2
    )

    trivial_layout = TrivialLayout(coupling_map)
    sabre_layout = SabreLayout(coupling_map=coupling_map, skip_routing=True)

    
    #### Standard qiskit pass manager inserted router
    configs = [
        (
            "TrivialLayout_SabreSwap",
            PassManager([SabreSwap(coupling_map=coupling_map)])
        ),
        (
            "TrivialLayout_AI_ibm",
            PassManager([ ai_routing])
        ),
        (
            "SabreLayout_SabreSwap",
            PassManager([sabre_layout, ApplyLayout(), SabreSwap(coupling_map=coupling_map)])
        ),
        # (
        #     "layout_AI_ibm",
        #     PassManager([ai_routing_op])
        # ),
        (
            "Op0 qiskit",
            generate_preset_pass_manager(
                optimization_level=0, coupling_map=coupling_map
            ),
        ),
    ]

    #### Pass manager with only routing stage
    # configs = [(title, PassManager([router])) for title, router in routers]

    bench_iterations = 100
    bench_circut_gate_count = 100
    bench = Benchmarker(n_qubits, bench_circut_gate_count, coupling_map)
    bench.run_mqt_benchmarks(configs)  # pyrefly: ignore
    print("\n")
    bench.run_rand_benchmarks(configs, bench_iterations)  # pyrefly: ignore
