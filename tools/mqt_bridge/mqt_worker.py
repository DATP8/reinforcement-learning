import sys
import json
import pickle
from mqt.bench.benchmarks import get_available_benchmark_names  # pyrefly: ignore
from mqt.bench import get_benchmark, BenchmarkLevel  # pyrefly: ignore
from mqt.bench.targets import get_target_for_gateset


def fetch_names():
    print(json.dumps(get_available_benchmark_names()))


def fetch_circuit(algo_name, qubits):
    for q in range(3, int(qubits)).__reversed__():
        try:
            # pyrefly: ignore [missing-argument]
            qc = get_benchmark(algo_name, BenchmarkLevel.NATIVEGATES, q, target=get_target_for_gateset('ibm_falcon', q))
            sys.stdout.buffer.write(pickle.dumps(qc))
            return
        except Exception:
            pass
    sys.stdout.buffer.write(b"err")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "names":
        fetch_names()
    elif mode == "circuit":
        fetch_circuit(sys.argv[2], int(sys.argv[3]))
