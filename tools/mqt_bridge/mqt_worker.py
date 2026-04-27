import sys
import json
from mqt.bench import get_available_benchmark_names, get_benchmark, BenchmarkLevel

def fetch_names():
    print(json.dumps(get_available_benchmark_names()))

def fetch_circuit(algo_name, qubits):
    try:
        # Note: Ensure BenchmarkLevel is available and correct for your version
        qc = get_benchmark(algo_name, BenchmarkLevel.INDEP, qubits)
        print(qc.qasm())
    except Exception as e:
        sys.stderr.write(str(e))
        sys.exit(1)

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "names":
        fetch_names()
    elif mode == "circuit":
        fetch_circuit(sys.argv[2], int(sys.argv[3]))