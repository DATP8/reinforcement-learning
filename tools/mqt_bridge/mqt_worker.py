import sys
import json
import io
from qiskit import qpy
from mqt.bench.benchmarks   import get_available_benchmark_names
from mqt.bench              import get_benchmark, BenchmarkLevel

def fetch_names():
    print(json.dumps(get_available_benchmark_names()))

def fetch_circuit(algo_name, qubits):
    for q in range(3, int(qubits)).__reversed__():
        try:
            qc = get_benchmark(algo_name, BenchmarkLevel.INDEP, q)
            buf = io.BytesIO()
            qpy.dump(qc, buf)
            buf.seek(0)
            sys.stdout.buffer.write(buf.read())
            return
        except Exception as e:
            pass
    print("err")

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "names":
        fetch_names()
    elif mode == "circuit":
        fetch_circuit(sys.argv[2], int(sys.argv[3]))