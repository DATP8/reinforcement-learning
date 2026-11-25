import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit


def generate_random_circuit(num_qubits, num_gates, gateset, seed=None):
    qc = QuantumCircuit(num_qubits)
    gate_set = ["cx"]
    for _ in range(num_gates):
        gate = np.random.choice(gate_set)
        if gate == "cx" and num_qubits > 1:
            q0, q1 = np.random.choice(num_qubits, size=2, replace=False)
            qc.cx(int(q0), int(q1))
        elif gate == "h":
            qc.h(int(np.random.choice(num_qubits)))
        else:
            qc.s(int(np.random.choice(num_qubits)))
    return qc


@dataclass
class RoutedCircuitMetrics:
    optimization_level: int
    depth: int
    num_gates: int


@dataclass
class CircuitMetrics:
    seed: int
    difficulty: int
    depth: int
    num_qubits: int
    gate_set: set[str]
    routed: list[RoutedCircuitMetrics]

    @property
    def id(self):
        return f"{self.difficulty}-{self.seed}"

    @property
    def gate_set_str(self):
        return "-".join(sorted(self.gate_set))

    def get_circuit(self):
        return generate_random_circuit(
            self.num_qubits, self.difficulty, self.gate_set, self.seed
        )

    def to_df(self):
        rows = []
        for r in self.routed:
            rows.append(
                {
                    "id": self.id,
                    "seed": self.seed,
                    "difficulty": self.difficulty,
                    "num_qubits": self.num_qubits,
                    "orig_depth": self.depth,
                    "opt_level": r.optimization_level,
                    "routed_depth": r.depth,
                    "routed_num_gates": r.num_gates,
                    "gate_set": self.gate_set_str,
                }
            )

        return pd.DataFrame(rows)

    @staticmethod
    def from_df(df: pd.DataFrame):
        circuits = []

        for cid, group in df.groupby("id"):
            first = group.iloc[0]

            routed = [
                RoutedCircuitMetrics(
                    optimization_level=row.opt_level,
                    depth=row.routed_depth,
                    num_gates=row.routed_num_gates,
                )
                for _, row in group.iterrows()
            ]

            circuits.append(
                CircuitMetrics(
                    seed=first.seed,
                    difficulty=first.difficulty,
                    depth=first.orig_depth,
                    num_qubits=first.num_qubits,
                    routed=routed,
                    gate_set={
                        item.strip()
                        for item in first.gate_set.split(",")
                        if item.strip()
                    },
                )
            )

        return circuits

    def to_rows(self):
        return [
            {
                "id": self.id,
                "seed": self.seed,
                "difficulty": self.difficulty,
                "num_qubits": self.num_qubits,
                "orig_depth": self.depth,
                "opt_level": r.optimization_level,
                "routed_depth": r.depth,
                "routed_num_gates": r.num_gates,
                "gate_set": self.gate_set_str,
            }
            for r in self.routed
        ]


class CurriculumHelper:
    def __init__(self, topology_name: str, num_qubits: int, gate_set: set[str]):
        self.topology_name = topology_name
        self.num_qubits = num_qubits
        self.gate_set = gate_set

    def get_file_path(self):
        return f"src/curriculum/benchmarks/{self.topology_name}_{'-'.join(sorted(self.gate_set))}_metrics.csv"

    def get_random_circuit(self, difficulty: int) -> CircuitMetrics:
        df = pd.read_csv(self.get_file_path())
        df = df[df["difficulty"] == difficulty]
        circuit_ids = df["id"].unique()
        cid = np.random.choice(circuit_ids)
        df = df[df["id"] == cid]
        (circuit,) = CircuitMetrics.from_df(df)
        return circuit

    def get_circuit(self, id):
        df = pd.read_csv(self.get_file_path())
        df = df[df["id"] == id]
        (circuit,) = CircuitMetrics.from_df(df)
        return circuit

    def save_circuits(self, circuits: list[CircuitMetrics]):
        path = self.get_file_path()

        write_header = not os.path.exists(path) or os.path.getsize(path) == 0

        # Flatten everything in one go
        rows = []
        for c in circuits:
            rows.extend(c.to_rows())

        df = pd.DataFrame(rows)

        df.to_csv(
            path,
            mode="a",
            header=write_header,
            index=False,
        )
