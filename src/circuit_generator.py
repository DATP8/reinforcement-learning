from qiskit.circuit import CircuitInstruction
from qiskit.circuit.library import standard_gates
from qiskit import QuantumCircuit
import numpy as np


class CircuitGenerator:
    _gate_registry = [
        (standard_gates.IGate, 1, 0),
        (standard_gates.SXGate, 1, 0),
        (standard_gates.XGate, 1, 0),
        (standard_gates.RZGate, 1, 1),
        (standard_gates.RGate, 1, 2),
        (standard_gates.HGate, 1, 0),
        (standard_gates.PhaseGate, 1, 1),
        (standard_gates.RXGate, 1, 1),
        (standard_gates.RYGate, 1, 1),
        (standard_gates.SGate, 1, 0),
        (standard_gates.SdgGate, 1, 0),
        (standard_gates.SXdgGate, 1, 0),
        (standard_gates.TGate, 1, 0),
        (standard_gates.TdgGate, 1, 0),
        (standard_gates.UGate, 1, 3),
        (standard_gates.U1Gate, 1, 1),
        (standard_gates.U2Gate, 1, 2),
        (standard_gates.U3Gate, 1, 3),
        (standard_gates.YGate, 1, 0),
        (standard_gates.ZGate, 1, 0),
        (standard_gates.CXGate, 2, 0),
        (standard_gates.DCXGate, 2, 0),
        (standard_gates.CHGate, 2, 0),
        (standard_gates.CPhaseGate, 2, 1),
        (standard_gates.CRXGate, 2, 1),
        (standard_gates.CRYGate, 2, 1),
        (standard_gates.CRZGate, 2, 1),
        (standard_gates.CSXGate, 2, 0),
        (standard_gates.CUGate, 2, 4),
        (standard_gates.CU1Gate, 2, 1),
        (standard_gates.CU3Gate, 2, 3),
        (standard_gates.CYGate, 2, 0),
        (standard_gates.CZGate, 2, 0),
        (standard_gates.RXXGate, 2, 1),
        (standard_gates.RYYGate, 2, 1),
        (standard_gates.RZZGate, 2, 1),
        (standard_gates.RZXGate, 2, 1),
        (standard_gates.XXMinusYYGate, 2, 2),
        (standard_gates.XXPlusYYGate, 2, 2),
        (standard_gates.ECRGate, 2, 0),
        (standard_gates.CSGate, 2, 0),
        (standard_gates.CSdgGate, 2, 0),
        (standard_gates.SwapGate, 2, 0),
        (standard_gates.iSwapGate, 2, 0),
        (standard_gates.CCXGate, 3, 0),
        (standard_gates.CSwapGate, 3, 0),
        (standard_gates.CCZGate, 3, 0),
        (standard_gates.RCCXGate, 3, 0),
        (standard_gates.C3SXGate, 4, 0),
        (standard_gates.RC3XGate, 4, 0),
    ]

    _gate_map = {cls.__name__: (cls, q, p) for cls, q, p in _gate_registry}

    @staticmethod
    def generate_random_circuit(
        num_qubits: int, num_gates: int, gateset: list[str], seed: int | None = None
    ) -> QuantumCircuit:
        """
        Modified from qiskit's qiskit.circuit.random.random_circuit
        """
        if num_qubits == 0:
            return QuantumCircuit()

        if seed is None:
            seed = np.random.randint(0, np.iinfo(np.int32).max)
        rng = np.random.default_rng(seed)

        filtered_gates = [
            CircuitGenerator._gate_map[name]
            for name in gateset
            if name in CircuitGenerator._gate_map
        ]

        if not filtered_gates:
            raise ValueError("Provided gateset contains no valid or supported gates.")

        dtype = [("class", object), ("num_qubits", np.int64), ("num_params", np.int64)]

        # Group by qubit count
        by_q_count: dict = {}
        for g in filtered_gates:
            by_q_count.setdefault(g[1], []).append(g)

        all_gate_lists = {
            q: np.array(data, dtype=dtype) for q, data in by_q_count.items()
        }
        available_q_counts = sorted(all_gate_lists.keys())

        max_operands = [q for q in available_q_counts if q <= num_qubits]
        if not max_operands:
            raise ValueError(
                f"Gateset requires more qubits ({min(available_q_counts)}) than provided ({num_qubits})."
            )

        # Generate weights only for the qubit counts available in gateset
        rand_dist = rng.dirichlet(np.ones(len(max_operands)))
        num_operand_distribution = {q: rand_dist[i] for i, q in enumerate(max_operands)}

        gates_to_consider = []
        distribution = []
        for n_qubits in max_operands:
            gate_list = all_gate_lists[n_qubits]
            ratio = num_operand_distribution[n_qubits]
            gates_to_consider.extend(gate_list)
            distribution.extend([ratio / len(gate_list)] * len(gate_list))

        gates = np.array(gates_to_consider, dtype=dtype)
        qc = QuantumCircuit(num_qubits)
        qubits = np.array(qc.qubits, dtype=object, copy=True)

        counter = {q: 0 for q in max_operands}
        total_gates = 0

        for _ in range(num_gates):
            gate_specs = rng.choice(gates, size=len(qubits), p=distribution)
            cumulative_qubits = np.cumsum(gate_specs["num_qubits"], dtype=np.int64)

            max_index = np.searchsorted(cumulative_qubits, num_qubits, side="right")
            gate_specs = gate_specs[:max_index]
            slack = num_qubits - (
                cumulative_qubits[max_index - 1] if max_index > 0 else 0
            )

            for q_count in gate_specs["num_qubits"]:
                counter[q_count] += 1
            total_gates += len(gate_specs)

            # Fill slack
            while slack > 0:
                gate_added_flag = False
                for q_val in sorted(max_operands, reverse=True):
                    if slack >= q_val and (
                        total_gates == 0
                        or counter[q_val] / total_gates
                        < num_operand_distribution[q_val]
                    ):
                        gate_to_add = rng.choice(all_gate_lists[q_val])
                        gate_specs = np.hstack((gate_specs, gate_to_add))
                        counter[q_val] += 1
                        total_gates += 1
                        slack -= q_val
                        gate_added_flag = True
                        break
                if not gate_added_flag:
                    break

            q_indices = np.zeros(len(gate_specs) + 1, dtype=np.int64)
            p_indices = np.zeros(len(gate_specs) + 1, dtype=np.int64)
            np.cumsum(gate_specs["num_qubits"], out=q_indices[1:])
            np.cumsum(gate_specs["num_params"], out=p_indices[1:])

            parameters = rng.uniform(0, 2 * np.pi, size=p_indices[-1])
            rng.shuffle(qubits)

            for gate, q_s, q_e, p_s, p_e in zip(
                gate_specs["class"], q_indices, q_indices[1:], p_indices, p_indices[1:]
            ):
                operation = gate(*parameters[p_s:p_e])
                qc._append(  # pyrefly: ignore
                    CircuitInstruction(operation=operation, qubits=qubits[q_s:q_e])
                )

        return qc
