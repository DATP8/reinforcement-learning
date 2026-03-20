import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import standard_gates


class CircuitGenerator:
    _GATE_MAP = {
        "c3sx": (standard_gates.C3SXGate, 4, 0),
        "ccx": (standard_gates.CCXGate, 3, 0),
        "ccz": (standard_gates.CCZGate, 3, 0),
        "ch": (standard_gates.CHGate, 2, 0),
        "cp": (standard_gates.CPhaseGate, 2, 1),
        "crx": (standard_gates.CRXGate, 2, 1),
        "cry": (standard_gates.CRYGate, 2, 1),
        "crz": (standard_gates.CRZGate, 2, 1),
        "cs": (standard_gates.CSGate, 2, 0),
        "csdg": (standard_gates.CSdgGate, 2, 0),
        "cswap": (standard_gates.CSwapGate, 3, 0),
        "csx": (standard_gates.CSXGate, 2, 0),
        "cu": (standard_gates.CUGate, 2, 4),
        "cu1": (standard_gates.CU1Gate, 2, 1),
        "cu3": (standard_gates.CU3Gate, 2, 3),
        "cx": (standard_gates.CXGate, 2, 0),
        "cy": (standard_gates.CYGate, 2, 0),
        "cz": (standard_gates.CZGate, 2, 0),
        "dcx": (standard_gates.DCXGate, 2, 0),
        "ecr": (standard_gates.ECRGate, 2, 0),
        "h": (standard_gates.HGate, 1, 0),
        "id": (standard_gates.IGate, 1, 0),
        "iswap": (standard_gates.iSwapGate, 2, 0),
        "p": (standard_gates.PhaseGate, 1, 1),
        "r": (standard_gates.RGate, 1, 2),
        "rcccx": (standard_gates.RC3XGate, 4, 0),
        "rccx": (standard_gates.RCCXGate, 3, 0),
        "rx": (standard_gates.RXGate, 1, 1),
        "rxx": (standard_gates.RXXGate, 2, 1),
        "ry": (standard_gates.RYGate, 1, 1),
        "ryy": (standard_gates.RYYGate, 2, 1),
        "rz": (standard_gates.RZGate, 1, 1),
        "rzx": (standard_gates.RZXGate, 2, 1),
        "rzz": (standard_gates.RZZGate, 2, 1),
        "s": (standard_gates.SGate, 1, 0),
        "sdg": (standard_gates.SdgGate, 1, 0),
        "swap": (standard_gates.SwapGate, 2, 0),
        "sx": (standard_gates.SXGate, 1, 0),
        "sxdg": (standard_gates.SXdgGate, 1, 0),
        "t": (standard_gates.TGate, 1, 0),
        "tdg": (standard_gates.TdgGate, 1, 0),
        "u": (standard_gates.UGate, 1, 3),
        "u1": (standard_gates.U1Gate, 1, 1),
        "u2": (standard_gates.U2Gate, 1, 2),
        "u3": (standard_gates.U3Gate, 1, 3),
        "x": (standard_gates.XGate, 1, 0),
        "xx_minus_yy": (standard_gates.XXMinusYYGate, 2, 2),
        "xx_plus_yy": (standard_gates.XXPlusYYGate, 2, 2),
        "y": (standard_gates.YGate, 1, 0),
        "z": (standard_gates.ZGate, 1, 0),
    }

    @staticmethod
    def generate_random_circuit(
        num_qubits: int,
        num_gates: int,
        gateset: set[str] | None = None,
        seed: int | None = None,
    ) -> QuantumCircuit:
        """
        Generates a random quantum circuits based on number of qubits, number of gates, and gateset.
        """
        if num_qubits < 0:
            raise ValueError("Number of qubits can not be negative.")
        elif num_gates < 0:
            raise ValueError("Number of gates can not be negative.")
        elif num_qubits == 0 and num_gates != 0:
            raise ValueError("A quantum circuit with 0 qubits can not have any gates.")
        elif num_qubits == 0:
            return QuantumCircuit()
        elif num_gates == 0:
            return QuantumCircuit(num_qubits)
        elif gateset is not None and len(gateset) == 0:
            raise ValueError("Gates can not be applied because the gateset is empty.")

        if gateset is None:  # Allow all gates
            gateset = set(CircuitGenerator._GATE_MAP.keys())

        if seed is None:
            seed = np.random.randint(0, np.iinfo(int).max)
        rng = np.random.default_rng(seed)

        filtered_gates = [
            CircuitGenerator._GATE_MAP[name]
            for name in gateset
            if name in CircuitGenerator._GATE_MAP
            and CircuitGenerator._GATE_MAP[name][1] <= num_qubits
        ]

        if not filtered_gates:
            raise ValueError(
                "Provided gateset contains no valid gates for the given qubit count."
            )

        qc = QuantumCircuit(num_qubits)

        for _ in range(num_gates):
            gate_cls, q_count, p_count = filtered_gates[rng.choice(len(filtered_gates))]

            q_indices = rng.choice(num_qubits, size=q_count, replace=False)
            qubits = [qc.qubits[i] for i in q_indices]

            params = rng.uniform(0, 2 * np.pi, size=p_count)
            operation = gate_cls(*params)

            qc._append(operation, qubits) # pyrefly: ignore

        return qc

    def generate_n_random_circuits(
        n: int,
        num_qubits: int,
        num_gates: int,
        gateset: set[str] | None = None,
        seed: int | None = None,
    ) -> list[QuantumCircuit]:
        """
        Generates n random quantum circuits based on number of qubits, number of gates, and gateset.
        """
        circuits = []
        for _ in range(n):
            circuits.append(
                CircuitGenerator.generate_random_circuit(
                    num_qubits, num_gates, gateset, seed
                )
            )

        return circuits
