from qiskit import QuantumCircuit

from src.circuit_generator import CircuitGenerator


class EvalCircuits:
    EVAL_SEED = 2026
    MULTIPLIER = 8
    NUM_COUNTS = 25

    @staticmethod
    def get_eval_circuits(
        n_eval_episodes: int, num_qubits: int
    ) -> list[QuantumCircuit]:
        return CircuitGenerator.generate_n_random_cx_circuits(
            n=n_eval_episodes,
            num_qubits=num_qubits,
            num_gates=[
                i * EvalCircuits.MULTIPLIER
                for i in range(1, EvalCircuits.NUM_COUNTS + 1)
            ],
            seed=EvalCircuits.EVAL_SEED,
        )
