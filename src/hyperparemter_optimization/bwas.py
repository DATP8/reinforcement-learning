from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import CommutativeCancellation
from qiskit.transpiler.passes import ApplyLayout
from src.routing.cnot_swap_cancel import CNOTSwapCancelation
from qiskit.transpiler.passes import TrivialLayout
from qiskit.transpiler import PassManager
from src.routing.swap_inserter.swap_inserter import SwapInserter
from src.routing.rl_routing_pass import RlRoutingPass
from src.routing.bwas_router import BWASRouter
from src.states.dense_circuit_graph_state_handler import DenseCircuitGraphStateHandler
from src.model import BiCircuitGNNDense
from src.circuit_generator import CircuitGenerator
from optuna import Trial
import optuna
import torch
import time
import json

N_CIRCUITS = 10
N_TRAILS = 1000
TRIAL_TIMEOUT = 500

NUM_QUBITS = 6
NUM_GATES = 30

SEED = 42


circuits = CircuitGenerator.generate_n_random_circuits(
    n=N_CIRCUITS, num_qubits=NUM_QUBITS, num_gates=NUM_GATES, seed=SEED
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = "models/dense_graph/difficulty31_iteration8000_0.3.pt"
model = BiCircuitGNNDense(NUM_QUBITS)
model.load_state_dict(torch.load(path, map_location=device))

topology = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
state_handler = DenseCircuitGraphStateHandler(NUM_QUBITS, topology)

coupling_map = CouplingMap(topology)
coupling_map.make_symmetric()

swap_inserter = SwapInserter(coupling_map, num_qubits=NUM_QUBITS)

trivial_layout = TrivialLayout(coupling_map)
apply_layout = ApplyLayout()
cnot_cancel = CNOTSwapCancelation()
commutative_cancellation = CommutativeCancellation()


def objective(trial: Trial):
    start_trial = time.time()
    batch_size = trial.suggest_int("batch_size", 1, 1000)
    weight = trial.suggest_float("weight", 0.01, 1.0)

    bwas_router = BWASRouter(model, state_handler, batch_size=batch_size, weight=weight)
    routing_pass = RlRoutingPass(
        bwas_router, swap_inserter
    )  # Assuming no swap inserter for simplicity
    pm = PassManager([trivial_layout, apply_layout, routing_pass, cnot_cancel])

    times = []
    costs = []
    for circuit in circuits:
        if time.time() - start_trial > TRIAL_TIMEOUT:
            raise optuna.TrialPruned()
        
        time_start = time.time()
        out_circuit = pm.run(circuit)
        time_end = time.time()
        times.append(time_end - time_start)
        costs.append(get_cost(out_circuit))

    routing_time = sum(times) / len(times)
    cost = sum(costs) / len(costs)

    return cost, routing_time


def get_cost(circuit) -> float:
    decomposed_circuit = commutative_cancellation.run(
        circuit.decompose(gates_to_decompose="swap").to_dag()
    ).to_circuit()

    return decomposed_circuit.depth()


def print_callback(study, trial):
    print(f"\nTrial {trial.number} finished")

    print("  Params:")
    for k, v in trial.params.items():
        print(f"    {k}: {v}")

    print(f"  Values: {trial.values}")

    print(f"  Best trials on Pareto front: {study.best_trials}")


study = optuna.create_study(
    directions=["minimize", "minimize"],
    study_name="routing_optimization",
    storage="sqlite:///routing_optimization.db",
    load_if_exists=True,
)

study.optimize(
    objective,
    n_trials=N_TRAILS,
    n_jobs=-1,
    timeout=None,
    callbacks=[print_callback],
    show_progress_bar=True,
)

print("Number of finished trials: ", len(study.trials))


pareto = [{"values": t.values, "params": t.params} for t in study.best_trials]

with open("pareto_trials.json", "w") as f:
    json.dump(pareto, f, indent=4)
