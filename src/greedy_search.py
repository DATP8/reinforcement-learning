from states.tensor_state_handler import TensorStateHandler
from states.state_handler import StateHandler
import torch


class GreedySearch:
    def __init__(self, model, game: StateHandler[torch.Tensor]):
        self.model = model
        self.game = game

    def search(self, state: torch.Tensor) -> list[int]:
        actions = []
        last_action = -1

        state_temp = state.clone()

        while not self.game.is_terminal(state):
            min_cost = float("inf")
            min_action = -1

            for action in self.game.get_possible_actions(state):
                next_state = self.game.get_next_state(state, action)

                if self.game.is_terminal(next_state):
                    cost = 0.0

                if last_action == action or torch.equal(
                    next_state[:, :, 0], state[:, :, 0]
                ):
                    continue

                with torch.no_grad():
                    cost = self.model.predict(next_state.unsqueeze(0)).item()

                if cost < min_cost:
                    min_cost = cost
                    min_action = action
                    state_temp = next_state

            print(min_action)
            state = state_temp
            last_action = min_action
            actions.append(min_action)

        return actions


# if __name__ == "__main__":
#    from model import ValueModel
#    from cnot_circuit import CNOTCircuit
#    import random
#    import time
#
#    def generate_random_circuit(game, n_qubits: int, n_gates: int, horizon: int):
#        qc = CNOTCircuit(n_qubits)
#
#        for i in range(n_gates):
#            q1, q2 = random.sample(range(n_qubits), 2)
#            while q1 == q2:
#                q2 = random.choice(range(n_qubits))
#
#            qc.add_cnot(q1, q2)
#
#        state = qc.to_tensor(horizon=horizon)
#        state, _ = game.prune(state)
#
#        if game.is_terminal(state):
#            return generate_random_circuit(game, n_qubits, n_gates, horizon)
#
#        return state
#
#    n_qubits = 6
#    horizon = 100
#    topology = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
#
#    game = TensorStateHandler(n_qubits, horizon, topology)
#    model = ValueModel(n_qubits, horizon, len(topology))
#    model.load_state_dict(
#        torch.load(
#            "models/value_model_deep_cube_a_exp_relu/difficulty11_iteration6790.pt"
#        )
#    )
#
#    # circuit = CNOTCircuit(n_qubits)
#    # circuit.add_cnot(0, 2)
#    # circuit.add_cnot(1, 2)
#    # circuit.add_cnot(2, 3)
#    # circuit.add_cnot(4, 0)
#
#    bwas = GreedySearch(model, game)
#    # root_state = generate_random_circuit(game, n_qubits, 15, horizon)
#    root_state = torch.load("circuits/dud.pt")
#    circuit = CNOTCircuit.from_tensor(root_state)
#    # root_state = circuit.to_tensor(horizon=horizon)
#
#    print(circuit)
#
#    # torch.save(root_state, "circuits/dud.pt")
#
#    start_time = time.time()
#    path = bwas.search(root_state)
#    end_time = time.time()
#
#    print(f"search time: {end_time - start_time:.4f}s")
#    state = root_state
#    for action in path:
#        state = game.get_next_state(state, action)
#
#    new_circuit = CNOTCircuit.from_tensor(state)
#    print(new_circuit)
#
#    print("Found path:", path)
