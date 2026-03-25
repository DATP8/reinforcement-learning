from qiskit.transpiler import Layout
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.basepasses import TransformationPass


class CompleteRlRoutingPass(TransformationPass):
    def __init__(
        self,
        agent,
        env,
        name: str,
    ):
        super().__init__()
        self.env = env
        self.agent = agent
        self.model_name = name

    def get_name(self):
        return self.model_name

    def run(self, dag):
        qc = dag_to_circuit(dag)

        obs, _ = self.env.unwrapped.set_circuit(qc)

        flag = True
        while flag:
            action_masks = self.env.action_masks()
            action, _ = self.agent.predict(
                obs, deterministic=True, action_masks=action_masks
            )
            obs, reward, terminated, _, _ = self.env.step(action)
            if terminated:
                break
        new_qc = self.env.unwrapped.get_routed_circuit()

        self.property_set["final_layout"] = Layout(
            self.env.unwrapped.get_final_mapping()
        )
        return circuit_to_dag(new_qc)
