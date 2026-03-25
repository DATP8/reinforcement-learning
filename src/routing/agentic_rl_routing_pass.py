from qiskit.transpiler import Layout
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.basepasses import TransformationPass


class AgenticRlRoutingPass(TransformationPass):
    def __init__(
        self,
        agent,
        env,
    ):
        super().__init__()
        self.env = env
        self.agent = agent

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
        new_dag = circuit_to_dag(new_qc)

        layout = Layout(self.env.unwrapped.get_final_mapping())

        self.property_set["final_layout"] = (
            layout
            if (prev := self.property_set["final_layout"]) is None
            else prev.compose(layout, new_dag.qubits)
        )

        return new_dag
