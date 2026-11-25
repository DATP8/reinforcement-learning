import gymnasium
import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.layout import Layout
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor

from src.curriculum_callback import CurriculumCallback
from src.policy_types import ActorCriticPolicyType
from src.routing_env import RoutingEnv


def mask_fn(env: gymnasium.Env) -> np.ndarray:
    return env.unwrapped.valid_action_mask()  # pyrefly: ignore


def make_env(
    coupling_map: CouplingMap,
    num_active_swaps: int,
    horizon: int,
    initial_difficulty: int,
    max_difficulty: int,
    diff_slope: float,
    layout_exponent: float,
    policy_type: ActorCriticPolicyType,
    render_mode: str | None = None,
):
    env = RoutingEnv(
        coupling_map=coupling_map,
        num_active_swaps=num_active_swaps,
        horizon=horizon,
        initial_difficulty=initial_difficulty,
        max_difficulty=max_difficulty,
        diff_slope=diff_slope,
        layout_exponent=layout_exponent,
        policy_type=policy_type,
        render_mode=render_mode,
    )
    env = ActionMasker(env, mask_fn)
    return env


def route_circuit(
    model: MaskablePPO, circuit: DAGCircuit | QuantumCircuit
) -> tuple[DAGCircuit, Layout]:
    if isinstance(circuit, DAGCircuit):
        circuit = dag_to_circuit(circuit)

    env: RoutingEnv = model.env.envs[0].unwrapped  # pyrefly: ignore
    obs, _ = env.reset(seed=model.seed, options={"circuit": circuit})

    if env.is_terminal():
        return circuit_to_dag(env.routed_circuit), Layout.generate_trivial_layout(
            *circuit.qregs
        )

    terminated = False
    while not terminated:
        mask = env.valid_action_mask()
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)

    layout_dict = {circuit.qubits[i]: int(p) for i, p in enumerate(env.l2p)}
    layout = Layout(layout_dict)
    return circuit_to_dag(env.routed_circuit), layout


class PostCurriculumEvalCallback(MaskableEvalCallback):
    def __init__(
        self,
        eval_env: Monitor,
        curriculum_callback: CurriculumCallback,
        eval_freq: int,
        n_eval_episodes: int,
        best_model_save_path: str,
        log_path: str,
    ):
        super().__init__(
            eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
        )
        self._curriculum_callback = curriculum_callback

    def _on_step(self) -> bool:
        current_diff = self.training_env.env_method("get_difficulty")[0]
        if current_diff < self._curriculum_callback.max_difficulty:
            return True

        return super()._on_step()
