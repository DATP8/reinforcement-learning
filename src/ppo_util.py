from stable_baselines3.common.monitor import Monitor
from src.curriculum_callback import CurriculumCallback
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from src.routing_env import RoutingEnv
import gymnasium
from qiskit.transpiler import CouplingMap
import numpy as np
from qiskit.converters import circuit_to_dag
from qiskit.converters import dag_to_circuit
from qiskit.transpiler.layout import Layout
from qiskit.dagcircuit import DAGCircuit


def mask_fn(env: gymnasium.Env) -> np.ndarray:
    return env.unwrapped.valid_action_mask()  # pyrefly: ignore


def make_env(
    num_qubits: int,
    coupling_map: CouplingMap,
    num_active_swaps: int,
    horizon: int,
    diff_slope: int,
    render_mode: str | None = None,
    initial_difficulty: int = 1,
    max_difficulty: int = 100,
):
    env = RoutingEnv(
        num_qubits=num_qubits,
        coupling_map=coupling_map,
        num_active_swaps=num_active_swaps,
        horizon=horizon,
        initial_difficulty=initial_difficulty,
        max_difficulty=max_difficulty,
        diff_slope=diff_slope,
        render_mode=render_mode,
    )
    env = ActionMasker(env, mask_fn)
    return env


def route_circuit(model: MaskablePPO, dag: DAGCircuit) -> tuple[DAGCircuit, Layout]:
    circuit = dag_to_circuit(dag)
    env: RoutingEnv = model.env.envs[0].unwrapped  # pyrefly: ignore
    obs, _ = env.reset(options={"circuit": circuit})

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
