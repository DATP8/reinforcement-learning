import sys

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
from src.eval_circuits import EvalCircuits
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
    sample_diff: bool = True,
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
        sample_diff=sample_diff,
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
    obs, _ = env.set_circuit(circuit, seed=model.seed)

    if env.is_terminal():
        return circuit_to_dag(circuit), Layout.generate_trivial_layout(*circuit.qregs)

    terminated = False
    while not terminated:
        mask = env.valid_action_mask()
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, _, terminated, _, opts = env.step(action)
        if opts["is_looping"]:
            raise ValueError("Model is looping")

    routed_qc = env.get_routed_circuit()
    layout = Layout(env.get_final_mapping())

    return circuit_to_dag(routed_qc), layout


def compute_avg_decomposed_cx(
    model: MaskablePPO,
    env: RoutingEnv,
    eval_circuits: list[QuantumCircuit],
    seed: int | None = None,
) -> float:
    num_decomposed_cx = 0
    for original_circuit in eval_circuits:
        circuit = dag_to_circuit(
            circuit_to_dag(original_circuit)
        )  # Make sure order is same as PassManager
        reset_kwargs = {"options": {"circuit": circuit}}
        obs, info = env.reset(seed=seed, **reset_kwargs)
        done = False
        while not done:
            mask = env.valid_action_mask()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated

        routed_circuit = env.get_routed_circuit()
        ops = routed_circuit.count_ops()
        num_decomposed_cx += ops.get("swap", 0) * 3
        num_decomposed_cx += ops.get("cx", 0)
    return num_decomposed_cx / len(eval_circuits)


class PostCurriculumEvalCallback(MaskableEvalCallback):
    def __init__(
        self,
        eval_env: Monitor,
        curriculum_callback: CurriculumCallback,
        eval_freq: int,
        n_eval_episodes: int,
        best_model_save_path: str,
        log_path: str,
        num_qubits: int,
        log_avg_s_cx: bool = False,
    ):
        super().__init__(
            eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
        )
        self._curriculum_callback = curriculum_callback
        self._target_env: RoutingEnv = eval_env.unwrapped  # pyrefly: ignore
        self._best_avg_decomposed_cx = sys.float_info.max
        self._log_avg_s_cx = log_avg_s_cx
        if self._log_avg_s_cx:
            self._eval_circuits = EvalCircuits.get_eval_circuits(
                n_eval_episodes=n_eval_episodes, num_qubits=num_qubits
            )

    def _on_step(self) -> bool:
        current_diff = self.training_env.env_method("get_difficulty")[0]
        if current_diff < self._curriculum_callback.max_difficulty:
            return True

        result = super()._on_step()

        if self._log_avg_s_cx:
            avg_decomposed_cx = self._compute_num_avg_decomposed_cx()
            if avg_decomposed_cx < self._best_avg_decomposed_cx:
                self._best_avg_decomposed_cx = avg_decomposed_cx

            self.logger.record("eval/best_avg_s_cx", self._best_avg_decomposed_cx)
            self.logger.record("eval/avg_s_cx", avg_decomposed_cx)

        return result

    def _compute_num_avg_decomposed_cx(self) -> float:
        if not isinstance(self.model, MaskablePPO):
            raise ValueError("Must be maskable PPO")

        return compute_avg_decomposed_cx(
            self.model, self._target_env, self._eval_circuits
        )
