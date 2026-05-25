import sys
import time
from typing import Callable

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
    gamma: float = 0.99,
    shaping_coef: float = 0.0,
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
        gamma=gamma,
        shaping_coef=shaping_coef,
        sample_diff=sample_diff,
        render_mode=render_mode,
    )
    env = ActionMasker(env, mask_fn)
    return env


def get_decomposed_size(circuit: QuantumCircuit):
    ops = circuit.count_ops()
    num_decomposed_cx = ops.get("cx", 0) + ops.get("swap", 0) * 3
    return num_decomposed_cx


def route_circuit(
    model: MaskablePPO,
    circuit: DAGCircuit | QuantumCircuit,
    samples: int | None = None,
    time_limit_s: float | None = None,
) -> tuple[DAGCircuit, Layout]:
    if samples and (not isinstance(samples, int) or samples < 1):
        raise ValueError("Samples must be a positive integer.")

    if time_limit_s and (not isinstance(time_limit_s, float) or not time_limit_s > 0.0):
        raise ValueError("Time limit must be a positive real.")

    if isinstance(circuit, DAGCircuit):
        circuit = dag_to_circuit(circuit)

    env: RoutingEnv = model.env.envs[0].unwrapped  # pyrefly: ignore
    obs, _ = env.set_circuit(circuit, seed=model.seed)

    if env.is_terminal():
        return circuit_to_dag(circuit), Layout.generate_trivial_layout(*circuit.qregs)

    routed_qcs = []
    i = 0
    start = time.time()

    if not samples and not time_limit_s:
        samples = 1

    while True:
        if i > 0 and (
            samples
            and i > samples
            or time_limit_s
            and time.time() - start > time_limit_s
        ):
            break

        obs, _ = env.set_circuit(circuit, seed=model.seed)
        terminated = False
        is_looping = False
        while not terminated:
            mask = env.valid_action_mask()
            action, _ = model.predict(obs, action_masks=mask, deterministic=(not i))
            obs, _, terminated, _, opts = env.step(action)
            if opts["is_looping"]:
                is_looping = True

        if not is_looping:
            routed_qc = env.get_routed_circuit()
            layout = Layout(env.get_final_mapping())
            routed_qcs.append((routed_qc, layout))

    if not routed_qcs:
        raise ValueError("Model is looping")

    routed_qcs.sort(key=lambda x: get_decomposed_size(x[0]))
    routed_qc, layout = routed_qcs[0]
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
        num_decomposed_cx += ops.get("cx", 0) + ops.get("swap", 0) * 3
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


def linear_schedule(
    initial_value: float, end_value: float = 0.0
) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return end_value + progress_remaining * (initial_value - end_value)

    return func
