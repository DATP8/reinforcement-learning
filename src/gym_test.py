from stable_baselines3.common.monitor import Monitor
from qiskit.converters import circuit_to_dag
from qiskit.converters import dag_to_circuit
from qiskit.transpiler.layout import Layout
from src.gym_extractor import HybridExtractor, SimpleExtractor
from src.curriculum_callback import CurriculumCallback
import gymnasium
from qiskit.transpiler import CouplingMap
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import multiprocessing as mp
import numpy as np

from src.routing_env import RoutingEnv
from qiskit.dagcircuit import DAGCircuit
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback


class EvalIfCurriculumFinishedCallback(BaseCallback):
    def __init__(
        self,
        eval_callback: MaskableEvalCallback,
        curriculum_callback: CurriculumCallback,
        verbose=0,
    ):
        super().__init__(verbose)
        self.eval_callback = eval_callback
        self.curriculum_callback = curriculum_callback

    def _init_callback(self) -> None:
        self.eval_callback.init_callback(self.model)

    def _on_step(self) -> bool:
        current_diff = self.training_env.env_method("get_difficulty")[0]
        # Only evaluate if we reached max difficulty
        if current_diff >= self.curriculum_callback.max_difficulty:
            return self.eval_callback.on_step()
        return True


### INFO
### When reporting results, take mean and standard deviation
### of at least 5 runs. Report the seeds for reproducability.


def mask_fn(env: gymnasium.Env) -> np.ndarray:
    return env.unwrapped.valid_action_mask()  # pyrefly: ignore


def make_env(
    num_qubits: int,
    coupling_map: CouplingMap,
    horizon: int,
    render_mode: str | None = None,
    initial_difficulty: int = 1,
    max_difficulty: int = 100,
):
    env = RoutingEnv(
        num_qubits=num_qubits,
        coupling_map=coupling_map,
        num_active_swaps=len(coupling_map.get_edges()),
        horizon=horizon,
        initial_difficulty=initial_difficulty,
        max_difficulty=max_difficulty,
        depth_slope=2,
        max_depth=128,
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

    layout_dict = {circuit.qubits[i]: int(p) for i, p in enumerate(env.locations)}
    layout = Layout(layout_dict)
    return circuit_to_dag(env.routed_circuit), layout


HORIZON = 16
MAX_DIFF = 2
NUM_QUBITS = 6

if __name__ == "__main__":
    coupling_map = CouplingMap.from_line(NUM_QUBITS)
    n_envs = mp.cpu_count() - 1
    print(f"Using {n_envs} envs")

    train_env = make_vec_env(
        lambda: make_env(
            num_qubits=NUM_QUBITS,
            coupling_map=coupling_map,
            horizon=HORIZON,
            initial_difficulty=1,
            max_difficulty=MAX_DIFF,
        ),
        n_envs=n_envs,
    )

    # Simple Extractor
    policy_kwargs = dict(
        features_extractor_class=SimpleExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
    )

    # Hybrid Graph Extractor
    policy_kwargs = dict(
        features_extractor_class=HybridExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
    )

    model = MaskablePPO("MlpPolicy", train_env, verbose=1, batch_size=2048)
    # model = MaskablePPO(
    #    "MultiInputPolicy", train_env, policy_kwargs=policy_kwargs, verbose=1
    # )

    eval_env = make_env(
        num_qubits=NUM_QUBITS,
        coupling_map=coupling_map,
        horizon=HORIZON,
        render_mode="ansi",
        initial_difficulty=MAX_DIFF,
        max_difficulty=MAX_DIFF,
    )
    eval_env = Monitor(eval_env)

    curriculum_callback = CurriculumCallback(threshold=0.85, verbose=1)

    eval_freq = max(100000 // n_envs, 1)
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path="./checkpoints/",
        log_path="./logs/",
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        verbose=1,
    )

    conditional_eval = EvalIfCurriculumFinishedCallback(
        eval_callback, curriculum_callback
    )

    model.learn(
        total_timesteps=25_000_000,
        progress_bar=True,
        callback=[curriculum_callback, conditional_eval],
    )
    model.save("test_model")

    for _ in range(5):
        obs, _ = eval_env.reset()
        flag = True
        while flag:
            action_masks = mask_fn(eval_env)  # pyrefly: ignore
            action, _ = model.predict(
                obs, deterministic=True, action_masks=action_masks
            )
            obs, reward, terminated, truncated, info = eval_env.step(action)
            if terminated:
                eval_env.render()
                flag = False
