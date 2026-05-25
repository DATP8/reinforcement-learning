import os
from typing import Any

from qiskit.transpiler import CouplingMap
from ray.tune import Checkpoint, ExperimentAnalysis, ResultGrid
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env

from src.policy_types import ActorCriticPolicyType
from src.ppo_util import make_env

RELATIVE_PATH = "results/basic_vs_cancel1"
EXPERIMENT_PATH = os.path.abspath(RELATIVE_PATH)

analysis = ExperimentAnalysis(EXPERIMENT_PATH)
results_grid = ResultGrid(analysis)

best_result = results_grid.get_best_result(metric="avg_d_cx", mode="min")

print(f"Best Trial Directory: {best_result.path}")
print(f"Best Metrics: {best_result.metrics}")
print(f"Best Hyperparameters: {best_result.config}")

best_config: dict[str, Any] = best_result.config  # pyrefly: ignore
coupling_map = CouplingMap.from_line(best_config["num_qubits"])

eval_env = make_vec_env(
    lambda: make_env(
        coupling_map=coupling_map,
        num_active_swaps=best_config["num_active_swaps"],
        horizon=best_config["horizon"],
        diff_slope=best_config["diff_slope"],
        layout_exponent=best_config["layout_exponent"],
        initial_difficulty=best_config["max_difficulty"],
        max_difficulty=best_config["max_difficulty"],
        policy_type=ActorCriticPolicyType[best_config["policy_type"]],
    ),
    n_envs=best_config["num_envs"],
)

best_checkpoint: Checkpoint = best_result.checkpoint  # pyrefly: ignore
with best_checkpoint.as_directory() as checkpoint_dir:
    model_path = os.path.join(checkpoint_dir, "model.zip")

    loaded_model = MaskablePPO.load(model_path, env=eval_env)
    print("Successfully reloaded the best MaskablePPO model checkpoint!")
