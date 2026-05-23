import multiprocessing as mp
import os
import sys
import tempfile
from typing import Any

import optuna
import torch
from numpy import random
from qiskit.transpiler import CouplingMap
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna.optuna_search import OptunaSearch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from eval_circuits import EvalCircuits
from src.curriculum_callback import CurriculumCallback
from src.policy_types import ActorCriticPolicyType
from src.ppo_util import compute_avg_decomposed_cx, make_env
from src.routing_env import RoutingEnv

CPUS_PER_TRIAL = 16
NUM_UNIQUE_SAMPLES = 128
REPEATS_PER_CONFIG = 1
GRACE_PERIOD = 5
REDUCTION_FACTOR = 3
BRACKETS = 1
NUM_QUBITS = 19 # 6
TOTAL_TIMESTEPS = 25_000_000 # 50_000_000
BASE_EVAL_FREQ = 256_000
GPUS = 1.0

EXPERIMENT_NAME = "bipartite_25TT_hh3_3" # 3 after reducing memory issues, but now Emil is on CPU

class RayTuneCurriculumCallback(BaseCallback):
    def __init__(
        self,
        eval_env: Monitor,
        curriculum_callback: CurriculumCallback,
        eval_freq: int,
        n_eval_episodes: int,
        num_qubits: int,
        seed: int,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._eval_env: RoutingEnv = eval_env.unwrapped  # pyrefly: ignore
        self._eval_freq = eval_freq
        self._curriculum_callback = curriculum_callback
        self._seed = seed
        self._post_curriculum_evals = 0
        self._best_avg_decomposed_cx = sys.float_info.max
        self._eval_circuits = EvalCircuits.get_eval_circuits(
            n_eval_episodes=n_eval_episodes, num_qubits=num_qubits
        )

    def _on_step(self) -> bool:
        current_diff = self.training_env.env_method("get_difficulty")[0]
        curriculum_done = current_diff >= self._curriculum_callback.max_difficulty

        if not curriculum_done:
            return True

        if self._eval_freq > 0 and self.n_calls % self._eval_freq == 0:
            self._post_curriculum_evals += 1

            avg_decomposed_cx = self._compute_num_avg_decomposed_cx()

            metrics = {
                "avg_s_cx": avg_decomposed_cx,
                "diff": current_diff,
                "seed": self._seed,
                "pc_evals": self._post_curriculum_evals,
            }

            if avg_decomposed_cx < self._best_avg_decomposed_cx:
                self._best_avg_decomposed_cx = avg_decomposed_cx
                metrics["best_avg_s_cx"] = self._best_avg_decomposed_cx
                with tempfile.TemporaryDirectory() as ckpt_dir:
                    self.model.save(os.path.join(ckpt_dir, "model"))
                    checkpoint = tune.Checkpoint.from_directory(ckpt_dir)
                    tune.report(metrics, checkpoint=checkpoint)
            else:
                metrics["best_avg_s_cx"] = self._best_avg_decomposed_cx
                tune.report(metrics)

        return True

    def _compute_num_avg_decomposed_cx(self) -> float:
        if not isinstance(self.model, MaskablePPO):
            raise ValueError("Must be maskable PPO")

        return compute_avg_decomposed_cx(
            self.model, self._eval_env, self._eval_circuits, seed=self._seed
        )


def maskable_ppo_obj(config):
    seed = random.randint(0, 2**31 - 1)
    policy_type = ActorCriticPolicyType[config["policy_type"]]
    # coupling_map = CouplingMap.from_line(config["num_qubits"])
    coupling_map = CouplingMap.from_heavy_hex(3)

    train_env = make_vec_env(
        lambda: make_env(
            coupling_map=coupling_map,
            num_active_swaps=config["num_active_swaps"],
            horizon=config["horizon"],
            diff_slope=config["diff_slope"],
            layout_exponent=config["layout_exponent"],
            initial_difficulty=config["initial_difficulty"],
            max_difficulty=config["max_difficulty"],
            policy_type=policy_type,
        ),
        n_envs=config["num_envs"],
        seed=seed,
    )

    eval_env = make_env(
        coupling_map=coupling_map,
        num_active_swaps=config["num_active_swaps"],
        horizon=config["horizon"],
        diff_slope=config["diff_slope"],
        layout_exponent=config["layout_exponent"],
        initial_difficulty=config["max_difficulty"],
        max_difficulty=config["max_difficulty"],
        policy_type=policy_type,
    )
    eval_env = Monitor(eval_env)

    vibe_kwargs = {}
    if policy_type == ActorCriticPolicyType.VIBE_GRAPH:
        vibe_kwargs = dict(
                features_dim=config["vibe_features_dim"],
                gnn_hidden=config["vibe_gnn_hidden"],
                gnn_heads=config["vibe_gnn_heads"],
                gnn_out=config["vibe_gnn_out"],
                matrix_out=config["vibe_matrix_out"],
            )
    elif policy_type == ActorCriticPolicyType.BIPARTITE:
        vibe_kwargs = dict(
                features_dim=config["bi_features_dim"],
                gnn_hidden=config["bi_gnn_hidden"],
                gnn_heads=config["bi_gnn_heads"],
                gnn_out=config["bi_gnn_out"],
                matrix_out=config["bi_matrix_out"],
            )

    # Restore model from checkpoint if one exists for this trial
    checkpoint = tune.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as ckpt_dir:
            model = MaskablePPO.load(os.path.join(ckpt_dir, "model"), env=train_env)
        print(f"Resumed model from checkpoint (seed={seed}, device={model.device})")
    else:
        model = MaskablePPO(
            policy=policy_type.get_sb3_policy(),
            policy_kwargs=policy_type.get_policy_kwargs(**vibe_kwargs),
            env=train_env,
            learning_rate=config["learning_rate"],
            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"],
            batch_size=config["batch_size"],
            n_steps=config["n_steps"],
            n_epochs=config["n_epochs"],
            seed=seed,
            ent_coef=config["ent_coef"],
        )

    curriculum_callback = CurriculumCallback(
        config["threshold"], use_fast_curriculum=True
    )

    eval_freq = max(config["base_eval_freq"] // config["num_envs"], 1)
    ray_tune_eval = RayTuneCurriculumCallback(
        eval_env,
        curriculum_callback,
        eval_freq,
        config["n_eval_episodes"],
        config["num_qubits"],
        seed,
    )

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=[curriculum_callback, ray_tune_eval],
    )


def optuna_space(trial: optuna.Trial | None) -> dict[str, Any] | None:
    if trial is None:
        return None

    policy_type = trial.suggest_categorical(
        "policy_type",
        [
            # p.name for p in [ActorCriticPolicyType.BASIC]
            p.name for p in [ActorCriticPolicyType.BIPARTITE]
        ],  # [p.name for p in ActorCriticPolicyType],
    )

    n_steps = trial.suggest_int("n_steps", 256, 4096)

    # batch_size must divide n_steps * num_envs
    num_envs = CPUS_PER_TRIAL
    buffer_size = n_steps * num_envs
    batch_divisor = trial.suggest_int("batch_divisor", 2, 32)
    batch_size = max(1, buffer_size // batch_divisor)

    MAX_BATCH_SIZE = 2048
    batch_size = min(MAX_BATCH_SIZE, batch_size)

    while buffer_size % batch_size:
        batch_size -= 1

    config = {
        "policy_type": policy_type,
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
        "gamma": trial.suggest_float("gamma", 0.8, 1.0),
        # "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 1.0),
        "gae_lambda": 0.95,
        "batch_size": batch_size,
        "horizon": trial.suggest_int("horizon", 8, 64),
        "n_steps": n_steps,
        "ent_coef": trial.suggest_float("ent_coef", 1e-5, 0.05, log=True),
        # "n_epochs": trial.suggest_int("na_epochs", 8, 12),
        "n_epochs": 10,
        "num_qubits": NUM_QUBITS,
        "num_active_swaps": NUM_QUBITS - 1,
        "initial_difficulty": 1,
        "max_difficulty": 256,
        "diff_slope": 0.9,
        "layout_exponent": 1.0,
        "threshold": 0.85,
        "base_eval_freq": BASE_EVAL_FREQ,
        "n_eval_episodes": 256,
        "total_timesteps": TOTAL_TIMESTEPS,
        "num_envs": num_envs,
    }

    if policy_type == ActorCriticPolicyType.VIBE_GRAPH.name:
        config["vibe_features_dim"] = trial.suggest_categorical(
            "vibe_features_dim", [128, 256, 512]
        )
        config["vibe_gnn_hidden"] = trial.suggest_categorical(
            "vibe_gnn_hidden", [32, 64, 128]
        )
        config["vibe_gnn_heads"] = trial.suggest_categorical(
            "vibe_gnn_heads", [2, 4, 8]
        )
        config["vibe_gnn_out"] = trial.suggest_categorical(
            "vibe_gnn_out", [32, 64, 128]
        )
        config["vibe_matrix_out"] = trial.suggest_categorical(
            "vibe_matrix_out", [64, 128, 256]
        )

    if policy_type == ActorCriticPolicyType.BIPARTITE.name:
        config["bi_features_dim"] = trial.suggest_categorical(
            "bi_features_dim", [64, 128, 256, 512]
        )
        config["bi_gnn_hidden"] = trial.suggest_categorical(
            "bi_gnn_hidden", [32, 64, 128]
        )
        config["bi_gnn_heads"] = trial.suggest_categorical(
            "bi_gnn_heads", [2, 4, 8]
        )
        config["bi_gnn_out"] = trial.suggest_categorical(
            "bi_gnn_out", [32, 64, 128]
        )
        config["bi_matrix_out"] = trial.suggest_categorical(
            "bi_matrix_out", [16, 32, 64, 128]
        )

    return config


if __name__ == "__main__":
    os.environ["RAY_DEDUP_LOGS"] = "0"
    os.environ["RAY_AIR_NEW_OUTPUT"] = "0"

    total_cpus = mp.cpu_count()
    num_concurrent_trials = max(1, total_cpus // CPUS_PER_TRIAL)

    experiment_path = os.path.join(os.path.expanduser("~/ray_results"), EXPERIMENT_NAME)

    num_samples = NUM_UNIQUE_SAMPLES

    gpus_per_trial = GPUS / num_concurrent_trials if torch.cuda.is_available() else 0.0

    algo = OptunaSearch(space=optuna_space, metric="best_avg_s_cx", mode="min")

    max_evals = TOTAL_TIMESTEPS // BASE_EVAL_FREQ

    scheduler = ASHAScheduler(
        time_attr="pc_evals",
        metric="avg_s_cx",
        mode="min",
        max_t=max_evals,
        grace_period=GRACE_PERIOD,
        reduction_factor=REDUCTION_FACTOR,
        brackets=BRACKETS,
    )

    reporter = CLIReporter(
        infer_limit=10,
        print_intermediate_tables=True,
        metric="best_avg_s_cx",
        mode="min",
        sort_by_metric=True,
    )

    trainable = tune.with_resources(
        maskable_ppo_obj, resources={"cpu": CPUS_PER_TRIAL, "gpu": gpus_per_trial}
    )

    if tune.Tuner.can_restore(experiment_path):
        print(f"Restoring existing experiment from {experiment_path}")
        tuner = tune.Tuner.restore(
            experiment_path,
            trainable=trainable,
            resume_unfinished=True,
            resume_errored=False,
            restart_errored=False,
        )
    else:
        print("Starting new experiment")
        tuner = tune.Tuner(
            trainable,
            tune_config=tune.TuneConfig(
                num_samples=num_samples, search_alg=algo, scheduler=scheduler
            ),
            run_config=tune.RunConfig(
                name=EXPERIMENT_NAME,
                progress_reporter=reporter,
                log_to_file=True,
                checkpoint_config=tune.CheckpointConfig(
                    checkpoint_score_attribute="best_avg_s_cx",
                    checkpoint_score_order="min",
                    num_to_keep=2,
                ),
            ),
        )

    results = tuner.fit()

    df = results.get_dataframe()

    config_cols = [col for col in df.columns if col.startswith("config/")]

    agg_df = (
        df.groupby(config_cols)
        .agg(
            best_avg_s_cx=("best_avg_s_cx", "min"),
            seeds_used=("seed", lambda x: list(x)),
        )
        .reset_index()
    )

    agg_df = agg_df.sort_values("best_avg_s_cx", ascending=True)

    print(f"\n--- Top Hyperparameters (Averaged over {REPEATS_PER_CONFIG} seeds) ---")
    print(agg_df.to_string(index=False))

    agg_df.to_csv("results.csv", index=False)
