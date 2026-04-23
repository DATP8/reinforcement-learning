from stable_baselines3.common.monitor import Monitor
from ray.tune.schedulers import ASHAScheduler
from numpy import random
from ray.tune.search.optuna.optuna_search import OptunaSearch
from qiskit.transpiler import CouplingMap
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from src.curriculum_callback import CurriculumCallback
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from src.ppo_util import make_env
from ray import tune
import multiprocessing as mp
import torch
import os
import numpy as np
import tempfile


class RayTuneCurriculumCallback(MaskableEvalCallback):
    def __init__(
        self,
        eval_env: Monitor,
        curriculum_callback: CurriculumCallback,
        eval_freq: int,
        n_eval_episodes: int,
        seed: int,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            verbose=verbose,
        )
        self._curriculum_callback = curriculum_callback
        self._seed = seed
        self._post_curriculum_evals = 0
        self._tune_best_reward = -float("inf")

    def _on_step(self) -> bool:
        current_diff = self.training_env.env_method("get_difficulty")[0]
        curriculum_done = current_diff >= self._curriculum_callback.max_difficulty

        if not curriculum_done:
            return True

        continue_training = super()._on_step()

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self._post_curriculum_evals += 1

            metrics = {
                "mean_reward": self.last_mean_reward,
                "difficulty": current_diff,
                "seed": self._seed,
                "post_curriculum_evals": self._post_curriculum_evals,
            }

            if self.last_mean_reward > self._tune_best_reward:
                self._tune_best_reward = self.last_mean_reward
                metrics["best_mean_reward"] = self._tune_best_reward

                with tempfile.TemporaryDirectory() as ckpt_dir:
                    self.model.save(os.path.join(ckpt_dir, "best_model"))
                    checkpoint = tune.Checkpoint.from_directory(ckpt_dir)
                    tune.report(metrics, checkpoint=checkpoint)
            else:
                metrics["best_mean_reward"] = self._tune_best_reward
                tune.report(metrics)

        return continue_training


def maskable_ppo_obj(config):
    seed = random.randint(0, 2**31 - 1)
    coupling_map = CouplingMap.from_line(config["num_qubits"])

    train_env = make_vec_env(
        lambda: make_env(
            num_qubits=config["num_qubits"],
            coupling_map=coupling_map,
            num_active_swaps=config["num_active_swaps"],
            horizon=config["horizon"],
            diff_slope=config["diff_slope"],
            layout_exponent=config["layout_exponent"],
            initial_difficulty=config["initial_difficulty"],
            max_difficulty=config["max_difficulty"],
        ),
        n_envs=config["num_envs"],
        seed=seed,
    )

    eval_env = make_env(
        num_qubits=config["num_qubits"],
        coupling_map=coupling_map,
        num_active_swaps=config["num_active_swaps"],
        horizon=config["horizon"],
        diff_slope=config["diff_slope"],
        layout_exponent=config["layout_exponent"],
        initial_difficulty=config["max_difficulty"],  # Strictly eval on max diff
        max_difficulty=config["max_difficulty"],
    )
    eval_env = Monitor(eval_env)

    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        batch_size=config["batch_size"],
        n_steps=config["n_steps"],
        n_epochs=config["n_epochs"],
        seed=seed,
    )

    curriculum_callback = CurriculumCallback(config["threshold"])

    eval_freq = max(config["base_eval_freq"] // config["num_envs"], 1)
    ray_tune_eval = RayTuneCurriculumCallback(
        eval_env, curriculum_callback, eval_freq, config["n_eval_episodes"], seed
    )

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=[curriculum_callback, ray_tune_eval],
    )


if __name__ == "__main__":
    cpus_per_trial = 4
    num_unique_samples = 32
    repeats_per_config = 1
    grace_period = 10

    total_cpus = mp.cpu_count()
    num_concurrent_trials = max(1, total_cpus // cpus_per_trial)

    # num_samples = repeats_per_config * num_unique_samples # when using Repeater
    num_samples = num_unique_samples

    gpus_per_trial = 1.0 / num_concurrent_trials if torch.cuda.is_available() else 0.0

    search_space = {
        "learning_rate": tune.loguniform(1e-5, 1e-1),
        "gamma": tune.uniform(0.8, 1.0),
        "gae_lambda": tune.uniform(0.9, 1.0),
        "batch_size": tune.choice([512, 1024, 2048, 4096]),
        "horizon": tune.randint(4, 64),
        "num_qubits": 6,
        "initial_difficulty": 1,
        "max_difficulty": 256,
        "diff_slope": 2,
        "layout_exponent": 1.0,
        "threshold": 0.85,
        "base_eval_freq": 100_000,
        "n_eval_episodes": 10,
        "total_timesteps": 25_000_000,
        "num_active_swaps": 6,
        "num_envs": cpus_per_trial,
        "n_steps": 2048,
        "n_epochs": 10,
    }

    algo = OptunaSearch(metric="mean_reward", mode="max")
    # repeated_algo = Repeater(algo, repeat=repeats_per_config) # Can cause problems when using scheduler

    max_evals = search_space["total_timesteps"] // search_space["base_eval_freq"]

    scheduler = ASHAScheduler(
        time_attr="post_curriculum_evals",
        metric="mean_reward",
        mode="max",
        max_t=max_evals,
        grace_period=grace_period,
    )

    tuner = tune.Tuner(
        tune.with_resources(
            maskable_ppo_obj, resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial}
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=num_samples, search_alg=algo, scheduler=scheduler
        ),
    )

    results = tuner.fit()

    df = results.get_dataframe()

    config_cols = [col for col in df.columns if col.startswith("config/")]

    agg_df = (
        df.groupby(config_cols)
        .agg(
            avg_reward=("mean_reward", "mean"),
            std_reward=("mean_reward", np.std),
            seeds_used=("seed", lambda x: list(x)),
        )
        .reset_index()
    )

    agg_df = agg_df.sort_values("avg_reward", ascending=False)

    print(f"\n--- Top Hyperparameters (Averaged over {repeats_per_config} seeds) ---")
    print(agg_df.to_string(index=False))
